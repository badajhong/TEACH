"""Concurrent supervised perception and student SAC, with optional teacher warmup.

Perception uses the current short depth/GRU rollout; SAC uses the existing flat
replay containing collected student latents. Past latents and recurrent states
are intentionally not recomputed after perception updates. SAC gradients do not
flow through the replay into perception. There are no refresh cycles.
"""

from dataclasses import dataclass, fields
from numbers import Integral
from typing import List

import torch
from hydra.core.config_store import ConfigStore

from .flashsac_vel import FlashSACVelConfig
from .flashsac_vel_flat import FlashSACVelFlat
from .flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig, _ADAPTIVE_SCHEDULE_FIELDS,
)
from ..ppo.ppo_vel import DEPTH_KEY


@dataclass
class JointFlashSACVelFinetuneConfig(FlashSACVelConfig):
    _target_: str = "active_adaptation.learning.flashsac.joint_flashsac_vel_finetune.JointFlashSACVelFinetune"
    name: str = "joint_flashsac_vel_finetune"
    num_envs: int = 4096
    enable_rtx: bool = False
    updates_per_interaction_step: float = 8.0
    # Optional teacher-rollout warmup. Set 0 to start student joint training immediately.
    perception_iters: int = 0
    teacher_perception_rollout_noise_scale: float = 1.6
    adapt_epochs: int = 2
    perception_ema_tau: float = 0.08
    in_keys: List[str] = (*FlashSACVelConfig.in_keys, DEPTH_KEY)
    learning_rate_end: float = 1e-5
    buffer_max_length: int = 10_000_000
    buffer_min_length: int = 2_600_000
    buffer_device_type: str = "cpu"


ConfigStore.instance().store(
    "joint_flashsac_vel_finetune", node=JointFlashSACVelFinetuneConfig, group="algo"
)


class JointFlashSACVelFinetune(FlashSACVelFinetune):
    reuse_rollout_predictions = True

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        # Reuse the one-shot teacher/student setup internally. Cycle controls are
        # not part of the public Hydra config and cannot turn joint training off.
        shared_cfg = FlashSACVelFinetuneConfig(
            **{f.name: getattr(cfg, f.name) for f in fields(JointFlashSACVelFinetuneConfig)},
            cycle_mode=False, num_cycles=0, final_sac_iters=0,
        )
        self.joint_perception_iters_completed = 0
        super().__init__(shared_cfg, observation_spec, action_spec, reward_spec, device, env)

    def _print_training_schedule(self):
        print(f"[Joint FlashSAC finetune] teacher perception {self.cfg.perception_iters}, "
              "then supervised perception + student SAC on every iteration; no refresh cycles.")

    def _training_progress_summary(self):
        return "joint perception + SAC" if self.stage == 2 else "teacher warmup before joint training"

    def _prepare_rl_perception(self):
        for online, ema in self._perception_pairs():
            online.requires_grad_(True).train()
            ema.requires_grad_(False).eval()
        return "perception continues training and EMA continues updating"

    @torch.no_grad()
    def add_transition(self, td):
        super().add_transition(td)
        if self.stage == 2:
            # Keep a short supervised sequence alongside the long SAC replay.
            # Neither replay nor carried recurrent states reset between updates.
            self._collect_perception_transition(td)

    def update(self):
        if self.stage == 1:
            return super().update()
        # SAC starts only when replay is ready. Perception trains even while the
        # initial replay is filling, and keeps training after SAC updates begin.
        FlashSACVelFlat.update(self)
        if self._pending_adapt is not None:
            info = self._train_perception(self._pending_adapt, distill_actor=False)
            self._pending_adapt = None
            self.joint_perception_iters_completed += 1
            for key, value in info.items():
                self._info_sum[key] = self._info_sum.get(key, 0.) + value.detach()
                self._info_cnt[key] = self._info_cnt.get(key, 0) + 1
        self._perception_iteration_ready = False

    def pop_info(self):
        info = FlashSACVelFlat.pop_info(self)
        info.update({
            "finetune/stage": self.stage,  # 1 = teacher warmup; 2 = joint
            "finetune/joint_active": int(self.stage == 2),
            "finetune/iters_completed": self.finetune_iters_completed,
            "finetune/teacher_perception_iters_completed": self.perception_iters_completed,
            "finetune/joint_perception_iters_completed": self.joint_perception_iters_completed,
            "finetune/joint_iters_completed": self.rl_phase_iters_completed,
        })
        optimizer = self.opt_adapt_actor if self.stage == 1 else self._student.optimizer
        info["actor/lr"] = optimizer.param_groups[0]["lr"]
        return info

    def state_dict(self):
        state = super().state_dict()
        state["finetune_method"] = "joint_v1"
        state["joint_finetune_version"] = 1
        state["joint_perception_iters_completed"] = self.joint_perception_iters_completed
        # Keep the shared network/optimizer checkpoint format, without suggesting
        # that any cycle setting controls this algorithm.
        for key in (
            "cycle_mode", "final_sac_iters", "final_sac_active", "training_target_iter",
            "student_perception_phase", "last_transition_reason", "cycles_completed",
            "sac_iters", "num_cycles", "cycle_perception_iters", "cycle_sac_iters",
            *_ADAPTIVE_SCHEDULE_FIELDS,
        ):
            state.pop(key, None)
        return state

    def load_state_dict(self, state_dict, strict=True):
        joint = state_dict.get("finetune_method") == "joint_v1"
        if "finetune_version" in state_dict and not joint:
            raise ValueError("Joint finetune requires a teacher or joint finetune checkpoint")
        if joint and state_dict.get("joint_finetune_version") != 1:
            raise ValueError("Unsupported joint finetune checkpoint version")
        completed = state_dict.get("joint_perception_iters_completed", 0)
        if isinstance(completed, bool) or not isinstance(completed, Integral) or completed < 0:
            raise ValueError("Invalid joint perception iteration counter")
        transfer = dict(state_dict)
        if joint:
            # The common loader restores warmup/joint stage, online and EMA
            # perception, SAC and perception optimizers, and LR schedules.
            transfer["finetune_method"] = "cycles_v2"
            transfer["cycle_mode"] = False
            transfer["cycles_completed"] = 0
            transfer["student_perception_phase"] = False
        result = super().load_state_dict(transfer, strict=strict)
        self.joint_perception_iters_completed = int(completed)
        return result

    def _warn_schedule_change(self, state):
        if "perception_iters" in state and state["perception_iters"] != self.cfg.perception_iters:
            print("[Joint FlashSAC finetune] WARNING: teacher warmup changed from "
                  f"{state['perception_iters']} to {self.cfg.perception_iters} iterations.")
