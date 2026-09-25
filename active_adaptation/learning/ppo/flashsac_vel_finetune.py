"""Repeat original teacher-rollout warmup and SAC with frozen perception.

Only the short Stage 1 rollout contains depth/hidden states. Stage 2 replay stores the
teacher replay fields followed by velocity commands and the frozen student latent.
Actions keep the teacher's residual/absolute convention, including ref_joint_pos_.
"""

import copy
from dataclasses import dataclass
from typing import List

import torch
from torch import nn, optim
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import Unbounded
from torchrl.envs.transforms import TensorDictPrimer

from .common import ACTION_KEY, OBS_KEY, CatTensors, make_batch, make_conv
from .flashsac_vel import FlashSACVel, FlashSACVelConfig, ACTOR_REPLAY_KEY
from .flashsac_vel_flat import ACTION_DR_KEY, U_KEY, FlashSACVelFlat, NStepTrajectoryReplay
from .flashsac_upstream.agent import _sample_flashsac_actions, _update_networks
from .flashsac_upstream.scheduler import warmup_cosine_decay_scheduler
from .flashsac_upstream.utils_network import Network
from .ppo_vel import (
    DEPTH_KEY, VEL_CMD_KEY, OBJECT_KEY, OBJECT_PRED_KEY, OBJECT_PRED_TRANS_KEY,
    PRIV_FEATURE_KEY, PRIV_PRED_KEY, REF_JPOS_KEY, TemporalDepthGRU,
)
from ..modules.rnn import set_recurrent_mode


@dataclass
class FlashSACVelFinetuneConfig(FlashSACVelConfig):
    _target_: str = "active_adaptation.learning.ppo.flashsac_vel_finetune.FlashSACVelFinetune"
    name: str = "flashsac_vel_finetune"
    num_envs: int = 4096
    # Depth comes from CUDA ray casting; headless training needs no RTX renderer.
    # train.py enables rendering again for eval_render=true.
    enable_rtx: bool = False
    updates_per_interaction_step: float = 8.0
    perception_warmup_iters: int = 500
    # Repeat the same warmup after each SAC phase. Zero keeps one-shot warmup.
    rl_phase_iters: int = 1000
    # Total Stage 1 warmups (sets). After the last set, SAC runs to the end. 0 = unlimited.
    num_warmup_cycles: int = 5
    # Warmup sets after the first: "perception_only" rolls out the frozen student SAC actor
    # (temperature 1.0) and trains perception only; "original" repeats the teacher warmup.
    repeat_warmup_mode: str = "perception_only"
    # Stage 1 rollout noise: u = tanh(mean + std * noise * scale). Stage 2 SAC always uses 1.0.
    teacher_perception_rollout_noise_scale: float = 1.6  # first set (and "original" repeats)
    student_perception_rollout_noise_scale: float = 1.0  # "perception_only" repeat sets
    adapt_epochs: int = 2
    perception_ema_tau: float = 0.08
    in_keys: List[str] = (*FlashSACVelConfig.in_keys, DEPTH_KEY)
    buffer_max_length: int = 10_000_000
    buffer_device_type: str = "cpu"


ConfigStore.instance().store(
    "flashsac_vel_finetune", node=FlashSACVelFinetuneConfig, group="algo"
)


class FinetuneRollout(TensorDictModuleBase):
    def __init__(self, policy, mode):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.train_mode = mode == "train"
        self.in_keys = list(dict.fromkeys([
            *policy.replay_env_keys, VEL_CMD_KEY, DEPTH_KEY,
            "is_init", "adapt_hx", "depth_hx",
        ]))
        self.out_keys = [ACTION_KEY, U_KEY, PRIV_PRED_KEY,
                         ("next", "adapt_hx"), ("next", "depth_hx")]

    @torch.no_grad()
    def forward(self, td):
        p = self.policy
        if not p._checkpoint_loaded:
            raise RuntimeError("flashsac_vel_finetune requires checkpoint_path to a teacher or finetune checkpoint")
        if ACTION_DR_KEY in p.replay_keys:
            td[ACTION_DR_KEY] = p.action_dr()
        p._perceive(td, ema=True)
        teacher_rollout = self.train_mode and p.stage == 1 and not p.perception_only_warmup
        if teacher_rollout:
            network, observation = p._actor, p.replay_obs(td)
        else:
            network = p._student if p._student is not None else p._student_rollout
            observation = p._actor_adapt_input(td)
        if self.train_mode:
            # Even with an empty replay, sample the warm-start policy (never uniform actions).
            p._cached_noise, u, p._cur_noise_repeat_count, p._cur_noise_repeat_n = (
                _sample_flashsac_actions(
                    actor=network, noise=p._cached_noise, observations=observation,
                    temperature=p._rollout_noise_scale(teacher_rollout),
                    cur_count=p._cur_noise_repeat_count,
                    cur_n=p._cur_noise_repeat_n, zeta_cdf=p._zeta_cdf,
                )
            )
        else:
            mean, _ = network.apply("get_mean_and_std", observation, training=False)
            u = torch.tanh(mean)
        td[U_KEY] = u
        td[ACTION_KEY] = p._joint_action(u, td)
        return td


class FlashSACVelFinetune(FlashSACVel):
    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.perception_warmup_iters < 0 or cfg.adapt_epochs < 1:
            raise ValueError("perception_warmup_iters must be >= 0 and adapt_epochs >= 1")
        if cfg.rl_phase_iters < 0 or cfg.num_warmup_cycles < 0:
            raise ValueError("rl_phase_iters and num_warmup_cycles must be >= 0")
        if cfg.repeat_warmup_mode not in ("perception_only", "original"):
            raise ValueError("repeat_warmup_mode must be 'perception_only' or 'original'")
        for name in ("teacher_perception_rollout_noise_scale", "student_perception_rollout_noise_scale"):
            if not 0 <= getattr(cfg, name) < float("inf"):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 < cfg.perception_ema_tau <= 1:
            raise ValueError("perception_ema_tau must be in (0, 1]")
        if cfg.adapt_module != "gru" or not cfg.use_object_adapt:
            raise ValueError("flashsac_vel_finetune requires GRU and object adaptation")
        if not cfg.enable_residual_distillation:
            raise ValueError("Stage 1 requires enable_residual_distillation=true")
        if observation_spec.get(DEPTH_KEY, None) is None:
            raise ValueError("Student finetune requires depth observations and enabled cameras")
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        if self.num_envs < cfg.adapt_num_minibatches or self.num_envs % cfg.adapt_num_minibatches:
            raise ValueError("num_envs must be divisible by adapt_num_minibatches")
        self.stage = 1
        self.warmup_iters_completed = 0
        self.rl_phase_iters_completed = 0
        self.cycles_completed = 0
        self.finetune_iters_completed = 0
        self._rl_steps_in_iteration = 0
        self._checkpoint_loaded = False
        self.needs_env_reset = False
        self._student = None
        self._student_rollout = Network(self.actor_adapt)
        self.actor.requires_grad_(False).eval()
        self.critic.requires_grad_(False)
        self.temperature.requires_grad_(False)
        self.vel_command_dim = observation_spec[VEL_CMD_KEY].shape[-1]
        self.student_replay_dim = self.replay_obs_dim + self.vel_command_dim + cfg.latent_dim
        layout = {k: (s, e) for k, s, e, _ in self.replay_layout}
        policy_start, policy_end = layout[OBS_KEY]
        self._student_replay_index = torch.tensor([
            *range(self.replay_obs_dim, self.replay_obs_dim + self.vel_command_dim),
            *range(policy_start, policy_end),
            *range(self.replay_obs_dim + self.vel_command_dim, self.student_replay_dim),
        ], device=self.device)
        print(f"[FlashSAC finetune] warmup {cfg.perception_warmup_iters} iterations; "
              f"SAC phase {cfg.rl_phase_iters} iterations (0 = no repeated warmup); "
              f"{cfg.num_warmup_cycles or 'unlimited'} warmup sets "
              f"(repeat mode {cfg.repeat_warmup_mode}); "
              f"student replay {self.student_replay_dim}, critic input {self.obs_dim} dims.")

    @property
    def perception_only_warmup(self):
        return (self.stage == 1 and self.cycles_completed > 0 and
                self.cfg.repeat_warmup_mode == "perception_only")

    def _rollout_noise_scale(self, teacher_rollout):
        if teacher_rollout:
            return self.cfg.teacher_perception_rollout_noise_scale
        if self.perception_only_warmup:
            return self.cfg.student_perception_rollout_noise_scale
        return 1.0

    def _build_adaptation(self, observation_spec):
        super()._build_adaptation(observation_spec)
        # Keep the teacher checkpoint's parameter names/shapes; switch only the input keys.
        keys = [OBS_KEY]
        if self.cfg.adapt_module_input_cmd:
            keys.append(VEL_CMD_KEY)
        keys += [OBJECT_PRED_KEY, OBJECT_PRED_TRANS_KEY]
        self.adapt_module = Seq(
            CatTensors(keys, "_adapt_inp", del_keys=False, sort=False),
            self.adapt_module[1],
            selected_out_keys=[PRIV_PRED_KEY, ("next", "adapt_hx")],
        ).to(self.device)
        self.adapt_ema = copy.deepcopy(self.adapt_module).requires_grad_(False)
        cnn = nn.Sequential(
            make_conv(num_channels=[8, 8, 8], activation=nn.Mish, kernel_sizes=5),
            nn.LazyLinear(self.depth_feature_dim), nn.LayerNorm(self.depth_feature_dim),
        )
        self.temporal_depth_gru = TemporalDepthGRU(cnn, self.depth_feature_dim).to(self.device)
        fake = observation_spec.zero()[:2].to(self.device)
        fake["is_init"] = torch.ones(2, 1, dtype=torch.bool, device=self.device)
        fake["depth_hx"] = torch.zeros(2, self.depth_feature_dim, device=self.device)
        with torch.no_grad():
            self.temporal_depth_gru(fake)
        for module in cnn.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.temporal_depth_gru_ema = copy.deepcopy(self.temporal_depth_gru).requires_grad_(False)
        self.opt_adapt = optim.Adam(self._perception_parameters(), lr=self.cfg.adapt_learning_rate)

    def _perception_pairs(self):
        return ((self.temporal_depth_gru, self.temporal_depth_gru_ema),
                (self.object_adapt, self.object_adapt_ema),
                (self.adapt_module, self.adapt_ema))

    def _perception_parameters(self):
        return [p for model in (self.temporal_depth_gru, self.object_adapt, self.adapt_module)
                for p in model.parameters()]

    def make_tensordict_primer(self):
        return TensorDictPrimer({
            "adapt_hx": Unbounded((self.num_envs, self.cfg.latent_dim), device=self.device),
            "depth_hx": Unbounded((self.num_envs, self.depth_feature_dim), device=self.device),
        }, reset_key="done")

    def get_rollout_policy(self, mode="train"):
        return FinetuneRollout(self, mode)

    def _perceive(self, td, ema):
        depth, obj, adapt = (
            (self.temporal_depth_gru_ema, self.object_adapt_ema, self.adapt_ema) if ema
            else (self.temporal_depth_gru, self.object_adapt, self.adapt_module)
        )
        # PPOVEL's single-step GRU assumes its caller resets hidden states.
        # Also mask explicitly here for manual next-observation and evaluation calls.
        if td.ndim == 1:
            for key in ("depth_hx", "adapt_hx"):
                td[key] = torch.where(td["is_init"], 0., td[key])
        depth(td)
        obj(td)
        td[OBJECT_PRED_TRANS_KEY] = self.actor.transform_object_pose(td[OBJECT_PRED_KEY])
        adapt(td)
        return td

    def _joint_action(self, u, td):
        if self.cfg.action_mode == "residual":
            return td[REF_JPOS_KEY] + self.cfg.residual_scale * u
        return self.cfg.action_scale * u

    @torch.no_grad()
    def add_transition(self, td):
        if self.stage == 1:
            keys = [OBS_KEY, VEL_CMD_KEY, DEPTH_KEY, OBJECT_KEY, REF_JPOS_KEY,
                    "is_init", "adapt_hx", "depth_hx"]
            step = td.select(*keys).clone()
            step[ACTOR_REPLAY_KEY] = self.replay_obs(td).detach().clone()
            self._adapt_steps.append(step)
            if len(self._adapt_steps) == self.cfg.train_every:
                self._pending_adapt = torch.stack(self._adapt_steps, dim=1)
                self._adapt_steps.clear()
            return

        next_td = td["next"]
        if ACTION_DR_KEY in self.replay_keys:
            next_td[ACTION_DR_KEY] = td[ACTION_DR_KEY]
        # Use the post-action observation and the hidden state produced at s_t. Do NOT
        # reset on done here: truncations bootstrap from the terminal, pre-reset state.
        perception_next = TensorDict({
            OBS_KEY: next_td[OBS_KEY], VEL_CMD_KEY: next_td[VEL_CMD_KEY],
            DEPTH_KEY: next_td[DEPTH_KEY],
            "is_init": torch.zeros_like(td["is_init"]),
            "adapt_hx": next_td["adapt_hx"], "depth_hx": next_td["depth_hx"],
        }, batch_size=td.batch_size, device=self.device)
        self._perceive(perception_next, ema=True)
        current_obs = self._student_replay_obs(td, td[PRIV_PRED_KEY])
        next_obs = self._student_replay_obs(next_td, perception_next[PRIV_PRED_KEY])
        if self.buffer is None:
            self._allocate_student_replay()
        terminated = next_td["terminated"].squeeze(-1)
        truncated = next_td["truncated"].squeeze(-1)
        cut = terminated
        if not self.cfg.bootstrap_on_command_finished:
            time_limit = next_td["step_count"].squeeze(-1) >= self.env.max_episode_length
            cut = cut | (truncated & ~time_limit)
        reward = next_td["reward"].sum(-1)
        self.buffer.add(
            obs=current_obs if self.buffer.step == 0 else None, next_obs=next_obs,
            action=td[U_KEY], reward=reward, discount=next_td["discount"].squeeze(-1),
            done=terminated | truncated, cut=cut, valid=td["step_count"].squeeze(-1) > 1,
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward, terminated, truncated)
        self._rl_steps_in_iteration += 1
        if self._rl_steps_in_iteration == self.cfg.train_every:
            self.finetune_iters_completed += 1
            self.rl_phase_iters_completed += 1
            self._rl_steps_in_iteration = 0

    def _student_replay_obs(self, td, latent):
        return torch.cat([self.replay_obs(td), td[VEL_CMD_KEY], latent], dim=-1)

    def _allocate_student_replay(self):
        storage = "cpu" if self.cfg.buffer_device_type == "cpu" else self.device
        if torch.device(storage).type == "cuda":
            capacity = max(self.cfg.buffer_max_length // self.num_envs, self.cfg.n_step + 2) * self.num_envs
            itemsize = torch.finfo(getattr(torch, self.cfg.buffer_obs_dtype)).bits // 8
            required = capacity * (self.student_replay_dim * itemsize + self.action_dim * 4 + 15)
            free, _ = torch.cuda.mem_get_info(self.device)
            if required > free - 4 * 2**30:
                raise MemoryError("Student replay exceeds free GPU memory; use algo.buffer_device_type=cpu "
                                  "or reduce algo.buffer_max_length")
        self.buffer = NStepTrajectoryReplay(
            num_envs=self.num_envs, obs_dim=self.student_replay_dim, action_dim=self.action_dim,
            n_step=self.cfg.n_step, gamma=self.cfg.gamma, max_length=self.cfg.buffer_max_length,
            min_length=self.cfg.buffer_min_length, batch_size=self.cfg.sample_batch_size,
            device=self.device, obs_dtype=getattr(torch, self.cfg.buffer_obs_dtype), storage_device=storage,
        )
        print(f"[FlashSAC finetune] replay {self.buffer.capacity} rows, "
              f"{self.buffer.nbytes() / 2**30:.2f} GiB on {self.buffer.storage}")

    def update(self):
        if self.stage == 1:
            if self._pending_adapt is not None:
                info = self._train_warmup(self._pending_adapt)
                self._pending_adapt = None
                for key, value in info.items():
                    self._info_sum[key] = self._info_sum.get(key, 0.) + value.detach()
                    self._info_cnt[key] = self._info_cnt.get(key, 0) + 1
                self.warmup_iters_completed += 1
                self.finetune_iters_completed += 1
                if self.warmup_iters_completed >= self.cfg.perception_warmup_iters:
                    self._start_rl()
            return
        FlashSACVelFlat.update(self)
        if (self.cfg.perception_warmup_iters > 0 and self.cfg.rl_phase_iters > 0 and
                self._rl_steps_in_iteration == 0 and
                self.rl_phase_iters_completed >= self.cfg.rl_phase_iters and
                (self.cfg.num_warmup_cycles == 0 or
                 self.cycles_completed + 1 < self.cfg.num_warmup_cycles)):
            self.cycles_completed += 1
            self._start_warmup()

    @set_recurrent_mode(True)
    def _train_warmup(self, rollout):
        with torch.no_grad():
            raw = rollout[ACTOR_REPLAY_KEY].flatten(0, 1)
            rollout[PRIV_FEATURE_KEY] = self.actor.encode_priv(raw).reshape(
                *rollout.batch_size, self.cfg.latent_dim)
            mean, _ = self.actor.get_mean_and_std(raw, training=False)
            rollout["_teacher_action"] = self._joint_action(
                torch.tanh(mean).reshape(*rollout.batch_size, self.action_dim), rollout)
        infos = []
        for _ in range(self.cfg.adapt_epochs):
            for batch in make_batch(rollout, self.cfg.adapt_num_minibatches, self.cfg.train_every):
                self._perceive(batch, ema=False)
                valid = (~batch["is_init"]).float()
                priv_loss = ((batch[PRIV_PRED_KEY] - batch[PRIV_FEATURE_KEY]).square() * valid).mean()
                object_loss = ((batch[OBJECT_PRED_KEY] - batch[OBJECT_KEY]).square() * valid).mean()
                priv_pred_norm = batch[PRIV_PRED_KEY].detach().norm(dim=-1).mean()
                depth_feature_norm = batch["_depth_feature"].detach().norm(dim=-1).mean()
                self.opt_adapt.zero_grad(set_to_none=True)
                (priv_loss + object_loss).backward()
                nn.utils.clip_grad_norm_(self._perception_parameters(), self.cfg.max_grad_norm)
                self.opt_adapt.step()
                # Distill on the rollout perception (EMA), which will be frozen for SAC.
                # This also keeps the actor loss completely separate from perception gradients.
                with torch.no_grad():
                    self._perceive(batch, ema=True)
                inputs = self._actor_adapt_input(batch).flatten(0, 1).detach()
                info = {
                    "adapt/priv_loss": priv_loss.detach(), "adapt/object_loss": object_loss.detach(),
                    "adapt/priv_feature_norm": batch[PRIV_FEATURE_KEY].norm(dim=-1).mean().detach(),
                    "adapt/priv_pred_norm": priv_pred_norm,
                    "adapt/depth_feature_norm": depth_feature_norm,
                }
                if self.perception_only_warmup:
                    # The SAC actor stays frozen; only monitor its distance from the teacher.
                    with torch.no_grad():
                        student_mean, _ = self.actor_adapt.get_mean_and_std(inputs, training=False)
                        student_action = self._joint_action(
                            torch.tanh(student_mean).reshape(*batch.batch_size, self.action_dim), batch)
                        mse = ((student_action - batch["_teacher_action"]).square() * valid).mean()
                    info["adapt/teacher_student_action_rmse"] = mse.sqrt()
                    infos.append(info)
                    continue
                student_mean, _ = self.actor_adapt.get_mean_and_std(inputs, training=True)
                student_action = self._joint_action(
                    torch.tanh(student_mean).reshape(*batch.batch_size, self.action_dim), batch)
                adapt_loss = ((student_action - batch["_teacher_action"]).square() * valid).mean()
                self.opt_adapt_actor.zero_grad(set_to_none=True)
                adapt_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_adapt.parameters(), self.cfg.max_grad_norm)
                self.opt_adapt_actor.step()
                self._normalize_actor_adapt_parameters()
                info["adapt/adapt_loss"] = adapt_loss.detach()
                info["adapt/teacher_student_action_rmse"] = adapt_loss.detach().sqrt()
                infos.append(info)
        with torch.no_grad():
            for source, target in self._perception_pairs():
                for src, dst in zip(source.parameters(), target.parameters()):
                    dst.lerp_(src, self.cfg.perception_ema_tau)
        return {k: torch.stack([i[k] for i in infos]).mean() for k in infos[0]}

    def _start_rl(self, schedule_updates=None):
        for pair in self._perception_pairs():
            for model in pair:
                model.requires_grad_(False).eval()
        self.actor_adapt.requires_grad_(True)
        self.critic.requires_grad_(True)
        self.temperature.requires_grad_(True)
        self.stage = 2
        self.rl_phase_iters_completed = 0
        self._reset_phase_rollout()
        print("[FlashSAC finetune] Stage 2: perception and EMA frozen; student SAC enabled. "
              "Resetting environments/hidden states; replay starts empty.")
        if self._student is not None:
            # Continue SAC optimizer/scheduler state after repeated supervised warmup.
            return
        # Existing critic/temperature optimizers are fresh when loading a teacher. They
        # have not stepped during warmup. Student SAC gets a separate fresh optimizer.
        optimizer = optim.Adam(self.actor_adapt.parameters(), lr=self.cfg.learning_rate_peak,
                               fused=self.device.type == "cuda")
        horizon = self.cfg.num_env_steps // self.num_envs
        warmup_steps = self.cfg.perception_warmup_iters * self.cfg.train_every
        rl_steps = self.cfg.rl_phase_iters * self.cfg.train_every
        # Count the RL env steps left after every warmup set that fits in the horizon.
        warmups, elapsed, steps = 1, warmup_steps, 0
        while (warmup_steps and rl_steps and elapsed + rl_steps < horizon and
               (self.cfg.num_warmup_cycles == 0 or warmups < self.cfg.num_warmup_cycles)):
            warmups += 1
            steps += rl_steps
            elapsed += rl_steps + warmup_steps
        steps = max(1, steps + max(0, horizon - elapsed))
        updates = schedule_updates or max(1, int(steps * self.cfg.updates_per_interaction_step))
        self._rl_schedule_updates = updates
        schedule = warmup_cosine_decay_scheduler(
            init_value=self.cfg.learning_rate_init, peak_value=self.cfg.learning_rate_peak,
            end_value=self.cfg.learning_rate_end,
            warmup_steps=int(self.cfg.learning_rate_warmup_rate * updates),
            decay_steps=max(1, int(self.cfg.learning_rate_decay_rate * updates)),
        )
        # Initialize schedules once; later cycles retain their SAC update progress.
        for network in (self._critic, self._temperature):
            network.scheduler = torch.optim.lr_scheduler.LambdaLR(
                network.optimizer, lr_lambda=lambda step: schedule(step) / self.cfg.learning_rate_peak)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: schedule(step) / self.cfg.learning_rate_peak)
        self._student = Network(
            self.actor_adapt, optimizer, scheduler,
            compile_network=self.cfg.use_compile, compile_mode=self.flash_cfg.compile_mode,
            use_weight_normalization=True,
        )
        if self.cfg.use_compile:
            self._student.network.get_mean_and_std = torch.compile(
                self._student.network.get_mean_and_std, mode=self.flash_cfg.compile_mode)
        self._student.normalize_parameters()
        self._update_step = 0

    def _reset_phase_rollout(self):
        self.buffer = None
        self._adapt_steps.clear()
        self._pending_adapt = None
        self._rl_steps_in_iteration = 0
        self._update_counter = 0.
        self._cur_noise_repeat_count.zero_()
        self._cur_noise_repeat_n.fill_(1)
        if self.reward_normalizer is not None:
            self.reward_normalizer.G_r.zero_()
        self.needs_env_reset = True

    def _start_warmup(self):
        self.stage = 1
        self.warmup_iters_completed = 0
        self._reset_phase_rollout()
        for online, ema in self._perception_pairs():
            online.requires_grad_(True).train()
            ema.requires_grad_(False).eval()
        # perception_only_warmup reads self.stage, which is already 1 here.
        self.actor_adapt.requires_grad_(not self.perception_only_warmup).train()
        for model in (self.critic, self.temperature):
            model.requires_grad_(False)
            model.zero_grad(set_to_none=True)
        # Same as the original warmup: supervised optimizers start without Adam moments.
        for optimizer in (self.opt_adapt, self.opt_adapt_actor):
            optimizer.state.clear()
        kind = ("frozen-student-rollout perception-only" if self.perception_only_warmup
                else "original teacher-rollout")
        print(f"[FlashSAC finetune] Stage 1: restarting {kind} warmup "
              f"after SAC cycle {self.cycles_completed}; replay released.")

    def _update_once(self):
        batch = self.buffer.sample()
        for key, actor_key in (("observation", "actor_observation"),
                               ("next_observation", "actor_next_observation")):
            obs = batch[key]
            batch[actor_key] = obs.index_select(-1, self._student_replay_index)
            batch[key] = self.critic_obs(obs)
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        info = _update_networks(
            batch=batch, actor=self._student, critic=self._critic, target_critic=self._target_critic,
            temperature=self._temperature, cfg=self.flash_cfg,
            do_actor_update=self._update_step % self.flash_cfg.actor_update_period == 0,
            device=self.device, grad_scaler=self._grad_scaler,
        )
        self._update_step += 1
        return info

    def pop_info(self):
        info = super().pop_info()
        info.update({"finetune/stage": self.stage,
                     "finetune/warmup_iters_completed": self.warmup_iters_completed,
                     "finetune/rl_phase_iters_completed": self.rl_phase_iters_completed,
                     "finetune/cycles_completed": self.cycles_completed,
                     "finetune/iters_completed": self.finetune_iters_completed})
        info["actor/lr"] = (self.opt_adapt_actor if self.stage == 1 and not self.perception_only_warmup
                            else self._student.optimizer).param_groups[0]["lr"]
        return info

    def state_dict(self):
        state = super().state_dict()
        state["finetune_version"] = 5
        state["finetune_method"] = "repeated_original_warmup_v1"
        state["finetune_stage"] = self.stage
        state["warmup_iters_completed"] = self.warmup_iters_completed
        state["finetune_iters_completed"] = self.finetune_iters_completed
        state["student_replay_dim"] = self.student_replay_dim
        state["student_replay_index"] = self._student_replay_index.cpu()
        state["perception_warmup_iters"] = self.cfg.perception_warmup_iters
        state["rl_phase_iters"] = self.cfg.rl_phase_iters
        state["num_warmup_cycles"] = self.cfg.num_warmup_cycles
        state["repeat_warmup_mode"] = self.cfg.repeat_warmup_mode
        state["rl_phase_iters_completed"] = self.rl_phase_iters_completed
        state["cycles_completed"] = self.cycles_completed
        for name in ("temporal_depth_gru", "temporal_depth_gru_ema"):
            state[name] = getattr(self, name).state_dict()
        if self._student is not None:
            state["rl_schedule_updates"] = self._rl_schedule_updates
            state["student_optimizer"] = self._student.optimizer.state_dict()
            state["student_scheduler"] = self._student.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        resume = state_dict.get("finetune_version") in (1, 5)
        if "finetune_version" in state_dict and not resume:
            raise ValueError("Unsupported finetune checkpoint version")
        if state_dict.get("actor_adapt_arch") != "flashsac_v1":
            raise ValueError("Finetune requires a teacher checkpoint with FlashSAC actor_adapt")
        if resume:
            if state_dict.get("finetune_version") == 5:
                if state_dict.get("finetune_method") != "repeated_original_warmup_v1":
                    raise ValueError("Unsupported finetune method")
                if state_dict["rl_phase_iters"] != self.cfg.rl_phase_iters:
                    raise ValueError("Use the checkpoint's rl_phase_iters when resuming")
                if state_dict.get("num_warmup_cycles", 0) != self.cfg.num_warmup_cycles:
                    raise ValueError("Use the checkpoint's num_warmup_cycles when resuming")
                if state_dict.get("repeat_warmup_mode", "original") != self.cfg.repeat_warmup_mode:
                    raise ValueError("Use the checkpoint's repeat_warmup_mode when resuming")
            if state_dict["finetune_stage"] not in (1, 2):
                raise ValueError("Invalid finetune stage in checkpoint")
            if (state_dict["student_replay_dim"] != self.student_replay_dim or
                    not torch.equal(state_dict["student_replay_index"].cpu(), self._student_replay_index.cpu())):
                raise ValueError("Student observation layout differs from the saved finetune checkpoint")
            if state_dict["perception_warmup_iters"] != self.cfg.perception_warmup_iters:
                raise ValueError("Use the checkpoint's perception_warmup_iters when resuming")
        # The parent's loader validates the teacher critic/replay/action layouts. On transfer,
        # omit teacher optimizer/counters: this is a new phase, not continued teacher training.
        transfer = dict(state_dict)
        if not resume:
            for key in list(transfer):
                if key.endswith(("_optimizer", "_scheduler")) or key in (
                    "opt_adapt", "opt_adapt_actor", "grad_scaler", "update_step", "last_iter"
                ):
                    transfer.pop(key)
        super().load_state_dict(transfer, strict=strict)
        if resume:
            for name in ("temporal_depth_gru", "temporal_depth_gru_ema"):
                getattr(self, name).load_state_dict(state_dict[name], strict=strict)
            self.finetune_iters_completed = state_dict["finetune_iters_completed"]
            self.cycles_completed = state_dict.get("cycles_completed", 0)
            # A repeated warmup also retains the preceding SAC optimizer/scheduler.
            if "student_optimizer" in state_dict:
                self._start_rl(schedule_updates=state_dict.get("rl_schedule_updates"))
                self._student.optimizer.load_state_dict(state_dict["student_optimizer"])
                self._student.scheduler.load_state_dict(state_dict["student_scheduler"])
                for name, network in (("critic", self._critic), ("temperature", self._temperature)):
                    network.optimizer.load_state_dict(state_dict[f"{name}_optimizer"])
                    network.scheduler.load_state_dict(state_dict[f"{name}_scheduler"])
                self._update_step = state_dict["update_step"]
                if state_dict["finetune_stage"] == 1:
                    self._start_warmup()
                    for name in ("opt_adapt", "opt_adapt_actor"):
                        if name in state_dict:
                            getattr(self, name).load_state_dict(state_dict[name])
            self.warmup_iters_completed = state_dict["warmup_iters_completed"]
            self.rl_phase_iters_completed = state_dict.get("rl_phase_iters_completed", 0)
        elif self.cfg.perception_warmup_iters == 0:
            self._start_rl()
        if self.reward_normalizer is not None:
            self.reward_normalizer.G_r.zero_()
        self._checkpoint_loaded = True
        # The training loop resets the simulator at startup. Replay/hidden state are intentionally
        # not checkpointed, and RL resumes by filling a fresh buffer with the warm-start actor.
        self.needs_env_reset = False
        self.env.set_progress(self.finetune_iters_completed)
        print(f"[FlashSAC finetune] {'resumed' if resume else 'loaded teacher'}; "
              f"stage={self.stage}, warmup={self.warmup_iters_completed}/{self.cfg.perception_warmup_iters}")
        return []
