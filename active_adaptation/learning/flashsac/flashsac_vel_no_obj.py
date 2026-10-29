"""``flashsac_vel_train`` for motion-only tracking, with an actor that reads only student inputs.

The actor's whole input is ``algo.actor_terms``: the ``policy`` and ``vel_command``
terms the ``_stu`` actor reads, without its depth/object latent, and the reference
joint positions the residual action is added to. It has no privileged encoder and no
object, depth or ray-cast input, so there is no adaptation module or student
distillation either; the trained actor is the policy to evaluate. The critic keeps
the privileged ``FlashSACVelFlat`` observation, which has no object groups here. The
task must not spawn an object, for example ``task=R1/rl/sub3_largebox_003``.

The FlashSAC side (critic, temperature, reward scaling, exploration, update rule)
is the shared ``FlashSACVelFlat`` that ``flashsac_vel_train`` also uses.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from termcolor import colored

from ..ppo.common import ACTION_KEY, CMD_KEY, OBS_KEY, OBS_PRIV_KEY
from .flashsac_upstream.agent import _update_networks
from .flashsac_upstream.network import FlashSACActor
from .flashsac_upstream.scheduler import ramp_cosine_decay_scheduler
from .flashsac_upstream.utils_network import Network
from .flashsac_vel_flat import (
    ACTION_DR_KEY,
    U_KEY,
    FlashSACVelFlat,
    FlashSACVelFlatConfig,
    NStepTrajectoryReplay,
    _raw,
)


HEIGHT_KEY = "height"
VEL_CMD_KEY = "vel_command"
REF_JPOS_KEY = "ref_joint_pos_"


@dataclass
class FlashSACVelNoObjConfig(FlashSACVelFlatConfig):
    _target_: str = "active_adaptation.learning.flashsac.flashsac_vel_no_obj.FlashSACVelNoObj"
    name: str = "flashsac_vel_no_obj"

    # Groups the env keeps (ref_joint_pos_ is always kept).
    in_keys: List[str] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY, HEIGHT_KEY, VEL_CMD_KEY)
    # Critic only: flashsac_vel_train's groups without object_; ref_joint_pos_ is appended.
    obs_keys: List[str] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY, HEIGHT_KEY, ACTION_DR_KEY)
    # The actor's whole input as group/term, in this order. Other terms of these groups are
    # not read, and a term the task does not define is an error.
    actor_terms: List[str] = (
        "policy/root_ang_vel_history",
        "policy/projected_gravity_history",
        "policy/joint_pos_history",
        "policy/prev_actions",
        "vel_command/ref_root_vel_future_b",
        "vel_command/ref_contact_flag_future",
        "ref_joint_pos_/ref_joint_pos_action_policy",
    )

    buffer_max_length: int = 10_000_000
    buffer_device_type: str = "cpu"


cs = ConfigStore.instance()
cs.store("flashsac_vel_train_no_obj", node=FlashSACVelNoObjConfig, group="algo")


class FlashSACVelNoObjRollout(TensorDictModuleBase):
    def __init__(self, policy: "FlashSACVelNoObj", mode: str):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.deterministic = mode != "train"
        self.train_mode = mode == "train"
        self.in_keys = list(policy.actor_groups)
        if policy.cfg.action_mode == "residual" and REF_JPOS_KEY not in self.in_keys:
            self.in_keys.append(REF_JPOS_KEY)
        self.out_keys = [ACTION_KEY, U_KEY]

    def forward(self, tensordict: TensorDictBase):
        policy = self.policy
        if self.train_mode and ACTION_DR_KEY in policy.obs_keys:
            # Only the critic reads it; add_transition stores it with this step.
            tensordict.set(ACTION_DR_KEY, policy.action_dr())
        u = policy.act(policy.actor_obs(tensordict), self.deterministic)
        tensordict.set(U_KEY, u)
        if policy.cfg.action_mode == "residual":
            action = tensordict[REF_JPOS_KEY] + policy.cfg.residual_scale * u
        else:
            action = policy.cfg.action_scale * u
        tensordict.set(ACTION_KEY, action)
        return tensordict


class FlashSACVelNoObj(FlashSACVelFlat):
    """FlashSAC with the flat privileged critic and an actor that reads only ``cfg.actor_terms``."""

    def __init__(self, cfg: FlashSACVelNoObjConfig, observation_spec, action_spec, reward_spec, device, env):
        base_env = getattr(env, "base_env", env)
        if getattr(base_env.command_manager, "object", None) is not None:
            raise ValueError(
                "flashsac_vel_train_no_obj needs a task without an object (no command.object_asset_name), "
                "for example task=R1/rl/sub3_largebox_003."
            )
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)

        self.actor_groups, actor_index, self.actor_layout = self._select_actor_terms(env, observation_spec)
        self._actor_index = torch.tensor(actor_index, device=self.device)
        self.actor_dim = len(actor_index)
        # Replay rows hold the critic observation followed by the actor observation.
        self.replay_obs_dim = self.obs_dim + self.actor_dim

        # Replace only the parent's actor, which read the critic observation.  The critic, target
        # critic, temperature, reward normalization, exploration process, and update rule remain
        # the FlashSAC implementation.
        self._actor = self._init_actor()
        self.actor = _raw(self._actor)

        print(colored(f"[FlashSAC no-obj] actor observation dim {self.actor_dim}:", "green"))
        for line in self.actor_layout:
            print(colored(f"    {line}", "green"))
        print(colored(
            f"[FlashSAC no-obj] critic reads the {self.obs_dim}-dim observation above; "
            f"replay rows are {self.replay_obs_dim} dims.",
            "green",
        ))

    def _select_actor_terms(self, env, observation_spec):
        """The groups of `cfg.actor_terms`, the terms' indices in those groups' concatenation, and
        a readable layout."""
        names = list(self.cfg.actor_terms)
        groups = list(dict.fromkeys(name.split("/", 1)[0] for name in names))
        undefined = [group for group in groups if observation_spec.get(group, None) is None]
        if undefined:
            raise ValueError(f"actor_terms use observation groups {undefined} that this task does not define")
        spans, offset = {}, 0
        for group in groups:
            for term, func in env.observation_funcs[group].funcs.items():
                dim = func().reshape(self.num_envs, -1).shape[-1]
                spans[f"{group}/{term}"] = (offset, dim)
                offset += dim
        missing = [name for name in names if name not in spans]
        if missing:
            raise ValueError(f"actor_terms {missing} are not defined by this task")
        index, layout = [], []
        for name in names:
            start, dim = spans[name]
            index.extend(range(start, start + dim))
            layout.append(f"{name} {dim}")
        unread = [name for name in spans if name not in names]
        if unread:
            print(colored(f"[FlashSAC no-obj] the actor does not read {unread}", "yellow"))
        return groups, index, layout

    def _init_actor(self):
        # upstream _init_flashsac_networks' actor, with the actor observation as its input
        cfg = self.flash_cfg
        actor_net = FlashSACActor(
            num_blocks=cfg.actor_num_blocks,
            input_dim=self.actor_dim,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=self.action_dim,
        ).to(self.device)
        use_fused = self.device.type == "cuda" and torch.cuda.is_available()
        optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
        schedule = ramp_cosine_decay_scheduler(
            init_value=cfg.learning_rate_init,
            peak_value=cfg.learning_rate_peak,
            end_value=cfg.learning_rate_end,
            ramp_steps=cfg.learning_rate_ramp_step,
            decay_steps=cfg.learning_rate_decay_step,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: schedule(step) / cfg.learning_rate_peak
        )
        actor = Network(
            network=actor_net,
            optimizer=optimizer,
            scheduler=scheduler,
            compile_network=cfg.use_compile,
            compile_mode=cfg.compile_mode,
            use_weight_normalization=True,
        )
        if cfg.use_compile:
            actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)
        actor.normalize_parameters()
        return actor

    def get_rollout_policy(self, mode: str = "train"):
        return FlashSACVelNoObjRollout(self, mode)

    def actor_obs(self, tensordict: TensorDictBase) -> torch.Tensor:
        batch_size = tensordict.batch_size
        obs = torch.cat([tensordict[k].reshape(*batch_size, -1).float() for k in self.actor_groups], dim=-1)
        return obs.index_select(-1, self._actor_index)

    def replay_obs(self, tensordict: TensorDictBase) -> torch.Tensor:
        return torch.cat([self.flat_obs(tensordict), self.actor_obs(tensordict)], dim=-1)

    @torch.no_grad()
    def add_transition(self, tensordict: TensorDictBase):
        next_td = tensordict["next"]
        if ACTION_DR_KEY in self.obs_keys:
            # the next observation belongs to the same episode even if the env has reset since
            next_td.set(ACTION_DR_KEY, tensordict[ACTION_DR_KEY])

        if self.buffer is None:
            storage_device = "cpu" if self.cfg.buffer_device_type == "cpu" else self.device
            if torch.device(storage_device).type == "cuda":
                capacity = max(
                    self.cfg.buffer_max_length // tensordict.shape[0], self.cfg.n_step + 2
                ) * tensordict.shape[0]
                obs_bytes = torch.finfo(getattr(torch, self.cfg.buffer_obs_dtype)).bits // 8
                required = capacity * (
                    self.replay_obs_dim * obs_bytes + self.action_dim * 4 + 15
                )
                free, _ = torch.cuda.mem_get_info(self.device)
                reserve = 4 * 2**30
                if required > free - reserve:
                    fit = max(0, int(capacity * (free - reserve) / required))
                    raise MemoryError(
                        f"FlashSAC no-obj replay needs {required / 2**30:.1f} GiB but only "
                        f"{free / 2**30:.1f} GiB is free; keeping 4 GiB for training requires "
                        f"algo.buffer_max_length about {fit} or less, or set "
                        "algo.buffer_device_type=cpu."
                    )
            self.buffer = NStepTrajectoryReplay(
                num_envs=tensordict.shape[0],
                obs_dim=self.replay_obs_dim,
                action_dim=self.action_dim,
                n_step=self.cfg.n_step,
                gamma=self.cfg.gamma,
                max_length=self.cfg.buffer_max_length,
                min_length=self.cfg.buffer_min_length,
                batch_size=self.cfg.sample_batch_size,
                device=self.device,
                obs_dtype=getattr(torch, self.cfg.buffer_obs_dtype),
                storage_device=storage_device,
            )
            print(colored(
                f"[FlashSAC no-obj] replay buffer: {self.buffer.capacity} rows, "
                f"{self.buffer.nbytes() / 2**30:.2f} GiB on {self.buffer.storage}",
                "green",
            ))

        reward = next_td["reward"].sum(-1)
        terminated = next_td["terminated"].squeeze(-1)
        truncated = next_td["truncated"].squeeze(-1)
        time_limit = next_td["step_count"].squeeze(-1) >= self.env.max_episode_length
        cut = terminated
        if not self.cfg.bootstrap_on_command_finished:
            cut = cut | (truncated & ~time_limit)
        self.buffer.add(
            obs=self.replay_obs(tensordict) if self.buffer.step == 0 or self.real_reset_observations else None,
            next_obs=self.replay_obs(next_td),
            action=tensordict[U_KEY],
            reward=reward,
            discount=next_td["discount"].squeeze(-1),
            done=terminated | truncated,
            cut=cut,
            valid=(torch.ones_like(terminated) if self.real_reset_observations
                   else tensordict["step_count"].squeeze(-1) > 1),
            reset_mask=tensordict["is_init"].squeeze(-1) if self.real_reset_observations else None,
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward, terminated, truncated)

    def _update_once(self):
        batch = self.buffer.sample()
        observation, next_observation = batch["observation"], batch["next_observation"]
        batch["observation"] = observation[..., :self.obs_dim]
        batch["next_observation"] = next_observation[..., :self.obs_dim]
        batch["actor_observation"] = observation[..., self.obs_dim:]
        batch["actor_next_observation"] = next_observation[..., self.obs_dim:]
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self.flash_cfg,
            do_actor_update=(self._update_step % self.flash_cfg.actor_update_period == 0),
            device=self.device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1
        return info

    def state_dict(self):
        state = super().state_dict()
        state["actor_layout"] = self.actor_layout
        return state

    def load_state_dict(self, state_dict, strict=True):
        saved_actor_layout = state_dict.get("actor_layout")
        if saved_actor_layout != self.actor_layout:
            raise ValueError(
                f"checkpoint actor reads {saved_actor_layout or 'an unrecorded layout'}, but the current "
                f"algo.actor_terms give {self.actor_layout}"
            )
        saved_critic_layout = state_dict.get("obs_layout")
        if saved_critic_layout is not None and saved_critic_layout != self.obs_layout:
            raise ValueError("checkpoint critic observation layout does not match the current config")
        saved_seam = state_dict.get("action_seam")
        if saved_seam is not None and tuple(saved_seam) != self._action_seam():
            raise ValueError(f"checkpoint action seam {tuple(saved_seam)} != {self._action_seam()}")
        for option in ("critic_state_encoder_dim", "critic_action_encoder_dim"):
            if state_dict.get(option) != getattr(self.cfg, option):
                raise ValueError(f"checkpoint {option} does not match the current config")
        uses_critic_encoder = self.cfg.critic_state_encoder_dim or self.cfg.critic_action_encoder_dim
        if uses_critic_encoder and state_dict.get("critic_encoder_arch") != "relu_branches_v1":
            raise ValueError("checkpoint critic encoder architecture does not match the current config")

        for name, network in self._networks().items():
            _raw(network).load_state_dict(state_dict[name], strict=strict)
            if network.optimizer is not None and f"{name}_optimizer" in state_dict:
                network.optimizer.load_state_dict(state_dict[f"{name}_optimizer"])
                network.scheduler.load_state_dict(state_dict[f"{name}_scheduler"])

        if self.reward_normalizer is not None and "reward_normalizer" in state_dict:
            rn, saved = self.reward_normalizer, state_dict["reward_normalizer"]
            if saved["G_r"].numel() == self.num_envs:
                rn.G_r = saved["G_r"].to(self.device)
            rn.G_r_max = saved["G_r_max"].to(self.device)
            rn.G_rms.mean = saved["G_rms_mean"].to(self.device)
            rn.G_rms.var = saved["G_rms_var"].to(self.device)
            rn.G_rms.count = saved["G_rms_count"].to(self.device)
        self._update_step = state_dict.get("update_step", 0)
        if "grad_scaler" in state_dict:
            self._grad_scaler.load_state_dict(state_dict["grad_scaler"])
        self.env.set_progress(state_dict.get("last_iter", 0))
        print(colored(f"[FlashSAC no-obj] loaded checkpoint at update step {self._update_step}.", "green"))
        return []
