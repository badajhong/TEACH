"""TGRL on top of the FlashSAC student finetune (copy of flashsac_vel_finetune's flow).

TGRL (Shenfeld et al., "TGRL: An Algorithm for Teacher Guided Reinforcement Learning", ICML 2023;
reference: idanshen/cleanrl tgrl_continuous_action.py) trains two student actors on one replay:

    pi(h)    main policy:      maximizes Q_R + c (Q_I - cost(s, h))     (+ entropy), exported
    pi_R(h)  auxiliary policy: maximizes Q_R only                        (+ entropy)
    cost(s, h) = imitation cost of pi vs the frozen teacher mu(s), clipped at C = tgrl_cost_clip
                 "kl" (default): sum_j KL(N(m_theta, s_theta) || N(m_T, s_T)) (pre-tanh)
                 "mse" (optional): sum_j (tanh m_T,j(s) - tanh m_theta,j(h))^2
                 and scaled by G_max (1 - gamma) / C, so the discounted sum fits Q_I's support
    Q_R: the FlashSAC categorical critic, target R^(n) + gamma^n (1 - d)(z - alpha log pi(a'|h'))
    Q_I: a second FlashSAC categorical critic (same class/support/EMA target as Q_R), target
         gamma (1 - d_1)(z - cost(s_{t+1}, h_{t+1})), a' ~ pi
    pi and pi_R are both the FlashSAC student actor (actor_adapt) architecture.
    c (teacher_coef) follows the constraint J_R(pi) >= J_R(pi_R): every
    teacher_coef_update_iters iterations c += step if pi's mean episode return beats pi_R's,
    else c -= step, clipped to [0, teacher_coef_max] (the reference code's simplified rule).

SAC rollouts split the envs: the first half acts with pi, the second half with pi_R.
Q_I uses the same sampled transition's one-step successor, so its Bellman recursion includes
every future imitation cost while Q_R retains FlashSAC's n-step reward target.
Everything else (warmup cycles, perception training, replay, rollout) follows flashsac_vel_finetune.
Actions keep the teacher's residual/absolute convention, including ref_joint_pos_.
"""

import copy
import functools
from dataclasses import dataclass
from typing import List
from unittest import mock

import torch
from torch import nn, optim
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential as Seq
from torchrl.data import Unbounded
from torchrl.envs.transforms import TensorDictPrimer

from .common import ACTION_KEY, OBS_KEY, CatTensors, make_batch, make_conv
from .flashsac_vel import FlashSACVel, FlashSACVelConfig
from .flashsac_vel_flat import (
    ACTION_DR_KEY, U_KEY, EncoderDoubleCritic, FlashSACVelFlat, NStepTrajectoryReplay, _raw,
)
from .flashsac_upstream import agent as flashsac_agent
from .flashsac_upstream.agent import _init_flashsac_networks, _sample_flashsac_actions, _sample_integer_from_cdf
from .flashsac_upstream.network import FlashSACDoubleCritic
from .flashsac_upstream.distribution import safe_tanh_log_det_jacobian
from .flashsac_upstream.update import (
    _compute_categorical_td_target, _select_min_q_log_probs, update_target_network, update_temperature,
)
from .flashsac_upstream.scheduler import warmup_cosine_decay_scheduler
from .flashsac_upstream.utils_network import Network
from .ppo_vel import (
    DEPTH_KEY, VEL_CMD_KEY, OBJECT_KEY, OBJECT_PRED_KEY, OBJECT_PRED_TRANS_KEY,
    PRIV_FEATURE_KEY, PRIV_PRED_KEY, REF_JPOS_KEY, TemporalDepthGRU,
)
from ..modules.rnn import set_recurrent_mode


@dataclass
class TGRLFlashSACVelFinetuneConfig(FlashSACVelConfig):
    _target_: str = "active_adaptation.learning.ppo.tgrl_flashsac_vel_finetune.TGRLFlashSACVelFinetune"
    name: str = "tgrl_flashsac_vel_finetune"
    num_envs: int = 4096
    # Depth comes from CUDA ray casting; headless training needs no RTX renderer.
    # train.py enables rendering again for eval_render=true.
    enable_rtx: bool = False
    updates_per_interaction_step: float = 8.0
    # First cycle (always runs): teacher warmup set, then the first SAC phase.
    perception_warmup_iters: int = 500
    rl_phase_iters: int = 1000
    # Repeated cycles after the first: a warmup set of cycle_warmup_iters, then a SAC phase of
    # cycle_rl_phase_iters. num_cycles counts these repeats only; 0 = one-shot warmup.
    # After the last repeat, SAC runs to the end of training.
    num_cycles: int = 10
    cycle_warmup_iters: int = 300
    cycle_rl_phase_iters: int = 500
    # Cycle warmup sets: "perception_only" rolls out the frozen student SAC actor
    # (temperature 1.0) and trains perception only; "original" repeats the teacher warmup.
    cycle_warmup_mode: str = "perception_only"
    # Stage 1 rollout noise: u = tanh(mean + std * noise * scale). Stage 2 SAC always uses 1.0.
    teacher_perception_rollout_noise_scale: float = 1.6  # first set (and "original" cycles)
    student_perception_rollout_noise_scale: float = 1.0  # "perception_only" cycle sets
    adapt_epochs: int = 2
    perception_ema_tau: float = 0.08
    in_keys: List[str] = (*FlashSACVelConfig.in_keys, DEPTH_KEY)
    # Cosine decay from learning_rate_peak to learning_rate_end over all SAC updates of the run
    # (length from total_frames, i.e. num_env_steps, and the cycle schedule).
    learning_rate_end: float = 1e-5
    buffer_max_length: int = 10_000_000
    # Rows before SAC updates start, after every replay reset (first SAC phase and each cycle).
    # ~one episode per env (4096 envs x ~635 steps), so updates never see only reset states.
    buffer_min_length: int = 2_600_000
    buffer_device_type: str = "cpu"
    # Match the linked TGRL implementation's KL(student || teacher), in pre-tanh space.
    # Mean-action MSE remains an optional alternative; KL is not the paper's cross-entropy.
    tgrl_cost: str = "kl"
    # Bound and scale the per-step cost to fit FlashSAC's categorical support. At the clip,
    # cost - Q_I approaches normalized_G_max with no episode end or extra environment discount.
    tgrl_cost_clip: float = 40.0
    teacher_coef_init: float = 1.0
    teacher_coef_update: float = 0.05  # step size of the sign rule
    teacher_coef_max: float = 10.0
    teacher_coef_update_iters: int = 50  # iterations of finished episodes per comparison
    teacher_coef_min_episodes: int = 1000  # per policy, else the comparison is skipped


ConfigStore.instance().store(
    "tgrl_flashsac_vel_finetune", node=TGRLFlashSACVelFinetuneConfig, group="algo"
)


class TGRLFinetuneRollout(TensorDictModuleBase):
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
            raise RuntimeError("tgrl_flashsac_vel_finetune requires checkpoint_path to a teacher or finetune checkpoint")
        if ACTION_DR_KEY in p.replay_keys:
            td[ACTION_DR_KEY] = p.action_dr()
        p._perceive(td, ema=True)
        teacher_rollout = self.train_mode and p.stage == 1 and not p.perception_only_warmup
        if teacher_rollout:
            network, observation = p._actor, p.replay_obs(td)
        else:
            network = p._student if p._student is not None else p._student_rollout
            observation = p._actor_adapt_input(td)
        if self.train_mode and p.stage == 2:
            u = p._sample_split_rollout(observation)
        elif self.train_mode:
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


class TGRLFlashSACVelFinetune(FlashSACVel):
    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.perception_warmup_iters < 0 or cfg.adapt_epochs < 1:
            raise ValueError("perception_warmup_iters must be >= 0 and adapt_epochs >= 1")
        schedule = (cfg.rl_phase_iters, cfg.cycle_warmup_iters, cfg.cycle_rl_phase_iters)
        if cfg.num_cycles < 0 or min(schedule) < 0:
            raise ValueError("num_cycles, rl_phase_iters and cycle_*_iters must be >= 0")
        if cfg.num_cycles > 0 and min(schedule) < 1:
            raise ValueError("repeated cycles need rl_phase_iters, cycle_warmup_iters and "
                             "cycle_rl_phase_iters >= 1")
        if cfg.tgrl_cost not in ("mse", "kl") or not cfg.tgrl_cost_clip > 0:
            raise ValueError("tgrl_cost must be 'mse' or 'kl' and tgrl_cost_clip > 0")
        if not 0 <= cfg.teacher_coef_init <= cfg.teacher_coef_max or cfg.teacher_coef_update < 0:
            raise ValueError("need 0 <= teacher_coef_init <= teacher_coef_max and teacher_coef_update >= 0")
        if cfg.teacher_coef_update_iters < 1 or cfg.teacher_coef_min_episodes < 1:
            raise ValueError("teacher_coef_update_iters and teacher_coef_min_episodes must be >= 1")
        if cfg.cycle_warmup_mode not in ("perception_only", "original"):
            raise ValueError("cycle_warmup_mode must be 'perception_only' or 'original'")
        for name in ("teacher_perception_rollout_noise_scale", "student_perception_rollout_noise_scale"):
            if not 0 <= getattr(cfg, name) < float("inf"):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 < cfg.perception_ema_tau <= 1:
            raise ValueError("perception_ema_tau must be in (0, 1]")
        if cfg.adapt_module != "gru" or not cfg.use_object_adapt:
            raise ValueError("tgrl_flashsac_vel_finetune requires GRU and object adaptation")
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
        self._aux = None
        self.teacher_coef = float(cfg.teacher_coef_init)
        self._student_rollout = Network(self.actor_adapt)
        self._adapt_buffer, self._adapt_t = None, 0
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
        print(f"[FlashSAC finetune] first cycle: warmup {cfg.perception_warmup_iters} + SAC "
              f"{cfg.rl_phase_iters} iterations; then {cfg.num_cycles} x (warmup "
              f"{cfg.cycle_warmup_iters} + SAC {cfg.cycle_rl_phase_iters}) "
              f"(cycle mode {cfg.cycle_warmup_mode}); "
              f"student replay {self.student_replay_dim}, critic input {self.obs_dim} dims.")
        # The first half of the envs acts with pi, the second half with pi_R during SAC.
        self._aux_envs = torch.arange(self.num_envs, device=self.device) >= self.num_envs // 2
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._reset_return_stats()
        self._build_imitation_critic()
        print(f"[TGRL] cost {cfg.tgrl_cost}, teacher_coef_init {cfg.teacher_coef_init}, "
              f"step {cfg.teacher_coef_update} every {cfg.teacher_coef_update_iters} iterations.")

    @property
    def perception_only_warmup(self):
        return (self.stage == 1 and self.cycles_completed > 0 and
                self.cfg.cycle_warmup_mode == "perception_only")

    def _warmup_length(self):
        return self.cfg.perception_warmup_iters if self.cycles_completed == 0 else self.cfg.cycle_warmup_iters

    def _rl_phase_length(self):
        return self.cfg.rl_phase_iters if self.cycles_completed == 0 else self.cfg.cycle_rl_phase_iters

    def _rollout_noise_scale(self, teacher_rollout):
        if teacher_rollout:
            return self.cfg.teacher_perception_rollout_noise_scale
        if self.perception_only_warmup:
            return self.cfg.student_perception_rollout_noise_scale
        return 1.0

    def _build_imitation_critic(self):
        """Q_I: a fresh FlashSAC critic + EMA target, built exactly like Q_R (FlashSACVelFlat)."""
        critic_cls = FlashSACDoubleCritic
        if self.cfg.critic_state_encoder_dim or self.cfg.critic_action_encoder_dim:
            critic_cls = functools.partial(
                EncoderDoubleCritic, state_dim=self.obs_dim,
                state_feature_dim=self.cfg.critic_state_encoder_dim,
                action_feature_dim=self.cfg.critic_action_encoder_dim,
            )
        with mock.patch.object(flashsac_agent, "FlashSACDoubleCritic", critic_cls):
            _, self._imitation_critic, self._imitation_target_critic, _ = _init_flashsac_networks(
                actor_observation_dim=self.obs_dim, critic_observation_dim=self.obs_dim,
                action_dim=self.action_dim, cfg=self.flash_cfg, device=self.device,
            )
        self.imitation_critic = _raw(self._imitation_critic)
        self.imitation_target_critic = _raw(self._imitation_target_critic)
        self._cost_scale = self.cfg.normalized_G_max * (1 - self.cfg.gamma) / self.cfg.tgrl_cost_clip

    def _reset_return_stats(self):
        # Finished-episode returns since the last teacher_coef comparison: [pi, pi_R].
        self._return_sum = torch.zeros(2, device=self.device)
        self._return_cnt = torch.zeros(2, device=self.device)
        self._coef_iters = 0
        self._last_perf_diff = float("nan")

    def _sample_split_rollout(self, observation):
        """FlashSAC's repeated-noise sampling, with pi on the first and pi_R on the second half."""
        mean, std = self._student.apply("get_mean_and_std", observation, training=False)
        # Compiled actors can reuse CUDA Graph output storage. Preserve the main policy's
        # outputs outside the compiled call before invoking the auxiliary actor.
        mean, std = mean.clone(), std.clone()
        aux_mean, aux_std = self._aux.apply("get_mean_and_std", observation, training=False)
        aux = self._aux_envs.unsqueeze(-1)
        mean, std = torch.where(aux, aux_mean, mean), torch.where(aux, aux_std, std)
        reinit = (self._cur_noise_repeat_count == 0) | (self._cur_noise_repeat_count >= self._cur_noise_repeat_n)
        self._cached_noise = torch.where(reinit, torch.randn_like(mean), self._cached_noise)
        self._cur_noise_repeat_n = torch.where(
            reinit, _sample_integer_from_cdf(self._zeta_cdf), self._cur_noise_repeat_n)
        self._cur_noise_repeat_count = torch.where(
            reinit, torch.zeros_like(self._cur_noise_repeat_count), self._cur_noise_repeat_count) + 1
        return torch.tanh(mean + std * self._cached_noise)

    def _imitation_cost(self, mean, std, teacher_mean, teacher_std):
        """Per-sample imitation cost of the student distribution vs the teacher (pre-tanh inputs)."""
        mean, std = mean.float(), std.float()
        if self.cfg.tgrl_cost == "mse":
            cost = (torch.tanh(teacher_mean) - torch.tanh(mean)).square().sum(-1)
        else:
            cost = (torch.log(teacher_std / std) +
                    (std.square() + (mean - teacher_mean).square()) / (2 * teacher_std.square()) - 0.5).sum(-1)
        return cost.clamp(min=0.0, max=self.cfg.tgrl_cost_clip) * self._cost_scale

    def _update_teacher_coef(self):
        """Reference sign rule on J_R(pi) - J_R(pi_R), from finished episode returns."""
        self._coef_iters += 1
        if self._coef_iters < self.cfg.teacher_coef_update_iters:
            return
        counts = self._return_cnt.tolist()
        if min(counts) >= self.cfg.teacher_coef_min_episodes:
            main_return, aux_return = (self._return_sum / self._return_cnt).tolist()
            self._last_perf_diff = main_return - aux_return
            step = self.cfg.teacher_coef_update if self._last_perf_diff > 0 else -self.cfg.teacher_coef_update
            self.teacher_coef = min(max(self.teacher_coef + step, 0.0), self.cfg.teacher_coef_max)
            self._return_sum.zero_()
            self._return_cnt.zero_()
        self._coef_iters = 0

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
        return TGRLFinetuneRollout(self, mode)

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
            step = td.select(*keys)
            # Teacher labels for _train_warmup, computed per step (the teacher is frozen), so
            # neither the full replay observation nor one large teacher call is needed later.
            raw = self.replay_obs(td)
            step[PRIV_FEATURE_KEY] = self.actor.encode_priv(raw)
            mean, _ = self.actor.get_mean_and_std(raw, training=False)
            step["_teacher_action"] = self._joint_action(torch.tanh(mean), td)
            if self._adapt_t == 0:
                # One [num_envs, train_every] buffer per iteration, filled in place.
                n, t = step.shape[0], self.cfg.train_every
                self._adapt_buffer = TensorDict(
                    {k: torch.empty(n, t, *v.shape[1:], dtype=v.dtype, device=v.device)
                     for k, v in step.items()}, batch_size=[n, t], device=self.device)
            self._adapt_buffer[:, self._adapt_t] = step
            self._adapt_t += 1
            if self._adapt_t == self.cfg.train_every:
                self._pending_adapt, self._adapt_buffer, self._adapt_t = self._adapt_buffer, None, 0
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
        # Raw task return per finished episode, grouped by the acting policy (pi / pi_R).
        self._episode_return += reward
        finished = (terminated | truncated).float()
        group = self._aux_envs.long()
        self._return_sum.index_add_(0, group, self._episode_return * finished)
        self._return_cnt.index_add_(0, group, finished)
        self._episode_return *= 1.0 - finished
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
            include_one_step=True,
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
                if self.warmup_iters_completed >= self._warmup_length():
                    self._start_rl()
            return
        FlashSACVelFlat.update(self)
        if self._rl_steps_in_iteration == 0:
            self._update_teacher_coef()
        if (self.cycles_completed < self.cfg.num_cycles and
                self._rl_steps_in_iteration == 0 and
                self.rl_phase_iters_completed >= self._rl_phase_length()):
            self.cycles_completed += 1
            self._start_warmup()

    @set_recurrent_mode(True)
    def _train_warmup(self, rollout):
        # rollout already holds the teacher labels PRIV_FEATURE_KEY and _teacher_action.
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
        # Auxiliary reward-only policy pi_R, starting from the same warm-started student
        # (copied before the student's get_mean_and_std is compiled below).
        aux_actor = copy.deepcopy(self.actor_adapt)
        # Existing critic/temperature optimizers are fresh when loading a teacher. They
        # have not stepped during warmup. Student SAC gets a separate fresh optimizer.
        optimizer = optim.Adam(self.actor_adapt.parameters(), lr=self.cfg.learning_rate_peak,
                               fused=self.device.type == "cuda")
        te, horizon = self.cfg.train_every, self.cfg.num_env_steps // self.num_envs
        # RL env steps in the horizon: first SAC phase, each repeated cycle's SAC phase, final SAC.
        elapsed, steps = self.cfg.perception_warmup_iters * te, 0
        phase = self.cfg.rl_phase_iters * te
        for _ in range(self.cfg.num_cycles):
            if elapsed + phase >= horizon:
                break
            steps += phase
            elapsed += phase + self.cfg.cycle_warmup_iters * te
            phase = self.cfg.cycle_rl_phase_iters * te
        steps = max(1, steps + max(0, horizon - elapsed))
        updates = schedule_updates or max(1, int(steps * self.cfg.updates_per_interaction_step))
        self._rl_schedule_updates = updates

        def lr_lambda(num_steps):
            schedule = warmup_cosine_decay_scheduler(
                init_value=self.cfg.learning_rate_init, peak_value=self.cfg.learning_rate_peak,
                end_value=self.cfg.learning_rate_end,
                warmup_steps=int(self.cfg.learning_rate_warmup_rate * num_steps),
                decay_steps=max(1, int(self.cfg.learning_rate_decay_rate * num_steps)),
            )
            return lambda step: schedule(step) / self.cfg.learning_rate_peak

        # Initialize schedules once; later cycles retain their SAC update progress. The critic
        # steps on every update, the actor and temperature every actor_update_period updates.
        actor_updates = max(1, updates // self.flash_cfg.actor_update_period)
        self._critic.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._critic.optimizer, lr_lambda=lr_lambda(updates))
        self._imitation_critic.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._imitation_critic.optimizer, lr_lambda=lr_lambda(updates))
        self._temperature.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._temperature.optimizer, lr_lambda=lr_lambda(actor_updates))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda(actor_updates))
        self._student = Network(
            self.actor_adapt, optimizer, scheduler,
            compile_network=self.cfg.use_compile, compile_mode=self.flash_cfg.compile_mode,
            use_weight_normalization=True,
        )
        if self.cfg.use_compile:
            self._student.network.get_mean_and_std = torch.compile(
                self._student.network.get_mean_and_std, mode=self.flash_cfg.compile_mode)
        self._student.normalize_parameters()
        aux_optimizer = optim.Adam(aux_actor.parameters(), lr=self.cfg.learning_rate_peak,
                                   fused=self.device.type == "cuda")
        self._aux = Network(
            aux_actor, aux_optimizer,
            torch.optim.lr_scheduler.LambdaLR(aux_optimizer, lr_lambda=lr_lambda(actor_updates)),
            compile_network=self.cfg.use_compile, compile_mode=self.flash_cfg.compile_mode,
            use_weight_normalization=True,
        )
        if self.cfg.use_compile:
            self._aux.network.get_mean_and_std = torch.compile(
                self._aux.network.get_mean_and_std, mode=self.flash_cfg.compile_mode)
        self._aux.normalize_parameters()
        self._update_step = 0

    def _reset_phase_rollout(self):
        self.buffer = None
        self._adapt_buffer, self._adapt_t = None, 0
        self._pending_adapt = None
        self._rl_steps_in_iteration = 0
        self._update_counter = 0.
        self._cur_noise_repeat_count.zero_()
        self._cur_noise_repeat_n.fill_(1)
        if self.reward_normalizer is not None:
            self.reward_normalizer.G_r.zero_()
        # Environments reset, so unfinished episode returns are dropped.
        self._episode_return.zero_()
        self._reset_return_stats()
        self.needs_env_reset = True
        if self.device.type == "cuda":
            # Return the finished phase's cached blocks so the simulator (PhysX) can use them.
            torch.cuda.empty_cache()

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
        for key, actor_key, teacher_key in (
                ("observation", "actor_observation", "teacher_observation"),
                ("next_observation", "actor_next_observation", "teacher_next_observation"),
                ("one_step_next_observation", "actor_one_step_next_observation",
                 "teacher_one_step_next_observation")):
            obs = batch[key]
            batch[actor_key] = obs.index_select(-1, self._student_replay_index)
            batch[teacher_key] = obs[..., :self.replay_obs_dim]  # the teacher's own input
            batch[key] = self.critic_obs(obs)
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        info = self._tgrl_update_networks(
            batch, do_actor_update=self._update_step % self.flash_cfg.actor_update_period == 0)
        self._update_step += 1
        return info

    @staticmethod
    def _sample(network, observations, training):
        """FlashSAC's tanh-Gaussian policy sample, also returning the pre-tanh mean and std."""
        mean, std = network.apply("get_mean_and_std", observations, training=training)
        # These distributions also survive calls to the other actor/critics during training.
        # clone() owns the storage while retaining the policy gradient through the compiled call.
        mean, std = mean.clone(), std.clone()
        raw = mean + std * torch.randn_like(mean)
        log_prob = torch.distributions.Normal(mean, std).log_prob(raw) - safe_tanh_log_det_jacobian(raw)
        return torch.tanh(raw), log_prob.sum(-1), mean, std

    def _optimizer_step(self, network, loss):
        # Same step as flashsac_upstream.update (shared AMP grad scaler, LR schedule, weight norm).
        network.optimizer.zero_grad(set_to_none=True)
        if self.flash_cfg.use_amp:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.step(network.optimizer)
            self._grad_scaler.update()
        else:
            loss.backward()
            network.optimizer.step()
        if network.scheduler is not None:
            network.scheduler.step()

    def _tgrl_update_networks(self, batch, do_actor_update):
        """flashsac_upstream.agent._update_networks with TGRL's two actors and Q_I critic."""
        cfg, device = self.flash_cfg, self.device
        coef = self.teacher_coef
        student, aux, critic, target_critic = self._student, self._aux, self._critic, self._target_critic
        with torch.no_grad():  # teacher mu(s), mu(s') from the frozen fully observable actor
            teacher_obs = torch.cat([batch["teacher_observation"], batch["teacher_one_step_next_observation"]])
            teacher_mean, teacher_std = self.actor.get_mean_and_std(teacher_obs, training=False)
            teacher_mean, teacher_std = teacher_mean.float().clone(), teacher_std.float().clone()
            mu, mu_next = teacher_mean.chunk(2)
            sigma, sigma_next = teacher_std.chunk(2)
        info = {}
        if do_actor_update:
            temperature = self._temperature().detach().clone()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
                # Match flashsac_vel_finetune's upstream actor update: BatchNorm sees both
                # current and n-step next observations; only the current half enters the loss.
                actor_obs_all = torch.cat([batch["actor_observation"], batch["actor_next_observation"]])
                actions, log_probs, mean, std = (
                    value.chunk(2, dim=0)[0] for value in self._sample(student, actor_obs_all, True))
                aux_actions, aux_log_probs, _, _ = (
                    value.chunk(2, dim=0)[0] for value in self._sample(aux, actor_obs_all, True))
                cost = self._imitation_cost(mean, std, mu, sigma)
                # Disable critic gradients to prevent CUDA graph overwriting (as upstream).
                critic.network.requires_grad_(False)
                qs, _ = critic(observations=batch["observation"], actions=actions, training=False)
                q = torch.minimum(qs[0], qs[1])
                aux_qs, _ = critic(observations=batch["observation"], actions=aux_actions, training=False)
                aux_q = torch.minimum(aux_qs[0], aux_qs[1])
                critic.network.requires_grad_(True)
                self._imitation_critic.network.requires_grad_(False)
                qis, _ = self._imitation_critic(observations=batch["observation"], actions=actions, training=False)
                qi = torch.minimum(qis[0], qis[1])
                self._imitation_critic.network.requires_grad_(True)
            # pi maximizes Q_R + c (Q_I - cost); pi_R maximizes Q_R only. Both keep the entropy term.
            actor_loss = (temperature * log_probs - q + coef * (cost - qi)).mean()
            aux_loss = (temperature * aux_log_probs - aux_q).mean()
            entropy = -log_probs.mean()
            info.update({"actor/loss": actor_loss.detach(), "actor/entropy": entropy.detach(),
                         "actor/mean_action": actions.detach().mean(),
                         "actor_aux/loss": aux_loss.detach(), "actor_aux/entropy": -aux_log_probs.detach().mean(),
                         "tgrl/cost": cost.detach().mean(), "tgrl/q_imitation": qi.detach().mean()})
            self._optimizer_step(student, actor_loss)
            self._optimizer_step(aux, aux_loss)
            # Finish both backwards before another compiled forward/normalization. CUDA
            # Graphs may otherwise reclaim the auxiliary actor's saved forward tensors.
            student.normalize_parameters()
            aux.normalize_parameters()
            info.update(update_temperature(
                temperature=self._temperature, entropy=entropy.detach(),
                target_entropy=cfg.temp_target_entropy))

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
            with torch.no_grad():
                next_actions, next_log_probs, _, _ = self._sample(
                    student, batch["actor_next_observation"], False)
                next_actions = next_actions.clone()
                entropy_penalty = self._temperature().detach().squeeze() * next_log_probs.clone().float()
                obs_all = torch.cat([batch["observation"], batch["next_observation"]])
                act_all = torch.cat([batch["action"], next_actions])
                qs_all, q_infos_all = target_critic(observations=obs_all, actions=act_all, training=True)
                next_qs = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs = _select_min_q_log_probs(next_qs, q_infos_all["log_prob"].chunk(2, dim=1)[1])
                target_probs = _compute_categorical_td_target(
                    target_log_probs=next_q_log_probs, reward=batch["reward"], done=batch["terminated"],
                    actor_entropy=entropy_penalty, gamma=cfg.gamma ** cfg.n_step, num_bins=cfg.critic_num_bins,
                    min_v=cfg.critic_min_v, max_v=cfg.critic_max_v)
            _, pred_q_infos = critic(observations=obs_all, actions=act_all, training=True)
            pred_log_probs = pred_q_infos["log_prob"].chunk(2, dim=1)[0]
            critic_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean()
        self._optimizer_step(critic, critic_loss)
        critic.normalize_parameters()
        update_target_network(target_network=target_critic)

        # Q_I excludes the current cost (the actor pays it directly). A one-step Bellman
        # target includes every future cost, independently of Q_R's n-step reward horizon.
        imitation_critic, imitation_target = self._imitation_critic, self._imitation_target_critic
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
            with torch.no_grad():
                next_actions, _, next_mean, next_std = self._sample(
                    student, batch["actor_one_step_next_observation"], False)
                next_actions = next_actions.clone()
                next_cost = self._imitation_cost(next_mean.clone(), next_std.clone(), mu_next, sigma_next)
                imitation_obs_all = torch.cat([batch["observation"], batch["one_step_next_observation"]])
                imitation_act_all = torch.cat([batch["action"], next_actions])
                qis_all, qi_infos_all = imitation_target(
                    observations=imitation_obs_all, actions=imitation_act_all, training=True)
                next_qi_log_probs = _select_min_q_log_probs(
                    qis_all.chunk(2, dim=1)[1], qi_infos_all["log_prob"].chunk(2, dim=1)[1])
                qi_target_probs = _compute_categorical_td_target(
                    target_log_probs=next_qi_log_probs, reward=torch.zeros_like(batch["reward"]),
                    done=batch["one_step_terminated"], actor_entropy=next_cost.float(),
                    gamma=cfg.gamma, num_bins=cfg.critic_num_bins,
                    min_v=cfg.critic_min_v, max_v=cfg.critic_max_v)
            _, pred_qi_infos = imitation_critic(
                observations=imitation_obs_all, actions=imitation_act_all, training=True)
            qi_loss = -(qi_target_probs.unsqueeze(0) *
                        pred_qi_infos["log_prob"].chunk(2, dim=1)[0]).sum(dim=-1).mean()
        self._optimizer_step(imitation_critic, qi_loss)
        imitation_critic.normalize_parameters()
        update_target_network(target_network=imitation_target)
        info.update({"critic/loss": critic_loss.detach(), "critic/max_penalty": entropy_penalty.max(),
                     "tgrl/q_imitation_loss": qi_loss.detach(), "tgrl/next_cost": next_cost.detach().mean()})
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
        info["tgrl/teacher_coef"] = self.teacher_coef
        info["tgrl/perf_diff"] = self._last_perf_diff
        cnt = self._return_cnt.clamp(min=1)
        info["tgrl/return_main"], info["tgrl/return_aux"] = (self._return_sum / cnt).tolist()
        return info

    def state_dict(self):
        state = super().state_dict()
        state["finetune_version"] = 5
        state["finetune_method"] = "cycles_v2"
        state["finetune_stage"] = self.stage
        state["warmup_iters_completed"] = self.warmup_iters_completed
        state["finetune_iters_completed"] = self.finetune_iters_completed
        state["student_replay_dim"] = self.student_replay_dim
        state["student_replay_index"] = self._student_replay_index.cpu()
        state["perception_warmup_iters"] = self.cfg.perception_warmup_iters
        state["rl_phase_iters"] = self.cfg.rl_phase_iters
        state["num_cycles"] = self.cfg.num_cycles
        state["cycle_warmup_iters"] = self.cfg.cycle_warmup_iters
        state["cycle_rl_phase_iters"] = self.cfg.cycle_rl_phase_iters
        state["cycle_warmup_mode"] = self.cfg.cycle_warmup_mode
        state["rl_phase_iters_completed"] = self.rl_phase_iters_completed
        state["cycles_completed"] = self.cycles_completed
        for name in ("temporal_depth_gru", "temporal_depth_gru_ema"):
            state[name] = getattr(self, name).state_dict()
        if self._student is not None:
            state["rl_schedule_updates"] = self._rl_schedule_updates
            state["student_optimizer"] = self._student.optimizer.state_dict()
            state["student_scheduler"] = self._student.scheduler.state_dict()
            state["tgrl_aux_actor"] = self._aux.network.state_dict()
            state["tgrl_aux_optimizer"] = self._aux.optimizer.state_dict()
            state["tgrl_aux_scheduler"] = self._aux.scheduler.state_dict()
        state["tgrl_imitation_critic"] = self.imitation_critic.state_dict()
        state["tgrl_imitation_target_critic"] = self.imitation_target_critic.state_dict()
        state["tgrl_imitation_critic_optimizer"] = self._imitation_critic.optimizer.state_dict()
        if self._student is not None:
            state["tgrl_imitation_critic_scheduler"] = self._imitation_critic.scheduler.state_dict()
        state["tgrl_teacher_coef"] = self.teacher_coef
        state["tgrl_cost"] = self.cfg.tgrl_cost
        return state

    def load_state_dict(self, state_dict, strict=True):
        resume = state_dict.get("finetune_version") in (1, 5)
        if "finetune_version" in state_dict and not resume:
            raise ValueError("Unsupported finetune checkpoint version")
        if state_dict.get("actor_adapt_arch") != "flashsac_v1":
            raise ValueError("Finetune requires a teacher checkpoint with FlashSAC actor_adapt")
        if resume:
            if state_dict.get("finetune_version") == 5 and state_dict.get("finetune_method") not in (
                    "repeated_original_warmup_v1", "cycles_v2"):
                raise ValueError("Unsupported finetune method")
            if state_dict["finetune_stage"] not in (1, 2):
                raise ValueError("Invalid finetune stage in checkpoint")
            if (state_dict["student_replay_dim"] != self.student_replay_dim or
                    not torch.equal(state_dict["student_replay_index"].cpu(), self._student_replay_index.cpu())):
                raise ValueError("Student observation layout differs from the saved finetune checkpoint")
            self._warn_schedule_change(state_dict)
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
                if "tgrl_aux_actor" in state_dict:  # other finetune checkpoints start pi_R from the student
                    self._aux.network.load_state_dict(state_dict["tgrl_aux_actor"])
                    self._aux.optimizer.load_state_dict(state_dict["tgrl_aux_optimizer"])
                    self._aux.scheduler.load_state_dict(state_dict["tgrl_aux_scheduler"])
            if "tgrl_imitation_critic" in state_dict:
                self.imitation_critic.load_state_dict(state_dict["tgrl_imitation_critic"])
                self.imitation_target_critic.load_state_dict(state_dict["tgrl_imitation_target_critic"])
                self._imitation_critic.optimizer.load_state_dict(state_dict["tgrl_imitation_critic_optimizer"])
                if self._student is not None and "tgrl_imitation_critic_scheduler" in state_dict:
                    self._imitation_critic.scheduler.load_state_dict(state_dict["tgrl_imitation_critic_scheduler"])
                self.teacher_coef = state_dict["tgrl_teacher_coef"]
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
              f"stage={self.stage}, warmup={self.warmup_iters_completed}/{self._warmup_length()}, "
              f"repeated cycles={self.cycles_completed}/{self.cfg.num_cycles}")
        return []

    def _warn_schedule_change(self, state):
        # The cycle schedule comes from the config, so it may change on resume (e.g. more cycles).
        keys = ("perception_warmup_iters", "rl_phase_iters", "num_cycles",
                "cycle_warmup_iters", "cycle_rl_phase_iters", "cycle_warmup_mode")
        changed = [f"{k} {state[k]} -> {getattr(self.cfg, k)}" for k in keys
                   if k in state and state[k] != getattr(self.cfg, k)]
        if changed:
            print("[FlashSAC finetune] WARNING: cycle schedule differs from the checkpoint; "
                  "continuing with the current config: " + ", ".join(changed))
