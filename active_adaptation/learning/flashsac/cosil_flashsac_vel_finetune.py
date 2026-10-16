"""CoSIL with the shared adaptive/fixed FlashSAC perception/SAC schedule.

CoSIL regularizes the partially observable student toward the frozen state expert:
    D(s, h) = sum_j (tanh m_T,j(s) - tanh m_theta,j(h))^2
    critic target penalty: beta * D(s', h')
    actor loss: E[beta * D(s, h) - min_i Q_i(s, a)]
    beta loss: beta * (target_divergence - E[D])

"cosil" replaces entropy with this penalty; "cosil_entropy" keeps SAC entropy
and adds the penalty. Perception phases, replay, final SAC tail and checkpoint
scheduling follow flashsac_vel_finetune. CoSIL updates run only in SAC phases.
With cosil_divergence_input="teacher", D evaluates the student actor with the
frozen teacher encoder's latent. Rollout, Q and entropy still use student latents.
"""

from dataclasses import dataclass

import torch
from torch import optim
from hydra.core.config_store import ConfigStore

from .flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
    FinetuneRollout as CoSILFinetuneRollout,
)
from .flashsac_upstream.distribution import safe_tanh_log_det_jacobian
from .flashsac_upstream.network import FlashSACTemperature
from .flashsac_upstream.update import (
    _compute_categorical_td_target, _select_min_q_log_probs, update_target_network, update_temperature,
)
from .flashsac_upstream.utils_network import Network


@dataclass
class CoSILFlashSACVelFinetuneConfig(FlashSACVelFinetuneConfig):
    _target_: str = "active_adaptation.learning.flashsac.cosil_flashsac_vel_finetune.CoSILFlashSACVelFinetune"
    name: str = "cosil_flashsac_vel_finetune"
    # All scheduling/perception/replay options and defaults match FlashSACVelFinetuneConfig.
    cosil_mode: str = "cosil_entropy"  # "cosil" or "cosil_entropy"
    # Latent supplied to the student actor only for D(s,h) and D(s',h').
    # "student": replayed perception output; "teacher": frozen privileged encoder.
    cosil_divergence_input: str = "student"
    # D is summed over normalized action dimensions, in [0, 4 * action_dim].
    # Beta grows above this target divergence and shrinks below it.
    target_divergence: float = 0.8
    beta_init: float = 0.01


ConfigStore.instance().store(
    "cosil_flashsac_vel_finetune", node=CoSILFlashSACVelFinetuneConfig, group="algo"
)


class CoSILFlashSACVelFinetune(FlashSACVelFinetune):
    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.cosil_mode not in ("cosil", "cosil_entropy"):
            raise ValueError("cosil_mode must be 'cosil' or 'cosil_entropy'")
        if cfg.cosil_divergence_input not in ("student", "teacher"):
            raise ValueError("cosil_divergence_input must be 'student' or 'teacher'")
        if not 0 <= cfg.target_divergence < float("inf") or not 0 < cfg.beta_init < float("inf"):
            raise ValueError("target_divergence must be finite and >= 0 and beta_init finite and > 0")
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        self._beta = None
        print(f"[CoSIL] mode {cfg.cosil_mode}, target_divergence {cfg.target_divergence}, "
              f"beta_init {cfg.beta_init}, divergence input {cfg.cosil_divergence_input}.")

    def _start_rl(self, schedule_updates=None):
        super()._start_rl(schedule_updates=schedule_updates)
        if self._beta is not None:
            return  # Retain beta and its optimizer/scheduler across perception refreshes.
        beta_net = FlashSACTemperature(self.cfg.beta_init).to(self.device)
        optimizer = optim.Adam(beta_net.parameters(), lr=self.cfg.learning_rate_peak,
                               fused=self.device.type == "cuda")
        # Beta steps with the actor. Reuse its LR function so both fixed update
        # budgets and the adaptive global-iteration clock match exactly on resume.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self._student.scheduler.lr_lambdas[0])
        self._beta = Network(beta_net, optimizer, scheduler)

    def _update_once(self):
        batch = self.buffer.sample()
        do_actor_update = self._update_step % self.flash_cfg.actor_update_period == 0
        for key, actor_key, teacher_key in (
                ("observation", "actor_observation", "teacher_observation"),
                ("next_observation", "actor_next_observation", "teacher_next_observation")):
            obs = batch[key]
            if do_actor_update or key == "next_observation":
                batch[actor_key] = obs.index_select(-1, self._student_replay_index)
                batch[teacher_key] = obs[..., :self.replay_obs_dim]  # the teacher's own input
            batch[key] = self.critic_obs(obs)
        if self.reward_normalizer is not None:
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])
        info = self._cosil_update_networks(batch, do_actor_update=do_actor_update)
        self._update_step += 1
        return info

    def _sample_student(self, observations, training):
        """Sample current actor-update rows or all bootstrap rows, with required statistics."""
        mean, std = self._student.apply("get_mean_and_std", observations, training=training)
        noise = torch.randn_like(mean)
        if training:
            # The full current/next forward is required for BatchNorm, but only
            # current actions enter the loss. Keep the full noise draw so future
            # sampling uses exactly the same RNG stream as before.
            mean, std, noise = (value.chunk(2)[0] for value in (mean, std, noise))
        raw = mean + std * noise
        log_prob = None
        if training or self.cfg.cosil_mode == "cosil_entropy":
            # Actor entropy is logged in both modes; the pure CoSIL bootstrap
            # has no entropy term and does not need a log probability.
            log_prob = (torch.distributions.Normal(mean, std).log_prob(raw)
                        - safe_tanh_log_det_jacobian(raw)).sum(-1)
        mean_action = torch.tanh(mean) if self.cfg.cosil_divergence_input == "student" else None
        return torch.tanh(raw), log_prob, mean_action

    def _divergence_mean_actions(self, observations):
        # The teacher-latent auxiliary branch must not mix privileged inputs into
        # the deployed actor's BatchNorm statistics. training=False freezes only
        # those statistics: actor gradients still flow through this mean action.
        mean, _ = self._student.apply("get_mean_and_std", observations, training=False)
        return torch.tanh(mean).float()

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

    def _cosil_update_networks(self, batch, do_actor_update):
        """flashsac_upstream.agent._update_networks with the CoSIL actor, beta and critic target."""
        cfg, device = self.flash_cfg, self.device
        entropy_on = self.cfg.cosil_mode == "cosil_entropy"
        teacher_input = self.cfg.cosil_divergence_input == "teacher"
        student, critic, target_critic = self._student, self._critic, self._target_critic
        with torch.no_grad():  # state expert mu(s), mu(s') from the frozen teacher
            # Critic-only updates need D(s',h') but never D(s,h). The frozen
            # teacher uses running BN statistics, so omitting current rows does
            # not change the next-state target or any teacher state.
            teacher_obs = (torch.cat([batch["teacher_observation"], batch["teacher_next_observation"]])
                           if do_actor_update else batch["teacher_next_observation"])
            if teacher_input:
                # Encode the needed states once for both expert actions and the
                # student's auxiliary input; no new replay fields are needed.
                feature, teacher_actions = self._label_teacher(teacher_obs)
                teacher_actions = teacher_actions.float().clone()
                prefix = batch["actor_next_observation"][..., :-self.cfg.latent_dim]
                if do_actor_update:
                    prefix = torch.cat([batch["actor_observation"][..., :-self.cfg.latent_dim], prefix])
                # cat owns the encoder output before a later compiled call reuses
                # it. Velocity commands and policy observations stay unchanged.
                divergence_next_obs = torch.cat([prefix, feature], dim=-1)
                if do_actor_update:
                    divergence_obs, divergence_next_obs = divergence_next_obs.chunk(2)
            else:
                teacher_mean, _ = self.actor.get_mean_and_std(teacher_obs, training=False)
                teacher_actions = torch.tanh(teacher_mean.float()).clone()
            if do_actor_update:
                mu, mu_next = teacher_actions.chunk(2)
            else:
                mu_next = teacher_actions
        info = {}
        if do_actor_update:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
                actor_obs_all = torch.cat([batch["actor_observation"], batch["actor_next_observation"]])
                actions, log_probs, mean_actions = self._sample_student(actor_obs_all, True)
                divergence_actions = (self._divergence_mean_actions(divergence_obs) if teacher_input
                                      else mean_actions.float())
                divergence = (mu - divergence_actions).square().sum(-1)
                # Disable critic gradients to prevent CUDA graph overwriting (as upstream).
                critic.network.requires_grad_(False)
                qs, _ = critic(observations=batch["observation"], actions=actions, training=False)
                q = torch.minimum(qs[0], qs[1])
                critic.network.requires_grad_(True)
                beta = self._beta().detach()
                actor_loss = beta * divergence - q
                if entropy_on:
                    actor_loss = actor_loss + self._temperature().detach() * log_probs
                actor_loss = actor_loss.mean()
                entropy = -log_probs.mean()
                mean_action = actions.mean()
            self._optimizer_step(student, actor_loss)
            student.normalize_parameters()
            info.update({"actor/loss": actor_loss.detach(), "actor/entropy": entropy.detach(),
                         "actor/mean_action": mean_action.detach()})
            # beta: grows while E[D] > target_divergence, shrinks below it.
            divergence_mean = divergence.detach().float().mean()
            beta_value = self._beta()
            beta_loss = beta_value * (self.cfg.target_divergence - divergence_mean)
            self._beta.optimizer.zero_grad(set_to_none=True)
            beta_loss.backward()
            self._beta.optimizer.step()
            self._beta.scheduler.step()
            info.update({"cosil/divergence": divergence_mean, "cosil/beta": beta_value.detach().squeeze(),
                         "cosil/beta_loss": beta_loss.detach().squeeze()})
            if entropy_on:
                info.update(update_temperature(
                    temperature=self._temperature, entropy=entropy.detach(),
                    target_entropy=cfg.temp_target_entropy))

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.use_amp):
            with torch.no_grad():
                next_actions, next_log_probs, next_mean_actions = self._sample_student(
                    batch["actor_next_observation"], False)
                next_actions = next_actions.clone()
                divergence_next_actions = (self._divergence_mean_actions(divergence_next_obs) if teacher_input
                                           else next_mean_actions.float())
                next_divergence = (mu_next - divergence_next_actions).square().sum(-1)
                penalty = self._beta().detach().squeeze() * next_divergence
                if entropy_on:
                    penalty = penalty + self._temperature().detach().squeeze() * next_log_probs.clone().float()
                obs_all = torch.cat([batch["observation"], batch["next_observation"]])
                act_all = torch.cat([batch["action"], next_actions])
                qs_all, q_infos_all = target_critic(observations=obs_all, actions=act_all, training=True)
                next_qs = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs = _select_min_q_log_probs(next_qs, q_infos_all["log_prob"].chunk(2, dim=1)[1])
                target_probs = _compute_categorical_td_target(
                    target_log_probs=next_q_log_probs, reward=batch["reward"], done=batch["terminated"],
                    actor_entropy=penalty, gamma=cfg.gamma ** cfg.n_step, num_bins=cfg.critic_num_bins,
                    min_v=cfg.critic_min_v, max_v=cfg.critic_max_v)
            _, pred_q_infos = critic(observations=obs_all, actions=act_all, training=True)
            pred_log_probs = pred_q_infos["log_prob"].chunk(2, dim=1)[0]
            critic_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean()
        self._optimizer_step(critic, critic_loss)
        critic.normalize_parameters()
        update_target_network(target_network=target_critic)
        info.update({"critic/loss": critic_loss.detach(), "critic/max_penalty": penalty.max()})
        return info

    def pop_info(self):
        info = super().pop_info()
        info["cosil/divergence_uses_teacher_latent"] = int(self.cfg.cosil_divergence_input == "teacher")
        if self._beta is not None:
            info["cosil/beta_value"] = self._beta().item()
        return info

    def state_dict(self):
        state = super().state_dict()
        state["cosil_mode"] = self.cfg.cosil_mode
        state["cosil_divergence_input"] = self.cfg.cosil_divergence_input
        if self._beta is not None:
            state["cosil_beta"] = self._beta.network.state_dict()
            state["cosil_beta_optimizer"] = self._beta.optimizer.state_dict()
            state["cosil_beta_scheduler"] = self._beta.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        # Translate the original CoSIL fixed-cycle checkpoint schema. New saves
        # use the shared finetune schema plus the CoSIL coefficient state.
        state = state_dict
        if state.get("finetune_version") in (1, 5):
            if state["finetune_version"] == 5 and state.get("finetune_method") not in (
                    "repeated_original_warmup_v1", "cycles_v2"):
                raise ValueError("Unsupported finetune method")
            original_refresh = (state.get("cycle_warmup_mode") == "original" or
                                state.get("finetune_method") == "repeated_original_warmup_v1")
            if original_refresh and state["finetune_stage"] == 1 and state.get("cycles_completed", 0) > 0:
                raise ValueError("Legacy teacher-rollout refresh is not supported; resume a SAC-phase "
                                 "checkpoint. New refreshes use the frozen student, as in flashsac_vel_finetune.")
            state = dict(state)
            for old, new in (
                ("perception_warmup_iters", "perception_iters"),
                ("rl_phase_iters", "sac_iters"),
                ("cycle_warmup_iters", "cycle_perception_iters"),
                ("cycle_rl_phase_iters", "cycle_sac_iters"),
                ("warmup_iters_completed", "perception_iters_completed"),
            ):
                if old in state:
                    state[new] = state[old]
            state.update(finetune_version=6, finetune_method="cycles_v2", cycle_mode=False,
                         student_perception_phase=state.get("cycles_completed", 0) > 0)
        beta_keys = ("cosil_beta", "cosil_beta_optimizer", "cosil_beta_scheduler")
        if any(key in state for key in beta_keys):
            if not all(key in state for key in beta_keys) or "student_optimizer" not in state:
                raise ValueError("Incomplete CoSIL beta checkpoint state")
        result = super().load_state_dict(state, strict=strict)
        if "cosil_beta" in state:
            self._beta.network.load_state_dict(state["cosil_beta"], strict=strict)
            self._beta.optimizer.load_state_dict(state["cosil_beta_optimizer"])
            self._beta.scheduler.load_state_dict(state["cosil_beta_scheduler"])
        if "cosil_mode" in state:
            saved_input = state.get("cosil_divergence_input", "student")
            if saved_input != self.cfg.cosil_divergence_input:
                print(f"[CoSIL] WARNING: divergence input changed from {saved_input} to "
                      f"{self.cfg.cosil_divergence_input}; continuing with the current config.")
        return result
