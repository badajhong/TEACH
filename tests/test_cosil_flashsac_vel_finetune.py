"""Real CPU updates exercise the shared schedule with both CoSIL objectives.

Run: PYTHONPATH=.:tests python -m unittest test_cosil_flashsac_vel_finetune
"""
import copy
import contextlib
from dataclasses import asdict
import io
from itertools import product
import unittest
import warnings
from unittest.mock import patch

import torch

import test_adaptive_flashsac_finetune as adaptive_fixtures
from active_adaptation.learning.flashsac import cosil_flashsac_vel_finetune as cosil_module
from active_adaptation.learning.flashsac.cosil_flashsac_vel_finetune import (
    CoSILFlashSACVelFinetune, CoSILFlashSACVelFinetuneConfig,
)
from active_adaptation.learning.flashsac.flashsac_vel_finetune import FlashSACVelFinetuneConfig
from active_adaptation.learning.flashsac.flashsac_upstream.distribution import safe_tanh_log_det_jacobian


class CoSILFinetuneTest(adaptive_fixtures.AdaptiveFinetuneTest):
    # Run the complete phase-boundary/reference/plateau/final-tail/resume suite
    # against CoSIL too; the inherited helpers execute real model updates.
    config_cls = CoSILFlashSACVelFinetuneConfig
    policy_cls = CoSILFlashSACVelFinetune

    def test_shared_config_defaults_and_cosil_validation(self):
        baseline = asdict(FlashSACVelFinetuneConfig())
        cosil = asdict(CoSILFlashSACVelFinetuneConfig())
        for key in baseline.keys() - {"_target_", "name"}:
            self.assertEqual(cosil[key], baseline[key], key)
        for overrides in ({"cosil_mode": "invalid"}, {"cosil_divergence_input": "invalid"},
                          {"target_divergence": -1},
                          {"target_divergence": float("nan")}, {"beta_init": 0},
                          {"beta_init": float("inf")}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.new_policy(**overrides)

    def test_both_cosil_modes_update_beta_and_keep_teacher_frozen(self):
        for mode, source in product(("cosil", "cosil_entropy"), ("student", "teacher")):
            with self.subTest(mode=mode, source=source):
                self.policy = self.new_policy(cosil_mode=mode, cosil_divergence_input=source)
                self.policy.load_state_dict(self.teacher_state)
                self.finish_teacher()
                p = self.policy
                beta = copy.deepcopy(p._beta.network.state_dict())
                actor = copy.deepcopy(p.actor_adapt.state_dict())
                critic = copy.deepcopy(p.critic.state_dict())
                temperature = copy.deepcopy(p.temperature.state_dict())
                self.sac_iteration()
                self.sac_iteration()
                self.assertGreater(p._update_step, 0)
                for model, before in ((p._beta.network, beta), (p.actor_adapt, actor),
                                      (p.critic, critic)):
                    self.assertTrue(any(not torch.equal(v, before[k])
                                        for k, v in model.state_dict().items()))
                temp_changed = any(not torch.equal(v, temperature[k])
                                   for k, v in p.temperature.state_dict().items())
                self.assertEqual(temp_changed, mode == "cosil_entropy")
                self.assert_model_equal(p.actor, self.teacher_state["actor"])
                info = p.pop_info()
                for key in ("cosil/beta_value", "cosil/divergence", "cosil/beta_loss",
                            "finetune/relative_error", "finetune/reference_error"):
                    self.assertIn(key, info)
                    self.assertTrue(torch.isfinite(torch.as_tensor(info[key])).all(), key)
                self.assertEqual(info["cosil/divergence_uses_teacher_latent"], int(source == "teacher"))

    def update_batch(self, p):
        batch = {"action": torch.rand(8, 3) * 2 - 1, "reward": torch.randn(8),
                 "terminated": torch.zeros(8)}
        for key, actor_key, teacher_key in (
                ("observation", "actor_observation", "teacher_observation"),
                ("next_observation", "actor_next_observation", "teacher_next_observation")):
            obs = torch.randn(8, p.student_replay_dim)
            batch[key] = p.critic_obs(obs)
            batch[actor_key] = obs.index_select(-1, p._student_replay_index)
            batch[teacher_key] = obs[..., :p.replay_obs_dim]
        return batch

    def test_sampling_skips_unused_outputs_but_keeps_gradients_bn_and_rng(self):
        for mode, source, training in product(("cosil", "cosil_entropy"),
                                               ("student", "teacher"), (False, True)):
            with self.subTest(mode=mode, source=source, training=training):
                p = self.new_policy(perception_iters=0, cosil_mode=mode, cosil_divergence_input=source)
                p.load_state_dict(self.teacher_state)
                reference = copy.deepcopy(p.actor_adapt)
                observations = torch.randn(16, p.actor_adapt.embedder.norm.running_mean.numel())
                with torch.set_grad_enabled(training):
                    torch.manual_seed(43)
                    mean, std = reference.get_mean_and_std(observations, training)
                    raw = mean + std * torch.randn_like(mean)
                    log_prob = (torch.distributions.Normal(mean, std).log_prob(raw)
                                - safe_tanh_log_det_jacobian(raw)).sum(-1)
                    expected = (raw.tanh(), log_prob, mean.tanh())
                    if training:
                        expected = tuple(value.chunk(2)[0] for value in expected)
                    expected = (expected[0], expected[1] if training or mode == "cosil_entropy" else None,
                                expected[2] if source == "student" else None)
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(43)
                    actual = p._sample_student(observations, training)
                    # Shorter elementwise kernels may round their tail lanes
                    # differently; the distribution and sampled noise are unchanged.
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                    if training:
                        sum(value.sum() for value in expected if value is not None).backward()
                        sum(value.sum() for value in actual if value is not None).backward()
                        torch.testing.assert_close([v.grad for v in p.actor_adapt.parameters()],
                                                   [v.grad for v in reference.parameters()], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(p.actor_adapt.state_dict(), reference.state_dict(), rtol=0, atol=0)

    def test_teacher_latent_routes_divergence_gradients_beta_and_target_only(self):
        for mode, actor_update in product(("cosil", "cosil_entropy"), (True, False)):
            with self.subTest(mode=mode, actor_update=actor_update):
                p = self.new_policy(perception_iters=0, cosil_mode=mode, cosil_divergence_input="teacher")
                p.load_state_dict(self.teacher_state)
                batch = self.update_batch(p)
                before = copy.deepcopy(batch)
                raw = torch.cat([batch["teacher_observation"], batch["teacher_next_observation"]])
                with torch.no_grad():
                    features = p.actor.encode_priv(raw).chunk(2)
                    mu, mu_next = p.actor.get_mean_and_std(raw, False)[0].tanh().chunk(2)
                expected_inputs = [torch.cat([batch[key][..., :-p.cfg.latent_dim], feature], -1)
                                   for key, feature in zip(("actor_observation", "actor_next_observation"), features)]
                beta_before = p._beta().detach().clone()
                beta_state = copy.deepcopy(p.state_dict())
                sample_calls, divergence_calls, gradients, penalties = [], [], [], []
                sample = p._sample_student
                mean_actions = p._divergence_mean_actions
                td_target = cosil_module._compute_categorical_td_target

                def tracked_sample(obs, training):
                    result = sample(obs, training)
                    sample_calls.append((obs.clone(), training,
                                         result[1].detach().clone() if result[1] is not None else None))
                    return result

                def tracked_divergence(obs):
                    stats = {k: v.clone() for k, v in p.actor_adapt.named_buffers()}
                    mean = mean_actions(obs)
                    # Auxiliary inputs must not alter the deployed actor's BN stats.
                    torch.testing.assert_close(dict(p.actor_adapt.named_buffers()), stats, rtol=0, atol=0)
                    divergence_calls.append((obs.clone(), mean.detach().clone()))
                    if mean.requires_grad:
                        mean.register_hook(lambda grad: gradients.append(grad.clone()))
                    return mean

                def tracked_target(**kwargs):
                    penalties.append(kwargs["actor_entropy"].clone())
                    return td_target(**kwargs)

                with patch.object(p, "_sample_student", side_effect=tracked_sample), \
                     patch.object(p, "_divergence_mean_actions", side_effect=tracked_divergence), \
                     patch.object(p.actor, "encode_priv", wraps=p.actor.encode_priv) as encode, \
                     patch.object(cosil_module, "_compute_categorical_td_target", side_effect=tracked_target):
                    info = p._cosil_update_networks(batch, do_actor_update=actor_update)
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(encode.call_args.args[0].shape[0], 16 if actor_update else 8)
                self.assertEqual(len(divergence_calls), 2 if actor_update else 1)
                if actor_update:
                    torch.testing.assert_close(divergence_calls[0][0], expected_inputs[0])
                    torch.testing.assert_close(sample_calls[0][0], torch.cat([
                        batch["actor_observation"], batch["actor_next_observation"]]))
                    self.assertTrue(sample_calls[0][1])
                    difference = divergence_calls[0][1] - mu
                    expected_d = difference.square().sum(-1).mean()
                    torch.testing.assert_close(info["cosil/divergence"], expected_d)
                    torch.testing.assert_close(info["cosil/beta_loss"],
                                               (beta_before * (p.cfg.target_divergence - expected_d)).squeeze())
                    self.assertEqual(len(gradients), 1)
                    torch.testing.assert_close(gradients[0], 2 * beta_before * difference / difference.shape[0])
                    self.assertGreater(gradients[0].abs().sum().item(), 0)
                else:
                    self.assert_beta_equal(p, beta_state)
                torch.testing.assert_close(divergence_calls[-1][0], expected_inputs[1])
                torch.testing.assert_close(sample_calls[-1][0], batch["actor_next_observation"])
                self.assertFalse(sample_calls[-1][1])
                expected_penalty = p._beta().detach().squeeze() * (mu_next - divergence_calls[-1][1]).square().sum(-1)
                if mode == "cosil_entropy":
                    expected_penalty += p._temperature().detach().squeeze() * sample_calls[-1][2]
                torch.testing.assert_close(penalties[0], expected_penalty)
                torch.testing.assert_close(batch, before, rtol=0, atol=0)
                self.assert_model_equal(p.actor, self.teacher_state["actor"])
                self.assertTrue(all(x.grad is None for x in p.actor.parameters()))
                self.assertTrue(all(x.grad is None for x in p._perception_parameters()))

    def test_divergence_source_checkpoint_metadata_and_explicit_override(self):
        p = self.new_policy(perception_iters=0, cosil_divergence_input="teacher")
        p.load_state_dict(self.teacher_state)
        saved = copy.deepcopy(p.state_dict())
        self.assertEqual(saved["cosil_divergence_input"], "teacher")
        resumed = self.new_policy(cosil_divergence_input="teacher")
        resumed.load_state_dict(saved)
        self.assert_beta_equal(resumed, saved)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.new_policy(cosil_divergence_input="student").load_state_dict(saved)
        self.assertIn("divergence input changed from teacher to student", output.getvalue())
        saved.pop("cosil_divergence_input")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            resumed.load_state_dict(saved)
        self.assertIn("divergence input changed from student to teacher", output.getvalue())

    def assert_beta_equal(self, policy, state):
        torch.testing.assert_close(policy._beta.network.state_dict(), state["cosil_beta"], rtol=0, atol=0)
        torch.testing.assert_close(policy._beta.optimizer.state_dict(), state["cosil_beta_optimizer"],
                                   rtol=0, atol=0)
        self.assertEqual(policy._beta.scheduler.state_dict(), state["cosil_beta_scheduler"])

    def test_beta_survives_refresh_and_checkpoint_resume(self):
        self.finish_teacher()
        self.sac_iteration()
        self.sac_iteration()
        p = self.policy
        saved_sac = copy.deepcopy(p.state_dict())
        resumed = self.new_policy()
        resumed.load_state_dict(saved_sac)
        self.assert_beta_equal(resumed, saved_sac)
        beta_network = p._beta
        p.cycles_completed += 1
        p._start_perception(student=True)
        self.perception_iteration()
        self.assert_beta_equal(p, saved_sac)
        saved_refresh = copy.deepcopy(p.state_dict())
        resumed.load_state_dict(saved_refresh)
        self.assertEqual(resumed.stage, 1)
        self.assert_beta_equal(resumed, saved_refresh)
        for _ in range(p.cfg.cycle_perception_iters_max):
            if p.stage == 2:
                break
            self.perception_iteration()
        self.assertEqual(p.stage, 2)
        self.assertIs(p._beta, beta_network)
        self.assert_beta_equal(p, saved_sac)
        self.sac_iteration()
        self.sac_iteration()
        self.assertGreater(p._beta.scheduler.last_epoch, saved_sac["cosil_beta_scheduler"]["last_epoch"])

    def test_fixed_cycles_and_beta_schedule(self):
        self.policy = self.new_policy(cycle_mode=False, num_cycles=1, sac_iters=2,
                                      cycle_perception_iters=2, cycle_sac_iters=3)
        self.policy.load_state_dict(self.teacher_state)
        self.perception_iteration()
        self.perception_iteration()
        p = self.policy
        beta = p._beta
        self.sac_iteration()
        self.sac_iteration()
        self.assertEqual((p.stage, p.cycles_completed), (1, 1))
        saved = copy.deepcopy(p.state_dict())
        self.perception_iteration()
        self.perception_iteration()
        self.assertEqual(p.stage, 2)
        self.assertIs(p._beta, beta)
        self.assert_beta_equal(p, saved)
        for _ in range(5):
            self.sac_iteration()
            self.assertEqual((p.stage, p.cycles_completed), (2, 1))
            self.assertEqual(p._beta.scheduler.last_epoch, p._student.scheduler.last_epoch)
            self.assertEqual(p._beta.scheduler.get_last_lr(), p._student.scheduler.get_last_lr())

    def test_beta_adaptive_lr_clock_is_restored_with_changed_budget(self):
        self.finish_teacher()
        self.sac_iteration()
        self.sac_iteration()
        original = self.policy
        saved = copy.deepcopy(original.state_dict())
        resumed = self.new_policy(num_env_steps=2 * original.cfg.num_env_steps)
        resumed.load_state_dict(saved)
        self.assert_beta_equal(resumed, saved)
        # Move past a refresh-sized gap: adaptive LR follows global progress,
        # while optimizer step counts remain unchanged across that gap.
        original.finetune_iters_completed += 10
        resumed.finetune_iters_completed = original.finetune_iters_completed
        for p in (original, resumed):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
                p._beta.scheduler.step()
                p._student.scheduler.step()
            self.assertEqual(p._beta.scheduler.get_last_lr(), p._student.scheduler.get_last_lr())
        self.assertEqual(original._beta.scheduler.get_last_lr(), resumed._beta.scheduler.get_last_lr())

    @staticmethod
    def legacy_checkpoint(policy):
        state = copy.deepcopy(policy.state_dict())
        state.update(finetune_version=5, finetune_method="cycles_v2", cycle_warmup_mode="perception_only")
        for old, new in (("perception_warmup_iters", "perception_iters"),
                         ("rl_phase_iters", "sac_iters"),
                         ("cycle_warmup_iters", "cycle_perception_iters"),
                         ("cycle_rl_phase_iters", "cycle_sac_iters"),
                         ("warmup_iters_completed", "perception_iters_completed")):
            state[old] = state.pop(new)
        for key in ("cycle_mode", "adaptive_schedule", "initial_reference_pending",
                    "adaptive_lr_start_iter", "adaptive_lr_num_steps", "student_perception_phase"):
            state.pop(key, None)
        return state

    def test_legacy_cosil_checkpoint_migration_preserves_beta_and_phase(self):
        self.policy = self.new_policy(cycle_mode=False)
        self.policy.load_state_dict(self.teacher_state)
        for phase in ("teacher", "sac", "refresh"):
            with self.subTest(phase=phase):
                p = self.policy
                if phase == "teacher":
                    self.perception_iteration()
                elif phase == "sac":
                    self.perception_iteration()
                    self.sac_iteration()
                    self.sac_iteration()
                else:
                    p.cycles_completed += 1
                    p._start_perception()
                    self.perception_iteration()
                saved = self.legacy_checkpoint(p)
                before = copy.deepcopy(saved)
                for adaptive in (False, True):
                    resumed = self.new_policy(cycle_mode=adaptive)
                    resumed.load_state_dict(saved)
                    self.assertEqual(resumed.stage, p.stage)
                    self.assertEqual(resumed.perception_iters_completed, p.perception_iters_completed)
                    self.assertEqual(resumed.rl_phase_iters_completed, p.rl_phase_iters_completed)
                    self.assertEqual(resumed.cycles_completed, p.cycles_completed)
                    if p._beta is not None:
                        self.assert_beta_equal(resumed, saved)
                    if phase == "sac" and adaptive:
                        self.assertTrue(resumed._initial_reference_pending)
                self.assertEqual(saved.keys(), before.keys())
                for key, value in before.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        self.assertEqual(saved[key], value, key)
        saved["cycle_warmup_mode"] = "original"
        with self.assertRaisesRegex(ValueError, "Legacy teacher-rollout refresh"):
            self.new_policy().load_state_dict(saved)

    def test_plain_finetune_checkpoint_starts_beta_fresh_and_partial_beta_is_rejected(self):
        self.finish_teacher()
        self.sac_iteration()
        self.sac_iteration()
        saved = copy.deepcopy(self.policy.state_dict())
        for key in ("cosil_mode", "cosil_beta", "cosil_beta_optimizer", "cosil_beta_scheduler"):
            saved.pop(key)
        resumed = self.new_policy(beta_init=.05)
        resumed.load_state_dict(saved)
        self.assertAlmostEqual(resumed._beta().item(), .05)
        self.assertFalse(resumed._beta.optimizer.state)
        saved["cosil_beta"] = self.policy._beta.network.state_dict()
        with self.assertRaisesRegex(ValueError, "Incomplete CoSIL beta"):
            self.new_policy().load_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
