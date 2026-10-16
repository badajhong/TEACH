"""Small real-model CPU tests for concurrent perception, CoSIL and intervention."""
import copy
import unittest
from contextlib import ExitStack
from types import MethodType
from unittest.mock import patch

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

import test_flashsac_vel_finetune as fixtures
import test_joint_flashsac_vel_finetune as joint_fixtures
from active_adaptation.learning.flashsac.dagger_cosil_joint_flashsac_vel_finetune import (
    INTERVENED_KEY, InterventionNStepReplay,
    DAggerCoSILJointFlashSACVelFinetune, DAggerCoSILJointFlashSACVelFinetuneConfig,
)
from active_adaptation.learning.flashsac.gaddpg_cosil_flashsac_vel_finetune import (
    GADDPGCoSILFlashSACVelFinetune, GADDPGCoSILFlashSACVelFinetuneConfig,
)
from active_adaptation.learning.flashsac.flashsac_vel_flat import U_KEY


class DAggerCoSILJointTest(unittest.TestCase):
    setUp = fixtures.FinetuneTest.setUp
    tearDown = fixtures.FinetuneTest.tearDown
    observation = fixtures.FinetuneTest.observation
    perception_iteration = fixtures.FinetuneTest.perception_iteration
    assert_model_equal = fixtures.FinetuneTest.assert_model_equal
    joint_step = joint_fixtures.JointFinetuneTest.joint_step
    joint_iteration = joint_fixtures.JointFinetuneTest.joint_iteration
    test_perception_trains_during_replay_fill = (
        joint_fixtures.JointFinetuneTest.test_starts_joint_and_perception_trains_while_replay_fills
    )
    test_joint_updates_without_resetting_replay_or_teacher = (
        joint_fixtures.JointFinetuneTest.test_joint_updates_both_models_without_resetting_replay_or_teacher
    )
    test_optional_warmup_and_resume = (
        joint_fixtures.JointFinetuneTest.test_optional_warmup_and_warmup_resume_then_joint
    )

    def new_policy(self, **overrides):
        cfg = DAggerCoSILJointFlashSACVelFinetuneConfig(**{
            **self.options, "adapt_epochs": 1, **overrides,
        })
        return DAggerCoSILJointFlashSACVelFinetune(
            cfg, self.spec, self.action_spec, None, "cpu", self.env)

    def test_joint_training_and_resume_do_not_call_gaddpg_implementation(self):
        self.assertFalse(issubclass(DAggerCoSILJointFlashSACVelFinetune,
                                   GADDPGCoSILFlashSACVelFinetune))
        # Fail if construction, rollouts, training or checkpoint handling dispatch
        # back to the formerly inherited algorithm after this separation.
        methods = ("__init__", "_intervene", "_update_intervention", "_start_rl",
                   "_collect_perception_transition", "_train_perception", "_update_once",
                   "_cosil_update_networks", "state_dict", "load_state_dict")
        with ExitStack() as stack:
            for name in methods:
                stack.enter_context(patch.object(GADDPGCoSILFlashSACVelFinetune, name,
                                                side_effect=AssertionError(f"GADDPG {name} called")))
            self.policy = self.new_policy(adapt_epochs=2)
            self.policy.load_state_dict(copy.deepcopy(self.teacher_state))
            for _ in range(3):
                self.joint_iteration()
            self.assertIs(type(self.policy.buffer), InterventionNStepReplay)
            self.assertGreater(self.policy._update_step, 0)
            saved = copy.deepcopy(self.policy.state_dict())
            self.policy = self.new_policy(adapt_epochs=2)
            self.policy.load_state_dict(saved)
            self.joint_iteration()
            self.assertEqual(self.policy.joint_perception_iters_completed, 4)

    def test_teacher_intervention_replay_and_student_only_metrics(self):
        p = self.policy
        p.intervention_tau = 1e-8
        self.assertTrue(torch.equal(p.stats_env_mask, ~p._mixed_envs))
        self.assertTrue(torch.equal(self.env.stats_env_mask, ~p._mixed_envs))
        td = p.get_rollout_policy("train")(self.observation())
        mean, _ = p.actor.get_mean_and_std(p.replay_obs(td), training=False)
        self.assertTrue(torch.equal(td[INTERVENED_KEY], p._mixed_envs))
        torch.testing.assert_close(td[U_KEY][p._mixed_envs], torch.tanh(mean)[p._mixed_envs])
        # Exercise add_transition through actual rollouts, including executed actions.
        for _ in range(p.cfg.train_every):
            self.joint_step()
        self.assertTrue(torch.equal(p.buffer.intervened[0], p._mixed_envs))
        self.assertEqual(p.sac_iters_completed, 1)
        info = p.pop_info()
        self.assertEqual(info["dagger/takeover_rate"], 1.0)
        self.assertIn("cosil/beta_value", info)
        self.assertNotIn("gaddpg/tau", info)
        self.assertGreater(p.intervention_tau, 1e-8)
        tau = p.intervention_tau
        p.update()  # No newly completed rollout: no second threshold update.
        self.assertEqual(p.intervention_tau, tau)
        self.assertEqual(p.joint_perception_iters_completed, 1)
        sums = p._intervention_sums.clone()
        evaluation = p.get_rollout_policy("eval")(self.observation())
        self.assertNotIn(INTERVENED_KEY, evaluation.keys())
        torch.testing.assert_close(p._intervention_sums, sums)
        expected, _ = p.actor_adapt.get_mean_and_std(p._actor_adapt_input(evaluation), training=False)
        torch.testing.assert_close(evaluation[U_KEY], torch.tanh(expected))

    def test_checkpoint_restores_joint_optimizers_beta_tau_and_decay_horizon(self):
        p = self.policy
        for _ in range(3):
            self.joint_iteration()
        saved = copy.deepcopy(p.state_dict())
        self.assertEqual(saved["finetune_method"], "dagger_cosil_joint_v1")
        self.assertNotIn("num_cycles", saved)
        resumed = self.new_policy(num_env_steps=8192)
        resumed.load_state_dict(saved)
        self.assertEqual(resumed._rl_total_iters, p._rl_total_iters)
        self.assertEqual(resumed._intervention_target(), p._intervention_target())
        self.assertEqual(resumed.intervention_tau, p.intervention_tau)
        self.assertEqual(resumed.sac_iters_completed, 3)
        self.assertEqual(resumed.joint_perception_iters_completed, 3)
        self.assertEqual(resumed._update_step, p._update_step)
        self.assertIsNone(resumed.buffer)
        for opt, old in ((resumed.opt_adapt, p.opt_adapt),
                         (resumed._student.optimizer, p._student.optimizer),
                         (resumed._critic.optimizer, p._critic.optimizer),
                         (resumed._beta.optimizer, p._beta.optimizer)):
            self.assertTrue(opt.state)
            torch.testing.assert_close(opt.state_dict(), old.state_dict(), rtol=0, atol=0)
        self.assert_model_equal(resumed._beta.network, p._beta.network.state_dict())
        for pairs in zip(resumed._perception_pairs(), p._perception_pairs()):
            for model, old in zip(*pairs):
                self.assert_model_equal(model, old.state_dict())
        self.policy = resumed
        self.joint_iteration()
        self.assertEqual(resumed.joint_perception_iters_completed, 4)
        self.assertTrue(all(v.requires_grad for v in resumed._perception_parameters()))

    def test_sac_updates_match_gaddpg_in_all_three_modes(self):
        for mode in ("cosil", "cosil_entropy", "entropy"):
            with self.subTest(mode=mode):
                ours = self.new_policy(cosil_mode=mode)
                base = GADDPGCoSILFlashSACVelFinetune(
                    GADDPGCoSILFlashSACVelFinetuneConfig(
                        **self.options, perception_iters=0, num_cycles=0,
                        adapt_epochs=1, cosil_mode=mode),
                    self.spec, self.action_spec, None, "cpu", self.env)
                for p in (base, ours):
                    p.load_state_dict(copy.deepcopy(self.teacher_state))
                    p._allocate_student_replay()
                g = torch.Generator().manual_seed(3)
                for step in range(12):
                    row = dict(
                        obs=torch.randn(4, ours.student_replay_dim, generator=g),
                        next_obs=torch.randn(4, ours.student_replay_dim, generator=g),
                        action=torch.rand(4, 3, generator=g) * 2 - 1,
                        reward=torch.randn(4, generator=g), discount=torch.ones(4),
                        done=torch.zeros(4, dtype=torch.bool), cut=torch.zeros(4, dtype=torch.bool),
                        valid=torch.ones(4, dtype=torch.bool),
                        intervened=torch.tensor([False, step % 3 == 0, False, step % 2 == 0]),
                    )
                    for p in (base, ours):
                        p.buffer.add(**row)
                for _ in range(3):
                    torch.manual_seed(11)
                    expected = base._update_once()
                    torch.manual_seed(11)
                    actual = ours._update_once()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in ("actor_adapt", "critic", "target_critic", "temperature"):
                    self.assert_model_equal(getattr(ours, name), getattr(base, name).state_dict())
                if mode != "entropy":
                    self.assert_model_equal(ours._beta.network, base._beta.network.state_dict())
                else:
                    self.assertIsNone(ours._beta)

    def test_checkpoint_methods_cannot_be_silently_interchanged(self):
        cfg = GADDPGCoSILFlashSACVelFinetuneConfig(**self.options)
        staged = GADDPGCoSILFlashSACVelFinetune(
            cfg, self.spec, self.action_spec, None, "cpu", self.env)
        with self.assertRaisesRegex(ValueError, "teacher or DAgger"):
            self.policy.load_state_dict(staged.state_dict())
        with self.assertRaisesRegex(ValueError, "Unsupported finetune method"):
            staged.load_state_dict(self.policy.state_dict())

    def test_hydra_config_has_no_cycle_controls(self):
        with initialize(version_base=None, config_path=None):
            cfg = compose(overrides=["+algo=dagger_cosil_joint_flashsac_vel_finetune"])
        self.assertEqual(cfg.algo.perception_iters, 0)
        self.assertEqual(cfg.algo.cosil_mode, "cosil_entropy")
        for key in ("cycle_mode", "num_cycles", "sac_iters", "cycle_perception_iters",
                    "cycle_sac_iters", "plateau_tol", "final_sac_iters"):
            self.assertNotIn(key, cfg.algo)
        runtime = OmegaConf.merge(cfg.algo, self.options, {"adapt_epochs": 1})
        self.policy = DAggerCoSILJointFlashSACVelFinetune(
            runtime, self.spec, self.action_spec, None, "cpu", self.env)
        self.policy.load_state_dict(self.teacher_state)
        self.joint_iteration()
        self.assertEqual(self.policy.joint_perception_iters_completed, 1)
        self.assertNotIn("num_cycles", self.policy.cfg)
        self.assertNotIn("sac_iters", self.policy.cfg)

    def test_rollout_reuses_actor_and_teacher_outputs(self):
        p = self.policy
        with (patch.object(p.actor, "encode_priv", wraps=p.actor.encode_priv) as encoder,
              patch.object(p.actor.policy, "get_mean_and_std",
                           wraps=p.actor.policy.get_mean_and_std) as teacher,
              patch.object(p.actor_adapt, "get_mean_and_std",
                           wraps=p.actor_adapt.get_mean_and_std) as student):
            self.joint_step()  # Includes labels and replay insertion, but no gradient update yet.
            self.assertEqual(encoder.call_count, 1)
            self.assertEqual(teacher.call_count, 1)
            self.assertEqual(student.call_count, 1)
        # Still collect perception labels after takeover stops entirely.
        p.sac_iters_completed = p._rl_total_iters
        with patch.object(p.actor, "encode_priv", wraps=p.actor.encode_priv) as encoder:
            self.joint_step()
            self.assertEqual(encoder.call_count, 1)
        self.assertFalse(p.buffer.intervened[1].any())

    def test_cached_rollout_matches_original_sampling_and_intervention(self):
        p = self.policy
        reference = self.new_policy()
        reference.load_state_dict(copy.deepcopy(p.state_dict()))
        # Loading normalizes actor weights once more; remove that roundoff so this
        # comparison isolates the forward-pass reuse, not checkpoint normalization.
        reference.actor_adapt.load_state_dict(p.actor_adapt.state_dict())
        reference.reuse_rollout_predictions = False
        reference._intervene = MethodType(GADDPGCoSILFlashSACVelFinetune._intervene, reference)
        for completed in (0, 1, p._rl_total_iters):
            td = self.observation()
            td["is_init"][1] = True
            for model in (p, reference):
                model.intervention_tau = 1e-3
                model.sac_iters_completed = completed
            torch.manual_seed(23)
            expected = reference.get_rollout_policy("train")(td.clone())
            torch.manual_seed(23)
            actual = p.get_rollout_policy("train")(td.clone())
            for key in (U_KEY, "action", INTERVENED_KEY, ("next", "adapt_hx"), ("next", "depth_hx")):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(p._intervention_sums, reference._intervention_sums, rtol=0, atol=0)
            labels = p._perception_labels(actual)
            expected_labels = GADDPGCoSILFlashSACVelFinetune._perception_labels(reference, expected)
            torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)

    def test_perception_updates_unchanged_without_monitoring_ema_forward(self):
        p = self.policy
        reference = self.new_policy(adapt_epochs=2)
        reference.load_state_dict(copy.deepcopy(p.state_dict()))
        p.cfg.adapt_epochs = 2
        # Collect one rollout without running updates, then compare with the former
        # monitoring path on exactly the same rollout and minibatch permutations.
        with patch.object(p, "update"):
            self.joint_iteration()
        rollout = p._pending_adapt.clone()
        uncached = rollout.exclude("_rollout_priv_pred").clone()
        torch.manual_seed(31)
        expected = reference._train_perception(uncached, distill_actor=False)
        torch.manual_seed(31)
        with patch.object(p, "_perceive", wraps=p._perceive) as perceive:
            actual = p._train_perception(rollout, distill_actor=False)
            self.assertEqual(perceive.call_count, p.cfg.adapt_epochs * p.cfg.adapt_num_minibatches)
            self.assertTrue(all(not call.kwargs["ema"] for call in perceive.call_args_list))
        for key in ("adapt/priv_loss", "adapt/object_loss"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        for pair, expected_pair in zip(p._perception_pairs(), reference._perception_pairs()):
            for model, old in zip(pair, expected_pair):
                self.assert_model_equal(model, old.state_dict())
        torch.testing.assert_close(p.opt_adapt.state_dict(), reference.opt_adapt.state_dict(), rtol=0, atol=0)

    def test_critic_only_teacher_forward_uses_next_states_only(self):
        p = self.policy
        batch = {"teacher_observation": torch.randn(8, p.replay_obs_dim),
                 "teacher_next_observation": torch.randn(8, p.replay_obs_dim)}
        with (torch.no_grad(), patch.object(p.actor, "get_mean_and_std",
                                            wraps=p.actor.get_mean_and_std) as teacher):
            mu, mu_next = p._cosil_teacher_actions(batch, do_actor_update=False)
            self.assertIsNone(mu)
            self.assertEqual(teacher.call_count, 1)
            self.assertEqual(teacher.call_args.args[0].shape[0], 8)
            self.assertEqual(mu_next.shape, (8, p.action_dim))

    def test_dagger_checkpoint_names_and_legacy_joint_checkpoint_restore(self):
        p = self.policy
        self.joint_iteration()
        state = copy.deepcopy(p.state_dict())
        self.assertFalse(any(key.startswith("gaddpg") for key in state))
        self.assertIn("dagger_intervention_tau", state)
        self.assertIn("dagger_sac_iters_completed", state)
        # Joint checkpoints produced before the naming cleanup remain loadable.
        state["gaddpg_intervention_tau"] = state.pop("dagger_intervention_tau")
        state["gaddpg_sac_iters_completed"] = state.pop("dagger_sac_iters_completed")
        resumed = self.new_policy()
        resumed.load_state_dict(state)
        self.assertEqual(resumed.intervention_tau, p.intervention_tau)
        self.assertEqual(resumed.sac_iters_completed, p.sac_iters_completed)

    def test_monitor_is_cached_across_epochs_without_changing_training_or_metrics(self):
        p = self.policy
        p.cfg.adapt_epochs = 2
        reference = self.new_policy(adapt_epochs=2)
        reference.load_state_dict(copy.deepcopy(p.state_dict()))
        reference.actor_adapt.load_state_dict(p.actor_adapt.state_dict())
        with patch.object(p, "update"):
            self.joint_iteration()
        rollout = p._pending_adapt.clone()
        rollout["is_init"][0, 0] = True  # Include a masked sample in both diagnostic paths.
        torch.manual_seed(41)
        with (patch.object(reference, "_prepare_perception_monitor"),
              patch.object(reference.actor_adapt, "get_mean_and_std",
                           wraps=reference.actor_adapt.get_mean_and_std) as previous):
            expected = reference._train_perception(rollout.clone(), distill_actor=False)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(41)
        with patch.object(p.actor_adapt, "get_mean_and_std",
                          wraps=p.actor_adapt.get_mean_and_std) as current:
            actual = p._train_perception(rollout.clone(), distill_actor=False)
        self.assertEqual(previous.call_count, p.cfg.adapt_epochs * p.cfg.adapt_num_minibatches)
        self.assertEqual(current.call_count, p.cfg.adapt_num_minibatches)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        torch.testing.assert_close(p.opt_adapt.state_dict(), reference.opt_adapt.state_dict(), rtol=0, atol=0)
        for pair, expected_pair in zip(p._perception_pairs(), reference._perception_pairs()):
            for model, old in zip(pair, expected_pair):
                self.assert_model_equal(model, old.state_dict())
        # A later call after SAC must not reuse errors from the previous actor.
        cached = rollout.clone()
        p._prepare_perception_monitor(cached, distill_actor=False)
        before = cached["_monitor_action_squared_error"].clone()
        with torch.no_grad():
            p.actor_adapt.predictor.mean_bias.add_(0.2)
        p._prepare_perception_monitor(cached, distill_actor=False)
        self.assertFalse(torch.equal(cached["_monitor_action_squared_error"], before))
        p._prepare_perception_monitor(cached, distill_actor=True)
        self.assertNotIn("_monitor_action_squared_error", cached.keys())

    def test_only_configured_optimizers_step_during_joint_training(self):
        p = self.policy
        p.cfg.adapt_epochs = 2
        p.cfg.updates_per_interaction_step = 8  # Same SAC update ratio as production.
        for _ in range(3):
            self.joint_iteration()
        before = p._update_step
        optimizers = {
            "perception": p.opt_adapt, "actor": p._student.optimizer,
            "critic": p._critic.optimizer, "beta": p._beta.optimizer,
            "temperature": p._temperature.optimizer, "bc": p.opt_adapt_actor,
            "teacher": p._actor.optimizer,
        }
        with ExitStack() as stack:
            steps = {name: stack.enter_context(patch.object(opt, "step", wraps=opt.step))
                     for name, opt in optimizers.items()}
            self.joint_iteration()
        total = int(p.cfg.train_every * p.cfg.updates_per_interaction_step)
        actor_updates = sum(step % p.flash_cfg.actor_update_period == 0
                            for step in range(before, before + total))
        self.assertEqual(p._update_step - before, total)
        self.assertEqual(steps["critic"].call_count, total)
        for name in ("actor", "beta", "temperature"):
            self.assertEqual(steps[name].call_count, actor_updates, name)
        self.assertEqual(steps["perception"].call_count, p.cfg.adapt_epochs * p.cfg.adapt_num_minibatches)
        self.assertEqual(steps["bc"].call_count, 0)
        self.assertEqual(steps["teacher"].call_count, 0)
        self.assertTrue(all(v.grad is None for v in p.actor.parameters()))
        self.assertTrue(all(v.grad is None for _, ema in p._perception_pairs() for v in ema.parameters()))


if __name__ == "__main__":
    unittest.main()
