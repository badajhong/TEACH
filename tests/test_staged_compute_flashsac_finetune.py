"""Preserve staged warmup/refresh updates while reusing fixed teacher/EMA outputs."""
import copy
import unittest
from unittest.mock import patch

import torch

import test_flashsac_vel_finetune as fixtures
from active_adaptation.learning.flashsac.flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
)
from active_adaptation.learning.modules.rnn import set_recurrent_mode


class StagedComputeTest(unittest.TestCase):
    setUp = fixtures.FinetuneTest.setUp
    tearDown = fixtures.FinetuneTest.tearDown
    observation = fixtures.FinetuneTest.observation
    perception_iteration = fixtures.FinetuneTest.perception_iteration
    assert_model_equal = fixtures.FinetuneTest.assert_model_equal

    def new_policy(self, **overrides):
        cfg = FlashSACVelFinetuneConfig(**{
            **self.options, "cycle_mode": True, "perception_iters": 5,
            "adapt_epochs": 2, **overrides,
        })
        return FlashSACVelFinetune(cfg, self.spec, self.action_spec, None, "cpu", self.env)

    def test_teacher_encoder_runs_once_for_identical_labels_and_error_measurement(self):
        p = self.policy
        td = p.get_rollout_policy("train")(self.observation())
        with torch.no_grad():
            raw = p.replay_obs(td)
            feature = p.actor.encode_priv(raw)
            mean, _ = p.actor.get_mean_and_std(raw, training=False)
            expected_action = p._joint_action(torch.tanh(mean), td)
        with patch.object(p.actor, "encode_priv", wraps=p.actor.encode_priv) as encoder:
            p.add_transition(td)
            self.assertEqual(encoder.call_count, 1)
        batch = p._adapt_buffer[:, 0]
        # Get the teacher-feature key through the label function's returned value
        # and the stored teacher action; schedule accumulators use the same target.
        actual_feature, actual_action = p._perception_labels(td)
        torch.testing.assert_close(actual_feature, feature, rtol=0, atol=0)
        torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)
        torch.testing.assert_close(batch["_teacher_action"], expected_action, rtol=0, atol=0)
        before = (p._perception_error_sum.clone(), p._perception_error_count.clone())
        p._perception_error_sum.zero_()
        p._perception_error_count.zero_()
        p._record_perception_error(td, target=feature)
        torch.testing.assert_close(before, (p._perception_error_sum, p._perception_error_count), rtol=0, atol=0)

    def test_warmup_and_refresh_match_uncached_losses_updates_and_rng(self):
        for refresh in (False, True):
            with self.subTest(refresh=refresh):
                self.policy = p = self.new_policy()
                p.load_state_dict(copy.deepcopy(self.teacher_state))
                if refresh:
                    p._start_rl()
                    p._start_perception(student=True)
                reference = self.new_policy()
                reference.load_state_dict(copy.deepcopy(p.state_dict()))
                reference.actor_adapt.load_state_dict(p.actor_adapt.state_dict())
                with patch.object(p, "update"):
                    self.perception_iteration()
                rollout = p._pending_adapt.clone()
                rollout["is_init"][0, 0] = True
                rollout["is_init"][1, 2] = True  # Reset inside the GRU sequence too.
                torch.manual_seed(97)
                with (patch.object(reference, "_prepare_perception_cache"),
                      patch.object(reference, "_perceive", wraps=reference._perceive) as old_perceive,
                      patch.object(reference.actor_adapt, "get_mean_and_std",
                                   wraps=reference.actor_adapt.get_mean_and_std) as old_actor):
                    expected = reference._train_perception(rollout.clone())
                expected_rng = torch.get_rng_state()
                torch.manual_seed(97)
                with (patch.object(p, "_perceive", wraps=p._perceive) as perceive,
                      patch.object(p.actor_adapt, "get_mean_and_std",
                                   wraps=p.actor_adapt.get_mean_and_std) as actor,
                      patch.object(p.opt_adapt_actor, "step", wraps=p.opt_adapt_actor.step) as bc,
                      patch.object(p.opt_adapt, "step", wraps=p.opt_adapt.step) as perception_steps):
                    actual = p._train_perception(rollout.clone())
                updates = p.cfg.adapt_epochs * p.cfg.adapt_num_minibatches
                self.assertEqual(sum(c.kwargs["ema"] for c in old_perceive.call_args_list), updates)
                self.assertEqual(sum(c.kwargs["ema"] for c in perceive.call_args_list), p.cfg.adapt_num_minibatches)
                self.assertEqual(sum(not c.kwargs["ema"] for c in perceive.call_args_list), updates)
                self.assertEqual(old_actor.call_count, updates)
                self.assertEqual(actor.call_count, p.cfg.adapt_num_minibatches if refresh else updates)
                self.assertEqual(bc.call_count, 0 if refresh else updates)
                self.assertEqual(perception_steps.call_count, updates)
                # Precomputing EMA sequences in a different env order can change
                # float32 kernel rounding; it must not change the BC objective.
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                torch.testing.assert_close(p.opt_adapt.state_dict(), reference.opt_adapt.state_dict(),
                                           rtol=0, atol=0)
                torch.testing.assert_close(p.opt_adapt_actor.state_dict(), reference.opt_adapt_actor.state_dict(),
                                           rtol=1e-6, atol=1e-7)
                for pair, old_pair in zip(p._perception_pairs(), reference._perception_pairs()):
                    for model, old in zip(pair, old_pair):
                        self.assert_model_equal(model, old.state_dict())
                torch.testing.assert_close(p.actor_adapt.state_dict(), reference.actor_adapt.state_dict(),
                                           rtol=1e-6, atol=1e-7)

    def test_cache_is_recomputed_after_ema_changes_and_skipped_for_one_epoch(self):
        p = self.policy
        with patch.object(p, "update"):
            self.perception_iteration()
        rollout = p._pending_adapt.clone()
        with set_recurrent_mode(True):
            p._prepare_perception_cache(rollout, distill_actor=True)
        previous = rollout["_perception_ema_latent"].clone()
        with torch.no_grad():
            for parameter in p.adapt_ema.parameters():
                parameter.add_(0.01)
        with set_recurrent_mode(True):
            p._prepare_perception_cache(rollout, distill_actor=True)
        self.assertFalse(torch.equal(rollout["_perception_ema_latent"], previous))
        p.cfg.adapt_epochs = 1
        p._prepare_perception_cache(rollout, distill_actor=True)
        self.assertNotIn("_perception_ema_latent", rollout.keys())
        self.assertNotIn("_monitor_action_squared_error", rollout.keys())


if __name__ == "__main__":
    unittest.main()
