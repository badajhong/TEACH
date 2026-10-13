"""CPU integration coverage for the adaptive perception/SAC phase schedule.

Run with ``PYTHONPATH=. python -m unittest discover -s tests -p
test_adaptive_flashsac_finetune.py``. The small real models exercise rollout,
supervised updates, replay insertion and checkpoint restore without a simulator.
"""
import copy
import math
import unittest
import warnings
from unittest.mock import patch

import torch

import test_flashsac_vel_finetune as fixtures
from active_adaptation.learning.flashsac.flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
)


class AdaptiveFinetuneTest(unittest.TestCase):
    # Reuse only the fixture helpers, not its legacy-schedule test methods.
    setUp = fixtures.FinetuneTest.setUp
    tearDown = fixtures.FinetuneTest.tearDown
    observation = fixtures.FinetuneTest.observation
    perception_iteration = fixtures.FinetuneTest.perception_iteration
    assert_model_equal = fixtures.FinetuneTest.assert_model_equal

    def new_policy(self, **overrides):
        schedule = dict(
            cycle_mode=True, perception_iters=2, adapt_epochs=1,
            perception_trigger=.03, perception_error_window=2,
            perception_trigger_patience=3,
            plateau_window=1, plateau_tol=.0025, plateau_patience=2,
            plateau_check_interval=1,
            cycle_perception_iters_min=2, cycle_perception_iters_max=4,
            cycle_sac_iters_min=2, cycle_sac_iters_max=4,
            # None of these legacy controls should prevent adaptive refresh.
            num_cycles=0, sac_iters=1000,
        )
        schedule.update(overrides)
        cfg = FlashSACVelFinetuneConfig(**{**self.options, **schedule})
        return FlashSACVelFinetune(cfg, self.spec, self.action_spec, None, "cpu", self.env)

    def finish_teacher(self):
        self.perception_iteration()
        self.perception_iteration()
        p = self.policy
        self.assertEqual(p.stage, 2)
        self.assertFalse(p.is_repeated_perception_phase)
        self.assertEqual(p.cycles_completed, 0)
        self.assertEqual(p.finetune_iters_completed, p.cfg.perception_iters)
        self.assertEqual(p.rl_phase_iters_completed, 0)
        self.assertIsNotNone(p._student)
        self.assertTrue(p._initial_reference_pending)
        self.assertIsNone(p._adaptive_schedule.reference_error)

    def sac_step(self, before_update=None):
        p = self.policy
        self.assertEqual(p.stage, 2)
        td = p.get_rollout_policy("train")(self.observation())
        future = self.observation()
        for key in ("adapt_hx", "depth_hx"):
            future[key] = td["next", key].clone()
        future["terminated"] = torch.zeros(4, 1, dtype=torch.bool)
        future["truncated"] = torch.zeros(4, 1, dtype=torch.bool)
        future["reward"] = torch.randn(4, 1)
        future["discount"] = torch.ones(4, 1)
        td["next"] = future
        p.add_transition(td)
        if before_update is not None:
            before_update()
        p.update()

    def sac_iteration(self):
        for _ in range(self.policy.cfg.train_every):
            self.sac_step()

    def test_teacher_boundary_enters_sac_without_extra_perception(self):
        self.perception_iteration()
        p = self.policy
        self.assertEqual(p.stage, 1)
        self.assertFalse(p.is_repeated_perception_phase)
        self.assertIsNone(p._student)
        self.perception_iteration()
        self.assertEqual(p.stage, 2)
        self.assertEqual(p.finetune_iters_completed, 2)
        self.assertEqual(p.cycles_completed, 0)
        self.assertEqual(p.rl_phase_iters_completed, 0)
        self.assertTrue(p._initial_reference_pending)
        self.assertIsNone(p._adaptive_schedule.reference_error)
        self.assertFalse(p._adaptive_schedule.state_dict()["history"])
        self.assertEqual(p._update_step, 0)
        self.assertTrue(all(x.requires_grad for x in p.actor_adapt.parameters()))
        self.assertTrue(all(not x.requires_grad for x in p._perception_parameters()))
        frozen = [x.detach().clone() for x in p._perception_parameters()]
        self.sac_iteration()
        self.assertEqual(p.stage, 2)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(p._perception_parameters(), frozen)))

    def test_initial_reference_precedes_first_gradient_and_includes_partial_iteration(self):
        self.finish_teacher()
        p = self.policy
        actor_before = copy.deepcopy(p.actor_adapt.state_dict())
        first_gradient_step = None
        expected = {}
        original_update = p._update_once

        def before_update():
            if p.can_start_training() and p._update_step == 0:
                history = p._adaptive_schedule.state_dict()["history"]
                current = [float(p._perception_error_sum), int(p._perception_error_count)]
                self.assertTrue(history)
                self.assertGreater(p._rl_steps_in_iteration, 0)
                observations = (history + [current])[-p.cfg.perception_error_window:]
                expected.update(history=history, current=current,
                                reference=math.sqrt(sum(x[0] for x in observations)
                                                    / sum(x[1] for x in observations)))

        def checked_update():
            nonlocal first_gradient_step
            if p._update_step == 0:
                first_gradient_step = p.buffer.step
                self.assertFalse(p._initial_reference_pending)
                self.assertAlmostEqual(p._adaptive_schedule.reference_error,
                                       expected["reference"], places=12)
                self.assertEqual(p._adaptive_schedule.state_dict()["history"], expected["history"])
                self.assertEqual([float(p._perception_error_sum), int(p._perception_error_count)],
                                 expected["current"])
                self.assert_model_equal(p.actor_adapt, actor_before)
            return original_update()

        with patch.object(p, "_update_once", side_effect=checked_update):
            for _ in range(2 * p.cfg.train_every):
                self.sac_step(before_update)
                if p._update_step:
                    break
        self.assertEqual(first_gradient_step, 6)
        self.assertEqual(p._update_step, 1)
        self.assertTrue(any(not torch.equal(v, actor_before[k])
                            for k, v in p.actor_adapt.state_dict().items()))

    def test_first_gradient_timing_matches_fixed_schedule_with_tiny_replay(self):
        first_gradient_steps = []
        for adaptive in (False, True):
            self.policy = self.new_policy(cycle_mode=adaptive, buffer_min_length=4,
                                          perception_error_window=50)
            self.policy.load_state_dict(self.teacher_state)
            self.perception_iteration()
            self.perception_iteration()
            p = self.policy
            self.assertEqual(p.stage, 2)
            for step in range(1, p.cfg.train_every + 1):
                self.sac_step()
                if p._update_step:
                    first_gradient_steps.append(step)
                    break
            self.assertEqual(p._update_step, 1)
            if adaptive:
                self.assertFalse(p._initial_reference_pending)
                self.assertFalse(p._adaptive_schedule.state_dict()["history"])
                self.assertAlmostEqual(p._adaptive_schedule.reference_error,
                                       math.sqrt(float(p._perception_error_sum)
                                                 / int(p._perception_error_count)), places=12)
        self.assertEqual(first_gradient_steps, [3, 3])

    def test_pending_baseline_checkpoint_resumes_in_sac_without_extra_refresh(self):
        self.finish_teacher()
        self.sac_iteration()
        original = self.policy
        self.assertEqual(original._update_step, 0)
        self.assertTrue(original._initial_reference_pending)
        saved = copy.deepcopy(original.state_dict())
        self.assertEqual(saved["finetune_version"], 9)
        self.assertTrue(saved["initial_reference_pending"])
        self.assertIn("student_optimizer", saved)
        self.policy = self.new_policy()
        self.policy.load_state_dict(saved)
        p = self.policy
        self.assertEqual(p.stage, 2)
        self.assertTrue(p._initial_reference_pending)
        self.assertIsNone(p._adaptive_schedule.reference_error)
        self.assertIsNone(p.buffer)
        self.assertEqual(p._adaptive_schedule.state_dict(),
                         original._adaptive_schedule.state_dict())
        self.sac_iteration()
        self.sac_step()
        self.sac_step()
        self.assertGreater(p._update_step, 0)
        self.assertEqual(p.stage, 2)
        self.assertFalse(p._initial_reference_pending)
        self.assertIsNotNone(p._adaptive_schedule.reference_error)

    def test_repeated_refresh_freezes_actor_and_restores_checkpoint(self):
        self.finish_teacher()
        while self.policy.stage == 2:
            self.sac_iteration()
        p = self.policy
        self.assertEqual(p.cycles_completed, 1)
        self.assertTrue(p.is_repeated_perception_phase)
        self.assertIsNotNone(p._student)
        actor_before = copy.deepcopy(p.actor_adapt.state_dict())
        perception_before = [x.detach().clone() for x in p._perception_parameters()]
        self.perception_iteration()
        self.assertEqual(p.stage, 1)
        self.assert_model_equal(p.actor_adapt, actor_before)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(p._perception_parameters(), perception_before)))
        saved = copy.deepcopy(p.state_dict())
        self.assertFalse(saved["initial_reference_pending"])
        self.policy = self.new_policy()
        self.policy.load_state_dict(saved)
        p = self.policy
        self.assertEqual(p.stage, 1)
        self.assertTrue(p.is_repeated_perception_phase)
        self.assertEqual(p.perception_iters_completed, 1)
        self.assertTrue(all(not x.requires_grad for x in p.actor_adapt.parameters()))
        for key, value in p.actor_adapt.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=1e-6, atol=1e-7)
        actor_before = copy.deepcopy(p.actor_adapt.state_dict())
        self.assertIn("actor/lr", p.pop_info())
        for _ in range(p.cfg.cycle_perception_iters_max):
            if p.stage == 2:
                break
            self.perception_iteration()
        self.assertEqual(p.stage, 2)
        self.assertFalse(p._initial_reference_pending)
        self.assertIsNotNone(p._adaptive_schedule.reference_error)
        self.assert_model_equal(p.actor_adapt, actor_before)

    def test_first_sac_uses_adaptive_cap_even_with_zero_legacy_cycles(self):
        self.finish_teacher()
        p = self.policy
        frozen = [x.detach().clone() for x in p._perception_parameters()]
        for iteration in range(1, p.cfg.cycle_sac_iters_max + 1):
            self.sac_iteration()
            if iteration < p.cfg.cycle_sac_iters_min:
                self.assertEqual(p.stage, 2)
            if p.stage == 1:
                break
        self.assertEqual(p.stage, 1)
        self.assertEqual(p.cycles_completed, 1)
        self.assertTrue(p.is_repeated_perception_phase)
        self.assertEqual(iteration, 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(p._perception_parameters(), frozen)))
        self.assertIsNone(p.buffer)
        self.assertTrue(all(not x.requires_grad for x in p.actor_adapt.parameters()))

    def test_sac_checkpoint_preserves_adaptive_history_and_replay_is_rebuilt(self):
        self.finish_teacher()
        self.sac_iteration()
        self.sac_iteration()
        p = self.policy
        self.assertEqual(p.stage, 2)
        self.assertIsNotNone(p.buffer)
        self.assertEqual(p.buffer.obs.shape[-1], p.student_replay_dim)
        saved = copy.deepcopy(p.state_dict())
        for version in (9, 8, 7):
            with self.subTest(checkpoint_version=version):
                checkpoint = copy.deepcopy(saved)
                checkpoint["finetune_version"] = version
                if version < 9:
                    for key in ("training_target_iter", "final_sac_active", "final_sac_iters"):
                        checkpoint.pop(key, None)
                if version == 7:
                    checkpoint["finetune_version"] = 7
                    checkpoint["finetune_method"] = "adaptive_cycles_v1"
                    checkpoint.pop("initial_reference_pending")
                resumed = self.new_policy()
                resumed.load_state_dict(checkpoint)
                self.assertEqual(resumed.stage, 2)
                self.assertEqual(resumed.rl_phase_iters_completed, 2)
                self.assertFalse(resumed._initial_reference_pending)
                self.assertIsNone(resumed.buffer)
                self.assertEqual(resumed._adaptive_schedule.state_dict(),
                                 p._adaptive_schedule.state_dict())
                self.assertEqual(resumed._rl_schedule_updates, p._rl_schedule_updates)
                self.assertEqual(resumed._update_step, p._update_step)

    def test_v7_teacher_checkpoint_finishes_remaining_teacher_then_direct_sac(self):
        self.perception_iteration()
        saved = copy.deepcopy(self.policy.state_dict())
        saved["finetune_version"] = 7
        saved["finetune_method"] = "adaptive_cycles_v1"
        saved.pop("initial_reference_pending")
        self.policy = self.new_policy()
        self.policy.load_state_dict(saved)
        self.assertEqual(self.policy.stage, 1)
        self.assertEqual(self.policy.perception_iters_completed, 1)
        self.perception_iteration()
        self.assertEqual(self.policy.stage, 2)
        self.assertEqual(self.policy.finetune_iters_completed, 2)
        self.assertTrue(self.policy._initial_reference_pending)
        self.assertIsNone(self.policy._adaptive_schedule.reference_error)

    def test_v7_initial_student_refresh_without_sac_optimizer_resumes_in_place(self):
        # Version 7 had an initial frozen-actor refresh before creating the SAC
        # optimizer. Reconstruct that checkpoint shape from a real teacher phase.
        self.perception_iteration()
        saved = copy.deepcopy(self.policy.state_dict())
        saved.update(finetune_version=7, finetune_method="adaptive_cycles_v1",
                     student_perception_phase=True, finetune_iters_completed=3,
                     last_transition_reason="initial_student_refresh")
        saved.pop("initial_reference_pending")
        self.assertNotIn("student_optimizer", saved)
        self.policy = self.new_policy()
        self.policy.load_state_dict(saved)
        p = self.policy
        self.assertEqual(p.stage, 1)
        self.assertEqual(p.cycles_completed, 0)
        self.assertEqual(p.perception_iters_completed, 1)
        self.assertTrue(p.is_repeated_perception_phase)
        self.assertIsNone(p._student)
        self.assertIsNone(p._adaptive_schedule.reference_error)
        self.assertTrue(all(not x.requires_grad for x in p.actor_adapt.parameters()))
        actor_before = copy.deepcopy(p.actor_adapt.state_dict())
        for _ in range(p.cfg.cycle_perception_iters_max):
            if p.stage == 2:
                break
            self.perception_iteration()
        self.assertEqual(p.stage, 2)
        self.assertIsNotNone(p._student)
        self.assertIsNotNone(p._adaptive_schedule.reference_error)
        self.assertFalse(p._initial_reference_pending)
        self.assertEqual(p._update_step, 0)
        # Creating the first SAC network normalizes UnitLinear weights once;
        # permit that float32 rounding while ruling out an actor update.
        for key, value in p.actor_adapt.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=1e-6, atol=1e-7)

    def test_resume_retains_sustained_trigger_progress(self):
        self.policy = self.new_policy(cycle_sac_iters_max=8)
        self.policy.load_state_dict(self.teacher_state)
        self.finish_teacher()
        p = self.policy
        self.sac_iteration()
        self.sac_iteration()
        self.assertFalse(p._initial_reference_pending)
        # A deliberately low reference makes real rollout errors unambiguously
        # exceed the threshold, so this tests phase timing rather than randomness.
        p._adaptive_schedule.reference_error = 1e-12
        self.sac_iteration()
        self.sac_iteration()
        self.assertEqual(p.stage, 2)
        self.assertEqual(p._adaptive_schedule.trigger_streak, 2)
        saved = copy.deepcopy(p.state_dict())
        self.policy = self.new_policy(cycle_sac_iters_max=8)
        self.policy.load_state_dict(saved)
        self.assertEqual(self.policy._adaptive_schedule.trigger_streak, 2)
        self.sac_iteration()
        self.assertEqual(self.policy.stage, 1)
        self.assertEqual(self.policy.cycles_completed, 1)

    def test_measurement_weights_valid_states_and_ignores_reset_rows(self):
        p = self.policy
        td = self.observation()
        td["priv_pred"] = torch.tensor([2., 3., 100., 100.]).unsqueeze(-1).expand(4, 8)
        td["step_count"] = torch.tensor([[2], [3], [1], [0]])
        p._record_perception_error(td, target=torch.zeros(4, 8))
        self.assertAlmostEqual(float(p._perception_error_sum), 13.)
        self.assertEqual(int(p._perception_error_count), 2)
        td["priv_pred"] = torch.ones(4, 8)
        td["step_count"] = torch.tensor([[2], [0], [0], [0]])
        p._record_perception_error(td, target=torch.zeros(4, 8))
        # A mean of the two step means would be 3.75, rather than 14 / 3.
        self.assertAlmostEqual(float(p._perception_error_sum), 14.)
        self.assertEqual(int(p._perception_error_count), 3)
        p._finish_perception_measurement()
        self.assertAlmostEqual(p._adaptive_schedule.last_error, math.sqrt(14 / 3), places=6)

    def test_replay_add_measures_current_rollout_ema_and_keeps_compact_storage(self):
        self.finish_teacher()
        p = self.policy
        td = p.get_rollout_policy("train")(self.observation())
        td["step_count"] = torch.tensor([[3], [4], [1], [0]])
        original_latent = td["priv_pred"].clone()
        with torch.no_grad():
            target = p.actor.encode_priv(p.replay_obs(td)).clone()
        expected_sum = (original_latent[:2] - target[:2]).square().mean(-1).sum()
        future = self.observation()
        future["adapt_hx"] = td["next", "adapt_hx"].clone()
        future["depth_hx"] = td["next", "depth_hx"].clone()
        future["terminated"] = torch.zeros(4, 1, dtype=torch.bool)
        future["truncated"] = torch.zeros(4, 1, dtype=torch.bool)
        future["reward"] = torch.zeros(4, 1)
        future["discount"] = torch.ones(4, 1)
        td["next"] = future
        p.add_transition(td)
        torch.testing.assert_close(p._perception_error_sum,
                                   expected_sum.to(p._perception_error_sum.dtype))
        self.assertEqual(int(p._perception_error_count), 2)
        torch.testing.assert_close(td["priv_pred"], original_latent)
        self.assertEqual(p.buffer.obs.shape[-1], p.student_replay_dim)
        self.assertEqual(p.student_replay_dim,
                         p.replay_obs_dim + p.vel_command_dim + p.cfg.latent_dim)
        torch.testing.assert_close(p.buffer.obs[0, :, -8:].float(), original_latent,
                                   atol=.002, rtol=.002)

    def test_adaptive_lr_budget_uses_global_horizon_and_ignores_fixed_schedule(self):
        self.finish_teacher()
        p = self.policy
        horizon = p.cfg.num_env_steps // p.num_envs - p.cfg.perception_iters * p.cfg.train_every
        expected_updates = max(1, int(horizon * p.cfg.updates_per_interaction_step))
        self.assertEqual(p._rl_schedule_updates, expected_updates)
        other = self.new_policy(sac_iters=7, num_cycles=80,
                                cycle_perception_iters=19, cycle_sac_iters=23)
        other.load_state_dict(self.teacher_state)
        other._start_rl()
        self.assertEqual(other._rl_schedule_updates, expected_updates)

    def test_resume_with_new_run_budget_preserves_adaptive_lr_clock(self):
        self.finish_teacher()
        self.sac_iteration()
        original = self.policy
        saved = copy.deepcopy(original.state_dict())
        self.assertEqual(saved["adaptive_lr_start_iter"], original._adaptive_lr_start_iter)
        self.assertEqual(saved["adaptive_lr_num_steps"], original._adaptive_lr_num_steps)

        self.options["num_env_steps"] *= 2
        resumed = self.new_policy()
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.cfg.num_env_steps, 2 * original.cfg.num_env_steps)
        self.assertEqual(resumed._adaptive_lr_start_iter, original._adaptive_lr_start_iter)
        self.assertEqual(resumed._adaptive_lr_num_steps, original._adaptive_lr_num_steps)
        self.assertEqual(resumed._rl_schedule_updates, original._rl_schedule_updates)

        # Compare the next scheduled LR at identical global progress. Calling
        # the scheduler directly isolates restoration from stochastic SAC updates.
        next_iteration = original.finetune_iters_completed + 1
        original.finetune_iters_completed = resumed.finetune_iters_completed = next_iteration
        for name in ("_student", "_critic", "_temperature"):
            expected, restored = getattr(original, name), getattr(resumed, name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
                expected.scheduler.step()
                restored.scheduler.step()
            self.assertEqual(expected.optimizer.param_groups[0]["lr"],
                             restored.optimizer.param_groups[0]["lr"], name)

    def test_legacy_sac_checkpoint_stays_in_sac_and_gets_baseline_during_refill(self):
        self.policy = self.new_policy(cycle_mode=False)
        self.policy.load_state_dict(self.teacher_state)
        self.perception_iteration()
        self.perception_iteration()
        self.assertEqual(self.policy.stage, 2)
        saved = copy.deepcopy(self.policy.state_dict())
        saved["finetune_version"] = 6
        saved["finetune_method"] = "cycles_v2"
        resumed = self.new_policy()
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.stage, 2)
        self.assertFalse(resumed.is_repeated_perception_phase)
        self.assertTrue(resumed._initial_reference_pending)
        self.assertIsNone(resumed._adaptive_schedule.reference_error)
        self.assertTrue(all(x.requires_grad for x in resumed.actor_adapt.parameters()))
        self.policy = resumed
        self.sac_iteration()
        self.sac_iteration()
        self.assertGreater(resumed._update_step, 0)
        self.assertFalse(resumed._initial_reference_pending)
        self.assertIsNotNone(resumed._adaptive_schedule.reference_error)


    def train_one_iteration(self):
        if self.policy.stage == 1:
            self.perception_iteration()
        else:
            self.sac_iteration()

    def constant_monitor(self, td, target=None):
        # Keep actual perception/actor optimization, replacing only the scalar
        # stopping signal so exact plateau timing does not depend on random data.
        self.policy._perception_error_sum.add_(4.)
        self.policy._perception_error_count.add_(4)

    def test_final_sac_config_and_budget_validation(self):
        self.assertEqual(FlashSACVelFinetuneConfig().final_sac_iters, 200)
        for invalid in (-1, 1.5, True, False):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "final_sac_iters"):
                self.new_policy(final_sac_iters=invalid)
        self.assertEqual(self.new_policy(final_sac_iters=0).cfg.final_sac_iters, 0)
        for invalid in (-1, 1.5, True):
            with self.subTest(budget=invalid), self.assertRaises(ValueError):
                self.policy.configure_training_budget(invalid)
        self.assertFalse(self.policy.training_complete())

    def test_final_sac_suppresses_boundary_cap_and_error_trigger_without_replay_reset(self):
        for boundary_reason in ("cap", "error_trigger"):
            with self.subTest(reason=boundary_reason):
                self.policy = self.new_policy(
                    final_sac_iters=4, cycle_sac_iters_max=4 if boundary_reason == "cap" else 8,
                    perception_trigger_patience=2)
                self.policy.load_state_dict(self.teacher_state)
                p = self.policy
                p.configure_training_budget(6)
                self.finish_teacher()
                p.needs_env_reset = False
                self.sac_iteration()
                self.sac_iteration()
                if boundary_reason == "error_trigger":
                    p._adaptive_schedule.reference_error = 1e-12
                self.sac_iteration()
                self.assertFalse(p._final_sac_active)
                self.assertFalse(p.training_complete())
                if boundary_reason == "error_trigger":
                    self.assertEqual(p._adaptive_schedule.trigger_streak, 1)
                replay = p.buffer
                updates = p._update_step
                self.sac_iteration()
                self.assertTrue(p._final_sac_active)
                self.assertTrue(p.training_complete())
                self.assertEqual(p.finetune_iters_completed, 6)
                self.assertEqual(p.stage, 2)
                self.assertEqual(p.rl_phase_iters_completed, 4)
                self.assertEqual(p.cycles_completed, 0)
                self.assertIs(p.buffer, replay)
                self.assertFalse(p.needs_env_reset)
                self.assertGreater(p._update_step, updates)

    def test_final_sac_counts_existing_phase_and_adds_only_missing_iterations(self):
        # The four-iteration nominal budget also ends the LR schedule; extra
        # SAC must continue at learning_rate_end instead of restarting decay.
        self.policy = self.new_policy(final_sac_iters=5, num_env_steps=64)
        self.policy.load_state_dict(self.teacher_state)
        p = self.policy
        p.configure_training_budget(4)
        self.finish_teacher()
        self.sac_iteration()
        self.sac_iteration()
        self.assertTrue(p._final_sac_active)
        self.assertFalse(p.training_complete())
        self.assertEqual(p.rl_phase_iters_completed, 2)
        for phase_iteration in (3, 4, 5):
            self.sac_iteration()
            self.assertEqual(p.stage, 2)
            self.assertEqual(p.training_complete(), phase_iteration == 5)
        self.assertEqual(p.finetune_iters_completed, 7)
        self.assertEqual(p.cycles_completed, 0)
        for name in ("_student", "_critic", "_temperature"):
            self.assertAlmostEqual(getattr(p, name).optimizer.param_groups[0]["lr"],
                                   p.cfg.learning_rate_end, places=14)

    def test_final_sac_does_not_activate_before_nominal_budget(self):
        self.policy = self.new_policy(final_sac_iters=6)
        self.policy.load_state_dict(self.teacher_state)
        p = self.policy
        p.configure_training_budget(8)
        self.finish_teacher()
        for _ in range(4):
            self.sac_iteration()
        # T-N is 2, but the ordinary SAC cap at 6 must still start refresh.
        self.assertEqual(p.finetune_iters_completed, 6)
        self.assertEqual(p.stage, 1)
        self.assertEqual(p.cycles_completed, 1)
        self.assertFalse(p._final_sac_active)
        self.assertFalse(p.training_complete())

    def test_disabled_and_fixed_final_sac_keep_nominal_stop_and_phase_schedule(self):
        for adaptive, tail in ((True, 0), (False, 0), (False, 200)):
            with self.subTest(adaptive=adaptive, tail=tail):
                self.policy = self.new_policy(cycle_mode=adaptive, final_sac_iters=tail,
                                              num_cycles=1, sac_iters=4)
                self.policy.load_state_dict(self.teacher_state)
                p = self.policy
                p.configure_training_budget(6)
                for iteration in range(6):
                    self.assertFalse(p.training_complete())
                    self.train_one_iteration()
                self.assertTrue(p.training_complete())
                self.assertEqual(p.finetune_iters_completed, 6)
                self.assertEqual(p.stage, 1)
                self.assertEqual(p.cycles_completed, 1)
                self.assertFalse(p._final_sac_active)

    def test_terminal_refresh_keeps_natural_plateau_and_maximum_rules(self):
        for end_reason, refresh_length in (("plateau", 3), ("perception_max", 4)):
            with self.subTest(reason=end_reason):
                self.policy = self.new_policy(final_sac_iters=3, cycle_sac_iters_max=2)
                self.policy.load_state_dict(self.teacher_state)
                p = self.policy
                p.configure_training_budget(5)
                self.finish_teacher()
                self.sac_iteration()
                self.sac_iteration()
                self.assertEqual(p.stage, 1)
                for iteration in range(1, refresh_length + 1):
                    # Constant error plateaus after two checks at 2 and 3;
                    # halving RMSE each iteration cannot plateau before the cap.
                    mse = 1. if end_reason == "plateau" else .25 ** iteration
                    def monitor(td, target=None, mse=mse):
                        p._perception_error_sum.add_(4 * mse)
                        p._perception_error_count.add_(4)
                    with patch.object(p, "_record_perception_error", side_effect=monitor):
                        self.perception_iteration()
                    self.assertTrue(p._final_sac_active)
                    self.assertFalse(p.training_complete())
                    self.assertEqual(p.stage, 2 if iteration == refresh_length else 1)
                self.assertEqual(p._last_transition_reason, end_reason)
                self.assertEqual(p.perception_iters_completed, refresh_length)
                self.assertEqual(p.rl_phase_iters_completed, 0)
                for iteration in range(1, 4):
                    self.sac_iteration()
                    self.assertEqual(p.stage, 2)
                    self.assertEqual(p.training_complete(), iteration == 3)
                self.assertEqual(p.finetune_iters_completed, 4 + refresh_length + 3)
                self.assertEqual(p.cycles_completed, 1)

    def test_terminal_budget_during_teacher_keeps_remaining_teacher_training(self):
        for budget in (0, 1):
            with self.subTest(budget=budget):
                self.policy = self.new_policy(final_sac_iters=3)
                self.policy.load_state_dict(self.teacher_state)
                p = self.policy
                p.configure_training_budget(budget)
                self.assertFalse(p.training_complete())
                self.perception_iteration()
                self.assertTrue(p._final_sac_active)
                self.assertEqual(p.stage, 1)
                self.assertEqual(p.perception_iters_completed, 1)
                self.assertFalse(p.training_complete())
                self.perception_iteration()
                self.assertEqual(p.stage, 2)
                for _ in range(3):
                    self.sac_iteration()
                self.assertTrue(p.training_complete())
                self.assertEqual(p.finetune_iters_completed, 5)

    def test_terminal_checkpoint_preserves_deadline_through_refresh_and_sac_resume(self):
        self.policy = self.new_policy(final_sac_iters=3, cycle_sac_iters_max=2)
        self.policy.load_state_dict(self.teacher_state)
        p = self.policy
        p.configure_training_budget(5)
        self.finish_teacher()
        self.sac_iteration()
        self.sac_iteration()
        with patch.object(p, "_record_perception_error", side_effect=self.constant_monitor):
            self.perception_iteration()
        self.assertTrue(p._final_sac_active)
        saved = copy.deepcopy(p.state_dict())
        self.assertEqual(saved["finetune_version"], 9)
        self.assertEqual(saved["training_target_iter"], 5)
        self.assertTrue(saved["final_sac_active"])
        self.policy = self.new_policy(final_sac_iters=3, cycle_sac_iters_max=2)
        self.policy.load_state_dict(saved)
        p = self.policy
        p.configure_training_budget(100)
        self.assertEqual(p._training_target_iter, 5)
        self.assertTrue(p._final_sac_active)
        self.assertEqual(p.stage, 1)
        self.assertEqual(p.perception_iters_completed, 1)
        with patch.object(p, "_record_perception_error", side_effect=self.constant_monitor):
            self.perception_iteration()
            self.perception_iteration()
        self.assertEqual(p.stage, 2)
        self.assertEqual(p.finetune_iters_completed, 7)
        self.sac_iteration()
        saved = copy.deepcopy(p.state_dict())
        self.policy = self.new_policy(final_sac_iters=3, cycle_sac_iters_max=2)
        self.policy.load_state_dict(saved)
        p = self.policy
        p.configure_training_budget(500)
        self.assertEqual(p._training_target_iter, 5)
        self.assertEqual(p.rl_phase_iters_completed, 1)
        self.assertFalse(p.training_complete())
        self.sac_iteration()
        self.assertFalse(p.training_complete())
        self.sac_iteration()
        self.assertTrue(p.training_complete())
        self.assertEqual(p.finetune_iters_completed, 10)

    def test_terminal_resume_honors_changed_tail_length_without_restarting_deadline(self):
        self.policy = self.new_policy(final_sac_iters=3)
        self.policy.load_state_dict(self.teacher_state)
        self.policy.configure_training_budget(4)
        self.finish_teacher()
        for _ in range(3):
            self.sac_iteration()
        self.assertTrue(self.policy.training_complete())
        saved = copy.deepcopy(self.policy.state_dict())
        for requested_tail in (5, 2):
            with self.subTest(tail=requested_tail):
                self.policy = self.new_policy(final_sac_iters=requested_tail)
                self.policy.load_state_dict(saved)
                p = self.policy
                p.configure_training_budget(100)
                self.assertEqual(p._training_target_iter, 4)
                self.assertTrue(p._final_sac_active)
                self.assertEqual(p.training_complete(), requested_tail <= 3)
                if requested_tail > 3:
                    self.sac_iteration()
                    self.assertFalse(p.training_complete())
                    self.sac_iteration()
                    self.assertTrue(p.training_complete())
                    self.assertEqual(p.finetune_iters_completed, 7)

    def test_disabling_tail_on_resume_starts_new_nominal_budget(self):
        self.policy = self.new_policy(final_sac_iters=3)
        self.policy.load_state_dict(self.teacher_state)
        self.policy.configure_training_budget(4)
        self.finish_teacher()
        for _ in range(3):
            self.sac_iteration()
        self.assertTrue(self.policy.training_complete())
        saved = copy.deepcopy(self.policy.state_dict())
        self.policy = self.new_policy(final_sac_iters=0)
        self.policy.load_state_dict(saved)
        p = self.policy
        p.configure_training_budget(2)
        self.assertEqual(p._training_target_iter, 7)
        self.assertFalse(p._final_sac_active)
        self.assertFalse(p.training_complete())
        self.sac_iteration()
        self.assertEqual(p.stage, 1)
        self.assertFalse(p.training_complete())
        self.perception_iteration()
        self.assertTrue(p.training_complete())
        self.assertEqual(p.finetune_iters_completed, 7)

    def test_nonterminal_resume_uses_new_invocation_budget(self):
        self.policy.configure_training_budget(30)
        self.finish_teacher()
        saved = copy.deepcopy(self.policy.state_dict())
        resumed = self.new_policy()
        resumed.load_state_dict(saved)
        resumed.configure_training_budget(10)
        self.assertEqual(resumed._training_target_iter, 12)
        self.assertFalse(resumed._final_sac_active)
        self.assertFalse(resumed.training_complete())


if __name__ == "__main__":
    unittest.main()
