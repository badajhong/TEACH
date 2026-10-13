"""CPU-only causal decision tests, without importing the simulator/policy registry."""

import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "finetune_schedule", Path(__file__).parents[1] / "active_adaptation/learning/flashsac/finetune_schedule.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AdaptivePerceptionSchedule = _MODULE.AdaptivePerceptionSchedule
validate_config = _MODULE.validate_config


def config(**changes):
    values = dict(
        perception_trigger=0.03, perception_error_window=3, perception_trigger_patience=2,
        plateau_window=2, plateau_tol=0.01, plateau_patience=2, plateau_check_interval=2,
        cycle_perception_iters_min=4, cycle_perception_iters_max=20,
        cycle_sac_iters_min=4, cycle_sac_iters_max=20,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def observation(schedule, rms, count=10):
    schedule.observe(rms * rms * count, count)


def with_reference(**changes):
    schedule = AdaptivePerceptionSchedule(config(**changes))
    observation(schedule, 1.0)
    schedule.establish_reference()
    schedule.reset_phase()
    return schedule


class AdaptiveScheduleTest(unittest.TestCase):
    def test_trigger_waits_for_cooldown_and_consecutive_smoothed_excess(self):
        schedule = with_reference()
        for iteration in range(1, 4):
            observation(schedule, 1.1)
            self.assertIsNone(schedule.sac_reason(iteration))
            self.assertEqual(schedule.trigger_streak, 0)
        observation(schedule, 1.1)
        self.assertIsNone(schedule.sac_reason(4))
        self.assertEqual(schedule.trigger_streak, 1)
        self.assertIsNone(schedule.sac_reason(4))  # Duplicate calls do not build patience.
        observation(schedule, 1.1)
        self.assertEqual(schedule.sac_reason(5), "error_trigger")

    def test_full_window_required_and_dip_breaks_streak(self):
        schedule = with_reference(cycle_sac_iters_min=1)
        for iteration in range(1, 4):
            observation(schedule, 1.1)
            self.assertIsNone(schedule.sac_reason(iteration))
        self.assertEqual(schedule.trigger_streak, 1)
        observation(schedule, 0.5)
        self.assertIsNone(schedule.sac_reason(4))
        self.assertEqual(schedule.trigger_streak, 0)
        for iteration in (5, 6, 7):
            observation(schedule, 1.1)
            self.assertIsNone(schedule.sac_reason(iteration))
        observation(schedule, 1.1)
        self.assertEqual(schedule.sac_reason(8), "error_trigger")

    def test_caps_are_unconditional_even_without_reference_or_valid_data(self):
        schedule = AdaptivePerceptionSchedule(config())
        self.assertIsNone(schedule.sac_reason(19))
        self.assertEqual(schedule.sac_reason(20), "sac_max")
        self.assertEqual(schedule.refresh_reason(20), "perception_max")

    def test_rms_weights_states_and_reference_uses_recent_tail_not_minimum(self):
        schedule = AdaptivePerceptionSchedule(config(perception_error_window=2))
        observation(schedule, 0.1)
        observation(schedule, 1.0, count=1)
        observation(schedule, 2.0, count=3)
        expected = math.sqrt(13 / 4)
        self.assertEqual(schedule.last_error, 2.0)
        self.assertAlmostEqual(schedule.smoothed_error, expected)
        self.assertAlmostEqual(schedule.establish_reference(), expected)
        self.assertEqual(schedule.relative_error, 0.0)
        schedule.reset_phase()
        self.assertEqual(schedule.reference_error, expected)
        self.assertIsNone(schedule.smoothed_error)
        with self.assertRaises(RuntimeError):
            schedule.establish_reference()

    def test_invalid_data_breaks_contiguous_windows_and_both_streaks(self):
        for bad in ((0, 0), (float("nan"), 10), (1, float("inf")), (-1, 10)):
            with self.subTest(bad=bad):
                schedule = with_reference(cycle_sac_iters_min=1)
                for iteration in range(1, 4):
                    observation(schedule, 1.1)
                    schedule.sac_reason(iteration)
                schedule.plateau_streak = 1
                schedule.observe(*bad)
                self.assertEqual(schedule.trigger_streak, 0)
                self.assertEqual(schedule.plateau_streak, 0)
                self.assertIsNone(schedule.last_error)
                self.assertIsNone(schedule.smoothed_error)
                self.assertIsNone(schedule.relative_error)
                self.assertEqual(schedule.reference_error, 1.0)
                self.assertIsNone(schedule.sac_reason(4))
                observation(schedule, 1.1)
                self.assertIsNone(schedule.sac_reason(5))
                with self.assertRaises(RuntimeError):
                    schedule.observe(0, 0)
                    schedule.establish_reference()

    def test_first_reference_weights_partial_iteration_without_recording_it(self):
        schedule = AdaptivePerceptionSchedule(config(perception_error_window=3))
        observation(schedule, 10.0, count=1)  # Outside the reference tail.
        observation(schedule, 1.0, count=1)
        observation(schedule, 2.0, count=3)
        history = copy.deepcopy(schedule.state_dict()["history"])
        self.assertTrue(schedule.try_initialize_reference(16, 1))
        self.assertAlmostEqual(schedule.reference_error, math.sqrt(29 / 5))
        self.assertEqual(schedule.state_dict()["history"], history)
        # The partial iteration eventually completes with one additional sample.
        # Its complete observation is recorded exactly once by the normal path.
        schedule.observe(20, 2)
        self.assertEqual(schedule.state_dict()["history"], [*history, [20.0, 2.0]])
        self.assertAlmostEqual(schedule.smoothed_error, math.sqrt(33 / 6))
        self.assertAlmostEqual(schedule.reference_error, math.sqrt(29 / 5))
        state = schedule.state_dict()
        self.assertFalse(schedule.try_initialize_reference(100, 1))
        self.assertEqual(schedule.state_dict(), state)

    def test_first_reference_can_use_only_partial_iteration_and_roundtrip(self):
        schedule = AdaptivePerceptionSchedule(config())
        self.assertTrue(schedule.try_initialize_reference(12, 3))
        self.assertEqual(schedule.reference_error, 2.0)
        self.assertEqual(schedule.state_dict()["history"], [])
        restored = AdaptivePerceptionSchedule(schedule.cfg)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.state_dict(), schedule.state_dict())
        self.assertFalse(restored.try_initialize_reference(0, 1))
        self.assertEqual(restored.reference_error, 2.0)

    def test_first_reference_skips_invalid_current_data_even_with_valid_history(self):
        for bad in ((0, 0), (float("nan"), 10), (1, float("inf")), (-1, 10), (None, None)):
            with self.subTest(bad=bad):
                schedule = AdaptivePerceptionSchedule(config())
                observation(schedule, 1.0)
                before = schedule.state_dict()
                self.assertFalse(schedule.try_initialize_reference(*bad))
                self.assertEqual(schedule.state_dict(), before)
                # Normal observation handling breaks history on an invalid
                # completed iteration; the later reference cannot bridge it.
                schedule.observe(0, 0)
                self.assertTrue(schedule.try_initialize_reference(9, 1))
                self.assertEqual(schedule.reference_error, 3.0)

    def test_plateau_needs_adjacent_windows_spaced_checks_and_patience(self):
        schedule = AdaptivePerceptionSchedule(config())
        for iteration in range(1, 6):
            observation(schedule, 1.0)
            self.assertIsNone(schedule.refresh_reason(iteration))
            self.assertEqual(schedule.plateau_streak, int(iteration >= 4))
        observation(schedule, 1.0)
        self.assertEqual(schedule.refresh_reason(6), "plateau")
        self.assertEqual(schedule.plateau_improvement, 0.0)
        self.assertEqual(schedule.refresh_reason(6), "plateau")
        self.assertEqual(schedule.plateau_streak, 2)

    def test_improvement_resets_plateau_and_worsening_counts_as_no_improvement(self):
        schedule = AdaptivePerceptionSchedule(config())
        for iteration, rms in enumerate((1, 1, 1, 1, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7), 1):
            observation(schedule, rms)
            result = schedule.refresh_reason(iteration)
            if iteration == 4:
                self.assertEqual(schedule.plateau_streak, 1)
            if iteration == 6:
                self.assertEqual(schedule.plateau_streak, 0)
                self.assertAlmostEqual(schedule.plateau_improvement, 0.5)
            if iteration == 8:
                self.assertEqual(schedule.plateau_streak, 1)
                self.assertLess(schedule.plateau_improvement, 0)
            if iteration < 10:
                self.assertIsNone(result)
        self.assertEqual(result, "plateau")

    def test_zero_reference_does_not_trigger_constant_zero_and_can_plateau(self):
        schedule = AdaptivePerceptionSchedule(config(cycle_sac_iters_min=1))
        observation(schedule, 0)
        self.assertEqual(schedule.establish_reference(), 0)
        schedule.reset_phase()
        for iteration in range(1, 7):
            observation(schedule, 0)
            self.assertIsNone(schedule.sac_reason(iteration))
            result = schedule.refresh_reason(iteration)
        self.assertEqual(schedule.relative_error, 0)
        self.assertEqual(schedule.plateau_improvement, 0)
        self.assertEqual(result, "plateau")

    def test_state_roundtrip_preserves_next_trigger_and_rejects_invalid_state_atomically(self):
        schedule = with_reference(cycle_sac_iters_min=1)
        for iteration in range(1, 4):
            observation(schedule, 1.1, count=iteration)
            schedule.sac_reason(iteration)
        state = schedule.state_dict()
        restored = AdaptivePerceptionSchedule(schedule.cfg)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        for instance in (schedule, restored):
            observation(instance, 1.1, count=4)
            self.assertEqual(instance.sac_reason(4), "error_trigger")
        before = restored.state_dict()
        broken = copy.deepcopy(before)
        broken["history"][0][1] = 0
        with self.assertRaises(ValueError):
            restored.load_state_dict(broken)
        self.assertEqual(restored.state_dict(), before)

    def test_state_roundtrip_preserves_plateau_patience(self):
        schedule = AdaptivePerceptionSchedule(config())
        for iteration in range(1, 5):
            observation(schedule, 1.0)
            schedule.refresh_reason(iteration)
        restored = AdaptivePerceptionSchedule(schedule.cfg)
        restored.load_state_dict(schedule.state_dict())
        for iteration in (5, 6):
            observation(restored, 1.0)
            result = restored.refresh_reason(iteration)
        self.assertEqual(result, "plateau")

    def test_validation_and_cap_only_configuration(self):
        for name, value in (("perception_trigger", float("nan")), ("plateau_tol", 0),
                            ("perception_error_window", 2.5), ("plateau_patience", True),
                            ("cycle_sac_iters_min", 21), ("cycle_perception_iters_max", 3)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_config(config(**{name: value}))
        # The hard cap remains useful if no full plateau comparison fits.
        schedule = AdaptivePerceptionSchedule(config(plateau_window=100))
        for iteration in range(1, 20):
            observation(schedule, 1.0)
            self.assertIsNone(schedule.refresh_reason(iteration))
        self.assertEqual(schedule.refresh_reason(20), "perception_max")


if __name__ == "__main__":
    unittest.main()
