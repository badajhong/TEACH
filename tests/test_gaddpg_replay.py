"""CPU tests for intervention replay: shortened returns use their actual horizon.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_gaddpg_replay.py
"""
import unittest

import torch

from active_adaptation.learning.flashsac.flashsac_vel_flat import NStepTrajectoryReplay
from active_adaptation.learning.flashsac.gaddpg_cosil_flashsac_vel_finetune import InterventionNStepReplay

GAMMA, N_STEP = 0.9, 3
OPTIONS = dict(num_envs=2, obs_dim=2, action_dim=1, n_step=N_STEP, gamma=GAMMA, max_length=16,
               min_length=2, batch_size=2, device="cpu", obs_dtype=torch.float32)


def bootstrap_discount(batch, gamma=GAMMA, n_step=N_STEP):
    # the critic target multiplies the bootstrap by gamma^n (1 - terminated)
    return gamma ** n_step * (1.0 - batch["terminated"])


class GADDPGReplayTest(unittest.TestCase):
    replay_cls = InterventionNStepReplay

    def add(self, replay, step, *, intervened=(False, False), done=(False, False), cut=(False, False),
            discount=(1.0, 1.0)):
        obs = torch.tensor([[10.0 * step, 0.0], [10.0 * step, 1.0]])
        replay.add(
            obs=obs, next_obs=obs + torch.tensor([10.0, 0.0]),
            action=torch.tensor([[step], [step + 0.5]]),
            reward=torch.tensor([step + 1.0, step + 2.0]),
            discount=torch.tensor(discount), done=torch.tensor(done), cut=torch.tensor(cut),
            valid=torch.tensor([True, True]), intervened=torch.tensor(intervened),
        )

    def test_matches_shared_replay_without_interventions_or_bootstrapped_ends(self):
        torch.manual_seed(0)
        ours, base = self.replay_cls(**OPTIONS), NStepTrajectoryReplay(**OPTIONS)
        for step in range(12):
            kwargs = dict(
                obs=torch.randn(2, 2), next_obs=torch.randn(2, 2), action=torch.randn(2, 1),
                reward=torch.randn(2), discount=torch.rand(2), done=torch.rand(2) < 0.3,
                cut=torch.rand(2) < 0.5, valid=torch.ones(2, dtype=torch.bool),
            )
            # The shared replay still uses gamma^n at early truncations. Compare
            # unchanged cases here; validate truncations against the return below.
            kwargs["cut"] |= kwargs["done"]
            base.add(**kwargs)
            ours.add(**kwargs, intervened=torch.zeros(2, dtype=torch.bool))
        t, env = torch.arange(8).repeat_interleave(2), torch.arange(2).repeat(8)
        expected, actual = base.gather(t, env), ours.gather(t, env)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_episode_end_uses_elapsed_horizon_and_terminal_observation(self):
        for gamma in (0.9, 0.99):
            for start in (0, 7):  # T=8: the latter window crosses the ring boundary.
                for length in (1, 2, 3):
                    for discount in (1.0, 0.5):
                        for cut in (False, True):
                            with self.subTest(gamma=gamma, start=start, length=length,
                                              discount=discount, cut=cut):
                                replay = self.replay_cls(**{**OPTIONS, "gamma": gamma})
                                for step in range(start + N_STEP + 1):
                                    ends = step == start + length - 1
                                    self.add(replay, step, done=(ends, False),
                                             cut=(ends and cut, False), discount=(discount, 1.0))
                                batch = replay.gather(torch.tensor([start % replay.T] * 2),
                                                      torch.tensor([0, 1]))
                                expected_reward = sum(
                                    (gamma * discount) ** k * (start + k + 1)
                                    for k in range(length))
                                # The other env keeps the full n-step return.
                                other_reward = sum(gamma ** k * (start + k + 2)
                                                   for k in range(N_STEP))
                                torch.testing.assert_close(
                                    batch["reward"], torch.tensor([expected_reward, other_reward]))
                                torch.testing.assert_close(batch["next_observation"], torch.tensor([
                                    [10.0 * (start + length), 0.0],
                                    [10.0 * (start + N_STEP), 1.0],
                                ]))
                                expected_bootstrap = 0.0 if cut else (gamma * discount) ** length
                                torch.testing.assert_close(
                                    bootstrap_discount(batch, gamma=gamma),
                                    torch.tensor([expected_bootstrap, gamma ** N_STEP]))

    def test_teacher_at_first_step_keeps_full_window(self):
        replay = self.replay_cls(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 0, False))
        batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0 + 0.9 * 2.0 + 0.81 * 3.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[30.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA ** 3]))

    def test_teacher_at_second_step_cuts_to_one_step(self):
        replay = self.replay_cls(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 1, False))
        batch = replay.gather(torch.tensor([0, 0]), torch.tensor([0, 1]))
        # env 0: r_0, bootstrap from s_1 with gamma; env 1 (no teacher): the full 3-step window
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0, 2.0 + 0.9 * 3.0 + 0.81 * 4.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[10.0, 0.0], [30.0, 1.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA, GAMMA ** 3]))
        torch.testing.assert_close(batch["action"], torch.tensor([[0.0], [0.5]]))

    def test_teacher_at_third_step_cuts_to_two_steps_with_soft_discount(self):
        replay = self.replay_cls(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 2, False), discount=(0.5, 1.0))
        batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0 + 0.9 * 0.5 * 2.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[20.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA ** 2 * 0.25]))

    def test_episode_end_before_teacher_takes_precedence(self):
        for cut in (False, True):
            with self.subTest(cut=cut):
                replay = self.replay_cls(**OPTIONS)
                for step in range(5):
                    self.add(replay, step, intervened=(step == 2, False), done=(step == 1, False),
                             cut=(step == 1 and cut, False))
                batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
                torch.testing.assert_close(batch["reward"], torch.tensor([1.0 + 0.9 * 2.0]))
                torch.testing.assert_close(batch["next_observation"], torch.tensor([[20.0, 0.0]]))
                torch.testing.assert_close(bootstrap_discount(batch),
                                           torch.tensor([0.0 if cut else GAMMA ** 2]))

    def test_teacher_action_stops_window_before_that_actions_episode_end(self):
        replay = self.replay_cls(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 1, False), done=(step == 1, False),
                     cut=(step == 1, False))
        batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
        # The teacher's terminal action/reward is outside this return.
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[10.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA]))

    def test_window_starting_inside_a_takeover_is_one_step(self):
        replay = self.replay_cls(**OPTIONS)
        for step in range(6):
            self.add(replay, step, intervened=(1 <= step <= 4, False))
        batch = replay.gather(torch.tensor([2]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([3.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[30.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA]))


if __name__ == "__main__":
    unittest.main()
