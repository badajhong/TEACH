"""CPU tests for the GA-DDPG CoSIL replay: n-step windows stop before teacher actions.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_gaddpg_replay.py
"""
import unittest

import torch

from active_adaptation.learning.ppo.flashsac_vel_flat import NStepTrajectoryReplay
from active_adaptation.learning.ppo.gaddpg_cosil_flashsac_vel_finetune import InterventionNStepReplay

GAMMA, N_STEP = 0.9, 3
OPTIONS = dict(num_envs=2, obs_dim=2, action_dim=1, n_step=N_STEP, gamma=GAMMA, max_length=16,
               min_length=2, batch_size=2, device="cpu", obs_dtype=torch.float32)


def bootstrap_discount(batch):
    # the critic target multiplies the bootstrap by gamma^n (1 - terminated)
    return GAMMA ** N_STEP * (1.0 - batch["terminated"])


class GADDPGReplayTest(unittest.TestCase):
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

    def test_matches_shared_replay_without_interventions(self):
        torch.manual_seed(0)
        ours, base = InterventionNStepReplay(**OPTIONS), NStepTrajectoryReplay(**OPTIONS)
        for step in range(12):
            kwargs = dict(
                obs=torch.randn(2, 2), next_obs=torch.randn(2, 2), action=torch.randn(2, 1),
                reward=torch.randn(2), discount=torch.rand(2), done=torch.rand(2) < 0.3,
                cut=torch.rand(2) < 0.5, valid=torch.ones(2, dtype=torch.bool),
            )
            base.add(**kwargs)
            ours.add(**kwargs, intervened=torch.zeros(2, dtype=torch.bool))
        t, env = torch.arange(8).repeat_interleave(2), torch.arange(2).repeat(8)
        expected, actual = base.gather(t, env), ours.gather(t, env)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_teacher_at_first_step_keeps_full_window(self):
        replay = InterventionNStepReplay(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 0, False))
        batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0 + 0.9 * 2.0 + 0.81 * 3.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[30.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA ** 3]))

    def test_teacher_at_second_step_cuts_to_one_step(self):
        replay = InterventionNStepReplay(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 1, False))
        batch = replay.gather(torch.tensor([0, 0]), torch.tensor([0, 1]))
        # env 0: r_0, bootstrap from s_1 with gamma; env 1 (no teacher): the full 3-step window
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0, 2.0 + 0.9 * 3.0 + 0.81 * 4.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[10.0, 0.0], [30.0, 1.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA, GAMMA ** 3]))
        torch.testing.assert_close(batch["action"], torch.tensor([[0.0], [0.5]]))

    def test_teacher_at_third_step_cuts_to_two_steps_with_soft_discount(self):
        replay = InterventionNStepReplay(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 2, False), discount=(0.5, 1.0))
        batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0 + 0.9 * 0.5 * 2.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[20.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA ** 2 * 0.25]))

    def test_episode_end_before_teacher_takes_precedence(self):
        replay = InterventionNStepReplay(**OPTIONS)
        for step in range(5):
            self.add(replay, step, intervened=(step == 2, False), done=(step == 1, False),
                     cut=(step == 1, False))
        batch = replay.gather(torch.tensor([0]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0 + 0.9 * 2.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[20.0, 0.0]]))
        torch.testing.assert_close(batch["terminated"], torch.tensor([1.0]))

    def test_window_starting_inside_a_takeover_is_one_step(self):
        replay = InterventionNStepReplay(**OPTIONS)
        for step in range(6):
            self.add(replay, step, intervened=(1 <= step <= 4, False))
        batch = replay.gather(torch.tensor([2]), torch.tensor([0]))
        torch.testing.assert_close(batch["reward"], torch.tensor([3.0]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[30.0, 0.0]]))
        torch.testing.assert_close(bootstrap_discount(batch), torch.tensor([GAMMA]))


if __name__ == "__main__":
    unittest.main()
