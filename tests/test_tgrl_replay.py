"""CPU tests for TGRL's one-step bootstrap shared with FlashSAC n-step replay.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_tgrl_replay.py
"""
import unittest

import torch

from active_adaptation.learning.ppo.flashsac_vel_flat import NStepTrajectoryReplay


class TGRLReplayTest(unittest.TestCase):
    def replay(self, **kwargs):
        options = dict(
            num_envs=2, obs_dim=2, action_dim=1, n_step=3, gamma=0.9,
            max_length=12, min_length=2, batch_size=2, device="cpu",
            obs_dtype=torch.float16, include_one_step=True,
        )
        options.update(kwargs)
        return NStepTrajectoryReplay(**options)

    def add(self, replay, step, *, discount=(1.0, 1.0), done=(False, False),
            cut=(False, False), valid=(True, True), next_obs=None):
        obs = torch.tensor([[10.0 * step, 0.0], [10.0 * step, 1.0]])
        if next_obs is None:
            next_obs = obs + torch.tensor([10.0, 0.0])
        replay.add(
            obs=obs, next_obs=next_obs,
            action=torch.tensor([[step], [step + 0.5]]),
            reward=torch.tensor([step + 1.0, step + 2.0]),
            discount=torch.tensor(discount), done=torch.tensor(done),
            cut=torch.tensor(cut), valid=torch.tensor(valid),
        )

    def test_one_step_and_n_step_observations_share_sampled_rows(self):
        replay = self.replay()
        for step in range(4):
            self.add(replay, step)
        batch = replay.gather(torch.tensor([1, 0]), torch.tensor([1, 0]))
        torch.testing.assert_close(batch["observation"], torch.tensor([[10.0, 1.0], [0.0, 0.0]]))
        torch.testing.assert_close(batch["action"], torch.tensor([[1.5], [0.0]]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[40.0, 1.0], [30.0, 0.0]]))
        torch.testing.assert_close(batch["one_step_next_observation"], torch.tensor([[20.0, 1.0], [10.0, 0.0]]))
        torch.testing.assert_close(batch["reward"], torch.tensor([3.0 + 0.9 * 4.0 + 0.81 * 5.0,
                                                                  1.0 + 0.9 * 2.0 + 0.81 * 3.0]))
        self.assertEqual(batch["one_step_next_observation"].dtype, torch.float32)

    def test_hard_terminal_keeps_final_observation_through_reset_placeholder(self):
        replay = self.replay()
        final_obs = torch.tensor([[99.0, 0.0], [99.0, 1.0]])
        self.add(replay, 0, done=(True, True), cut=(True, True), next_obs=final_obs)
        self.add(replay, 1, valid=(False, False))
        self.add(replay, 2)
        batch = replay.gather(torch.tensor([0, 0]), torch.tensor([0, 1]))
        torch.testing.assert_close(batch["one_step_next_observation"], final_obs)
        torch.testing.assert_close(batch["next_observation"], final_obs)
        torch.testing.assert_close(batch["one_step_terminated"], torch.ones(2))
        torch.testing.assert_close(batch["terminated"], torch.ones(2))
        torch.testing.assert_close(batch["reward"], torch.tensor([1.0, 2.0]))

    def test_truncation_bootstraps_from_final_observation(self):
        replay = self.replay()
        final_obs = torch.tensor([[77.0, 0.0], [77.0, 1.0]])
        self.add(replay, 0, done=(True, True), cut=(False, False),
                 discount=(0.6, 1.0), next_obs=final_obs)
        self.add(replay, 1, valid=(False, False))
        self.add(replay, 2)
        batch = replay.gather(torch.tensor([0, 0]), torch.tensor([0, 1]))
        torch.testing.assert_close(batch["one_step_next_observation"], final_obs)
        torch.testing.assert_close(batch["next_observation"], final_obs)
        torch.testing.assert_close(batch["one_step_terminated"], torch.tensor([0.4, 0.0]))
        torch.testing.assert_close(batch["terminated"], torch.tensor([0.4, 0.0]))

    def test_one_step_discount_ignores_later_discounts_and_termination(self):
        replay = self.replay()
        self.add(replay, 0, discount=(0.5, 0.8))
        self.add(replay, 1, discount=(0.8, 0.5), done=(False, True), cut=(False, True))
        self.add(replay, 2, discount=(0.75, 1.0))
        batch = replay.gather(torch.tensor([0, 0]), torch.tensor([0, 1]))
        torch.testing.assert_close(batch["one_step_terminated"], torch.tensor([0.5, 0.2]))
        torch.testing.assert_close(batch["terminated"], torch.tensor([0.7, 1.0]))
        torch.testing.assert_close(batch["one_step_next_observation"], torch.tensor([[10.0, 0.0], [10.0, 1.0]]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[30.0, 0.0], [20.0, 1.0]]))

    def test_one_step_wraps_with_same_ring_row(self):
        replay = self.replay()
        self.assertEqual(replay.T, 6)
        for step in range(9):
            self.add(replay, step)
        batch = replay.gather(torch.tensor([5, 4]), torch.tensor([1, 0]))
        torch.testing.assert_close(batch["observation"], torch.tensor([[50.0, 1.0], [40.0, 0.0]]))
        torch.testing.assert_close(batch["one_step_next_observation"], torch.tensor([[60.0, 1.0], [50.0, 0.0]]))
        torch.testing.assert_close(batch["next_observation"], torch.tensor([[80.0, 1.0], [70.0, 0.0]]))

    def test_opt_in_preserves_n_step_outputs_and_does_not_duplicate_storage(self):
        extended = self.replay()
        standard = self.replay(include_one_step=False)
        self.assertEqual(extended.nbytes(), standard.nbytes())
        for replay in (extended, standard):
            self.add(replay, 0, discount=(0.6, 0.8))
            self.add(replay, 1, done=(False, True), cut=(False, False))
            self.add(replay, 2)
        args = torch.tensor([0, 0]), torch.tensor([1, 0])
        expected, actual = standard.gather(*args), extended.gather(*args)
        self.assertEqual(set(expected), {"observation", "action", "reward", "terminated", "next_observation"})
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        one_step = self.replay(n_step=1, include_one_step=False)
        self.add(one_step, 0, discount=(0.6, 0.8))
        one_step_batch = one_step.gather(*args)
        torch.testing.assert_close(actual["one_step_next_observation"], one_step_batch["next_observation"])
        torch.testing.assert_close(actual["one_step_terminated"], one_step_batch["terminated"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for pinned replay transfer")
    def test_pinned_staging_and_reuse_match_cpu_replay(self):
        cpu = self.replay()
        cuda = self.replay(device="cuda", storage_device="cpu")
        for replay in (cpu, cuda):
            self.add(replay, 0, discount=(0.6, 0.8))
            self.add(replay, 1, done=(False, True), cut=(False, False))
            self.add(replay, 2, done=(True, False), cut=(True, False))
            self.add(replay, 3)
        self.assertEqual(cuda._staging[0][0].shape, (6, 2))
        self.assertEqual(cuda._staging[0][1].shape, (2, 4))
        # Eight configured-size batches reuse every staging slot; the final smaller batch
        # exercises gather's plain transfer path with the same additional fields.
        for index in range(9):
            rows = torch.tensor([index % 2, 0]) if index < 8 else torch.tensor([0])
            envs = torch.tensor([0, index % 2]) if index < 8 else torch.tensor([1])
            expected, actual = cpu.gather(rows, envs), cuda.gather(rows, envs)
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assertEqual(actual[key].device.type, "cuda")
                torch.testing.assert_close(actual[key].cpu(), expected[key])


if __name__ == "__main__":
    unittest.main()
