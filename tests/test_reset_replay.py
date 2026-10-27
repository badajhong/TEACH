"""Episode-start observations must not overwrite terminal bootstrap states."""
import unittest

import torch

from active_adaptation.learning.flashsac.flashsac_vel_flat import NStepTrajectoryReplay


class ResetReplayTest(unittest.TestCase):
    def replay(self, **kwargs):
        args = dict(num_envs=2, obs_dim=2, action_dim=1, n_step=3, gamma=.9,
                    max_length=16, min_length=2, batch_size=2, device="cpu",
                    obs_dtype=torch.float16, include_one_step=True)
        args.update(kwargs)
        return NStepTrajectoryReplay(**args)

    def add(self, replay, step, *, reset=(False, False), done=(False, False), cut=(False, False)):
        # At a boundary current and previous next observations are deliberately
        # different. They must remain distinct after storing and sampling.
        obs = torch.tensor([[step * 10. + 1, 0.], [step * 10. + 1, 1.]])
        next_obs = torch.tensor([[(step + 1) * 10., 0.], [(step + 1) * 10., 1.]])
        replay.add(obs, next_obs, torch.ones(2, 1), torch.ones(2), torch.ones(2),
                   torch.tensor(done), torch.tensor(cut), torch.ones(2, dtype=torch.bool),
                   reset_mask=torch.tensor(reset))
        return obs, next_obs

    def check_boundaries(self, replay):
        self.add(replay, 0, reset=(True, True), done=(True, True), cut=(False, True))
        initial, _ = self.add(replay, 1, reset=(True, True))
        self.add(replay, 2)
        _, final = self.add(replay, 3)
        rows, envs = torch.tensor([0, 0], device=replay.storage), torch.tensor([0, 1], device=replay.storage)
        before = replay.gather(rows, envs)
        torch.testing.assert_close(before["next_observation"].cpu(), torch.tensor([[10., 0.], [10., 1.]]))
        torch.testing.assert_close(before["one_step_next_observation"], before["next_observation"])
        torch.testing.assert_close(before["terminated"].cpu(), torch.tensor([0., 1.]))
        after = replay.gather(torch.tensor([1, 1], device=replay.storage), envs)
        torch.testing.assert_close(after["observation"].cpu(), initial)
        torch.testing.assert_close(after["next_observation"].cpu(), final)
        torch.testing.assert_close(after["reward"].cpu(), torch.full((2,), 1. + .9 + .81))

    def test_hard_end_soft_end_and_first_action_are_distinct(self):
        self.check_boundaries(self.replay())

    def test_sparse_reset_cache_expires_on_ring_wraparound(self):
        replay = self.replay()
        self.check_boundaries(replay)
        self.assertIn(1, replay._reset_obs)
        for step in range(4, 13):
            self.add(replay, step)
        self.assertFalse(replay._reset_obs)
        self.assertFalse(replay._reset_rows.any())
        batch = replay.gather(torch.tensor([1, 1]), torch.tensor([0, 1]))
        torch.testing.assert_close(batch["observation"], torch.tensor([[90., 0.], [90., 1.]]))

    def test_async_env_resets_only_replace_selected_current_states(self):
        replay = self.replay()
        self.add(replay, 0, done=(True, False))
        initial, _ = self.add(replay, 1, reset=(True, False))
        self.add(replay, 2)
        self.add(replay, 3)
        batch = replay.gather(torch.tensor([1, 1]), torch.tensor([0, 1]))
        expected = initial.clone()
        expected[1, 0] = 10.
        torch.testing.assert_close(batch["observation"], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_pinned_and_cuda_storage_restore_first_states(self):
        for storage in ("cpu", "cuda"):
            with self.subTest(storage=storage):
                self.check_boundaries(self.replay(device="cuda", storage_device=storage))


if __name__ == "__main__":
    unittest.main()
