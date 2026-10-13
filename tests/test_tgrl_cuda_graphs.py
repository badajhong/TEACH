"""Opt-in regressions for the actual TGRL CUDA graph and AMP training path.

Run: VAIC_TEST_CUDA_GRAPHS=1 PYTHONPATH=. python -m unittest discover \
    -s tests -p test_tgrl_cuda_graphs.py -v

The tiny policy uses real FlashSAC actors, both categorical critics, optimizers,
normalization and target updates. No simulator or teacher training is required.
Keep this separate from the CPU fixture, which deliberately disables Dynamo.
"""
import os
from types import SimpleNamespace
import unittest

import torch
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.flashsac.tgrl_flashsac_vel_finetune import (
    TGRLFlashSACVelFinetune, TGRLFlashSACVelFinetuneConfig,
)


@unittest.skipUnless(os.environ.get("VAIC_TEST_CUDA_GRAPHS") == "1",
                     "set VAIC_TEST_CUDA_GRAPHS=1 to run compiled CUDA regressions")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TGRLCudaGraphsTest(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.compile_enabled = torch._dynamo.config.patch(disable=False)
        self.compile_enabled.__enter__()
        self.addCleanup(self.compile_enabled.__exit__, None, None, None)
        torch.manual_seed(7)
        self.device = torch.device("cuda")
        # Match production's rollout : update minibatch ratio (4096 : 2048).
        self.n, self.batch_size = 32, 16
        dims = {"command": 10, "policy": 6, "priv": 12, "object_": 22,
                "object_geo_": 384, "vel_command": 4, "ref_joint_pos_": 3}
        spec = Composite({k: Unbounded((self.n, d)) for k, d in dims.items()}, shape=(self.n,))
        spec["depth"] = Unbounded((self.n, 1, 36, 64))
        self.spec = spec.to(self.device)
        action_spec = Unbounded((self.n, 3), device=self.device)
        env = SimpleNamespace(
            transform=SimpleNamespace(transforms=[]), current_iter=0, max_episode_length=100,
            action_manager=SimpleNamespace(
                delay=torch.zeros(self.n, 1, device=self.device),
                alpha=torch.ones(self.n, 1, device=self.device)),
            observation_funcs={},
            command_manager=SimpleNamespace(future_steps=torch.tensor([1, 2], device=self.device)),
        )
        env.set_progress = lambda i: setattr(env, "current_iter", i)
        for key, dim in dims.items():
            env.observation_funcs[key] = SimpleNamespace(
                funcs={key: lambda dim=dim: torch.zeros(self.n, dim, device=self.device)},
                _compute=lambda dim=dim: torch.zeros(self.n, dim, device=self.device),
            )
        cfg = TGRLFlashSACVelFinetuneConfig(
            num_envs=self.n, train_every=4, num_env_steps=4096,
            use_compile=True, compile_mode="reduce-overhead", use_amp=True,
            normalize_reward=False, obs_future_steps=None, obs_drop_terms=(), latent_dim=256,
            actor_hidden_dim=256, critic_hidden_dim=256,
            critic_state_encoder_dim=256, critic_action_encoder_dim=256, critic_num_bins=101,
            adapt_num_minibatches=2, buffer_max_length=128, buffer_min_length=16,
            sample_batch_size=self.batch_size, updates_per_interaction_step=1,
            perception_iters=2, adapt_epochs=1,
        )
        self.policy = TGRLFlashSACVelFinetune(
            cfg, self.spec, action_spec, None, self.device, env)
        # Exercise the real perception-to-SAC transition without 500 simulator iterations.
        self.policy._start_rl()

    def rl_batch(self):
        p = self.policy
        batch = {}
        for key, actor_key, teacher_key in (
                ("observation", "actor_observation", "teacher_observation"),
                ("next_observation", "actor_next_observation", "teacher_next_observation"),
                ("one_step_next_observation", "actor_one_step_next_observation",
                 "teacher_one_step_next_observation")):
            td = self.spec.rand()[:self.batch_size]
            td["action_dr"] = p.action_dr()[:self.batch_size]
            obs = p._student_replay_obs(
                td, torch.randn(self.batch_size, p.cfg.latent_dim, device=self.device))
            batch[key] = p.critic_obs(obs)
            batch[actor_key] = obs.index_select(-1, p._student_replay_index)
            batch[teacher_key] = obs[:, :p.replay_obs_dim]
        batch.update(
            action=torch.rand(self.batch_size, p.action_dim, device=self.device) * 2 - 1,
            reward=torch.tensor([.1, .2, -.3, .4], device=self.device).repeat(4),
            terminated=torch.tensor([1., 1., 0., 0.], device=self.device).repeat(4),
            one_step_terminated=torch.tensor([0., 1., 0., 0.], device=self.device).repeat(4),
        )
        return batch

    def test_split_rollout_and_repeated_actor_critic_updates(self):
        p = self.policy
        self.assertEqual(p.flash_cfg.compile_mode, "reduce-overhead")
        self.assertTrue(p._grad_scaler.is_enabled())
        for network in (p._student, p._aux, p._critic, p._imitation_critic):
            self.assertTrue(hasattr(network.network, "_orig_mod"))
        before = [next(network.network.parameters()).detach().clone()
                  for network in (p._student, p._aux, p._critic, p._imitation_critic)]

        # Repetition reaches graph capture/replay, and alternation tests pending
        # backwards plus two successive forwards of the same actor/critic code.
        # The production GradScaler may skip initial random-batch updates while
        # calibrating its scale. Run enough steps to check successful updates too.
        for actor_update in (False, True) * 15:
            batch = self.rl_batch()
            rollout_obs = batch["actor_observation"].repeat(2, 1)
            with torch.no_grad():
                for _ in range(3):
                    actions = p._sample_split_rollout(rollout_obs)
                    self.assertEqual(actions.shape, (self.n, p.action_dim))
                    self.assertTrue(torch.isfinite(actions).all())
                    self.assertTrue((actions.abs() <= 1).all())
            info = p._tgrl_update_networks(batch, do_actor_update=actor_update)
            torch.cuda.synchronize()
            self.assertTrue(info)
            for key, value in info.items():
                self.assertTrue(torch.isfinite(value).all(), key)

        for name, network, initial in zip(
                ("student", "auxiliary", "reward critic", "imitation critic"),
                (p._student, p._aux, p._critic, p._imitation_critic), before):
            self.assertTrue(network.optimizer.state, name)
            self.assertTrue(any(state["step"] > 0 for state in network.optimizer.state.values()), name)
            self.assertFalse(torch.equal(next(network.network.parameters()), initial), name)
            gradients = [parameter.grad for parameter in network.network.parameters()
                         if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), name)


if __name__ == "__main__":
    unittest.main()
