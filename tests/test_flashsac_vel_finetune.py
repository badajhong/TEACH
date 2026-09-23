"""CPU regression tests for phase boundaries, frozen replay and checkpoint transfer.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_flashsac_vel_finetune.py
The simulator/camera/compiled CUDA path is covered separately with scripts/train.py.
"""
import copy
import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.ppo.flashsac_vel import FlashSACVel, FlashSACVelConfig
from active_adaptation.learning.ppo.flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
)


class FinetuneTest(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.no_compile = torch._dynamo.config.patch(disable=True)
        self.no_compile.__enter__()
        torch.manual_seed(7)
        n = 4
        dims = {"command": 10, "policy": 6, "priv": 12, "object_": 22,
                "object_geo_": 384, "vel_command": 4, "ref_joint_pos_": 3}
        self.spec = Composite({k: Unbounded((n, d)) for k, d in dims.items()}, shape=(n,))
        self.spec["depth"] = Unbounded((n, 1, 36, 64))
        self.action_spec = Unbounded((n, 3))
        self.env = SimpleNamespace(
            transform=SimpleNamespace(transforms=[]), current_iter=0, max_episode_length=100,
            action_manager=SimpleNamespace(delay=torch.zeros(n, 1), alpha=torch.ones(n, 1)),
            observation_funcs={}, command_manager=SimpleNamespace(future_steps=torch.tensor([1, 2])),
        )
        self.env.set_progress = lambda i: setattr(self.env, "current_iter", i)
        for k, d in dims.items():
            self.env.observation_funcs[k] = SimpleNamespace(
                funcs={k: lambda d=d: torch.zeros(n, d)},
                _compute=lambda d=d: torch.zeros(n, d),
            )
        self.options = dict(
            num_envs=n, train_every=4, num_env_steps=4096,
            use_compile=False, use_amp=False, normalize_reward=False,
            obs_future_steps=None, obs_drop_terms=(), latent_dim=8,
            actor_hidden_dim=16, critic_hidden_dim=16,
            critic_state_encoder_dim=8, critic_action_encoder_dim=8, critic_num_bins=11,
            adapt_num_minibatches=2, buffer_max_length=128, buffer_min_length=16,
            sample_batch_size=8, updates_per_interaction_step=1,
        )
        self.teacher = FlashSACVel(FlashSACVelConfig(**self.options), self.spec,
                                  self.action_spec, None, "cpu", self.env)
        self.teacher_state = copy.deepcopy(self.teacher.state_dict())
        self.policy = self.new_policy()
        self.policy.load_state_dict(self.teacher_state)

    def tearDown(self):
        self.no_compile.__exit__(None, None, None)
        torch.set_num_threads(self.old_threads)

    def new_policy(self):
        cfg = FlashSACVelFinetuneConfig(**self.options, perception_warmup_iters=2, adapt_epochs=1)
        return FlashSACVelFinetune(cfg, self.spec, self.action_spec, None, "cpu", self.env)

    def observation(self):
        td = self.spec.rand()
        td["is_init"] = torch.zeros(4, 1, dtype=torch.bool)
        td["adapt_hx"] = torch.randn(4, 8)
        td["depth_hx"] = torch.randn(4, 64)
        td["step_count"] = torch.ones(4, 1, dtype=torch.long) * 3
        return td

    def warmup_iteration(self):
        p = self.policy
        rollout = p.get_rollout_policy("train")
        for _ in range(p.cfg.train_every):
            td = rollout(self.observation())
            p.add_transition(td)
            p.update()

    def assert_model_equal(self, model, state):
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_warmup_updates_only_students_then_freezes(self):
        p = self.policy
        self.assertFalse(p._critic.optimizer.state)
        student_before = copy.deepcopy(p.actor_adapt.state_dict())
        perception_before = [x.detach().clone() for x in p._perception_parameters()]
        self.warmup_iteration()
        self.assertEqual(p.stage, 1)
        self.assertIsNone(p.buffer)
        self.assert_model_equal(p.actor, self.teacher_state["actor"])
        self.assert_model_equal(p.critic, self.teacher_state["critic"])
        self.assertTrue(any(not torch.equal(v, student_before[k])
                            for k, v in p.actor_adapt.state_dict().items()))
        self.assertTrue(any(not torch.equal(a, b)
                            for a, b in zip(p._perception_parameters(), perception_before)))
        self.assertTrue(all(x.grad is None for x in p.actor.parameters()))
        # Stage 1 checkpoint resumes the local counter and new depth optimizer.
        mid = copy.deepcopy(p.state_dict())
        self.policy = self.new_policy()
        self.policy.load_state_dict(mid)
        self.assertEqual(self.policy.warmup_iters_completed, 1)
        self.warmup_iteration()
        p = self.policy
        self.assertEqual(p.stage, 2)
        self.assertTrue(p.needs_env_reset)
        self.assertIsNone(p.buffer)
        self.assertTrue(all(not x.requires_grad for pair in p._perception_pairs()
                            for model in pair for x in model.parameters()))
        self.assertTrue(all(x.requires_grad for x in p.actor_adapt.parameters()))
        self.assertFalse(p._student.optimizer.state)
        self.assert_model_equal(p.actor, self.teacher_state["actor"])
        self.assert_model_equal(p.critic, self.teacher_state["critic"])

    def test_frozen_replay_terminal_latent_and_resume(self):
        p = self.policy
        self.warmup_iteration()
        self.warmup_iteration()
        p.needs_env_reset = False
        frozen = [x.detach().clone() for x in p._perception_parameters()]
        rollout = p.get_rollout_policy("train")
        carry = self.observation()
        for i in range(12):
            td = rollout(carry)
            future = self.observation()
            for key in ("adapt_hx", "depth_hx"):
                future[key] = td["next", key].clone()
            future["terminated"] = torch.zeros(4, 1, dtype=torch.bool)
            future["truncated"] = torch.full((4, 1), i == 11, dtype=torch.bool)
            future["reward"] = torch.randn(4, 1)
            future["discount"] = torch.ones(4, 1)
            td["next"] = future
            with torch.no_grad():
                expected = future.select("policy", "vel_command", "depth", "is_init",
                                         "adapt_hx", "depth_hx").clone()
                p._perceive(expected, ema=True)
            p.add_transition(td)
            stored = p.buffer.obs[p.buffer.step % p.buffer.T]
            torch.testing.assert_close(stored[:, -8:].float(), expected["priv_pred"], atol=.002, rtol=.002)
            # The full teacher fields are preserved, not replaced by critic-selected command.
            torch.testing.assert_close(stored[:, :10].float(), future["command"], atol=.002, rtol=.002)
            if i > 0:
                current = p._student_replay_obs(td, td["priv_pred"])
                torch.testing.assert_close(
                    p.buffer.obs[(p.buffer.step - 1) % p.buffer.T].float(), current, atol=.002, rtol=.002)
            p.update()
            carry = future.select(*self.spec.keys(), "is_init", "adapt_hx", "depth_hx", "step_count").clone()
        self.assertGreater(p._update_step, 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(p._perception_parameters(), frozen)))
        self.assertFalse(p.buffer.cut[(p.buffer.step - 1) % p.buffer.T].any())
        saved = copy.deepcopy(p.state_dict())
        resumed = self.new_policy()
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.stage, 2)
        self.assertEqual(resumed._update_step, p._update_step)
        self.assertEqual(resumed._rl_schedule_updates, p._rl_schedule_updates)
        self.assertIsNone(resumed.buffer)
        # Loading applies UnitLinear normalization again; allow its float32 roundoff.
        for key, value in resumed.actor_adapt.state_dict().items():
            torch.testing.assert_close(value, saved["actor_adapt"][key], rtol=1e-6, atol=1e-7)
        self.assert_model_equal(resumed.critic, saved["critic"])
        self.assertEqual(resumed._student.optimizer.param_groups[0]["lr"],
                         p._student.optimizer.param_groups[0]["lr"])
        self.assertTrue(resumed._student.optimizer.state)

    def test_replay_keeps_full_command_while_critic_selects(self):
        self.options["obs_future_steps"] = (2,)
        self.env.observation_funcs["command"].funcs = {
            "ref_future": lambda: torch.zeros(4, 10)
        }
        p = self.new_policy()
        td = self.observation()
        td["command"] = torch.arange(10).float().expand(4, 10)
        td["action_dr"] = p.action_dr()
        latent = torch.randn(4, 8)
        obs = p._student_replay_obs(td, latent)
        torch.testing.assert_close(obs[:, :10], td["command"])
        torch.testing.assert_close(p.critic_obs(obs)[:, :5], td["command"][:, 5:])
        student = obs.index_select(-1, p._student_replay_index)
        torch.testing.assert_close(student, torch.cat([td["vel_command"], td["policy"], latent], -1))

    def test_rollout_reset_clears_both_hidden_states(self):
        p = self.policy
        first = self.observation()
        first["is_init"].fill_(True)
        second = first.clone()
        second["adapt_hx"].zero_()
        second["depth_hx"].zero_()
        with torch.no_grad():
            p._perceive(first, ema=True)
            p._perceive(second, ema=True)
        torch.testing.assert_close(first["priv_pred"], second["priv_pred"])
        torch.testing.assert_close(first["next", "adapt_hx"], second["next", "adapt_hx"])
        torch.testing.assert_close(first["next", "depth_hx"], second["next", "depth_hx"])


if __name__ == "__main__":
    unittest.main()
