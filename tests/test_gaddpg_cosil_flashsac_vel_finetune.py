"""CPU tests for gaddpg_cosil_flashsac_vel_finetune: teacher intervention and the "entropy" mode.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_gaddpg_cosil_flashsac_vel_finetune.py
The simulator/camera/compiled CUDA path is covered separately with scripts/train.py.
"""
import copy
import unittest
from types import SimpleNamespace

import torch
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.ppo.flashsac_vel import FlashSACVel, FlashSACVelConfig
from active_adaptation.learning.ppo.flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
)
from active_adaptation.learning.ppo.flashsac_vel_flat import U_KEY
from active_adaptation.learning.ppo.gaddpg_cosil_flashsac_vel_finetune import (
    INTERVENED_KEY, GADDPGCoSILFlashSACVelFinetune, GADDPGCoSILFlashSACVelFinetuneConfig,
)


class GADDPGFinetuneTest(unittest.TestCase):
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
            perception_warmup_iters=2, adapt_epochs=1,
        )
        teacher = FlashSACVel(FlashSACVelConfig(**{k: v for k, v in self.options.items()
                                                   if k in FlashSACVelConfig.__dataclass_fields__}),
                              self.spec, self.action_spec, None, "cpu", self.env)
        self.teacher_state = copy.deepcopy(teacher.state_dict())

    def tearDown(self):
        self.no_compile.__exit__(None, None, None)
        torch.set_num_threads(self.old_threads)

    def gaddpg(self, **kwargs):
        cfg = GADDPGCoSILFlashSACVelFinetuneConfig(**self.options, **kwargs)
        p = GADDPGCoSILFlashSACVelFinetune(cfg, self.spec, self.action_spec, None, "cpu", self.env)
        p.load_state_dict(copy.deepcopy(self.teacher_state))
        return p

    def baseline(self):
        cfg = FlashSACVelFinetuneConfig(**self.options)
        p = FlashSACVelFinetune(cfg, self.spec, self.action_spec, None, "cpu", self.env)
        p.load_state_dict(copy.deepcopy(self.teacher_state))
        return p

    def observation(self):
        td = self.spec.rand()
        td["is_init"] = torch.zeros(4, 1, dtype=torch.bool)
        td["adapt_hx"] = torch.randn(4, 8)
        td["depth_hx"] = torch.randn(4, 64)
        td["step_count"] = torch.ones(4, 1, dtype=torch.long) * 3
        return td

    def test_entropy_mode_update_matches_flashsac_vel_finetune(self):
        base, ours = self.baseline(), self.gaddpg(cosil_mode="entropy")
        for p in (base, ours):
            p._start_rl()
            p._allocate_student_replay()
        self.assertIsNone(ours._beta)
        self.assertEqual(base._rl_schedule_updates, ours._rl_schedule_updates)
        g = torch.Generator().manual_seed(3)
        dim = ours.student_replay_dim
        for step in range(12):
            row = dict(obs=torch.randn(4, dim, generator=g), next_obs=torch.randn(4, dim, generator=g),
                       action=torch.rand(4, 3, generator=g) * 2 - 1, reward=torch.randn(4, generator=g),
                       discount=torch.ones(4), done=torch.zeros(4, dtype=torch.bool),
                       cut=torch.zeros(4, dtype=torch.bool), valid=torch.ones(4, dtype=torch.bool))
            base.buffer.add(**row)
            ours.buffer.add(**row, intervened=torch.zeros(4, dtype=torch.bool))
        for _ in range(3):  # actor_update_period covers updates with and without the actor
            torch.manual_seed(11)
            info_base = base._update_once()
            torch.manual_seed(11)
            info_ours = ours._update_once()
            self.assertEqual(sorted(info_base), sorted(info_ours))
            for key in info_base:
                torch.testing.assert_close(info_ours[key], info_base[key], rtol=0, atol=0)
        for name in ("actor_adapt", "critic", "target_critic", "temperature"):
            a, b = getattr(base, name).state_dict(), getattr(ours, name).state_dict()
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), f"{name}.{key}")

    def test_entropy_mode_checkpoint_has_no_beta_and_resumes(self):
        p = self.gaddpg(cosil_mode="entropy")
        p._start_rl()
        state = copy.deepcopy(p.state_dict())
        self.assertNotIn("cosil_beta", state)
        resumed = self.gaddpg(cosil_mode="entropy")
        resumed.load_state_dict(state)
        self.assertEqual(resumed.stage, 2)
        self.assertIsNone(resumed._beta)
        # a CoSIL-mode run can resume it and starts beta fresh
        cosil = self.gaddpg(cosil_mode="cosil_entropy")
        cosil.load_state_dict(state)
        self.assertIsNotNone(cosil._beta)

    def test_intervention_rollout_mask_and_tau(self):
        p = self.gaddpg(intervention_tau_init=1e-3)  # always above tau: every mixed env is taken over
        self.assertIsNone(p.stats_env_mask)
        p._start_rl()
        p.needs_env_reset = False
        mixed = torch.tensor([False, True, False, True])
        self.assertTrue(torch.equal(p._mixed_envs, mixed))
        self.assertTrue(torch.equal(p.stats_env_mask, ~mixed))
        self.assertTrue(torch.equal(self.env.stats_env_mask, ~mixed))
        td = p.get_rollout_policy("train")(self.observation())
        teacher_mean, _ = p.actor.get_mean_and_std(p.replay_obs(td), training=False)
        self.assertTrue(torch.equal(td[INTERVENED_KEY], mixed))
        torch.testing.assert_close(td[U_KEY][mixed], torch.tanh(teacher_mean)[mixed])
        self.assertFalse(torch.allclose(td[U_KEY][~mixed], torch.tanh(teacher_mean)[~mixed]))
        self.assertTrue(torch.equal(p._hold_left, mixed.long() * (p.cfg.intervention_hold_steps - 1)))
        # takeover rate 1.0 > target 0.6, so tau rises; a new episode clears the hold
        tau = p.intervention_tau
        p._update_intervention()
        self.assertGreater(p.intervention_tau, tau)
        self.assertEqual(p._intervention_info["gaddpg/takeover_rate"], 1.0)
        p.intervention_tau = 1e3
        reset = self.observation()
        reset["is_init"][1] = True
        td = p.get_rollout_policy("train")(reset)
        self.assertTrue(torch.equal(td[INTERVENED_KEY], torch.tensor([False, False, False, True])))
        # warmup clears the logging mask; evaluation rollouts never intervene
        p._start_warmup()
        self.assertIsNone(p.stats_env_mask)
        self.assertIsNone(self.env.stats_env_mask)


if __name__ == "__main__":
    unittest.main()
