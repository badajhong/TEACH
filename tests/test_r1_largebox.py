"""CPU checks for the R1 task and robot-dependent FlashSAC dimensions."""
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.flashsac.flashsac_vel import FlashSACVel, FlashSACVelConfig


ROOT = Path(__file__).resolve().parents[1]


class R1LargeboxTest(unittest.TestCase):
    def test_task_alias_and_motion_joint_mapping(self):
        with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
            canonical = compose("train", overrides=["algo=flashsac_vel_train", "task=R1/TEACH/largebox_tea"])
            alias = compose("train", overrides=["algo=flashsac_vel_train", "task=G1/TEACH/largebox_tea"])
        self.assertEqual(OmegaConf.to_container(canonical.task), OmegaConf.to_container(alias.task))
        task = canonical.task
        self.assertEqual(task.robot.name, "r1")
        with (ROOT / task.command.data_path / "meta.json").open() as f:
            meta = json.load(f)
        actions = []
        for pattern, scale in task.action.action_scaling.items():
            if scale is None:
                continue
            matches = [j for j in meta["joint_names"] if re.fullmatch(pattern, j)]
            self.assertTrue(matches, pattern)
            self.assertGreater(scale, 0)
            actions.extend(matches)
        self.assertEqual(len(actions), 24)
        self.assertEqual(len(set(actions)), 24)
        self.assertFalse(any(j.startswith("head_") for j in actions))
        self.assertEqual(list(task.action.fixed_joint_names), ["head_pitch_joint", "head_yaw_joint"])
        self.assertTrue(set(task.action.fixed_joint_names).isdisjoint(actions))
        self.assertNotIn("waist_pitch_joint", actions)
        self.assertIn(task.command.root_body_name, meta["body_names"])
        self.assertTrue(task.reward.feet.feet_contact.enabled)
        self.assertIsNone(task.reward.feet.feet_air_lift)
        self.assertIsNone(task.reward.feet.feet_air_time)
        self.assertNotIn("waist_pos_limits", task.reward.loco)
        self.assertEqual(task.reward.loco.joint_pos_limits.soft_factor, 0.95)
        self.assertEqual(task.robot.override_params.soft_joint_pos_limit_factor, 0.95)
        self.assertEqual(task.reward.debug.joint_pos_limits.soft_factor, 0.95)

    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def make_policy(self, actions, joints, reference_actions=None):
        n = 4
        # Exercise different motion-command and proprioception widths as well
        # as action widths; no network layer should assume G1's dimensions.
        dims = {"command": 5 * (42 + actions) + 1, "policy": 6 + 6 * joints + 3 * actions,
                "priv": 12 + 9 * joints, "object_": 22, "object_geo_": 384,
                "vel_command": 20, "ref_joint_pos_": reference_actions or actions}
        spec = Composite({k: Unbounded((n, d)) for k, d in dims.items()}, shape=(n,))
        env = SimpleNamespace(
            transform=SimpleNamespace(transforms=[]),
            action_manager=SimpleNamespace(delay=torch.zeros(n, 1), alpha=torch.ones(n, 1)),
            observation_funcs={},
        )
        for key, dim in dims.items():
            env.observation_funcs[key] = SimpleNamespace(
                funcs={key: lambda dim=dim: torch.zeros(n, dim)},
                _compute=lambda dim=dim: torch.zeros(n, dim),
            )
        cfg = FlashSACVelConfig(
            num_envs=n, num_env_steps=256, use_compile=False, use_amp=False, obs_future_steps=None,
            obs_drop_terms=(), latent_dim=8, actor_hidden_dim=16, critic_hidden_dim=16,
            critic_state_encoder_dim=8, critic_action_encoder_dim=8, critic_num_bins=11,
        )
        return FlashSACVel(cfg, spec, Unbounded((n, actions)), None, "cpu", env), spec

    def test_teacher_student_and_critic_follow_robot_dimensions(self):
        with torch._dynamo.config.patch(disable=True):
            for actions, joints in [(23, 29), (24, 26)]:
                with self.subTest(actions=actions, joints=joints):
                    policy, spec = self.make_policy(actions, joints)
                    td = spec.rand()
                    td = policy.get_rollout_policy("eval")(td)
                    replay = policy.replay_obs(td)
                    mean, _ = policy.actor.get_mean_and_std(replay, training=False)
                    self.assertEqual(mean.shape, (4, actions))
                    torch.testing.assert_close(
                        td["action"], td["ref_joint_pos_"] + policy.cfg.residual_scale * mean.tanh()
                    )
                    student_input = torch.cat([td["vel_command"], td["policy"], torch.zeros(4, 8)], -1)
                    student_mean, _ = policy.actor_adapt.get_mean_and_std(student_input, training=False)
                    self.assertEqual(student_mean.shape, (4, actions))
                    q, _ = policy.critic(policy.critic_obs(replay), mean.tanh(), training=False)
                    self.assertTrue(torch.isfinite(q).all())

    def test_rejects_reference_with_uncontrolled_head_joints(self):
        with self.assertRaisesRegex(ValueError, "26 values.*24 policy actions"):
            self.make_policy(actions=24, joints=26, reference_actions=26)


if __name__ == "__main__":
    unittest.main()
