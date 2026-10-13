"""Real-model CPU checks for joint perception/SAC updates and checkpoint restore."""
import copy
import unittest

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

import test_flashsac_vel_finetune as fixtures
from active_adaptation.learning.flashsac.flashsac_vel_finetune import (
    FlashSACVelFinetune, FlashSACVelFinetuneConfig,
)
from active_adaptation.learning.flashsac.joint_flashsac_vel_finetune import (
    JointFlashSACVelFinetune, JointFlashSACVelFinetuneConfig,
)


class JointFinetuneTest(unittest.TestCase):
    setUp = fixtures.FinetuneTest.setUp
    tearDown = fixtures.FinetuneTest.tearDown
    observation = fixtures.FinetuneTest.observation
    perception_iteration = fixtures.FinetuneTest.perception_iteration
    assert_model_equal = fixtures.FinetuneTest.assert_model_equal

    def new_policy(self, **overrides):
        cfg = JointFlashSACVelFinetuneConfig(**{
            **self.options, "perception_iters": 0, "adapt_epochs": 1, **overrides,
        })
        return JointFlashSACVelFinetune(cfg, self.spec, self.action_spec, None, "cpu", self.env)

    def joint_step(self):
        p = self.policy
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
        p.update()

    def joint_iteration(self):
        for _ in range(self.policy.cfg.train_every):
            self.joint_step()

    def test_starts_joint_and_perception_trains_while_replay_fills(self):
        p = self.policy
        self.assertEqual(p.stage, 2)
        self.assertEqual(p.perception_iters_completed, 0)
        actor = copy.deepcopy(p.actor_adapt.state_dict())
        online = [v.detach().clone() for v in p._perception_parameters()]
        ema = [v.detach().clone() for _, m in p._perception_pairs() for v in m.parameters()]
        self.joint_iteration()
        self.assertEqual(p._update_step, 0)
        self.assertEqual(p.joint_perception_iters_completed, 1)
        self.assert_model_equal(p.actor_adapt, actor)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(online, p._perception_parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            ema, (v for _, m in p._perception_pairs() for v in m.parameters()))))
        self.assertFalse(p.opt_adapt_actor.state)
        self.assertTrue(all(v.requires_grad for v in p._perception_parameters()))
        self.assertTrue(all(not v.requires_grad for _, m in p._perception_pairs() for v in m.parameters()))

    def test_joint_updates_both_models_without_resetting_replay_or_teacher(self):
        p = self.policy
        actor = copy.deepcopy(p.actor_adapt.state_dict())
        critic = copy.deepcopy(p.critic.state_dict())
        teacher = copy.deepcopy(p.actor.state_dict())
        self.joint_iteration()
        replay = p.buffer
        old_rows = replay.obs[:3].clone()
        online = [v.detach().clone() for v in p._perception_parameters()]
        for _ in range(3):
            self.joint_iteration()
        self.assertIs(p.buffer, replay)
        self.assertEqual(p.buffer.step, 16)
        torch.testing.assert_close(p.buffer.obs[:3], old_rows)
        self.assertFalse(p.needs_env_reset)
        self.assertEqual(p.stage, 2)
        self.assertEqual(p.finetune_iters_completed, 4)
        self.assertEqual(p.joint_perception_iters_completed, 4)
        self.assertGreater(p._update_step, 0)
        self.assertTrue(any(not torch.equal(v, actor[k]) for k, v in p.actor_adapt.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, critic[k]) for k, v in p.critic.state_dict().items()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(online, p._perception_parameters())))
        self.assert_model_equal(p.actor, teacher)
        self.assertTrue(all(v.grad is None for v in p.actor.parameters()))
        info = p.pop_info()
        self.assertIn("adapt/priv_loss", info)
        self.assertIn("adapt/object_loss", info)
        self.assertEqual(info["finetune/joint_active"], 1)
        self.assertNotIn("finetune/cycles_completed", info)

    def test_joint_checkpoint_restores_perception_and_sac_optimizers(self):
        for _ in range(3):
            self.joint_iteration()
        p = self.policy
        saved = copy.deepcopy(p.state_dict())
        self.assertEqual(saved["finetune_method"], "joint_v1")
        self.assertNotIn("cycle_mode", saved)
        self.policy = self.new_policy()
        self.policy.load_state_dict(saved)
        resumed = self.policy
        self.assertEqual(resumed.stage, 2)
        self.assertEqual(resumed._update_step, p._update_step)
        self.assertEqual(resumed.joint_perception_iters_completed, 3)
        self.assertIsNone(resumed.buffer)
        self.assertTrue(resumed.opt_adapt.state)
        self.assertTrue(resumed._student.optimizer.state)
        for pair, expected_pair in zip(resumed._perception_pairs(), p._perception_pairs()):
            for model, expected in zip(pair, expected_pair):
                self.assert_model_equal(model, expected.state_dict())
        self.joint_iteration()
        self.assertEqual(resumed.joint_perception_iters_completed, 4)
        self.assertTrue(all(v.requires_grad for v in resumed._perception_parameters()))
        resumed.configure_training_budget(1)
        self.assertFalse(resumed.training_complete())
        self.joint_iteration()
        self.assertTrue(resumed.training_complete())

    def test_optional_warmup_and_warmup_resume_then_joint(self):
        self.policy = self.new_policy(perception_iters=2)
        self.policy.load_state_dict(self.teacher_state)
        self.perception_iteration()
        saved = copy.deepcopy(self.policy.state_dict())
        self.policy = self.new_policy(perception_iters=2)
        self.policy.load_state_dict(saved)
        self.assertEqual(self.policy.stage, 1)
        self.assertEqual(self.policy.perception_iters_completed, 1)
        self.perception_iteration()
        self.assertEqual(self.policy.stage, 2)
        self.assertTrue(all(v.requires_grad for v in self.policy._perception_parameters()))
        self.joint_iteration()
        self.assertEqual(self.policy.joint_perception_iters_completed, 1)

    def test_checkpoint_methods_cannot_be_silently_interchanged(self):
        staged = FlashSACVelFinetune(
            FlashSACVelFinetuneConfig(**self.options, cycle_mode=False),
            self.spec, self.action_spec, None, "cpu", self.env)
        with self.assertRaisesRegex(ValueError, "teacher or joint"):
            self.policy.load_state_dict(staged.state_dict())
        with self.assertRaisesRegex(ValueError, "Unsupported finetune method"):
            staged.load_state_dict(self.policy.state_dict())

    def test_hydra_config_has_no_cycle_controls(self):
        with initialize(version_base=None, config_path=None):
            cfg = compose(overrides=["+algo=joint_flashsac_vel_finetune"])
        self.assertEqual(cfg.algo.perception_iters, 0)
        self.assertEqual(cfg.algo.name, "joint_flashsac_vel_finetune")
        for key in ("cycle_mode", "num_cycles", "sac_iters", "perception_trigger",
                    "plateau_window", "plateau_tol", "final_sac_iters"):
            self.assertNotIn(key, cfg.algo)
        # scripts/train.py passes a DictConfig, rather than the dataclass used
        # by the other small-model tests.
        runtime_cfg = OmegaConf.merge(cfg.algo, self.options, {"adapt_epochs": 1})
        self.policy = JointFlashSACVelFinetune(
            runtime_cfg, self.spec, self.action_spec, None, "cpu", self.env)
        self.policy.load_state_dict(self.teacher_state)
        self.joint_iteration()
        self.assertEqual(self.policy.joint_perception_iters_completed, 1)


if __name__ == "__main__":
    unittest.main()
