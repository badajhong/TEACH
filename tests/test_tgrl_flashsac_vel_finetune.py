"""CPU regressions for TGRL targets, actor inputs and compiled-output ownership.

Run: PYTHONPATH=. python -m unittest discover -s tests -p test_tgrl_flashsac_vel_finetune.py
The inherited tests also exercise perception warmup, replay and checkpoint transfer.
"""
import copy
import unittest
from unittest import mock

import torch

import test_flashsac_vel_finetune as finetune_fixture
from active_adaptation.learning.ppo import tgrl_flashsac_vel_finetune as tgrl


class TGRLFinetuneTest(finetune_fixture.FinetuneTest):
    def new_policy(self):
        cfg = tgrl.TGRLFlashSACVelFinetuneConfig(
            **self.options, perception_warmup_iters=2, adapt_epochs=1)
        return tgrl.TGRLFlashSACVelFinetune(
            cfg, self.spec, self.action_spec, None, "cpu", self.env)

    def rl_batch(self):
        p = self.policy
        p._start_rl()
        batch = {}
        for key, actor_key, teacher_key in (
                ("observation", "actor_observation", "teacher_observation"),
                ("next_observation", "actor_next_observation", "teacher_next_observation"),
                ("one_step_next_observation", "actor_one_step_next_observation",
                 "teacher_one_step_next_observation")):
            td = self.observation()
            td["action_dr"] = p.action_dr()
            obs = p._student_replay_obs(td, torch.randn(4, p.cfg.latent_dim))
            batch[key] = p.critic_obs(obs)
            batch[actor_key] = obs.index_select(-1, p._student_replay_index)
            batch[teacher_key] = obs[:, :p.replay_obs_dim]
        batch.update(
            action=torch.rand(4, p.action_dim) * 2 - 1,
            reward=torch.tensor([.1, .2, -.3, .4]),
            # Row 0 terminates after the first step; row 1 terminates immediately.
            terminated=torch.tensor([1., 1., 0., 0.]),
            one_step_terminated=torch.tensor([0., 1., 0., 0.]),
        )
        return batch

    def test_default_kl_is_student_to_teacher_with_finite_gradients(self):
        p = self.policy
        self.assertEqual(p.cfg.tgrl_cost, "kl")
        mean = torch.tensor([[.2, -.3, .6], [-.1, .4, -.2]], requires_grad=True)
        std = torch.tensor([[.7, 1.1, .9], [1.2, .8, .6]], requires_grad=True)
        teacher_mean = torch.tensor([[-.1, .1, .3], [.2, -.2, .1]])
        teacher_std = torch.tensor([[1.1, .8, 1.2], [.7, 1.1, .9]])
        student = torch.distributions.Normal(mean, std)
        teacher = torch.distributions.Normal(teacher_mean, teacher_std)
        expected = torch.distributions.kl_divergence(student, teacher).sum(-1)
        reverse = torch.distributions.kl_divergence(teacher, student).sum(-1)
        cost = p._imitation_cost(mean, std, teacher_mean, teacher_std)
        torch.testing.assert_close(cost, expected * p._cost_scale)
        self.assertFalse(torch.allclose(cost, reverse * p._cost_scale))
        cost.sum().backward()
        for gradient in (mean.grad, std.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        torch.testing.assert_close(
            p._imitation_cost(teacher_mean, teacher_std, teacher_mean, teacher_std),
            torch.zeros(2))
        clipped = p._imitation_cost(mean.detach() + 100, std.detach(), teacher_mean, teacher_std)
        torch.testing.assert_close(clipped, torch.full_like(clipped, p.cfg.tgrl_cost_clip * p._cost_scale))

    def test_split_rollout_preserves_main_outputs_when_aux_reuses_storage(self):
        p = self.policy
        p._start_rl()
        shared_mean = torch.empty_like(p._cached_noise)
        shared_std = torch.empty_like(shared_mean)

        def main_forward(*args, **kwargs):
            return shared_mean.fill_(.25), shared_std.fill_(.5)

        def aux_forward(*args, **kwargs):
            return shared_mean.fill_(-.4), shared_std.fill_(1.5)

        # Keep the existing repeated noise so the expected action is deterministic.
        p._cached_noise.fill_(.3)
        p._cur_noise_repeat_count.fill_(1)
        p._cur_noise_repeat_n.fill_(8)
        with mock.patch.object(p._student, "apply", side_effect=main_forward), \
                mock.patch.object(p._aux, "apply", side_effect=aux_forward):
            actions = p._sample_split_rollout(torch.zeros(4, p.obs_dim))

        expected_mean = torch.where(p._aux_envs.unsqueeze(-1), -.4, .25)
        expected_std = torch.where(p._aux_envs.unsqueeze(-1), 1.5, .5)
        torch.testing.assert_close(
            actions, torch.tanh(expected_mean + expected_std * p._cached_noise))
        self.assertFalse(torch.equal(actions[0], actions[-1]))

    def test_sample_owns_distribution_storage_and_keeps_gradients(self):
        source_mean = torch.full((4, 3), .2, requires_grad=True)
        source_std = torch.full((4, 3), .7, requires_grad=True)
        network = mock.Mock()
        network.apply.return_value = source_mean, source_std
        actions, log_probs, mean, std = self.policy._sample(
            network, torch.zeros(4, 6), training=True)

        # Simulate the next compiled invocation overwriting its output buffers.
        with torch.no_grad():
            source_mean.fill_(-10)
            source_std.fill_(8)
        torch.testing.assert_close(mean, torch.full_like(mean, .2))
        torch.testing.assert_close(std, torch.full_like(std, .7))
        loss = mean.square().mean() + std.square().mean() + actions.mean() + log_probs.mean()
        loss.backward()
        for gradient in (source_mean.grad, source_std.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_both_actor_backwards_finish_before_compiled_normalization(self):
        p = self.policy
        batch = self.rl_batch()
        completed = set()
        original_step = p._optimizer_step
        main_normalize = p._student.normalize_parameters
        aux_normalize = p._aux.normalize_parameters

        def record_step(network, loss):
            original_step(network, loss)
            if network is p._student or network is p._aux:
                completed.add(network)

        def checked_normalize(normalize):
            self.assertEqual(completed, {p._student, p._aux})
            normalize()

        with mock.patch.object(p, "_optimizer_step", side_effect=record_step), \
                mock.patch.object(p._student, "normalize_parameters", side_effect=lambda:
                                  checked_normalize(main_normalize)) as main_call, \
                mock.patch.object(p._aux, "normalize_parameters", side_effect=lambda:
                                  checked_normalize(aux_normalize)) as aux_call:
            p._tgrl_update_networks(batch, do_actor_update=True)
        self.assertEqual(main_call.call_count, 1)
        self.assertEqual(aux_call.call_count, 1)

    def test_imitation_uses_one_step_target_while_reward_keeps_three_steps(self):
        p = self.policy
        batch = self.rl_batch()
        self.assertEqual(p.flash_cfg.n_step, 3)
        samples = []
        original_sample = p._sample

        def record_sample(network, observations, training):
            result = original_sample(network, observations, training)
            samples.append((observations.clone(), training,
                            tuple(value.detach().clone() for value in result)))
            return result

        with mock.patch.object(p, "_sample", side_effect=record_sample), \
                mock.patch.object(p.actor, "get_mean_and_std",
                                  wraps=p.actor.get_mean_and_std) as teacher_forward, \
                mock.patch.object(tgrl, "_compute_categorical_td_target",
                                  wraps=tgrl._compute_categorical_td_target) as target, \
                mock.patch.object(p._target_critic.network, "forward",
                                  wraps=p._target_critic.network.forward) as reward_forward, \
                mock.patch.object(p._imitation_target_critic.network, "forward",
                                  wraps=p._imitation_target_critic.network.forward) as imitation_forward:
            info = p._tgrl_update_networks(batch, do_actor_update=False)

        self.assertEqual(target.call_count, 2)
        self.assertEqual(teacher_forward.call_count, 1)
        self.assertFalse(teacher_forward.call_args.kwargs["training"])
        torch.testing.assert_close(
            teacher_forward.call_args.args[0],
            torch.cat([batch["teacher_observation"], batch["teacher_one_step_next_observation"]]))
        reward_target, imitation_target = [call.kwargs for call in target.call_args_list]
        self.assertEqual(reward_target["gamma"], p.flash_cfg.gamma ** 3)
        torch.testing.assert_close(reward_target["reward"], batch["reward"])
        torch.testing.assert_close(reward_target["done"], batch["terminated"])
        self.assertEqual(imitation_target["gamma"], p.flash_cfg.gamma)
        torch.testing.assert_close(imitation_target["reward"], torch.zeros_like(batch["reward"]))
        torch.testing.assert_close(imitation_target["done"], batch["one_step_terminated"])

        for forward, next_key, actor_next_key in (
                (reward_forward, "next_observation", "actor_next_observation"),
                (imitation_forward, "one_step_next_observation", "actor_one_step_next_observation")):
            self.assertEqual(forward.call_count, 1)
            arguments = forward.call_args.kwargs
            self.assertTrue(arguments["training"])
            torch.testing.assert_close(
                arguments["observations"], torch.cat([batch["observation"], batch[next_key]]))
            matches = [result for observations, training, result in samples
                       if not training and torch.equal(observations, batch[actor_next_key])]
            self.assertEqual(len(matches), 1)
            torch.testing.assert_close(
                arguments["actions"], torch.cat([batch["action"], matches[0][0]]))
            if actor_next_key == "actor_one_step_next_observation":
                one_step_mean, one_step_std = matches[0][2:]

        with torch.no_grad():
            teacher_mean, teacher_std = p.actor.get_mean_and_std(
                batch["teacher_one_step_next_observation"], training=False)
            expected_cost = p._imitation_cost(
                one_step_mean, one_step_std, teacher_mean.float(), teacher_std.float())
        torch.testing.assert_close(imitation_target["actor_entropy"], expected_cost)
        self.assertTrue(all(torch.isfinite(value).all() for value in info.values()))
        self.assertTrue(p._critic.optimizer.state)
        self.assertTrue(p._imitation_critic.optimizer.state)

    def test_both_actor_updates_use_flashsac_combined_observations_and_resume(self):
        p = self.policy
        batch = self.rl_batch()
        with mock.patch.object(p._student.network, "get_mean_and_std",
                               wraps=p._student.network.get_mean_and_std) as main_forward, \
                mock.patch.object(p._aux.network, "get_mean_and_std",
                                  wraps=p._aux.network.get_mean_and_std) as aux_forward:
            info = p._tgrl_update_networks(batch, do_actor_update=True)

        expected = torch.cat([batch["actor_observation"], batch["actor_next_observation"]])
        for forward in (main_forward, aux_forward):
            training_calls = [call for call in forward.call_args_list if call.kwargs.get("training")]
            self.assertEqual(len(training_calls), 1)
            torch.testing.assert_close(training_calls[0].args[0], expected)
        self.assertTrue(all(torch.isfinite(value).all() for value in info.values()))
        for network in (p._student, p._aux):
            self.assertTrue(network.optimizer.state)
            gradients = [parameter.grad for parameter in network.network.parameters()
                         if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

        saved = copy.deepcopy(p.state_dict())
        resumed = self.new_policy()
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.cfg.tgrl_cost, "kl")
        self.assertEqual(resumed.teacher_coef, p.teacher_coef)
        self.assertTrue(resumed._aux.optimizer.state)
        self.assertTrue(resumed._imitation_critic.optimizer.state)
        for network, key in ((resumed._aux.network, "tgrl_aux_actor"),
                             (resumed.imitation_critic, "tgrl_imitation_critic"),
                             (resumed.imitation_target_critic, "tgrl_imitation_target_critic")):
            for name, value in network.state_dict().items():
                torch.testing.assert_close(value, saved[key][name], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
