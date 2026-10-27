"""Isaac check of shared train/play initialization and a short student update.

Run in the vaic environment with the same algorithm/task/checkpoint overrides
as train.py. This test uses a small temporary replay, not the training buffer.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from isaaclab.app import AppLauncher
import active_adaptation.learning


@hydra.main(config_path="../cfg", config_name="train", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.app.headless = True
    cfg.app.enable_cameras = False  # Ray casting does not need RTX.
    cfg.task.num_envs = cfg.algo.num_envs = 4
    cfg.algo.use_compile = False
    cfg.algo.train_every = 4
    cfg.algo.adapt_num_minibatches = 2
    cfg.algo.adapt_epochs = 1
    cfg.algo.buffer_max_length = 128  # Test-only replay, no production config change.
    cfg.algo.buffer_min_length = 16
    cfg.algo.sample_batch_size = 8
    cfg.algo.updates_per_interaction_step = 1
    cfg.algo.normalize_reward = False
    cfg.task.command.reset_range = [0, 1]
    for term in cfg.task.observation.depth.values():
        term.noise_std = 0.
        term.episode_noise_range = [0., 0.]
        term.dropout_prob = 0.
    app = AppLauncher(cfg.app).app
    from helpers import make_env_policy
    from isaaclab.utils.math import combine_frame_transforms

    env, policy, _ = make_env_policy(cfg)
    base = env.base_env
    assert base.real_reset_observations
    depth = base.observation_funcs["depth"].funcs["depth_camera"]
    camera = depth.camera
    action = base.action_manager
    command = base.command_manager
    head = base.robot.find_bodies(cfg.task.camera_sensor.parent_link)[0][0]

    def check_reference():
        expected = command.dataset.get_slice(command.motion_ids, command.t, steps=command.future_steps)
        torch.testing.assert_close(command.ref_joint_pos_future_,
                                   expected.joint_pos[..., command.tracking_joint_indices_motion])
        index = (command.motion_starts + command.t)[:, None] + command.future_steps[None]
        index.clamp_max_(command.motion_ends[:, None] - 1)
        torch.testing.assert_close(command.ref_body_contact_future, command._body_contact[index])

    with torch.no_grad():
        # Both full starts and random-phase cold starts must be real inputs.
        command.reset_range = None
        command.full_start_probability = 1.
        env.train()
        env.reset()
        assert not command.t.any()
        command.full_start_probability = 0.
        env.reset()
        assert command.t.max() > 0, "Random phase initialization was lost"
        check_reference()
        command.full_start_probability = cfg.task.command.full_start_probability
        command.reset_range = (0, 1)
        for training in (True, False):
            env.train(training)
            env.set_seed(0)
            td = env.reset()
            assert td["is_init"].all() and not td["step_count"].any()
            assert not td["adapt_hx"].any() and not td["depth_hx"].any()
            assert not command.t.any(), "Reset consumed a motion frame"
            check_reference()
            for key in ("policy", "vel_command", "ref_joint_pos_", "depth"):
                assert torch.isfinite(td[key]).all(), key
                assert td[key].abs().sum() > 0, key
            raw = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(depth.depth_img_buffer, raw[:, None].expand_as(depth.depth_img_buffer))
            normalized = ((raw.nan_to_num(nan=1.5, posinf=1.5, neginf=.1).clamp(.1, 1.5) - .1) / 1.4)
            normalized = (normalized * depth.num_bins).floor().clamp(0, depth.num_bins) / depth.num_bins
            torch.testing.assert_close(td["depth"].squeeze(1), normalized)
            pos, quat = combine_frame_transforms(base.robot.data.body_link_pos_w[:, head],
                base.robot.data.body_link_quat_w[:, head], camera._offset_pos, camera._offset_quat)
            torch.testing.assert_close(camera.data.pos_w, pos)
            torch.testing.assert_close(camera.data.quat_w_world, quat)
            held = action.default_joint_pos[:, action.joint_ids] + action.offset[:, action.joint_ids]
            held += action.applied_action * action.action_scaling
            torch.testing.assert_close(held, base.robot.data.joint_pos[:, action.joint_ids])

            rollout = policy.get_rollout_policy("train" if training else "eval")
            for step in range(4):
                transition, td = env.step_and_maybe_reset(rollout(td))
                if training:
                    policy.add_transition(transition)
                    with torch.enable_grad():
                        policy.update()
                assert (command.t == step + 1).all()
                check_reference()
            if training:
                assert policy.buffer.valid[:4].all(), "First actions were excluded from SAC"
                assert policy.joint_perception_iters_completed > 0
                for _ in range(4):
                    transition, td = env.step_and_maybe_reset(rollout(td))
                    policy.add_transition(transition)
                    with torch.enable_grad():
                        policy.update()
                assert policy._update_step > 0, "No student SAC update ran"

            # Reset some robots after camera/history have followed a moving pose.
            selected = torch.tensor([1, 3], device=env.device)
            keep = torch.tensor([0, 2], device=env.device)
            old = (depth.depth_img_buffer[keep].clone(), action.action_buf[keep].clone(),
                   command.t[keep].clone(), command.ref_joint_pos_future_[keep].clone(),
                   command.ref_body_contact_future[keep].clone())
            mask = torch.zeros(4, 1, device=env.device, dtype=torch.bool)
            mask[selected] = True
            reset = env.reset(TensorDict({"_reset": mask}, [4], device=env.device))
            current = (depth.depth_img_buffer[keep], action.action_buf[keep], command.t[keep],
                       command.ref_joint_pos_future_[keep], command.ref_body_contact_future[keep])
            for actual, expected in zip(current, old):
                torch.testing.assert_close(actual, expected)
            assert not command.t[selected].any()
            assert not reset["adapt_hx"][selected].any() and not reset["depth_hx"][selected].any()
            raw = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(depth.depth_img_buffer[selected],
                raw[selected, None].expand_as(depth.depth_img_buffer[selected]))
            check_reference()
            print(f"PASS: {'train' if training else 'play'} first observations, sensor pose, "
                  "reference/contact timing, depth/history and partial reset.", flush=True)
    print("PASS: initial actions stored for SAC; perception and student SAC updates ran.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
