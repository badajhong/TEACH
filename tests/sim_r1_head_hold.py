"""Isaac Sim integration check: python tests/sim_r1_head_hold.py (vaic env)."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import active_adaptation  # Register Hydra algorithms before composition.
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher


def main():
    app = AppLauncher(headless=True).app
    from helpers import make_env_policy
    from active_adaptation.envs.mdp.action import JointPosition

    with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = compose("train", overrides=[
            "algo=flashsac_vel_train", "task=R1/TEACH/largebox_tea",
            "task.num_envs=4", "algo.num_envs=4", "algo.use_compile=false",
            "total_frames=256", "algo.adapt_num_minibatches=2",
        ])
    OmegaConf.resolve(cfg)
    with tempfile.TemporaryDirectory(prefix="vaic-head-hold-") as tmp:
        os.chdir(tmp)
        env, policy, _ = make_env_policy(cfg)
        base = env.base_env
        robot = base.scene["robot"]
        action = base.action_manager
        head_ids = action.fixed_joint_ids
        defaults = robot.data.default_joint_pos[:, head_ids]
        assert robot.num_joints == 26
        assert action.action_dim == policy.action_dim == 24
        assert set(action.fixed_joint_names) == {"head_pitch_joint", "head_yaw_joint"}
        assert not set(action.fixed_joint_names) & set(action.joint_names)

        # A partial reset restores only the selected environments and ignores
        # non-default head poses/velocities, as loaded by motion initialization.
        robot.write_joint_state_to_sim(
            defaults + 0.2, torch.ones_like(defaults), joint_ids=head_ids,
        )
        selected = torch.tensor([1, 3], device=env.device)
        action.reset(selected)
        torch.testing.assert_close(robot.data.joint_pos[selected][:, head_ids], defaults[selected])
        torch.testing.assert_close(robot.data.joint_vel[selected][:, head_ids], torch.zeros_like(defaults[selected]))
        torch.testing.assert_close(robot.data.joint_pos[0, head_ids], defaults[0] + 0.2)

        try:
            JointPosition(base, action_scaling={"head_pitch_joint": 1.0},
                          fixed_joint_names=["head_pitch_joint"])
        except ValueError as error:
            assert "excluded from policy actions" in str(error)
        else:
            raise AssertionError("Fixed joints were allowed in the actor output")

        # Default targets must win even if another component writes an offset.
        action.offset[:, head_ids] = 0.25
        action(torch.randn(4, 24, device=env.device), 0)
        torch.testing.assert_close(robot.data.joint_pos_target[:, head_ids], defaults)

        max_head_deviation = 0.0
        for training in (True, False):
            env.train(training)
            # Frame 250 is in the real motion, rather than the neutral lead-in.
            base.command_manager.reset_range = (250, 251)
            carry = env.reset()
            torch.testing.assert_close(robot.data.joint_pos[:, head_ids], defaults)
            torch.testing.assert_close(robot.data.joint_vel[:, head_ids], torch.zeros_like(defaults))
            torch.testing.assert_close(action.offset[:, head_ids], torch.zeros_like(defaults))
            rollout = policy.get_rollout_policy("train" if training else "eval")
            with torch.no_grad():
                for _ in range(32):
                    carry = rollout(carry)
                    assert carry["action"].shape == (4, 24)
                    _, carry = env.step_and_maybe_reset(carry)
                    torch.testing.assert_close(robot.data.joint_pos_target[:, head_ids], defaults)
                    max_head_deviation = max(
                        max_head_deviation, (robot.data.joint_pos[:, head_ids] - defaults).abs().max().item()
                    )
        print(f"PASS: 26 simulated joints, 24 actor outputs; default head resets/targets "
              f"in train and eval. Maximum physical head deviation: {max_head_deviation:.6f} rad.", flush=True)
        env.close()
        os.chdir(ROOT)
    # Use the same process-exit cleanup as scripts/train.py.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
