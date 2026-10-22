"""Isaac Sim check: python tests/sim_r1_largebox_randomization.py (vaic env)."""
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import active_adaptation  # Register Hydra algorithms before composition.
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher


def main():
    app = AppLauncher(headless=True).app
    from active_adaptation.envs import SimpleEnv
    import omni.usd

    with initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = compose("train", overrides=[
            "algo=flashsac_vel_train", "task=R1/TEACH/largebox_tea", "task.num_envs=8",
        ])
    OmegaConf.set_struct(cfg, False)
    # Match make_env_policy: disabled camera observations are removed before
    # the task builds its observation manager.
    for key in list(cfg.task.observation):
        if key not in cfg.algo.in_keys and not key.endswith("_"):
            cfg.task.observation.pop(key)

    with tempfile.TemporaryDirectory(prefix="vaic-box-randomization-") as tmp:
        os.chdir(tmp)
        env = SimpleEnv(cfg.task)
        box = env.scene.rigid_objects["largebox_link"]
        scale = box.cfg.spawn.scale.cpu()
        masses = box.root_physx_view.get_masses().cpu()

        assert scale.shape == (8, 3)
        assert torch.all((0.5 <= scale) & (scale <= 1.2))
        assert torch.unique(scale).numel() > 3
        assert torch.all((1.0 <= masses[:, 0]) & (masses[:, 0] <= 1.8))
        assert masses[:, 0].std() > 0

        stage = omni.usd.get_context().get_stage()
        for i in (0, 1, 7):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/largebox_link")
            assert prim.IsValid()
            actual_scale = torch.tensor(prim.GetAttribute("xformOp:scale").Get())
            torch.testing.assert_close(actual_scale, scale[i])

        base_contact_offsets = torch.tensor(
            cfg.task.command.contact_target_pos_offset, device=env.device
        )
        expected_offsets = base_contact_offsets.unsqueeze(0) * scale.to(env.device).unsqueeze(1)
        torch.testing.assert_close(
            env.command_manager.contact_target_pos_offset, expected_offsets
        )
        env.reset()
        print(
            f"PASS: mass {masses[:, 0].min():.3f}-{masses[:, 0].max():.3f} kg; "
            f"scale {scale.min():.3f}-{scale.max():.3f}; source and clones match offsets.",
            flush=True,
        )
        env.close()
        os.chdir(ROOT)

    # Isaac's Kit shutdown can hang after the checks have completed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
