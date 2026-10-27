"""Convert Holosoma R1 ``rl/<object>/<clip>.npz`` clips for VAIC replay.

The source archives are left intact. Each clip gets a sibling directory with
``motion.npz`` and ``meta.json``, as expected by MotionDataset.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "data/motion/r1/rl"
CONTACT_BODIES = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
)


def convert_clip(source: Path, object_root: Path) -> Path:
    raw = np.load(source, allow_pickle=False)
    joint_names = raw["joint_names"].tolist()
    body_names = raw["body_names"].tolist()
    fps = float(raw["fps"].item())
    frames = raw["body_pos_w"].shape[0]
    if raw["joint_pos"].shape != (frames, 7 + len(joint_names)):
        raise ValueError(f"Unexpected joint_pos shape in {source}")
    if raw["joint_vel"].shape != (frames, 6 + len(joint_names)):
        raise ValueError(f"Unexpected joint_vel shape in {source}")
    if body_names[0] != "world" or raw["body_quat_w"].shape != (frames, len(body_names), 4):
        raise ValueError(f"Unexpected body layout in {source}")

    source_contact_names = raw["contact_object_names"].tolist()
    source_contact = raw["contact_object_label"].astype(bool)
    source_contact_by_name = dict(zip(source_contact_names, source_contact.T))
    false = np.zeros(frames, dtype=bool)
    body_contact = np.zeros((frames, len(CONTACT_BODIES)), dtype=bool)
    for column, name in enumerate(CONTACT_BODIES):
        body_contact[:, column] = source_contact_by_name.get(name, false)
        if name.endswith("ankle_roll_link"):
            body_contact[:, column] |= source_contact_by_name.get(
                name.replace("ankle_roll_link", "ankle_constraint_A_link"), false
            )
        elif name.endswith("wrist_roll_link"):
            body_contact[:, column] |= source_contact_by_name.get(
                name.replace("wrist_roll_link", "hand_contact_link"), false
            )

    foot_contact = np.zeros((frames, 2), dtype=bool)
    for column, side in enumerate(("left", "right")):
        foot_names = [
            f"{side}_foot_{part}_{position}_link"
            for part in ("front", "rear") for position in ("outer", "inner")
        ]
        indices = [body_names.index(name) for name in foot_names]
        foot_contact[:, column] = raw["body_pos_w"][:, indices, 2].min(axis=1) < 0.015

    sample_points_path = object_root / source.parent.name / "sample_points.npy"
    if not sample_points_path.is_file():
        raise FileNotFoundError(sample_points_path)
    sample_points = np.load(sample_points_path)
    sample_indices = np.linspace(0, len(sample_points) - 1, 128, dtype=int)
    object_points = sample_points[sample_indices][None].astype(np.float32)

    # Holosoma stores the floating root in the first 7 position and 6 velocity
    # columns. Its body quaternions are already wxyz, as used by VAIC.
    motion = {
        key: raw[key][:, 1:].astype(np.float32)
        for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    }
    motion.update(
        joint_pos=raw["joint_pos"][:, 7:].astype(np.float32),
        joint_vel=raw["joint_vel"][:, 6:].astype(np.float32),
        object_contact=source_contact.any(axis=1, keepdims=True),
        body_contact=body_contact,
        feet_contact=foot_contact,
        object_points=object_points,
    )
    destination = source.with_suffix("")
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / "motion.npz", **motion)
    try:
        source_name = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_name = str(source)
    (destination / "meta.json").write_text(
        json.dumps({
            "body_names": body_names[1:],
            "joint_names": joint_names,
            "fps": fps,
            "source": source_name,
            "conversion": {"format": "holosoma_r1_rl_to_vaic", "quaternion_order": "wxyz"},
        }, indent=2) + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--object-root", type=Path,
        default=Path(os.environ.get("VAIC_HOI_ROOT", "/home/hcc/research/HOI")) / "train_r1/objects",
    )
    args = parser.parse_args()
    clips = sorted(args.source_root.glob("*/*.npz"))
    if not clips:
        parser.error(f"No R1 clips found under {args.source_root}")
    for source in clips:
        print(f"{source.relative_to(args.source_root)} -> {convert_clip(source, args.object_root)}")


if __name__ == "__main__":
    main()
