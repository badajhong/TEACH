"""Save ray-camera depth as a preview and a millimeter-valued PNG."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class DepthPngWriter:
    def __init__(self, camera, directory, env_id=0, interval=None, max_frames=0):
        if not 0 <= env_id < camera.num_instances:
            raise ValueError(f"depth_png.env_id must be between 0 and {camera.num_instances - 1}.")
        if max_frames < 0:
            raise ValueError("depth_png.max_frames must be nonnegative; 0 means unlimited.")
        self.interval = camera.cfg.update_period if interval is None else interval
        if self.interval <= 0:
            raise ValueError("Set a positive depth_png.interval when the camera has no update period.")
        self.camera = camera
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.env_id = env_id
        self.max_frames = max_frames
        self.frames_written = 0
        self.elapsed = 0.0
        print(f"[INFO] Depth PNG output: {self.directory}")

    def update(self, dt):
        """Call after physics/sensor updates; elapsed time survives replay resets."""
        if self.max_frames and self.frames_written >= self.max_frames:
            return False
        self.elapsed += dt
        if self.frames_written and self.elapsed + 1e-9 < self.interval:
            return False

        # Read the sensor directly: teacher policies may discard the depth observation.
        output = self.camera.data.output.get("distance_to_image_plane")
        if output is None:
            raise ValueError("Depth PNG export requires distance_to_image_plane camera output.")
        depth = output[self.env_id, :, :, 0].detach().float().cpu().numpy()
        valid = np.isfinite(depth) & (depth > 0)
        clean = np.where(valid, depth, 0.0)
        depth_mm = np.rint(np.clip(clean * 1000, 0, 65535)).astype(np.uint16)
        near, far = self.camera.cfg.min_distance, self.camera.cfg.max_distance
        preview = np.rint(np.clip((clean - near) / (far - near), 0, 1) * 255).astype(np.uint8)
        preview[~valid] = 0

        stem = f"depth_env{self.env_id:03d}_{self.frames_written:06d}"
        Image.fromarray(preview).save(self.directory / f"{stem}.png")
        Image.fromarray(depth_mm).save(self.directory / f"{stem}_mm.png")
        if self.frames_written == 0:
            metadata = {
                "source": "distance_to_image_plane",
                "env_id": self.env_id,
                "image_shape_hw": list(depth.shape),
                "metric_png_suffix": "_mm.png",
                "metric_png_unit": "millimeter",
                "invalid_metric_value": 0,
                "preview_depth_range_m": [near, far],
                "preview_mapping": "near=0, far=255; invalid=0",
                "capture_interval_seconds": self.interval,
            }
            (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"[INFO] Saved depth preview and millimeter PNG: {stem}")
        self.frames_written += 1
        self.elapsed = 0.0
        return True
