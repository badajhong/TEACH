"""Depth export preserves metric values and obeys the capture interval."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from scripts.depth_png import DepthPngWriter


class DepthPngTest(unittest.TestCase):
    def make_camera(self):
        pixels = torch.tensor([
            [[0.1, 0.8, 1.5, float("nan")], [float("inf"), -1.0, 2.0, 0.0]],
            [[0.1, 0.8, 1.5, float("nan")], [float("inf"), -1.0, 2.0, 0.0]],
        ])
        pixels[0] = 1.5
        return SimpleNamespace(
            num_instances=2,
            cfg=SimpleNamespace(update_period=0.1, min_distance=0.1, max_distance=1.5),
            data=SimpleNamespace(output={"distance_to_image_plane": pixels.unsqueeze(-1)}),
        )

    def test_selected_env_png_values_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DepthPngWriter(self.make_camera(), directory, env_id=1)
            self.assertTrue(writer.update(0.005))
            folder = Path(directory)
            with Image.open(folder / "depth_env001_000000_mm.png") as im:
                np.testing.assert_array_equal(
                    np.asarray(im), [[100, 800, 1500, 0], [0, 0, 2000, 0]]
                )
            with Image.open(folder / "depth_env001_000000.png") as im:
                self.assertEqual(im.mode, "L")
                np.testing.assert_array_equal(
                    np.asarray(im), [[0, 128, 255, 0], [0, 0, 255, 0]]
                )
            metadata = json.loads((folder / "metadata.json").read_text())
            self.assertEqual(metadata["metric_png_unit"], "millimeter")
            self.assertEqual(metadata["image_shape_hw"], [2, 4])

    def test_capture_interval_and_frame_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DepthPngWriter(self.make_camera(), directory, max_frames=2)
            self.assertTrue(writer.update(0.005))
            for _ in range(19):
                self.assertFalse(writer.update(0.005))
            self.assertTrue(writer.update(0.005))
            self.assertFalse(writer.update(1.0))
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 4)


if __name__ == "__main__":
    unittest.main()
