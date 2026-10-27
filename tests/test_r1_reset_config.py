"""Train/play must select the same reset contract without altering R1 optics."""
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
import active_adaptation.learning


class R1ResetConfigTest(unittest.TestCase):
    def test_train_play_and_legacy_tasks(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / "cfg"), version_base=None):
            configs = [compose(name, overrides=["algo=joint_flashsac_vel_finetune",
                        "task=R1/rl/sub3_largebox_003_stu"]) for name in ("train", "play")]
            teacher = compose("train", overrides=["algo=flashsac_vel_train", "task=R1/rl/sub3_largebox_003_tea"])
            g1 = compose("train", overrides=["algo=joint_flashsac_vel_finetune", "task=G1/vaic/skateboard_general_tracking_stu"])
        for cfg in configs:
            self.assertEqual(cfg.task.reset_observation_mode, "current")
            self.assertEqual(cfg.task.command.full_start_probability, .2)
            self.assertEqual((cfg.task.camera_width, cfg.task.camera_height), (68, 56))
            sensor = cfg.task.camera_sensor
            self.assertEqual((sensor.horizontal_fov_deg, sensor.vertical_fov_deg), (146., 124.))
            self.assertEqual(sensor.update_period, .1)
            self.assertEqual((sensor.min_distance, sensor.max_distance), (.1, 1.5))
            self.assertEqual(cfg.algo.buffer_max_length, 10_000_000)
        self.assertEqual(teacher.task.reset_observation_mode, "zero")
        self.assertEqual(g1.task.get("reset_observation_mode", "zero"), "zero")


if __name__ == "__main__":
    unittest.main()
