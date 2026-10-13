"""Algorithm registration and compatibility with saved pre-move configurations."""
import importlib
import pickle
import pkgutil
import subprocess
import sys
import unittest
from pathlib import Path

from hydra import compose, initialize
from hydra.utils import get_class

import active_adaptation.learning  # Register both PPO and FlashSAC algorithm groups.


NEW_PACKAGE = "active_adaptation.learning.flashsac"
OLD_PACKAGE = "active_adaptation.learning.ppo"
ALGORITHMS = (
    ("flashsac_vel_flat_train", "flashsac_vel_flat", "FlashSACVelFlat"),
    ("flashsac_vel_train", "flashsac_vel", "FlashSACVel"),
    ("flashsac_vel_finetune", "flashsac_vel_finetune", "FlashSACVelFinetune"),
    ("joint_flashsac_vel_finetune", "joint_flashsac_vel_finetune", "JointFlashSACVelFinetune"),
    ("cosil_flashsac_vel_finetune", "cosil_flashsac_vel_finetune", "CoSILFlashSACVelFinetune"),
    ("gaddpg_cosil_flashsac_vel_finetune", "gaddpg_cosil_flashsac_vel_finetune",
     "GADDPGCoSILFlashSACVelFinetune"),
    ("tgrl_flashsac_vel_finetune", "tgrl_flashsac_vel_finetune", "TGRLFlashSACVelFinetune"),
)


class FlashSACPackageLayoutTest(unittest.TestCase):
    def test_algorithm_names_compose_with_new_targets_and_old_targets_still_resolve(self):
        with initialize(version_base=None, config_path=None):
            for algorithm, module_name, class_name in ALGORITHMS:
                with self.subTest(algorithm=algorithm):
                    cfg = compose(overrides=[f"+algo={algorithm}"])
                    target = f"{NEW_PACKAGE}.{module_name}.{class_name}"
                    old_target = f"{OLD_PACKAGE}.{module_name}.{class_name}"
                    policy_class = getattr(importlib.import_module(f"{NEW_PACKAGE}.{module_name}"), class_name)
                    self.assertEqual(cfg.algo._target_, target)
                    self.assertIs(get_class(target), policy_class)
                    self.assertIs(get_class(old_target), policy_class)

    def test_old_module_aliases_share_classes_and_vendored_module_state(self):
        module_names = [entry[1] for entry in ALGORITHMS] + ["finetune_schedule", "flashsac_upstream"]
        upstream = importlib.import_module(f"{NEW_PACKAGE}.flashsac_upstream")
        module_names.extend(f"flashsac_upstream.{entry.name}"
                            for entry in pkgutil.iter_modules(upstream.__path__))
        for module_name in module_names:
            with self.subTest(module=module_name):
                current = importlib.import_module(f"{NEW_PACKAGE}.{module_name}")
                legacy = importlib.import_module(f"{OLD_PACKAGE}.{module_name}")
                self.assertIs(legacy, current)

    def test_configs_pickled_with_old_module_paths_load_as_current_classes(self):
        for _, module_name, class_name in ALGORITHMS:
            with self.subTest(module=module_name):
                new_module = f"{NEW_PACKAGE}.{module_name}"
                old_module = f"{OLD_PACKAGE}.{module_name}"
                config_class = getattr(importlib.import_module(new_module), f"{class_name}Config")
                cfg = config_class()
                cfg._target_ = f"{old_module}.{class_name}"
                payload = pickle.dumps(cfg, protocol=2)
                # Protocol 2 GLOBAL encodes module/class names as newline-delimited
                # text. Only replace that reference, leaving object state intact.
                current_global = f"c{new_module}\n{config_class.__name__}\n".encode()
                legacy_global = f"c{old_module}\n{config_class.__name__}\n".encode()
                self.assertIn(current_global, payload)
                restored = pickle.loads(payload.replace(current_global, legacy_global, 1))
                self.assertIs(type(restored), config_class)
                self.assertEqual(restored, cfg)
                self.assertIs(get_class(restored._target_), getattr(importlib.import_module(new_module), class_name))

    def test_legacy_import_and_nested_pickle_work_first_in_a_fresh_interpreter(self):
        # Test the startup path independently of this process's registration and
        # import order, as a checkpoint or archived Hydra target encounters it.
        code = """
import importlib
import pickle
old = importlib.import_module('active_adaptation.learning.ppo.flashsac_vel_finetune')
new = importlib.import_module('active_adaptation.learning.flashsac.flashsac_vel_finetune')
assert old is new
from active_adaptation.learning.ppo import flashsac_vel_finetune
assert flashsac_vel_finetune is new
actor = pickle.loads(b'cactive_adaptation.learning.ppo.flashsac_upstream.network\\nFlashSACActor\\n.')
module = importlib.import_module('active_adaptation.learning.flashsac.flashsac_upstream.network')
assert actor is module.FlashSACActor
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ppo_algorithm_targets_remain_in_ppo_package(self):
        targets = {
            "ppo": "ppo.PPOPolicy",
            "ppo_vel_train": "ppo_vel.PPOVEL",
            "ppo_vel_finetune": "ppo_vel.PPOVEL",
            "ppo_vel_flat_train": "ppo_vel_flat.PPOVelFlat",
            "critics": "critics.Critics",
        }
        with initialize(version_base=None, config_path=None):
            for algorithm, suffix in targets.items():
                with self.subTest(algorithm=algorithm):
                    cfg = compose(overrides=[f"+algo={algorithm}"])
                    target = f"{OLD_PACKAGE}.{suffix}"
                    self.assertEqual(cfg.algo._target_, target)
                    self.assertEqual(get_class(target).__module__, f"{OLD_PACKAGE}.{suffix.rsplit('.', 1)[0]}")


if __name__ == "__main__":
    unittest.main()
