"""Register FlashSAC algorithms and preserve imports used by older checkpoints."""

import importlib as _importlib
import sys as _sys


__all__ = [
    "flashsac_vel_flat",
    "flashsac_vel",
    "flashsac_vel_finetune",
    "joint_flashsac_vel_finetune",
    "cosil_flashsac_vel_finetune",
    "tgrl_flashsac_vel_finetune",
    "gaddpg_cosil_flashsac_vel_finetune",
    "finetune_schedule",
    "flashsac_upstream",
]

# These imports register the existing Hydra algo names using their new targets.
for _name in __all__:
    globals()[_name] = _importlib.import_module(f".{_name}", __name__)


def _register_legacy_imports():
    # Old checkpoints pickle structured-config classes from learning.ppo, and
    # saved Hydra configs contain that path. Alias the modules themselves so old
    # and new imports share class identities, including vendored submodules.
    ppo = _importlib.import_module("..ppo", __name__)
    prefix = f"{__name__}."
    for name, module in list(_sys.modules.items()):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            _sys.modules[f"{ppo.__name__}.{suffix}"] = module
            if "." not in suffix:
                setattr(ppo, suffix, module)


_register_legacy_imports()
