import os

from .objects import *
from .g1 import *
from .r1 import *


ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CYLINDER_CFG,
    "r1": R1_26DOF_CFG,
}

OBJECTS = {
    "mybox": MYBOX_CFG,
    "skateboard": SKATEBOARD_CFG,
    "cart": CART_CFG,
    "wheelchair": WHEELCHAIR_CFG,
    "stairslope": STAIRSLOPE_CFG,
    # Match the scene names stored in the R1 motion archives.
    "largebox_link": R1_LARGEBOX_CFG,
    "whitechair_link": R1_WHITECHAIR_CFG,
    "smalltable_link": R1_SMALLTABLE_CFG,
    "suitcase_link": R1_SUITCASE_CFG,
    "plasticbox_link": R1_PLASTICBOX_CFG,
}


def get_asset_meta(asset: Articulation):
    if not asset.is_initialized:
        raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asset.cfg.init_state.to_dict(),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
    }
    if asset.is_initialized: # parsed values
        meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
        meta["stiffness"] = asset.data.joint_stiffness[0].tolist()
        meta["damping"] = asset.data.joint_damping[0].tolist()

    for actuator_name, actuator in asset.actuators.items():
        meta["actuators"][actuator_name] = actuator.cfg.to_dict()
    return meta

