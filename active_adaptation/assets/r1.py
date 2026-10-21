"""R1 robot and interaction-object assets used by R1 tracking tasks.

The heavy USD layers stay in the neighboring HOI checkout.  Set
``VAIC_HOI_ROOT`` when that checkout is located somewhere other than the
default path used on this workstation.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg


HOI_ROOT = os.path.realpath(
    os.path.expanduser(os.environ.get("VAIC_HOI_ROOT", "/home/hcc/research/HOI"))
)
R1_USD_PATH = os.path.join(
    HOI_ROOT,
    "src/holosoma_retargeting/holosoma_retargeting/models",
    "converted_rank0/r1_26dof.usd",
)
R1_LARGEBOX_USD_PATH = os.path.join(
    HOI_ROOT,
    "train_r1/objects/largebox/converted_rank0/largebox.usd",
)
R1_WHITECHAIR_USD_PATH = os.path.join(
    HOI_ROOT,
    "train_r1/objects/whitechair/converted_rank0/whitechair.usd",
)
R1_SMALLTABLE_USD_PATH = os.path.join(
    HOI_ROOT,
    "train_r1/objects/smalltable/converted_rank0/smalltable.usd",
)
R1_SUITCASE_USD_PATH = os.path.join(
    HOI_ROOT,
    "train_r1/objects/suitcase/converted_rank0/suitcase.usd",
)
R1_PLASTICBOX_USD_PATH = os.path.join(
    HOI_ROOT,
    "train_r1/objects/plasticbox/converted_rank0/plasticbox.usd",
)


# Holosoma computes these gains in its torque controller.  VAIC's action
# manager sends position targets, so the equivalent gains belong on implicit
# actuators here (zero-gain actuators would leave the robot unactuated).
R1_26DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_ankle_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.87,
            ".*_wrist_roll_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips_and_knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=18.8,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
            friction=0.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            friction=0.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_(roll|yaw)_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=18.8,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
            friction=0.0,
        ),
        "shoulder_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_(pitch|roll)_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=18.8,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            friction=0.0,
        ),
        "small_upper_body": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                "head_(pitch|yaw)_joint",
            ],
            effort_limit_sim=33.0,
            velocity_limit_sim=33.4,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
            friction=0.0,
        ),
    },
)


def _r1_object_cfg(usd_path: str, scene_name: str) -> RigidObjectCfg:
    """Create the common Holosoma rigid-object configuration."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{scene_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ),
    )


R1_LARGEBOX_CFG = _r1_object_cfg(R1_LARGEBOX_USD_PATH, "largebox_link")
R1_WHITECHAIR_CFG = _r1_object_cfg(R1_WHITECHAIR_USD_PATH, "whitechair_link")
R1_SMALLTABLE_CFG = _r1_object_cfg(R1_SMALLTABLE_USD_PATH, "smalltable_link")
R1_SUITCASE_CFG = _r1_object_cfg(R1_SUITCASE_USD_PATH, "suitcase_link")
R1_PLASTICBOX_CFG = _r1_object_cfg(R1_PLASTICBOX_USD_PATH, "plasticbox_link")
