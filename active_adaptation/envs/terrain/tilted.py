"""Planar floor tiles tilted by a fixed set of angles, one uphill direction per column.

With curriculum generation each column is one sub-terrain type (uphill direction) and
row ``i`` gets ``TILT_MAGNITUDES_DEG[i]``. Every plane passes through its tile center at
height 0, which is the tile's origin. ``RobotTiltedFloorTracking`` measures each tile's
slope from the ground mesh and moves envs between tiles at reset.
"""

import math
from dataclasses import MISSING

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

# Tilting a tile-sized plane shrinks its xy footprint by up to half_size * (1 - cos(tilt))
# per side (2.7 mm at 3 deg on a 4 m tile), which leaves gaps between tiles. A body above a
# gap gets no ground raycast hit and body_height becomes 3.4e38. Each plane overhangs its
# tile by this margin so neighbouring tiles overlap.
OVERHANG = 0.01


def tilted_plane_terrain(difficulty: float, cfg: "TiltedPlaneTerrainCfg"):
    # curriculum rows give difficulty in [row / num_rows, (row + 1) / num_rows)
    index = min(int(difficulty * len(cfg.tilt_degrees)), len(cfg.tilt_degrees) - 1)
    tilt = math.radians(cfg.tilt_degrees[index])
    direction = math.radians(cfg.direction_deg)
    center = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    mesh = make_plane((cfg.size[0] + 2 * OVERHANG, cfg.size[1] + 2 * OVERHANG), 0.0, center_zero=False)
    mesh.apply_translation([-OVERHANG, -OVERHANG, 0.0])
    # rotating about (sin, -cos, 0) raises the side toward (cos, sin, 0)
    axis = [math.sin(direction), -math.cos(direction), 0.0]
    mesh.apply_transform(trimesh.transformations.rotation_matrix(tilt, axis, point=center))
    return [mesh], center


@configclass
class TiltedPlaneTerrainCfg(SubTerrainBaseCfg):
    function = tilted_plane_terrain

    tilt_degrees: tuple[float, ...] = MISSING
    """Tilt of each curriculum row, in degrees."""
    direction_deg: float = MISSING
    """Uphill direction in the world xy plane, in degrees from +x."""


TILT_MAGNITUDES_DEG = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
NUM_TILT_DIRECTIONS = 36  # 10 degree steps

TILTED_PLANES = TerrainGeneratorCfg(
    seed=0,
    # A tile must hold a whole clip around its center; sub3_largebox_003 stays within 1 m.
    size=(4.0, 4.0),
    border_width=1.0,
    num_rows=len(TILT_MAGNITUDES_DEG),
    num_cols=NUM_TILT_DIRECTIONS,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        f"tilt_dir_{k * 360 // NUM_TILT_DIRECTIONS:03d}": TiltedPlaneTerrainCfg(
            proportion=1.0,
            tilt_degrees=TILT_MAGNITUDES_DEG,
            direction_deg=k * 360.0 / NUM_TILT_DIRECTIONS,
        )
        for k in range(NUM_TILT_DIRECTIONS)
    },
)

TERRAINS = {
    # Contact and look match the "plane" terrain.
    "tilted_planes": TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TILTED_PLANES,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        debug_vis=False,
    ),
}
