# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S3 physical predicates (prism.predicates) on synthetic grids -- deterministic, no data needed.

`occluded` is the edge predicate a box-only tool cannot express: it reads the dense occupancy
between the ego and a target point. `velocity` reads a per-object speed and is explicit about an
unknown (None, never a silent 0). `ttc` ships flagged: the dynfield pre-registered NEGATIVE means
occupancy/velocity is action-equivalent to boxes, so ttc is a primitive, not a validated signal --
the warning MUST be in its docstring.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from probe.grid import EgoPose, OccupancyGrid, UnknownPolicy
from probe.raycast import FREE, OCCUPIED, UNKNOWN
from probe.scene import Frame, Scene, TrackedBox
from prism.ir import Entity, Pose, SceneIR
from prism.predicates import occluded, ttc, velocity


def _grid(occ: np.ndarray, voxel_size: float = 1.0) -> OccupancyGrid:
    return OccupancyGrid(occupancy=occ, voxel_size=voxel_size, origin=(0.0, 0.0, 0.0), ground_height=-0.5)


def _scene_one_frame(grid: OccupancyGrid, ego: EgoPose, objects=()) -> SceneIR:
    sc = Scene(frames=(Frame(grid=grid, ego=ego, time=0.0, objects=tuple(objects)),), name="synthetic")
    return SceneIR(scene=sc)


# --- occluded -------------------------------------------------------------------------------

def test_occluded_true_behind_an_obstacle():
    # 10x1x3 corridor along +x; a solid wall at x=5 across the full z-band.
    occ = np.full((10, 1, 3), FREE, dtype=int)
    occ[5, 0, :] = OCCUPIED
    grid = _grid(occ)
    ego = EgoPose(position=(0.0, 0.0, 1.0), heading=0.0)
    ir = _scene_one_frame(grid, ego)
    # target at x=9 (the far side of the wall, same z) -> the ray must cross the wall.
    assert occluded(ir, 0, 9.0, 0.0, target_z=1.0) is True


def test_occluded_false_in_open_space():
    occ = np.full((10, 1, 3), FREE, dtype=int)
    grid = _grid(occ)
    ego = EgoPose(position=(0.0, 0.0, 1.0), heading=0.0)
    ir = _scene_one_frame(grid, ego)
    assert occluded(ir, 0, 9.0, 0.0, target_z=1.0) is False


def test_occluded_unknown_blocks_only_when_requested():
    occ = np.full((10, 1, 3), FREE, dtype=int)
    occ[5, 0, :] = UNKNOWN
    grid = _grid(occ)
    ego = EgoPose(position=(0.0, 0.0, 1.0), heading=0.0)
    ir = _scene_one_frame(grid, ego)
    # default: UNKNOWN does not block -> visible.
    assert occluded(ir, 0, 9.0, 0.0, target_z=1.0) is False
    # conservative: UNKNOWN blocks -> occluded.
    assert occluded(ir, 0, 9.0, 0.0, target_z=1.0, unknown_blocks=True) is True


# --- velocity -------------------------------------------------------------------------------

def test_velocity_reads_a_moving_entity():
    e = Entity(entity_id="v0", category="vehicle", pose=Pose.from_yaw((3.0, 0.0, 0.0), 0.0),
               size=(4.0, 2.0, 1.6), velocity=(3.0, 4.0))
    assert velocity(e) == pytest.approx(5.0)  # hypot(3, 4)


def test_velocity_is_none_for_unknown_velocity():
    e = Entity(entity_id="v1", category="vehicle", pose=Pose.from_yaw((3.0, 0.0, 0.0), 0.0),
               size=(4.0, 2.0, 1.6), velocity=(float("nan"), float("nan")))
    assert velocity(e) is None  # explicit unknown, NOT a silent 0


def test_velocity_zero_is_a_real_zero_not_unknown():
    e = Entity(entity_id="v2", category="vehicle", pose=Pose.from_yaw((3.0, 0.0, 0.0), 0.0),
               size=(4.0, 2.0, 1.6), velocity=(0.0, 0.0))
    assert velocity(e) == 0.0  # a measured-stationary object is 0.0, not None


# --- ttc (flagged primitive) ----------------------------------------------------------------

def test_ttc_closing_scenario_is_sane():
    # ego at origin facing +x, a lead box 10 m ahead closing at 2 m/s relative.
    occ = np.full((20, 1, 3), FREE, dtype=int)
    grid = _grid(occ)
    ego = EgoPose(position=(0.0, 0.0, 0.0), heading=0.0, speed=0.0)
    lead = TrackedBox(center=(10.0, 0.0, 0.0), size=(4.0, 2.0, 1.6), yaw=0.0, label="vehicle",
                      velocity=(-2.0, 0.0))  # moving back toward the ego at 2 m/s
    ir = _scene_one_frame(grid, ego, objects=(lead,))
    t = ttc(ir, 0)
    assert t == pytest.approx(5.0, abs=0.5)  # 10 m / 2 m/s


def test_ttc_is_inf_when_not_closing():
    occ = np.full((20, 1, 3), FREE, dtype=int)
    grid = _grid(occ)
    ego = EgoPose(position=(0.0, 0.0, 0.0), heading=0.0, speed=0.0)
    lead = TrackedBox(center=(10.0, 0.0, 0.0), size=(4.0, 2.0, 1.6), yaw=0.0, label="vehicle",
                      velocity=(2.0, 0.0))  # moving away
    ir = _scene_one_frame(grid, ego, objects=(lead,))
    assert ttc(ir, 0) == math.inf


def test_ttc_carries_the_dynfield_negative_warning():
    assert "NEGATIVE" in ttc.__doc__
    assert "dynfield" in ttc.__doc__
