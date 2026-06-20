# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""TDD for the OccQuery v0 predicates -- the executable spec for grid + predicates.

Synthetic occupancy grids with hand-known geometry, no dataset. voxel_size=1.0 and
origin=(0,0,0) so voxel (i,j,k) sits at world (i,j,k); ground_height=0.5 so z>=1 is non-ground.
Ego defaults: width 1.85, length 4.6, height 1.9. Physical clearance subtracts the ego
half-width (0.925) and a voxel half-extent (0.5), i.e. 1.425 off the centerline distance.
"""
import math

import numpy as np
import pytest

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid, UnknownPolicy
from probe.predicates.clearance import centerline_lateral_distance, lateral_clearance
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path

_OFFSET = 1.425  # ego half-width (0.925) + voxel half-extent (0.5) at voxel_size=1.0


def _empty(n: int = 40) -> np.ndarray:
    return np.full((n, n, n), FREE, dtype=int)


def _grid(occ: np.ndarray, ground_height: float = 0.5, voxel_size: float = 1.0) -> OccupancyGrid:
    return OccupancyGrid(occ, voxel_size=voxel_size, origin=(0.0, 0.0, 0.0), ground_height=ground_height)


# --- centerline_lateral_distance (raw geometric quantity) ---

def test_centerline_distance_is_perpendicular_offset():
    occ = _empty()
    occ[5, 2, 2] = OCCUPIED  # 5 m ahead, 2 m left, above ground
    assert centerline_lateral_distance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == 2.0


def test_centerline_distance_takes_the_minimum():
    occ = _empty()
    occ[5, 3, 2] = OCCUPIED
    occ[8, 1, 2] = OCCUPIED  # farther ahead but nearer laterally
    assert centerline_lateral_distance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == 1.0


def test_centerline_distance_follows_heading():
    occ = _empty()
    occ[20, 25, 2] = OCCUPIED  # due north of ego
    ego = EgoPose((20, 20, 0), math.pi / 2)  # facing north -> dead ahead
    assert centerline_lateral_distance(_grid(occ), ego) == pytest.approx(0.0, abs=1e-9)


# --- lateral_clearance (physical free gap) ---

def test_lateral_clearance_is_physical_gap():
    occ = _empty()
    occ[5, 3, 2] = OCCUPIED  # centerline distance 3.0 -> gap 3.0 - 1.425
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == pytest.approx(3.0 - _OFFSET)


def test_lateral_clearance_floors_at_zero_on_overlap():
    occ = _empty()
    occ[5, 0, 2] = OCCUPIED  # dead ahead, centerline distance 0 -> would be negative -> 0
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == 0.0


def test_lateral_clearance_empty_corridor_is_inf():
    assert lateral_clearance(_grid(_empty()), EgoPose((0, 0, 0), 0.0)) == math.inf


def test_clearance_excludes_ground_voxels():
    occ = _empty()
    occ[5, 1, 0] = OCCUPIED  # z=0 <= ground_height -> ground, ignored
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == math.inf


def test_clearance_excludes_voxels_above_ego_height():
    occ = _empty()
    occ[5, 1, 5] = OCCUPIED  # z=5 > ground(0.5)+ego.height(1.9)=2.4 -> overhead, ignored
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == math.inf


def test_clearance_ignores_obstacles_behind_ego():
    occ = _empty()
    occ[15, 21, 2] = OCCUPIED  # behind ego standing at (20, 20)
    assert lateral_clearance(_grid(occ), EgoPose((20, 20, 0), 0.0)) == math.inf


def test_clearance_translation_invariant():
    occ = _empty()
    occ[25, 23, 2] = OCCUPIED  # 5 m ahead, 3 m left of ego at (20, 20)
    g1 = OccupancyGrid(occ, 1.0, (0.0, 0.0, 0.0), 0.5)
    g2 = OccupancyGrid(occ, 1.0, (100.0, 100.0, 0.0), 0.5)  # shift world origin
    e1 = EgoPose((20, 20, 0), 0.0)
    e2 = EgoPose((120, 120, 0), 0.0)  # shift ego the same
    assert lateral_clearance(g1, e1) == lateral_clearance(g2, e2)


# --- unknown-voxel policy ---

def test_unknown_policy_free_vs_occupied():
    occ = _empty()
    occ[5, 3, 2] = UNKNOWN
    g = _grid(occ)
    ego = EgoPose((0, 0, 0), 0.0)
    assert lateral_clearance(g, ego, unknown_policy=UnknownPolicy.FREE) == math.inf
    assert lateral_clearance(g, ego, unknown_policy=UnknownPolicy.OCCUPIED) == pytest.approx(3.0 - _OFFSET)


# --- free_along_ego_path ---

def _ego(speed: float = 10.0) -> EgoPose:
    return EgoPose((20, 20, 0), 0.0, speed=speed, width=1.85, length=4.6)


def test_free_path_empty_is_true():
    assert free_along_ego_path(_grid(_empty()), _ego(), horizon=2.0) is True


def test_free_path_obstacle_on_centerline_blocks():
    occ = _empty()
    occ[25, 20, 2] = OCCUPIED  # 5 m ahead on the centerline, within reach
    assert free_along_ego_path(_grid(occ), _ego(), horizon=2.0) is False


def test_free_path_obstacle_outside_width_is_free():
    occ = _empty()
    occ[25, 25, 2] = OCCUPIED  # ahead but 5 m to the side
    assert free_along_ego_path(_grid(occ), _ego(), horizon=2.0) is True


def test_free_path_obstacle_beyond_horizon_is_free():
    occ = _empty()
    occ[39, 20, 2] = OCCUPIED  # 19 m ahead; reach = 2.3 + 10*0.5 + 0.5 = 7.8
    assert free_along_ego_path(_grid(occ), _ego(), horizon=0.5) is True


def test_free_path_horizon_zero_tests_the_body():
    occ = _empty()
    occ[21, 20, 2] = OCCUPIED  # 1 m ahead, inside the ego body half-length
    assert free_along_ego_path(_grid(occ), _ego(), horizon=0.0) is False


def test_free_path_unknown_policy():
    occ = _empty()
    occ[25, 20, 2] = UNKNOWN
    g = _grid(occ)
    assert free_along_ego_path(g, _ego(), 2.0, unknown_policy=UnknownPolicy.FREE) is True
    assert free_along_ego_path(g, _ego(), 2.0, unknown_policy=UnknownPolicy.OCCUPIED) is False


# --- min_free_width_along_path ---

def test_min_free_width_symmetric_corridor():
    occ = _empty()
    for x in range(21, 28):
        occ[x, 22, 2] = OCCUPIED  # lateral +2
        occ[x, 18, 2] = OCCUPIED  # lateral -2
    # inner surfaces at +1.5 and -1.5 -> width 3.0
    assert min_free_width_along_path(_grid(occ), _ego(), horizon=0.5) == pytest.approx(3.0)


def test_min_free_width_asymmetric_corridor():
    occ = _empty()
    occ[24, 22, 2] = OCCUPIED  # lateral +2 -> inner 1.5
    occ[24, 17, 2] = OCCUPIED  # lateral -3 -> inner 2.5
    assert min_free_width_along_path(_grid(occ), _ego(), horizon=0.5) == pytest.approx(4.0)


def test_min_free_width_narrowing_returns_tightest():
    occ = _empty()
    occ[22, 23, 2] = OCCUPIED
    occ[22, 17, 2] = OCCUPIED  # forward 2: +/-3 -> width 5.0
    occ[25, 21, 2] = OCCUPIED
    occ[25, 19, 2] = OCCUPIED  # forward 5: +/-1 -> width 1.0
    assert min_free_width_along_path(_grid(occ), _ego(), horizon=0.5) == pytest.approx(1.0)


def test_min_free_width_one_sided_is_inf():
    occ = _empty()
    for x in range(21, 28):
        occ[x, 22, 2] = OCCUPIED  # only a left wall -> corridor not bounded both sides
    assert min_free_width_along_path(_grid(occ), _ego(), horizon=0.5) == math.inf


def test_min_free_width_empty_is_inf():
    assert min_free_width_along_path(_grid(_empty()), _ego(), horizon=0.5) == math.inf
