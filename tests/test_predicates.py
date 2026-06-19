# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""TDD for the OccQuery v0 predicates — the executable spec for grid + predicates.

Synthetic occupancy grids with hand-known geometry, no dataset. voxel_size=1.0 and
origin=(0,0,0) so voxel (i,j,k) sits at world (i,j,k) — keeps the asserted numbers obvious.
"""
import math

import numpy as np
import pytest

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.predicates.clearance import lateral_clearance
from probe.predicates.freepath import free_along_ego_path


def _empty(n: int = 40) -> np.ndarray:
    return np.full((n, n, n), FREE, dtype=int)


def _grid(occ: np.ndarray, ground_height: float = 0.5) -> OccupancyGrid:
    return OccupancyGrid(occ, voxel_size=1.0, origin=(0.0, 0.0, 0.0), ground_height=ground_height)


# --- lateral_clearance ---

def test_clearance_empty_corridor_is_inf():
    assert lateral_clearance(_grid(_empty()), EgoPose((0, 0, 0), 0.0)) == math.inf


def test_clearance_is_perpendicular_offset():
    occ = _empty()
    occ[5, 2, 2] = OCCUPIED  # 5 m ahead, 2 m to the left, above ground
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == 2.0


def test_clearance_excludes_ground_voxels():
    occ = _empty()
    occ[5, 1, 0] = OCCUPIED  # z=0 <= ground_height -> ground, ignored
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == math.inf


def test_clearance_ignores_obstacles_behind_ego():
    occ = _empty()
    occ[15, 21, 2] = OCCUPIED  # behind ego standing at (20, 20)
    assert lateral_clearance(_grid(occ), EgoPose((20, 20, 0), 0.0)) == math.inf


def test_clearance_follows_heading():
    occ = _empty()
    occ[20, 25, 2] = OCCUPIED  # due north of ego
    ego = EgoPose((20, 20, 0), math.pi / 2)  # facing north -> obstacle is dead ahead
    assert lateral_clearance(_grid(occ), ego) == pytest.approx(0.0, abs=1e-9)


def test_clearance_takes_the_minimum():
    occ = _empty()
    occ[5, 3, 2] = OCCUPIED
    occ[8, 1, 2] = OCCUPIED  # farther ahead but nearer laterally
    assert lateral_clearance(_grid(occ), EgoPose((0, 0, 0), 0.0)) == 1.0


# --- free_along_ego_path ---

def _ego(speed: float = 10.0) -> EgoPose:
    return EgoPose((20, 20, 0), 0.0, speed=speed, width=1.85, length=4.6)


def test_free_path_empty_is_true():
    assert free_along_ego_path(_grid(_empty()), _ego(), horizon=2.0) is True


def test_free_path_obstacle_on_centerline_blocks():
    occ = _empty()
    occ[25, 20, 2] = OCCUPIED  # 5 m ahead, within speed*horizon
    assert free_along_ego_path(_grid(occ), _ego(), horizon=2.0) is False


def test_free_path_obstacle_outside_width_is_free():
    occ = _empty()
    occ[25, 25, 2] = OCCUPIED  # ahead but 5 m to the side
    assert free_along_ego_path(_grid(occ), _ego(), horizon=2.0) is True


def test_free_path_obstacle_beyond_horizon_is_free():
    occ = _empty()
    occ[39, 20, 2] = OCCUPIED  # 19 m ahead; reach = 2.3 + 10*0.5 = 7.3
    assert free_along_ego_path(_grid(occ), _ego(), horizon=0.5) is True


def test_free_path_horizon_zero_tests_the_body():
    occ = _empty()
    occ[21, 20, 2] = OCCUPIED  # 1 m ahead, inside the ego body half-length
    assert free_along_ego_path(_grid(occ), _ego(), horizon=0.0) is False
