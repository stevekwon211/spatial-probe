# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""TDD for the C-space reachable free-space field (the v1 predicate substrate).

Synthetic occupancy with hand-known geometry. voxel_size=1.0, origin=(0,0,0) so voxel (i,j,k)
sits at world (i,j,k); ground_height=0.5 so z>=1 is non-ground. Ego stands at (20,20) facing +x,
so ego-frame forward = world_x - 20 and lateral = world_y - 20.
"""
import numpy as np

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.predicates.reachable import reachable_free_field


def _empty(n: int = 40) -> np.ndarray:
    return np.full((n, n, n), FREE, dtype=int)


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, voxel_size=1.0, origin=(0.0, 0.0, 0.0), ground_height=0.5)


def _ego(speed: float = 10.0) -> EgoPose:
    return EgoPose((20, 20, 0), 0.0, speed=speed, width=1.85, length=4.6)


def test_empty_field_all_reachable():
    f = reachable_free_field(_grid(_empty()), _ego(), 2.0)
    assert f.is_reachable(5, 0)
    assert f.is_reachable(7, 3)
    assert f.is_reachable(0, 0)


def test_frontal_wall_blocks_everything_behind():
    occ = _empty(44)  # wider than the lateral window so the wall fully seals the road
    occ[25, :, 2] = OCCUPIED  # wall spanning the whole road at forward 5 (a dead end)
    f = reachable_free_field(_grid(occ), _ego(), 2.0)
    assert f.is_reachable(3, 0)        # ahead of the wall
    assert not f.is_reachable(7, 0)    # behind the wall -- unreachable (no way around)


def test_corridor_between_two_walls():
    occ = _empty()
    for x in range(20, 31):
        occ[x, 22, 2] = OCCUPIED  # left wall  (lateral +2)
        occ[x, 18, 2] = OCCUPIED  # right wall (lateral -2)
    f = reachable_free_field(_grid(occ), _ego(), 2.0)
    assert f.is_reachable(5, 0)        # inside the corridor
    assert not f.is_reachable(5, 2)    # on the +2 wall itself -- C-obstacle, not free
    # NOTE: cells beyond the wall (e.g. lateral 6) ARE globally reachable by going around the
    # finite wall's ends -- correct for reachability. Corridor WIDTH must therefore be measured
    # as the lateral run containing the centerline at a forward station, not global reachability.


def test_frontal_clump_is_bypassed_not_a_wall():
    # the v1 anti-false-corridor property: a small frontal clump on the centerline does NOT make
    # everything behind it unreachable, because the ego can go around the side.
    occ = _empty()
    occ[25:28, 19:22, 2] = OCCUPIED  # forward 5-7, lateral -1..+1, a small clump
    f = reachable_free_field(_grid(occ), _ego(), 2.0)
    assert f.is_reachable(3, 0)        # ahead of the clump
    assert f.is_reachable(10, 5)       # behind+beside the clump, reached by going around


def test_clearance_is_distance_to_nearest_wall():
    occ = _empty()
    for x in range(20, 31):
        occ[x, 22, 2] = OCCUPIED  # lateral +2
        occ[x, 18, 2] = OCCUPIED  # lateral -2
    f = reachable_free_field(_grid(occ), _ego(), 2.0)
    # nearest obstacle center to the centerline (lateral 0) at forward 5 is the +/-2 voxel: 2.0 m
    assert f.clearance_at(5, 0) == 2.0


def test_ego_in_collision_yields_empty_reachable():
    occ = _empty()
    occ[20, 20, 2] = OCCUPIED  # an obstacle right on the ego footprint center
    f = reachable_free_field(_grid(occ), _ego(), 2.0)
    assert not f.is_reachable(5, 0)
    assert not f.reachable.any()
