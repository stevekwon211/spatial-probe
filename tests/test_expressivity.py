# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Expressivity witness (OccQuery H1) -- the strongest oracle-free separation.

Two scenes with IDENTICAL tracked-box observables that differ only by an UNBOXED occupancy
obstacle. A box-only query language is structurally blind to the difference (its observation is
identical), so it MUST return the same answer; the occupancy predicate returns different answers.
This is a non-identifiability witness, not a score: it holds by construction, anyone can re-run
it, and it needs no oracle.

Scope note: this witnesses separation against the *tracked-box observable set* used here (object
center/size/yaw/class/velocity). It is not a proof against RefAV's entire function language; the
honest claim is "box observations cannot distinguish these scenes, therefore no box-only function
of them can."
"""
import math

import numpy as np

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.predicates.clearance import lateral_clearance
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path
from probe.predicates.objects import distance_to_nearest_object
from probe.scene import Frame, Scene, TrackedBox

_VOXEL = 0.2
_N = 120
_GROUND = 0.5


def _blank() -> np.ndarray:
    return np.full((_N, _N, _N), FREE, dtype=int)


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, _VOXEL, (0.0, 0.0, 0.0), _GROUND)


def _occupy(occ: np.ndarray, x: float, y: float, z: float = 1.0) -> None:
    occ[round(x / _VOXEL), round(y / _VOXEL), round(z / _VOXEL)] = OCCUPIED


def _witness_pair() -> tuple[Scene, Scene]:
    """(clear, unboxed_wall): same tracked vehicle box (20 m to the side, identical), differing
    only by an unboxed wall in the ego corridor of the second scene."""
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=10.0)
    box = TrackedBox(center=(10.0, 30.0, 1.0), size=(4.5, 2.0, 1.8), yaw=0.0, label="vehicle")
    occ_clear = _blank()
    occ_wall = _blank()
    lateral = ego.width / 2.0 + _VOXEL / 2.0 + 0.4  # an unboxed wall ~0.4 m off the ego body side
    for fx in (-2.0, -1.0, 0.0, 1.0, 2.0):  # ALONGSIDE the ego body (abeam) -- the side pass the
        _occupy(occ_wall, 10.0 + fx, 10.0 + lateral)  # ego is making now, which lateral_clearance reads
    clear = Scene((Frame(_grid(occ_clear), ego, 0.0, objects=(box,)),), "witness_clear")
    wall = Scene((Frame(_grid(occ_wall), ego, 0.0, objects=(box,)),), "witness_unboxed_wall")
    return clear, wall


def test_box_backend_cannot_distinguish_the_pair():
    clear, wall = _witness_pair()
    # identical box observable -> a box-only language MUST return the same answer
    assert distance_to_nearest_object(clear, 0, object_class="vehicle") == distance_to_nearest_object(
        wall, 0, object_class="vehicle"
    )


def test_occupancy_predicate_distinguishes_the_pair():
    clear, wall = _witness_pair()
    c_clear = lateral_clearance(clear.grid_at(0), clear.ego_at(0))
    c_wall = lateral_clearance(wall.grid_at(0), wall.ego_at(0))
    assert math.isinf(c_clear)  # corridor is clear
    assert math.isfinite(c_wall)  # the unboxed wall is seen
    assert c_clear != c_wall


def test_witness_survives_array_roundtrip():
    # the fixture is machine-reproducible: rebuilding from serialized occupancy + box yields the
    # same denotation (no hidden state)
    clear, wall = _witness_pair()
    occ = wall.grid_at(0).occupancy
    reloaded = Scene(
        (Frame(OccupancyGrid(occ.copy(), _VOXEL, (0.0, 0.0, 0.0), _GROUND), wall.ego_at(0), 0.0),),
        "reloaded",
    )
    assert lateral_clearance(reloaded.grid_at(0), reloaded.ego_at(0)) == lateral_clearance(
        wall.grid_at(0), wall.ego_at(0)
    )


# --- H1 family 2: free-path (free_along_ego_path) ---
# Honest scope (same as family 1): the box+map observable channel is held identical (both scenes
# carry an empty, therefore identical, map), so no function of that channel can distinguish the pair;
# this is NOT a claim that no RefAV composition can. The separation is oracle-free and by construction.


def _freepath_witness_pair() -> tuple[Scene, Scene, np.ndarray, np.ndarray]:
    """(clear, blocked): identical box + ego + (empty) map; differ ONLY by an unboxed obstacle
    CLUSTER on the ego centerline. Separates free_along_ego_path. A 4-voxel cluster (not a lone
    voxel) so it survives the realistic min_cluster_voxels noise gate. The sibling predicate
    min_free_width_along_path TIES on this pair (a frontal block is never a two-sided corridor),
    asserted below as the scope boundary."""
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=10.0)
    box = TrackedBox(center=(10.0, 30.0, 1.0), size=(4.5, 2.0, 1.8), yaw=0.0, label="vehicle")
    occ_clear = _blank()
    occ_block = _blank()
    # 4-voxel 8-connected cluster straddling the centerline ~8 m ahead, inside the forward reach.
    for dx in (0.0, 0.2):
        for dy in (0.0, 0.2):
            _occupy(occ_block, 18.0 + dx, 10.0 + dy)
    clear = Scene((Frame(_grid(occ_clear), ego, 0.0, objects=(box,)),), "witness")
    block = Scene((Frame(_grid(occ_block), ego, 0.0, objects=(box,)),), "witness")
    return clear, block, occ_clear, occ_block


def test_freepath_box_backend_cannot_distinguish():
    clear, block, _, _ = _freepath_witness_pair()
    # every box field the box-only language reads (center, label) is identical -> forced tie
    assert distance_to_nearest_object(clear, 0, object_class="vehicle") == distance_to_nearest_object(
        block, 0, object_class="vehicle"
    )


def test_freepath_occupancy_distinguishes():
    clear, block, _, _ = _freepath_witness_pair()
    assert free_along_ego_path(clear.grid_at(0), clear.ego_at(0), 1.0) is True
    assert free_along_ego_path(block.grid_at(0), block.ego_at(0), 1.0) is False
    # noise-robust: a cluster, not a lone voxel, so it survives min_cluster_voxels >= 2
    assert free_along_ego_path(block.grid_at(0), block.ego_at(0), 1.0, min_cluster_voxels=2) is False


def test_freepath_inputs_differ_only_in_occupancy():
    clear, block, oc, ob = _freepath_witness_pair()
    assert int((oc != ob).sum()) == 4  # exactly the 4-voxel cluster, nothing else
    # scope boundary: the SIBLING predicate min_free_width TIES (frontal block, never two-sided)
    assert math.isinf(min_free_width_along_path(clear.grid_at(0), clear.ego_at(0), 1.0))
    assert math.isinf(min_free_width_along_path(block.grid_at(0), block.ego_at(0), 1.0))


# --- H1 family 3: corridor (min_free_width_along_path) ---


def _corridor_witness_pair() -> tuple[Scene, Scene, np.ndarray, np.ndarray]:
    """(narrow, wide): identical box + ego + (empty) map; differ ONLY by unboxed walls bounding the
    corridor on BOTH sides -- near (narrow) vs far (wide). Walls on clean voxel centers to avoid
    round() ties; assertions are robust (both finite, narrow < wide), never an exact width."""
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=10.0)
    box = TrackedBox(center=(10.0, 30.0, 1.0), size=(4.5, 2.0, 1.8), yaw=0.0, label="vehicle")
    occ_narrow = _blank()
    occ_wide = _blank()
    for i in range(0, 60):
        x = 10.0 + 0.2 * i  # forward 0 .. ~11.8 m, inside the reach
        _occupy(occ_narrow, x, 11.0)  # +1.0 m left, > ego.width/2 so the centerline stays reachable
        _occupy(occ_narrow, x, 9.0)   # -1.0 m right
        _occupy(occ_wide, x, 15.0)    # +5.0 m
        _occupy(occ_wide, x, 5.0)     # -5.0 m
    narrow = Scene((Frame(_grid(occ_narrow), ego, 0.0, objects=(box,)),), "witness")
    wide = Scene((Frame(_grid(occ_wide), ego, 0.0, objects=(box,)),), "witness")
    return narrow, wide, occ_narrow, occ_wide


def test_corridor_box_backend_cannot_distinguish():
    narrow, wide, _, _ = _corridor_witness_pair()
    assert distance_to_nearest_object(narrow, 0, object_class="vehicle") == distance_to_nearest_object(
        wide, 0, object_class="vehicle"
    )


def test_corridor_occupancy_distinguishes():
    narrow, wide, _, _ = _corridor_witness_pair()
    w_narrow = min_free_width_along_path(narrow.grid_at(0), narrow.ego_at(0), 2.0)
    w_wide = min_free_width_along_path(wide.grid_at(0), wide.ego_at(0), 2.0)
    # robust: both two-sided corridors are finite; the unboxed walls make narrow < wide.
    # exact value avoided -- it rides a round() tie and is fragile to a sub-voxel origin shift.
    assert math.isfinite(w_narrow) and math.isfinite(w_wide)
    assert w_narrow < w_wide  # the box channel is blind to the unboxed walls
