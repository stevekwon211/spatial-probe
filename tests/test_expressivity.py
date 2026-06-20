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
