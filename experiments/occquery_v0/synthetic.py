# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Synthetic, hand-labeled scenes for occquery_v0 -- exercise the full retrieval loop with NO
dataset. Each scene's geometry is built so its ground-truth query membership is known by
construction, and the scenes are designed so each occupancy query matches exactly its GT scene
(clean denotation), plus controls and one deliberately unknown-sensitive scene.

This is a SMOKE harness, not science. At M2 it is replaced by the Occ3D-nuScenes adapter: same
`Scene` type, real geometry, GT from hand-labeling instead of construction.
"""
from __future__ import annotations

import numpy as np

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox

VOXEL = 0.2          # m per voxel edge (fine enough for sub-0.5 m clearance)
GRID_N = 160         # 32 m cube
ORIGIN = (0.0, 0.0, 0.0)
GROUND_H = 0.5
_EGO_XY = (10.0, 10.0)

# centerline offset that yields a ~0.4 m physical clearance: half-width + voxel-half + 0.4
_WALL_LAT = 1.85 / 2.0 + VOXEL / 2.0 + 0.4


def _blank() -> np.ndarray:
    return np.full((GRID_N, GRID_N, GRID_N), FREE, dtype=int)


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, VOXEL, ORIGIN, GROUND_H)


def _ego(speed: float) -> EgoPose:
    return EgoPose((*_EGO_XY, 0.0), 0.0, speed=speed)


def _place(occ: np.ndarray, ego: EgoPose, forward: float, lateral: float, height: float = 1.0, value: int = OCCUPIED) -> None:
    """Set one voxel at (forward, lateral, height) in the ego frame."""
    fx, fy = ego.forward
    lx, ly = ego.left
    wx = ego.position[0] + forward * fx + lateral * lx
    wy = ego.position[1] + forward * fy + lateral * ly
    occ[round(wx / VOXEL), round(wy / VOXEL), round(height / VOXEL)] = value


def _wall(occ: np.ndarray, ego: EgoPose, forwards, lateral: float, value: int = OCCUPIED) -> None:
    for f in forwards:
        _place(occ, ego, float(f), lateral, value=value)


def tight_pass() -> Scene:
    """0.4 m clearance to an unboxed wall at 43 km/h -> matches tight_clearance_at_speed only."""
    ego = _ego(12.0)
    occ = _blank()
    _wall(occ, ego, np.arange(2.0, 8.0, VOXEL), _WALL_LAT)
    return Scene((Frame(_grid(occ), ego, 0.0),), "tight_pass")


def slow_near() -> Scene:
    """Same 0.4 m wall but at 18 km/h -> fails the speed gate (precision control)."""
    ego = _ego(5.0)
    occ = _blank()
    _wall(occ, ego, np.arange(2.0, 8.0, VOXEL), _WALL_LAT)
    return Scene((Frame(_grid(occ), ego, 0.0),), "slow_near")


def open_road() -> Scene:
    """Empty corridor -> matches nothing (recall control)."""
    return Scene((Frame(_grid(_blank()), _ego(12.0), 0.0),), "open_road")


def narrowing_corridor() -> Scene:
    """Walls 0.8 m off each side -> free width ~1.4 m < ego width 1.85 m. Speed 18 km/h so the
    speed-gated tight_clearance query does not also fire. Matches corridor_narrows only."""
    ego = _ego(5.0)
    occ = _blank()
    _wall(occ, ego, np.arange(2.0, 9.0, VOXEL), +0.8)
    _wall(occ, ego, np.arange(2.0, 9.0, VOXEL), -0.8)
    return Scene((Frame(_grid(occ), ego, 0.0),), "narrowing_corridor")


def blocked_then_clears() -> Scene:
    """Frame 0: a one-sided wall blocks the swept path (but only the left half, so it is not a
    two-sided 'narrowing'). Frames 1-2: clear. Matches blocked_then_clears (a real transition)."""
    ego = _ego(8.0)
    occ0 = _blank()
    for lat in np.arange(0.0, 1.2, VOXEL):
        _place(occ0, ego, 4.0, float(lat))
    frames = (
        Frame(_grid(occ0), ego, 0.0),
        Frame(_grid(_blank()), ego, 0.5),
        Frame(_grid(_blank()), ego, 1.0),
    )
    return Scene(frames, "blocked_then_clears")


def unknown_side() -> Scene:
    """An UNKNOWN (unobserved) wall 0.4 m off the ego side at 43 km/h. tight_clearance FLIPS with
    the unknown policy (OCCUPIED -> match, FREE -> no match): the unknown-sensitivity demo. Not in
    any ground-truth set, because its membership is genuinely undetermined."""
    ego = _ego(12.0)
    occ = _blank()
    _wall(occ, ego, np.arange(2.0, 8.0, VOXEL), _WALL_LAT, value=UNKNOWN)
    return Scene((Frame(_grid(occ), ego, 0.0),), "unknown_side")


def near_vehicle() -> Scene:
    """A tracked vehicle box 1.5 m ahead, empty occupancy -> matches the tracking-baseline query
    near_a_tracked_vehicle only (the fairness control)."""
    ego = _ego(10.0)
    box = TrackedBox(center=(11.5, 10.0, 1.0), size=(4.5, 2.0, 1.8), yaw=0.0, label="vehicle")
    return Scene((Frame(_grid(_blank()), ego, 0.0, objects=(box,)),), "near_vehicle")


SCENES = [
    tight_pass(),
    slow_near(),
    open_road(),
    narrowing_corridor(),
    blocked_then_clears(),
    unknown_side(),
    near_vehicle(),
]

# Hand-labeled retrieval ground truth (scene names) per query id. unknown_side is deliberately
# absent: it is the unknown-sensitive scene whose membership depends on the policy, so it is
# scored as a false positive under OCCUPIED and excluded (undetermined) under IGNORED.
GROUND_TRUTH = {
    "tight_clearance_at_speed": {"tight_pass"},
    "corridor_narrows_below_vehicle_width": {"narrowing_corridor"},
    "blocked_then_clears": {"blocked_then_clears"},
    "near_a_tracked_vehicle": {"near_vehicle"},
}
