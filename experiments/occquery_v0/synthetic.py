# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Synthetic, hand-labeled scenes for occquery_v0 — exercise the retrieval loop with NO
dataset. Each scene's geometry is built so its ground-truth query membership is known by
construction, letting run.py measure denotation P/R/F1 before nuScenes is available.

At M2 this module is replaced by the Occ3D-nuScenes adapter: same `Scene` type, real
geometry, GT from hand-labeling instead of construction.
"""
from __future__ import annotations

import numpy as np

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene

VOXEL = 0.2          # m per voxel edge (fine enough to express sub-0.5 m clearance)
GRID_N = 100         # 20 m cube
ORIGIN = (0.0, 0.0, 0.0)
GROUND_H = 0.5       # world z at/below which OCCUPIED is the road surface
_EGO_XY = (10.0, 10.0)


def _blank() -> np.ndarray:
    return np.full((GRID_N, GRID_N, GRID_N), FREE, dtype=int)


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, VOXEL, ORIGIN, GROUND_H)


def _occupy_ahead(occ: np.ndarray, ego: EgoPose, *, forward: float, lateral: float, height: float = 1.0) -> None:
    """Set one OCCUPIED voxel at (forward, lateral, height) in the ego frame."""
    fx, fy = ego.forward
    lx, ly = ego.left
    wx = ego.position[0] + forward * fx + lateral * lx
    wy = ego.position[1] + forward * fy + lateral * ly
    i = round((wx - ORIGIN[0]) / VOXEL)
    j = round((wy - ORIGIN[1]) / VOXEL)
    k = round((height - ORIGIN[2]) / VOXEL)
    occ[i, j, k] = OCCUPIED


def tight_pass() -> Scene:
    """0.4 m clearance at 43 km/h -> matches tight_clearance_at_speed."""
    ego = EgoPose((*_EGO_XY, 0.0), 0.0, speed=12.0)
    occ = _blank()
    _occupy_ahead(occ, ego, forward=5.0, lateral=0.4)
    return Scene((Frame(_grid(occ), ego, 0.0),), "tight_pass")


def slow_near() -> Scene:
    """Same 0.4 m clearance but at 18 km/h -> fails the speed gate (precision control)."""
    ego = EgoPose((*_EGO_XY, 0.0), 0.0, speed=5.0)
    occ = _blank()
    _occupy_ahead(occ, ego, forward=5.0, lateral=0.4)
    return Scene((Frame(_grid(occ), ego, 0.0),), "slow_near")


def open_road() -> Scene:
    """Empty corridor -> clearance is infinite, matches nothing (recall control)."""
    ego = EgoPose((*_EGO_XY, 0.0), 0.0, speed=12.0)
    return Scene((Frame(_grid(_blank()), ego, 0.0),), "open_road")


SCENES = [tight_pass(), slow_near(), open_road()]

# Hand-labeled retrieval ground truth (scene names) per query id in queries.yaml.
# Only queries whose predicates exist in the v0 core are labeled; the rest are reported
# as SKIP by run.py until their predicate ships.
GROUND_TRUTH = {
    "tight_clearance_at_speed": {"tight_pass"},
}
