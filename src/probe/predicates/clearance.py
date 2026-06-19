# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""lateral_clearance: the headline OccQuery predicate.

Min horizontal distance from the ego's forward corridor to the nearest occupied,
non-ground voxel. It answers "how close did the ego come to *anything solid*", including
obstacles with no object box (walls, debris, curbs, protruding loads) — exactly what a
box-only query language (RefAV's cuboid functions) cannot express.
"""
from __future__ import annotations

import math

import numpy as np

from probe.grid import EgoPose, OccupancyGrid


def lateral_clearance(grid: OccupancyGrid, ego: EgoPose, *, lookahead: float = 20.0) -> float:
    """Meters to the nearest non-ground obstacle within `lookahead` ahead of the ego.

    Considers only voxels in front of the ego (forward in [0, lookahead]) and returns the
    smallest |lateral| offset among them; math.inf if the corridor ahead is clear. The ego
    half-width is NOT subtracted in v0 — clearance is measured to the ego centerline, so
    subtract width/2 when comparing against a physical gap.
    """
    centers = grid.nonground_occupied_centers()
    if len(centers) == 0:
        return math.inf
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    ahead = (forward >= 0.0) & (forward <= lookahead)
    if not ahead.any():
        return math.inf
    return float(np.min(np.abs(lateral[ahead])))
