# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Lateral-clearance predicates: how close the ego comes to anything solid (including
obstacles with no object box -- walls, debris, curbs, protruding loads), which is exactly
what a box-only query language cannot express.

Two distinct quantities, named honestly:

- `centerline_lateral_distance` -- raw geometric distance from the ego centerline to the
  nearest obstacle voxel CENTER. A measurement, not a physical gap.
- `lateral_clearance` -- the physical free gap between the ego BODY SIDE and the obstacle
  SURFACE: centerline distance minus the ego half-width minus the voxel half-extent, floored
  at 0 (0 means the footprints overlap). This is the one a planner cares about.
"""
from __future__ import annotations

import math

import numpy as np

from probe.grid import EgoPose, OccupancyGrid, UnknownPolicy


def centerline_lateral_distance(
    grid: OccupancyGrid,
    ego: EgoPose,
    *,
    lookahead: float = 20.0,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
) -> float:
    """Min |lateral| from the ego centerline to the nearest obstacle voxel center ahead.

    Considers obstacle voxels (see `OccupancyGrid.obstacle_centers`, restricted to the ego
    height band) with forward in [0, lookahead]; returns math.inf if the corridor ahead is
    clear. This is a centerline-to-center distance, NOT a physical clearance -- use
    `lateral_clearance` for that.
    """
    centers = grid.obstacle_centers(unknown_policy=unknown_policy, max_height_agl=ego.height)
    if len(centers) == 0:
        return math.inf
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    ahead = (forward >= 0.0) & (forward <= lookahead)
    if not ahead.any():
        return math.inf
    return float(np.min(np.abs(lateral[ahead])))


def lateral_clearance(
    grid: OccupancyGrid,
    ego: EgoPose,
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
) -> float:
    """Physical lateral free gap (m) to the nearest obstacle ABEAM the ego body, v1.

    Only obstacles beside the ego body -- |forward| <= ego.length/2 and |lateral| beyond the
    half-width -- count; the gap is |lateral| - half-width - voxel-half. An obstacle dead ahead
    (inside the corridor) is a longitudinal blockage that `free_along_ego_path` handles, and an
    obstacle far ahead is not a side clearance at all. math.inf if nothing is abeam.

    v1 restricts to the abeam band. v0 took the minimum side gap over the whole 0-20 m forward
    window, so a wall far ahead where the road bends read as a gap beside the ego (the
    'frontal-edge-as-side-clearance' false positive on real Occ3D data 2026-06-20).
    """
    centers = grid.obstacle_centers(unknown_policy=unknown_policy, max_height_agl=ego.height)
    if len(centers) == 0:
        return math.inf
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    half = ego.width / 2.0 + grid.voxel_size / 2.0
    abeam = ego.length / 2.0
    beside = (np.abs(forward) <= abeam) & (np.abs(lateral) > half)
    if not beside.any():
        return math.inf
    return float(np.min(np.abs(lateral[beside]) - half))
