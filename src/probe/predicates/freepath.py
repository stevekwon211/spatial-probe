# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Free-path predicates over the ego's swept footprint.

`free_along_ego_path` asks whether the ego can move forward over a time horizon without its
body sweeping through an obstacle. `min_free_width_along_path` measures how wide the free
corridor stays along that path -- the quantity that decides whether the only gap ahead has
narrowed below the vehicle's width. Both are properties of empty SPACE, not of any object box,
so a box-only query language cannot express them. v0 models the sweep as a straight,
constant-velocity extrusion of the ego box along its heading; logged trajectories arrive with
the M2 adapter.
"""
from __future__ import annotations

import math

import numpy as np

from probe.grid import EgoPose, OccupancyGrid, UnknownPolicy


def free_along_ego_path(
    grid: OccupancyGrid,
    ego: EgoPose,
    horizon: float,
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
) -> bool:
    """True iff no obstacle lies in the ego swept footprint over `horizon` seconds.

    Footprint = forward in [0, length/2 + speed*horizon + voxel/2] and |lateral| <= width/2 +
    voxel/2 -- the ego body plus a straight constant-velocity reach, padded by a voxel
    half-extent so a voxel grazing the corridor edge counts as a hit. horizon=0 tests the body
    alone (is it blocked right now); a larger horizon extends the corridor by speed*horizon
    meters. Ground and over-height voxels are excluded via `obstacle_centers`.
    """
    centers = grid.obstacle_centers(unknown_policy=unknown_policy, max_height_agl=ego.height)
    if len(centers) == 0:
        return True
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    reach = ego.length / 2.0 + ego.speed * horizon + grid.voxel_size / 2.0
    half_w = ego.width / 2.0 + grid.voxel_size / 2.0
    hit = (forward >= 0.0) & (forward <= reach) & (np.abs(lateral) <= half_w)
    return not bool(hit.any())


def min_free_width_along_path(
    grid: OccupancyGrid,
    ego: EgoPose,
    horizon: float,
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
) -> float:
    """Minimum free corridor width (m) across longitudinal stations along the ego path.

    At each station s in [0, length/2 + speed*horizon] (stepped by one voxel), find the nearest
    obstacle surface to the left and to the right of the centerline; free_width = left_extent +
    right_extent, each measured centerline-to-inner-surface (voxel center minus a voxel
    half-extent). Returns the minimum over stations where BOTH sides are bounded; math.inf if
    the path is never laterally bounded on both sides (an open or one-sided edge does not define
    a corridor that can 'narrow'). Ground and over-height voxels excluded.
    """
    centers = grid.obstacle_centers(unknown_policy=unknown_policy, max_height_agl=ego.height)
    reach = ego.length / 2.0 + ego.speed * horizon
    if len(centers) == 0:
        return math.inf
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    half = grid.voxel_size / 2.0
    step = grid.voxel_size
    widths: list[float] = []
    k = 0
    while k * step <= reach + 1e-9:
        s = k * step
        k += 1
        near = np.abs(forward - s) <= half
        if not near.any():
            continue
        lat = lateral[near]
        left = lat[lat > 0]
        right = lat[lat < 0]
        if left.size and right.size:
            widths.append((float(left.min()) - half) + (-float(right.max()) - half))
    return float(min(widths)) if widths else math.inf
