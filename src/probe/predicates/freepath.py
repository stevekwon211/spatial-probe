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
from probe.predicates.reachable import reachable_free_field


def free_along_ego_path(
    grid: OccupancyGrid,
    ego: EgoPose,
    horizon: float,
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
    min_cluster_voxels: int = 1,
) -> bool:
    """True iff the ego can move straight forward over `horizon` seconds without its body sweeping
    through an obstacle, v1 (C-space reachability).

    The ego centerline must stay reachable from the ego footprint out to a constant-velocity
    forward reach (length/2 + speed*horizon). Obstacles are inflated by the ego half-width, so a
    clear centerline means the whole body clears -- equivalent to the v0 swept-box test, but on the
    same reachability substrate as the corridor/clearance predicates. horizon=0 tests the body
    alone. `min_cluster_voxels` > 1 drops lone-voxel noise (kills single-voxel-noise-as-blocked);
    set it at the query layer for real data. Temporal persistence / relative motion are dynfield's
    job, not this single-frame predicate.
    """
    f = reachable_free_field(
        grid, ego, horizon, unknown_policy=unknown_policy, min_cluster_voxels=min_cluster_voxels
    )
    res = f.resolution
    fi0, li0 = f.ego_cell
    reach = ego.length / 2.0 + ego.speed * horizon
    reach_fi = int(round((reach - f.forward_min) / res))
    for fi in range(fi0, min(reach_fi, f.obstacle.shape[0] - 1) + 1):
        if f.obstacle[fi, li0] or not f.reachable[fi, li0]:
            return False  # centerline blocked within reach
    return True


def min_free_width_along_path(
    grid: OccupancyGrid,
    ego: EgoPose,
    horizon: float,
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
    min_cluster_voxels: int = 1,
) -> float:
    """Minimum free corridor width (m) along the ego's straight-ahead path, v1 (C-space).

    Walks the centerline forward one voxel at a time. At each station the centerline cell must be
    free AND reachable from the ego footprint; the first cell that is itself an obstacle or is
    unreachable is a frontal blockage and stops the walk (nothing past it is a corridor the ego
    drives THROUGH -- it is a thing to stop for or go around). Where the walk continues, the
    width is the surface-to-surface gap between the nearest obstacle to the left and to the right
    of the centerline. Returns the minimum over stations bounded on BOTH sides; math.inf if the
    path is never two-sided (an open / one-sided edge is not a corridor that can 'narrow').
    Sub-voxel gaps clamp to 0 (blocked).

    This replaces the v0 version that paired the nearest left/right voxel at every station with no
    reachability test -- the source of the frontal-object-as-corridor and far-wall false
    positives on real data. (A single isolated-voxel 'wall' is a separate noise concern for the
    persistence layer, not here.)
    """
    f = reachable_free_field(
        grid, ego, horizon, unknown_policy=unknown_policy, min_cluster_voxels=min_cluster_voxels
    )
    res = f.resolution
    fi0, li0 = f.ego_cell
    reach = ego.length / 2.0 + ego.speed * horizon
    reach_fi = int(round((reach - f.forward_min) / res))
    obst = f.obstacle
    widths: list[float] = []
    for fi in range(fi0, min(reach_fi, obst.shape[0] - 1) + 1):
        if obst[fi, li0]:
            break  # a raw obstacle ON the centerline is a frontal blockage -- stop. An
            # inflated-only narrowing (walls closer than the ego half-width) is NOT an obstacle on
            # the centerline; it stays a corridor we keep measuring (that IS 'narrows below width').
        row = obst[fi]
        left = np.argwhere(row[li0 + 1:]).ravel()  # nearest obstacle left of the centerline
        right = np.argwhere(row[:li0]).ravel()      # nearest obstacle right of the centerline
        if left.size and right.size:
            left_li = li0 + 1 + int(left[0])
            right_li = int(right[-1])
            width = (left_li - right_li) * res - res  # surface-to-surface free gap
            widths.append(max(0.0, width))
    if not widths:
        return math.inf
    narrowest = min(widths)
    return 0.0 if narrowest < res else float(narrowest)
