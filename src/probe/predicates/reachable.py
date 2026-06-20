# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""C-space reachable free-space -- the v1 substrate under the free-space predicates.

v0 predicates reasoned over the nearest obstacle voxels on the ego's straight centerline, with
no notion of whether the ego can actually REACH a gap. That single shortcut produced every one
of the five false-positive modes found on real Occ3D data (a frontal object read as a two-sided
corridor, a far wall read as a side clearance, a lone voxel read as a blockage). This module
replaces it with the textbook construct the frontier uses: configuration-space free space.

Pipeline (single frame, ego frame, top-down):
  1. rasterize obstacle voxels (ego height band) into a 2D BEV grid in ego (forward, lateral);
  2. Minkowski-inflate obstacles by the ego half-width (a cell within half-width of any obstacle
     is C-obstacle -- the ego center cannot sit there);
  3. flood-fill the free region 8-connected from the ego footprint -> the REACHABLE region (a
     gap the ego cannot reach, e.g. behind a frontal object, is excluded);
  4. a Euclidean distance transform gives, at every reachable cell, the physical gap (m) to the
     nearest obstacle surface.

The predicates then read geometry off this field: a corridor is a narrowing of the reachable
channel, a side clearance is the gap beside the ego within the reachable region, a blockage is
the centerline leaving the reachable region. Noise/dynamics (cluster size, persistence) are the
NEXT layer (a Bayes occupancy filter); this layer is single-frame geometry only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from probe.grid import EgoPose, OccupancyGrid, UnknownPolicy

_LATERAL_HALF_WINDOW = 20.0  # meters kept each side of the ego centerline in the BEV


@dataclass(frozen=True)
class ReachableField:
    """The reachable free region around the ego, in an ego-frame BEV grid.

    reachable: (NF, NL) bool -- free cells 8-connected to the ego footprint.
    clearance: (NF, NL) float meters -- centerline-to-nearest-obstacle-CENTER distance at each
        cell, +inf where unreachable (so a min over unreachable cells never wins). Physical
        surface gaps subtract a voxel half-extent at the predicate layer, matching v0.
    obstacle: (NF, NL) bool -- the raw (un-inflated) obstacle BEV, so a predicate can separate
        the nearest LEFT vs RIGHT obstacle to recover a physical corridor width.
    forward_min / lateral_min: ego-frame coord (m) of cell index 0 on each axis.
    resolution: meters per cell (= grid voxel size).
    ego_cell: (fi, li) index of the ego footprint center.
    """

    reachable: np.ndarray
    clearance: np.ndarray
    obstacle: np.ndarray
    forward_min: float
    lateral_min: float
    resolution: float
    ego_cell: tuple[int, int]

    def _index(self, forward: float, lateral: float) -> tuple[int, int] | None:
        fi = int(round((forward - self.forward_min) / self.resolution))
        li = int(round((lateral - self.lateral_min) / self.resolution))
        if 0 <= fi < self.reachable.shape[0] and 0 <= li < self.reachable.shape[1]:
            return fi, li
        return None

    def is_reachable(self, forward: float, lateral: float) -> bool:
        ix = self._index(forward, lateral)
        return bool(self.reachable[ix]) if ix is not None else False

    def clearance_at(self, forward: float, lateral: float) -> float:
        ix = self._index(forward, lateral)
        return float(self.clearance[ix]) if ix is not None else float("inf")


def _drop_small_clusters(obstacle: np.ndarray, min_size: int) -> np.ndarray:
    """Drop 8-connected obstacle components smaller than `min_size` voxels -- a lone sensor-noise
    voxel is not an obstacle. Single-frame geometry only; temporal persistence is dynfield's job."""
    labels, n = ndimage.label(obstacle, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return obstacle
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False  # background label 0 is never an obstacle
    return keep[labels]


def reachable_free_field(
    grid: OccupancyGrid,
    ego: EgoPose,
    horizon: float,
    *,
    unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
    margin: float = 0.0,
    min_cluster_voxels: int = 1,
) -> ReachableField:
    """Build the ego's reachable free-space field over a forward horizon.

    `margin` widens the C-space inflation beyond the ego half-width (a safety buffer).
    `min_cluster_voxels` > 1 drops obstacle clusters smaller than that many voxels as noise (a lone
    voxel is not an obstacle). The BEV spans the ego body plus a constant-velocity forward reach,
    and +/-`_LATERAL_HALF_WINDOW` laterally. Unknown voxels follow `unknown_policy` as v0 did.
    """
    res = grid.voxel_size
    reach = ego.length / 2.0 + ego.speed * horizon
    f_min = -ego.length / 2.0 - res
    f_max = reach + res
    l_min, l_max = -_LATERAL_HALF_WINDOW, _LATERAL_HALF_WINDOW
    nf = int(np.ceil((f_max - f_min) / res)) + 1
    nl = int(np.ceil((l_max - l_min) / res)) + 1

    obstacle = np.zeros((nf, nl), dtype=bool)
    centers = grid.obstacle_centers(unknown_policy=unknown_policy, max_height_agl=ego.height)
    if len(centers):
        fwd, lat = ego.to_ego_frame(centers[:, :2])
        fi = np.round((fwd - f_min) / res).astype(int)
        li = np.round((lat - l_min) / res).astype(int)
        inb = (fi >= 0) & (fi < nf) & (li >= 0) & (li < nl)
        obstacle[fi[inb], li[inb]] = True
    if min_cluster_voxels > 1:
        obstacle = _drop_small_clusters(obstacle, min_cluster_voxels)

    # centerline-to-center distance (m) from every cell to the nearest obstacle voxel center
    raw_dist = ndimage.distance_transform_edt(~obstacle) * res
    # C-space: a cell within (half-width + margin) of an obstacle cannot hold the ego center
    free = raw_dist > (ego.width / 2.0 + margin)

    fi0 = int(round((0.0 - f_min) / res))
    li0 = int(round((0.0 - l_min) / res))
    if free[fi0, li0]:
        labels, _ = ndimage.label(free, structure=np.ones((3, 3), dtype=int))  # 8-connected
        reachable = labels == labels[fi0, li0]
    else:
        reachable = np.zeros_like(free)  # ego footprint itself is in collision
    clearance = np.where(reachable, raw_dist, np.inf)
    return ReachableField(reachable, clearance, obstacle, f_min, l_min, res, (fi0, li0))
