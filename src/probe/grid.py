# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""OccupancyGrid + EgoPose + UnknownPolicy: the spatial substrate the predicates run on.

An OccupancyGrid is a dense 3D voxel field (OCCUPIED / FREE / UNKNOWN) plus its world
calibration (voxel size + origin). EgoPose is the vehicle pose at one instant. UnknownPolicy
is how a predicate treats unobserved voxels -- the single most important validity knob for
occupancy queries. The dataset adapter (M2) fills these from Occ3D-nuScenes; tests fill them
synthetically. Occupancy encoding constants live with the primitive in `probe.raycast`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from probe.raycast import FREE, OCCUPIED, UNKNOWN, Cell

__all__ = ["OccupancyGrid", "EgoPose", "UnknownPolicy", "FREE", "OCCUPIED", "UNKNOWN"]


class UnknownPolicy(Enum):
    """How a predicate treats UNKNOWN (unobserved) voxels.

    PLAN section 4 requires reporting denotation sensitivity across all three: the result of
    an occupancy query can flip depending on whether the unseen is assumed solid or empty, and
    that flip rate IS the validity test. An enum (not boolean flags) makes an invalid
    combination unrepresentable.
    """

    OCCUPIED = "occupied"  # unknown blocks: conservative (assume the unseen is solid)
    FREE = "free"          # unknown is passable: optimistic (assume the unseen is empty)
    IGNORED = "ignored"    # value computed as FREE; the experiment layer drops any (scene,
    #                        frame) whose verdict would flip under OCCUPIED as "undetermined".


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """A dense voxel occupancy field with its world calibration.

    occupancy: (X, Y, Z) int array of OCCUPIED / FREE / UNKNOWN.
    voxel_size: meters per voxel edge.
    origin: world coordinate of the center of voxel (0, 0, 0).
    ground_height: world z at/below which an OCCUPIED voxel is the road surface (excluded as a
        non-ground obstacle).
    """

    occupancy: np.ndarray
    voxel_size: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_height: float = 0.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.occupancy.shape  # type: ignore[return-value]

    def world_to_voxel(self, point) -> Cell:
        o = np.asarray(self.origin, dtype=float)
        i, j, k = np.round((np.asarray(point, dtype=float) - o) / self.voxel_size).astype(int)
        return int(i), int(j), int(k)

    def voxel_to_world(self, cell: Cell) -> tuple[float, float, float]:
        o = np.asarray(self.origin, dtype=float)
        x, y, z = o + np.asarray(cell, dtype=float) * self.voxel_size
        return float(x), float(y), float(z)

    def obstacle_centers(
        self,
        *,
        unknown_policy: UnknownPolicy = UnknownPolicy.FREE,
        max_height_agl: float | None = None,
    ) -> np.ndarray:
        """World-space centers (M, 3) of obstacle voxels.

        An obstacle is an OCCUPIED voxel above `ground_height` -- plus UNKNOWN voxels when
        `unknown_policy` is OCCUPIED (the conservative reading). `max_height_agl` (meters above
        ground) caps the vertical band, excluding voxels above the querying vehicle's envelope
        (e.g. a high overhang the ego passes safely under). Vectorized: argwhere -> world -> z
        filter.
        """
        occ = self.occupancy
        if unknown_policy is UnknownPolicy.OCCUPIED:
            mask = (occ == OCCUPIED) | (occ == UNKNOWN)
        else:
            mask = occ == OCCUPIED
        idx = np.argwhere(mask)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        centers = np.asarray(self.origin, dtype=float) + idx * self.voxel_size
        z = centers[:, 2]
        keep = z > self.ground_height
        if max_height_agl is not None:
            keep = keep & (z <= self.ground_height + max_height_agl)
        return centers[keep]


@dataclass(frozen=True)
class EgoPose:
    """The ego vehicle pose at one instant. Heading is yaw in radians, 0 == +x, CCW.

    width / length / height are the vehicle's physical extents in meters (nuScenes ego is
    ~1.85 x 4.6 x 1.9), used by the predicates to turn voxel-center geometry into a physical
    free gap.
    """

    position: tuple[float, float, float]
    heading: float
    speed: float = 0.0  # m/s
    width: float = 1.85
    length: float = 4.6
    height: float = 1.9

    @property
    def forward(self) -> tuple[float, float]:
        return math.cos(self.heading), math.sin(self.heading)

    @property
    def left(self) -> tuple[float, float]:
        return -math.sin(self.heading), math.cos(self.heading)

    def to_ego_frame(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world XY points (N, 2) into ego (forward, lateral) coordinates in meters.

        forward = signed distance ahead of the ego; lateral = signed distance to the ego's
        left. The predicates reason entirely in this frame.
        """
        d = np.asarray(points_xy, dtype=float) - np.asarray(self.position[:2], dtype=float)
        fx, fy = self.forward
        lx, ly = self.left
        return d[:, 0] * fx + d[:, 1] * fy, d[:, 0] * lx + d[:, 1] * ly
