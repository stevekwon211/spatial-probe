# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""OccupancyGrid + EgoPose: the spatial substrate the predicates run on.

An OccupancyGrid is a dense 3D voxel field (OCCUPIED / FREE / UNKNOWN) plus the
world<->voxel calibration (voxel size + origin). EgoPose is the vehicle's pose at one
instant. Both are deliberately minimal value objects: the dataset adapter (M2) fills them
from Occ3D-nuScenes, and tests fill them synthetically. The occupancy encoding constants
live with the primitive in `probe.raycast` and are re-exported here for convenience.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from probe.raycast import FREE, OCCUPIED, UNKNOWN, Cell

__all__ = ["OccupancyGrid", "EgoPose", "FREE", "OCCUPIED", "UNKNOWN"]


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """A dense voxel occupancy field with its world calibration.

    occupancy: (X, Y, Z) int array of OCCUPIED / FREE / UNKNOWN.
    voxel_size: meters per voxel edge.
    origin: world coordinate of the center of voxel (0, 0, 0).
    ground_height: world z at/below which an OCCUPIED voxel is treated as ground (the
        drivable surface) and excluded from "non-ground" obstacle queries.
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

    def nonground_occupied_centers(self) -> np.ndarray:
        """World-space centers (M, 3) of every OCCUPIED voxel above `ground_height`.

        Ground voxels are excluded because clearance / free-path care about obstacles, not
        the road surface. Vectorized: argwhere -> affine to world -> z filter.
        """
        idx = np.argwhere(self.occupancy == OCCUPIED)
        if idx.size == 0:
            return np.empty((0, 3), dtype=float)
        centers = np.asarray(self.origin, dtype=float) + idx * self.voxel_size
        return centers[centers[:, 2] > self.ground_height]


@dataclass(frozen=True)
class EgoPose:
    """The ego vehicle's pose at one instant. Heading is yaw in radians, 0 == +x, CCW."""

    position: tuple[float, float, float]
    heading: float
    speed: float = 0.0  # m/s
    width: float = 1.85  # m   (nuScenes ego vehicle ~1.85 x 4.6)
    length: float = 4.6  # m

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
