# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Core primitive: DDA line-of-sight / ray traversal over a voxel occupancy grid.

This is the load-bearing operation for the whole instrument. The occlusion paper's
visibility result, OccQuery's clearance/free-path predicates, and the gt-distrust
occlusion-depth score all reduce to "march a ray cell-by-cell and test what it hits".

Implementation uses 3D DDA (Amanatides & Woo, 1987) over an integer voxel grid.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

# Occupancy encoding shared across the package (see probe.grid / probe.adapters).
OCCUPIED = 1
FREE = 0
UNKNOWN = -1

Cell = tuple[int, int, int]


def traverse(src: Cell, dst: Cell) -> Iterator[Cell]:
    """Yield the voxel cells a ray from `src` to `dst` passes through, in order,
    EXCLUSIVE of both endpoints (the cells strictly between).

    Integer-grid 3D DDA (Amanatides & Woo): the ray leaves the center of the `src`
    voxel toward the center of `dst`, stepping one axis at a time across whichever voxel
    boundary it reaches first. Deterministic and order-stable.
    """
    x, y, z = int(src[0]), int(src[1]), int(src[2])
    x1, y1, z1 = int(dst[0]), int(dst[1]), int(dst[2])
    if (x, y, z) == (x1, y1, z1):
        return
    dx, dy, dz = x1 - x, y1 - y, z1 - z
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    sz = (dz > 0) - (dz < 0)
    inf = float("inf")
    # t-distance to cross one full voxel per axis, and to reach the first boundary
    # (the ray starts at a voxel center, so the first boundary is half a voxel away).
    t_delta_x = 1.0 / abs(dx) if dx else inf
    t_delta_y = 1.0 / abs(dy) if dy else inf
    t_delta_z = 1.0 / abs(dz) if dz else inf
    t_max_x = 0.5 / abs(dx) if dx else inf
    t_max_y = 0.5 / abs(dy) if dy else inf
    t_max_z = 0.5 / abs(dz) if dz else inf
    while True:
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            x += sx
            t_max_x += t_delta_x
        elif t_max_y <= t_max_z:
            y += sy
            t_max_y += t_delta_y
        else:
            z += sz
            t_max_z += t_delta_z
        if (x, y, z) == (x1, y1, z1):
            return
        yield (x, y, z)


def line_of_sight(
    grid: np.ndarray,
    src: Cell,
    dst: Cell,
    *,
    occupied_value: int = OCCUPIED,
    unknown_blocks: bool = False,
) -> bool:
    """Return True iff no blocking voxel lies strictly between `src` and `dst`.

    grid: 3D int array of OCCUPIED/FREE/UNKNOWN.
    occupied_value: the value treated as solid.
    unknown_blocks: if True, UNKNOWN cells also block (one of the v0 "unobserved-voxel"
        handling rules; the experiment reports sensitivity to this choice).

    A target is *occluded* iff `line_of_sight(...) is False`.
    """
    for x, y, z in traverse(src, dst):
        v = int(grid[x, y, z])
        if v == occupied_value:
            return False
        if unknown_blocks and v == UNKNOWN:
            return False
    return True


def occlusion_depth(grid: np.ndarray, src: Cell, dst: Cell) -> int:
    """Number of OCCUPIED cells strictly between `src` and `dst`.

    0 == directly visible. Used later by gt-distrust as the per-voxel "this label is
    only supported through an occluder, so distrust it" score. Shares the traversal;
    not used by OccQuery v0.
    """
    n = 0
    for x, y, z in traverse(src, dst):
        if int(grid[x, y, z]) == OCCUPIED:
            n += 1
    return n
