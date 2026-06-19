# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""TDD for the core primitive — written BEFORE the implementation.

M1 = make these pass (then delete the xfail markers). Each case uses a tiny synthetic
grid with hand-known visibility, so the primitive is verified without any dataset.
"""
import numpy as np

from probe.raycast import FREE, OCCUPIED, UNKNOWN, line_of_sight, occlusion_depth, traverse


def _empty(n: int = 8) -> np.ndarray:
    return np.full((n, n, n), FREE, dtype=int)


def test_clear_path_is_visible():
    grid = _empty()
    assert line_of_sight(grid, (0, 0, 0), (7, 0, 0)) is True


def test_wall_blocks_line_of_sight():
    grid = _empty()
    grid[4, 0, 0] = OCCUPIED  # a wall cell strictly between src and dst
    assert line_of_sight(grid, (0, 0, 0), (7, 0, 0)) is False


def test_endpoints_excluded():
    # an occupied endpoint must NOT count as a blocker (we test cells strictly between)
    grid = _empty()
    grid[7, 0, 0] = OCCUPIED
    assert line_of_sight(grid, (0, 0, 0), (7, 0, 0)) is True


def test_unknown_blocks_only_when_asked():
    grid = _empty()
    grid[4, 0, 0] = UNKNOWN
    assert line_of_sight(grid, (0, 0, 0), (7, 0, 0), unknown_blocks=False) is True
    assert line_of_sight(grid, (0, 0, 0), (7, 0, 0), unknown_blocks=True) is False


def test_traverse_is_ordered_and_exclusive():
    cells = list(traverse((0, 0, 0), (3, 0, 0)))
    assert cells == [(1, 0, 0), (2, 0, 0)]  # strictly-between, in order


def test_occlusion_depth_counts_occluders():
    grid = _empty()
    grid[3, 0, 0] = OCCUPIED
    grid[5, 0, 0] = OCCUPIED
    assert occlusion_depth(grid, (0, 0, 0), (7, 0, 0)) == 2
