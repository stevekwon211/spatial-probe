# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Occ3D adapter: the occupancy mapping is unit-tested with no dataset, and a real-scene load smoke
runs only when the gated Occ3D-nuScenes data is present locally (data/ is gitignored)."""
import pathlib

import numpy as np
import pytest

from probe.adapters import occ3d
from probe.grid import FREE, OCCUPIED, UNKNOWN

_DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
_HAS_DATA = (_DATA / "annotations.json").exists() and (_DATA / "gts").exists()


def test_map_occupancy_encoding():
    # free class, ground-surface class, obstacle (car), and a masked-out (unobserved) voxel
    semantics = np.array([[[17, 11, 4, 4]]], dtype=np.uint8)  # free, driveable_surface, car, car
    mask = np.array([[[1, 1, 1, 0]]], dtype=np.uint8)         # last voxel is unobserved
    occ = occ3d.map_occupancy(semantics, mask)
    assert occ[0, 0, 0] == FREE       # 17 -> free
    assert occ[0, 0, 1] == FREE       # ground surface -> free (passable, not an obstacle)
    assert occ[0, 0, 2] == OCCUPIED   # car -> occupied
    assert occ[0, 0, 3] == UNKNOWN    # mask 0 overrides the semantic label -> unknown


def test_grid_spec_constants():
    assert occ3d.GRID_SHAPE == (200, 200, 16)
    assert occ3d.VOXEL_SIZE == 0.4
    assert occ3d.FREE_CLASS == 17


@pytest.mark.skipif(not _HAS_DATA, reason="Occ3D-nuScenes data not present (gated)")
def test_load_real_scene_smoke():
    scene = occ3d.load_scene("scene-0061", _DATA)
    assert len(scene) > 0
    grid = scene.grid_at(0)
    assert grid.occupancy.shape == occ3d.GRID_SHAPE
    assert scene.ego_speed(0) >= 0.0
    assert (grid.occupancy == OCCUPIED).sum() > 0  # a real driving scene has obstacles
