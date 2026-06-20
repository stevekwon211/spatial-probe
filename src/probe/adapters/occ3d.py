# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Occ3D-nuScenes -> probe.Scene adapter (M2). NOT YET IMPLEMENTED -- requires the dataset.

This module is the CONTRACT for M2: the signature plus the label/coordinate mapping a real
implementation must satisfy, so the data-independent core (predicates, retrieval, metrics) wires to
real scenes by filling in `load_scene` and nothing else. Once it returns scenes,
`experiments/occquery_v0/run.py` works unchanged with `SCENES` sourced from here.

The dataset is gated behind a free nuScenes research account + terms, so this file does NOT download
anything. See docs/m2-adapter-contract.md for the expected layout, versions, integrity checks, the
first real-data smoke command, and the output schema.
"""
from __future__ import annotations

import pathlib

from probe.scene import Scene

# Occ3D-nuScenes voxel-grid spec. PROVISIONAL -- confirm against the official Occ3D-nuScenes release
# before implementing (commonly cited: 0.4 m voxels over [-40, 40] x [-40, 40] x [-1.0, 5.4] m ->
# 200 x 200 x 16, one semantic label per voxel + a per-voxel camera/lidar visibility mask).
OCC3D_VOXEL_SIZE_M = 0.4               # PROVISIONAL
OCC3D_GRID_SHAPE = (200, 200, 16)      # PROVISIONAL
OCC3D_RANGE_M = ((-40.0, 40.0), (-40.0, 40.0), (-1.0, 5.4))  # PROVISIONAL
OCC3D_FREE_LABEL = 17                  # PROVISIONAL -- confirm the free/empty label id


def load_scene(scene_token: str, data_root: pathlib.Path) -> Scene:
    """Load one nuScenes scene as a probe.Scene (one Frame per keyframe sample).

    Contract per frame (see docs/m2-adapter-contract.md):
    - OccupancyGrid.occupancy: Occ3D voxel labels mapped to OCCUPIED / FREE / UNKNOWN, where
      UNKNOWN = not observed per the visibility mask, FREE = the empty label, and any non-free,
      non-ground class = OCCUPIED. Ground classes are handled via OccupancyGrid.ground_height, not
      as obstacles.
    - OccupancyGrid.voxel_size / origin / ground_height: from the Occ3D grid spec + the ego frame.
    - EgoPose: position + yaw from the nuScenes ego pose; speed from consecutive sample timestamps.
    - objects: tuple[TrackedBox] from nuScenes sample_annotation (center, size, yaw, category,
      velocity).
    - time: sample timestamp in seconds.

    Raises NotImplementedError until M2; the dataset is gated and must be set up locally first.
    """
    raise NotImplementedError(
        "M2 adapter requires Occ3D-nuScenes data (gated behind a nuScenes account + terms). "
        "See docs/m2-adapter-contract.md for setup and the output schema."
    )
