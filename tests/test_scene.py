# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Scene container behavior -- indexing, frame accessors, ego + object helpers."""
import numpy as np

from probe.grid import FREE, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox


def _grid() -> OccupancyGrid:
    return OccupancyGrid(np.full((4, 4, 4), FREE, dtype=int), voxel_size=1.0)


def test_scene_indexes_frames_and_reads_ego():
    scene = Scene(
        (
            Frame(_grid(), EgoPose((0, 0, 0), 0.0, speed=5.0), time=0.0),
            Frame(_grid(), EgoPose((5, 0, 0), 0.0, speed=7.0), time=0.5),
        ),
        name="two_frames",
    )
    assert len(scene) == 2
    assert scene.times() == (0, 1)
    assert scene.ego_speed(1) == 7.0
    assert scene.ego_width() == 1.85
    assert scene.ego_at(0).position == (0, 0, 0)


def test_scene_exposes_tracked_objects_per_frame():
    box = TrackedBox(center=(3, 0, 1), size=(4.5, 2.0, 1.8), yaw=0.0, label="vehicle")
    scene = Scene(
        (
            Frame(_grid(), EgoPose((0, 0, 0), 0.0), objects=(box,)),
            Frame(_grid(), EgoPose((0, 0, 0), 0.0)),  # no objects this frame
        ),
        name="boxed",
    )
    assert scene.objects_at(0) == (box,)
    assert scene.objects_at(1) == ()
    assert scene.objects_at(0)[0].label == "vehicle"
