# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""PRISM validate -- a good IR passes; corrupt calibration/timestamps RAISE a clear error.

Covers the honest-instrument failures: non-monotonic timestamps, a missing/undeclared coordinate
frame, NaN poses, sensor-vs-frame timestamp disagreement, broken parent chains.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from probe.grid import FREE, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox
from prism.ir import (
    CoordinateFrame,
    Entity,
    GroundTruth,
    Observation,
    Pose,
    Provenance,
    SceneIR,
)
from prism.validate import ValidationError, validate_scene


def _grid() -> OccupancyGrid:
    return OccupancyGrid(np.full((3, 3, 3), FREE, dtype=int), voxel_size=1.0)


def _good_ir(times=(0.0, 0.5, 1.0)) -> SceneIR:
    frames = tuple(Frame(_grid(), EgoPose((0, 0, 0), 0.0, speed=1.0), time=t) for t in times)
    scene = Scene(frames, name="good")
    cfs = (
        CoordinateFrame("world", None, Pose((0.0, 0.0, 0.0))),
        CoordinateFrame("ego", "world", Pose((1.0, 2.0, 0.0))),
        CoordinateFrame("lidar", "ego", Pose((0.0, 0.0, 0.0))),
    )
    obs = tuple(Observation("lidar", t, i) for i, t in enumerate(times))
    return SceneIR(scene=scene, coordinate_frames=cfs, observations=obs,
                   provenance=Provenance(dataset="synthetic", log_id="good"))


def test_good_ir_passes_and_returns_itself():
    ir = _good_ir()
    assert validate_scene(ir) is ir


def test_non_monotonic_timestamps_raise():
    ir = _good_ir(times=(0.0, 0.5, 0.4))  # third stamp goes backwards
    with pytest.raises(ValidationError, match="strictly increasing"):
        validate_scene(ir)


def test_duplicate_timestamps_raise():
    ir = _good_ir(times=(0.0, 0.5, 0.5))
    with pytest.raises(ValidationError, match="strictly increasing"):
        validate_scene(ir)


def test_nan_ego_pose_raises():
    frames = (
        Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.0),
        Frame(_grid(), EgoPose((float("nan"), 0, 0), 0.0), time=0.5),
    )
    ir = SceneIR(scene=Scene(frames, "bad"),
                 coordinate_frames=(CoordinateFrame("ego", None, Pose((0.0, 0.0, 0.0))),))
    with pytest.raises(ValidationError, match="non-finite"):
        validate_scene(ir)


def test_nan_box_center_raises():
    box = TrackedBox(center=(float("nan"), 0.0, 0.0), size=(1, 1, 1), yaw=0.0, label="vehicle")
    frames = (Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.0, objects=(box,)),
              Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.5))
    ir = SceneIR(scene=Scene(frames, "bad"))
    with pytest.raises(ValidationError, match="non-finite"):
        validate_scene(ir)


def test_missing_referenced_coordinate_frame_raises():
    """An entity declares it lives in frame 'lidar' but only 'world'/'ego' are declared."""
    box = TrackedBox(center=(1.0, 0.0, 0.0), size=(1, 1, 1), yaw=0.0, label="vehicle")
    frames = (Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.0, objects=(box,)),
              Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.5))
    ent = Entity.from_tracked_box(box, "vehicle#0", frame="lidar")  # 'lidar' not declared below
    ir = SceneIR(
        scene=Scene(frames, "bad"),
        coordinate_frames=(CoordinateFrame("world", None, Pose((0.0, 0.0, 0.0))),
                           CoordinateFrame("ego", "world", Pose((0.0, 0.0, 0.0)))),
        ground_truth=(GroundTruth(0, (ent,)),),
    )
    with pytest.raises(ValidationError, match="lidar"):
        validate_scene(ir)


def test_unknown_parent_frame_raises():
    ir = SceneIR(
        scene=Scene((Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.0),
                     Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.5)), "bad"),
        coordinate_frames=(CoordinateFrame("ego", "world", Pose((0.0, 0.0, 0.0))),),  # 'world' undeclared
    )
    with pytest.raises(ValidationError, match="unknown parent"):
        validate_scene(ir)


def test_sensor_timestamp_mismatch_raises():
    ir = _good_ir()
    bad_obs = (Observation("lidar", 0.0, 0), Observation("camera", 99.0, 1), Observation("lidar", 1.0, 2))
    ir = replace(ir, observations=bad_obs)  # camera says t=99 for a frame stamped 0.5
    with pytest.raises(ValidationError, match="mismatches"):
        validate_scene(ir)


def test_empty_scene_raises():
    ir = SceneIR(scene=Scene((), "empty"))
    with pytest.raises(ValidationError, match="no frames"):
        validate_scene(ir)


def test_collect_gathers_multiple_errors():
    frames = (Frame(_grid(), EgoPose((float("nan"), 0, 0), 0.0), time=0.0),
              Frame(_grid(), EgoPose((0, 0, 0), 0.0), time=0.0))  # NaN pose AND duplicate timestamp
    ir = SceneIR(scene=Scene(frames, "bad"))
    with pytest.raises(ValidationError) as exc:
        validate_scene(ir, collect=True)
    msg = str(exc.value)
    assert "non-finite" in msg and "strictly increasing" in msg
