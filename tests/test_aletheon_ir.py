# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Aletheon Scene IR -- object invariants, probe<->IR lifting, calibration frame transforms.

Builds an IR two ways: (1) a synthetic minimal IR (always runs, deterministic), and (2) from a
real AV2-Sensor log via the adapter when present (skip-if-missing). The AV2 path proves the
CoordinateFrame is built from the REAL city_SE3_egovehicle calibration, not faked.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from probe.grid import FREE, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox
from aletheon.ir import (
    SCHEMA_VERSION,
    CoordinateFrame,
    Entity,
    Pose,
    Provenance,
    SceneIR,
    Track,
)

_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")


def _grid() -> OccupancyGrid:
    return OccupancyGrid(np.full((3, 3, 3), FREE, dtype=int), voxel_size=1.0)


def _minimal_ir() -> SceneIR:
    box = TrackedBox(center=(3.0, 0.0, 1.0), size=(4.5, 2.0, 1.8), yaw=0.5, label="vehicle", velocity=(1.0, 0.0))
    scene = Scene(
        (
            Frame(_grid(), EgoPose((0, 0, 0), 0.0, speed=5.0), time=0.0, objects=(box,)),
            Frame(_grid(), EgoPose((5, 0, 0), 0.0, speed=6.0), time=0.5),
        ),
        name="mini",
    )
    return SceneIR(
        scene=scene,
        coordinate_frames=(
            CoordinateFrame("world", None, Pose((0.0, 0.0, 0.0))),
            CoordinateFrame("ego", "world", Pose.from_yaw((10.0, 20.0, 0.0), math.pi / 2)),
            CoordinateFrame("lidar", "ego", Pose((0.0, 0.0, 0.0))),
        ),
        tracks=(
            Track(
                entity_id="vehicle#0",
                category="vehicle",
                timestamps=(0.0,),
                states=(Entity.from_tracked_box(box, "vehicle#0", "ego"),),
            ),
        ),
        provenance=Provenance(dataset="synthetic", log_id="mini"),
    )


def test_scene_ir_wraps_probe_scene_losslessly():
    ir = _minimal_ir()
    assert len(ir) == 2
    assert ir.name == "mini"
    # the embedded probe.Scene is unchanged -- every probe accessor still works
    assert ir.scene.ego_speed(0) == 5.0
    assert ir.scene.objects_at(0)[0].label == "vehicle"
    assert ir.frames[1].time == 0.5


def test_entity_round_trips_through_tracked_box():
    box = TrackedBox(center=(3.0, -2.0, 1.0), size=(4.5, 2.0, 1.8), yaw=0.75, label="vehicle", velocity=(1.5, -0.5))
    ent = Entity.from_tracked_box(box, entity_id="e1", frame="ego")
    back = ent.to_tracked_box()
    assert back.center == box.center
    assert back.size == box.size
    assert back.label == box.label
    assert back.velocity == box.velocity
    assert math.isclose(back.yaw, box.yaw, abs_tol=1e-9)


def test_pose_yaw_round_trip_and_matrix_is_se3():
    p = Pose.from_yaw((1.0, 2.0, 3.0), 0.6)
    assert math.isclose(p.yaw, 0.6, abs_tol=1e-9)
    m = p.matrix()
    assert m.shape == (4, 4)
    # rotation block is orthonormal, translation block matches
    R = m[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(m[:3, 3], [1.0, 2.0, 3.0])


def test_coordinate_frame_transform_and_inverse_compose_to_identity():
    cf = CoordinateFrame("ego", "world", Pose.from_yaw((10.0, 5.0, 0.0), math.pi / 3))
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, -1.0, 2.0]])
    world = cf.transform_points(pts)
    back = cf.inverse().transform_points(world)
    assert np.allclose(back, pts, atol=1e-9)


def test_coordinate_frame_project_requires_intrinsics():
    no_intr = CoordinateFrame("lidar", "ego", Pose((0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        no_intr.project(np.array([[1.0, 1.0, 2.0]]))
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    cam = CoordinateFrame("cam", "ego", Pose((0.0, 0.0, 0.0)), intrinsics=K)
    px = cam.project(np.array([[0.0, 0.0, 2.0]]))
    assert np.allclose(px, [[50.0, 40.0]])  # principal point for an on-axis point


def test_track_length_invariant_enforced():
    with pytest.raises(ValueError):
        Track(entity_id="x", category="vehicle", timestamps=(0.0, 1.0), states=())


def test_schema_version_is_stamped():
    assert _minimal_ir().provenance.schema_version == SCHEMA_VERSION


@pytest.mark.skipif(not _AV2_ROOT.is_dir() or not any(_AV2_ROOT.iterdir()), reason="no AV2-Sensor logs on disk")
def test_ir_from_real_av2_log_has_real_calibration():
    from aletheon.adapt import ingest

    log = sorted(p for p in _AV2_ROOT.iterdir() if p.is_dir())[0]
    ir = ingest(log)
    assert len(ir) > 0
    assert ir.provenance.dataset == "av2_sensor"
    assert ir.provenance.log_id == log.name
    # the ego->world frame must carry the REAL SE3 from city_SE3_egovehicle.feather, not identity
    ego = ir.frame_for("ego")
    assert ego is not None
    assert ego.parent == "world"
    assert ego.pose.translation != (0.0, 0.0, 0.0)  # real AV2 city translation is large, never origin
    # entities lifted from boxes carry a full pose + the ego frame name
    if ir.tracks:
        st = ir.tracks[0].states[0]
        assert st.frame == "ego"
        assert isinstance(st.pose, Pose)
