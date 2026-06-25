# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""PRISM serialization -- lossless parquet round-trip (content hash) + OpenLABEL/JSONL views.

The load-bearing claim: from_parquet(to_parquet(s)) reproduces s at the content level, including
the dense occupancy grids and NaN velocities. Asserted via content_hash equality AND field equality.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox
from prism.ir import (
    CoordinateFrame,
    DatasetSlice,
    Entity,
    Event,
    Failure,
    GroundTruth,
    Observation,
    Pose,
    Prediction,
    Provenance,
    Relation,
    SceneIR,
    Track,
)
from prism.serialize import (
    content_hash,
    from_parquet,
    scene_ir_from_dict,
    scene_ir_to_dict,
    to_jsonl,
    to_openlabel_json,
    to_parquet,
)

_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")


def _rich_ir() -> SceneIR:
    """An IR exercising every layer + the awkward values (NaN velocity, mixed occupancy, intrinsics)."""
    rng = np.random.default_rng(0)
    occ0 = rng.integers(UNKNOWN, OCCUPIED + 1, size=(4, 4, 4)).astype(int)
    occ1 = np.full((4, 4, 4), FREE, dtype=int)
    occ1[0, 0, 0] = OCCUPIED
    box = TrackedBox(center=(3.0, -1.0, 0.5), size=(4.5, 2.0, 1.8), yaw=0.4, label="vehicle",
                     velocity=(float("nan"), float("nan")))
    scene = Scene(
        (
            Frame(OccupancyGrid(occ0, 0.4, (-40.0, -40.0, -1.0), -1.0), EgoPose((0, 0, 0), 0.1, speed=5.0), time=0.0, objects=(box,)),
            Frame(OccupancyGrid(occ1, 0.4, (-40.0, -40.0, -1.0), -1.0), EgoPose((1, 0, 0), 0.1, speed=6.0), time=0.5),
        ),
        name="rich",
    )
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    ent = Entity.from_tracked_box(box, "vehicle#0", "ego")
    return SceneIR(
        scene=scene,
        coordinate_frames=(
            CoordinateFrame("world", None, Pose((0.0, 0.0, 0.0))),
            CoordinateFrame("ego", "world", Pose.from_yaw((5568.3, 2152.2, 74.4), 1.2)),
            CoordinateFrame("cam", "ego", Pose((0.1, 0.0, 1.5)), intrinsics=K),
        ),
        tracks=(Track("vehicle#0", "vehicle", (0.0,), (ent,)),),
        observations=(Observation("lidar", 0.0, 0), Observation("lidar", 0.5, 1)),
        relations=(Relation("vehicle#0", "ahead_of", "ego", 0),),
        events=(Event("near_miss", 0, 1, "vehicle#0"),),
        predictions=(Prediction(0, (ent,), "model_v0"),),
        ground_truth=(GroundTruth(0, (ent,)), GroundTruth(1, ())),
        failures=(Failure(1, "missed_detection", "fp", "vehicle#0"),),
        slices=(DatasetSlice("danger", (0,)),),
        provenance=Provenance(dataset="synthetic", log_id="rich", adapter="test", commit="deadbeef"),
    )


def test_parquet_round_trip_content_hash(tmp_path):
    s = _rich_ir()
    p = to_parquet(s, tmp_path / "ir.parquet")
    assert pathlib.Path(p).exists()
    back = from_parquet(p)
    assert content_hash(back) == content_hash(s)


def test_parquet_round_trip_grids_exact(tmp_path):
    s = _rich_ir()
    back = from_parquet(to_parquet(s, tmp_path / "ir.parquet"))
    for a, b in zip(s.scene.frames, back.scene.frames):
        assert a.grid.occupancy.dtype == b.grid.occupancy.dtype
        assert np.array_equal(a.grid.occupancy, b.grid.occupancy)
        assert a.grid.voxel_size == b.grid.voxel_size
        assert a.grid.origin == b.grid.origin


def test_parquet_round_trip_preserves_nan_velocity(tmp_path):
    s = _rich_ir()
    back = from_parquet(to_parquet(s, tmp_path / "ir.parquet"))
    vx, vy = back.scene.frames[0].objects[0].velocity
    assert math.isnan(vx) and math.isnan(vy)


def test_parquet_round_trip_preserves_intrinsics(tmp_path):
    s = _rich_ir()
    back = from_parquet(to_parquet(s, tmp_path / "ir.parquet"))
    cam = back.frame_for("cam")
    assert cam is not None and cam.intrinsics is not None
    assert np.allclose(cam.intrinsics, s.frame_for("cam").intrinsics)


def test_dict_round_trip_is_identity():
    s = _rich_ir()
    d = scene_ir_to_dict(s)
    grids = [fr.grid.occupancy for fr in s.scene.frames]
    back = scene_ir_from_dict(d, grids)
    assert content_hash(back) == content_hash(s)


def test_a_changed_voxel_changes_the_hash(tmp_path):
    s = _rich_ir()
    h0 = content_hash(s)
    occ = s.scene.frames[1].grid.occupancy.copy()
    occ[1, 1, 1] = OCCUPIED
    frames = list(s.scene.frames)
    frames[1] = Frame(OccupancyGrid(occ, 0.4, (-40.0, -40.0, -1.0), -1.0), frames[1].ego, frames[1].time, frames[1].objects)
    s2 = s.with_scene(Scene(tuple(frames), s.scene.name))
    assert content_hash(s2) != h0


def test_openlabel_export_is_valid_json_with_expected_keys():
    s = _rich_ir()
    ol = to_openlabel_json(s)
    text = json.dumps(ol)  # must be JSON-serializable
    reparsed = json.loads(text)
    assert "openlabel" in reparsed
    body = reparsed["openlabel"]
    for key in ("metadata", "frames", "objects", "coordinate_systems"):
        assert key in body, f"missing OpenLABEL key {key}"
    assert body["metadata"]["name"] == "rich"
    assert "0" in body["frames"]  # frame 0 present
    assert len(body["objects"]) >= 1  # the one box exported


def test_jsonl_export_one_line_per_frame():
    s = _rich_ir()
    lines = to_jsonl(s).splitlines()
    assert len(lines) == len(s.scene.frames)
    first = json.loads(lines[0])
    assert first["frame_index"] == 0
    assert "ego" in first and "objects" in first


def test_from_parquet_rejects_non_prism_file(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    plain = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), plain)
    with pytest.raises(ValueError):
        from_parquet(plain)


@pytest.mark.skipif(not _AV2_ROOT.is_dir() or not any(_AV2_ROOT.iterdir()), reason="no AV2-Sensor logs on disk")
def test_real_av2_log_round_trips(tmp_path):
    from prism.adapt import ingest

    log = sorted(p for p in _AV2_ROOT.iterdir() if p.is_dir())[0]
    ir = ingest(log)
    back = from_parquet(to_parquet(ir, tmp_path / "av2.parquet"))
    assert content_hash(back) == content_hash(ir)
