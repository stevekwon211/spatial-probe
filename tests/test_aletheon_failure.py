# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S5b failure-taxonomy mining + S6 `aletheon find` wow demo.

Honest framing (mirrors the module + repo CLAUDE.md): there are NO model predictions on disk, so
the only buildable failure signal is the H1 occupancy-vs-box SET DIFFERENCE, which needs no
predictions:

- `path_blocked_no_box` -- occupancy blocks the ego in-path band AND no tracked box explains it ->
  an unboxed-obstacle / FP candidate. The FP DIRECTION is externally anchored by the traversal-v0.1
  oracle (RELIABLE: occupancy does not hallucinate obstacles on the driven path).
- `box_in_free` -- a LiDAR-seen (num_interior_pts>=5) box whose footprint occupancy marks FREE ->
  a recall-miss CANDIDATE, CONSISTENCY-ONLY (same-modality box-recall oracle; external recall is
  honestly closed -- stereo + DAv2 both killed).

Synthetic tests plant one of each on a hand-built SceneIR so the signature mechanics are exercised
deterministically with no data; the `find` test runs on a real on-disk AV2 log (skip-if-missing).
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox
from aletheon.failure import (
    SIGNATURES,
    FailureCandidate,
    cluster,
    find,
    mine,
    resolve_signature,
    signature_for_query,
    similar_frames,
)
from aletheon.ir import Failure, Provenance, SceneIR

_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")


# --- synthetic SceneIR builders ------------------------------------------------------------------
# Grid spec mirrors the AV2 voxelizer (0.4 m, x[-40,40] fwd, y[-40,40] left, ground at -1.0) so the
# free-path / footprint geometry behaves the way it does on real data, at a tiny size.

_VOXEL = 0.4
_NX, _NY, _NZ = 60, 60, 8  # ~ x[-12,12], y[-12,12] -- enough for a forward block + a side box
_ORIGIN = (-12.0 + _VOXEL / 2.0, -12.0 + _VOXEL / 2.0, -1.0 + _VOXEL / 2.0)
_GROUND = -1.0


def _empty_occ() -> np.ndarray:
    return np.full((_NX, _NY, _NZ), FREE, dtype=int)


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, _VOXEL, _ORIGIN, _GROUND)


def _ix(world_x: float, world_y: float, world_z: float) -> tuple[int, int, int]:
    return _grid(_empty_occ()).world_to_voxel((world_x, world_y, world_z))


def _ego(speed: float = 5.0) -> EgoPose:
    # default nuScenes-ish ego; heading +x so "forward" == +x, matching the AV2 ego frame.
    return EgoPose(position=(0.0, 0.0, 0.0), heading=0.0, speed=speed, width=1.85, length=4.6, height=1.9)


def _scene_ir(frames, name="synthetic") -> SceneIR:
    sc = Scene(frames=tuple(frames), name=name)
    prov = Provenance(dataset="synthetic", log_id=name, adapter="test")
    return SceneIR(scene=sc, provenance=prov)


def _wall_across_path(occ: np.ndarray, world_x: float) -> None:
    """Fill a solid wall (a cluster, so min_cluster_voxels=2 keeps it) across the ego corridor at a
    forward x, spanning the ego-width lateral band and the ego height band."""
    i, _, _ = _ix(world_x, 0.0, 0.0)
    for di in (0, 1):  # two forward voxels -> a real cluster, not lone-voxel noise
        for jy in range(-3, 4):  # +/- ~1.2 m lateral, covers the ego centerline + body
            _, j, _ = _ix(world_x, jy * _VOXEL, 0.0)
            for kz in range(1, 4):  # above the ground floor, within the ego height band
                occ[i + di, j, kz] = OCCUPIED


# === path_blocked_no_box =========================================================================


def test_path_blocked_no_box_finds_a_planted_unboxed_obstacle():
    # An occupancy wall 4 m ahead on the ego path, and NO tracked box anywhere -> the occupancy
    # blocks the path but no box explains it: the unboxed-obstacle / FP candidate.
    occ = _empty_occ()
    _wall_across_path(occ, world_x=4.0)
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=())  # no boxes at all
    ir = _scene_ir([fr])
    cands = mine([ir], "path_blocked_no_box", params={"horizon": 1.0, "box_radius_m": 5.0})
    assert len(cands) >= 1
    c = cands[0]
    assert c.signature == "path_blocked_no_box"
    assert c.frame_index == 0
    assert "external" in c.honesty.lower()  # FP side is externally anchored (traversal-v0.1)
    assert math.isfinite(c.features["forward_range_m"])


def test_path_blocked_no_box_is_silent_when_a_box_explains_the_block():
    # Same wall, but now a tracked box sits ON the block -> the block IS explained, NOT a candidate.
    occ = _empty_occ()
    _wall_across_path(occ, world_x=4.0)
    box = TrackedBox(center=(4.0, 0.0, 0.0), size=(2.0, 2.0, 1.6), yaw=0.0, label="vehicle")
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=(box,))
    ir = _scene_ir([fr])
    cands = mine([ir], "path_blocked_no_box", params={"horizon": 1.0, "box_radius_m": 5.0})
    assert cands == []


def test_path_blocked_no_box_is_silent_on_a_clear_path():
    fr = Frame(grid=_grid(_empty_occ()), ego=_ego(), time=0.0, objects=())
    ir = _scene_ir([fr])
    assert mine([ir], "path_blocked_no_box", params={"horizon": 1.0, "box_radius_m": 5.0}) == []


# === box_in_free =================================================================================


def test_box_in_free_finds_a_box_in_occupancy_free_space():
    # A tracked box well ahead/aside in EMPTY occupancy (its footprint marks FREE) -> a recall-miss
    # consistency candidate. The interior-pts gate is satisfied via the side-channel lookup.
    occ = _empty_occ()  # entirely free
    box = TrackedBox(center=(6.0, 3.0, 0.0), size=(2.0, 2.0, 1.6), yaw=0.0, label="vehicle")
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=(box,))
    ir = _scene_ir([fr])
    # interior-pts side channel: {(frame_index, box_slot): num_interior_pts}; >=5 passes the gate.
    cands = mine([ir], "box_in_free", params={"interior_pts": {(0, 0): 20}, "n_interior_min": 5})
    assert len(cands) >= 1
    c = cands[0]
    assert c.signature == "box_in_free"
    assert "consistency-only" in c.honesty.lower()  # NOT externally validated recall


def test_box_in_free_is_silent_when_occupancy_covers_the_box():
    # Same box, but occupancy is OCCUPIED at its footprint -> occupancy recalls it, NOT a miss.
    occ = _empty_occ()
    i, j, _ = _ix(6.0, 3.0, 0.0)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for kz in range(1, 4):
                occ[i + di, j + dj, kz] = OCCUPIED
    box = TrackedBox(center=(6.0, 3.0, 0.0), size=(2.0, 2.0, 1.6), yaw=0.0, label="vehicle")
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=(box,))
    ir = _scene_ir([fr])
    cands = mine([ir], "box_in_free", params={"interior_pts": {(0, 0): 20}, "n_interior_min": 5})
    assert cands == []


def test_box_in_free_gate_drops_sensor_blind_boxes():
    # A box with too few interior points (LiDAR barely saw it) is below the gate -> NOT a recall
    # miss (occupancy can't mark what LiDAR never returned). The gate must drop it.
    occ = _empty_occ()
    box = TrackedBox(center=(6.0, 3.0, 0.0), size=(2.0, 2.0, 1.6), yaw=0.0, label="vehicle")
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=(box,))
    ir = _scene_ir([fr])
    cands = mine([ir], "box_in_free", params={"interior_pts": {(0, 0): 1}, "n_interior_min": 5})
    assert cands == []


# === clustering + similarity =====================================================================


def test_clustering_groups_same_signature_failures_by_feature():
    # Two near-identical blocked-path frames (same range bin) -> one cluster; a third far-block ->
    # a second cluster. Feature-distance clustering, not semantic.
    def blocked_at(x):
        occ = _empty_occ()
        _wall_across_path(occ, world_x=x)
        return Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=())

    ir = _scene_ir([blocked_at(3.0), blocked_at(3.2), blocked_at(9.0)])
    cands = mine([ir], "path_blocked_no_box", params={"horizon": 2.0, "box_radius_m": 5.0})
    assert len(cands) == 3
    clusters = cluster(cands, range_bin_m=2.0)
    assert len(clusters) == 2  # {3.0, 3.2} together; {9.0} apart
    sizes = sorted(len(cl.candidates) for cl in clusters)
    assert sizes == [1, 2]
    # each cluster lowers to a DatasetSlice over its member frames
    big = max(clusters, key=lambda cl: len(cl.candidates))
    assert set(big.slice.frame_indices) == {0, 1}
    assert big.slice.name.startswith("path_blocked_no_box")


def test_similar_frames_ranks_by_feature_distance_to_centroid():
    def blocked_at(x):
        occ = _empty_occ()
        _wall_across_path(occ, world_x=x)
        return Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=())

    ir = _scene_ir([blocked_at(3.0), blocked_at(3.1), blocked_at(10.0)])
    cands = mine([ir], "path_blocked_no_box", params={"horizon": 2.0, "box_radius_m": 5.0})
    clusters = cluster(cands, range_bin_m=2.0)
    near = min(clusters, key=lambda cl: cl.centroid["forward_range_m"])
    ranked = similar_frames(near, cands, k=3)
    # the two ~3 m blocks rank above the 10 m block (closer to the near centroid)
    assert len(ranked) == 3
    ranges = [r.candidate.features["forward_range_m"] for r in ranked]
    assert ranges[0] <= ranges[1] <= ranges[2]
    assert ranges[-1] == max(ranges)  # the far block is least similar


# === registry + query mapping ====================================================================


def test_signature_registry_is_declarative_and_extensible():
    assert "path_blocked_no_box" in SIGNATURES
    assert "box_in_free" in SIGNATURES
    # resolve by name AND by alias
    assert resolve_signature("path_blocked_no_box").name == "path_blocked_no_box"
    assert resolve_signature("recall-miss").name == "box_in_free"


def test_signature_for_query_maps_natural_text():
    assert signature_for_query("path blocked but no tracked object explains it").name == "path_blocked_no_box"
    assert signature_for_query("a box the occupancy missed").name == "box_in_free"
    with pytest.raises(ValueError):
        signature_for_query("something totally unrelated to the corpus")


def test_failure_candidate_lowers_to_ir_failure():
    occ = _empty_occ()
    _wall_across_path(occ, world_x=4.0)
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=())
    ir = _scene_ir([fr])
    c = mine([ir], "path_blocked_no_box", params={"horizon": 1.0, "box_radius_m": 5.0})[0]
    f = c.to_ir_failure()
    assert isinstance(f, Failure)
    assert f.frame_index == 0
    assert f.kind == "path_blocked_no_box"


# === S6 find (the wow) -- synthetic + real -------------------------------------------------------


def test_find_returns_a_summary_on_synthetic():
    occ = _empty_occ()
    _wall_across_path(occ, world_x=4.0)
    fr = Frame(grid=_grid(occ), ego=_ego(), time=0.0, objects=())
    ir = _scene_ir([fr])
    out = find("path blocked but no tracked object explains it", [ir],
               params={"horizon": 1.0, "box_radius_m": 5.0})
    assert out["signature"] == "path_blocked_no_box"
    assert out["n_matches"] >= 1
    assert "honesty" in out
    assert "external" in out["honesty"].lower()
    assert out["clusters"]  # at least one cluster
    assert "human_summary" in out


@pytest.mark.skipif(not _AV2_ROOT.is_dir() or not any(_AV2_ROOT.iterdir()), reason="no AV2 on disk")
def test_find_runs_on_real_av2_corpus():
    from aletheon.adapt import ingest

    logs = sorted(p for p in _AV2_ROOT.iterdir() if p.is_dir())[:1]  # one log keeps it fast
    irs = [ingest(p) for p in logs]
    out = find("path blocked but no tracked object explains it", irs,
               params={"horizon": 1.0, "box_radius_m": 5.0, "limit_frames": 30})
    assert out["signature"] == "path_blocked_no_box"
    assert out["n_frames_scanned"] > 0
    # structural shape always present (n_matches may be 0 or more -- honest either way)
    assert "human_summary" in out and isinstance(out["human_summary"], str)
    assert "clusters" in out
    assert "external" in out["honesty"].lower()


# === missed_detection: 2D-detector recall vs GT (REAL model-eval) --------------------------------
# Synthetic: build a forward-looking camera with no distortion so an ego box projects to a known 2D
# box, then feed the signature synthetic detections. A GT box with NO matching detection -> flagged;
# a GT box WITH a matching detection -> not flagged. No model file or onnxruntime needed.

from aletheon.failure import AV2CameraCalib, match_detections, project_ego_boxes


def _forward_camera() -> AV2CameraCalib:
    # cam_x -> -ego_y (image right), cam_y -> -ego_z (image down), cam_z -> +ego_x (forward).
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return AV2CameraCalib(
        name="cam", fx=1000.0, fy=1000.0, cx=800.0, cy=600.0, k1=0.0, k2=0.0, k3=0.0,
        width=1600, height=1200, R_cam2ego=R, t_cam_in_ego=np.zeros(3),
    )


class _FakeDetection:
    """Duck-typed aletheon.detect.Detection (box_xyxy / center / av2_label / score)."""

    def __init__(self, box, label, score):
        self.box_xyxy = box
        self.av2_label = label
        self.score = score

    @property
    def center(self):
        x0, y0, x1, y1 = self.box_xyxy
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _frame_with_box(center, label="vehicle"):
    box = TrackedBox(center=center, size=(2.0, 2.0, 1.6), yaw=0.0, label=label)
    return Frame(grid=_grid(_empty_occ()), ego=_ego(), time=0.0, objects=(box,))


def test_missed_detection_flags_a_visible_gt_box_with_no_detection():
    # A vehicle 10 m ahead projects into the camera; the detector returned NOTHING -> a missed detection.
    ir = _scene_ir([_frame_with_box((10.0, 0.0, 0.0))])
    cam = _forward_camera()
    cands = mine([ir], "missed_detection",
                 params={"camera_calib": cam, "detections": {0: []}})
    assert len(cands) == 1
    c = cands[0]
    assert c.signature == "missed_detection"
    assert c.frame_index == 0
    assert "model-eval" in c.honesty.lower()  # REAL detector-eval, not occupancy
    assert c.features["category_code"] == 1.0  # vehicle
    assert math.isfinite(c.features["forward_range_m"])


def test_missed_detection_is_silent_when_a_matching_detection_exists():
    # Same box, but the detector output a matching vehicle detection on it -> NOT a miss.
    ir = _scene_ir([_frame_with_box((10.0, 0.0, 0.0))])
    cam = _forward_camera()
    vis, box2d, _ = project_ego_boxes(ir.scene.objects_at(0), cam)[0]
    det = _FakeDetection(box2d, "vehicle", 0.9)
    cands = mine([ir], "missed_detection",
                 params={"camera_calib": cam, "detections": {0: [det]}})
    assert cands == []


def test_missed_detection_ignores_boxes_the_camera_cannot_see():
    # A vehicle BEHIND the ego cannot project into a forward camera -> not a miss (projection artifact).
    ir = _scene_ir([_frame_with_box((-10.0, 0.0, 0.0))])
    cam = _forward_camera()
    cands = mine([ir], "missed_detection", params={"camera_calib": cam, "detections": {0: []}})
    assert cands == []


def test_missed_detection_ignores_non_coco_classes():
    # An 'other' class (e.g. a bollard collapsed to "other") is NOT a COCO-detectable class -> the
    # detector was never trained to find it, so a miss is not the detector's failure.
    ir = _scene_ir([_frame_with_box((10.0, 0.0, 0.0), label="other")])
    cam = _forward_camera()
    cands = mine([ir], "missed_detection", params={"camera_calib": cam, "detections": {0: []}})
    assert cands == []


def test_missed_detection_class_agnostic_match_suppresses_a_miss():
    # With class_agnostic, a detection of the WRONG class on the box still counts as "the detector saw
    # something there" -> not a miss. (Default class-aware would still flag a wrong-class detection.)
    ir = _scene_ir([_frame_with_box((10.0, 0.0, 0.0))])
    cam = _forward_camera()
    vis, box2d, _ = project_ego_boxes(ir.scene.objects_at(0), cam)[0]
    wrong = _FakeDetection(box2d, "pedestrian", 0.9)
    # class-aware: pedestrian detection does NOT explain a vehicle GT -> still a miss
    aware = mine([ir], "missed_detection", params={"camera_calib": cam, "detections": {0: [wrong]}})
    assert len(aware) == 1
    # class-agnostic: any detection on the box explains it -> not a miss
    agn = mine([ir], "missed_detection",
               params={"camera_calib": cam, "detections": {0: [wrong]}, "class_agnostic": True})
    assert agn == []


def test_missed_detection_clusters_group_by_category():
    # Two missed vehicles in one range bin and a missed pedestrian -> clustering splits by category
    # (category_code is part of the cluster key).
    boxes = (
        TrackedBox(center=(10.0, 0.0, 0.0), size=(2.0, 2.0, 1.6), yaw=0.0, label="vehicle"),
        TrackedBox(center=(11.0, 2.0, 0.0), size=(2.0, 2.0, 1.6), yaw=0.0, label="vehicle"),
        TrackedBox(center=(10.0, -2.0, 0.0), size=(0.8, 0.8, 1.7), yaw=0.0, label="pedestrian"),
    )
    fr = Frame(grid=_grid(_empty_occ()), ego=_ego(), time=0.0, objects=boxes)
    ir = _scene_ir([fr])
    cam = _forward_camera()
    cands = mine([ir], "missed_detection", params={"camera_calib": cam, "detections": {0: []}})
    assert len(cands) == 3  # all three visible, in-range, unmatched
    clusters = cluster(cands, range_bin_m=8.0)
    # vehicles (2) cluster together; the pedestrian (1) is its own cluster -> 2 clusters
    assert len(clusters) == 2
    sizes = sorted(len(cl.candidates) for cl in clusters)
    assert sizes == [1, 2]


def test_missed_detection_query_routes_and_is_silent_without_a_camera():
    from aletheon.failure import signature_for_query

    assert signature_for_query("vehicle missed by the detector").name == "missed_detection"
    assert signature_for_query("pedestrian missed by the model").name == "missed_detection"
    # no camera calibration available (synthetic log, no params) -> the signature is honestly silent
    ir = _scene_ir([_frame_with_box((10.0, 0.0, 0.0))])
    cands = mine([ir], "missed_detection", params={})
    assert cands == []


@pytest.mark.skipif(
    not (_AV2_ROOT / "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c").is_dir()
    or not pathlib.Path("data/models/yolov8n.onnx").exists(),
    reason="no camera AV2 log or detector model on disk",
)
def test_missed_detection_find_runs_on_real_av2_log():
    from aletheon.adapt import ingest

    log = _AV2_ROOT / "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c"
    out = find("pedestrian missed by the model", [ingest(log)],
               params={"limit_frames": 5})
    assert out["signature"] == "missed_detection"
    assert out["n_frames_scanned"] == 5
    assert "model-eval" in out["honesty"].lower()
    assert "human_summary" in out and isinstance(out["human_summary"], str)
