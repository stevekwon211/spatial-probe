# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S5b failure-taxonomy mining + S6 `prism find` wow demo.

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
from prism.failure import (
    SIGNATURES,
    FailureCandidate,
    cluster,
    find,
    mine,
    resolve_signature,
    signature_for_query,
    similar_frames,
)
from prism.ir import Failure, Provenance, SceneIR

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
    from prism.adapt import ingest

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
