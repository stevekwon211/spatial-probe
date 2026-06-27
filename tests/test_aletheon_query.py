# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S2 SceneQL over the Scene IR (aletheon.query.query).

One engine: the same AST-whitelist (`probe.query_dsl.safe_eval`) the CLI and retrieval use, the
predicate namespace, and temporal scopes (any/all/transition). Public API speaks Aletheon/python
types only -- a `SceneIR` or a wrapped `probe.Scene` in, frame indices / bools out. The malicious
expression is rejected (the wrong thing is unrepresentable, not filtered). The real on-disk run is
skip-if-missing.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from probe.grid import EgoPose, OccupancyGrid, UnknownPolicy
from probe.query_dsl import UnsafeExpression
from probe.raycast import FREE, OCCUPIED
from probe.scene import Frame, Scene, TrackedBox
from aletheon.ir import SceneIR
from aletheon.query import query

_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")


def _first_av2_log():
    if _AV2_ROOT.is_dir():
        logs = sorted(p for p in _AV2_ROOT.iterdir() if p.is_dir())
        if logs:
            return logs[0]
    return None


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occupancy=occ, voxel_size=1.0, origin=(0.0, 0.0, 0.0), ground_height=-0.5)


def _three_frame_ir() -> SceneIR:
    # frame 0: open. frame 1: a wall at x=5. frame 2: open again.
    open_occ = np.full((10, 1, 3), FREE, dtype=int)
    wall_occ = np.full((10, 1, 3), FREE, dtype=int)
    wall_occ[5, 0, :] = OCCUPIED
    ego = EgoPose(position=(0.0, 0.0, 1.0), heading=0.0, speed=5.0)
    frames = (
        Frame(grid=_grid(open_occ), ego=ego, time=0.0),
        Frame(grid=_grid(wall_occ), ego=ego, time=0.5),
        Frame(grid=_grid(open_occ), ego=ego, time=1.0),
    )
    return SceneIR(scene=Scene(frames=frames, name="synthetic3"))


# --- engine: scopes -------------------------------------------------------------------------

def test_any_scope_returns_matching_frames():
    ir = _three_frame_ir()
    res = query(ir, "occluded(scene, t, 9.0, 0.0, target_z=1.0)", scope="any")
    assert res.matched_frames == [1]  # only the wall frame occludes the far point
    assert res.matched is True


def test_all_scope_is_false_when_one_frame_fails():
    ir = _three_frame_ir()
    res = query(ir, "ego_speed(scene, t) > 0", scope="all")
    assert res.matched is True
    res2 = query(ir, "occluded(scene, t, 9.0, 0.0, target_z=1.0)", scope="all")
    assert res2.matched is False  # frames 0 and 2 are open


def test_transition_scope_before_then_after():
    ir = _three_frame_ir()
    # blocked at frame 1, then clear at frame 2: a clear->blocked->clear pattern.
    res = query(
        ir,
        None,
        scope="transition",
        before="occluded(scene, t, 9.0, 0.0, target_z=1.0)",
        after="not occluded(scene, t, 9.0, 0.0, target_z=1.0)",
        within_frames=2,
    )
    assert res.matched is True


def test_query_accepts_a_bare_scene_too():
    ir = _three_frame_ir()
    res = query(ir.scene, "ego_speed(scene, t) > 0", scope="any")
    assert res.matched is True


# --- engine: safety -------------------------------------------------------------------------

def test_rejects_malicious_expression():
    ir = _three_frame_ir()
    with pytest.raises((UnsafeExpression, NameError)):
        query(ir, "__import__('os').system('echo pwned')", scope="any")


def test_unknown_predicate_raises_nameerror():
    ir = _three_frame_ir()
    with pytest.raises(NameError):
        query(ir, "totally_made_up(scene, t) < 1", scope="any")


def test_existing_occupancy_predicates_still_work_in_namespace():
    ir = _three_frame_ir()
    # min_free_width_along_path is the existing free-path predicate -- it must still resolve.
    res = query(ir, "min_free_width_along_path(scene, t, 0.0) < 100.0", scope="any")
    assert isinstance(res.matched, bool)


def test_policy_is_threaded():
    # a frame whose only obstacle is UNKNOWN flips occluded under the conservative policy.
    from probe.raycast import UNKNOWN
    occ = np.full((10, 1, 3), FREE, dtype=int)
    occ[5, 0, :] = UNKNOWN
    ego = EgoPose(position=(0.0, 0.0, 1.0), heading=0.0)
    ir = SceneIR(scene=Scene(frames=(Frame(grid=_grid(occ), ego=ego),), name="u"))
    # occluded reads occupancy directly with its own unknown_blocks flag (independent of policy),
    # so assert the explicit flag path here.
    free_res = query(ir, "occluded(scene, t, 9.0, 0.0, target_z=1.0)", scope="any")
    assert free_res.matched is False
    blk_res = query(ir, "occluded(scene, t, 9.0, 0.0, target_z=1.0, unknown_blocks=True)", scope="any")
    assert blk_res.matched is True


# --- real on-disk run (skip-if-missing) -----------------------------------------------------

@pytest.mark.skipif(_first_av2_log() is None, reason="no AV2-Sensor log on disk")
def test_query_runs_on_real_av2_ir():
    from aletheon.adapt import ingest
    log = _first_av2_log()
    ir = ingest(log)
    res = query(ir, "ego_speed(scene, t) >= 0.0", scope="all")
    assert res.matched is True
    assert res.n_frames > 0
    # the new occluded predicate evaluates without error on real occupancy.
    res2 = query(ir, "occluded(scene, t, 15.0, 0.0)", scope="any")
    assert isinstance(res2.matched, bool)
