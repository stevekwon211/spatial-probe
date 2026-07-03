# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""search_yield_occ3d unit gates, pinned BEFORE the sealed corpus run (pre-reg
`search_yield_occ3d_preregistration.md`):

1. memoized namespace == the sealed direct evaluator (`retrieval.frame_true`) — the
   semantics-preservation gate: if memoization changes any query verdict, the run must not start.
2. band_unknown_fraction against a HAND-COMPUTED value on a fixed toy grid (the `_roc_auc` lesson).
3. the tri-state honesty tag truth table, including the sealed boundary (uf == eps -> EXONERATED).
4. yield + scene-clustered bootstrap on hand values.
5. horizon extraction from the sealed predicate strings fails loudly when absent.
6. the occ3d annotations cache parses once per root.
"""
from __future__ import annotations

import json
import math

import numpy as np

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid, UnknownPolicy
from probe.query_spec import Query
from probe.retrieval import frame_true
from probe.scene import Frame, Scene

from experiments.occquery_v0.search_yield_occ3d import (
    band_unknown_fraction,
    horizon_of,
    memoized_namespace,
    tag_decision,
    yield_ci,
)


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, 1.0, (0.0, 0.0, 0.0), 0.5)


def _blank() -> np.ndarray:
    return np.full((30, 30, 4), FREE, dtype=int)


def _blocked_grid() -> OccupancyGrid:
    occ = _blank()
    for lat in (-1, 0, 1):
        occ[14, 10 + lat, 1] = OCCUPIED  # wall 4 m ahead across the ego corridor at (10, 10)
    return _grid(occ)


def _scene(grids: list[OccupancyGrid]) -> Scene:
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=5.0)
    return Scene(tuple(Frame(g, ego, float(i)) for i, g in enumerate(grids)), "s")


# --- 1. semantics-preservation gate -----------------------------------------------------------
_EXPRS = [
    "lateral_clearance(scene, t) < 0.5",
    "not free_along_ego_path(scene, t, horizon=1.0)",
    "0 < min_free_width_along_path(scene, t, horizon=2.0) < ego_width(scene)",
    "min_free_width_along_path(scene, t, horizon=4.0) == 0.0",
    "ego_speed(scene, t) > 8.33 and not free_along_ego_path(scene, t, horizon=4.0)",
    "not free_along_ego_path(scene, t, horizon=1.0) or lateral_clearance(scene, t) < 0.5",
]


def test_memoized_namespace_matches_direct_evaluator():
    from probe.query_dsl import safe_eval

    scene = _scene([_grid(_blank()), _blocked_grid()])
    for policy in (UnknownPolicy.FREE, UnknownPolicy.OCCUPIED):
        ns, cache = memoized_namespace(scene, policy)
        for expr in _EXPRS:
            for t in scene.times():
                ns["t"] = t
                assert bool(safe_eval(expr, ns)) == frame_true(scene, t, expr, policy), (
                    f"memoized != direct for {expr!r} t={t} policy={policy}"
                )
    assert cache, "cache never populated — memoization is not wired"


def test_memoized_namespace_caches_repeat_calls():
    from probe.query_dsl import safe_eval

    scene = _scene([_blocked_grid()])
    ns, cache = memoized_namespace(scene, UnknownPolicy.FREE)
    ns["t"] = 0
    safe_eval("not free_along_ego_path(scene, t, horizon=1.0)", ns)
    n_after_first = len(cache)
    safe_eval("not free_along_ego_path(scene, t, horizon=1.0)", ns)
    assert len(cache) == n_after_first  # second eval hits the cache, no new entries


# --- 2. band_unknown_fraction hand value -------------------------------------------------------
def test_band_unknown_fraction_hand_computed():
    # grid: voxel 1.0, origin (0,0,0), ground 0.5; ego (10,10,0) heading 0, speed 5, w 1.85 l 4.6
    # h 1.9. horizon 1.0 -> reach = 4.6/2 + 5*1.0 = 7.3 -> x centers 10..17 (fwd in [0, 7.3]).
    # band_half = 1.85/2 + 1.0 = 1.925 -> y centers 9, 10, 11. z slab: 0.5 < z <= 2.4 -> z = 1, 2.
    # band volume = 8 * 3 * 2 = 48 voxels.
    occ = _blank()
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=5.0)
    occ[10:13, 9, 1] = UNKNOWN   # 3 unknown inside the band
    occ[10:13, 9, 3] = UNKNOWN   # z=3 is above the ego-height slab -> excluded
    occ[0, 0, 1] = UNKNOWN       # far outside the band -> excluded
    uf = band_unknown_fraction(_grid(occ), ego, horizon=1.0)
    assert math.isclose(uf, 3 / 48), uf


def test_band_unknown_fraction_fully_observed_is_zero():
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=5.0)
    assert band_unknown_fraction(_grid(_blank()), ego, horizon=1.0) == 0.0


# --- 3. tag truth table ------------------------------------------------------------------------
def test_tag_decision_truth_table():
    assert tag_decision(hit=True, unknown_fraction=0.99, eps=0.05) == "CONFIRMED_HIT"
    assert tag_decision(hit=True, unknown_fraction=0.0, eps=0.05) == "CONFIRMED_HIT"
    assert tag_decision(hit=False, unknown_fraction=0.04, eps=0.05) == "EXONERATED"
    assert tag_decision(hit=False, unknown_fraction=0.05, eps=0.05) == "EXONERATED"  # sealed: <= eps
    assert tag_decision(hit=False, unknown_fraction=0.051, eps=0.05) == "UNRESOLVED"


# --- 4. yield + bootstrap hand values -----------------------------------------------------------
def test_yield_ci_hand_values():
    rng = np.random.default_rng(0)
    out = yield_ci([1, 0, 0, 1], rng)
    assert math.isclose(out["mean"], 0.5)
    assert 0.0 <= out["lo"] <= out["mean"] <= out["hi"] <= 1.0
    flat = yield_ci([0, 0, 0, 0], rng)
    assert flat["mean"] == 0.0 and flat["lo"] == 0.0 and flat["hi"] == 0.0


# --- 5. horizon extraction ----------------------------------------------------------------------
def _q(predicate: str) -> Query:
    return Query(id="q", nl="", backend="occupancy", status="implemented", scope="any",
                 refav_expressible=False, rationale="", predicate=predicate)


def test_horizon_of_reads_sealed_predicates():
    assert horizon_of(_q("not free_along_ego_path(scene, t, horizon=0.5)")) == 0.5
    assert horizon_of(_q("0 < min_free_width_along_path(scene, t, horizon=4.0) < ego_width(scene)")) == 4.0


def test_horizon_of_fails_loudly_without_horizon():
    try:
        horizon_of(_q("lateral_clearance(scene, t) < 0.5"))
    except ValueError:
        return
    raise AssertionError("horizon_of must raise on a predicate without a horizon")


# --- 6. occ3d annotations cache -----------------------------------------------------------------
def test_occ3d_annotations_parsed_once_per_root(tmp_path):
    from probe.adapters.occ3d import _annotations

    (tmp_path / "annotations.json").write_text(json.dumps({"scene_infos": {}}))
    first = _annotations(tmp_path)
    second = _annotations(tmp_path)
    assert first is second
