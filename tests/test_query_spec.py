# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""probe.query_spec -- schema validation for query files. An invalid query must fail at load
time with a clear error, never silently or mid-experiment."""
import pathlib

import pytest

from probe.query_spec import QuerySpecError, load_queries, validate_query

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _ok(**over) -> dict:
    base = {
        "id": "q",
        "nl": "n",
        "backend": "occupancy",
        "status": "implemented",
        "scope": "any",
        "refav_expressible": False,
        "rationale": "r",
        "predicate": "lateral_clearance(scene, t) < 0.5",
    }
    base.update(over)
    return base


def test_valid_any_query():
    q = validate_query(_ok())
    assert q.id == "q" and q.is_occupancy is True


def test_baseline_only_is_not_occupancy():
    q = validate_query(_ok(backend="tracking", status="baseline_only",
                           predicate="distance_to_nearest_object(scene, t) < 2.0"))
    assert q.is_occupancy is False


def test_transition_query_requires_before_after_window():
    q = validate_query({
        "id": "t", "nl": "n", "backend": "occupancy", "status": "implemented",
        "scope": "transition", "refav_expressible": False, "rationale": "r",
        "before": "not free_along_ego_path(scene, t, horizon=1.0)",
        "after": "free_along_ego_path(scene, t, horizon=1.0)",
        "within_frames": 3,
    })
    assert q.scope == "transition" and q.within_frames == 3


def test_missing_field_raises():
    bad = _ok()
    del bad["backend"]
    with pytest.raises(QuerySpecError):
        validate_query(bad)


def test_unknown_field_raises():
    with pytest.raises(QuerySpecError):
        validate_query(_ok(typo=1))


def test_bad_backend_raises():
    with pytest.raises(QuerySpecError):
        validate_query(_ok(backend="lidar"))


def test_transition_missing_window_raises():
    with pytest.raises(QuerySpecError):
        validate_query({
            "id": "t", "nl": "n", "backend": "occupancy", "status": "implemented",
            "scope": "transition", "refav_expressible": False, "rationale": "r",
            "before": "x", "after": "y",
        })


def test_any_scope_with_transition_fields_raises():
    with pytest.raises(QuerySpecError):
        validate_query(_ok(before="x"))


def test_repo_queries_file_is_valid():
    queries = load_queries(_REPO / "experiments" / "occquery_v0" / "queries.yaml")
    ids = [q.id for q in queries]
    assert "tight_clearance_at_speed" in ids
    # occquery is static per PLAN s0; the temporal blocked->clears moved to the dynfield experiment
    assert "free_path_is_blocked" in ids
    assert any(q.backend == "tracking" and q.status == "baseline_only" for q in queries)
