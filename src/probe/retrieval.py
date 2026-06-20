# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Scope-aware query evaluation over scenes, with unknown-policy sensitivity.

Instrument-level so the synthetic runner, the future M2 nuScenes runner, and the tests share one
evaluator. A query is evaluated against a scene under a chosen UnknownPolicy:

- scope `any`        -- some frame satisfies `predicate`
- scope `all`        -- every frame satisfies `predicate`
- scope `transition` -- some frame satisfies `before`, and a later frame within `within_frames`
                        satisfies `after` (a genuine temporal pattern, not any-frame-blocked)

Predicate expressions are evaluated by `probe.query_dsl.safe_eval` (AST whitelist), never eval().
"""
from __future__ import annotations

from probe.grid import UnknownPolicy
from probe.predicates.clearance import centerline_lateral_distance, lateral_clearance
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path
from probe.predicates.objects import distance_to_nearest_object
from probe.query_dsl import safe_eval
from probe.query_spec import Query
from probe.scene import Scene

__all__ = ["namespace", "frame_true", "scene_matches", "retrieved"]


def namespace(scene: Scene, policy: UnknownPolicy) -> dict:
    """The only identifiers a query expression may reference, bound to the given scene and unknown
    policy. The occupancy predicates carry `policy`; the tracking baseline ignores it."""
    return {
        "scene": scene,
        "lateral_clearance": lambda sc, t: lateral_clearance(sc.grid_at(t), sc.ego_at(t), unknown_policy=policy),
        "centerline_lateral_distance": lambda sc, t: centerline_lateral_distance(sc.grid_at(t), sc.ego_at(t), unknown_policy=policy),
        "free_along_ego_path": lambda sc, t, horizon: free_along_ego_path(sc.grid_at(t), sc.ego_at(t), horizon, unknown_policy=policy),
        "min_free_width_along_path": lambda sc, t, horizon: min_free_width_along_path(sc.grid_at(t), sc.ego_at(t), horizon, unknown_policy=policy),
        "ego_speed": lambda sc, t: sc.ego_speed(t),
        "ego_width": lambda sc: sc.ego_width(),
        "distance_to_nearest_object": lambda sc, t, object_class=None: distance_to_nearest_object(sc, t, object_class=object_class),
    }


def frame_true(scene: Scene, t: int, expr: str, policy: UnknownPolicy) -> bool:
    ns = namespace(scene, policy)
    ns["t"] = t
    return bool(safe_eval(expr, ns))


def scene_matches(scene: Scene, query: Query, policy: UnknownPolicy) -> bool:
    """Whether `scene` satisfies `query` under `policy`, per the query's temporal scope."""
    ts = scene.times()
    if query.scope == "any":
        return any(frame_true(scene, t, query.predicate, policy) for t in ts)
    if query.scope == "all":
        return all(frame_true(scene, t, query.predicate, policy) for t in ts)
    if query.scope == "transition":
        for offset, i in enumerate(ts):
            if not frame_true(scene, i, query.before, policy):
                continue
            for j in ts[offset + 1:]:
                if j - i <= query.within_frames and frame_true(scene, j, query.after, policy):
                    return True
        return False
    raise ValueError(f"unknown scope: {query.scope}")


def retrieved(scenes, query: Query, policy: UnknownPolicy) -> set[str]:
    """Names of the scenes that satisfy `query` under `policy`."""
    return {s.name for s in scenes if scene_matches(s, query, policy)}
