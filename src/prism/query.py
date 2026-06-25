# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S2 SceneQL -- run a query expression over a Scene IR, ONE engine shared with the CLI.

A SceneQL string is evaluated by `probe.query_dsl.safe_eval` (the AST whitelist -- untrusted-input
safe, never Python eval) against a namespace bound to a `SceneIR` (or a bare `probe.Scene`). The
namespace is the existing occupancy/box predicates (`probe.retrieval.namespace`) PLUS the S3
physical predicates (`prism.predicates`: occluded / object_speed / ttc), registered in ONE place
(`namespace()` here). Temporal scopes mirror the experiment retrieval layer:

- `any`        -- some frame satisfies `expr`
- `all`        -- every frame satisfies `expr`
- `transition` -- some frame satisfies `before`, and a later frame within `within_frames`
                  satisfies `after`

The public API speaks PRISM/python types only: a `SceneIR`/`Scene` in, a `QueryResult` out.
`prism.cli`'s query verb routes through this so the CLI and any programmatic caller share one
evaluator (no second, drifting query path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from probe.grid import UnknownPolicy
from probe.query_dsl import safe_eval
from probe.retrieval import namespace as _probe_namespace
from probe.scene import Scene
from prism.ir import SceneIR
from prism.predicates import object_speed, occluded, ttc, velocity

__all__ = ["query", "namespace", "frame_value", "QueryResult"]

SceneLike = Union[SceneIR, Scene]


def _as_scene(scene_like: SceneLike) -> Scene:
    return scene_like.scene if isinstance(scene_like, SceneIR) else scene_like


def namespace(scene_like: SceneLike, policy: UnknownPolicy = UnknownPolicy.FREE) -> dict:
    """The identifiers a SceneQL expression may reference: the probe predicate namespace EXTENDED
    with the S3 physical predicates. Registered in this single place so the CLI and the API agree.

    `scene` binds to the underlying `probe.Scene` (so the existing predicates' `scene.grid_at(t)`
    etc. keep working verbatim); the S3 predicates accept that Scene and unwrap it themselves.
    """
    sc = _as_scene(scene_like)
    ns = _probe_namespace(sc, policy)
    ns.update(
        {
            "occluded": occluded,
            "object_speed": object_speed,
            "ttc": ttc,
            "velocity": velocity,
        }
    )
    return ns


def frame_value(scene_like: SceneLike, t: int, expr: str, policy: UnknownPolicy):
    """Raw value of `expr` at frame `t` (not coerced to bool -- a predicate may return a number)."""
    ns = namespace(scene_like, policy)
    ns["t"] = t
    return safe_eval(expr, ns)


def _frame_true(scene_like: SceneLike, t: int, expr: str, policy: UnknownPolicy) -> bool:
    return bool(frame_value(scene_like, t, expr, policy))


@dataclass(frozen=True)
class QueryResult:
    """The outcome of a SceneQL query over one scene.

    `matched` is the scope-level verdict (does the SCENE satisfy the query). `matched_frames` are
    the frame indices that individually satisfied the per-frame `expr` (the `before` predicate for
    transition scope) -- empty/partial for `all`/`transition` but always populated for `any`.
    """

    matched: bool
    matched_frames: list[int]
    n_frames: int
    scope: str
    expr: Optional[str] = None


def query(
    scene_like: SceneLike,
    expr: Optional[str] = None,
    *,
    scope: str = "any",
    before: Optional[str] = None,
    after: Optional[str] = None,
    within_frames: Optional[int] = None,
    policy: UnknownPolicy = UnknownPolicy.FREE,
) -> QueryResult:
    """Run SceneQL `expr` over `scene_like` under `scope`, returning a `QueryResult`.

    - scope 'any'/'all': `expr` is required (a per-frame predicate); 'before'/'after' must be unset.
    - scope 'transition': `before`, `after`, `within_frames` are required; `expr` must be unset.

    Raises ValueError for a malformed scope/argument combination, NameError for an unknown
    identifier, UnsafeExpression/SyntaxError for an expression outside the whitelist (propagated
    from `safe_eval` -- a bad query fails loudly, never silently).
    """
    sc = _as_scene(scene_like)
    ts = list(sc.times())
    n = len(ts)

    if scope in ("any", "all"):
        if expr is None:
            raise ValueError(f"scope {scope!r} requires `expr`")
        if before is not None or after is not None or within_frames is not None:
            raise ValueError(f"scope {scope!r} must not set transition fields (before/after/within_frames)")
        matched_frames = [t for t in ts if _frame_true(scene_like, t, expr, policy)]
        matched = len(matched_frames) == n if scope == "all" else len(matched_frames) > 0
        # 'all' over an empty scene is vacuously... there are no frames to satisfy; keep it False to
        # avoid a vacuous-true claim on a degenerate scene.
        if scope == "all" and n == 0:
            matched = False
        return QueryResult(matched=matched, matched_frames=matched_frames, n_frames=n, scope=scope, expr=expr)

    if scope == "transition":
        if expr is not None:
            raise ValueError("scope 'transition' must not set `expr`; use before/after")
        if before is None or after is None or within_frames is None:
            raise ValueError("scope 'transition' requires before, after, and within_frames")
        if within_frames < 1:
            raise ValueError("within_frames must be a positive int")
        before_frames = [t for t in ts if _frame_true(scene_like, t, before, policy)]
        matched = False
        for i in before_frames:
            for j in ts:
                if 0 < j - i <= within_frames and _frame_true(scene_like, j, after, policy):
                    matched = True
                    break
            if matched:
                break
        return QueryResult(matched=matched, matched_frames=before_frames, n_frames=n, scope=scope, expr=None)

    raise ValueError(f"unknown scope: {scope!r}")
