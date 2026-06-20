# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""probe.retrieval scope evaluation -- frame-scoped (any/all) and temporal transition queries.

`blocked_then_clears` must be a real temporal pattern (blocked, then clear later), not just
"some frame is blocked".
"""
import numpy as np

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid, UnknownPolicy
from probe.query_spec import Query
from probe.retrieval import scene_matches
from probe.scene import Frame, Scene


def _grid(occ: np.ndarray) -> OccupancyGrid:
    return OccupancyGrid(occ, 1.0, (0.0, 0.0, 0.0), 0.5)


def _blank() -> np.ndarray:
    return np.full((30, 30, 30), FREE, dtype=int)


def _blocked() -> OccupancyGrid:
    occ = _blank()
    for lat in (-1, 0, 1):
        occ[12, 10 + lat, 2] = OCCUPIED  # wall 2 m ahead across the ego corridor at (10, 10)
    return _grid(occ)


def _scene(grids: list[OccupancyGrid]) -> Scene:
    ego = EgoPose((10, 10, 0), 0.0, speed=5.0)
    return Scene(tuple(Frame(g, ego, float(i)) for i, g in enumerate(grids)), "s")


def _transition(within: int = 3) -> Query:
    return Query(
        id="t", nl="", backend="occupancy", status="implemented", scope="transition",
        refav_expressible=False, rationale="",
        before="not free_along_ego_path(scene, t, horizon=0.0)",
        after="free_along_ego_path(scene, t, horizon=0.0)", within_frames=within,
    )


def _any(predicate: str) -> Query:
    return Query(id="a", nl="", backend="occupancy", status="implemented", scope="any",
                 refav_expressible=False, rationale="", predicate=predicate)


def _all(predicate: str) -> Query:
    return Query(id="a", nl="", backend="occupancy", status="implemented", scope="all",
                 refav_expressible=False, rationale="", predicate=predicate)


_FREE = UnknownPolicy.FREE
_BLOCKED_PRED = "not free_along_ego_path(scene, t, horizon=0.0)"
_CLEAR_PRED = "free_along_ego_path(scene, t, horizon=0.0)"


def test_transition_blocked_then_clears_matches():
    s = _scene([_blocked(), _grid(_blank())])
    assert scene_matches(s, _transition(), _FREE) is True


def test_transition_clears_then_blocks_does_not_match():
    s = _scene([_grid(_blank()), _blocked()])  # wrong order
    assert scene_matches(s, _transition(), _FREE) is False


def test_transition_never_clears_does_not_match():
    s = _scene([_blocked(), _blocked()])
    assert scene_matches(s, _transition(), _FREE) is False


def test_transition_multiple_frames_clear_to_block_to_clear():
    s = _scene([_grid(_blank()), _blocked(), _grid(_blank())])  # blocked at 1, clear at 2
    assert scene_matches(s, _transition(), _FREE) is True


def test_any_scope_matches_one_frame():
    s = _scene([_grid(_blank()), _blocked()])
    assert scene_matches(s, _any(_BLOCKED_PRED), _FREE) is True


def test_all_scope_requires_every_frame():
    assert scene_matches(_scene([_grid(_blank()), _grid(_blank())]), _all(_CLEAR_PRED), _FREE) is True
    assert scene_matches(_scene([_grid(_blank()), _blocked()]), _all(_CLEAR_PRED), _FREE) is False
