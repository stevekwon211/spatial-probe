# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Pin the SYNTHETIC denotation-mechanism harness: the constructed-GT placement, the raycast
occlusion, and a HAND-CHECKED single-blocker scene (one blocker at 10 m -> the predicate denotes
BLOCKED there and FREE elsewhere). Pure numpy; no sklearn/torch (repo rule)."""
from __future__ import annotations

import numpy as np

from probe.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, UnknownPolicy
from probe.predicates.freepath import free_along_ego_path

from experiments.occquery_v0.synth_denotation import (
    BAND_MARGIN,
    GROUND_H,
    HORIZON,
    VOXEL,
    _blank,
    _build_scenes,
    _ego,
    _ego_cell,
    _grade_scene,
    _occlude,
    _path_truly_blocked,
    _place_block,
    band_blocked_bev,
)


def _grid(occ):
    return OccupancyGrid(occ, VOXEL, (0.0, 0.0, 0.0), GROUND_H)


# --- constructed-GT placement -------------------------------------------------------------------
def test_place_block_sets_occupied_at_expected_world_cell():
    ego = _ego(10.0)
    occ = _blank()
    _place_block(occ, ego, forward=10.0, lateral=0.0, lat_w=1.0, fwd_d=1.0)
    # heading 0: world x = ego_x(5)+10 = 15 -> i=30; y = ego_y(10)+0 = 10 -> j=20; z band k=1..4
    assert occ[30, 20, 2] == OCCUPIED
    # outside the block stays FREE
    assert occ[20, 20, 2] == FREE
    # ground layer (k=0) is never filled by the column (HEIGHT_K starts at 1)
    assert occ[30, 20, 0] == FREE


def test_path_truly_blocked_independent_label():
    ego = _ego(10.0)
    occ_blocked = _blank()
    _place_block(occ_blocked, ego, forward=10.0, lateral=0.0, lat_w=1.0, fwd_d=1.0)
    assert _path_truly_blocked(occ_blocked, _grid(occ_blocked), ego) is True
    # a blocker off the corridor (lateral 5 m) is NOT a path block
    occ_off = _blank()
    _place_block(occ_off, ego, forward=10.0, lateral=5.0, lat_w=1.0, fwd_d=1.0)
    assert _path_truly_blocked(occ_off, _grid(occ_off), ego) is False
    # empty grid -> not blocked
    assert _path_truly_blocked(_blank(), _grid(_blank()), ego) is False


# --- HAND-CHECKED single blocker at 10 m --------------------------------------------------------
def test_hand_checked_single_blocker_at_10m():
    ego = _ego(10.0)  # reach = 4.6/2 + 10*1.0 = 12.3 m -> 10 m is within reach
    occ = _blank()
    _place_block(occ, ego, forward=10.0, lateral=0.0, lat_w=1.0, fwd_d=1.0)
    grid = _grid(occ)

    # 1) predicate verdict: BLOCKED with the blocker, FREE on the empty grid
    assert free_along_ego_path(grid, ego, HORIZON) is False
    assert free_along_ego_path(_grid(_blank()), ego, HORIZON) is True

    # 2) cell denotation: the band cell at forward~10, lateral 0 is BLOCKED; a near cell at
    #    forward~3 on the centerline is FREE.
    blocked = band_blocked_bev(grid, ego, horizon=HORIZON, band_margin=BAND_MARGIN,
                               unknown_policy=UnknownPolicy.FREE)
    res = VOXEL
    band_half = ego.width / 2.0 + BAND_MARGIN
    li0 = int(round(band_half / res))            # centerline lateral index
    fi10 = int(round(10.0 / res))                # forward 10 m
    fi3 = int(round(3.0 / res))                  # forward 3 m
    # blocked at the 10 m centerline (allow +-1 cell for the column's finite depth)
    assert blocked[fi10 - 1:fi10 + 2, li0 - 1:li0 + 2].any()
    # free near the ego at 3 m on the centerline
    assert not blocked[fi3, li0]


# --- raycast occlusion --------------------------------------------------------------------------
def test_occlusion_hides_obstacle_behind_a_visible_one():
    ego = _ego(10.0)
    occ = _blank()
    _place_block(occ, ego, forward=6.0, lateral=0.0, lat_w=1.0, fwd_d=1.0)   # front, visible
    _place_block(occ, ego, forward=9.0, lateral=0.0, lat_w=1.0, fwd_d=1.0)   # behind, occluded
    grid = _grid(occ)
    ego_cell = _ego_cell(grid, ego)
    obs = _occlude(occ, ego_cell)

    # front block (forward 6, fwd_d 1) spans world x 10.5..11.5 -> i=21..23; front FACE = i=21.
    # rear block (forward 9) world x 13.5..14.5 -> i=27..29; centerline j=20, k=2.
    assert occ[21, 20, 2] == OCCUPIED and occ[28, 20, 2] == OCCUPIED       # both true obstacles
    assert obs[21, 20, 2] == OCCUPIED      # the front FACE stays visible (first hit on the ray)
    assert obs[28, 20, 2] == UNKNOWN       # the rear block is fully occluded -> UNKNOWN
    assert obs[22, 20, 2] == UNKNOWN       # even the front block's INTERIOR is occluded by its face

    # under unknown=FREE the occluded rear obstacle reads FREE in the band -> a MISS vs the TRUE GT
    ref_blocked = band_blocked_bev(grid, ego, horizon=HORIZON, band_margin=BAND_MARGIN,
                                   unknown_policy=UnknownPolicy.FREE)
    obs_blocked = band_blocked_bev(_grid(obs), ego, horizon=HORIZON, band_margin=BAND_MARGIN,
                                   unknown_policy=UnknownPolicy.FREE)
    # TRUE GT blocks more cells than the occluded observation sees (the rear obstacle is missed)
    assert int(ref_blocked.sum()) > int(obs_blocked.sum())


def test_occlusion_keeps_a_lone_visible_obstacle():
    ego = _ego(10.0)
    occ = _blank()
    _place_block(occ, ego, forward=8.0, lateral=0.0, lat_w=1.0, fwd_d=1.0)
    grid = _grid(occ)
    obs = _occlude(occ, _ego_cell(grid, ego))
    # block at forward 8 (center world x=13, fwd_d 1) spans i=25..27; its front FACE (i=25) is seen
    assert obs[25, 20, 2] == OCCUPIED   # nearest face has clear line-of-sight -> stays OCCUPIED


# --- mechanism invariant: unknown=FREE never FALSE-BLOCKS truly-free space ----------------------
def test_unknown_free_never_false_blocks_truly_free_space():
    # Across constructed scenes, the FREE-class FN (observed BLOCKED where GT FREE) must be exactly 0
    # under unknown=FREE: occlusion only turns OCCUPIED->UNKNOWN->free, never free->blocked.
    rng = np.random.default_rng(0)
    scenes = _build_scenes(24, rng)
    for sc in scenes:
        rec = _grade_scene(sc)
        tp, fp, fn, tn = rec["free"]
        assert fn == 0, f"{rec['name']}: unknown=FREE false-blocked {fn} truly-free cells"


def test_obstacle_bearing_scenes_are_nonvacuous():
    # the construction must actually produce blocked band cells (non-vacuity, unlike the 99.5%-free
    # Occ3D L1 band)
    rng = np.random.default_rng(0)
    scenes = _build_scenes(40, rng)
    recs = [_grade_scene(sc) for sc in scenes]
    obstacle_bearing = [r for r in recs if r["ref_blocked_cells"] > 0]
    assert len(obstacle_bearing) >= 10
    total_blocked = sum(r["ref_blocked_cells"] for r in obstacle_bearing)
    total_cells = sum(r["ref_total_cells"] for r in obstacle_bearing)
    blocked_rate = total_blocked / total_cells
    assert blocked_rate > 0.02, f"band blocked-rate {blocked_rate:.4f} too low -> vacuous"
