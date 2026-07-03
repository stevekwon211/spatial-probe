# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""autolabel_v0 unit gates, pinned BEFORE the sealed corpus run (pre-reg
`experiments/autolabel_v0/preregistration.md`):

1. the greedy nuScenes-style matcher on a fixed 3-GT / 3-proposal layout (hand-derived TP set).
2. precision/recall/F1 from a hand toy confusion (the `_roc_auc` lesson — numpy, no sklearn).
3. connected-component proposer finds the right number of clusters + centroids on a toy grid,
   and the [tau, tau_max] gate drops specks and structure.
4. range slicing puts a GT box in the right near/mid/far bin.
"""
from __future__ import annotations

import math

import numpy as np

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.scene import TrackedBox

from experiments.autolabel_v0.run import (
    Proposal,
    _fp_after_other,
    match_greedy,
    pr_f1,
    propose_from_occupancy,
    range_bin,
    score_frame,
)


# --- 1. greedy matcher --------------------------------------------------------------------------
def test_match_greedy_hand_layout():
    # GT at x=5,10,30 (y=0); proposals at x=5.5 (near g0), 11 (near g1), 100 (far from all).
    gts = [(5.0, 0.0), (10.0, 0.0), (30.0, 0.0)]
    props = [(5.5, 0.0), (11.0, 0.0), (100.0, 0.0)]
    tp, fn_idx, fp_idx = match_greedy(gts, props, d=2.0)
    assert tp == 2                       # g0<->p0 (0.5), g1<->p1 (1.0); g2 unmatched
    assert set(fn_idx) == {2}            # g2 has no proposal within 2 m
    assert set(fp_idx) == {2}            # p2 matches nothing


def test_match_greedy_nearest_first_not_greedy_wrong():
    # one proposal between two GTs: must go to the NEARER gt, leaving the farther as FN.
    gts = [(0.0, 0.0), (3.0, 0.0)]
    props = [(1.0, 0.0)]                 # 1.0 from g0, 2.0 from g1
    tp, fn_idx, fp_idx = match_greedy(gts, props, d=4.0)
    assert tp == 1 and set(fn_idx) == {1} and fp_idx == []


def test_match_greedy_distance_gate():
    tp, fn_idx, fp_idx = match_greedy([(0.0, 0.0)], [(3.0, 0.0)], d=2.0)
    assert tp == 0 and set(fn_idx) == {0} and set(fp_idx) == {0}


# --- 2. rates -----------------------------------------------------------------------------------
def test_pr_f1_hand_values():
    m = pr_f1(tp=6, fp=2, fn=1)
    assert math.isclose(m["precision"], 6 / 8)
    assert math.isclose(m["recall"], 6 / 7)
    assert math.isclose(m["f1"], 2 * 6 / (2 * 6 + 2 + 1))


def test_pr_f1_zero_safe():
    m = pr_f1(tp=0, fp=0, fn=0)
    assert math.isnan(m["precision"]) and math.isnan(m["recall"])


# --- 3. proposer + size gate --------------------------------------------------------------------
# grid: voxel 1.0, origin (0,0,0), ground 0.5, ego.height 1.9 -> ego-height slab keeps z-index 1,2
# (world z 1.0, 2.0 in (0.5, 2.4]); z-index 0,3+ are excluded. Blobs live at z 1:3 to sit in-slab.
def _grid_with_blobs() -> OccupancyGrid:
    occ = np.full((40, 40, 6), FREE, dtype=int)
    occ[10:12, 10:12, 1:3] = OCCUPIED   # blob A: 2x2x2 = 8 voxels, fully in the ego-height slab
    occ[30, 30, 1] = OCCUPIED           # blob B: 1-voxel speck (in-slab), gated by tau
    return OccupancyGrid(occ, 1.0, (0.0, 0.0, 0.0), 0.5)


def test_propose_from_occupancy_finds_blobs_and_gates_speck():
    grid = _grid_with_blobs()
    ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=0.0)
    props = propose_from_occupancy(grid, ego, tau=2, tau_max=2000)
    assert len(props) == 1                       # the 1-voxel speck is gated by tau=2
    p = props[0]
    assert p.n_voxels == 8
    assert abs(p.cx - 10.5) < 1e-6 and abs(p.cy - 10.5) < 1e-6  # centroid of the 2x2 BEV footprint


def test_propose_tau_max_gates_structure():
    occ = np.full((40, 40, 6), FREE, dtype=int)
    occ[0:25, 0:25, 1:3] = OCCUPIED             # 25*25*2 = 1250-voxel "wall" (in-slab)
    grid = OccupancyGrid(occ, 1.0, (0.0, 0.0, 0.0), 0.5)
    ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=0.0)
    assert propose_from_occupancy(grid, ego, tau=2, tau_max=1000) == []   # 1250 > 1000 -> structure
    assert len(propose_from_occupancy(grid, ego, tau=2, tau_max=2000)) == 1


# --- 4. range binning ---------------------------------------------------------------------------
def test_range_bin():
    assert range_bin(10.0) == "near"      # < 20
    assert range_bin(25.0) == "mid"       # [20, 35]
    assert range_bin(40.0) == "far"       # > 35


# --- 5. 'other'-explained proposals are removed from FP (sealed fairness rule) ------------------
def test_fp_after_other_removes_explained():
    prop_xy = [(10.0, 0.0), (50.0, 0.0)]      # both are unmatched-to-scored-GT (FP candidates)
    other_xy = [(10.3, 0.0)]                  # a barrier explains proposal 0, not proposal 1
    assert _fp_after_other(prop_xy, [0, 1], other_xy, d=2.0) == 1   # only proposal 1 is a true FP
    assert _fp_after_other(prop_xy, [0, 1], [], d=2.0) == 2         # no 'other' -> both FP


def test_score_frame_slices_and_fp_filter():
    gt = [{"xy": (5.0, 0.0), "label": "vehicle", "range": 5.0, "rbin": "near"}]
    other = [(50.0, 0.0)]
    props = [Proposal(5.2, 0.0, 8), Proposal(50.1, 0.0, 8)]  # p0 hits GT, p1 explained by 'other'
    rec = score_frame(gt, other, props)["2.0"]
    assert rec["overall"] == [1, 0, 0]                     # 1 TP, 0 FP (p1 removed), 0 FN
    assert rec["class:vehicle"] == [1, 0, 0]
    assert rec["rbin:near"] == [1, 0, 0]
