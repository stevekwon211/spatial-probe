# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Pin the L1 numpy FREE-class set-metrics against HAND-COMPUTED values (the `_roc_auc` lesson:
a hand-rolled metric is unit-tested against values computed by hand on a fixed toy confusion before
it is trusted). Pure numpy -- no sklearn, no torch (repo rule)."""
from __future__ import annotations

import math

import numpy as np

from experiments.occquery_v0.l1_denotation_occ3d import (
    confusion_from_masks,
    free_set_metrics,
)


def test_fixed_toy_confusion_hand_computed():
    # Fixed confusion (FREE = positive): TP=6, FP=2, FN=1, TN=3.
    m = free_set_metrics(6, 2, 1, 3)
    # precision = 6/(6+2) = 0.75
    assert math.isclose(m["precision"], 0.75)
    # recall = 6/(6+1) = 0.857142857...
    assert math.isclose(m["recall"], 6 / 7)
    # IoU = 6/(6+2+1) = 0.6666...
    assert math.isclose(m["iou"], 6 / 9)
    # F1 = 2*6/(2*6+2+1) = 12/15 = 0.8
    assert math.isclose(m["f1"], 0.8)
    # false-block rate = FN/(TP+FN) = 1/7
    assert math.isclose(m["false_block_rate"], 1 / 7)
    # miss rate = FP/(FP+TN) = 2/5 = 0.4
    assert math.isclose(m["miss_rate"], 0.4)
    # false-block rate is exactly 1 - recall
    assert math.isclose(m["false_block_rate"], 1.0 - m["recall"])


def test_all_free_prediction_degenerate():
    # all-free predicts every cell FREE on a GT of 8 free + 4 blocked:
    # TP=8, FP=4, FN=0, TN=0.
    m = free_set_metrics(8, 4, 0, 0)
    assert math.isclose(m["precision"], 8 / 12)
    assert math.isclose(m["recall"], 1.0)
    assert math.isclose(m["iou"], 8 / 12)
    # F1 = 2*8/(16+4) = 16/20 = 0.8
    assert math.isclose(m["f1"], 0.8)
    assert math.isclose(m["false_block_rate"], 0.0)   # never blocks
    assert math.isclose(m["miss_rate"], 1.0)          # misses every obstacle


def test_all_blocked_prediction_degenerate():
    # all-blocked predicts every cell BLOCKED on a GT of 8 free + 4 blocked:
    # TP=0, FP=0, FN=8, TN=4.
    m = free_set_metrics(0, 0, 8, 4)
    assert math.isnan(m["precision"])  # 0/0 -> NaN, never a silent 0
    assert math.isclose(m["recall"], 0.0)
    assert math.isclose(m["iou"], 0.0)
    assert math.isclose(m["f1"], 0.0)
    assert math.isclose(m["false_block_rate"], 1.0)
    assert math.isclose(m["miss_rate"], 0.0)


def test_perfect_identity_iou_one():
    # observed == GT (the degeneracy the run actually hits): IoU = F1 = 1.0, no false-block, no miss.
    m = free_set_metrics(10, 0, 0, 5)
    assert math.isclose(m["iou"], 1.0)
    assert math.isclose(m["f1"], 1.0)
    assert math.isclose(m["precision"], 1.0)
    assert math.isclose(m["recall"], 1.0)
    assert math.isclose(m["false_block_rate"], 0.0)
    assert math.isclose(m["miss_rate"], 0.0)


def test_confusion_from_masks_matches_hand_count():
    # pred (observed FREE) vs ref (GT FREE), booleans. Hand-count the four cells.
    pred = np.array([[True, True, False], [False, True, True]])
    ref = np.array([[True, False, False], [True, True, False]])
    # cell-by-cell (pred, ref):
    #  (T,T)=TP (T,F)=FP (F,F)=TN | (F,T)=FN (T,T)=TP (T,F)=FP
    # TP=2, FP=2, FN=1, TN=1
    tp, fp, fn, tn = confusion_from_masks(pred, ref)
    assert (tp, fp, fn, tn) == (2, 2, 1, 1)
    assert tp + fp + fn + tn == pred.size


def test_confusion_empty_band():
    # zero cells -> all zero, metrics NaN (undefined, never silent 0 for ratios over empty sets)
    tp, fp, fn, tn = confusion_from_masks(np.zeros((0,), bool), np.zeros((0,), bool))
    assert (tp, fp, fn, tn) == (0, 0, 0, 0)
    m = free_set_metrics(0, 0, 0, 0)
    assert math.isnan(m["iou"]) and math.isnan(m["f1"])
