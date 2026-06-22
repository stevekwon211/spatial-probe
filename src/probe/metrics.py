# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Pure scoring kernel: denotation P/R/F1, cluster bootstrap CIs, and the relative-gap curve.

Scope (per `experiments/occquery_v0/preregistration.md`, committed 2026-06-21): **H1 (expressivity)
is the sole headline.** The relative gap scored here is an INTERNAL-CONSISTENCY comparison against
whatever oracle the caller supplies, NOT external denotation correctness -- on this hardware the only
constructible oracle shares the predicate's data source (same Occ3D/nuScenes LiDAR), so any real-data
number from this module is labeled "internal-consistency", never "externally validated".

This module computes nothing about whether an oracle is independent, and ENFORCES no movable gate:
there is no absolute F1 cutoff anywhere, the decision is read off a curve, and `taus` is caller-
supplied with no default -- a reviewer cannot move a number that does not exist (preregistration: the
0.90 gate is removed; every threshold is a curve, never a cutoff).

Two F1 policies share ONE core so there is a single P/R/F1 definition in the repo:
- ``report``: empty retrieved -> precision 1.0, empty truth -> recall 1.0 (the legacy run.py:_prf1
  convention, kept byte-compatible so `prf1` is a drop-in).
- ``strict``: NO positives in truth -> F1 is **undefined (NaN)**, never a vacuous 1.0. Used by the
  bootstrap and the gap curve, because the empty=1.0 convention silently inverts the gap exactly where
  real val lives (sparse / zero positives -- the mini set has none).

Deferred until an oracle module exists (it is demoted/unbuilt per the preregistration): continuous
clearance/free-width error, tolerance curves, and inf-aware sentinel accuracy. Building them now would
be a core abstraction with no second consumer (PLAN: core gains code only on the second use).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EMPTY_POLICIES = ("report", "strict")


# --- 1. the single P/R/F1 core ---------------------------------------------------------------------


def _f1_core(tp: int, fp: int, fn: int, *, empty_policy: str) -> float:
    """F1 from confusion counts under an explicit empty-domain policy. The ONLY F1 definition here.

    ``report`` reproduces run.py:_prf1 (empties -> 1.0). ``strict`` returns NaN when the truth set has
    no positives (F1 of a positive-class metric is undefined with no positive class), and 0.0 when
    positives exist but nothing was retrieved (a real miss, not undefined)."""
    if empty_policy not in _EMPTY_POLICIES:
        raise ValueError(f"empty_policy must be one of {_EMPTY_POLICIES}, got {empty_policy!r}")
    retrieved = tp + fp
    positives = tp + fn
    if empty_policy == "strict":
        if positives == 0:
            return math.nan
        precision = tp / retrieved if retrieved else 0.0
        recall = tp / positives
    else:  # report
        precision = tp / retrieved if retrieved else 1.0
        recall = tp / positives if positives else 1.0
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def prf1(retrieved: set, truth: set) -> dict:
    """Denotation P/R/F1 over scene-name sets. Byte-compatible drop-in for run.py:_prf1."""
    r, t = set(retrieved), set(truth)
    tp = len(r & t)
    precision = tp / len(r) if r else 1.0
    recall = tp / len(t) if t else 1.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(_f1_core(tp, len(r - t), len(t - r), empty_policy="report"), 3),
        "false_positives": sorted(r - t),
        "false_negatives": sorted(t - r),
    }


def _counts(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    """(tp, fp, fn) from two boolean masks of equal length."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    return tp, fp, fn


def f1_from_masks(pred: np.ndarray, truth: np.ndarray, *, empty_policy: str = "strict") -> float:
    """F1 of a boolean predicted mask vs a boolean truth mask (strict policy by default)."""
    tp, fp, fn = _counts(pred, truth)
    return _f1_core(tp, fp, fn, empty_policy=empty_policy)


def _stat_from_masks(pred: np.ndarray, truth: np.ndarray, metric: str) -> float:
    """precision / recall / f1 from masks under the strict policy (NaN when no positives)."""
    tp, fp, fn = _counts(pred, truth)
    if (tp + fn) == 0:  # no positives -> every positive-class metric undefined
        return math.nan
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else 0.0
    if metric == "recall":
        return tp / (tp + fn)
    if metric == "f1":
        return _f1_core(tp, fp, fn, empty_policy="strict")
    raise ValueError(f"metric must be precision/recall/f1, got {metric!r}")


# --- 2. cluster bootstrap CI (the scene is the resampling unit) -------------------------------------


@dataclass(frozen=True)
class CI:
    """A percentile confidence interval over bootstrap replicates.

    ``defined=False`` means under-powered / undefined (fewer than ``min_positive`` positive units): lo/hi
    are NaN and the interval must NOT be read as a result. ``defined=True`` means only that the statistic
    was bootstrappable (>= min_positive positives) -- NOT that it is well-powered. At the ``min_positive``
    boundary a sizable share of resamples can miss the positives and drop to NaN, leaving a WIDE,
    near-vacuous [lo, hi]; read ``effective_fraction`` and the interval WIDTH, never ``defined`` alone.
    A low-power interval is already inert at the decision layer: ``gap_decision_summary`` counts a tau
    only when ``gap_ci.lo > 0``, which a wide straddling interval fails, so low power cannot manufacture
    a positive."""

    point: float
    lo: float
    hi: float
    n_boot: int
    level: float
    n_effective: int
    defined: bool
    reason: str = ""

    @property
    def effective_fraction(self) -> float:
        """Share of bootstrap replicates that were finite; the rest hit a no-positive resample and were
        dropped. ~1.0 = well-powered; well below 1.0 = a shaky interval even when ``defined`` is True."""
        return self.n_effective / self.n_boot if self.n_boot else math.nan


def _percentile_ci(point: float, samples: list[float], n_boot: int, level: float) -> CI:
    """Percentile CI over the finite bootstrap replicates; undefined if none are finite."""
    if not samples:
        return CI(point, math.nan, math.nan, n_boot, level, 0, False, "all bootstrap replicates undefined")
    alpha = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(samples, [alpha, 100.0 - alpha])
    return CI(float(point), float(lo), float(hi), n_boot, level, len(samples), True)


def bootstrap_f1(
    per_scene: dict,
    *,
    metric: str = "f1",
    n_boot: int = 1000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
    min_positive: int = 2,
) -> CI:
    """Cluster bootstrap CI for a denotation metric. ``per_scene`` maps scene_name -> (pred, truth)
    booleans for ONE query. Scene KEYS are the resampling unit (frames within a scene are correlated).

    Keys are sorted before drawing, so the CI is reproducible regardless of dict insertion order. With
    fewer than ``min_positive`` positive scenes the metric cannot be bootstrapped (a single positive
    unit yields a vacuous [0,1] interval), so the CI is returned ``defined=False`` -- the wrong state
    is unrepresentable as a usable interval."""
    keys = sorted(per_scene)
    pred = np.array([bool(per_scene[k][0]) for k in keys])
    truth = np.array([bool(per_scene[k][1]) for k in keys])
    n_pos = int(np.count_nonzero(truth))
    point = _stat_from_masks(pred, truth, metric)
    if n_pos < min_positive:
        return CI(point, math.nan, math.nan, n_boot, level, n_pos, False,
                  f"under-powered: n_pos={n_pos} < min_positive={min_positive}")
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(keys)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stat = _stat_from_masks(pred[idx], truth[idx], metric)
        if math.isfinite(stat):
            samples.append(stat)
    return _percentile_ci(point, samples, n_boot, level)


# --- 3. the relative-gap curve (consistency comparison, NOT the headline) ---------------------------


@dataclass(frozen=True)
class GapPoint:
    """One decision-threshold point: free-space F1 minus best-box-only F1, with a paired CI on the
    difference. Both backends are scored against the SAME truth, so the gap is not a different-
    yardstick artifact."""

    tau: float
    f1_free: float
    f1_box: float
    gap: float
    gap_ci: CI


def _reduce_scene(score: np.ndarray, scene_ids: np.ndarray, units: np.ndarray, direction: str, reduce: str) -> np.ndarray:
    """Collapse per-row scores to one score per scene, hoisted out of the tau loop. For
    direction='greater' a scene fires when score>=tau, so reduce='any' -> the scene's MAX score,
    reduce='all' -> its MIN; direction='less' mirrors it."""
    use_max = (direction == "greater") == (reduce == "any")
    out = np.empty(len(units), dtype=float)
    for i, u in enumerate(units):
        rows = score[scene_ids == u]
        out[i] = rows.max() if use_max else rows.min()
    return out


def _reduce_truth(truth_mask: np.ndarray, scene_ids: np.ndarray, units: np.ndarray, reduce: str) -> np.ndarray:
    out = np.empty(len(units), dtype=bool)
    for i, u in enumerate(units):
        rows = truth_mask[scene_ids == u]
        out[i] = rows.any() if reduce == "any" else rows.all()
    return out


def _predict(scene_score: np.ndarray, tau: float, direction: str) -> np.ndarray:
    return scene_score >= tau if direction == "greater" else scene_score <= tau


def _tau_rng(seed: int, tau: float) -> np.random.Generator:
    """A generator seeded by the tau VALUE, so a tau's CI depends only on (seed, tau) -- never on the
    other taus in the grid. Adding/removing/reordering taus cannot move an existing tau's interval,
    which closes the 'add taus until one is significant' forking path."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(round(float(tau) * 1_000_000))]))


def _paired_gap_ci(pred_free: np.ndarray, pred_box: np.ndarray, truth: np.ndarray,
                   n_pos: int, n_boot: int, level: float, rng: np.random.Generator, min_positive: int) -> CI:
    """Paired bootstrap CI on (F1_free - F1_box): resample scene units ONCE per replicate and recompute
    BOTH F1s on that one resample, so positive cross-correlation tightens the interval correctly."""
    point = f1_from_masks(pred_free, truth) - f1_from_masks(pred_box, truth)
    if n_pos < min_positive:
        return CI(point, math.nan, math.nan, n_boot, level, n_pos, False,
                  f"under-powered: n_pos={n_pos} < min_positive={min_positive}")
    n = len(truth)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gap = f1_from_masks(pred_free[idx], truth[idx]) - f1_from_masks(pred_box[idx], truth[idx])
        if math.isfinite(gap):
            samples.append(gap)
    return _percentile_ci(point, samples, n_boot, level)


def gap_curve(
    free_score: np.ndarray,
    box_score: np.ndarray,
    truth_mask: np.ndarray,
    scene_ids: np.ndarray,
    *,
    taus: np.ndarray,
    direction: str = "greater",
    reduce: str = "any",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    min_positive: int = 2,
) -> list[GapPoint]:
    """The relative-gap CONSISTENCY curve over the decision threshold ``taus`` (caller-supplied, no
    default -- a pre-registered knob, never a movable gate).

    ``free_score`` / ``box_score`` are the two backends' continuous scores per row; ``truth_mask`` is
    the oracle membership per row; ``scene_ids`` labels each row's bootstrap unit. Rows are reduced to
    scenes by ``reduce`` ('any' matches retrieval scope='any'); both backends are thresholded at each
    tau and scored against the SAME reduced truth. Each tau carries a paired bootstrap CI on the gap,
    seeded by the tau value. With fewer than ``min_positive`` positive scenes the whole curve is
    undefined (the mini reality), reported as such -- never a spurious gap."""
    if direction not in ("greater", "less"):
        raise ValueError(f"direction must be greater/less, got {direction!r}")
    if reduce not in ("any", "all"):
        raise ValueError(f"reduce must be any/all, got {reduce!r}")
    free_score = np.asarray(free_score, dtype=float)
    box_score = np.asarray(box_score, dtype=float)
    truth_mask = np.asarray(truth_mask, dtype=bool)
    scene_ids = np.asarray(scene_ids)
    units = np.unique(scene_ids)
    scene_free = _reduce_scene(free_score, scene_ids, units, direction, reduce)
    scene_box = _reduce_scene(box_score, scene_ids, units, direction, reduce)
    scene_truth = _reduce_truth(truth_mask, scene_ids, units, reduce)
    n_pos = int(np.count_nonzero(scene_truth))
    out = []
    for tau in np.asarray(taus, dtype=float):
        pred_free = _predict(scene_free, tau, direction)
        pred_box = _predict(scene_box, tau, direction)
        f_free = f1_from_masks(pred_free, scene_truth)
        f_box = f1_from_masks(pred_box, scene_truth)
        ci = _paired_gap_ci(pred_free, pred_box, scene_truth, n_pos, n_boot, level,
                            _tau_rng(seed, tau), min_positive)
        out.append(GapPoint(float(tau), f_free, f_box, f_free - f_box, ci))
    return out


def gap_decision_summary(curve: list[GapPoint]) -> dict:
    """Describe the curve WITHOUT deciding pass/fail. The primary signal is the fraction of taus whose
    paired gap CI excludes 0 from below (gap_ci.lo>0) -- grid-shape-robust, with no AUC (an AUC is a
    movable number that leaks back through the grid). A human reads the relative claim off this."""
    defined = [g for g in curve if g.gap_ci.defined]
    positive = [g for g in defined if g.gap_ci.lo > 0.0]
    best = max(defined, key=lambda g: g.gap, default=None)
    taus_positive = [g.tau for g in positive]
    return {
        "n_taus": len(curve),
        "n_defined": len(defined),
        "all_undefined": len(defined) == 0,
        "frac_taus_gap_ci_positive": (len(positive) / len(defined)) if defined else math.nan,
        "tau_range_gap_ci_positive": [min(taus_positive), max(taus_positive)] if taus_positive else [],
        "max_gap": best.gap if best is not None else math.nan,
        "max_gap_tau": best.tau if best is not None else math.nan,
    }
