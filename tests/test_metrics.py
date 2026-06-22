# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Scoring-kernel tests (occquery H3 internal-consistency arm).

These test the MATH of `probe.metrics`, the pure scoring kernel. Per the committed
`experiments/occquery_v0/preregistration.md`, H1 (expressivity) is the SOLE headline; the relative
gap scored here is an internal-consistency comparison, NOT external denotation correctness. A green
test here is a smoke test of the kernel arithmetic, never evidence about real val behavior.

The load-bearing tests are the rare-positive ones (`*_all_negative_*`, `*_one_positive_*`): they force
the kernel to report "no usable signal" instead of a vacuous F1=1.0 / a spurious gap, which is exactly
where the naive empty-set convention silently breaks on real data (the mini set has zero positives).
"""
from __future__ import annotations

import ast
import math
import pathlib

import numpy as np

from probe import metrics


# The pre-extraction run.py:33-46 behavior, frozen here as the drop-in contract. `metrics.prf1` MUST
# match this byte-for-byte so swapping it into run.py is a no-op.
def _legacy_prf1(ret: set, truth: set) -> dict:
    ret, truth = set(ret), set(truth)
    tp = len(ret & truth)
    precision = tp / len(ret) if ret else 1.0
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": sorted(ret - truth),
        "false_negatives": sorted(truth - ret),
    }


# --- 1. prf1 is a byte-compatible drop-in for run.py:_prf1 ---


def test_prf1_byte_compatible_with_legacy():
    cases = [
        ({"a", "b", "c"}, {"a", "b"}),       # mixed
        (set(), set()),                       # both empty -> legacy vacuous 1.0 (report policy)
        ({"a"}, set()),                       # empty truth
        (set(), {"a"}),                       # empty retrieved
        ({"x", "y"}, {"y", "z"}),             # partial overlap
        ({"a", "b"}, {"a", "b"}),             # perfect
    ]
    for ret, truth in cases:
        assert metrics.prf1(ret, truth) == _legacy_prf1(ret, truth), (ret, truth)


def test_prf1_keys_exact():
    out = metrics.prf1({"a"}, {"a", "b"})
    assert set(out) == {"precision", "recall", "f1", "false_positives", "false_negatives"}
    assert out["false_negatives"] == ["b"]


# --- 2. one f1 core, two policies (the report=vacuous vs strict=undefined split) ---


def test_f1_core_empty_truth_report_is_vacuous_strict_is_nan():
    # report: empty retrieved AND empty truth -> vacuous F1=1.0 (legacy convention, fine for the runner)
    assert metrics._f1_core(0, 0, 0, empty_policy="report") == 1.0
    # strict: NO positives in truth -> F1 undefined, never a vacuous 1.0 (the load-bearing fix)
    assert math.isnan(metrics._f1_core(0, 0, 0, empty_policy="strict"))
    # strict: positives present but nothing retrieved -> F1=0 (missed them), NOT undefined
    assert metrics._f1_core(0, 0, 5, empty_policy="strict") == 0.0
    # strict: retrieved garbage against zero positives -> still undefined (no positive class exists)
    assert math.isnan(metrics._f1_core(0, 7, 0, empty_policy="strict"))


def test_f1_from_masks_strict_zero_positive_is_nan():
    pred = np.array([True, True, False])
    truth = np.array([False, False, False])  # zero positives
    assert math.isnan(metrics.f1_from_masks(pred, truth, empty_policy="strict"))


# --- 3. LOAD-BEARING: truth-all-negative makes the gap UNDEFINED, never a spurious number ---


def test_gap_curve_all_negative_truth_is_undefined_not_spurious():
    # The mini reality: zero positives. The naive empty=1.0 convention manufactures gap=-1.0 punishing
    # the better backend. The kernel must instead report every tau undefined.
    n = 20
    scene_ids = np.arange(n)
    truth = np.zeros(n, dtype=bool)               # NO positives anywhere
    free = np.linspace(0, 1, n)                   # free fires on some scenes
    box = np.zeros(n)                             # box fires on none
    curve = metrics.gap_curve(
        free, box, truth, scene_ids, taus=np.array([0.25, 0.5, 0.75]), n_boot=200
    )
    assert all(not g.gap_ci.defined for g in curve)             # every tau flagged undefined
    summary = metrics.gap_decision_summary(curve)
    assert summary["all_undefined"] is True
    assert summary["n_defined"] == 0
    # crucially, no defined GapPoint reports a spurious negative gap
    assert not any(g.gap_ci.defined and g.gap < 0 for g in curve)


# --- 4. LOAD-BEARING: one positive in twenty is under-powered, not a vacuous [0,1] CI ---


def test_bootstrap_f1_one_positive_is_underpowered():
    per_scene = {f"s{i}": (False, False) for i in range(19)}
    per_scene["s19"] = (True, True)               # exactly one positive scene
    ci = metrics.bootstrap_f1(per_scene, n_boot=1000)
    assert ci.defined is False                    # cannot bootstrap a positive-class metric from 1 unit
    assert ci.n_effective == 1                    # surfaces the effective positive count
    assert "under-powered" in ci.reason


def test_bootstrap_f1_enough_positives_is_defined():
    per_scene = {f"p{i}": (True, True) for i in range(8)}
    per_scene.update({f"n{i}": (False, False) for i in range(12)})
    ci = metrics.bootstrap_f1(per_scene, n_boot=1000)
    assert ci.defined is True
    assert ci.point == 1.0                        # all 8 positives retrieved correctly
    assert ci.lo <= ci.point <= ci.hi


def test_min_positive_boundary_reports_low_power_honestly():
    # exactly 2 positive scenes (== min_positive): bootstrappable, but a sizable share of resamples miss
    # BOTH positives and drop to NaN, so the interval is low-powered. defined=True must NOT read as solid
    # -- effective_fraction carries the truth (the n_pos=2 boundary foot-gun, pinned as a known property).
    n = 20
    sid = np.arange(n)
    truth = np.array([i < 2 for i in range(n)])       # scenes 0,1 positive (== min_positive default)
    free = np.where(truth, 1.0, 0.0)
    box = np.zeros(n)
    box[0] = 1.0                                      # box catches only one of the two positives
    pt = metrics.gap_curve(free, box, truth, sid, taus=np.array([0.5]), n_boot=1000, seed=2)[0]
    assert pt.gap_ci.defined is True                  # n_pos == min_positive -> bootstrappable
    assert pt.gap_ci.n_effective < pt.gap_ci.n_boot   # resamples that missed both positives were dropped
    assert pt.gap_ci.effective_fraction < 1.0         # shakiness is a named, visible field, not hand-math


def test_decision_summary_ignores_wide_or_straddling_ci():
    # Low power surfaces as a wide CI straddling 0. The decision signal counts a tau ONLY when the gap CI
    # excludes 0 from below (lo>0), so a shaky/vacuous interval cannot manufacture a positive -- the
    # foot-gun is contained at the decision layer (asserted, not merely reasoned).
    def gp(tau, lo, hi):
        ci = metrics.CI(point=(lo + hi) / 2, lo=lo, hi=hi, n_boot=1000, level=0.95, n_effective=900, defined=True)
        return metrics.GapPoint(tau, 0.0, 0.0, (lo + hi) / 2, ci)

    curve = [gp(0.3, -0.4, 0.9), gp(0.5, -0.1, 0.8), gp(0.7, 0.2, 0.6)]  # only tau=0.7 excludes 0
    summary = metrics.gap_decision_summary(curve)
    assert summary["frac_taus_gap_ci_positive"] == 1 / 3   # the two straddling CIs do not count
    assert summary["tau_range_gap_ci_positive"] == [0.7, 0.7]


# --- 5. determinism does NOT depend on caller dict ordering (sorted units) ---


def test_bootstrap_determinism_independent_of_dict_order():
    base = {"a": (True, True), "b": (True, True), "c": (False, True), "d": (False, False)}
    shuffled = {k: base[k] for k in ["d", "b", "a", "c"]}     # same data, different insertion order
    ci1 = metrics.bootstrap_f1(base, n_boot=500, rng=np.random.default_rng(0))
    ci2 = metrics.bootstrap_f1(shuffled, n_boot=500, rng=np.random.default_rng(0))
    assert (ci1.point, ci1.lo, ci1.hi) == (ci2.point, ci2.lo, ci2.hi)


# --- 6. anti-forking-path: a tau's CI depends ONLY on its value, not the rest of the grid ---


def _signal_fixture():
    n = 30
    scene_ids = np.arange(n)
    truth = np.array([i < 15 for i in range(n)])          # 15 positives
    free = np.where(truth, 1.0, 0.0)                      # free separates positives perfectly
    box = np.full(n, 0.5)                                 # box cannot discriminate
    return free, box, truth, scene_ids


def test_gap_curve_tau_ci_is_grid_invariant():
    free, box, truth, sid = _signal_fixture()
    only = metrics.gap_curve(free, box, truth, sid, taus=np.array([0.5]), n_boot=300, seed=7)
    prepended = metrics.gap_curve(free, box, truth, sid, taus=np.array([0.3, 0.5]), n_boot=300, seed=7)
    reordered = metrics.gap_curve(free, box, truth, sid, taus=np.array([0.9, 0.5, 0.1]), n_boot=300, seed=7)
    at = lambda curve: next(g for g in curve if g.tau == 0.5).gap_ci
    a, b, c = at(only), at(prepended), at(reordered)
    # adding/reordering OTHER taus must not move tau=0.5's interval (no sequential RNG coupling)
    assert (a.lo, a.hi) == (b.lo, b.hi) == (c.lo, c.hi)


# --- 7. the kernel detects a real relative gap when one exists ---


def test_gap_curve_detects_real_signal():
    free, box, truth, sid = _signal_fixture()
    taus = np.linspace(0.1, 0.9, 9)
    curve = metrics.gap_curve(free, box, truth, sid, taus=taus, n_boot=500, seed=1)
    assert all(g.gap_ci.defined for g in curve)           # 15 positives -> well-powered
    summary = metrics.gap_decision_summary(curve)
    assert summary["frac_taus_gap_ci_positive"] > 0.0     # some tau range where gap CI excludes 0
    assert summary["max_gap"] > 0.0


# --- 8. paired bootstrap captures correlation (paired CI tighter than independent) ---


def test_paired_gap_ci_tighter_when_errors_correlated():
    # both backends make the IDENTICAL false positive on one scene -> their F1s move together, so the
    # PAIRED gap is identically 0 on every resample (width ~0). An independent differencing cannot.
    n = 20
    sid = np.arange(n)
    truth = np.array([i < 10 for i in range(n)])
    pred_both = truth.copy()
    pred_both[10] = True                                  # same FP for free and box
    free = np.where(pred_both, 1.0, 0.0)
    box = np.where(pred_both, 1.0, 0.0)                   # identical predictions
    curve = metrics.gap_curve(free, box, truth, sid, taus=np.array([0.5]), n_boot=500, seed=3)
    gap_ci = curve[0].gap_ci
    assert gap_ci.defined
    assert abs(gap_ci.hi - gap_ci.lo) < 1e-9              # paired width collapses to ~0
    assert abs(curve[0].gap) < 1e-9                       # identical backends -> zero gap


# --- 9. reduce='any' vs 'all' is an explicit, recorded scoping choice (not a hidden forking path) ---


def test_reduce_any_vs_all_changes_membership():
    # one scene, two frames: frame 0 crosses tau, frame 1 does not.
    scene_ids = np.array(["s", "s"])
    truth = np.array([True, True])
    free = np.array([1.0, 0.0])
    box = np.array([0.0, 0.0])
    taus = np.array([0.5])
    any_curve = metrics.gap_curve(free, box, truth, scene_ids, taus=taus, reduce="any", n_boot=50, min_positive=1)
    all_curve = metrics.gap_curve(free, box, truth, scene_ids, taus=taus, reduce="all", n_boot=50, min_positive=1)
    # 'any': scene counts as predicted-positive (frame 0 crossed) -> free F1 finite & >0
    # 'all': scene NOT predicted-positive (frame 1 did not cross) -> free misses it
    assert any_curve[0].f1_free > all_curve[0].f1_free


# --- 10. structural: no movable gate is representable in the module source ---


def test_no_0_90_literal_as_comparison_operand():
    src = pathlib.Path(metrics.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for op in operands:
                if isinstance(op, ast.Constant) and isinstance(op.value, (int, float)):
                    assert float(op.value) not in (0.9, 0.90), "a 0.90 gate is representable -- forbidden"


def test_taus_has_no_default():
    # `taus` is the movable knob; the caller MUST supply it (pre-registered), so it has no default.
    import inspect

    sig = inspect.signature(metrics.gap_curve)
    assert sig.parameters["taus"].default is inspect.Parameter.empty


def test_gap_decision_summary_uses_no_removed_numpy_symbol():
    src = pathlib.Path(metrics.__file__).read_text()
    assert "np.trapz" not in src and "numpy.trapz" not in src  # removed in numpy 2.x -> would crash
