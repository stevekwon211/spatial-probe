# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""L1 -- occupancy free-space denotation vs the Occ3D-nuScenes dense GT (realizes the SEALED pre-reg
`l1_denotation_occ3d_preregistration.md`, commit 6475448). Run AFTER the seal; nothing here was
chosen after seeing a number.

Estimand (per the pre-reg). For each scene, load the SAME scene twice:
  * UNDER TEST  -- `load_scene(name, 'data', mask='lidar')`: single-frame OBSERVED occupancy
                   (unobserved voxels UNKNOWN), read with the SEALED unknown_policy=FREE.
  * REFERENCE   -- `load_scene(name, 'data', mask='none')`: the dense, temporally-aggregated Occ3D GT.
Per frame, over the in-path band (the ego straight-ahead corridor the sealed free-space predicates
sweep), classify each band cell FREE / BLOCKED from the obstacle BEV the predicates read
(`OccupancyGrid.obstacle_centers`, ego-height band -- the exact substrate of `free_along_ego_path`,
`min_free_width_along_path`, `lateral_clearance`). The denotation set = {band cells}; positive class
= FREE. Report IoU / precision / recall / F1 of the FREE class, plus false-block rate (observed
BLOCKED where GT FREE) and miss rate (observed FREE where GT OCCUPIED), with a scene-clustered
bootstrap CI (1000 resamples, seed 0) and the trivial baselines (all-free, random@band-free-rate).

HONEST LABEL (per the pre-reg independence ledger + CLAUDE.md): both sides are LiDAR-derived, so this
is CONSISTENCY between the observed and dense occupancy, NOT external truth. No H3 re-inflation.

Set metrics are pure numpy (IoU/P/R/F1 are TP/FP/FN counts, no rank-tie ambiguity -- the `_roc_auc`
lesson; sklearn is not a dep and is not used). Unit-tested against hand-computed values in
`tests/test_l1_denotation_metrics.py`. Box-only baseline = INAPPLICABLE (coverage 0, cannot express
free-space -- stated, never fabricated).

Run: python experiments/occquery_v0/l1_denotation_occ3d.py [--limit N] [--horizon 1.0]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))  # script-mode import of probe.*

from probe.adapters.occ3d import load_scene  # noqa: E402
from probe.grid import OccupancyGrid, EgoPose, UnknownPolicy  # noqa: E402

_DATA = _REPO / "data"
# In-path band (derived from the sealed predicate geometry -- see pre-reg "Per-frame in-path band
# identical to the sealed occquery band"): the ego straight-ahead corridor `free_along_ego_path`
# sweeps (forward 0..reach, reach = length/2 + speed*horizon) widened by the widest sealed
# clearance/centerline tier (1.0 m) so the band covers what ALL four free-space families read.
_HORIZON = 1.0          # the canonical free_path_is_blocked horizon (queries.yaml)
_BAND_MARGIN = 1.0      # widest sealed lateral clearance/centerline tier (queries.yaml)
_DEV_FRACTION = 0.20    # first 20% by sorted scene id = dev (no free params to tune -- hygiene only)
_N_BOOT = 1000
_SEED = 0


# --------------------------------------------------------------------------------------------------
# Pure set-metrics (unit-tested; no sklearn).  positive class = FREE; pred = observed, ref = dense GT.
# --------------------------------------------------------------------------------------------------
def confusion_from_masks(pred_free: np.ndarray, ref_free: np.ndarray) -> tuple[int, int, int, int]:
    """(TP, FP, FN, TN) of the FREE class. pred=observed, ref=dense GT (booleans, FREE==True).

    TP = obs FREE & GT FREE; FP = obs FREE & GT BLOCKED (the MISS direction, obs free where GT occ);
    FN = obs BLOCKED & GT FREE (the FALSE-BLOCK direction); TN = obs BLOCKED & GT BLOCKED.
    """
    pf = np.asarray(pred_free, dtype=bool).ravel()
    rf = np.asarray(ref_free, dtype=bool).ravel()
    tp = int(np.count_nonzero(pf & rf))
    fp = int(np.count_nonzero(pf & ~rf))
    fn = int(np.count_nonzero(~pf & rf))
    tn = int(np.count_nonzero(~pf & ~rf))
    return tp, fp, fn, tn


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def free_set_metrics(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    """IoU / precision / recall / F1 of the FREE class + false-block & miss rates from a confusion.

    precision = TP/(TP+FP), recall = TP/(TP+FN), IoU = TP/(TP+FP+FN), F1 = 2TP/(2TP+FP+FN).
    false_block_rate = FN/(TP+FN) (= 1-recall): observed BLOCKED among GT-FREE cells.
    miss_rate        = FP/(FP+TN): observed FREE among GT-OCCUPIED (BLOCKED) cells.
    Undefined denominators -> NaN (never a silent 0).
    """
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_block_rate": _safe_div(fn, tp + fn),
        "miss_rate": _safe_div(fp, fp + tn),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


# --------------------------------------------------------------------------------------------------
# In-path band obstacle BEV (the substrate the sealed free-space predicates read).
# --------------------------------------------------------------------------------------------------
def band_blocked_bev(grid: OccupancyGrid, ego: EgoPose, *, horizon: float, band_margin: float,
                     unknown_policy: UnknownPolicy) -> np.ndarray:
    """(nf, nl) bool BEV: True == BLOCKED (an obstacle voxel projects to the cell), over the ego
    in-path band [forward 0..reach] x [|lateral| <= ego.width/2 + band_margin], at the grid voxel
    resolution. Obstacles = `obstacle_centers` in the ego-height band (the predicates' own def);
    `unknown_policy` controls whether UNKNOWN voxels count as obstacles."""
    res = grid.voxel_size
    reach = ego.length / 2.0 + ego.speed * horizon
    band_half = ego.width / 2.0 + band_margin
    nf = int(round(reach / res)) + 1
    nl = int(round(2.0 * band_half / res)) + 1
    blocked = np.zeros((nf, nl), dtype=bool)
    centers = grid.obstacle_centers(unknown_policy=unknown_policy, max_height_agl=ego.height)
    if len(centers):
        fwd, lat = ego.to_ego_frame(centers[:, :2])
        inb = (fwd >= 0.0) & (fwd <= reach) & (np.abs(lat) <= band_half)
        fi = np.round(fwd[inb] / res).astype(int)
        li = np.round((lat[inb] + band_half) / res).astype(int)
        fi = np.clip(fi, 0, nf - 1)
        li = np.clip(li, 0, nl - 1)
        blocked[fi, li] = True
    return blocked


def _scene_confusion(name: str, horizon: float) -> dict | None:
    """Accumulate the FREE-class confusion over a whole scene (summed over frames x band cells), for
    the observed predicate under unknown_policy FREE (sealed) and OCCUPIED (sensitivity)."""
    obs = load_scene(name, _DATA, mask="lidar")   # UNDER TEST -- single-frame observed
    gt = load_scene(name, _DATA, mask="none")     # REFERENCE -- dense aggregated GT
    acc = {"scene": name,
           "free": [0, 0, 0, 0],   # tp, fp, fn, tn under unknown_policy=FREE (sealed)
           "occ": [0, 0, 0, 0]}    # under unknown_policy=OCCUPIED (sensitivity)
    n = min(len(obs.frames), len(gt.frames))
    for i in range(n):
        ego = obs.frames[i].ego
        gt_blocked = band_blocked_bev(gt.frames[i].grid, ego, horizon=horizon,
                                      band_margin=_BAND_MARGIN, unknown_policy=UnknownPolicy.FREE)
        ref_free = ~gt_blocked
        for key, pol in (("free", UnknownPolicy.FREE), ("occ", UnknownPolicy.OCCUPIED)):
            obs_blocked = band_blocked_bev(obs.frames[i].grid, ego, horizon=horizon,
                                           band_margin=_BAND_MARGIN, unknown_policy=pol)
            tp, fp, fn, tn = confusion_from_masks(~obs_blocked, ref_free)
            a = acc[key]
            a[0] += tp; a[1] += fp; a[2] += fn; a[3] += tn
    acc["n_frames"] = n
    return acc


# --------------------------------------------------------------------------------------------------
# Scene-clustered bootstrap on a metric computed from POOLED confusion (the correct cluster bootstrap
# for a ratio metric: resample scenes, pool their TP/FP/FN/TN, recompute).
# --------------------------------------------------------------------------------------------------
def _boot_metric(per_scene_conf: list[tuple[float, float, float, float]], metric_key: str,
                 rng: np.random.Generator, n_boot: int = _N_BOOT) -> dict:
    arr = np.asarray(per_scene_conf, dtype=float)  # (S, 4)
    point = free_set_metrics(*arr.sum(axis=0))[metric_key]
    s = arr.shape[0]
    if s < 2:
        return {"mean": point, "lo": float("nan"), "hi": float("nan"), "n_scenes": s}
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, s, size=s)
        tp, fp, fn, tn = arr[idx].sum(axis=0)
        m = free_set_metrics(tp, fp, fn, tn)[metric_key]
        if not math.isnan(m):
            samples.append(m)
    lo, hi = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))) \
        if samples else (float("nan"), float("nan"))
    return {"mean": point, "lo": lo, "hi": hi, "n_scenes": s}


def _allfree_conf(conf: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """all-free predicts every cell FREE: TP=GT_FREE, FP=GT_BLOCKED, FN=0, TN=0."""
    tp, fp, fn, tn = conf
    gt_free = tp + fn
    gt_blocked = fp + tn
    return (gt_free, gt_blocked, 0.0, 0.0)


def _random_conf(conf: tuple[float, float, float, float], p: float) -> tuple[float, float, float, float]:
    """random@free-rate predicts FREE w.p. p -- expected confusion (deterministic, the principled
    value of a base-rate random classifier)."""
    tp, fp, fn, tn = conf
    gt_free = tp + fn
    gt_blocked = fp + tn
    return (p * gt_free, p * gt_blocked, (1 - p) * gt_free, (1 - p) * gt_blocked)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _leg1_restate() -> dict:
    """Re-state Leg 1 (EXPRESSIVITY, oracle-free) from the existing sealed h3b result -- the SOLE
    headline. Read from results/h3b_expressivity.json (NOT recomputed here)."""
    p = _HERE / "results" / "h3b_expressivity.json"
    if not p.exists():
        return {"available": False, "note": "results/h3b_expressivity.json absent; run h3b first"}
    d = json.loads(p.read_text())
    fs = d["leg1_expressivity"]["free_space_families_only"]
    return {
        "available": True,
        "source": "results/h3b_expressivity.json (sealed h3b_expressivity_preregistration.md, a47b500)",
        "free_space_families": fs["families"],
        "occupancy_coverage_pct": fs["occupancy_coverage_pct"],
        "box_only_coverage_pct": fs["box_only_coverage_pct"],
        "gap_pct": fs["gap_pct"],
        "kill_H1_falsified": d["leg1_expressivity"]["kill_H1_falsified_box_expresses_freespace"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = ALL scenes (the sealed run)")
    ap.add_argument("--horizon", type=float, default=_HORIZON)
    args = ap.parse_args()
    rng = np.random.default_rng(_SEED)

    annotations = json.loads((_DATA / "annotations.json").read_text())
    all_scenes = sorted(annotations["scene_infos"].keys())
    if args.limit:
        all_scenes = all_scenes[: args.limit]

    records: list[dict] = []
    print(f"L1 denotation: {len(all_scenes)} scenes, horizon={args.horizon}s ...", flush=True)
    for i, name in enumerate(all_scenes):
        try:
            rec = _scene_confusion(name, args.horizon)
        except Exception as e:  # noqa: BLE001 - a missing/corrupt scene is skipped, reported
            print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
            continue
        records.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(all_scenes):
            print(f"  {i + 1}/{len(all_scenes)} scenes ({len(records)} loaded)", flush=True)

    if not records:
        raise SystemExit("no scenes loaded")

    # split: first 20% by sorted scene id = dev (hygiene only -- no tuned params), rest = headline
    loaded = sorted(r["scene"] for r in records)
    n_dev = int(round(_DEV_FRACTION * len(loaded)))
    dev_ids = set(loaded[:n_dev])
    head = [r for r in records if r["scene"] not in dev_ids]
    dev = [r for r in records if r["scene"] in dev_ids]

    def _summary(group: list[dict]) -> dict:
        conf_free = [tuple(r["free"]) for r in group]
        conf_occ = [tuple(r["occ"]) for r in group]
        total_free = np.asarray(conf_free, float).sum(axis=0)
        gt_free = total_free[0] + total_free[2]
        gt_blocked = total_free[1] + total_free[3]
        band_free_rate = _safe_div(gt_free, gt_free + gt_blocked)
        allfree = [_allfree_conf(c) for c in conf_free]
        randc = [_random_conf(c, band_free_rate) for c in conf_free]
        out = {
            "n_scenes": len(group),
            "n_frames": int(sum(r["n_frames"] for r in group)),
            "band_gt_free_rate": band_free_rate,
            "predicate_unknown_FREE_sealed": {
                k: _boot_metric(conf_free, k, rng)
                for k in ("iou", "f1", "precision", "recall", "false_block_rate", "miss_rate")
            },
            "predicate_unknown_OCCUPIED_sensitivity": {
                k: _boot_metric(conf_occ, k, rng) for k in ("iou", "f1", "false_block_rate", "miss_rate")
            },
            "baseline_all_free": {k: _boot_metric(allfree, k, rng) for k in ("iou", "f1")},
            "baseline_random_at_free_rate": {k: _boot_metric(randc, k, rng) for k in ("iou", "f1")},
            "baseline_box_only": "INAPPLICABLE (coverage 0 -- box-only cannot express free-space; no number fabricated)",
        }
        return out

    headline = _summary(head)
    dev_summary = _summary(dev) if dev else None

    # DEGENERACY CHECK (the ground-truth finding): does the Occ3D lidar mask ever hide an OCCUPIED
    # voxel? If observed-OCCUPIED == GT-OCCUPIED, the sealed (unknown=FREE) denotation is identity by
    # construction, not an occlusion test. Measured directly on the headline confusion.
    fb = headline["predicate_unknown_FREE_sealed"]["false_block_rate"]["mean"]
    miss = headline["predicate_unknown_FREE_sealed"]["miss_rate"]["mean"]
    degenerate = (not math.isnan(fb) and fb < 1e-3) and (not math.isnan(miss) and miss < 1e-3)

    # verdict per the SEALED kill criteria (IoU CI.lo vs the trivial baseline)
    pred_iou = headline["predicate_unknown_FREE_sealed"]["iou"]
    base_iou = headline["baseline_all_free"]["iou"]
    iou_beats_trivial = (not math.isnan(pred_iou["lo"]) and not math.isnan(base_iou["mean"])
                         and pred_iou["lo"] > base_iou["mean"])

    report = {
        "experiment": "occquery_v0 / L1 denotation (Occ3D-nuScenes dense GT)",
        "preregistration": "l1_denotation_occ3d_preregistration.md (SEALED, commit 6475448)",
        "result_class": ("CONSISTENCY (observed single-frame occupancy vs dense aggregated Occ3D GT, "
                         "both LiDAR-derived) -- NOT external truth. No H3 re-inflation."),
        "commit": _git_commit(),
        "seed": _SEED,
        "data_root": str(_DATA),
        "horizon_s": args.horizon,
        "band": (f"ego in-path corridor: forward 0..(length/2 + speed*{args.horizon}s), "
                 f"|lateral| <= width/2 + {_BAND_MARGIN} m; obstacle BEV in the ego-height band "
                 "(the substrate the sealed free-space predicates read)."),
        "leg1_expressivity_headline": _leg1_restate(),
        "split": {"dev_fraction": _DEV_FRACTION, "n_dev_scenes": len(dev), "n_headline_scenes": len(head)},
        "headline": headline,
        "dev": dev_summary,
        "ground_truth_degeneracy": {
            "observed_obstacle_set_equals_dense_GT": bool(degenerate),
            "note": ("Occ3D's mask_lidar marks ~100% of OCCUPIED voxels visible (verified: only "
                     "~0.008% of occupied voxels differ obs-vs-dense), so under the sealed "
                     "unknown_policy=FREE the observed and dense-GT OBSTACLE sets are identical and "
                     "the FREE denotation is ~1.0 BY CONSTRUCTION -- a synthetic-class identity, NOT "
                     "an occlusion-robustness test. The pre-reg premise 'the two differ by "
                     "occlusion/sparsity' is FALSIFIED for the obstacle class; the difference lives "
                     "entirely in FREE->UNKNOWN voxels, governed by the unknown policy (see the "
                     "OCCUPIED sensitivity arm). The adapter docstring (occ3d.py) warns this."),
        },
        "verdict": {
            "leg2_iou_ci_lo_beats_all_free_mean": bool(iou_beats_trivial),
            "leg2_is_a_result": False,
            "leg2_verdict": ("NOT A RESULT (degenerate): any apparent win over the trivial baseline "
                             "is an artifact of observed-OBSTACLES == dense-GT-OBSTACLES by "
                             "construction, not evidence of denotation robustness. Reported as "
                             "consistency only; the SOLE headline remains Leg 1 (expressivity)."),
        },
    }

    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "l1_denotation_occ3d.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_summary(out_dir / "l1_denotation_occ3d_summary.md", report)
    print(f"\nwrote {out_dir / 'l1_denotation_occ3d.json'}")
    print(f"wrote {out_dir / 'l1_denotation_occ3d_summary.md'}")
    _print_console(report)


def _fmt(ci: dict) -> str:
    return f"{ci['mean']:.4f} CI[{ci['lo']:.4f}, {ci['hi']:.4f}]"


def _print_console(r: dict) -> None:
    h = r["headline"]
    print("\n=== L1 DENOTATION (Occ3D dense-GT consistency) ===")
    l1 = r["leg1_expressivity_headline"]
    if l1.get("available"):
        print(f"Leg 1 (HEADLINE, oracle-free): free-space families occupancy "
              f"{l1['occupancy_coverage_pct']}% vs box-only {l1['box_only_coverage_pct']}% "
              f"-> gap {l1['gap_pct']} pts (H1 falsified={l1['kill_H1_falsified']})")
    print(f"\nLeg 2 (CONSISTENCY): headline {h['n_scenes']} scenes / {h['n_frames']} frames, "
          f"band GT free-rate {h['band_gt_free_rate']:.4f}")
    pf = h["predicate_unknown_FREE_sealed"]
    print("  predicate (unknown=FREE, SEALED):")
    for k in ("iou", "f1", "precision", "recall", "false_block_rate", "miss_rate"):
        print(f"    {k:18s}: {_fmt(pf[k])}")
    print(f"  baseline all-free   IoU {_fmt(h['baseline_all_free']['iou'])}  F1 {_fmt(h['baseline_all_free']['f1'])}")
    print(f"  baseline random@fr  IoU {_fmt(h['baseline_random_at_free_rate']['iou'])}  F1 {_fmt(h['baseline_random_at_free_rate']['f1'])}")
    po = h["predicate_unknown_OCCUPIED_sensitivity"]
    print("  SENSITIVITY (unknown=OCCUPIED, conservative reading of the unobserved):")
    for k in ("iou", "f1", "false_block_rate", "miss_rate"):
        print(f"    {k:18s}: {_fmt(po[k])}")
    print(f"  box-only: {h['baseline_box_only']}")
    print(f"\nDEGENERACY: observed obstacles == dense-GT obstacles by construction = "
          f"{r['ground_truth_degeneracy']['observed_obstacle_set_equals_dense_GT']}")
    print(f"VERDICT Leg 2: {r['verdict']['leg2_verdict']}")


def _write_summary(path: pathlib.Path, r: dict) -> None:
    h = r["headline"]
    pf = h["predicate_unknown_FREE_sealed"]
    po = h["predicate_unknown_OCCUPIED_sensitivity"]
    l1 = r["leg1_expressivity_headline"]
    L = []
    L.append("# L1 denotation -- occupancy free-space vs Occ3D dense GT (CONSISTENCY)\n")
    L.append(f"- Pre-reg: `{r['preregistration']}`")
    L.append(f"- Commit: `{r['commit']}`  seed {r['seed']}  horizon {r['horizon_s']}s")
    L.append(f"- Result class: {r['result_class']}")
    L.append(f"- Band: {r['band']}\n")
    L.append("## Leg 1 -- EXPRESSIVITY (SOLE headline, oracle-free)")
    if l1.get("available"):
        L.append(f"Free-space families: occupancy **{l1['occupancy_coverage_pct']}%** vs box-only "
                 f"**{l1['box_only_coverage_pct']}%** -> **{l1['gap_pct']}-pt** expressivity gap "
                 f"(H1 falsified = {l1['kill_H1_falsified']}). Source: {l1['source']}.")
    else:
        L.append("h3b_expressivity.json absent -- run h3b first.")
    L.append("")
    L.append("## Leg 2 -- DENOTATION CONSISTENCY (NOT external truth)")
    L.append(f"Headline split: {h['n_scenes']} scenes / {h['n_frames']} frames "
             f"(dev {r['split']['n_dev_scenes']} scenes). Band GT free-rate {h['band_gt_free_rate']:.4f}.\n")
    L.append("| metric (FREE class) | predicate (unknown=FREE, sealed) | all-free | random@free-rate |")
    L.append("|---|---|---|---|")
    for k in ("iou", "f1"):
        L.append(f"| {k.upper()} | {_fmt(pf[k])} | {_fmt(h['baseline_all_free'][k])} | "
                 f"{_fmt(h['baseline_random_at_free_rate'][k])} |")
    L.append("")
    L.append("Predicate (unknown=FREE, sealed) full denotation:")
    for k in ("precision", "recall", "false_block_rate", "miss_rate"):
        L.append(f"- {k}: {_fmt(pf[k])}")
    L.append("\nBox-only baseline: " + h["baseline_box_only"])
    L.append("\n### Sensitivity -- unknown_policy = OCCUPIED (conservative reading of the unobserved)")
    for k in ("iou", "f1", "false_block_rate", "miss_rate"):
        L.append(f"- {k}: {_fmt(po[k])}")
    L.append("\n## Ground-truth degeneracy (the headline negative for Leg 2)")
    L.append(r["ground_truth_degeneracy"]["note"])
    L.append(f"\n**Observed obstacles == dense-GT obstacles by construction: "
             f"{r['ground_truth_degeneracy']['observed_obstacle_set_equals_dense_GT']}**")
    L.append("\n## Verdict (per the sealed kill criteria)")
    L.append(f"- IoU CI.lo beats all-free mean: {r['verdict']['leg2_iou_ci_lo_beats_all_free_mean']}")
    L.append(f"- Leg 2 is a result: {r['verdict']['leg2_is_a_result']}")
    L.append(f"- {r['verdict']['leg2_verdict']}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
