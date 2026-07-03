# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""P3 — occupancy-to-box AUTO-LABEL recovery + the human residual (realizes the SEALED pre-reg
`experiments/autolabel_v0/preregistration.md`; run AFTER the seal, nothing chosen post-hoc).

The auto-labeler: 3D connected-component clustering (`scipy.ndimage.label`, 26-connectivity) of
the Occ3D-nuScenes occupied voxels in the ego-height band → BEV proposal boxes (centroid + voxel
count as confidence), gated to voxel-count ∈ [τ, τ_max]. Scored against the real nuScenes GT
boxes with the nuScenes detection matching rule (greedy BEV center-distance, d ∈ {0.5,1,2,4} m;
verdict at d=2.0 m). Measures the automation ceiling (C3-A) and whether the failure is structured
by class/range (C3-B).

Honest scope: occupancy is nuScenes-LiDAR-derived, so this is a SAME-MODALITY recovery study —
the floor P2's camera-LiDAR fusion detector is meant to beat. The harness (`score_frame`, the
matcher, the rate machinery) accepts external proposals unchanged, so P2's mmdet3d predictions
drop in as `Proposal` lists with no other change.

Run:  python experiments/autolabel_v0/run.py            # full sealed run
      python experiments/autolabel_v0/run.py --limit 5  # smoke only, never reported
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from probe.adapters.occ3d import _annotations, load_scene  # noqa: E402
from probe.grid import OCCUPIED, EgoPose, OccupancyGrid  # noqa: E402
from probe.scene import TrackedBox  # noqa: E402

_DATA = _REPO / "data"

# --- SEALED constants (preregistration.md) ------------------------------------------------------
_TAU_SWEEP = (2, 5, 10, 20, 40)
_TAU_MAX = 2000
_DISTANCES = (0.5, 1.0, 2.0, 4.0)
_D_VERDICT = 2.0
_DEV_FRACTION = 0.20
_N_BOOT = 1000
_SEED = 0
_NEAR_MAX = 20.0
_FAR_MIN = 35.0
_SCORED_CLASSES = ("vehicle", "pedestrian", "bicycle", "motorcycle")  # 'other' excluded from GT
_C3A_QUALITY = 0.90
_C3B_GAP = 0.20
_C3B_UNSTRUCTURED = 0.10


@dataclass(frozen=True)
class Proposal:
    cx: float
    cy: float
    n_voxels: int


# --------------------------------------------------------------------------------------------------
# Auto-labeler: occupancy -> BEV proposals
# --------------------------------------------------------------------------------------------------
def propose_from_occupancy(grid: OccupancyGrid, ego: EgoPose, *, tau: int, tau_max: int) -> list[Proposal]:
    """Connected-component clusters of occupied voxels in the ego-height band -> BEV proposals,
    gated to voxel count in [tau, tau_max]. Ego frame is the occupancy frame (ego at origin)."""
    occ = grid.occupancy
    res = grid.voxel_size
    origin = np.asarray(grid.origin, dtype=float)
    zc = origin[2] + np.arange(occ.shape[2]) * res
    zsel = (zc > grid.ground_height) & (zc <= grid.ground_height + ego.height)
    mask = np.zeros_like(occ, dtype=bool)
    mask[:, :, zsel] = occ[:, :, zsel] == OCCUPIED
    if not mask.any():
        return []
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return []
    props: list[Proposal] = []
    # centroid in index space -> world (ego) BEV; count voxels per component
    counts = np.bincount(labels.ravel())
    idx = np.argwhere(labels > 0)
    comp = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    for c in range(1, n + 1):
        nv = int(counts[c])
        if nv < tau or nv > tau_max:
            continue
        sel = idx[comp == c]
        cx, cy = (origin[:2] + sel[:, :2].mean(axis=0) * res)
        props.append(Proposal(cx=float(cx), cy=float(cy), n_voxels=nv))
    return props


# --------------------------------------------------------------------------------------------------
# Matching (nuScenes greedy BEV center-distance) + rates
# --------------------------------------------------------------------------------------------------
def match_greedy(gt_xy: list[tuple[float, float]], prop_xy: list[tuple[float, float]],
                 d: float) -> tuple[int, list[int], list[int]]:
    """Greedy nuScenes-style match: GT nearest-first (by range), each matched to the nearest
    unmatched proposal within d. Returns (TP, unmatched-GT indices, unmatched-proposal indices)."""
    gts = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    props = np.asarray(prop_xy, dtype=float).reshape(-1, 2)
    used = np.zeros(len(props), dtype=bool)
    order = np.argsort(np.linalg.norm(gts, axis=1)) if len(gts) else []
    tp = 0
    fn: list[int] = []
    for gi in order:
        if not len(props) or used.all():
            fn.append(int(gi))
            continue
        dist = np.linalg.norm(props - gts[gi], axis=1)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= d:
            used[j] = True
            tp += 1
        else:
            fn.append(int(gi))
    fp = [int(j) for j in range(len(props)) if not used[j]]
    return tp, fn, fp


def pr_f1(*, tp: float, fp: float, fn: float) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def range_bin(r: float) -> str:
    if r < _NEAR_MAX:
        return "near"
    if r > _FAR_MIN:
        return "far"
    return "mid"


# --------------------------------------------------------------------------------------------------
# Per-frame scoring: proposals vs GT, sliced by class and range, over the (tau, d) grid.
# --------------------------------------------------------------------------------------------------
def _scored_gt(objects: tuple[TrackedBox, ...]) -> list[dict]:
    out = []
    for o in objects:
        if o.label not in _SCORED_CLASSES:
            continue
        r = math.hypot(o.center[0], o.center[1])
        out.append({"xy": (o.center[0], o.center[1]), "label": o.label, "range": r,
                    "rbin": range_bin(r)})
    return out


def _other_xy(objects: tuple[TrackedBox, ...]) -> list[tuple[float, float]]:
    """BEV centers of non-scored ('other' — barrier/cone/debris) objects: a proposal near one is
    explained by a real obstacle and is removed from FP (sealed pre-reg fairness rule)."""
    return [(o.center[0], o.center[1]) for o in objects if o.label not in _SCORED_CLASSES]


def _fp_after_other(prop_xy: list[tuple[float, float]], fp_idx: list[int],
                    other_xy: list[tuple[float, float]], d: float) -> int:
    """FP proposals with no scored-GT match, minus those within d of an 'other'-class object."""
    if not other_xy:
        return len(fp_idx)
    oth = np.asarray(other_xy, dtype=float).reshape(-1, 2)
    penalized = 0
    for j in fp_idx:
        p = np.asarray(prop_xy[j], dtype=float)
        if np.min(np.linalg.norm(oth - p, axis=1)) > d:
            penalized += 1  # not explained by any real obstacle -> a true false positive
    return penalized


def score_frame(gt: list[dict], other_xy: list[tuple[float, float]],
                proposals: list[Proposal]) -> dict:
    """Score one proposal set over all d, sliced. Returns confusion counts overall + per class +
    per range bin (TP/FP/FN). 'other'-explained proposals are removed from FP (sealed)."""
    prop_xy = [(p.cx, p.cy) for p in proposals]
    res: dict = {}
    for d in _DISTANCES:
        tp, fn_idx, fp_idx = match_greedy([g["xy"] for g in gt], prop_xy, d)
        fp = _fp_after_other(prop_xy, fp_idx, other_xy, d)
        rec = {"overall": [tp, fp, len(fn_idx)]}
        # slice recall denominators: TP per slice needs which GT matched. Recompute matched set.
        matched = _matched_gt_set([g["xy"] for g in gt], prop_xy, d)
        # slice recall by class (gt key 'label') and by range bin (gt key 'rbin'); slot name is the
        # reported prefix, gt_key is where the value lives.
        for slot_prefix, gt_key, keyset in (("class", "label", _SCORED_CLASSES),
                                            ("rbin", "rbin", ("near", "mid", "far"))):
            for k in keyset:
                idxs = [i for i, g in enumerate(gt) if g[gt_key] == k]
                s_tp = sum(1 for i in idxs if i in matched)
                rec[f"{slot_prefix}:{k}"] = [s_tp, 0, len(idxs) - s_tp]  # recall only (FP not sliced)
        res[str(d)] = rec
    return res


def _matched_gt_set(gt_xy, prop_xy, d) -> set[int]:
    gts = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    props = np.asarray(prop_xy, dtype=float).reshape(-1, 2)
    used = np.zeros(len(props), dtype=bool)
    order = np.argsort(np.linalg.norm(gts, axis=1)) if len(gts) else []
    matched: set[int] = set()
    for gi in order:
        if not len(props) or used.all():
            continue
        dist = np.linalg.norm(props - gts[gi], axis=1)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= d:
            used[j] = True
            matched.add(int(gi))
    return matched


def process_scene(name: str) -> dict:
    sc = load_scene(name, _DATA, mask="none", with_boxes=True)
    rec: dict = {"scene": name, "n_frames": len(sc.frames), "by_tau": {}}
    for tau in _TAU_SWEEP:
        # accumulate confusion over frames, per d and per slice
        acc: dict[str, dict[str, list[int]]] = {str(d): {} for d in _DISTANCES}
        for fr in sc.frames:
            gt = _scored_gt(fr.objects)
            other = _other_xy(fr.objects)
            props = propose_from_occupancy(fr.grid, fr.ego, tau=tau, tau_max=_TAU_MAX)
            fs = score_frame(gt, other, props)
            for d, rec_d in fs.items():
                for slot, (tp, fp, fn) in rec_d.items():
                    a = acc[d].setdefault(slot, [0, 0, 0])
                    a[0] += tp; a[1] += fp; a[2] += fn
        rec["by_tau"][str(tau)] = acc
    return rec


# --------------------------------------------------------------------------------------------------
# Aggregation + verdicts
# --------------------------------------------------------------------------------------------------
def _boot_rate(per_scene: list[tuple[float, float]], rng: np.random.Generator,
               kind: str, n_boot: int = _N_BOOT) -> dict:
    """per_scene = (numerator, denominator); pooled ratio + scene bootstrap CI."""
    arr = np.asarray(per_scene, dtype=float)
    if not arr.size:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    num, den = arr.sum(axis=0)
    point = num / den if den else float("nan")
    if arr.shape[0] < 2 or not den:
        return {"mean": point, "lo": float("nan"), "hi": float("nan")}
    s = [n / d for n, d in (arr[rng.integers(0, arr.shape[0], size=arr.shape[0])].sum(axis=0)
                            for _ in range(n_boot)) if d]
    lo, hi = (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))) if s else (float("nan"),) * 2
    return {"mean": float(point), "lo": lo, "hi": hi}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_report(records: list[dict]) -> dict:
    rng = np.random.default_rng(_SEED)
    loaded = sorted(r["scene"] for r in records)
    n_dev = int(round(_DEV_FRACTION * len(loaded)))
    dev_ids = set(loaded[:n_dev])
    head = [r for r in records if r["scene"] not in dev_ids]
    dv = str(_D_VERDICT)

    def _slice_pr(group, tau, slot, num_idx, den_idx_pair):
        pairs = []
        for r in group:
            a = r["by_tau"][str(tau)][dv].get(slot, [0, 0, 0])
            num = a[num_idx]
            den = a[den_idx_pair[0]] + a[den_idx_pair[1]]
            pairs.append((num, den))
        return _boot_rate(pairs, rng, slot)

    # per-tau overall precision/recall/f1 at the verdict distance
    tau_rows = {}
    for tau in _TAU_SWEEP:
        rec = _slice_pr(head, tau, "overall", 0, (0, 2))    # recall = tp/(tp+fn)
        prec = _slice_pr(head, tau, "overall", 0, (0, 1))   # precision = tp/(tp+fp)
        # f1 from pooled
        tp = sum(r["by_tau"][str(tau)][dv]["overall"][0] for r in head)
        fp = sum(r["by_tau"][str(tau)][dv]["overall"][1] for r in head)
        fn = sum(r["by_tau"][str(tau)][dv]["overall"][2] for r in head)
        tau_rows[tau] = {"precision": prec, "recall": rec, "f1": pr_f1(tp=tp, fp=fp, fn=fn)["f1"],
                         "tp": tp, "fp": fp, "fn": fn,
                         "min_pr": min(prec["mean"], rec["mean"])}

    # C3-A: no tau reaches precision>=0.9 AND recall>=0.9  <=>  max_tau min(P,R) < 0.9
    best_minpr = max(v["min_pr"] for v in tau_rows.values())
    c3a = {
        "d_verdict": _D_VERDICT,
        "max_over_tau_min_precision_recall": best_minpr,
        "holds_automation_ceiling": best_minpr < _C3A_QUALITY,
        "killed_occupancy_suffices": best_minpr >= _C3A_QUALITY,
        "per_tau": {str(t): {"precision": v["precision"]["mean"], "recall": v["recall"]["mean"],
                             "f1": v["f1"]} for t, v in tau_rows.items()},
    }

    # C3-B: at F1-max tau, class + range recall gaps
    tau_star = max(_TAU_SWEEP, key=lambda t: (tau_rows[t]["f1"] if not math.isnan(tau_rows[t]["f1"]) else -1))
    veh = _slice_pr(head, tau_star, "class:vehicle", 0, (0, 2))["mean"]
    ped = _slice_pr(head, tau_star, "class:pedestrian", 0, (0, 2))["mean"]
    near = _slice_pr(head, tau_star, "rbin:near", 0, (0, 2))["mean"]
    far = _slice_pr(head, tau_star, "rbin:far", 0, (0, 2))["mean"]
    class_gap = (veh - ped) if not (math.isnan(veh) or math.isnan(ped)) else float("nan")
    range_gap = (near - far) if not (math.isnan(near) or math.isnan(far)) else float("nan")
    both_struct = (not math.isnan(class_gap) and class_gap >= _C3B_GAP) and \
                  (not math.isnan(range_gap) and range_gap >= _C3B_GAP)
    both_unstruct = (not math.isnan(class_gap) and class_gap < _C3B_UNSTRUCTURED) and \
                    (not math.isnan(range_gap) and range_gap < _C3B_UNSTRUCTURED)
    c3b = {
        "tau_star_f1max": tau_star,
        "recall": {"vehicle": veh, "pedestrian": ped, "near": near, "far": far,
                   "mid": _slice_pr(head, tau_star, "rbin:mid", 0, (0, 2))["mean"],
                   "bicycle": _slice_pr(head, tau_star, "class:bicycle", 0, (0, 2))["mean"],
                   "motorcycle": _slice_pr(head, tau_star, "class:motorcycle", 0, (0, 2))["mean"]},
        "class_gap_vehicle_minus_pedestrian": class_gap,
        "range_gap_near_minus_far": range_gap,
        "holds_structured": both_struct,
        "killed_unstructured": both_unstruct,
    }

    return {
        "experiment": "autolabel_v0 / occupancy-to-box auto-label recovery (Occ3D-nuScenes)",
        "preregistration": "experiments/autolabel_v0/preregistration.md (SEALED, commit 9b34df9)",
        "result_class": ("SAME-MODALITY recovery/consistency (occupancy auto-labeler vs nuScenes "
                         "GT, both LiDAR-derived) — the floor P2's camera-LiDAR fusion detector is "
                         "meant to beat, NOT independent detector field-eval."),
        "commit": _git_commit(), "seed": _SEED,
        "n_headline_scenes": len(head), "n_dev_scenes": len(dev_ids),
        "tau_sweep": list(_TAU_SWEEP), "tau_max": _TAU_MAX,
        "tau_table": {str(t): {"precision": tau_rows[t]["precision"], "recall": tau_rows[t]["recall"],
                               "f1": tau_rows[t]["f1"], "tp": tau_rows[t]["tp"],
                               "fp": tau_rows[t]["fp"], "fn": tau_rows[t]["fn"]} for t in _TAU_SWEEP},
        "c3a_verdict": c3a,
        "c3b_verdict": c3b,
    }


def _fmt(d: dict) -> str:
    return f"{d['mean']:.3f} CI[{d['lo']:.3f}, {d['hi']:.3f}]"


def write_summary(path: pathlib.Path, r: dict) -> None:
    a, b = r["c3a_verdict"], r["c3b_verdict"]
    L = ["# Auto-label recovery + the human residual — Occ3D-nuScenes (P3)\n"]
    L.append(f"- Pre-reg: `{r['preregistration']}`")
    L.append(f"- Commit: `{r['commit']}`  seed {r['seed']}")
    L.append(f"- Result class: {r['result_class']}")
    L.append(f"- Headline {r['n_headline_scenes']} scenes / dev {r['n_dev_scenes']}; "
             f"τ_max {r['tau_max']}, verdict d={a['d_verdict']} m\n")
    v_a = "C3-A HOLDS (automation ceiling)" if a["holds_automation_ceiling"] else \
          "C3-A KILLED (occupancy suffices)"
    v_b = "C3-B HOLDS (structured)" if b["holds_structured"] else \
          ("C3-B KILLED (unstructured)" if b["killed_unstructured"] else "C3-B NO CLAIM (mixed)")
    L.append(f"## Verdicts: {v_a} / {v_b}\n")

    L.append(f"## C3-A — precision/recall vs τ (d={a['d_verdict']} m, headline)")
    L.append(f"best over τ of min(precision, recall) = **{a['max_over_tau_min_precision_recall']:.3f}** "
             f"(< {_C3A_QUALITY} → ceiling holds).\n")
    L.append("| τ | precision | recall | F1 | TP | FP | FN |")
    L.append("|---|---|---|---|---|---|---|")
    for t in r["tau_sweep"]:
        row = r["tau_table"][str(t)]
        L.append(f"| {t} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {row['f1']:.3f} | "
                 f"{row['tp']} | {row['fp']} | {row['fn']} |")
    L.append("")
    L.append(f"## C3-B — recall by slice (τ*={b['tau_star_f1max']}, F1-max)")
    L.append(f"class gap vehicle−pedestrian = **{b['class_gap_vehicle_minus_pedestrian']:.3f}**, "
             f"range gap near−far = **{b['range_gap_near_minus_far']:.3f}** "
             f"(both ≥ {_C3B_GAP} → structured).\n")
    rc = b["recall"]
    L.append("| slice | recall |")
    L.append("|---|---|")
    for k in ("vehicle", "pedestrian", "bicycle", "motorcycle", "near", "mid", "far"):
        val = rc.get(k)
        L.append(f"| {k} | {val:.3f} |" if val is not None and not math.isnan(val) else f"| {k} | n/a |")
    L.append("")
    L.append("## Reading")
    L.append("The occupancy-only auto-labeler is the FLOOR: a same-modality proposer with no "
             "semantics. The residual it leaves (low precision from static structure, recall "
             "collapse on small/distant objects) is exactly the human queue a Data PM owns — and "
             "what P2's camera-LiDAR fusion detector is meant to shrink. Numbers are recovery/"
             "consistency, not detector field-eval.")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = ALL scenes (the sealed run)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "autolabel_v0_scenes.jsonl"
    done: dict[str, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["scene"]] = rec

    if not args.report_only:
        all_scenes = sorted(_annotations(_DATA)["scene_infos"].keys())
        if args.limit:
            all_scenes = all_scenes[: args.limit]
        todo = [s for s in all_scenes if s not in done]
        print(f"autolabel_v0: {len(all_scenes)} scenes ({len(done)} checkpointed, {len(todo)} to run)",
              flush=True)
        t0 = time.time()
        with ckpt.open("a") as fh:
            for i, name in enumerate(todo):
                try:
                    rec = process_scene(name)
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
                    continue
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                done[name] = rec
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    rate = (time.time() - t0) / (i + 1)
                    print(f"  {i + 1}/{len(todo)}  ({rate:.1f}s/scene, "
                          f"~{rate * (len(todo) - i - 1) / 60:.0f} min left)", flush=True)

    records = list(done.values())
    if not records:
        raise SystemExit("no scenes processed")
    if args.limit and not args.report_only:
        print("\n--limit smoke run: no report written (never reported, per the pre-reg)")
        return

    report = build_report(records)
    (out_dir / "autolabel_v0.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(out_dir / "summary.md", report)
    print(f"\nwrote {out_dir / 'autolabel_v0.json'}\nwrote {out_dir / 'summary.md'}")
    a, b = report["c3a_verdict"], report["c3b_verdict"]
    print(f"\nC3-A: holds={a['holds_automation_ceiling']} "
          f"(best min(P,R)={a['max_over_tau_min_precision_recall']:.3f})")
    print(f"C3-B: holds={b['holds_structured']} killed_unstructured={b['killed_unstructured']} "
          f"(class gap {b['class_gap_vehicle_minus_pedestrian']:.3f}, "
          f"range gap {b['range_gap_near_minus_far']:.3f})")


if __name__ == "__main__":
    main()
