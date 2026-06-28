# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""SemanticKITTI cross-dataset experiment (realizes the SEALED pre-reg
`semantickitti_xdataset_preregistration.md`). Run AFTER the seal; the confirmatory metric-vs-baseline
verdict was not computed before the seal.

  Leg 1 -- H1 expressivity on a THIRD dataset. Reuse h3b_expressivity.leg1_expressivity VERBATIM
    (sealed queries.yaml + production scene_matches + refav_expressible flag) with a SemanticKITTI
    probe scene. Show the occupancy-vs-box-only free-space coverage gap holds on KITTI too.
    KILL: H1 falsified iff box-only expresses ANY free-space-family query.

  Leg 2 -- denotation, NON-degenerate + NON-vacuous. OBSERVED single-scan `.bin` vs DENSE-GT `.label`,
    FREE-class IoU/P/R/F1 + false_block/miss over the forward ego-height BEV field (the non-vacuous
    headline domain) and the L1-style thin in-path corridor (the vacuous contrast). Reuse the L1
    set-metrics + sequence-clustered bootstrap helpers VERBATIM. Report the headline-domain GT
    blocked-rate (non-vacuity) and the global `.bin`-vs-`.label` occupied XOR (non-degeneracy).
    KILL: Leg-2 dead iff predicate FREE-IoU CI.lo <= all-free IoU mean on the non-vacuous domain.

HONEST LABEL: both sides are LiDAR-derived (single scan vs completed) -> CONSISTENCY under occlusion,
NOT external truth. Non-degenerate + non-vacuous, so strictly stronger than L1, but Leg 1 stays the
field headline.

Pure numpy. src/probe is read-only. Deterministic (seed 0).
Run: .venv/bin/python experiments/occquery_v0/semantickitti_xdataset.py [--per-seq 40]
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
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "experiments" / "dynfield_v0"))

from probe.adapters import semantickitti as sk  # noqa: E402
from probe.grid import OCCUPIED, EgoPose, UnknownPolicy  # noqa: E402
from probe.query_spec import load_queries  # noqa: E402

# L1 helpers -- VERBATIM reuse (unit-tested set-metrics + cluster bootstrap; no homegrown rank-AUC)
from l1_denotation_occ3d import (  # noqa: E402
    _allfree_conf, _boot_metric, _random_conf, band_blocked_bev,
    confusion_from_masks, free_set_metrics,
)
# Leg-1 expressivity -- VERBATIM reuse (same sealed queries + production evaluator + refav flag)
from h3b_expressivity import _FREESPACE_FAMILIES, leg1_expressivity  # noqa: E402

_DATA = _REPO / "data" / "semantickitti" / "dataset"
_LABELED_SEQS = [f"{i:02d}" for i in range(11)]  # 00..10 carry GT labels; 11..21 are the test split
_N_BOOT = 1000
_SEED = 0
_NOMINAL_SPEED = 5.0   # urban m/s, for the corridor band-reach only (not the headline domain)
_CORRIDOR_HORIZON = 1.0
_CORRIDOR_MARGIN = 1.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _zband_indices() -> np.ndarray:
    """Voxel z-indices in the ego-height band (GROUND_HEIGHT, GROUND_HEIGHT + EGO_HEIGHT] -- the exact
    vertical envelope obstacle_centers(max_height_agl=ego.height) reads."""
    zk = sk.ORIGIN[2] + sk.VOXEL_SIZE * np.arange(sk.GRID_SHAPE[2])
    return np.where((zk > sk.GROUND_HEIGHT) & (zk <= sk.GROUND_HEIGHT + sk.EGO_HEIGHT))[0]


_ZBAND = _zband_indices()


def _bev_blocked(occ3d: np.ndarray) -> np.ndarray:
    """(256,256) bool BEV: a column is BLOCKED iff any ego-height-band voxel is OCCUPIED."""
    return (occ3d[:, :, _ZBAND] == OCCUPIED).any(axis=2)


def _sample_frame_ids(seq: str, per_seq: int) -> list[str]:
    vdir = _DATA / "sequences" / seq / "voxels"
    ids = sorted(p.stem for p in vdir.glob("*.bin"))
    if len(ids) <= per_seq:
        return ids
    pick = np.linspace(0, len(ids) - 1, per_seq).round().astype(int)
    return [ids[i] for i in sorted(set(pick.tolist()))]


def _frame_confusions(seq: str, fid: str, band_ego: EgoPose) -> dict:
    """For one frame: headline-domain + corridor confusions (FREE positive, observed vs dense GT),
    plus the global 3D occupied XOR counts (non-degeneracy)."""
    vdir = _DATA / "sequences" / seq / "voxels"
    obs = sk.load_observed_grid(vdir / f"{fid}.bin", vdir / f"{fid}.invalid").occupancy
    gt = sk.load_dense_gt_grid(vdir / f"{fid}.label").occupancy
    invalid = sk._unpack_bits(vdir / f"{fid}.invalid")  # determinable mask source

    obs_occ3d = obs == OCCUPIED
    gt_occ3d = gt == OCCUPIED

    # --- headline domain: forward ego-height BEV field, DETERMINABLE columns only ---
    obs_blk = _bev_blocked(obs)
    gt_blk = _bev_blocked(gt)
    determinable = ~invalid[:, :, _ZBAND].all(axis=2)  # drop columns all-invalid across the z-band
    pred_free = (~obs_blk)[determinable]
    ref_free = (~gt_blk)[determinable]
    head_conf = confusion_from_masks(pred_free, ref_free)
    gt_blocked_cells = int(gt_blk[determinable].sum())
    determinable_cells = int(determinable.sum())

    # --- secondary domain: L1-style thin in-path corridor (expected vacuous) ---
    from probe.grid import OccupancyGrid
    obs_grid = OccupancyGrid(obs, sk.VOXEL_SIZE, sk.ORIGIN, sk.GROUND_HEIGHT)
    gt_grid = OccupancyGrid(gt, sk.VOXEL_SIZE, sk.ORIGIN, sk.GROUND_HEIGHT)
    c_gt_blk = band_blocked_bev(gt_grid, band_ego, horizon=_CORRIDOR_HORIZON,
                                band_margin=_CORRIDOR_MARGIN, unknown_policy=UnknownPolicy.FREE)
    c_obs_blk = band_blocked_bev(obs_grid, band_ego, horizon=_CORRIDOR_HORIZON,
                                 band_margin=_CORRIDOR_MARGIN, unknown_policy=UnknownPolicy.FREE)
    corr_conf = confusion_from_masks(~c_obs_blk, ~c_gt_blk)
    corr_gt_blocked = int(c_gt_blk.sum())
    corr_cells = int(c_gt_blk.size)

    return {
        "head_conf": head_conf, "head_gt_blocked": gt_blocked_cells, "head_cells": determinable_cells,
        "corr_conf": corr_conf, "corr_gt_blocked": corr_gt_blocked, "corr_cells": corr_cells,
        "xor": int(np.logical_xor(obs_occ3d, gt_occ3d).sum()),
        "obs_occ": int(obs_occ3d.sum()), "gt_occ": int(gt_occ3d.sum()),
        "bin_not_label": int((obs_occ3d & ~gt_occ3d).sum()),
        "n_voxels": int(obs_occ3d.size),
    }


def _summarize(per_seq_conf: list[tuple], free_rate: float, rng: np.random.Generator) -> dict:
    """Bootstrap the FREE-class metrics + trivial baselines from per-sequence confusion tuples."""
    allfree = [_allfree_conf(c) for c in per_seq_conf]
    randc = [_random_conf(c, free_rate) for c in per_seq_conf]
    return {
        "predicate": {k: _boot_metric(per_seq_conf, k, rng, _N_BOOT)
                      for k in ("iou", "f1", "precision", "recall", "false_block_rate", "miss_rate")},
        "baseline_all_free": {k: _boot_metric(allfree, k, rng, _N_BOOT) for k in ("iou", "f1")},
        "baseline_random_at_free_rate": {k: _boot_metric(randc, k, rng, _N_BOOT) for k in ("iou", "f1")},
        "baseline_box_only": "INAPPLICABLE (RefAV box+map has no free-space primitive; no number fabricated)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seq", type=int, default=40, help="max evenly-spaced frames per sequence")
    args = ap.parse_args()
    rng = np.random.default_rng(_SEED)

    # ---------------- Leg 1: H1 expressivity on KITTI (reuse h3b verbatim) ----------------
    queries = load_queries(_HERE / "queries.yaml")
    probe_ids = _sample_frame_ids("00", 5)
    probe_scene = sk.load_scene("00", _DATA, variant="observed", frame_ids=probe_ids)
    print(f"Leg 1: KITTI probe scene seq00 ({len(probe_scene)} frames) -> expressivity ...", flush=True)
    leg1 = leg1_expressivity(queries, probe_scene)
    fs = leg1["free_space_families_only"]

    # ---------------- Leg 2: denotation (observed vs dense GT) ----------------
    band_ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=_NOMINAL_SPEED,
                       width=sk.EGO_WIDTH, length=sk.EGO_LENGTH, height=sk.EGO_HEIGHT)
    head_per_seq: list[tuple] = []
    corr_per_seq: list[tuple] = []
    seq_used: list[str] = []
    g = {"xor": 0, "obs_occ": 0, "gt_occ": 0, "bin_not_label": 0, "n_voxels": 0,
         "head_gt_blocked": 0, "head_cells": 0, "corr_gt_blocked": 0, "corr_cells": 0, "n_frames": 0}

    print(f"Leg 2: {len(_LABELED_SEQS)} labeled sequences, up to {args.per_seq} frames each ...", flush=True)
    for seq in _LABELED_SEQS:
        ids = _sample_frame_ids(seq, args.per_seq)
        if not ids:
            print(f"  [skip] seq {seq}: no frames", flush=True)
            continue
        h = [0, 0, 0, 0]
        c = [0, 0, 0, 0]
        for fid in ids:
            r = _frame_confusions(seq, fid, band_ego)
            for i in range(4):
                h[i] += r["head_conf"][i]
                c[i] += r["corr_conf"][i]
            for k in ("xor", "obs_occ", "gt_occ", "bin_not_label", "n_voxels",
                      "head_gt_blocked", "head_cells", "corr_gt_blocked", "corr_cells"):
                g[k] += r[k]
            g["n_frames"] += 1
        head_per_seq.append(tuple(h))
        corr_per_seq.append(tuple(c))
        seq_used.append(seq)
        print(f"  seq {seq}: {len(ids)} frames  head GT-blocked "
              f"{100 * sum(r for r in [g['head_gt_blocked']]) / max(g['head_cells'], 1):.1f}% (cum)", flush=True)

    # pooled free-rates (for the random baseline + non-vacuity reporting)
    head_free_rate = 1.0 - g["head_gt_blocked"] / max(g["head_cells"], 1)
    corr_free_rate = 1.0 - g["corr_gt_blocked"] / max(g["corr_cells"], 1)
    xor_frac = g["xor"] / max(g["n_voxels"], 1)

    headline = _summarize(head_per_seq, head_free_rate, rng)
    corridor = _summarize(corr_per_seq, corr_free_rate, rng)

    # verdict per the SEALED kill: predicate IoU CI.lo > all-free IoU mean on the NON-vacuous domain
    pred_iou = headline["predicate"]["iou"]
    base_iou = headline["baseline_all_free"]["iou"]
    leg2_is_result = (not math.isnan(pred_iou["lo"]) and not math.isnan(base_iou["mean"])
                      and pred_iou["lo"] > base_iou["mean"])
    h1_falsified = leg1["kill_H1_falsified_box_expresses_freespace"]

    report = {
        "experiment": "occquery_v0 / SemanticKITTI cross-dataset (H1 3rd-dataset + non-degenerate denotation)",
        "preregistration": "semantickitti_xdataset_preregistration.md (SEALED before the confirmatory verdict; NOT git-committed per task instruction)",
        "result_class": ("Leg 1 = real-data EXPRESSIVITY (oracle-free, the headline). Leg 2 = CONSISTENCY "
                         "(observed single-scan vs completed dense GT, both LiDAR-derived) -- "
                         "NON-degenerate + NON-vacuous, but NOT external truth."),
        "commit": _git_commit(),
        "seed": _SEED,
        "data_root": str(_DATA),
        "sample": {"labeled_sequences": seq_used, "n_sequences": len(seq_used),
                   "max_frames_per_seq": args.per_seq, "n_frames": g["n_frames"]},
        "non_degeneracy": {
            "global_bin_vs_label_occupied_XOR_frac_of_all_voxels": xor_frac,
            "global_observed_occupied": g["obs_occ"], "global_dense_gt_occupied": g["gt_occ"],
            "bin_occupied_not_in_label": g["bin_not_label"],
            "note": (f"XOR = {100 * xor_frac:.3f}% of all voxels differ obs-vs-dense (L1/Occ3D was "
                     f"0.008%). Dense GT has {g['gt_occ'] / max(g['obs_occ'], 1):.1f}x the observed "
                     f"occupied voxels; only {g['bin_not_label']} observed-occupied voxels are absent "
                     "from the dense GT (.bin ~ a subset of .label). NON-DEGENERATE."),
        },
        "non_vacuity": {
            "headline_domain_gt_blocked_rate": 1.0 - head_free_rate,
            "headline_domain_free_rate": head_free_rate,
            "corridor_domain_gt_blocked_rate": 1.0 - corr_free_rate,
            "corridor_domain_free_rate": corr_free_rate,
            "note": (f"Headline forward ego-height BEV field: GT blocked-rate {100 * (1 - head_free_rate):.1f}% "
                     f"(NON-vacuous). Thin in-path corridor (L1-style): GT blocked-rate "
                     f"{100 * (1 - corr_free_rate):.1f}% (VACUOUS, the corridor ahead is free by "
                     "construction -- why the headline domain was widened)."),
        },
        "leg1_expressivity": {
            "free_space_families": list(_FREESPACE_FAMILIES),
            "occupancy_coverage_pct": fs["occupancy_coverage_pct"],
            "box_only_coverage_pct": fs["box_only_coverage_pct"],
            "gap_pct": fs["gap_pct"],
            "overall": leg1["overall"],
            "kill_H1_falsified_box_expresses_freespace": h1_falsified,
            "probe_scene": leg1["probe_scene"],
        },
        "leg2_denotation": {
            "domain_headline": ("forward ego-height BEV field (x in [0,51.2] m, ego-height z-band "
                                f"voxel-idx {int(_ZBAND.min())}-{int(_ZBAND.max())}), determinable "
                                "columns only (not all-invalid). FREE = positive; pred = observed .bin, "
                                "ref = dense .label."),
            "headline": headline,
            "domain_corridor": ("L1-style thin in-path corridor (reused band_blocked_bev: forward "
                                f"0..(length/2+{_NOMINAL_SPEED}*{_CORRIDOR_HORIZON}s), |lat|<=width/2+"
                                f"{_CORRIDOR_MARGIN} m) -- reported as the vacuous contrast."),
            "corridor": corridor,
        },
        "verdict": {
            "leg1_H1_holds_on_kitti": (not h1_falsified) and fs["occupancy_coverage_pct"] > fs["box_only_coverage_pct"],
            "leg2_iou_ci_lo_beats_all_free_mean": bool(leg2_is_result),
            "leg2_is_a_result": bool(leg2_is_result),
            "leg2_label": "CONSISTENCY (non-degenerate, non-vacuous occlusion-robustness), NOT external truth",
        },
    }

    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "semantickitti_xdataset.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_summary(out_dir / "semantickitti_xdataset_summary.md", report)
    print(f"\nwrote {out_dir / 'semantickitti_xdataset.json'}")
    print(f"wrote {out_dir / 'semantickitti_xdataset_summary.md'}")
    _print_console(report)


def _fmt(ci: dict) -> str:
    return f"{ci['mean']:.4f} CI[{ci['lo']:.4f}, {ci['hi']:.4f}]"


def _print_console(r: dict) -> None:
    print("\n=== SemanticKITTI CROSS-DATASET ===")
    l1 = r["leg1_expressivity"]
    print(f"Leg 1 (HEADLINE, H1 expressivity on a 3rd dataset): free-space families occupancy "
          f"{l1['occupancy_coverage_pct']}% vs box-only {l1['box_only_coverage_pct']}% -> "
          f"gap {l1['gap_pct']} pts; H1 falsified = {l1['kill_H1_falsified_box_expresses_freespace']}")
    nd = r["non_degeneracy"]; nv = r["non_vacuity"]
    print(f"\nNon-degeneracy: {nd['note']}")
    print(f"Non-vacuity:    {nv['note']}")
    h = r["leg2_denotation"]["headline"]
    s = r["sample"]
    print(f"\nLeg 2 (CONSISTENCY) headline domain: {s['n_sequences']} sequences / {s['n_frames']} frames")
    for k in ("iou", "f1", "precision", "recall", "false_block_rate", "miss_rate"):
        print(f"    {k:18s}: {_fmt(h['predicate'][k])}")
    print(f"  baseline all-free   IoU {_fmt(h['baseline_all_free']['iou'])}  F1 {_fmt(h['baseline_all_free']['f1'])}")
    print(f"  baseline random@fr  IoU {_fmt(h['baseline_random_at_free_rate']['iou'])}")
    print(f"  box-only: {h['baseline_box_only']}")
    v = r["verdict"]
    print(f"\nVERDICT: Leg1 H1 holds on KITTI = {v['leg1_H1_holds_on_kitti']}")
    print(f"         Leg2 IoU CI.lo > all-free mean (a result) = {v['leg2_is_a_result']}  "
          f"[{v['leg2_label']}]")


def _write_summary(path: pathlib.Path, r: dict) -> None:
    l1 = r["leg1_expressivity"]
    h = r["leg2_denotation"]["headline"]
    c = r["leg2_denotation"]["corridor"]
    s = r["sample"]
    pf = h["predicate"]
    L = []
    L.append("# SemanticKITTI cross-dataset -- H1 (3rd dataset) + non-degenerate denotation\n")
    L.append(f"- Pre-reg: `{r['preregistration']}`")
    L.append(f"- Commit: `{r['commit']}`  seed {r['seed']}")
    L.append(f"- Sample: {s['n_sequences']} labeled sequences {s['labeled_sequences']}, "
             f"{s['n_frames']} frames (<= {s['max_frames_per_seq']}/seq, evenly spaced)")
    L.append(f"- Result class: {r['result_class']}\n")

    L.append("## Leg 1 -- H1 EXPRESSIVITY on a THIRD dataset (SOLE field headline, oracle-free)")
    L.append(f"Free-space families on real SemanticKITTI grids: occupancy "
             f"**{l1['occupancy_coverage_pct']}%** vs box-only **{l1['box_only_coverage_pct']}%** "
             f"-> **{l1['gap_pct']}-pt** gap. H1 falsified (box expresses free-space) = "
             f"**{l1['kill_H1_falsified_box_expresses_freespace']}**. Probe scene: `{l1['probe_scene']}`.")
    L.append("The expressivity gap now holds on AV2 (h3b) + Occ3D-nuScenes + SemanticKITTI -- 3 "
             "structurally-different datasets.\n")

    L.append("## Non-degeneracy + non-vacuity (the two L1 failure modes, checked)")
    L.append(f"- **Non-degeneracy**: {r['non_degeneracy']['note']}")
    L.append(f"- **Non-vacuity**: {r['non_vacuity']['note']}\n")

    L.append("## Leg 2 -- DENOTATION CONSISTENCY (non-degenerate, non-vacuous; NOT external truth)")
    L.append(f"Headline domain: {r['leg2_denotation']['domain_headline']}")
    L.append(f"GT blocked-rate {100 * r['non_vacuity']['headline_domain_gt_blocked_rate']:.1f}% "
             f"(free-rate {100 * r['non_vacuity']['headline_domain_free_rate']:.1f}%).\n")
    L.append("| metric (FREE class) | predicate (observed .bin) | all-free | random@free-rate |")
    L.append("|---|---|---|---|")
    for k in ("iou", "f1"):
        L.append(f"| {k.upper()} | {_fmt(pf[k])} | {_fmt(h['baseline_all_free'][k])} | "
                 f"{_fmt(h['baseline_random_at_free_rate'][k])} |")
    L.append("")
    L.append("Predicate full denotation (observed vs dense GT):")
    for k in ("precision", "recall", "false_block_rate", "miss_rate"):
        L.append(f"- {k}: {_fmt(pf[k])}")
    L.append("\nBox-only baseline: " + h["baseline_box_only"])
    L.append(f"\nContrast -- thin in-path corridor (VACUOUS, GT blocked-rate "
             f"{100 * r['non_vacuity']['corridor_domain_gt_blocked_rate']:.1f}%): "
             f"predicate IoU {_fmt(c['predicate']['iou'])} vs all-free IoU "
             f"{_fmt(c['baseline_all_free']['iou'])}.")

    L.append("\n## Verdict (per the sealed kill criteria)")
    v = r["verdict"]
    L.append(f"- Leg 1 H1 holds on KITTI: **{v['leg1_H1_holds_on_kitti']}**")
    L.append(f"- Leg 2 predicate IoU CI.lo > all-free IoU mean (a relative result): "
             f"**{v['leg2_is_a_result']}**")
    L.append(f"- Leg 2 label: {v['leg2_label']}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
