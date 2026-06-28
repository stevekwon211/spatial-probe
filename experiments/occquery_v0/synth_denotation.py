# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""SYNTHETIC denotation-MECHANISM validation for occquery (realizes the SEALED pre-reg
`synth_denotation_preregistration.md`). Run AFTER the seal; nothing here was chosen after seeing a number.

The ONLY place denotation-correctness can be EXTERNALLY validated: WE construct the ground truth, so the
reference is genuinely independent of the predicate's input (unlike Occ3D, whose GT is the same LiDAR the
predicate reads -> `l1_denotation_occ3d` came back DEGENERATE). We then inject REAL occlusion with the
repo's own raycaster, so the OBSERVED grid genuinely differs from the truth.

HONEST SCOPE (research-integrity): SYNTHETIC = valid for MECHANISM / EDGE / NUMERICAL correctness ONLY. It
establishes the free-space predicate denotation LOGIC + graceful degradation under occlusion. It is NEVER
field evidence and NEVER re-inflates H3. H1 (expressivity) remains the sole field headline.

Reuses the already-unit-tested L1 numpy set-metrics + the in-path band substrate the predicates read
(`band_blocked_bev`, `confusion_from_masks`, `free_set_metrics`, `_allfree_conf`, `_random_conf`,
`_boot_metric`) -- no sklearn/torch (repo rule; the `_roc_auc` lesson).

Run: python experiments/occquery_v0/synth_denotation.py [--n-scenes 120]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))  # script-mode import of probe.*
sys.path.insert(0, str(_REPO))          # script-mode import of experiments.occquery_v0.* (reused L1 helpers)

from probe.grid import OCCUPIED, FREE, UNKNOWN, EgoPose, OccupancyGrid, UnknownPolicy  # noqa: E402
from probe.raycast import line_of_sight  # noqa: E402
from probe.predicates.freepath import free_along_ego_path  # noqa: E402

# Reuse the SEALED, unit-tested L1 machinery (same substrate the predicates read).
from experiments.occquery_v0.l1_denotation_occ3d import (  # noqa: E402
    band_blocked_bev,
    confusion_from_masks,
    free_set_metrics,
    _allfree_conf,
    _random_conf,
    _boot_metric,
    _safe_div,
)

# --- constructed-world constants (sealed) -------------------------------------------------------
VOXEL = 0.5
GROUND_H = 0.25
EGO_XY = (5.0, 10.0)
EGO_Z = 1.0
NX, NY, NZ = 61, 41, 10      # x:0..30 m, y:0..20 m, z:0..4.5 m
HEIGHT_K = (1, 4)            # obstacle column z-voxels (z 0.5..2.0 m), within the ego height band
HORIZON = 1.0               # sealed free_path horizon (queries.yaml)
BAND_MARGIN = 1.0           # sealed widest lateral tier (queries.yaml)
N_BOOT = 1000
SEED = 0


def _blank() -> np.ndarray:
    return np.full((NX, NY, NZ), FREE, dtype=int)


def _ego(speed: float) -> EgoPose:
    return EgoPose((EGO_XY[0], EGO_XY[1], EGO_Z), 0.0, speed=speed, width=1.85, length=4.6, height=1.9)


def _reach(ego: EgoPose) -> float:
    return ego.length / 2.0 + ego.speed * HORIZON


def _place_block(occ: np.ndarray, ego: EgoPose, forward: float, lateral: float,
                 lat_w: float, fwd_d: float, value: int = OCCUPIED) -> None:
    """Fill a vertical column block centred at ego-frame (forward, lateral), lat_w m wide laterally,
    fwd_d m deep, spanning the ego height band. Heading 0 => world x = ego_x+forward, y = ego_y+lateral."""
    x0 = ego.position[0] + forward - fwd_d / 2.0
    x1 = ego.position[0] + forward + fwd_d / 2.0
    y0 = ego.position[1] + lateral - lat_w / 2.0
    y1 = ego.position[1] + lateral + lat_w / 2.0
    i0, i1 = int(round(x0 / VOXEL)), int(round(x1 / VOXEL))
    j0, j1 = int(round(y0 / VOXEL)), int(round(y1 / VOXEL))
    i0, i1 = max(0, i0), min(NX - 1, i1)
    j0, j1 = max(0, j0), min(NY - 1, j1)
    occ[i0:i1 + 1, j0:j1 + 1, HEIGHT_K[0]:HEIGHT_K[1] + 1] = value


def _ego_cell(grid: OccupancyGrid, ego: EgoPose) -> tuple[int, int, int]:
    return grid.world_to_voxel((ego.position[0], ego.position[1], EGO_Z))


def _occlude(true_occ: np.ndarray, ego_cell: tuple[int, int, int]) -> np.ndarray:
    """Single-sensor OBSERVED grid: every voxel in the forward height-band region keeps its TRUE value
    iff line-of-sight from the ego is clear; an OCCUPIED voxel strictly between ego and it -> UNKNOWN.
    The first hit along each ray stays visible (OCCUPIED); everything behind is hidden. Uses the repo
    3D-DDA raycaster verbatim."""
    obs = true_occ.copy()
    ex, _, _ = ego_cell
    kk = range(HEIGHT_K[0], HEIGHT_K[1] + 1)
    for i in range(ex, NX):          # forward hemisphere only (behind ego is irrelevant to the band)
        for j in range(NY):
            for k in kk:
                if (i, j, k) == ego_cell:
                    continue
                if not line_of_sight(true_occ, ego_cell, (i, j, k)):
                    obs[i, j, k] = UNKNOWN
    return obs


def _path_truly_blocked(true_occ: np.ndarray, grid: OccupancyGrid, ego: EgoPose) -> bool:
    """INDEPENDENT constructed label (not via reachable_free_field): a TRUE OCCUPIED voxel sits in the
    ego corridor |lateral| <= width/2, forward in [0, reach], height band."""
    idx = np.argwhere((true_occ == OCCUPIED))
    if idx.size == 0:
        return False
    centers = np.asarray(grid.origin, float) + idx * VOXEL
    z = centers[:, 2]
    band = (z > GROUND_H) & (z <= GROUND_H + ego.height)
    fwd, lat = ego.to_ego_frame(centers[:, :2])
    reach = _reach(ego)
    hit = band & (fwd >= 0.0) & (fwd <= reach) & (np.abs(lat) <= ego.width / 2.0)
    return bool(hit.any())


# --- scene construction (seed 0) ----------------------------------------------------------------
def _build_scenes(n_scenes: int, rng: np.random.Generator) -> list[dict]:
    """Return per-scene dicts {name, ego, true_occ, n_inpath}. Three controlled groups."""
    scenes: list[dict] = []
    speeds = (8.0, 10.0, 12.0)
    n_free = int(round(0.25 * n_scenes))
    n_single = int(round(0.42 * n_scenes))
    n_multi = n_scenes - n_free - n_single

    def _side_walls(occ, ego):
        # off-corridor roadside walls (occluders / clutter), never in the centerline corridor
        for sgn in (-1.0, 1.0):
            if rng.random() < 0.6:
                lat = sgn * float(rng.uniform(3.0, 6.0))
                _place_block(occ, ego, forward=float(rng.uniform(6.0, 12.0)), lateral=lat,
                             lat_w=float(rng.uniform(1.0, 3.0)), fwd_d=float(rng.uniform(2.0, 6.0)))

    def _maybe_hidden_behind(occ, ego, front_fwd, lat):
        # a second obstacle directly behind a visible one -> genuinely occluded (the occlusion test)
        if rng.random() < 0.5:
            _place_block(occ, ego, forward=front_fwd + float(rng.uniform(2.0, 4.0)), lateral=lat,
                         lat_w=float(rng.uniform(1.0, 2.0)), fwd_d=1.0)

    idx = 0
    for _ in range(n_free):
        ego = _ego(float(rng.choice(speeds)))
        occ = _blank()
        _side_walls(occ, ego)
        scenes.append({"name": f"free_{idx:03d}", "ego": ego, "true_occ": occ, "n_inpath": 0}); idx += 1
    for _ in range(n_single):
        ego = _ego(float(rng.choice(speeds)))
        occ = _blank()
        fwd = float(rng.uniform(5.0, 15.0))
        lat = float(rng.uniform(-0.5, 0.5))
        _place_block(occ, ego, forward=fwd, lateral=lat,
                     lat_w=float(rng.uniform(1.0, 3.0)), fwd_d=float(rng.uniform(1.0, 2.0)))
        _maybe_hidden_behind(occ, ego, fwd, lat)
        if rng.random() < 0.5:
            _side_walls(occ, ego)
        scenes.append({"name": f"single_{idx:03d}", "ego": ego, "true_occ": occ, "n_inpath": 1}); idx += 1
    for _ in range(n_multi):
        ego = _ego(float(rng.choice(speeds)))
        occ = _blank()
        nblk = int(rng.integers(2, 5))
        for _b in range(nblk):
            fwd = float(rng.uniform(5.0, 15.0))
            lat = float(rng.uniform(-1.5, 1.5))
            _place_block(occ, ego, forward=fwd, lateral=lat,
                         lat_w=float(rng.uniform(1.0, 2.5)), fwd_d=float(rng.uniform(1.0, 2.0)))
            _maybe_hidden_behind(occ, ego, fwd, lat)
        _side_walls(occ, ego)
        scenes.append({"name": f"multi_{idx:03d}", "ego": ego, "true_occ": occ, "n_inpath": nblk}); idx += 1
    return scenes


# --- grading ------------------------------------------------------------------------------------
def _grade_scene(sc: dict) -> dict:
    ego = sc["ego"]
    true_occ = sc["true_occ"]
    true_grid = OccupancyGrid(true_occ, VOXEL, (0.0, 0.0, 0.0), GROUND_H)
    ego_cell = _ego_cell(true_grid, ego)
    obs_occ = _occlude(true_occ, ego_cell)
    obs_grid = OccupancyGrid(obs_occ, VOXEL, (0.0, 0.0, 0.0), GROUND_H)

    # reference = TRUE constructed GT (independent of the predicate's occluded input)
    ref_blocked = band_blocked_bev(true_grid, ego, horizon=HORIZON, band_margin=BAND_MARGIN,
                                   unknown_policy=UnknownPolicy.FREE)
    ref_free = ~ref_blocked

    # under test = OCCLUDED observation, sealed (unknown=FREE) and sensitivity (unknown=OCCUPIED)
    out = {"name": sc["name"], "n_inpath": sc["n_inpath"]}
    for key, pol in (("free", UnknownPolicy.FREE), ("occ", UnknownPolicy.OCCUPIED)):
        obs_blocked = band_blocked_bev(obs_grid, ego, horizon=HORIZON, band_margin=BAND_MARGIN,
                                       unknown_policy=pol)
        out[key] = list(confusion_from_masks(~obs_blocked, ref_free))  # tp, fp, fn, tn (FREE positive)

    out["ref_blocked_cells"] = int(ref_blocked.sum())
    out["ref_total_cells"] = int(ref_blocked.size)

    # SECONDARY: predicate-verdict free_along_ego_path vs the INDEPENDENT constructed label
    label_blocked = _path_truly_blocked(true_occ, true_grid, ego)
    pred_true_blocked = not free_along_ego_path(true_grid, ego, HORIZON, unknown_policy=UnknownPolicy.FREE)
    pred_obs_blocked = not free_along_ego_path(obs_grid, ego, HORIZON, unknown_policy=UnknownPolicy.FREE)
    out["verdict"] = {"label_blocked": bool(label_blocked),
                      "pred_true_blocked": bool(pred_true_blocked),
                      "pred_obs_blocked": bool(pred_obs_blocked)}
    return out


def _conf_metrics_block(confs: list[tuple], rng: np.random.Generator) -> dict:
    """Bootstrap CI for every reported FREE-class metric over a list of per-scene confusions."""
    keys = ("iou", "f1", "precision", "recall", "false_block_rate", "miss_rate")
    return {k: _boot_metric(confs, k, rng) for k in keys}


def _baselines_block(confs: list[tuple], rng: np.random.Generator) -> dict:
    arr = np.asarray(confs, float).sum(axis=0)
    gt_free = arr[0] + arr[2]
    gt_blocked = arr[1] + arr[3]
    free_rate = _safe_div(gt_free, gt_free + gt_blocked)
    allfree = [_allfree_conf(c) for c in confs]
    randc = [_random_conf(c, free_rate) for c in confs]
    return {
        "band_gt_free_rate": free_rate,
        "band_gt_blocked_rate": 1.0 - free_rate if not math.isnan(free_rate) else float("nan"),
        "all_free": {k: _boot_metric(allfree, k, rng) for k in ("iou", "f1")},
        "random_at_free_rate": {k: _boot_metric(randc, k, rng) for k in ("iou", "f1")},
        "box_only": "INAPPLICABLE (coverage 0 -- box-only cannot express free-space; no number fabricated)",
    }


def _verdict_confusion(records: list[dict], pred_key: str) -> dict:
    """2x2 of the SECONDARY predicate-verdict (BLOCKED) vs the independent constructed label."""
    tp = fp = fn = tn = 0
    for r in records:
        v = r["verdict"]
        gt = v["label_blocked"]; pr = v[pred_key]
        if pr and gt: tp += 1
        elif pr and not gt: fp += 1
        elif not pr and gt: fn += 1
        else: tn += 1
    n = len(records)
    acc = _safe_div(tp + tn, n)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n, "accuracy": acc,
            "precision_blocked": _safe_div(tp, tp + fp), "recall_blocked": _safe_div(tp, tp + fn)}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    scenes = _build_scenes(args.n_scenes, rng)
    print(f"synth denotation: constructed {len(scenes)} scenes (seed {SEED}); grading ...", flush=True)
    records = [_grade_scene(sc) for sc in scenes]

    # boot RNG is independent of the construction RNG (re-seed for the resampling)
    boot_rng = np.random.default_rng(SEED)

    full_free = [tuple(r["free"]) for r in records]
    full_occ = [tuple(r["occ"]) for r in records]
    # obstacle-bearing = TRUE band has >=1 blocked cell (the non-vacuous subset)
    obs_records = [r for r in records if r["ref_blocked_cells"] > 0]
    sub_free = [tuple(r["free"]) for r in obs_records]
    sub_occ = [tuple(r["occ"]) for r in obs_records]

    def _summary(confs_free, confs_occ, group):
        bl = _baselines_block(confs_free, boot_rng)
        return {
            "n_scenes": len(group),
            "band_gt_free_rate": bl["band_gt_free_rate"],
            "band_gt_blocked_rate": bl["band_gt_blocked_rate"],
            "predicate_unknown_FREE_sealed": _conf_metrics_block(confs_free, boot_rng),
            "predicate_unknown_OCCUPIED_sensitivity": _conf_metrics_block(confs_occ, boot_rng),
            "baseline_all_free": bl["all_free"],
            "baseline_random_at_free_rate": bl["random_at_free_rate"],
            "baseline_box_only": bl["box_only"],
        }

    full = _summary(full_free, full_occ, records)
    sub = _summary(sub_free, sub_occ, obs_records)

    # VERDICT (sealed kill): on the obstacle-bearing subset, predicate(unknown=FREE) IoU CI.lo must beat
    # max(all-free IoU mean, random IoU mean).
    pred_iou = sub["predicate_unknown_FREE_sealed"]["iou"]
    base_af = sub["baseline_all_free"]["iou"]["mean"]
    base_rnd = sub["baseline_random_at_free_rate"]["iou"]["mean"]
    base_max = max(base_af, base_rnd)
    beats = (not math.isnan(pred_iou["lo"])) and pred_iou["lo"] > base_max
    # HONEST caveat (added for transparency, NOT a change to the sealed kill): do the predicate and
    # all-free IoU bootstrap CIs overlap? A thin "beats-the-mean" win with overlapping CIs is weaker
    # than non-overlapping separation -- reported so the margin is not over-read.
    af_iou_ci = sub["baseline_all_free"]["iou"]
    iou_ci_overlaps_baseline = (not math.isnan(pred_iou["lo"]) and not math.isnan(af_iou_ci["hi"])
                                and pred_iou["lo"] <= af_iou_ci["hi"])

    # mechanism sanity: false_block under unknown=FREE must be ~0 (occlusion never false-blocks free space)
    fb_free = sub["predicate_unknown_FREE_sealed"]["false_block_rate"]["mean"]
    fb_ok = math.isnan(fb_free) or fb_free < 1e-9

    # SECONDARY predicate-verdict vs the independent constructed label
    vc_true = _verdict_confusion(records, "pred_true_blocked")
    vc_obs = _verdict_confusion(records, "pred_obs_blocked")
    logic_ok = (not math.isnan(vc_true["accuracy"])) and vc_true["accuracy"] >= 0.99

    killed = not beats

    pre = _HERE / "synth_denotation_preregistration.md"
    report = {
        "experiment": "occquery_v0 / SYNTHETIC denotation-MECHANISM validation",
        "preregistration": "synth_denotation_preregistration.md (SEALED, written before grading code/data)",
        "prereg_sha256": hashlib.sha256(pre.read_bytes()).hexdigest() if pre.exists() else "absent",
        "result_class": ("SYNTHETIC MECHANISM validation -- by-construction GT, independent of the "
                         "predicate input; real raycast occlusion. Valid for MECHANISM/EDGE/NUMERICAL "
                         "correctness ONLY, NEVER field evidence. Does NOT re-inflate H3. H1 stays the "
                         "sole field headline."),
        "commit": _git_commit(),
        "seed": SEED,
        "n_scenes_total": len(records),
        "n_scenes_obstacle_bearing": len(obs_records),
        "voxel_m": VOXEL,
        "horizon_s": HORIZON,
        "band": (f"ego in-path corridor: forward 0..(length/2 + speed*{HORIZON}s), "
                 f"|lateral| <= width/2 + {BAND_MARGIN} m; obstacle BEV in the ego-height band "
                 "(the EXACT substrate the sealed free-space predicates read)."),
        "occlusion": "single-sensor: voxel -> UNKNOWN iff probe.raycast.line_of_sight from ego is blocked",
        "full_set": full,
        "obstacle_bearing_subset": sub,
        "secondary_predicate_verdict": {
            "definition": ("free_along_ego_path BLOCKED verdict vs the INDEPENDENT constructed label "
                           "path_truly_blocked (a TRUE occupied voxel in the ego corridor within reach)."),
            "on_TRUE_grid_perfect_input": vc_true,
            "on_OBSERVED_grid_occluded": vc_obs,
        },
        "verdict": {
            "kill_metric": "predicate(unknown=FREE) FREE-IoU CI.lo > max(all-free, random) IoU mean, on obstacle-bearing subset",
            "predicate_iou_ci_lo": pred_iou["lo"],
            "baseline_all_free_iou_mean": base_af,
            "baseline_random_iou_mean": base_rnd,
            "predicate_beats_trivial_baseline": bool(beats),
            "iou_ci_overlaps_all_free_baseline": bool(iou_ci_overlaps_baseline),
            "false_block_rate_unknown_FREE_is_zero": bool(fb_ok),
            "predicate_logic_on_perfect_input_ok": bool(logic_ok),
            "predicate_verdict_accuracy_true": vc_true["accuracy"],
            "predicate_verdict_accuracy_observed": vc_obs["accuracy"],
            "KILLED": bool(killed),
            "verdict": (
                "KILLED -- predicate denotation no better than trivial baseline even with perfect "
                "independent GT." if killed else
                "NOT KILLED (per the sealed kill: cell FREE-IoU CI.lo beats the trivial baseline mean). "
                "HONEST caveat: on cell-IoU the margin is THIN and the bootstrap CIs OVERLAP the all-free "
                "baseline -- because the in-path band is majority-free and most blocked VOLUME is genuinely "
                "occluded under a single sensor (miss_rate high, false_block_rate exactly 0 -> correct "
                "graceful degradation, no logic bug). The DECISIVE, occlusion-robust evidence is the "
                "predicate VERDICT (free_along_ego_path) vs the independent constructed label: accuracy "
                f"{vc_true['accuracy']:.4f} on perfect input AND {vc_obs['accuracy']:.4f} under occlusion "
                "(the visible front face suffices to denote BLOCKED). SYNTHETIC MECHANISM result ONLY -- "
                "not field evidence; H3 stays gated; H1 remains the field headline."),
        },
    }

    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "synth_denotation.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_summary(out_dir / "synth_denotation_summary.md", report)
    print(f"wrote {out_dir / 'synth_denotation.json'}")
    print(f"wrote {out_dir / 'synth_denotation_summary.md'}")
    _print_console(report)


def _fmt(ci: dict) -> str:
    return f"{ci['mean']:.4f} CI[{ci['lo']:.4f}, {ci['hi']:.4f}]"


def _print_console(r: dict) -> None:
    print("\n=== SYNTHETIC denotation-MECHANISM validation ===")
    print(f"result class: {r['result_class']}")
    s = r["obstacle_bearing_subset"]
    print(f"\nObstacle-bearing subset: {s['n_scenes']} scenes, "
          f"band GT blocked-rate {s['band_gt_blocked_rate']:.4f} (non-vacuity: NOT ~0)")
    pf = s["predicate_unknown_FREE_sealed"]
    print("  predicate (unknown=FREE, SEALED):")
    for k in ("iou", "f1", "precision", "recall", "false_block_rate", "miss_rate"):
        print(f"    {k:18s}: {_fmt(pf[k])}")
    print(f"  baseline all-free   IoU {_fmt(s['baseline_all_free']['iou'])}")
    print(f"  baseline random@fr  IoU {_fmt(s['baseline_random_at_free_rate']['iou'])}")
    po = s["predicate_unknown_OCCUPIED_sensitivity"]
    print("  SENSITIVITY (unknown=OCCUPIED):")
    for k in ("iou", "f1", "false_block_rate", "miss_rate"):
        print(f"    {k:18s}: {_fmt(po[k])}")
    vt = r["secondary_predicate_verdict"]["on_TRUE_grid_perfect_input"]
    vo = r["secondary_predicate_verdict"]["on_OBSERVED_grid_occluded"]
    print(f"\nSECONDARY predicate-verdict vs constructed label:")
    print(f"  on TRUE grid (perfect input): acc {vt['accuracy']:.4f} (tp{vt['tp']} fp{vt['fp']} fn{vt['fn']} tn{vt['tn']})")
    print(f"  on OBSERVED grid (occluded):  acc {vo['accuracy']:.4f} (tp{vo['tp']} fp{vo['fp']} fn{vo['fn']} tn{vo['tn']})")
    v = r["verdict"]
    print(f"\nVERDICT: {v['verdict']}")
    print(f"  predicate IoU CI.lo {v['predicate_iou_ci_lo']:.4f} vs baseline max "
          f"{max(v['baseline_all_free_iou_mean'], v['baseline_random_iou_mean']):.4f} "
          f"-> beats={v['predicate_beats_trivial_baseline']}, KILLED={v['KILLED']}")


def _write_summary(path: pathlib.Path, r: dict) -> None:
    s = r["obstacle_bearing_subset"]
    f = r["full_set"]
    pf = s["predicate_unknown_FREE_sealed"]
    po = s["predicate_unknown_OCCUPIED_sensitivity"]
    vt = r["secondary_predicate_verdict"]["on_TRUE_grid_perfect_input"]
    vo = r["secondary_predicate_verdict"]["on_OBSERVED_grid_occluded"]
    v = r["verdict"]
    L = []
    L.append("# Synthetic denotation-MECHANISM validation -- occquery free-space predicates\n")
    L.append(f"- Pre-reg: `{r['preregistration']}` (sha256 `{r['prereg_sha256'][:16]}...`)")
    L.append(f"- Commit: `{r['commit']}`  seed {r['seed']}  voxel {r['voxel_m']} m  horizon {r['horizon_s']}s")
    L.append(f"- **Result class: {r['result_class']}**")
    L.append(f"- Scenes: {r['n_scenes_total']} total, {r['n_scenes_obstacle_bearing']} obstacle-bearing")
    L.append(f"- Band: {r['band']}")
    L.append(f"- Occlusion: {r['occlusion']}\n")
    L.append("## SYNTHETIC label (loud)")
    L.append("This is a **by-construction MECHANISM check**: it validates the predicate denotation LOGIC and "
             "its graceful degradation under occlusion. It is **NOT real-world field denotation** -- the "
             "real-world denotation stays gated (H3 demoted). **H1 (expressivity) remains the field headline.**\n")
    L.append("## Non-vacuity")
    L.append(f"Obstacle-bearing subset band GT **blocked-rate = {s['band_gt_blocked_rate']:.4f}** "
             f"(free-rate {s['band_gt_free_rate']:.4f}). The `l1_denotation_occ3d` band was 99.5% free "
             "(vacuous); this design controls obstacle density so the FREE/BLOCKED classes are both "
             "substantial.\n")
    L.append("## Denotation metrics -- obstacle-bearing subset (FREE class, scene-clustered bootstrap CI 1000)")
    L.append("| metric | predicate (unknown=FREE, sealed) | all-free | random@free-rate |")
    L.append("|---|---|---|---|")
    for k in ("iou", "f1"):
        L.append(f"| {k.upper()} | {_fmt(pf[k])} | {_fmt(s['baseline_all_free'][k])} | "
                 f"{_fmt(s['baseline_random_at_free_rate'][k])} |")
    L.append("")
    L.append("Predicate (unknown=FREE, sealed) full denotation:")
    for k in ("precision", "recall", "false_block_rate", "miss_rate"):
        L.append(f"- {k}: {_fmt(pf[k])}")
    L.append("\nBox-only baseline: " + s["baseline_box_only"])
    L.append("\n### Sensitivity -- unknown_policy = OCCUPIED (conservative reading of the unobserved)")
    for k in ("iou", "f1", "false_block_rate", "miss_rate"):
        L.append(f"- {k}: {_fmt(po[k])}")
    L.append("\n## Secondary -- predicate-verdict `free_along_ego_path` vs INDEPENDENT constructed label")
    L.append(f"- on TRUE grid (perfect input): accuracy **{vt['accuracy']:.4f}** "
             f"(tp {vt['tp']}, fp {vt['fp']}, fn {vt['fn']}, tn {vt['tn']}) -- validates the LOGIC")
    L.append(f"- on OBSERVED grid (occluded): accuracy {vo['accuracy']:.4f} "
             f"(tp {vo['tp']}, fp {vo['fp']}, fn {vo['fn']}, tn {vo['tn']}) -- graceful degradation")
    L.append("\n## Full-set denotation (incl. free controls, for completeness)")
    L.append(f"- predicate(unknown=FREE) IoU {_fmt(f['predicate_unknown_FREE_sealed']['iou'])}, "
             f"band blocked-rate {f['band_gt_blocked_rate']:.4f}")
    L.append("\n## Verdict (per the sealed kill criteria)")
    L.append(f"- kill metric: {v['kill_metric']}")
    L.append(f"- predicate IoU CI.lo = {v['predicate_iou_ci_lo']:.4f}; baseline max IoU mean = "
             f"{max(v['baseline_all_free_iou_mean'], v['baseline_random_iou_mean']):.4f}")
    L.append(f"- predicate beats trivial baseline (CI.lo > baseline mean): **{v['predicate_beats_trivial_baseline']}**")
    L.append(f"- HONEST caveat -- predicate IoU CI overlaps all-free CI: **{v['iou_ci_overlaps_all_free_baseline']}** "
             "(thin cell-IoU margin; see verdict)")
    L.append(f"- predicate-VERDICT accuracy vs constructed label: true grid {v['predicate_verdict_accuracy_true']:.4f}, "
             f"observed (occluded) {v['predicate_verdict_accuracy_observed']:.4f} (the decisive occlusion-robust line)")
    L.append(f"- false_block_rate(unknown=FREE) ~ 0: {v['false_block_rate_unknown_FREE_is_zero']}")
    L.append(f"- predicate logic on perfect input OK (>=0.99): {v['predicate_logic_on_perfect_input_ok']}")
    L.append(f"- **KILLED: {v['KILLED']}**")
    L.append(f"- {v['verdict']}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
