# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""H3b -- occupancy-native vs box-only, the SOLO/CPU honest version (realizes the SEALED pre-reg
`h3b_expressivity_preregistration.md`, commit a47b500). Run AFTER the seal; nothing here was chosen
after seeing a number.

Two legs, both solo/CPU, both on REAL AV2 (not synthetic):

  Leg 1 -- EXPRESSIVITY coverage (oracle-free, structural, made into a real-data number). Over the
    SEALED query set (`queries.yaml`), per family, what fraction can each backend EXPRESS?
      * occupancy coverage = the occupancy queries that actually RUN over real AV2 logs without error
        (a real-data evaluation, not an assumed flag).
      * box-only coverage  = the `refav_expressible` flag (RefAV's released 32-function set: tracked
        boxes + HD-map polygons, NO dense-occupancy / free-space primitive -- verified against
        RefAV's atomic_functions.py, recorded in queries.yaml). Box-only's free-space queries are
        STRUCTURALLY inexpressible.
    Kill (declared before data): H1 falsified iff box-only CAN express the free-space family.

  Leg 2 -- FP-SIDE DENOTATION vs the INDEPENDENT traversal oracle. For the free-space queries
    occupancy CAN express, is its denotation CORRECT on the driven-free ribbon? The traversal oracle
    (oracle_traversal.py, sealed oracle_traversal_v0_1, verdict RELIABLE) gives FREE ground truth: the
    ego's future swept ribbon is space the vehicle physically traversed = provably FREE. We reuse that
    EXACT ribbon (ego future sweep, `_rect_mask`, far=3.9, window=10, same usable-frame gating) on the
    same held-out free-driving logs, and on each driven frame ask whether occupancy's free-space query
    predicates (`free_along_ego_path`, `min_free_width_along_path`) denote FREE/passable. They should
    (traversal RELIABLE => occupancy ~never false-blocks the driven path). Box-only has NO free-space
    denotation to grade => INAPPLICABLE, not 0.
    Kill (declared before data): Leg-2 fails iff occupancy false-blocks the driven path (predicate
    denotes BLOCKED where the ego physically drove).

Honest bound (per the pre-reg): the full both-sided >=20-F1 gap additionally needs BLOCKED-side truth
(unboxed-obstacle recall), empirically CLOSED solo/CPU this session, GPU-gated. This file does NOT
claim it. We claim: expressivity dominance (Leg 1) + FP-side denotation correctness with box-only
inapplicability (Leg 2).

Independence ledger (Leg 2): traversal oracle modality (recorded GPS/IMU-derived city_SE3 poses) !=
active LiDAR; algorithm (rigid-body sweep) != voxelization. Genuinely independent of the occupancy it
grades.

Pure numpy/scipy. src/probe is read-only (imported, never modified). Deterministic (rng seed 0).
Run: python experiments/occquery_v0/h3b_expressivity.py [--window 10] [--far 3.9] [--limit 8]
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
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "experiments" / "dynfield_v0"))

import pyarrow as pa

from harness_v2 import _boot_mean  # bootstrap CI by log (reused; same as oracle_traversal)
from probe.adapters import av2_sensor
from probe.grid import UnknownPolicy
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path
from probe.query_spec import load_queries

# Reuse the traversal oracle's ribbon primitives verbatim (the RELIABLE oracle) -- DO NOT reimplement.
from oracle_traversal import _bev_centers, _pose_at, _rect_mask, _world_poses

_AV2 = _HERE.parents[1] / "data" / "danger" / "av2_sensor"
_HELDOUT = _HERE / "oracle_heldout_logs.json"  # held-out free-driving logs = the RELIABLE substrate
_VOX = av2_sensor.VOXEL_SIZE
_OX, _OY = av2_sensor.ORIGIN[0], av2_sensor.ORIGIN[1]
_NX, _NY = av2_sensor.GRID_SHAPE[0], av2_sensor.GRID_SHAPE[1]
_MIN_SWEEP_CELLS = 5  # estimand undefined when the ego barely moves (stopped) -> skip (sealed)

# query family <- query id prefix (the comment-delimited groups in queries.yaml)
_FAMILIES = {
    "clearance": ("grazing_side_clearance", "tight_clearance_at_speed",
                  "moderate_side_clearance_at_speed", "tight_side_clearance", "moderate_side_clearance"),
    "centerline": ("centerline_obstacle_grazing", "centerline_obstacle_tight",
                   "centerline_obstacle_moderate_at_urban_speed", "centerline_obstacle_moderate",
                   "centerline_obstacle_tight_at_speed"),
    "free_path": ("free_path_blocked_within_body_length", "free_path_is_blocked",
                  "free_path_blocked_at_two_second_horizon", "free_path_blocked_at_long_range_high_speed",
                  "free_path_blocked_or_side_clearance_tight"),
    "corridor": ("corridor_pinches_fully_shut", "near_corridor_below_vehicle_width",
                 "corridor_narrows_below_vehicle_width", "far_corridor_below_vehicle_width",
                 "corridor_below_half_meter"),
    "box_baseline": ("near_a_tracked_vehicle", "within_one_carlength_of_a_tracked_vehicle",
                     "near_a_tracked_pedestrian", "near_a_tracked_bicycle"),
}
# the free-space families box-only structurally cannot express (the H1 gap)
_FREESPACE_FAMILIES = ("clearance", "centerline", "free_path", "corridor")
# horizons the free-path query family samples (queries.yaml: 0.5 / 1.0 / 2.0 / 4.0 s)
_FREEPATH_HORIZONS = (0.5, 1.0, 2.0, 4.0)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _family_of(qid: str) -> str:
    for fam, ids in _FAMILIES.items():
        if qid in ids:
            return fam
    return "unknown"


# --------------------------------------------------------------------------------------------------
# Leg 1 -- expressivity coverage over the SEALED query set, occupancy run over REAL AV2.
# --------------------------------------------------------------------------------------------------
def _occupancy_runs_on_real_av2(query, scene) -> bool:
    """Does this occupancy query actually EVALUATE over a real AV2 scene without error? A real-data
    check that the occupancy backend can EXPRESS (denote a truth value for) the query, not just an
    assumed flag. We reuse the production evaluator (probe.retrieval.scene_matches)."""
    from probe.retrieval import scene_matches  # reused production evaluator
    try:
        scene_matches(scene, query, UnknownPolicy.FREE)
        return True
    except Exception:  # noqa: BLE001 - any failure = NOT expressible on real occupancy
        return False


def leg1_expressivity(queries, probe_scene) -> dict:
    """Per family: occupancy coverage (occupancy queries that RUN on real AV2) vs box-only coverage
    (refav_expressible flag). Returns per-family + overall + free-space-only rollups."""
    per_family: dict[str, dict] = {}
    for fam in _FAMILIES:
        fam_qs = [q for q in queries if _family_of(q.id) == fam]
        n = len(fam_qs)
        # box-only: the verified RefAV-expressibility flag (structural, from queries.yaml)
        box_n = sum(1 for q in fam_qs if q.refav_expressible)
        # occupancy: how many occupancy-backend queries actually evaluate on real AV2
        occ_qs = [q for q in fam_qs if q.is_occupancy]
        occ_runs = sum(1 for q in occ_qs if _occupancy_runs_on_real_av2(q, probe_scene))
        # occupancy "expresses" a query iff it is an occupancy-backend query that runs on real data;
        # the 4 box_baseline queries are tracking-backend (not in the occupancy retrieval set).
        per_family[fam] = {
            "n_queries": n,
            "occupancy_expressible": occ_runs,
            "occupancy_coverage_pct": round(100.0 * occ_runs / n, 1) if n else 0.0,
            "box_only_expressible": box_n,
            "box_only_coverage_pct": round(100.0 * box_n / n, 1) if n else 0.0,
            "gap_pct": round(100.0 * (occ_runs - box_n) / n, 1) if n else 0.0,
            "query_ids": [q.id for q in fam_qs],
            "occupancy_ran_real_av2": [q.id for q in occ_qs
                                       if _occupancy_runs_on_real_av2(q, probe_scene)],
        }

    def _roll(fams) -> dict:
        n = sum(per_family[f]["n_queries"] for f in fams)
        occ = sum(per_family[f]["occupancy_expressible"] for f in fams)
        box = sum(per_family[f]["box_only_expressible"] for f in fams)
        return {
            "n_queries": n,
            "occupancy_expressible": occ, "occupancy_coverage_pct": round(100.0 * occ / n, 1) if n else 0.0,
            "box_only_expressible": box, "box_only_coverage_pct": round(100.0 * box / n, 1) if n else 0.0,
            "gap_pct": round(100.0 * (occ - box) / n, 1) if n else 0.0,
        }

    overall = _roll(list(_FAMILIES))
    freespace = _roll(_FREESPACE_FAMILIES)
    # KILL check: H1 falsified iff box-only can express ANY free-space-family query.
    h1_falsified = freespace["box_only_expressible"] > 0
    return {
        "per_family": per_family,
        "overall": overall,
        "free_space_families_only": {**freespace, "families": list(_FREESPACE_FAMILIES)},
        "probe_scene": probe_scene.name,
        "kill_H1_falsified_box_expresses_freespace": h1_falsified,
    }


# --------------------------------------------------------------------------------------------------
# Leg 2 -- FP-side denotation vs the traversal-FREE oracle (reuse oracle_traversal ribbon).
# --------------------------------------------------------------------------------------------------
def _heldout_logs(limit: int) -> list[str]:
    d = json.loads(_HELDOUT.read_text())
    return list(d["logs"])[:limit]


def leg2_fp_denotation(window: int, far: float, limit: int, rng) -> dict:
    """On the driven-free ribbon (= traversal-FREE truth), does occupancy's free-space query
    predicate denote FREE/passable? Reuses oracle_traversal's ribbon EXACTLY (ego future sweep,
    `_rect_mask`, far, window, _MIN_SWEEP_CELLS gating).

    Two graded denotations on the SAME usable driven frames:
      (a) PREDICATE-level: free_along_ego_path(.) should be True (path passable) over each query
          horizon; min_free_width_along_path(.) should be > 0 (corridor open, not pinched shut).
          Agreement = fraction of driven frames where the predicate denotes FREE.
      (b) VOXEL-level ribbon FP: occupied voxels inside the provably-free swept ribbon (the exact
          oracle_traversal FP estimand), reported here per the query predicates' substrate.
    Box-only: INAPPLICABLE (no free-space denotation).
    """
    names = _heldout_logs(limit)
    XX, YY = _bev_centers()

    # frame-level predicate denotation accumulators
    fap_free = {h: 0 for h in _FREEPATH_HORIZONS}   # free_along_ego_path == True (FREE/passable)
    fap_total = {h: 0 for h in _FREEPATH_HORIZONS}
    mfw_open = {h: 0 for h in _FREEPATH_HORIZONS}    # min_free_width > 0 (corridor not pinched shut)
    mfw_total = {h: 0 for h in _FREEPATH_HORIZONS}
    # voxel-level ribbon FP (the oracle estimand) per frame, for the bootstrap CI
    ribbon_fp_rows: list[dict] = []
    n_usable = 0

    print(f"Leg 2: traversal-FREE denotation (W={window}, far={far}) over {len(names)} held-out "
          f"free-driving logs ...", flush=True)
    for li, name in enumerate(names):
        log_dir = _AV2 / name
        if not (log_dir / "city_SE3_egovehicle.feather").exists():
            continue
        all_sweeps = sorted(int(p.stem) for p in (log_dir / "sensors" / "lidar").glob("*.feather"))
        idx_of = {ts: i for i, ts in enumerate(all_sweeps)}
        poses = _world_poses(log_dir)
        # held-out free-driving = ALL driven frames (matches oracle_traversal held-out branch).
        scene = av2_sensor.load_scene(name, _AV2, timestamps=all_sweeps)
        dsorted = list(all_sweeps)
        used = 0
        for fi, ts in enumerate(dsorted):
            if fi >= len(scene.frames):
                break
            i = idx_of.get(ts)
            if i is None or i + window >= len(all_sweeps):
                continue
            fr = scene.frames[fi]
            ego = fr.ego
            txi, tyi, yawi = _pose_at(poses, ts)
            ci, si = math.cos(yawi), math.sin(yawi)
            sweep = np.zeros((_NX, _NY), dtype=bool)
            for k in range(1, window + 1):
                tsk = all_sweeps[i + k]
                txk, tyk, yawk = _pose_at(poses, tsk)
                dX, dY = txk - txi, tyk - tyi
                fwd = dX * ci + dY * si
                lat = -dX * si + dY * ci
                sweep |= _rect_mask(XX, YY, fwd, lat, yawk - yawi, ego.length, ego.width)
            sweep &= XX > far  # exploratory far-zone restriction (RELIABLE config: far=3.9)
            den = int(sweep.sum())
            if den < _MIN_SWEEP_CELLS:
                continue  # ego barely moved -> estimand undefined
            n_usable += 1
            used += 1

            # (a) frame-level predicate denotation over the query horizons
            for h in _FREEPATH_HORIZONS:
                fap = free_along_ego_path(fr.grid, ego, h, unknown_policy=UnknownPolicy.FREE,
                                          min_cluster_voxels=2)
                fap_total[h] += 1
                fap_free[h] += int(bool(fap))  # True = denotes FREE/passable (agrees with driven-FREE)
                mfw = min_free_width_along_path(fr.grid, ego, h, unknown_policy=UnknownPolicy.FREE)
                mfw_total[h] += 1
                mfw_open[h] += int(mfw > 0.0)   # >0 = corridor open (not pinched shut == 0.0)

            # (b) voxel-level ribbon FP (the exact oracle estimand on this frame)
            centers = fr.grid.obstacle_centers(max_height_agl=ego.height)
            occ_bev = np.zeros((_NX, _NY), dtype=bool)
            if len(centers):
                bi = np.clip(np.round((centers[:, 0] - _OX) / _VOX).astype(int), 0, _NX - 1)
                bj = np.clip(np.round((centers[:, 1] - _OY) / _VOX).astype(int), 0, _NY - 1)
                occ_bev[bi, bj] = True
            ribbon_fp_rows.append({"scene": name,
                                   "fp": float((sweep & occ_bev).sum()) / den})
        print(f"  {li + 1}/{len(names)} {name[:12]} -> {used} usable frames "
              f"({n_usable} total)", flush=True)

    if n_usable == 0:
        raise SystemExit("Leg 2: no usable frames (ego never moved enough?).")

    # frame-level agreement rate: occupancy denotes FREE on the driven-FREE ribbon
    predicate_agreement = {}
    for h in _FREEPATH_HORIZONS:
        predicate_agreement[f"free_along_ego_path@{h}s"] = {
            "denotes_free": fap_free[h], "n_frames": fap_total[h],
            "agreement_rate": round(fap_free[h] / fap_total[h], 4) if fap_total[h] else None,
        }
        predicate_agreement[f"min_free_width_open@{h}s"] = {
            "open_gt0": mfw_open[h], "n_frames": mfw_total[h],
            "agreement_rate": round(mfw_open[h] / mfw_total[h], 4) if mfw_total[h] else None,
        }
    # headline single agreement: pooled over all (frame, horizon) free_along_ego_path evaluations
    tot = sum(fap_total.values())
    free = sum(fap_free.values())
    headline_agreement = round(free / tot, 4) if tot else None

    # voxel-level ribbon FP CI (the oracle estimand, bootstrap by log)
    scenes = [r["scene"] for r in ribbon_fp_rows]
    fp_boot = _boot_mean([r["fp"] for r in ribbon_fp_rows], scenes, rng)

    # KILL check: Leg-2 fails iff occupancy false-blocks the driven path (agreement well below 1).
    leg2_fails = headline_agreement is not None and headline_agreement < 0.95

    return {
        "substrate": (f"held-out free-driving logs ({_HELDOUT.name}), ego in-path swept ribbon, "
                      f"far={far}, window={window} frames"),
        "n_logs": len({*scenes}),
        "n_usable_frames": n_usable,
        "predicate_denotation_vs_traversal_FREE": predicate_agreement,
        "headline_free_path_agreement": headline_agreement,
        "headline_free_path_evals": {"denotes_free": free, "n": tot},
        "voxel_ribbon_fp": {
            "true_fp_mean": fp_boot["mean"],
            "true_fp_ci": [fp_boot["lo"], fp_boot["hi"]],
            "note": ("occupancy false-positive rate inside the provably-FREE driven ribbon = the "
                     "oracle_traversal estimand on the query predicates' substrate; 0 => occupancy "
                     "never places an obstacle in space the ego physically traversed."),
        },
        "box_only_free_space_denotation": "INAPPLICABLE",
        "box_only_note": ("box-only (RefAV cuboid+map) has NO free-space primitive, so it cannot "
                          "produce a free-space denotation to grade -- INAPPLICABLE, not 0."),
        "kill_Leg2_fails_occupancy_false_blocks_driven_path": leg2_fails,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)   # RELIABLE oracle config
    ap.add_argument("--far", type=float, default=3.9)   # RELIABLE oracle config (beyond ego self-return)
    ap.add_argument("--limit", type=int, default=8)     # the 8 held-out free-driving logs
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    queries = load_queries(_HERE / "queries.yaml")

    # Leg 1 needs ONE real AV2 scene to prove the occupancy queries actually evaluate on real data.
    probe_name = _heldout_logs(1)[0]
    print(f"Leg 1: loading probe scene {probe_name[:12]} for real-AV2 expressivity check ...",
          flush=True)
    probe_scene = av2_sensor.load_scene(probe_name, _AV2, timestamps=None)
    leg1 = leg1_expressivity(queries, probe_scene)

    leg2 = leg2_fp_denotation(args.window, args.far, args.limit, rng)

    fs = leg1["free_space_families_only"]
    verdict = {
        "expressivity_dominance": (not leg1["kill_H1_falsified_box_expresses_freespace"]
                                   and fs["occupancy_coverage_pct"] > fs["box_only_coverage_pct"]),
        "fp_side_denotation_correct": (not leg2["kill_Leg2_fails_occupancy_false_blocks_driven_path"]),
        "both_sided_20F1_gap": "NOT CLAIMED (GPU-gated: BLOCKED-side unboxed-obstacle recall "
                               "empirically closed solo/CPU this session, per the pre-reg honest bound)",
    }

    report = {
        "experiment": "occquery_v0 / H3b",
        "preregistration": "h3b_expressivity_preregistration.md (SEALED, commit a47b500)",
        "result_class": "real-AV2 (solo/CPU, non-circular for both legs)",
        "commit": _git_commit(),
        "seed": 0,
        "data_root": str(_AV2),
        "leg1_expressivity": leg1,
        "leg2_fp_denotation": leg2,
        "verdict": verdict,
        "honest_bound": ("Full both-sided >=20-F1 denotation gap needs BLOCKED-side truth (unboxed-"
                         "obstacle recall) = a cross-modal oracle empirically CLOSED solo/CPU this "
                         "session (stereo AUC 0.259 / DAv2 scale >9 m / free-driving vacuity), GPU-"
                         "gated. NOT claimed here. Claimed: expressivity dominance (Leg 1) + FP-side "
                         "denotation correctness with box-only inapplicability (Leg 2)."),
    }

    out = _HERE / "results" / "h3b_expressivity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    # console summary
    print("\n=== H3b RESULT (real AV2, solo/CPU) ===")
    print("Leg 1 -- expressivity coverage (occupancy run on real AV2 vs box-only refav flag):")
    for fam, r in leg1["per_family"].items():
        print(f"  {fam:12s}: occupancy {r['occupancy_coverage_pct']:5.1f}%  "
              f"box-only {r['box_only_coverage_pct']:5.1f}%  gap {r['gap_pct']:+5.1f}%  "
              f"(n={r['n_queries']})")
    print(f"  {'OVERALL':12s}: occupancy {leg1['overall']['occupancy_coverage_pct']:5.1f}%  "
          f"box-only {leg1['overall']['box_only_coverage_pct']:5.1f}%  "
          f"gap {leg1['overall']['gap_pct']:+5.1f}%  (n={leg1['overall']['n_queries']})")
    print(f"  {'FREE-SPACE':12s}: occupancy {fs['occupancy_coverage_pct']:5.1f}%  "
          f"box-only {fs['box_only_coverage_pct']:5.1f}%  gap {fs['gap_pct']:+5.1f}%  "
          f"(n={fs['n_queries']})")
    print(f"  KILL H1-falsified (box expresses free-space)? "
          f"{leg1['kill_H1_falsified_box_expresses_freespace']}")
    print("\nLeg 2 -- FP-side denotation vs traversal-FREE truth (driven ribbon):")
    print(f"  n_logs={leg2['n_logs']}  n_usable_frames={leg2['n_usable_frames']}")
    print(f"  headline free_along_ego_path agreement (denotes FREE on driven path): "
          f"{leg2['headline_free_path_agreement']}  "
          f"({leg2['headline_free_path_evals']['denotes_free']}/"
          f"{leg2['headline_free_path_evals']['n']})")
    for k, v in leg2["predicate_denotation_vs_traversal_FREE"].items():
        print(f"    {k:28s}: agreement={v['agreement_rate']}  (n={v['n_frames']})")
    vfp = leg2["voxel_ribbon_fp"]
    print(f"  voxel ribbon FP (oracle estimand): {vfp['true_fp_mean']:.4f} "
          f"CI[{vfp['true_fp_ci'][0]:.4f},{vfp['true_fp_ci'][1]:.4f}]")
    print(f"  box-only free-space denotation: {leg2['box_only_free_space_denotation']}")
    print(f"  KILL Leg2-fails (occupancy false-blocks driven path)? "
          f"{leg2['kill_Leg2_fails_occupancy_false_blocks_driven_path']}")
    print("\nVerdict:")
    print(f"  expressivity dominance: {verdict['expressivity_dominance']}")
    print(f"  FP-side denotation correct: {verdict['fp_side_denotation_correct']}")
    print(f"  both-sided >=20-F1 gap: {verdict['both_sided_20F1_gap']}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
