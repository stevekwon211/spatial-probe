# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Oracle-v2 -- GT-box RECALL oracle for occupancy denotation-COMPLETENESS.

Design sealed in oracle_box_recall_preregistration.md BEFORE this run; this module realizes that
pre-registration literally (no redesign). A human-annotated tracked box with num_interior_pts >= N is a
location where the SAME LiDAR sweep physically returned >=N points on a real object. If the voxelization
marks that object's above-ground/in-range/non-ego sub-volume FREE, the pipeline lost returns it provably
had -- an internal-completeness MISS. Estimand = the rate of such misses; the load-bearing claim is the
RELATIVE gap between miss-rate at real boxes and at size/range-matched random on-road relocations.

SCOPE CEILING (repo CLAUDE.md H3 demotion): gating on num_interior_pts ties this to the SAME LiDAR
modality as the voxelizer -> a same-modality internal-consistency check of the voxelization/threshold/
filter chain, NOT external truth. Earns provenance + algorithm independence only.

Run:
    python experiments/occquery_v0/oracle_box_recall.py --self-check          # geometry; no confirmatory data
    python experiments/occquery_v0/oracle_box_recall.py --logs ALL --n-interior-min 5 --min-boxes 3 \
        --range-bin-m 8 --null on-road-matched --n-curve 1,3,5,10,20 --seed 0 \
        --out experiments/occquery_v0/results/oracle_box_recall.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "experiments" / "dynfield_v0"))

from harness_v2 import _boot_mean  # noqa: E402  scene-clustered bootstrap, reused verbatim
from oracle_traversal import _bev_centers, _rect_mask  # noqa: E402  BEV footprint, reused verbatim
from probe.adapters import av2_sensor  # noqa: E402
from probe.grid import OCCUPIED  # noqa: E402

_AV2 = _HERE.parents[1] / "data" / "danger" / "av2_sensor"
_ROAD_Z = av2_sensor._ROAD_Z          # 0.3 -- floor of the above-road slab (== _voxelize)
_VOX = av2_sensor.VOXEL_SIZE          # 0.4
_NX, _NY, _NZ = av2_sensor.GRID_SHAPE  # 200, 200, 16
(_X0, _X1), (_Y0, _Y1), (_Z0, _Z1) = av2_sensor.RANGE  # ((-40,40),(-40,40),(-1,5.4))
# voxel-z centers (axis2 = up); admissibility keeps centers strictly above _ROAD_Z and below _Z1.
_ZC = _Z0 + (np.arange(_NZ) + 0.5) * _VOX
# tall/large classes (z=5.4 cap straddle, confound C3) -> reported as a separate stratum
_TALL = ("BUS", "LARGE_VEHICLE", "TRUCK", "ARTICULATED_BUS", "SCHOOL_BUS", "TRUCK_CAB")


def _annotations(log_dir: pathlib.Path) -> dict:
    """Read annotations.feather directly (pyarrow; venv has no pandas) -> column dict of numpy arrays.
    SPEC: oracle reads num_interior_pts (+ pose/size/quat/category/timestamp) DIRECTLY, not via TrackedBox."""
    t = av2_sensor._read_feather(log_dir / "annotations.feather").to_pydict()
    return {
        "ts": np.asarray(t["timestamp_ns"], dtype=np.int64),
        "tx": np.asarray(t["tx_m"], float), "ty": np.asarray(t["ty_m"], float), "tz": np.asarray(t["tz_m"], float),
        "L": np.asarray(t["length_m"], float), "W": np.asarray(t["width_m"], float), "H": np.asarray(t["height_m"], float),
        "qw": np.asarray(t["qw"], float), "qx": np.asarray(t["qx"], float),
        "qy": np.asarray(t["qy"], float), "qz": np.asarray(t["qz"], float),
        "cat": np.asarray(t["category"]), "pts": np.asarray(t["num_interior_pts"], dtype=np.int64),
    }


def _admissible_column(XX, YY, cx, cy, yaw, length, width) -> np.ndarray:
    """BEV oriented-rect footprint (NX,NY) intersected with in-range + non-ego columns (the column-wise
    part of _voxelize admissibility). SPEC-NOTE: _voxelize clips out-of-range points INTO the edge voxel
    via np.clip rather than dropping them, but its mask `m` first requires x in [x0,x1) & y in [y0,y1);
    a BEV center grid built from ORIGIN already lies inside range, and the rect mask cannot select a
    column outside it -- so the in-range filter is a no-op here and the literal reading needs only the
    ego-cuboid removal. Ego cuboid uses _voxelize's exact bounds (open interval, |y|<half_w)."""
    foot = _rect_mask(XX, YY, cx, cy, yaw, length, width)
    ego = (XX > av2_sensor._EGO_X0) & (XX < av2_sensor._EGO_X1) & (np.abs(YY) < av2_sensor._EGO_HALF_W)
    inrange = (XX >= _X0) & (XX < _X1) & (YY >= _Y0) & (YY < _Y1)
    return foot & inrange & ~ego


def _slab_levels(tz: float, H: float) -> np.ndarray:
    """Voxel-z indices whose CENTER lies in [max(tz-H/2,_ROAD_Z), min(tz+H/2,_Z1)] -- the above-road slab.
    Admissibility (== _voxelize): center strictly above _ROAD_Z and strictly below _Z1."""
    lo = max(tz - H / 2.0, _ROAD_Z)
    hi = min(tz + H / 2.0, _Z1)
    return np.flatnonzero((_ZC >= lo) & (_ZC <= hi) & (_ZC > _ROAD_Z) & (_ZC < _Z1))


def _box_voxels(XX, YY, cx, cy, yaw, length, width, tz, H):
    """(col_i, col_j, z_levels): the admissible voxel set of one box footprint -- BEV columns x z-slab."""
    cols = _admissible_column(XX, YY, cx, cy, yaw, length, width)
    bi, bj = np.nonzero(cols)
    return bi, bj, _slab_levels(tz, H)


def _covered(occ, bi, bj, levels) -> int:
    """|box_voxels & (occ==OCCUPIED)|. Binary primary uses covered==0 (the box is ENTIRELY FREE)."""
    if len(bi) == 0 or len(levels) == 0:
        return 0
    sub = occ[bi[:, None], bj[:, None], levels[None, :]]  # (n_cols, n_levels)
    return int((sub == OCCUPIED).sum())


def _self_check() -> bool:
    """Geometry only, NO confirmatory data (pre-reg Self-check (a)/(b)/(c))."""
    XX, YY = _bev_centers()
    ok = True

    # (a) a synthetic box at a known ego location rasterizes to exactly the voxel _voxelize assigns for a
    #     point at its center -- 0-voxel round-trip error. Pick an admissible center (above road, in range,
    #     beyond the ego cuboid). Call _voxelize on that single point; the box rasterizer must hit it.
    cx, cy, cz = 12.0, -7.0, 1.5
    pocc = av2_sensor._voxelize(np.array([cx]), np.array([cy]), np.array([cz]))
    pidx = np.argwhere(pocc == OCCUPIED)
    assert len(pidx) == 1, "synthetic point did not light exactly one voxel"
    pi, pj, pk = (int(v) for v in pidx[0])
    bi, bj, levels = _box_voxels(XX, YY, cx, cy, 0.0, 1.6, 1.6, cz, 1.6)
    hit = (pi in bi.tolist()) and (pj in bj.tolist()) and (pk in levels.tolist())
    # confirm the box column that matches the point is exactly the point's BEV column (0-voxel error)
    col_hit = any(i == pi and j == pj for i, j in zip(bi.tolist(), bj.tolist()))
    pass_a = bool(hit and col_hit)
    print(f"  (a) box-rasterizer hits the _voxelize voxel of its center {(pi, pj, pk)}: "
          f"{'PASS' if pass_a else 'FAIL'}  (0-voxel round-trip)")
    ok &= pass_a

    # (b) a box centered in the ego cuboid -> 0 admissible voxels.
    ecx = (av2_sensor._EGO_X0 + av2_sensor._EGO_X1) / 2.0  # 1.4, inside the ego cuboid
    bi, bj, levels = _box_voxels(XX, YY, ecx, 0.0, 0.0, 1.0, 1.0, 1.5, 1.6)
    pass_b = (len(bi) == 0)
    print(f"  (b) box centered in ego cuboid -> {len(bi)} admissible columns: {'PASS' if pass_b else 'FAIL'}")
    ok &= pass_b

    # (c) a box entirely below _ROAD_Z -> 0 admissible voxels (z-slab empty).
    bi, bj, levels = _box_voxels(XX, YY, 15.0, 5.0, 0.0, 2.0, 2.0, tz=-0.5, H=0.4)  # top = -0.3 < _ROAD_Z
    pass_c = (len(levels) == 0)
    print(f"  (c) box entirely below _ROAD_Z -> {len(levels)} admissible z-levels: {'PASS' if pass_c else 'FAIL'}")
    ok &= pass_c

    print(f"SELF-CHECK: {'PASS' if ok else 'FAIL'}")
    return ok


def _verdict(true_b, null_b, null_mean) -> str:
    """Falsifiable kill (pre-reg). Null-reachability pre-condition gates first."""
    if not (0.3 < null_mean < 0.97):
        return "INDETERMINATE-BY-NULL"
    if not (true_b["defined"] and null_b["defined"]):
        return "INDETERMINATE"
    if true_b["hi"] < null_b["lo"]:
        return "RECALL-SUPPORTED"
    if true_b["lo"] >= null_b["lo"]:
        return "FAIL"
    return "INDETERMINATE"


def _collect(names, n_min, min_boxes, range_bin_m, rng, n_curve):
    """One pass per log: build per-box miss rows for the true arm + the region-local matched null arm,
    aggregated to per-frame means. Returns the row dicts the bootstrap consumes plus the strata tallies."""
    XX, YY = _bev_centers()
    true_rows, null_rows = [], []                       # {scene, val} per usable frame (mean miss over boxes)
    curve = {n: {"true": [], "null": []} for n in n_curve}  # per-frame means at each N (scene-tagged)
    strata = {"pts0": [0, 0], "pts1to4": [0, 0],        # [miss_sum, count] for the descriptive strata
              "height_below_road": [0, 0], "height_above_road": [0, 0]}
    cls_strata: dict[str, list] = {}
    dropped_logs = []

    for li, name in enumerate(names):
        log_dir = _AV2 / name
        if not (log_dir / "annotations.feather").exists():
            continue
        ann = _annotations(log_dir)
        # C1 guard (oracle-validity falsifier #5): ann_ts must be a subset of the LiDAR sweeps.
        sweeps_sorted = sorted(int(p.stem) for p in (log_dir / "sensors" / "lidar").glob("*.feather"))
        if not set(int(t) for t in np.unique(ann["ts"])).issubset(sweeps_sorted):
            dropped_logs.append(name)
            continue
        scene = av2_sensor.load_scene(name, _AV2, with_boxes=False)
        # Key occupancy by the EXACT integer sweep timestamp (load_scene builds frames in sorted-sweep
        # order). NOT via fr.time: ts/1e9*1e9 round-trips through float64 and ns > 2^53 lose precision,
        # which would silently miss the int-exact box timestamps (the same class of bug as the labeler).
        occ_by_ts = {ts: fr.grid.occupancy for ts, fr in zip(sweeps_sorted, scene.frames)}

        # empirical on-road support: union of BEV columns covered by any INCLUDED (pts>=primary N) box.
        # SPEC-NOTE: "any included box across the log" -- "included" = the primary inclusion set (pts>=n_min);
        # this is the substrate the null relocates onto. Built per log before the null draws.
        primary_idx = np.flatnonzero(ann["pts"] >= n_min)
        support = np.zeros((_NX, _NY), dtype=bool)
        for i in primary_idx:
            yaw = av2_sensor._quat_yaw(ann["qw"][i], ann["qx"][i], ann["qy"][i], ann["qz"][i])
            support |= _admissible_column(XX, YY, ann["tx"][i], ann["ty"][i], yaw, ann["L"][i], ann["W"][i])
        sup_i, sup_j = np.nonzero(support)
        sup_x = _bev_x(sup_i)  # forward (x) coord of each support column center

        used = 0
        for ts in np.unique(ann["ts"]):
            occ = occ_by_ts.get(int(ts))
            if occ is None:
                continue
            fi = np.flatnonzero(ann["ts"] == ts)
            frame_true, frame_null = [], []
            per_n = {n: [] for n in n_curve}
            for i in fi:
                pts = int(ann["pts"][i])
                yaw = av2_sensor._quat_yaw(ann["qw"][i], ann["qx"][i], ann["qy"][i], ann["qz"][i])
                bi, bj, levels = _box_voxels(XX, YY, ann["tx"][i], ann["ty"][i], yaw,
                                             ann["L"][i], ann["W"][i], ann["tz"][i], ann["H"][i])
                miss = 1.0 if _covered(occ, bi, bj, levels) == 0 else 0.0
                cat = str(ann["cat"][i])

                # descriptive strata (recorded regardless of inclusion)
                _tally(strata, "pts0" if pts == 0 else ("pts1to4" if pts < n_min else None), miss)
                bottom = float(ann["tz"][i]) - float(ann["H"][i]) / 2.0
                _tally(strata, "height_below_road" if bottom < _ROAD_Z else "height_above_road", miss)
                if pts >= n_min:
                    key = "tall" if cat.startswith(_TALL) else "other"
                    cls_strata.setdefault(key, [0, 0])
                    cls_strata[key][0] += miss; cls_strata[key][1] += 1

                # N-curve inclusion: pts>=n for each n in the curve
                for n in n_curve:
                    if pts >= n:
                        per_n[n].append(miss)
                # primary arm + matched null arm: only on the sealed primary inclusion set (pts>=n_min)
                if pts >= n_min:
                    frame_true.append(miss)
                    null_miss = _null_box(occ, XX, YY, ann, i, yaw, sup_i, sup_j, sup_x, range_bin_m, rng)
                    if null_miss is not None:
                        frame_null.append(null_miss)

            # per-frame aggregation: drop frames with < min_boxes included boxes (primary arm)
            if len(frame_true) >= min_boxes:
                true_rows.append({"scene": name, "val": float(np.mean(frame_true))})
                if frame_null:
                    null_rows.append({"scene": name, "val": float(np.mean(frame_null))})
                used += 1
            # N-curve per-frame means (same >=min_boxes frame gate per N)
            for n in n_curve:
                if len(per_n[n]) >= min_boxes:
                    curve[n]["true"].append({"scene": name, "val": float(np.mean(per_n[n]))})
        print(f"  {li + 1}/{len(names)} {name[:12]} -> {used} usable frames "
              f"({len(true_rows)} true rows)", flush=True)

    return true_rows, null_rows, curve, strata, cls_strata, dropped_logs


def _bev_x(i):
    """forward-x center of BEV column index i (axis0)."""
    return _X0 + (np.asarray(i) + 0.5) * _VOX


def _null_box(occ, XX, YY, ann, i, yaw, sup_i, sup_j, sup_x, range_bin_m, rng):
    """Region-local matched null: relocate the box footprint (keep length/width/yaw/vertical-slab) to a
    uniformly random admissible support column within +-1 range-bin (range_bin_m) forward-x of the real
    box. Returns null miss (0/1) or None if the matched region is empty."""
    if len(sup_i) == 0:
        return None
    x_real = float(ann["tx"][i])
    in_bin = np.flatnonzero(np.abs(sup_x - x_real) <= range_bin_m)
    if len(in_bin) == 0:
        return None
    pick = in_bin[rng.integers(len(in_bin))]
    ncx, ncy = float(_bev_x(sup_i[pick])), float(_Y0 + (sup_j[pick] + 0.5) * _VOX)
    bi, bj, levels = _box_voxels(XX, YY, ncx, ncy, yaw, ann["L"][i], ann["W"][i], ann["tz"][i], ann["H"][i])
    return 1.0 if _covered(occ, bi, bj, levels) == 0 else 0.0


def _tally(strata, key, miss):
    if key is not None:
        strata[key][0] += miss
        strata[key][1] += 1


def _rate(pair):
    return (pair[0] / pair[1]) if pair[1] else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true", help="run geometry self-check only (no data)")
    ap.add_argument("--logs", default="ALL", help="'ALL' = every log dir with annotations + lidar")
    ap.add_argument("--n-interior-min", type=int, default=5, help="primary num_interior_pts gate (N=5)")
    ap.add_argument("--min-boxes", type=int, default=3, help="drop frames with < this many included boxes")
    ap.add_argument("--range-bin-m", type=float, default=8.0, help="+-1 range-bin for the matched null")
    ap.add_argument("--null", default="on-road-matched", choices=["on-road-matched"])
    ap.add_argument("--n-curve", default="1,3,5,10,20", help="comma N-curve for the inclusion gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    if args.self_check:
        sys.exit(0 if _self_check() else 1)

    rng = np.random.default_rng(args.seed)
    n_curve = [int(x) for x in args.n_curve.split(",")]
    if args.logs == "ALL":
        names = sorted(p.name for p in _AV2.iterdir()
                       if (p / "annotations.feather").exists() and (p / "sensors" / "lidar").exists())
    else:
        names = [s.strip() for s in args.logs.split(",") if s.strip()]

    print(f"oracle-v2 box-recall over {len(names)} logs (N>={args.n_interior_min}, "
          f"min_boxes={args.min_boxes}, range_bin={args.range_bin_m}m) ...", flush=True)
    true_rows, null_rows, curve, strata, cls_strata, dropped = _collect(
        names, args.n_interior_min, args.min_boxes, args.range_bin_m, rng, n_curve)

    if not true_rows:
        sys.exit("no usable frames (no log had >= min_boxes included boxes).")

    true_b = _boot_mean([r["val"] for r in true_rows], [r["scene"] for r in true_rows], rng)
    null_b = _boot_mean([r["val"] for r in null_rows], [r["scene"] for r in null_rows], rng) if null_rows \
        else {"defined": False, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    null_mean = null_b["mean"]
    verdict = _verdict(true_b, null_b, null_mean)

    # N-curve bootstrapped true miss-rate (the curve, not a movable cutoff)
    curve_out = {}
    for n in n_curve:
        rows = curve[n]["true"]
        b = _boot_mean([r["val"] for r in rows], [r["scene"] for r in rows], rng) if rows \
            else {"defined": False, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        curve_out[str(n)] = {"true_miss_mean": b["mean"], "true_miss_ci": [b["lo"], b["hi"]],
                             "n_frames": b["n"], "defined": b["defined"]}
    # sealed sanity check: miss-rate must DECREASE monotonically with N (confound C4 / falsifier #2)
    curve_means = [curve_out[str(n)]["true_miss_mean"] for n in n_curve]
    monotone = all(a >= b or math.isnan(a) or math.isnan(b)
                   for a, b in zip(curve_means, curve_means[1:]))

    report = {
        "estimand": "binary per-box occupancy MISS (covered==0) rate over num_interior_pts>=N boxes; "
                    "relative gap vs a region-local size/range-matched on-road null.",
        "n_logs": len({r["scene"] for r in true_rows}), "n_frames_true": len(true_rows),
        "n_frames_null": len(null_rows), "dropped_logs_c1": dropped,
        "n_interior_min": args.n_interior_min, "min_boxes": args.min_boxes,
        "range_bin_m": args.range_bin_m, "null": args.null, "seed": args.seed,
        "true_miss_mean": true_b["mean"], "true_miss_ci": [true_b["lo"], true_b["hi"]],
        "null_miss_mean": null_b["mean"], "null_miss_ci": [null_b["lo"], null_b["hi"]],
        "null_reachability_ok": bool(0.3 < null_mean < 0.97) if not math.isnan(null_mean) else False,
        "verdict": verdict,
        "n_curve": curve_out, "n_curve_monotone_decreasing": bool(monotone),
        "strata": {
            "pts0_sensor_blind_miss": _rate(strata["pts0"]), "pts0_n": strata["pts0"][1],
            "pts1to4_sparse_miss": _rate(strata["pts1to4"]), "pts1to4_n": strata["pts1to4"][1],
            "height_below_road_miss": _rate(strata["height_below_road"]),
            "height_below_road_n": strata["height_below_road"][1],
            "height_above_road_miss": _rate(strata["height_above_road"]),
            "height_above_road_n": strata["height_above_road"][1],
        },
        "class_strata": {k: {"miss_rate": _rate(v), "n": v[1]} for k, v in cls_strata.items()},
        "framing": "RECALL, one-sided: real LiDAR-seen obstacles occupancy marks FREE, complement of the "
                   "PASSED FP oracle. SAME-MODALITY internal-consistency check (provenance + algorithm "
                   "independence only, NOT external truth). true_miss is an UPPER BOUND (C2/C4/C6 inflate "
                   "it) -> RECALL-SUPPORTED is conservative. Not a verified recall P/R/F1.",
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\noracle-v2 box-recall ({report['n_logs']} logs, {len(true_rows)} usable frames):")
    print(f"  true occupancy MISS rate at real boxes (pts>={args.n_interior_min}): "
          f"{true_b['mean']:.4f} CI[{true_b['lo']:.4f},{true_b['hi']:.4f}]")
    print(f"  region-local matched on-road null:                  "
          f"{null_b['mean']:.4f} CI[{null_b['lo']:.4f},{null_b['hi']:.4f}]")
    print(f"  N-curve (true miss): " + ", ".join(f"N{n}={curve_out[str(n)]['true_miss_mean']:.3f}" for n in n_curve)
          + f"  monotone_decreasing={monotone}")
    print(f"  VERDICT: {verdict}")
    if args.out:
        print(f"  wrote {args.out}")
    print("  scope: SAME-MODALITY internal-consistency check; true_miss is an UPPER BOUND; not a P/R/F1.")


if __name__ == "__main__":
    main()
