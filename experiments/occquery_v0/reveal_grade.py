# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""occquery H3 future-reveal grading harness -- per future_reveal_preregistration.md (design b).

Grades the occupancy reachable-free predicate (run on the single-frame-t OBSERVED view, extrapolating
into UNOBSERVED space) against the INDEPENDENT raw t+k LiDAR carve (reveal_oracle.py), on the revealed
voxels (unobserved@t, directly observed@t+k, near, static). Reports the pre-registered statistic: the
GAP F1(occ) - F1(box-only) with OCCUPIED (box-less structure) as the positive class, scene-clustered
bootstrap 95% CI, over k. Everything is 2D BEV (the predicate's native plane); the 3D reveal/observed
grids are z-collapsed over the ego-height band the predicate itself uses (max_height_agl=ego.height),
so occ_pred and reveal_truth share the same projection. Sealed scope: 10 mini scenes with raw sweeps,
horizon=3.0 s (fixed a priori, reported), unknown->free primary.

Run: python experiments/occquery_v0/reveal_grade.py [--limit N]
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
sys.path.insert(0, str(_HERE))

from probe.adapters.occ3d import (GRID_SHAPE, GROUND_HEIGHT, ORIGIN, VOXEL_SIZE, _box_index,
                                  _ego_frame_boxes, _ordered_tokens, map_occupancy)
from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid, UnknownPolicy
from probe.predicates.reachable import reachable_free_field

_POLICY = {"free": UnknownPolicy.FREE, "occupied": UnknownPolicy.OCCUPIED, "ignored": UnknownPolicy.IGNORED}
from reveal_oracle import _index, reveal_truth_in_frame, sweep_points_ego

_DATA = _HERE.parents[1] / "data"
_NUSC = _DATA / "nuscenes" / "v1.0-trainval"
_ORIGIN = np.asarray(ORIGIN, float)
_NEAR = 15.0
_KS = (1, 3, 5)
_HORIZON = 3.0          # s, fixed a priori (standard planning look-ahead); reported, not tuned
_EGO_H = 1.9            # ego height -> z-collapse band, matches reachable's max_height_agl
_BAND = (_ORIGIN[2] + np.arange(GRID_SHAPE[2]) * VOXEL_SIZE) <= (GROUND_HEIGHT + _EGO_H)  # z in ego body slab


def _collapse(grid3d: np.ndarray, min_occ: int = 1) -> np.ndarray:
    """z-collapse a FREE/OCCUPIED/UNKNOWN grid over the ego body slab -> 2D BEV.

    min_occ = how many occupied voxels in the column count as an OCCUPIED cell (1 = any, the
    pre-registered default; >1 = a robustness variant that drops single-return noise)."""
    band = grid3d[:, :, _BAND]
    occ = (band == OCCUPIED).sum(axis=2) >= min_occ
    free = (band == FREE).any(axis=2)
    out = np.full(band.shape[:2], UNKNOWN, dtype=np.int8)
    out[free] = FREE
    out[occ] = OCCUPIED                      # occupied overrides free (something blocks the column)
    return out


def _occ_pred_bev(observed3d, speed, policy=UnknownPolicy.FREE) -> tuple[np.ndarray, np.ndarray]:
    """reachable-free predicate on the single-frame observed grid -> (occ_pred_OCCUPIED bev, in_window bev).

    occ_pred(cell) = OCCUPIED where in the predicate window and NOT reachable-free; in_window marks where
    the predicate makes any claim at all (out-of-window cells are excluded from grading)."""
    grid = OccupancyGrid(observed3d, VOXEL_SIZE, ORIGIN, GROUND_HEIGHT)
    ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=speed)
    rf = reachable_free_field(grid, ego, _HORIZON, unknown_policy=policy, min_cluster_voxels=2)
    nx, ny = GRID_SHAPE[0], GRID_SHAPE[1]
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    xw = _ORIGIN[0] + ii * VOXEL_SIZE       # ego forward (ego at origin, heading 0)
    yw = _ORIGIN[1] + jj * VOXEL_SIZE       # ego lateral
    fi = np.round((xw - rf.forward_min) / rf.resolution).astype(int)
    li = np.round((yw - rf.lateral_min) / rf.resolution).astype(int)
    nf, nl = rf.reachable.shape
    inwin = (fi >= 0) & (fi < nf) & (li >= 0) & (li < nl)
    free = np.zeros((nx, ny), bool)
    free[inwin] = rf.reachable[fi[inwin], li[inwin]]
    return inwin & ~free, inwin


def _box_pred_bev(boxes) -> np.ndarray:
    """box-only baseline: cell OCCUPIED iff inside any box footprint (BEV rotated rectangle)."""
    nx, ny = GRID_SHAPE[0], GRID_SHAPE[1]
    occ = np.zeros((nx, ny), bool)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    xw = _ORIGIN[0] + ii * VOXEL_SIZE
    yw = _ORIGIN[1] + jj * VOXEL_SIZE
    for b in boxes:
        cx, cy = b.center[0], b.center[1]
        l, w = b.size[0], b.size[1]
        c, s = math.cos(-b.yaw), math.sin(-b.yaw)
        dx, dy = xw - cx, yw - cy
        fwd = dx * c - dy * s
        lat = dx * s + dy * c
        occ |= (np.abs(fwd) <= l / 2) & (np.abs(lat) <= w / 2)
    return occ


def _f1(pred_occ: np.ndarray, truth_occ: np.ndarray) -> float:
    tp = int((pred_occ & truth_occ).sum())
    fp = int((pred_occ & ~truth_occ).sum())
    fn = int((~pred_occ & truth_occ).sum())
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--policy", choices=list(_POLICY), default="free")
    ap.add_argument("--collapse-min", type=int, default=1, help="occupied voxels/column for OCCUPIED (1=pre-reg any; 2=majority robustness)")
    args = ap.parse_args()
    policy = _POLICY[args.policy]
    scene_infos = json.loads((_DATA / "annotations.json").read_text())["scene_infos"]
    lidar, cs, ep = _index(_NUSC)
    box_index = _box_index(_NUSC)
    # only scenes whose frames all have a local raw sweep
    scenes = [n for n in sorted(scene_infos)
              if all(t in lidar and pathlib.Path(_DATA / lidar[t][0]).exists()
                     for t in _ordered_tokens(scene_infos[n]))][: args.limit]
    rng = np.random.default_rng(0)

    rows = {k: [] for k in _KS}             # per (scene, frame): (occ_occ[], box_occ[], reveal_occ[])
    print(f"grading future-reveal H3 on {len(scenes)} scenes (raw-sweep) ...", flush=True)
    near_x = np.abs(_ORIGIN[0] + np.arange(GRID_SHAPE[0]) * VOXEL_SIZE) <= _NEAR
    near_y = np.abs(_ORIGIN[1] + np.arange(GRID_SHAPE[1]) * VOXEL_SIZE) <= _NEAR
    near = near_x[:, None] & near_y[None, :]
    for s_i, name in enumerate(scenes):
        si = scene_infos[name]
        toks = _ordered_tokens(si)
        cache: dict[int, tuple] = {}
        def frame(i):
            if i not in cache:
                tok = toks[i]; fr = si[tok]
                lab = np.load(_DATA / fr["gt_path"])
                fn, cst, _ = lidar[tok]
                pts, o = sweep_points_ego(_DATA, fn, *cs[cst])
                R_w, t_w = ep[lidar[tok][2]]
                cache[i] = (lab["semantics"], lab["mask_lidar"], R_w, t_w, pts, o,
                            _ego_frame_boxes(tok, fr["ego_pose"], box_index), si[tok])
            return cache[i]
        for i in range(len(toks)):
            sem_t, ml_t, R_t, t_t, _, _, boxes_t, fr_t = frame(i)
            spd = _ego_speed(si, toks, i)
            observed_t = map_occupancy(sem_t, ml_t)            # single-frame observed (UNKNOWN where unseen)
            occ_occ_bev, inwin = _occ_pred_bev(observed_t, spd, policy)
            box_occ_bev = _box_pred_bev(boxes_t)
            obs_t_bev = (ml_t[:, :, _BAND] == 1).any(axis=2)   # observed@t (any z in band)
            dyn_bev = np.isin(sem_t[:, :, _BAND], np.arange(11)).any(axis=2)  # dynamic class present
            for k in _KS:
                j = i + k
                if j >= len(toks):
                    continue
                sem_b, ml_b, R_b, t_b, pts_b, o_b, _, _ = frame(j)
                reveal3d = reveal_truth_in_frame(pts_b, o_b, R_b, t_b, R_t, t_t)
                reveal_bev = _collapse(reveal3d, args.collapse_min)
                # pre-registered leak-channel-2 part 2: drop revealed cells whose t+k carve conflicts with
                # a t+k box of a DYNAMIC object that RE-ENTERED -- re-audit fix: filter to MOVING boxes
                # (speed > 0.5 m/s); the prior "any t+k box" form over-dropped PARKED cars (validly
                # gradeable static structure) and that over-drop was the sole maker of the apparent gap.
                moving = [b for b in _ego_frame_boxes(toks[j], fr_t["ego_pose"], box_index)
                          if not math.isnan(b.velocity[0]) and math.hypot(b.velocity[0], b.velocity[1]) > 0.5]
                reentry = (reveal_bev == OCCUPIED) & _box_pred_bev(moving)
                revealed = (reveal_bev != UNKNOWN) & ~obs_t_bev & near & ~dyn_bev & inwin & ~reentry
                if not revealed.any():
                    continue
                rows[k].append((occ_occ_bev[revealed], box_occ_bev[revealed],
                                reveal_bev[revealed] == OCCUPIED, name))
        if (s_i + 1) % 2 == 0:
            print(f"  {s_i + 1}/{len(scenes)} scenes", flush=True)

    report = {"scenes": len(scenes), "horizon_s": _HORIZON, "near_m": _NEAR, "unknown_policy": args.policy, "by_k": {}}
    print(f"\noccquery H3 future-reveal ({len(scenes)} scenes, horizon={_HORIZON}s, unknown->{args.policy}, OCCUPIED=positive):\n")
    print(f"  {'k':>3} {'cells':>8} {'occF1':>7} {'boxF1':>7} {'gap':>7}  {'gap 95% CI (scene-bootstrap)':>30}")
    for k in _KS:
        per = rows[k]
        if not per:
            continue
        occ = np.concatenate([r[0] for r in per]); box = np.concatenate([r[1] for r in per])
        tru = np.concatenate([r[2] for r in per]); sc = np.concatenate([[r[3]] * len(r[0]) for r in per])
        f_occ, f_box = _f1(occ, tru), _f1(box, tru)
        uniq = np.unique(sc)
        boot = []
        for _ in range(2000):
            drawn = rng.choice(uniq, len(uniq), replace=True)
            m = np.concatenate([np.where(sc == u)[0] for u in drawn])
            boot.append(_f1(occ[m], tru[m]) - _f1(box[m], tru[m]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        report["by_k"][k] = {"cells": int(len(occ)), "occ_f1": f_occ, "box_f1": f_box,
                             "gap": f_occ - f_box, "gap_ci": [float(lo), float(hi)],
                             "occ_pos_rate": float(tru.mean())}
        print(f"  {k:>3} {len(occ):>8} {f_occ:>7.3f} {f_box:>7.3f} {f_occ - f_box:>+7.3f}  [{lo:>+.3f}, {hi:>+.3f}]")
    (_HERE / "results").mkdir(exist_ok=True)
    out = _HERE / "results" / f"h3_future_reveal_{args.policy}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  wrote {out}")
    print("  gap CI lower-bound > 0 across k => H3 HOLDS (occupancy beats box-only at box-less structure,")
    print("  independently verified). CI includes 0 => FALSIFIED / under-power, report the negative.")


def _ego_speed(si, toks, i):
    a = si[toks[max(i - 1, 0)]]; b = si[toks[min(i + 1, len(toks) - 1)]]
    pa = np.asarray(a["ego_pose"]["translation"][:2], float); pb = np.asarray(b["ego_pose"]["translation"][:2], float)
    dt = abs(b["timestamp"] - a["timestamp"]) / 1e6
    return float(np.linalg.norm(pb - pa) / dt) if dt > 0 else 0.0


if __name__ == "__main__":
    main()
