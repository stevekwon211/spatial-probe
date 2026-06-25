# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Oracle-v0 -- ego-trajectory traversal oracle for occupancy denotation-correctness.

Design sealed in oracle_traversal_preregistration.md BEFORE this run. The ego's RECORDED future
trajectory is ground truth for "free" (a vehicle cannot drive through a real obstacle), so the ego's
future swept volume over W frames marks space that was physically FREE at frame t -- an oracle
independent of the occupancy perception in both data source (recorded poses, not LiDAR) and algorithm
(rigid-body sweep, not voxelization). Estimand: occupancy FALSE-POSITIVE rate inside that swept volume
vs a shuffled-occupancy null. Kill: true_fp not clearly below shuffled. One-sided (false positives in
the driven ribbon only; recall is oracle-v1).

Run: python experiments/occquery_v0/oracle_traversal.py [--window 5] [--limit 18]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "experiments" / "dynfield_v0"))

import pyarrow as pa

from harness_v2 import _boot_mean
from probe.adapters import av2_sensor
from probe.grid import OCCUPIED

_AV2 = _HERE.parents[1] / "data" / "danger" / "av2_sensor"
_LOGS = _HERE.parents[1] / "experiments" / "dynfield_v0" / "av2_danger_logs.json"
_VOX = av2_sensor.VOXEL_SIZE
_OX, _OY = av2_sensor.ORIGIN[0], av2_sensor.ORIGIN[1]
_NX, _NY = av2_sensor.GRID_SHAPE[0], av2_sensor.GRID_SHAPE[1]
_MIN_SWEEP_CELLS = 5  # estimand undefined when the ego barely moves (stopped) -> skip


def _quat_yaw(qw, qx, qy, qz):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _world_poses(log_dir: pathlib.Path):
    """timestamp_ns(sorted) -> world (tx, ty, yaw) from city_SE3_egovehicle.feather."""
    t = pa.ipc.open_file(pa.memory_map(str(log_dir / "city_SE3_egovehicle.feather"), "r")).read_all()
    ts = np.asarray(t.column("timestamp_ns").to_pylist(), dtype=np.int64)
    tx = np.asarray(t.column("tx_m").to_pylist(), float)
    ty = np.asarray(t.column("ty_m").to_pylist(), float)
    yaw = np.array([_quat_yaw(float(t.column("qw")[i].as_py()), float(t.column("qx")[i].as_py()),
                              float(t.column("qy")[i].as_py()), float(t.column("qz")[i].as_py()))
                    for i in range(len(ts))])
    o = np.argsort(ts)
    return ts[o], tx[o], ty[o], yaw[o]


def _pose_at(poses, ts: int):
    pts, tx, ty, yaw = poses
    k = int(np.argmin(np.abs(pts - ts)))  # nearest recorded pose (ego poses are dense)
    return tx[k], ty[k], yaw[k]


def _bev_centers():
    xs = _OX + np.arange(_NX) * _VOX
    ys = _OY + np.arange(_NY) * _VOX
    return np.meshgrid(xs, ys, indexing="ij")  # XX, YY  (NX, NY)


def _rect_mask(XX, YY, cx, cy, yaw, length, width):
    dx, dy = XX - cx, YY - cy
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = dx * c + dy * s
    lat = -dx * s + dy * c
    return (np.abs(fwd) <= length / 2.0) & (np.abs(lat) <= width / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5)   # W frames ~ 0.5 s at 10 Hz (sealed)
    ap.add_argument("--far", type=float, default=-1e9)  # EXPLORATORY: restrict ribbon to x>far (beyond the ego self-return zone, x>3.9); default = sealed (no restriction)
    ap.add_argument("--limit", type=int, default=18)
    args = ap.parse_args()
    danger = json.loads(_LOGS.read_text())
    names = list(danger)[: args.limit]
    rng = np.random.default_rng(0)
    XX, YY = _bev_centers()

    rows = []  # {scene, fp, shuf_fp}
    print(f"oracle-v0 traversal (W={args.window} frames) over {len(names)} danger logs ...", flush=True)
    for li, name in enumerate(names):
        log_dir = _AV2 / name
        if not (log_dir / "city_SE3_egovehicle.feather").exists():
            continue
        all_sweeps = sorted(int(p.stem) for p in (log_dir / "sensors" / "lidar").glob("*.feather"))
        idx_of = {ts: i for i, ts in enumerate(all_sweeps)}
        poses = _world_poses(log_dir)
        danger_ts = [int(t) for t in danger[name]]
        scene = av2_sensor.load_scene(name, _AV2, timestamps=danger_ts)
        dsorted = sorted(set(danger_ts))  # frame fi <-> dsorted[fi] (load_scene keeps sorted-sweep order)
        used = 0
        for fi, ts in enumerate(dsorted):
            if fi >= len(scene.frames):
                break
            i = idx_of.get(ts)
            if i is None or i + args.window >= len(all_sweeps):
                continue
            fr = scene.frames[fi]
            ego = fr.ego
            txi, tyi, yawi = _pose_at(poses, ts)
            # ego future swept volume in frame-i ego coords (k = 1..W)
            ci, si = math.cos(yawi), math.sin(yawi)
            sweep = np.zeros((_NX, _NY), dtype=bool)
            for k in range(1, args.window + 1):
                tsk = all_sweeps[i + k]
                txk, tyk, yawk = _pose_at(poses, tsk)
                dX, dY = txk - txi, tyk - tyi
                fwd = dX * ci + dY * si
                lat = -dX * si + dY * ci
                sweep |= _rect_mask(XX, YY, fwd, lat, yawk - yawi, ego.length, ego.width)
            sweep &= XX > args.far  # exploratory far-zone restriction (sealed default = no-op)
            den = int(sweep.sum())
            if den < _MIN_SWEEP_CELLS:
                continue  # ego barely moved -> estimand undefined
            # BEV occupied mask (obstacle voxels above road, capped at ego height) -> (NX,NY)
            centers = fr.grid.obstacle_centers(max_height_agl=ego.height)
            occ_bev = np.zeros((_NX, _NY), dtype=bool)
            if len(centers):
                bi = np.clip(np.round((centers[:, 0] - _OX) / _VOX).astype(int), 0, _NX - 1)
                bj = np.clip(np.round((centers[:, 1] - _OY) / _VOX).astype(int), 0, _NY - 1)
                occ_bev[bi, bj] = True
            n_occ = int(occ_bev.sum())
            fp = float((sweep & occ_bev).sum()) / den
            # shuffled null: relocate the same count of occupied cells at random in the grid
            shuf = np.zeros(_NX * _NY, dtype=bool)
            if n_occ:
                shuf[rng.choice(_NX * _NY, size=min(n_occ, _NX * _NY), replace=False)] = True
            shuf_fp = float((shuf.reshape(_NX, _NY) & sweep).sum()) / den
            rows.append({"scene": name, "fp": fp, "shuf_fp": shuf_fp})
            used += 1
        print(f"  {li + 1}/{len(names)} {name[:12]} -> {used} usable frames ({len(rows)} total)", flush=True)
    if not rows:
        sys.exit("no usable frames (ego never moved enough?).")

    scenes = [r["scene"] for r in rows]
    true_b = _boot_mean([r["fp"] for r in rows], scenes, rng)
    shuf_b = _boot_mean([r["shuf_fp"] for r in rows], scenes, rng)
    verdict = "INDETERMINATE"
    if true_b["defined"] and shuf_b["defined"]:
        verdict = "RELIABLE" if true_b["hi"] < shuf_b["lo"] else (
            "UNRELIABLE" if true_b["lo"] >= shuf_b["lo"] else "INDETERMINATE")
    report = {
        "substrate": "AV2-Sensor val danger logs (av2_danger_logs.json), ego in-path swept ribbon",
        "window_frames": args.window, "n_logs": len({*scenes}), "n_frames": len(rows),
        "true_fp_mean": true_b["mean"], "true_fp_ci": [true_b["lo"], true_b["hi"]],
        "shuffled_fp_mean": shuf_b["mean"], "shuffled_fp_ci": [shuf_b["lo"], shuf_b["hi"]],
        "verdict": verdict,
        "framing": "occupancy false-positive rate in the ego's physically-driven ribbon vs a shuffled-occupancy null; label-free, box-independent. One-sided (FP only). RELIABLE = true CI below shuffled CI.",
    }
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "oracle_traversal.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\noracle-v0 traversal ({report['n_logs']} logs, {len(rows)} usable frames):")
    print(f"  true occupancy false-positive rate in driven ribbon: {true_b['mean']:.4f} CI[{true_b['lo']:.4f},{true_b['hi']:.4f}]")
    print(f"  shuffled-occupancy null:                             {shuf_b['mean']:.4f} CI[{shuf_b['lo']:.4f},{shuf_b['hi']:.4f}]")
    print(f"  VERDICT: {verdict}  (RELIABLE = true CI strictly below shuffled CI = occupancy keeps the driven path clearer than random, validated label-free)")
    print(f"  wrote {_HERE / 'results' / 'oracle_traversal.json'}")
    print("  scope: one-sided (false positives only; recall = oracle-v1 camera+multi-sweep). NOT a danger/safety claim.")


if __name__ == "__main__":
    main()
