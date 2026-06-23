# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Resource-zero feasibility probe: is there enough FUTURE-REVEAL material on Occ3D-nuScenes?

A future-reveal oracle grades a free-space claim made at frame t (where part of the region is
UNOBSERVED, so the dense-GT label there is an accumulation guess) against what a LATER frame t+k
DIRECTLY observes once the ego has driven forward and the previously-occluded space becomes visible.
The later direct observation is independent of the t-frame claim in time and vantage -- the
"second witness" route that does not need a second vehicle or any new download.

Before designing that oracle, this probe answers ONE precondition: across the 850-scene Occ3D-nuScenes
trainval already on disk, how many NEAR, STATIC voxels are UNOBSERVED at frame t but DIRECTLY OBSERVED
at t+k (after registering the two ego-centric grids via ego_pose)? If that count is large across many
scenes, the reveal mechanism has material; if near-zero, future-reveal is dead and we fall back.

SCOPE (honest): this measures the GEOMETRY OPPORTUNITY (mask_lidar transitions, ego-pose-registered),
NOT the oracle itself. Whether the t+k observation is *independent* of the predicate's label (the Occ3D
dense GT may have been accumulated FROM the t+k sweep) is a DESIGN question the pre-registration must
resolve -- the clean grader is the RAW t+k LiDAR sweep treated as a fresh measurement, with the
predicate forbidden from having seen it. This probe only sizes the opportunity. Static = semantics>=11
(ground/manmade/veg/free; dynamic classes 0-10 excluded -- a moving object revealed later says nothing
about frame t). Read-only, pure numpy, no GPU, no new data.

Run: python experiments/occquery_v0/future_reveal_density.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from probe.adapters.occ3d import GRID_SHAPE, ORIGIN, VOXEL_SIZE, _ordered_tokens, map_occupancy
from probe.grid import FREE, OCCUPIED

_DATA = _HERE.parents[1] / "data"
_NEAR = 15.0          # m, ego-frame |x|,|y| bound = the predicate-relevant near free-space zone
_KS = (1, 3, 5, 10)   # frame gaps t -> t+k to probe
_ORIGIN = np.asarray(ORIGIN, dtype=float)
_SHAPE = np.asarray(GRID_SHAPE)


def _rotmat(q) -> np.ndarray:
    """nuScenes ego_pose quaternion [w,x,y,z] -> 3x3 ego->world rotation."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _frame(si, tok):
    """Load one frame: (semantics, mask_lidar, R ego->world, t translation)."""
    fr = si[tok]
    lab = np.load(_DATA / fr["gt_path"])
    ep = fr["ego_pose"]
    return lab["semantics"], lab["mask_lidar"], _rotmat(ep["rotation"]), np.asarray(ep["translation"], float)


def _revealed(sem_b, ml_b, R_b, t_b, ml_a, R_a, t_a):
    """Count NEAR STATIC voxels OBSERVED at frame b(=t+k) that were UNOBSERVED at frame a(=t).

    Registers b's observed voxels into a's ego frame via ego_pose. Returns (n_free, n_occ, n_total)."""
    obs = np.argwhere(ml_b == 1)                          # (M,3) voxel idx observed at b
    if obs.size == 0:
        return 0, 0, 0
    p_b = _ORIGIN + obs * VOXEL_SIZE                       # ego-b metric centers
    world = p_b @ R_b.T + t_b                              # ego-b -> world
    p_a = (world - t_a) @ R_a                              # world -> ego-a (R_a^T x = x @ R_a for rows)
    near = (np.abs(p_a[:, 0]) <= _NEAR) & (np.abs(p_a[:, 1]) <= _NEAR)
    if not near.any():
        return 0, 0, 0
    cls_b = sem_b[obs[:, 0], obs[:, 1], obs[:, 2]]
    static = cls_b >= 11                                  # ground/manmade/veg/free; exclude dynamic 0-10
    cand = near & static
    if not cand.any():
        return 0, 0, 0
    vox_a = np.round((p_a - _ORIGIN) / VOXEL_SIZE).astype(int)
    inb = np.all((vox_a >= 0) & (vox_a < _SHAPE), axis=1)
    cand &= inb
    if not cand.any():
        return 0, 0, 0
    va = vox_a[cand]
    unobs_a = ml_a[va[:, 0], va[:, 1], va[:, 2]] == 0     # was UNOBSERVED at a
    occ_b = map_occupancy(sem_b)                          # dense FREE/OCCUPIED at b
    occ_vals = occ_b[obs[cand, 0], obs[cand, 1], obs[cand, 2]][unobs_a]
    n_free = int((occ_vals == FREE).sum())
    n_occ = int((occ_vals == OCCUPIED).sum())
    return n_free, n_occ, n_free + n_occ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    scene_infos = json.loads((_DATA / "annotations.json").read_text())["scene_infos"]
    names = sorted(scene_infos)[: args.limit]

    per_k = {k: {"free": [], "occ": [], "pairs": 0} for k in _KS}
    print(f"probing future-reveal density on {len(names)} scenes ...", flush=True)
    for n_i, name in enumerate(names):
        si = scene_infos[name]
        toks = _ordered_tokens(si)
        cache: dict[int, tuple] = {}
        for i in range(len(toks)):
            for k in _KS:
                j = i + k
                if j >= len(toks):
                    continue
                if i not in cache:
                    cache[i] = _frame(si, toks[i])
                if j not in cache:
                    cache[j] = _frame(si, toks[j])
                sem_a, ml_a, R_a, t_a = cache[i]
                sem_b, ml_b, R_b, t_b = cache[j]
                nf, no, _ = _revealed(sem_b, ml_b, R_b, t_b, ml_a, R_a, t_a)
                per_k[k]["free"].append(nf)
                per_k[k]["occ"].append(no)
                per_k[k]["pairs"] += 1
        if (n_i + 1) % 10 == 0:
            print(f"  {n_i + 1}/{len(names)} scenes", flush=True)

    out = {"scenes": len(names), "near_m": _NEAR, "by_k": {}}
    print(f"\nFUTURE-REVEAL DENSITY ({len(names)} scenes, near |x,y|<={_NEAR} m, static voxels):\n")
    print(f"  {'k (frames)':>10} {'pairs':>6} {'free/frame':>11} {'occ/frame':>10} {'total/frame':>12} {'%pairs>=50':>11}")
    for k in _KS:
        fr = np.array(per_k[k]["free"], float)
        oc = np.array(per_k[k]["occ"], float)
        tot = fr + oc
        frac = float((tot >= 50).mean()) * 100 if tot.size else 0.0
        out["by_k"][k] = {"pairs": per_k[k]["pairs"], "free_per_frame": float(fr.mean()) if fr.size else 0.0,
                          "occ_per_frame": float(oc.mean()) if oc.size else 0.0,
                          "total_per_frame": float(tot.mean()) if tot.size else 0.0, "pct_pairs_ge50": frac}
        print(f"  {k:>10} {per_k[k]['pairs']:>6} {fr.mean():>11.1f} {oc.mean():>10.1f} {tot.mean():>12.1f} {frac:>10.1f}%")
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "future_reveal_density.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {_HERE / 'results' / 'future_reveal_density.json'}")
    print("  READING: total/frame = NEAR STATIC voxels unobserved@t but directly observed@t+k.")
    print("  Large + both free AND occ present across many pairs => future-reveal has material to grade")
    print("  (then pre-register the INDEPENDENT grader = raw t+k LiDAR sweep). Near-zero => fall back.")


if __name__ == "__main__":
    main()
