# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Can the future-reveal carve isolate REAL box-less structure INDEPENDENTLY (no Occ3D GT)?

The re-audit showed the carve's revealed-OCCUPIED "truth" is 0.2% real structure -- 97.5% is ground/
free single returns. The open question (owner, 2026-06-24): is that fixable by cleaning the carve truth
INDEPENDENTLY (height above the road + multi-return), keeping the oracle independent, and how many real
structures survive? If height+multi-return isolates structure and yields enough cells, the substrate may
be salvageable; if it stays a handful, the 10-scene oracle is confirmed underpowered for H3.

For each revealed-OCCUPIED cell (unobserved@t, observed@t+k, near, static, in-window -- same selection
as reveal_grade), this counts how many survive three INDEPENDENT truth definitions and VALIDATES each
against Occ3D dense GT class (validation only -- the cleaning itself never uses GT):
  raw            = any occupied voxel in the ego-body slab (the polluted current truth)
  height         = an occupied voxel ABOVE road (z > ground + 0.5 m)  [excludes road-surface returns]
  height+multi   = >= 2 occupied voxels above road                    [drops single-return noise]
Read-only, pure numpy, no GPU, no new data.

Run: python experiments/occquery_v0/structure_density.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

import reveal_grade as R
from probe.adapters.occ3d import GRID_SHAPE, GROUND_HEIGHT, VOXEL_SIZE, _ordered_tokens, map_occupancy
from probe.grid import OCCUPIED, UNKNOWN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    si = json.loads((R._DATA / "annotations.json").read_text())["scene_infos"]
    lidar, cs, ep = R._index(R._NUSC)
    scenes = [n for n in sorted(si)
              if all(t in lidar and (R._DATA / lidar[t][0]).exists() for t in _ordered_tokens(si[n]))][: args.limit]

    zk = R._ORIGIN[2] + np.arange(GRID_SHAPE[2]) * VOXEL_SIZE          # world z per voxel level
    struct_band = zk > (GROUND_HEIGHT + 0.5)                          # above the road surface
    near_x = np.abs(R._ORIGIN[0] + np.arange(GRID_SHAPE[0]) * VOXEL_SIZE) <= R._NEAR
    near_y = np.abs(R._ORIGIN[1] + np.arange(GRID_SHAPE[1]) * VOXEL_SIZE) <= R._NEAR
    near = near_x[:, None] & near_y[None, :]

    sets = {"raw": [0, 0], "height": [0, 0], "height+multi": [0, 0]}  # [count, gt_structure_count]
    print(f"probing independent structure-isolation on {len(scenes)} scenes ...", flush=True)
    for name in scenes:
        s = si[name]; toks = _ordered_tokens(s)
        for i in range(len(toks) - 1):
            tok = toks[i]; fr = s[tok]; lab = np.load(R._DATA / fr["gt_path"])
            fn, cst, _ = lidar[tok]; pts, o = R.sweep_points_ego(R._DATA, fn, *cs[cst]); Rt, tt = ep[lidar[tok][2]]
            obs = map_occupancy(lab["semantics"], lab["mask_lidar"]); spd = R._ego_speed(s, toks, i)
            _, inwin = R._occ_pred_bev(obs, spd)
            obs_t = (lab["mask_lidar"][:, :, R._BAND] == 1).any(2)
            dyn = np.isin(lab["semantics"][:, :, R._BAND], np.arange(11)).any(2)
            j = i + 1; tk = toks[j]; fn2, cst2, _ = lidar[tk]
            p2, o2 = R.sweep_points_ego(R._DATA, fn2, *cs[cst2]); Rb, tb = ep[lidar[tk][2]]
            reveal3d = R.reveal_truth_in_frame(p2, o2, Rb, tb, Rt, tt)
            reveal_bev = R._collapse(reveal3d)
            base = (reveal_bev != UNKNOWN) & ~obs_t & near & ~dyn & inwin
            occ_struct = (reveal3d[:, :, struct_band] == OCCUPIED).sum(axis=2)     # occupied count above road
            sem2 = np.load(R._DATA / s[tk]["gt_path"])["semantics"][:, :, R._BAND]  # GT (validation only)
            gt_struct = np.isin(sem2, (15, 16)).any(2)                              # manmade/vegetation
            for key, mask in (("raw", base & (reveal_bev == OCCUPIED)),
                              ("height", base & (occ_struct >= 1)),
                              ("height+multi", base & (occ_struct >= 2))):
                sets[key][0] += int(mask.sum())
                sets[key][1] += int((mask & gt_struct).sum())

    print(f"\nINDEPENDENT STRUCTURE-ISOLATION ({len(scenes)} scenes):\n")
    print(f"  {'truth definition':>16} {'cells':>8} {'GT-real-structure':>18} {'precision':>10}")
    out = {"scenes": len(scenes), "sets": {}}
    for key, (n, g) in sets.items():
        prec = g / n if n else 0.0
        out["sets"][key] = {"cells": n, "gt_structure": g, "structure_precision": prec}
        print(f"  {key:>16} {n:>8} {g:>18} {prec:>9.1%}")
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "structure_density.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n  wrote {_HERE / 'results' / 'structure_density.json'}")
    print("  READING: if height+multi yields MANY cells that are MOSTLY GT-real-structure (high precision),")
    print("  independent cleaning salvages the truth. If it stays a handful, the 10-scene oracle is")
    print("  confirmed underpowered -- a structure-bearing dataset (more sweeps / cross-modal) is needed.")


if __name__ == "__main__":
    main()
