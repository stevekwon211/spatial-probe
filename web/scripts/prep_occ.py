# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Precompute Occ3D-nuScenes free-space data for the web free-space view. Dumps, per scene:
  public/occ/<scene>.occ.bin        dense occupancy uint8 [x*ny*nz+y*nz+z], 1=solid (WASM meshes it)
  public/occ/<scene>.freespace.json corridor honesty (aggregated clearance vs single-sweep confirmed,
                                     fog fraction, observed points) — the "honest gap"
  public/occ/scenes.json            {scenes, dims, voxel_size} index

Static files, so the Next app serves them with no Python at runtime. Run:
  cd spatial-probe && PYTHONPATH=src .venv/bin/python web/scripts/prep_occ.py --limit 12
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from probe.adapters.occ3d import load_scene, _annotations
from probe.grid import OCCUPIED, FREE, UNKNOWN

_ROOT = str(pathlib.Path.home() / "Projects/Personal/spatial-probe/data")
_OUT = pathlib.Path(__file__).resolve().parents[1] / "public" / "occ"
_VEHICLE_Z_MAX = 2.0


def corridor(dense, obs, half_width=1.0, horizon=20.0, step=0.4):
    D, O = dense.occupancy, obs.occupancy
    origin, vs = np.asarray(dense.origin), dense.voxel_size
    nx, ny, nz = O.shape
    zc = origin[2] + np.arange(nz) * vs
    zsel = (zc > dense.ground_height) & (zc <= dense.ground_height + _VEHICLE_Z_MAX)
    states, conf_free = [], None
    x = step
    while x < horizon:
        ix = int(round((x - origin[0]) / vs))
        jy0 = int(round((-half_width - origin[1]) / vs)); jy1 = int(round((half_width - origin[1]) / vs))
        if not (0 <= ix < nx):
            break
        band = O[ix, max(0, jy0):min(ny, jy1 + 1)][:, zsel]
        total = band.size
        ff = float((band == FREE).sum() / total) if total else 0.0
        st = "blocked" if (band == OCCUPIED).any() else ("free" if not (band == UNKNOWN).any() else "unknown")
        if ff < 0.99 and conf_free is None:
            conf_free = round(x, 1)
        states.append({"x": round(x, 1), "state": st, "free_frac": round(ff, 2)})
        x += step
    return states, conf_free


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    scenes = sorted(_annotations(_ROOT)["scene_infos"].keys())[: args.limit]
    dims = None
    for i, sc in enumerate(scenes):
        dense = load_scene(sc, _ROOT, mask="none").frames[0].grid
        obs = load_scene(sc, _ROOT, mask="lidar").frames[0].grid
        occ = np.ascontiguousarray((dense.occupancy == OCCUPIED).astype(np.uint8))
        nx, ny, nz = occ.shape
        dims = {"nx": nx, "ny": ny, "nz": nz, "voxel_size": dense.voxel_size, "origin": list(dense.origin)}
        (_OUT / f"{sc}.occ.bin").write_bytes(occ.tobytes())

        states, conf_free = corridor(dense, obs)
        oc = dense.obstacle_centers(max_height_agl=_VEHICLE_Z_MAX)
        corr = oc[(oc[:, 0] > 0) & (oc[:, 0] < 20) & (np.abs(oc[:, 1]) < 1.0)]
        agg_clear = round(float(corr[:, 0].min()), 1) if len(corr) else None
        observed = obs.obstacle_centers(max_height_agl=_VEHICLE_Z_MAX)
        if len(observed) > 6000:
            observed = observed[np.linspace(0, len(observed) - 1, 6000).astype(int)]
        fs = {
            "scene": sc, **dims,
            "unknown_frac_single_sweep": round(float((obs.occupancy == UNKNOWN).mean()), 3),
            "observed": observed.round(2).tolist(),
            "corridor": {"half_width": 1.0, "horizon": 20.0,
                         "aggregated_clearance": agg_clear, "confirmed_free_to": conf_free, "states": states},
        }
        (_OUT / f"{sc}.freespace.json").write_text(json.dumps(fs))
        print(f"[{i+1}/{len(scenes)}] {sc}: occ {nx}x{ny}x{nz} + corridor {len(states)} steps, clear {agg_clear}/{conf_free}", flush=True)

    (_OUT / "scenes.json").write_text(json.dumps({"scenes": scenes, **(dims or {})}))
    print(f"wrote {len(scenes)} scenes to {_OUT}")


if __name__ == "__main__":
    main()
