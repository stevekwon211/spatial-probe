# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Export occquery real-data scenes to JSON for the web 3D viewer (`web/public/data/occquery`).

Layout (split so the viewer loads a light meta up front and a frame's voxels only on demand):
  occquery/index.json            -- scene list + a global note
  occquery/<scene>.json          -- meta: ego, voxel_size, and per-frame predicate readings/counts
  occquery/<scene>/f<t>.json     -- just that frame's obstacle voxel centers (ego frame)

The predicate readings in the meta are the REAL outputs of the same predicates the retrieval runs,
so the viewer's log is the instrument's own numbers, not a re-derivation.

Usage: python experiments/occquery_v0/export_web.py
"""
from __future__ import annotations

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import numpy as np  # noqa: E402

from probe.adapters.occ3d import load_scene  # noqa: E402
from probe.grid import OCCUPIED  # noqa: E402
from probe.predicates.clearance import lateral_clearance  # noqa: E402
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path  # noqa: E402

_DATA = _HERE.parents[1] / "data"
_OUT = _HERE.parents[1] / "web" / "public" / "data" / "occquery"
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]


def _finite(v: float) -> float | None:
    """JSON has no inf; map it to null so the viewer shows 'none' instead of a bogus number."""
    return None if v == float("inf") else round(float(v), 2)


def _band_obstacles(scene, t):
    grid, ego = scene.grid_at(t), scene.ego_at(t)
    occ = grid.occupancy
    idx = np.argwhere(occ == OCCUPIED)
    centers = np.asarray(grid.origin) + idx * grid.voxel_size
    z = centers[:, 2]
    band = (z > grid.ground_height) & (z <= grid.ground_height + ego.height)
    return grid, ego, np.round(centers[band], 1), int(band.sum()), int(len(idx))


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for name in MINI:
        scene = load_scene(name, _DATA, mask="none")
        e0, g0 = scene.ego_at(0), scene.grid_at(0)
        (_OUT / name).mkdir(exist_ok=True)
        metas = []
        for t in scene.times():
            grid, ego, obstacles, n_band, n_total = _band_obstacles(scene, t)
            (_OUT / name / f"f{t}.json").write_text(json.dumps({"t": t, "obstacles": obstacles.tolist()}))
            metas.append({
                "t": t,
                "speed": round(float(ego.speed), 2),
                "n_obstacles_band": n_band,
                "n_obstacles_total": n_total,
                "predicates": {
                    "ego_width": round(float(ego.width), 2),
                    "min_free_width": _finite(min_free_width_along_path(grid, ego, 2.0)),
                    "lateral_clearance": _finite(lateral_clearance(grid, ego)),
                    "free_path_blocked": not free_along_ego_path(grid, ego, 1.0, min_cluster_voxels=2),
                },
            })
        (_OUT / f"{name}.json").write_text(json.dumps({
            "scene": name,
            "voxel_size": g0.voxel_size,
            "ground_height": g0.ground_height,
            "ego": {"width": round(e0.width, 2), "length": round(e0.length, 2), "height": round(e0.height, 2)},
            "n_frames": len(metas),
            "frames": metas,
        }))
        index.append({"scene": name, "n_frames": len(metas)})
        print(f"  {name}: {len(metas)} frames")
    (_OUT / "index.json").write_text(json.dumps({
        "scenes": index,
        "note": "occquery real Occ3D-nuScenes mini (dense GT). corridor MATCH judgment is DEFERRED "
                "(see results/summary.md); min_free_width is a measurement only, not a verdict.",
    }))
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
