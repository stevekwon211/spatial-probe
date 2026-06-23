# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Export raw LiDAR point clouds to the web viewer, ego-aligned with the Occ3D voxels.

The viewer renders the Occ3D VOXELS (discretized state). This exports the RAW LiDAR scan (the
measurement the voxels are built FROM) so the viewer can toggle voxel <-> point-cloud and you can SEE
what discretization keeps vs drops -- the repo's state-vs-render theme, made literal.

Per Occ3D frame (token = nuScenes sample_token), find the LIDAR_TOP sweep file + its calibrated_sensor
extrinsic, load the float32 [x,y,z,intensity,ring] points (sensor frame), transform sensor->ego
(p_ego = R @ p_sensor + t), keep the ego height band, subsample for web perf, and write
web/public/data/occquery/<scene>/lidar<t>.json as ego-frame [forward, left, up, intensity]. Same ego
frame the voxels use, so the two overlay exactly. Run: python experiments/occquery_v0/export_lidar.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from projection import quat_to_rotmat  # reuse the verified quaternion->rotation

_DATA = _HERE.parents[1] / "data"
_NUSC = _DATA / "nuscenes" / "v1.0-mini"
_OUT = _HERE.parents[1] / "web" / "public" / "data" / "occquery"
_MAX_POINTS = 9000   # web render budget; subsample beyond this
_GROUND = -1.0       # ego-frame ground (Occ3D RANGE z-min); keep points above it
_HEIGHT = 5.4        # keep points up to ego-frame this high (matches the occupancy band ceiling)
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]


def _lidar_index() -> dict[str, dict]:
    """{sample_token: {filename, R(sensor->ego), t}} for LIDAR_TOP keyframes."""
    cs = {c["token"]: c for c in json.loads((_NUSC / "calibrated_sensor.json").read_text())}
    idx: dict[str, dict] = {}
    for d in json.loads((_NUSC / "sample_data.json").read_text()):
        if d["is_key_frame"] and d["filename"].startswith("samples/LIDAR_TOP"):
            c = cs[d["calibrated_sensor_token"]]
            idx[d["sample_token"]] = {
                "filename": d["filename"],
                "R": quat_to_rotmat(c["rotation"]),
                "t": np.asarray(c["translation"], dtype=float),
            }
    return idx


def _export_frame(rec: dict, out_path: pathlib.Path, rng: np.random.Generator) -> int:
    pts = np.fromfile(_DATA / rec["filename"], dtype=np.float32).reshape(-1, 5)
    xyz = pts[:, :3].astype(float)
    ego = (rec["R"] @ xyz.T).T + rec["t"]              # sensor -> ego (x fwd, y left, z up)
    band = (ego[:, 2] > _GROUND) & (ego[:, 2] <= _HEIGHT)
    ego, inten = ego[band], pts[band, 3]
    if len(ego) > _MAX_POINTS:
        keep = rng.choice(len(ego), _MAX_POINTS, replace=False)
        ego, inten = ego[keep], inten[keep]
    out = [[round(float(x), 2), round(float(y), 2), round(float(z), 2), round(float(i), 1)]
           for (x, y, z), i in zip(ego, inten)]
    out_path.write_text(json.dumps({"points": out}))  # [forward, left, up, intensity], ego frame
    return len(out)


def main() -> None:
    ann = json.loads((_DATA / "annotations.json").read_text())
    lidar = _lidar_index()
    rng = np.random.default_rng(0)
    for name in MINI:
        si = ann["scene_infos"][name]
        # temporal order via prev/next (same as the adapter); fall back to dict order
        head = next((tok for tok, fr in si.items() if fr.get("prev") in (None, "", "EOF")), next(iter(si)))
        order, tok, seen = [], head, set()
        while tok and tok not in seen and tok in si:
            order.append(tok); seen.add(tok)
            nxt = si[tok].get("next"); tok = nxt if nxt not in (None, "", "EOF") else None
        order = order or list(si)
        (_OUT / name).mkdir(parents=True, exist_ok=True)
        n = 0
        for t, sample_token in enumerate(order):
            rec = lidar.get(sample_token)
            if rec is None:
                continue
            n += _export_frame(rec, _OUT / name / f"lidar{t}.json", rng)
        print(f"  {name}: {len(order)} frames, ~{n // max(len(order),1)} pts/frame -> lidar*.json")
    print(f"wrote LiDAR point clouds to {_OUT}/<scene>/lidar<t>.json")


if __name__ == "__main__":
    main()
