# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Honest finer-than-0.4m reconstruction from the RAW LiDAR (Horizon B).

The web viewer meshes the 0.4m Occ3D occupancy — a coarse SUMMARY that already threw away the LiDAR's
density, so the mesh is blocky/blobby. This goes back to the source: accumulate all 39 keyframe LiDAR
sweeps into a common (frame-0 ego) frame via the ego poses (real measurements, ego-motion aligned — NOT
morphological invention), REMOVE moving objects (nuScenes 3D box tracks, velocity-gated) so they don't
ghost, and voxelize the static survivors at 0.2m (2x finer). Emits <scene>.lidar02.occ.bin (uint8
[x*ny*nz+y*nz+z], surface-hit occupancy) + <scene>.lidar02.json (dims). The web meshes it with the SAME
parametric mesher (blocky/qef) + projective texture.

Honesty: this is a SURFACE-HIT occupancy (occupied = a real LiDAR return landed in the cell) — denser and
finer than 0.4m Occ3D, but it is not the 3-state free/occupied/unknown QA grid; far range stays sparse
(beam spread) and only keyframes are on disk (intermediate 20Hz sweeps not downloaded = even denser
possible). Every occupied cell is a measured return; nothing is interpolated or invented.

Run:  cd spatial-probe && PYTHONPATH=src .venv/bin/python web/scripts/prep_lidar_recon.py --scenes scene-0061
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from probe.adapters.occ3d import load_scene, _annotations, _ordered_tokens

_ROOT = pathlib.Path.home() / "Projects/Personal/spatial-probe/data"
_OUT = pathlib.Path(__file__).resolve().parents[1] / "public" / "occ"
# match the Occ3D extent so the recon overlays the 0.4m grid 1:1, but at a finer voxel
RANGE = ((-40.0, 40.0), (-40.0, 40.0), (-1.0, 5.4))
VS = 0.2
MOVER_MPS = 0.5   # a box faster than this is dynamic -> its points ghost -> remove
BOX_MARGIN = 0.1  # m, grow the box slightly to catch edge returns


def _q2r(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _mat(q, t):
    m = np.eye(4)
    m[:3, :3] = _q2r(q)
    m[:3, 3] = np.asarray(t, float)
    return m


def _lidar_extrinsics(root: pathlib.Path) -> dict:
    """filename -> LIDAR_TOP calibrated_sensor extrinsic (sensor->ego), from the nuScenes mini tables."""
    mini = root / "nuscenes" / "v1.0-mini"
    cs = {c["token"]: c for c in json.load(open(mini / "calibrated_sensor.json"))}
    sd = json.load(open(mini / "sample_data.json"))
    return {r["filename"]: cs[r["calibrated_sensor_token"]] for r in sd if "LIDAR_TOP" in r["filename"]}


def _remove_movers(p_ek: np.ndarray, boxes) -> np.ndarray:
    """Drop points inside any MOVING box (oriented box test in the ego-k frame the boxes live in)."""
    keep = np.ones(len(p_ek), bool)
    for b in boxes:
        vx, vy = b.velocity
        if vx != vx or (vx * vx + vy * vy) ** 0.5 <= MOVER_MPS:  # NaN or slow -> static, keep
            continue
        cx, cy, cz = b.center
        l, w, h = b.size            # TrackedBox size = (length, width, height)
        cyaw, syaw = np.cos(-b.yaw), np.sin(-b.yaw)
        dx, dy, dz = p_ek[:, 0] - cx, p_ek[:, 1] - cy, p_ek[:, 2] - cz
        lx = cyaw * dx - syaw * dy  # into box-local (un-yaw about z)
        ly = syaw * dx + cyaw * dy
        inside = (np.abs(lx) <= l / 2 + BOX_MARGIN) & (np.abs(ly) <= w / 2 + BOX_MARGIN) & (np.abs(dz) <= h / 2 + BOX_MARGIN)
        keep &= ~inside
    return p_ek[keep]


def reconstruct(scene: str, extr: dict) -> tuple[np.ndarray, dict]:
    frames = _annotations(_ROOT)["scene_infos"][scene]
    toks = _ordered_tokens(frames)
    sc_boxes = load_scene(scene, _ROOT, mask="none", with_boxes=True).frames  # per-frame ego boxes + velocity
    ego0 = _mat(frames[toks[0]]["ego_pose"]["rotation"], frames[toks[0]]["ego_pose"]["translation"])
    ego0i = np.linalg.inv(ego0)
    kept = []
    for i, tk in enumerate(toks):
        info = frames[tk]
        prefix = pathlib.Path(info["camera_sensor"]["CAM_FRONT"]["img_path"]).name.split("__")[0]
        rel = f"samples/LIDAR_TOP/{prefix}__LIDAR_TOP__{info['timestamp']}.pcd.bin"
        f = _ROOT / rel
        if not f.exists() or rel not in extr:
            continue
        p = np.fromfile(f, np.float32).reshape(-1, 5)[:, :3]            # lidar-sensor frame
        ex = extr[rel]
        Rl, tl = _q2r(ex["rotation"]), np.asarray(ex["translation"], float)
        p_ek = p @ Rl.T + tl                                           # -> ego_k
        p_ek = _remove_movers(p_ek, sc_boxes[i].objects)               # drop dynamic returns
        T = ego0i @ _mat(info["ego_pose"]["rotation"], info["ego_pose"]["translation"])
        kept.append(p_ek @ T[:3, :3].T + T[:3, 3])                     # -> ego_0
    pts = np.vstack(kept)

    origin = np.array([RANGE[0][0], RANGE[1][0], RANGE[2][0]])
    nx = int(round((RANGE[0][1] - RANGE[0][0]) / VS))
    ny = int(round((RANGE[1][1] - RANGE[1][0]) / VS))
    nz = int(round((RANGE[2][1] - RANGE[2][0]) / VS))
    gi = np.floor((pts - origin) / VS).astype(np.int64)
    m = (gi[:, 0] >= 0) & (gi[:, 0] < nx) & (gi[:, 1] >= 0) & (gi[:, 1] < ny) & (gi[:, 2] >= 0) & (gi[:, 2] < nz)
    gi = gi[m]
    occ = np.zeros((nx, ny, nz), np.uint8)
    occ[gi[:, 0], gi[:, 1], gi[:, 2]] = 1
    dims = {
        "nx": nx, "ny": ny, "nz": nz, "voxel_size": VS,
        "origin": [origin[0] + VS / 2, origin[1] + VS / 2, origin[2] + VS / 2],  # voxel-0 CENTER (matches Occ3D convention)
        "occupied": int(occ.sum()), "points": int(len(pts)),
    }
    return np.ascontiguousarray(occ), dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["scene-0061"])
    args = ap.parse_args()
    extr = _lidar_extrinsics(_ROOT)
    for sc in args.scenes:
        occ, dims = reconstruct(sc, extr)
        (_OUT / f"{sc}.lidar02.occ.bin").write_bytes(occ.tobytes())
        (_OUT / f"{sc}.lidar02.json").write_text(json.dumps(dims))
        print(f"{sc}: {dims['nx']}x{dims['ny']}x{dims['nz']} @ {VS}m · {dims['occupied']} occupied · {dims['points']} static points")


if __name__ == "__main__":
    main()
