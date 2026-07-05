# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Precompute Occ3D-nuScenes free-space data for the web free-space view. Per scene (only scenes
that have BOTH occupancy GT and the 6 camera images on disk):
  public/occ/<scene>.occ.bin        dense occupancy uint8 [x*ny*nz+y*nz+z], 1=solid (WASM meshes it)
  public/occ/<scene>.color.bin      per-voxel RGB uint8 [x*ny*nz+y*nz+z]*3, 0=uncolored — projective
                                     color from the cameras (nearest unoccluded view), used as the
                                     mesh's vertex color = "camera-textured occupancy"
  public/occ/<scene>.freespace.json corridor honesty (aggregated clear vs single-sweep confirmed)
  public/occ/scenes.json            index

Projective coloring is HONEST diffuse: nuScenes RGB has lighting baked in (no albedo/roughness/
metallic — true PBR needs inverse rendering), and geometry is 0.4 m voxels, so texture detail
exceeds geometry. Occlusion is a z-buffer per camera (a voxel is colored only from a view where it
is the front surface). Run:  cd spatial-probe && PYTHONPATH=src .venv/bin/python web/scripts/prep_occ.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np
from PIL import Image

from probe.adapters.occ3d import load_scene, _annotations
from probe.grid import OCCUPIED, FREE, UNKNOWN

_ROOT = str(pathlib.Path.home() / "Projects/Personal/spatial-probe/data")
_SAMPLES = pathlib.Path(_ROOT) / "samples"
_OUT = pathlib.Path(__file__).resolve().parents[1] / "public" / "occ"
_VEHICLE_Z_MAX = 2.0


def _quat_to_rot(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _dual_available(ann, limit):
    gts = set(os.listdir(pathlib.Path(_ROOT) / "gts"))
    out = []
    for sc, frames in ann["scene_infos"].items():
        if sc not in gts:
            continue
        info = frames[next(iter(frames))]
        cams = info["camera_sensor"]
        if all((_SAMPLES / cs["img_path"]).exists() for cs in cams.values()):
            out.append(sc)
        if len(out) >= limit:
            break
    return sorted(out)


def voxel_colors(scene, dense, cam_info) -> np.ndarray:
    """Per-occupied-voxel RGB via projective coloring with z-buffer occlusion. Voxels are in the
    ego frame (the occupancy grid's frame); each camera's extrinsic is sensor->ego."""
    origin, vs = np.asarray(dense.origin, float), dense.voxel_size
    nx, ny, nz = dense.occupancy.shape
    occ_idx = np.argwhere(dense.occupancy == OCCUPIED)  # (M,3)
    if not len(occ_idx):
        return np.zeros((nx * ny * nz, 3), np.uint8)
    centers = origin + (occ_idx + 0.5) * vs             # ego-frame world centers
    colors = np.zeros((len(occ_idx), 3), np.uint8)
    best_z = np.full(len(occ_idx), np.inf)

    for cam, cs in cam_info.items():
        K = np.asarray(cs["intrinsics"], float)
        R = _quat_to_rot(cs["extrinsic"]["rotation"])   # sensor->ego
        t = np.asarray(cs["extrinsic"]["translation"], float)
        img = np.asarray(Image.open(_SAMPLES / cs["img_path"]).convert("RGB"))
        H, W = img.shape[:2]
        pc = (centers - t) @ R                          # ego -> camera
        z = pc[:, 2]
        uv = (pc @ K.T)
        zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = uv[:, 0] / zz
        v = uv[:, 1] / zz
        infr = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.clip(u, 0, W - 1).astype(int)
        vi = np.clip(v, 0, H - 1).astype(int)
        # z-buffer occlusion: nearest voxel wins each 4px pixel bucket -> only the front surface is colored
        seen: dict[tuple[int, int], int] = {}
        for i in np.argsort(z):
            if not infr[i]:
                continue
            key = (int(vi[i]) >> 2, int(ui[i]) >> 2)
            if key in seen:
                continue
            seen[key] = i
            if z[i] < best_z[i]:
                best_z[i] = z[i]
                colors[i] = img[vi[i], ui[i]]
    full = np.zeros((nx * ny * nz, 3), np.uint8)
    lin = occ_idx[:, 0] * ny * nz + occ_idx[:, 1] * nz + occ_idx[:, 2]
    full[lin] = colors
    return full


def _mat4(quat, t):
    m = np.eye(4); m[:3, :3] = _quat_to_rot(quat); m[:3, 3] = np.asarray(t, float); return m


def voxel_colors_multiframe(dense, frames) -> np.ndarray:
    """Per-voxel RGB projected from ALL keyframes' cameras (not just frame-0), each keyframe's camera
    transformed into the frame-0 ego frame (the grid's frame) via the ego poses:
    cam->ego0 = inv(ego0) @ egok @ extrinsic_k. The ego drives forward, so far/side surfaces one
    keyframe missed get colored by a keyframe that drove up to them -> near-full coverage, nearest
    (smallest-z) view wins per voxel. This feeds the web's fallback color layer (no shader change)."""
    origin, vs = np.asarray(dense.origin, float), dense.voxel_size
    nx, ny, nz = dense.occupancy.shape
    occ_idx = np.argwhere(dense.occupancy == OCCUPIED)
    if not len(occ_idx):
        return np.zeros((nx * ny * nz, 3), np.uint8)
    centers = origin + (occ_idx + 0.5) * vs
    colors = np.zeros((len(occ_idx), 3), np.uint8)
    best_z = np.full(len(occ_idx), np.inf)
    f0 = next(iter(frames.values()))
    ego0_inv = np.linalg.inv(_mat4(f0["ego_pose"]["rotation"], f0["ego_pose"]["translation"]))
    for info in frames.values():
        egok = _mat4(info["ego_pose"]["rotation"], info["ego_pose"]["translation"])
        for cam, cs in info["camera_sensor"].items():
            cam2ego0 = ego0_inv @ egok @ _mat4(cs["extrinsic"]["rotation"], cs["extrinsic"]["translation"])
            R, t = cam2ego0[:3, :3], cam2ego0[:3, 3]   # sensor->ego0
            K = np.asarray(cs["intrinsics"], float)
            img = np.asarray(Image.open(_SAMPLES / cs["img_path"]).convert("RGB"))
            H, W = img.shape[:2]
            pc = (centers - t) @ R                     # ego0 -> camera
            z = pc[:, 2]
            zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
            u = (pc @ K.T)[:, 0] / zz; v = (pc @ K.T)[:, 1] / zz
            infr = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < best_z)
            ui = np.clip(u, 0, W - 1).astype(int); vi = np.clip(v, 0, H - 1).astype(int)
            idx = np.where(infr)[0]
            colors[idx] = img[vi[idx], ui[idx]]
            best_z[idx] = z[idx]
    full = np.zeros((nx * ny * nz, 3), np.uint8)
    lin = occ_idx[:, 0] * ny * nz + occ_idx[:, 1] * nz + occ_idx[:, 2]
    full[lin] = colors
    return full


def export_cameras(scene, cam_info) -> list[dict]:
    """Export the frame-0 six camera images + their intrinsics/extrinsic so the web can do
    render-time PROJECTIVE texturing — sampling the full-res image per fragment instead of the
    0.4 m per-voxel color (which smears). Extrinsic is sensor->ego; we store its inverse rotation
    Rt = R^T (ego->sensor, row-major) + sensor origin t so the shader computes p_sensor =
    Rt·(p_ego − t) directly. Same frame as the mesh (both are frame-0 ego)."""
    import shutil
    cam_dir = _OUT / "cams" / scene
    cam_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for cam, cs in cam_info.items():
        src = _SAMPLES / cs["img_path"]
        W, H = Image.open(src).size
        shutil.copyfile(src, cam_dir / f"{cam}.jpg")  # native nuScenes JPEG, K unchanged
        K = np.asarray(cs["intrinsics"], float)
        R = _quat_to_rot(cs["extrinsic"]["rotation"])          # sensor->ego
        Rt = R.T                                               # ego->sensor
        out.append({
            "cam": cam, "img": f"cams/{scene}/{cam}.jpg", "w": W, "h": H,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "Rt": [float(x) for x in Rt.flatten()],           # row-major ego->sensor
            "t": [float(x) for x in cs["extrinsic"]["translation"]],
        })
    return out


def corridor(dense, obs, half_width=1.0, horizon=20.0, step=0.4):
    O = obs.occupancy
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
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    ann = _annotations(_ROOT)
    scenes = _dual_available(ann, args.limit)
    print(f"scenes with occupancy + camera images: {len(scenes)} -> {scenes}", flush=True)
    dims = None
    for i, sc in enumerate(scenes):
        dense = load_scene(sc, _ROOT, mask="none").frames[0].grid
        obs = load_scene(sc, _ROOT, mask="lidar").frames[0].grid
        occ = np.ascontiguousarray((dense.occupancy == OCCUPIED).astype(np.uint8))
        nx, ny, nz = occ.shape
        dims = {"nx": nx, "ny": ny, "nz": nz, "voxel_size": dense.voxel_size, "origin": list(dense.origin)}
        (_OUT / f"{sc}.occ.bin").write_bytes(occ.tobytes())

        scene_frames = ann["scene_infos"][sc]
        cam_info = scene_frames[next(iter(scene_frames))]["camera_sensor"]
        col = voxel_colors_multiframe(dense, scene_frames)  # all 39 keyframes -> near-full coverage
        (_OUT / f"{sc}.color.bin").write_bytes(col.tobytes())
        n_colored = int((col.reshape(-1, 3).any(axis=1)).sum())
        cams = export_cameras(sc, cam_info)  # full-res images for render-time projective texturing
        (_OUT / f"{sc}.cams.json").write_text(json.dumps({"cameras": cams}))

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
        print(f"[{i+1}/{len(scenes)}] {sc}: {nx}x{ny}x{nz} · {n_colored} voxels colored · clear {agg_clear}/{conf_free}", flush=True)

    (_OUT / "scenes.json").write_text(json.dumps({"scenes": scenes, "textured": True, **(dims or {})}))
    print(f"wrote {len(scenes)} scenes to {_OUT}")


if __name__ == "__main__":
    main()
