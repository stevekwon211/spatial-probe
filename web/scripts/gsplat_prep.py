# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Prep a nuScenes scene for 3D Gaussian Splatting (path C, the photoreal track — separate from the
deterministic DC-mesh pipeline). Converts one scene's surround cameras across all keyframes into a
nerfstudio/instant-ngp `transforms.json` (posed images in a consistent GLOBAL frame) + a LiDAR init
point cloud (`points3d.ply`), the two inputs a gsplat trainer needs.

This step is CPU + runs now (verifiable). Training is the GPU step (see gsplat-plan.md): a static
3DGS reconstructs the street well; MOVING objects (cars/peds over the ~20s drive) ghost unless
decomposed with the 3D boxes (StreetGaussians/OmniRe) — v1 is static-with-ghosting, honestly scoped.

Run:  cd spatial-probe && PYTHONPATH=src .venv/bin/python web/scripts/gsplat_prep.py --scene scene-0061
Out:  web/public/gsplat/<scene>/{transforms.json, points3d.ply, images -> symlinked}
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from probe.adapters.occ3d import load_scene, _annotations
from probe.grid import OCCUPIED

_ROOT = pathlib.Path.home() / "Projects/Personal/spatial-probe/data"
_SAMPLES = _ROOT / "samples"
_OUT = pathlib.Path(__file__).resolve().parents[1] / "public" / "gsplat"


def _quat_to_rot(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _mat(rot_q, t):
    m = np.eye(4)
    m[:3, :3] = _quat_to_rot(rot_q)
    m[:3, 3] = np.asarray(t, float)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0061")
    args = ap.parse_args()
    ann = _annotations(str(_ROOT))
    frames = ann["scene_infos"][args.scene]
    out = _OUT / args.scene
    out.mkdir(parents=True, exist_ok=True)

    K0 = None
    nerf_frames = []
    W = H = None
    for tok, info in frames.items():
        # ego -> global for this keyframe
        ego = _mat(info["ego_pose"]["rotation"], info["ego_pose"]["translation"])
        for cam, cs in info["camera_sensor"].items():
            img = _SAMPLES / cs["img_path"]
            if not img.exists():
                continue
            # cam -> ego (extrinsic) then ego -> global => cam -> world (nerfstudio wants cam2world)
            cam2ego = _mat(cs["extrinsic"]["rotation"], cs["extrinsic"]["translation"])
            c2w = ego @ cam2ego
            K = np.asarray(cs["intrinsics"], float)
            if K0 is None:
                K0 = K
            nerf_frames.append({
                "file_path": f"../../../samples/{cs['img_path']}",  # served/read directly
                "transform_matrix": c2w.tolist(),
                "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            })
    # image size from the first image
    from PIL import Image
    W, H = Image.open(_SAMPLES / next(iter(frames.values()))["camera_sensor"]["CAM_FRONT"]["img_path"]).size

    transforms = {
        "camera_model": "OPENCV", "w": W, "h": H,
        "fl_x": nerf_frames[0]["fl_x"], "fl_y": nerf_frames[0]["fl_y"],
        "cx": nerf_frames[0]["cx"], "cy": nerf_frames[0]["cy"],
        "ply_file_path": "points3d.ply",
        "frames": nerf_frames,
    }
    (out / "transforms.json").write_text(json.dumps(transforms, indent=1))

    # LiDAR init: occupied voxel centers of the aggregated scene, in GLOBAL frame (matches c2w)
    f0 = next(iter(frames.values()))
    dense = load_scene(args.scene, str(_ROOT), mask="none").frames[0].grid
    ego0 = _mat(f0["ego_pose"]["rotation"], f0["ego_pose"]["translation"])
    occ_idx = np.argwhere(dense.occupancy == OCCUPIED)
    centers_ego = np.asarray(dense.origin) + (occ_idx + 0.5) * dense.voxel_size
    centers_h = np.c_[centers_ego, np.ones(len(centers_ego))]
    centers_world = (ego0 @ centers_h.T).T[:, :3]
    _write_ply(out / "points3d.ply", centers_world)

    print(f"{args.scene}: {len(nerf_frames)} posed images ({len(frames)} keyframes x 6 cams), "
          f"{len(centers_world)} init points, {W}x{H} -> {out}")


def _write_ply(path: pathlib.Path, pts: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} 128 128 128\n")


if __name__ == "__main__":
    main()
