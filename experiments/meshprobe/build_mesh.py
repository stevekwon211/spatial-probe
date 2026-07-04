# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Mesh probe: Doeon's hypothesis — voxel+meshing an Occ3D-nuScenes scene into a solid surface
makes free-space / collision problems POP for a human/agent reviewer, in a way a sparse point
cloud doesn't.

Honest design (the point that makes it not a pretty lie):
 - Mesh the DENSE aggregated Occ3D occupancy (unknown≈0) → a clean surface.
 - ALSO carry the uncertainty: overlay what a SINGLE LiDAR sweep actually confirms (observed-
   occupied voxels) vs the aggregated surface — so the reviewer sees the confident mesh AND how
   little one sweep verifies (the 88%-unknown occlusion reality).
 - Overlay the ego forward CORRIDOR (free_path band) + the nearest-obstacle CLEARANCE, so
   "free-space is tight/violated HERE" is visual, not a number.

Meshing = exposed-face voxel surface (Minecraft-style): a face is emitted only where an occupied
voxel borders a non-occupied one. Zero-dependency (numpy), and HONEST for voxels — no marching-
cubes/DC interpolation inventing surface between samples. (DC/QEF from the SPACE0 engine is the
upgrade for sharp smooth surfaces; exposed-face is the right first probe.)

Output: mesh.json (verts/faces/colors + observed markers + corridor + clearance) for the viewer.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from probe.adapters.occ3d import load_scene, _annotations  # noqa: E402
from probe.grid import OCCUPIED, UNKNOWN  # noqa: E402

DATA = str(pathlib.Path(__file__).resolve().parents[2] / "data")
OUT = pathlib.Path(__file__).resolve().parent / "mesh.json"

# 6 face directions: (axis, +/-), the 4 corner offsets of that face, and an outward shade factor.
_FACES = [
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),   # +x
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),  # -x
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),   # +y
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),  # -y
    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),   # +z (top)
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),  # -z
]


def exposed_face_mesh(occ: np.ndarray, origin, vsize: float):
    """occupied bool grid -> (verts (N,3) world, faces (M,4) quad indices, face_z (M,) world-z)."""
    o = occ == OCCUPIED
    verts: list[tuple] = []
    vidx: dict[tuple, int] = {}
    quads: list[tuple] = []
    fz: list[float] = []
    ox, oy, oz = origin

    def vid(ix, iy, iz):
        key = (ix, iy, iz)
        j = vidx.get(key)
        if j is None:
            j = len(verts)
            vidx[key] = j
            verts.append((ox + ix * vsize, oy + iy * vsize, oz + iz * vsize))
        return j

    nx, ny, nz = o.shape
    for (dx, dy, dz), corners in _FACES:
        # exposed where occupied AND neighbor (shifted) is not occupied (or off-grid)
        shifted = np.zeros_like(o)
        sx0, sx1 = max(dx, 0), nx + min(dx, 0)
        sy0, sy1 = max(dy, 0), ny + min(dy, 0)
        sz0, sz1 = max(dz, 0), nz + min(dz, 0)
        shifted[max(-dx, 0):nx - max(dx, 0), max(-dy, 0):ny - max(dy, 0), max(-dz, 0):nz - max(dz, 0)] = \
            o[sx0:sx1, sy0:sy1, sz0:sz1]
        exposed = o & ~shifted
        idx = np.argwhere(exposed)
        for x, y, z in idx:
            q = [vid(x + cx, y + cy, z + cz) for cx, cy, cz in corners]
            quads.append(tuple(q))
            fz.append(oz + (z + 0.5) * vsize)
    return np.array(verts, dtype=float), np.array(quads, dtype=int), np.array(fz, dtype=float)


def main():
    names = sorted(_annotations(DATA)["scene_infos"].keys())
    # pick a scene with a decent obstacle count near the ego
    nm = names[0]
    dense = load_scene(nm, DATA, mask="none").frames[0].grid
    obs = load_scene(nm, DATA, mask="lidar").frames[0].grid
    origin, vs = dense.origin, dense.voxel_size

    verts, quads, fz = exposed_face_mesh(dense.occupancy, origin, vs)
    print(f"{nm}: {int((dense.occupancy==OCCUPIED).sum())} occupied voxels -> "
          f"{len(verts)} verts, {len(quads)} faces")

    # single-sweep confirmed obstacles (what one LiDAR actually sees) — the honesty overlay
    obs_centers = obs.obstacle_centers(max_height_agl=2.0)
    # unknown fraction (the occlusion reality)
    unk_frac = float((obs.occupancy == UNKNOWN).mean())

    # ego forward corridor: a band along +x (ego forward), ego half-width 1.0 m, horizon 20 m.
    # nearest obstacle in that band = the clearance the planner cares about.
    occ_centers = dense.obstacle_centers(max_height_agl=2.0)
    in_corr = (occ_centers[:, 0] > 0) & (occ_centers[:, 0] < 20) & (np.abs(occ_centers[:, 1]) < 1.0)
    corr_obs = occ_centers[in_corr]
    clearance = float(corr_obs[:, 0].min()) if len(corr_obs) else None

    OUT.write_text(json.dumps({
        "scene": nm, "voxel_size": vs, "origin": list(origin),
        "verts": verts.round(3).tolist(),
        "faces": quads.tolist(),
        "face_z": fz.round(2).tolist(),
        "observed": obs_centers.round(2).tolist(),   # single-sweep confirmed points
        "unknown_frac_single_sweep": round(unk_frac, 3),
        "corridor": {"half_width": 1.0, "horizon": 20.0, "clearance": clearance,
                     "n_obstacles": int(len(corr_obs))},
        "n_occupied": int((dense.occupancy == OCCUPIED).sum()),
    }))
    print(f"wrote {OUT} | single-sweep unknown {unk_frac:.1%} | corridor clearance {clearance} m")


if __name__ == "__main__":
    main()
