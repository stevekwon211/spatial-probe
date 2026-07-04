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


def _mesh_dict(occ_mask, origin, vs):
    v, q, fz = exposed_face_mesh(occ_mask.astype(np.uint8) * OCCUPIED, origin, vs)
    return {"verts": v.round(3).tolist(), "faces": q.tolist(), "face_z": fz.round(2).tolist()}


def _corridor_states(dense, obs, half_width=1.0, horizon=20.0, step=0.4, z_max=2.0):
    """Walk the ego forward centerline (+x). At each x-step, classify the corridor band from the
    SINGLE sweep: BLOCKED (an obstacle), FREE (all observed & empty), or UNKNOWN (unseen — could
    hide anything). This is the honest free-space story: the planner's clearance comes from the
    aggregated GT, but a single sweep only CONFIRMS free so far — beyond that the 'clear path' is fog.
    Returns (states list, confirmed_free_distance, aggregated_clearance)."""
    D, O = dense.occupancy, obs.occupancy
    origin, vs = np.asarray(dense.origin), dense.voxel_size
    nx, ny, nz = O.shape
    zc = origin[2] + np.arange(nz) * vs
    zsel = (zc > dense.ground_height) & (zc <= dense.ground_height + z_max)
    states, agg_clear, conf_free = [], None, None
    x = step
    while x < horizon:
        # voxel index range for this x-slab and the |y|<half_width band
        ix = int(round((x - origin[0]) / vs))
        jy0 = int(round((-half_width - origin[1]) / vs)); jy1 = int(round((half_width - origin[1]) / vs))
        if not (0 <= ix < nx):
            break
        Dband = D[ix, max(0, jy0):min(ny, jy1 + 1)][:, zsel]
        Oband = O[ix, max(0, jy0):min(ny, jy1 + 1)][:, zsel]
        if (Dband == OCCUPIED).any() and agg_clear is None:
            agg_clear = round(x, 1)
        total = Oband.size
        free_frac = float((Oband == FREE).sum() / total) if total else 0.0
        if (Oband == OCCUPIED).any():
            st = "blocked"
        elif (Oband == UNKNOWN).any():
            st = "unknown"
        else:
            st = "free"
        if free_frac < 0.99 and conf_free is None:
            conf_free = round(x, 1)      # single-sweep fully-confirms free only up to here
        states.append({"x": round(x, 1), "state": st, "free_frac": round(free_frac, 2)})
        x += step
    return states, conf_free, agg_clear


def main():
    names = sorted(_annotations(DATA)["scene_infos"].keys())
    nm = names[0]
    dense = load_scene(nm, DATA, mask="none").frames[0].grid   # aggregated GT (unknown≈0)
    obs = load_scene(nm, DATA, mask="lidar").frames[0].grid     # single sweep (mostly unknown)
    origin, vs = dense.origin, dense.voxel_size

    # Obstacles = the aggregated solid world, meshed (legible terrain).
    mesh_obst = _mesh_dict(dense.occupancy == OCCUPIED, origin, vs)
    # What the single sweep actually confirmed as an obstacle (bright dots — real observation).
    obs_pts = obs.obstacle_centers(max_height_agl=2.0)
    unk_frac = float((obs.occupancy == UNKNOWN).mean())

    states, conf_free, _ = _corridor_states(dense, obs)
    # aggregated clearance from the reliable continuous obstacle centers (matches the red marker)
    occ_c = dense.obstacle_centers(max_height_agl=2.0)
    corr = occ_c[(occ_c[:, 0] > 0) & (occ_c[:, 0] < 20) & (np.abs(occ_c[:, 1]) < 1.0)]
    agg_clear = round(float(corr[:, 0].min()), 1) if len(corr) else None
    n_free = sum(1 for s in states if s["state"] == "free")
    n_unk = sum(1 for s in states if s["state"] == "unknown")
    print(f"{nm}: {len(mesh_obst['faces'])} obstacle faces | corridor "
          f"free {n_free} / unknown {n_unk} / total {len(states)} steps | "
          f"aggregated clearance {agg_clear} m, single-sweep confirmed-free to {conf_free} m")

    OUT.write_text(json.dumps({
        "scene": nm, "voxel_size": vs, "origin": list(origin),
        "obstacles": mesh_obst,
        "observed": obs_pts.round(2).tolist(),
        "unknown_frac_single_sweep": round(unk_frac, 3),
        "corridor": {"half_width": 1.0, "horizon": 20.0,
                     "aggregated_clearance": agg_clear,   # planner's number (from dense GT)
                     "confirmed_free_to": conf_free,      # how far ONE sweep confirms free
                     "states": states},                   # per-0.4m free/unknown/blocked
    }))
    print(f"wrote {OUT} | single-sweep unknown {unk_frac:.1%}")


if __name__ == "__main__":
    main()
