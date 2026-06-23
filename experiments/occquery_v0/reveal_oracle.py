# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Future-reveal H3 oracle (occquery) -- the INDEPENDENT grader built per future_reveal_preregistration.md.

The grader carves the RAW t+k LiDAR sweep into a per-voxel FREE/OCCUPIED/UNKNOWN reveal-truth, by its
own algorithm (direct point-carving), independent of the Occ3D accumulated semantics the predicate's
data lineage shares. A free-space claim the occupancy predicate extrapolated at frame t (over space the
t-sweep did NOT see) is then graded against what the t+k sweep DIRECTLY observed once the ego drove on.

This module is the carver + nuScenes-metadata indices; the grading harness (occ_pred vs box_pred vs
reveal_truth, gap + bootstrap) is `reveal_grade.py`. Carve is pure numpy (vectorized ray-sampling at
voxel resolution): each LiDAR return marks its endpoint voxel OCCUPIED and every voxel its ray passes
through FREE; voxels no ray reaches stay UNKNOWN (excluded from grading). Sealed scope: 10 mini scenes
with local raw sweeps, static-only, temporal (not cross-modal) independence.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from probe.adapters.occ3d import GRID_SHAPE, ORIGIN, VOXEL_SIZE
from probe.grid import FREE, OCCUPIED, UNKNOWN

_ORIGIN = np.asarray(ORIGIN, dtype=float)
_SHAPE = np.asarray(GRID_SHAPE)


def rotmat(q) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation (body->world / sensor->ego, nuScenes convention)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _index(nusc_root: pathlib.Path) -> tuple[dict, dict, dict]:
    """Build {sample_token -> (lidar_filename, cs_token, ego_pose_token)}, {cs_token -> (R,t)},
    {ego_pose_token -> (R,t)} for LIDAR_TOP key-frames."""
    sd = json.loads((nusc_root / "sample_data.json").read_text())
    cs = {c["token"]: (rotmat(c["rotation"]), np.asarray(c["translation"], float))
          for c in json.loads((nusc_root / "calibrated_sensor.json").read_text())}
    ep = {e["token"]: (rotmat(e["rotation"]), np.asarray(e["translation"], float))
          for e in json.loads((nusc_root / "ego_pose.json").read_text())}
    lidar = {}
    for d in sd:
        if d.get("is_key_frame") and "LIDAR_TOP" in d.get("filename", ""):
            lidar[d["sample_token"]] = (d["filename"], d["calibrated_sensor_token"], d["ego_pose_token"])
    return lidar, cs, ep


def sweep_points_ego(data_root: pathlib.Path, filename: str, R_cs: np.ndarray, t_cs: np.ndarray):
    """Load a nuScenes LIDAR_TOP .pcd.bin (float32 [x,y,z,intensity,ring], sensor frame) -> ego frame.

    Returns (points_ego (N,3), sensor_origin_ego (3,))."""
    raw = np.fromfile(data_root / filename, dtype=np.float32).reshape(-1, 5)[:, :3].astype(float)
    return raw @ R_cs.T + t_cs, t_cs.copy()


def carve(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Vectorized ray-sampling carve of a point cloud (already in the TARGET ego frame) into a
    FREE/OCCUPIED/UNKNOWN grid matching the Occ3D voxel spec.

    Each ray sensor_origin->point: voxels strictly before the endpoint (sampled at voxel resolution)
    -> FREE; the endpoint voxel -> OCCUPIED (set last, so a hit overrides a pass-through); voxels no
    ray reaches -> UNKNOWN."""
    grid = np.full(tuple(GRID_SHAPE), UNKNOWN, dtype=np.int8)
    o = np.asarray(origin, float)
    d = points - o
    rng = np.linalg.norm(d, axis=1)
    keep = rng > VOXEL_SIZE
    d, rng, pts = d[keep], rng[keep], points[keep]
    nsteps = np.floor(rng / VOXEL_SIZE).astype(int)               # free samples per ray (excl. endpoint)
    total = int(nsteps.sum())
    if total:
        ray_idx = np.repeat(np.arange(len(pts)), nsteps)
        start = np.cumsum(nsteps) - nsteps
        within = np.arange(total) - start[ray_idx]                # 0 .. nsteps[i]-1
        frac = (within + 1) * VOXEL_SIZE / rng[ray_idx]
        samp = o + frac[:, None] * d[ray_idx]
        fv = np.round((samp - _ORIGIN) / VOXEL_SIZE).astype(int)
        inb = np.all((fv >= 0) & (fv < _SHAPE), axis=1)
        fv = fv[inb]
        grid[fv[:, 0], fv[:, 1], fv[:, 2]] = FREE
    ov = np.round((pts - _ORIGIN) / VOXEL_SIZE).astype(int)
    inb = np.all((ov >= 0) & (ov < _SHAPE), axis=1)
    ov = ov[inb]
    grid[ov[:, 0], ov[:, 1], ov[:, 2]] = OCCUPIED
    return grid


def reveal_truth_in_frame(points_ego_b, origin_ego_b, R_b, t_b, R_a, t_a) -> np.ndarray:
    """Carve frame-b's sweep (given in ego-b frame) into the ego-a frame grid, via ego_pose b->world->a.

    Returns a FREE/OCCUPIED/UNKNOWN grid indexed in ego-a (so it aligns with the frame-a predicate)."""
    world = points_ego_b @ R_b.T + t_b
    pts_a = (world - t_a) @ R_a
    o_world = origin_ego_b @ R_b.T + t_b
    o_a = (o_world - t_a) @ R_a
    return carve(pts_a, o_a)
