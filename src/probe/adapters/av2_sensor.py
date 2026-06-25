# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Argoverse-2 SENSOR -> probe.Scene adapter (R0-danger substrate).

R0-v3 returned a clean negative on SAFE nuScenes following; the signal, if any, lives in DANGER, which
nuScenes barely has. AV2-Sensor val (150 logs) + the RefAV scenario-mining val feather give ~22,846
REFERRED danger frames (1038x nuScenes). This adapter loads AV2-Sensor logs into the SAME probe.Scene
contract occ3d.py produces, so R0 runs UNCHANGED (swap the one load_scene import).

The dense-3D side is the only net-new piece: AV2 ships RAW LiDAR (not a voxel grid), so each ego-frame
sweep is voxelized into the Occ3D-IDENTICAL OccupancyGrid spec (200x200x16, 0.4 m, x[-40,40] forward,
y[-40,40] left, z[-1,5.4] up). A voxel with >=1 above-road LiDAR point -> OCCUPIED, else FREE (R0's
occ_gap reads OCCUPIED voxels; a full FREE/UNKNOWN raycast is not needed for the nearest-obstacle scan).
Boxes are already AV2 ego-frame (+x fwd, +y left, ego at origin); velocity is finite-differenced per track.

NOTE (verify at first run): AV2 LiDAR feather cols x/y/z(halffloat),intensity; annotations.feather cols
timestamp_ns,track_uuid,category,length_m/width_m/height_m,qw/qx/qy/qz,tx_m/ty_m/tz_m,num_interior_pts;
city_SE3_egovehicle.feather cols timestamp_ns,qw..qz,tx_m,ty_m,tz_m. Column names are checked on load.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pyarrow as pa

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox

# Occ3D-identical grid spec (so the predicates run unchanged on AV2 occupancy)
VOXEL_SIZE = 0.4
RANGE = ((-40.0, 40.0), (-40.0, 40.0), (-1.0, 5.4))
GRID_SHAPE = (200, 200, 16)
ORIGIN = (RANGE[0][0] + VOXEL_SIZE / 2.0, RANGE[1][0] + VOXEL_SIZE / 2.0, RANGE[2][0] + VOXEL_SIZE / 2.0)
GROUND_HEIGHT = RANGE[2][0]  # -1.0 (grid floor); AV2 ego-frame road is ~z=0, so road returns are dropped below
_ROAD_Z = 0.3  # drop LiDAR returns below this (ego-frame, ~road surface) so occupancy = obstacles, not ground
# ego self-return removal: the roof LiDAR sees the ego body (verified: stray returns at fwd~0.5 m, z~1 m).
# AV2 ego cuboid = rear-axle origin, center ~x=1.42 m, ~4.9 x 2.0 m, so drop points inside it. A real lead
# sits beyond the front bumper (x > _EGO_X1), so removing the ego cuboid CANNOT drop a real obstacle.
_EGO_X0, _EGO_X1, _EGO_HALF_W = -1.1, 3.9, 1.05

# AV2 fine category -> the coarse labels the probe queries use (first prefix match wins)
_COARSE = (
    ("BICYCLE", "bicycle"), ("BICYCLIST", "bicycle"),
    ("MOTORCYCLE", "motorcycle"), ("MOTORCYCLIST", "motorcycle"),
    ("PEDESTRIAN", "pedestrian"), ("STROLLER", "pedestrian"), ("WHEELCHAIR", "pedestrian"),
    ("REGULAR_VEHICLE", "vehicle"), ("LARGE_VEHICLE", "vehicle"), ("BUS", "vehicle"),
    ("TRUCK", "vehicle"), ("BOX_TRUCK", "vehicle"), ("VEHICLE", "vehicle"), ("TRAILER", "vehicle"),
)


def _read_feather(p: pathlib.Path):
    return pa.ipc.open_file(pa.memory_map(str(p), "r")).read_all()


def _coarse(cat: str) -> str:
    for prefix, lab in _COARSE:
        if cat.startswith(prefix):
            return lab
    return "other"


def _quat_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """BEV yaw (rad about +z) from an (w,x,y,z) quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _voxelize(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Raw ego-frame LiDAR sweep -> OccupancyGrid occupancy (200,200,16) int. FREE default; an
    above-road in-range point marks its voxel OCCUPIED. axis0=forward(x), axis1=left(y), axis2=up(z)."""
    occ = np.full(GRID_SHAPE, FREE, dtype=int)
    (x0, x1), (y0, y1), (z0, z1) = RANGE
    ego = (x > _EGO_X0) & (x < _EGO_X1) & (np.abs(y) < _EGO_HALF_W)  # ego self-returns: not obstacles
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z > _ROAD_Z) & (z < z1) & ~ego
    if not m.any():
        return occ
    ix = ((x[m] - x0) / VOXEL_SIZE).astype(np.intp)
    iy = ((y[m] - y0) / VOXEL_SIZE).astype(np.intp)
    iz = ((z[m] - z0) / VOXEL_SIZE).astype(np.intp)
    np.clip(ix, 0, GRID_SHAPE[0] - 1, out=ix)
    np.clip(iy, 0, GRID_SHAPE[1] - 1, out=iy)
    np.clip(iz, 0, GRID_SHAPE[2] - 1, out=iz)
    occ[ix, iy, iz] = OCCUPIED
    return occ


def _ego_speeds(city_se3: pathlib.Path) -> dict[int, float]:
    """timestamp_ns -> ego speed (m/s) by finite-differencing the city-frame ego translations."""
    t = _read_feather(city_se3)
    ts = np.asarray(t.column("timestamp_ns").to_pylist(), dtype=np.int64)
    tx = np.asarray(t.column("tx_m").to_pylist(), dtype=float)
    ty = np.asarray(t.column("ty_m").to_pylist(), dtype=float)
    order = np.argsort(ts)
    ts, tx, ty = ts[order], tx[order], ty[order]
    speed: dict[int, float] = {}
    for i in range(len(ts)):
        j = i + 1 if i + 1 < len(ts) else i - 1
        if j < 0:
            speed[int(ts[i])] = 0.0
            continue
        dt = abs(ts[j] - ts[i]) / 1e9
        d = math.hypot(tx[j] - tx[i], ty[j] - ty[i])
        speed[int(ts[i])] = d / dt if dt > 0 else 0.0
    return speed


def _box_index(annotations: pathlib.Path):
    """Build per-timestamp ego-frame boxes with finite-differenced per-track velocity.
    Returns {timestamp_ns: [TrackedBox, ...]}."""
    t = _read_feather(annotations)
    cols = {c: np.asarray(t.column(c).to_pylist()) for c in
            ["timestamp_ns", "track_uuid", "category", "length_m", "width_m", "height_m",
             "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"]}
    n = len(cols["timestamp_ns"])
    # per-track forward velocity by finite-differencing tx across the track's frames (ego frame)
    by_track: dict[str, list[int]] = {}
    for i in range(n):
        by_track.setdefault(str(cols["track_uuid"][i]), []).append(i)
    vel = np.full((n, 2), np.nan)
    for idxs in by_track.values():
        idxs.sort(key=lambda k: int(cols["timestamp_ns"][k]))
        for a, i in enumerate(idxs):
            j = idxs[a + 1] if a + 1 < len(idxs) else (idxs[a - 1] if a > 0 else None)
            if j is None:
                continue
            dt = abs(int(cols["timestamp_ns"][j]) - int(cols["timestamp_ns"][i])) / 1e9
            if dt > 0:
                vel[i] = [(float(cols["tx_m"][j]) - float(cols["tx_m"][i])) / dt,
                          (float(cols["ty_m"][j]) - float(cols["ty_m"][i])) / dt]
    out: dict[int, list[TrackedBox]] = {}
    for i in range(n):
        cat = str(cols["category"][i])
        out.setdefault(int(cols["timestamp_ns"][i]), []).append(TrackedBox(
            center=(float(cols["tx_m"][i]), float(cols["ty_m"][i]), float(cols["tz_m"][i])),
            size=(float(cols["length_m"][i]), float(cols["width_m"][i]), float(cols["height_m"][i])),
            yaw=_quat_yaw(float(cols["qw"][i]), float(cols["qx"][i]), float(cols["qy"][i]), float(cols["qz"][i])),
            label=_coarse(cat),
            velocity=(float(vel[i, 0]), float(vel[i, 1])),
        ))
    return out


def load_scene(scene_name: str, data_root: pathlib.Path | str, *, mask: str = "none",
               with_boxes: bool = True, timestamps: list[int] | None = None) -> Scene:
    """Load one AV2-Sensor log as a probe.Scene (ego-centric occupancy voxelized from raw LiDAR).

    `scene_name` = the AV2 log UUID; `data_root/<log>/` must hold sensors/lidar/<ts>.feather,
    city_SE3_egovehicle.feather, annotations.feather. `mask` is ignored (single-sweep observed view;
    no accumulated GT). `timestamps` restricts to a danger window (the REFERRED frames); None = all sweeps.
    Frames are temporal. Boxes are AV2 ego-frame (ego at origin, +x fwd); occupancy matches the Occ3D spec.
    """
    root = pathlib.Path(data_root) / scene_name
    sweeps = sorted(int(p.stem) for p in (root / "sensors" / "lidar").glob("*.feather"))
    if timestamps is not None:
        keep = set(int(t) for t in timestamps)
        sweeps = [s for s in sweeps if s in keep]
    speeds = _ego_speeds(root / "city_SE3_egovehicle.feather")
    boxes = _box_index(root / "annotations.feather") if with_boxes else {}
    frames: list[Frame] = []
    for ts in sweeps:
        lt = _read_feather(root / "sensors" / "lidar" / f"{ts}.feather")
        x = np.asarray(lt.column("x").to_pylist(), dtype=float)
        y = np.asarray(lt.column("y").to_pylist(), dtype=float)
        z = np.asarray(lt.column("z").to_pylist(), dtype=float)
        grid = OccupancyGrid(_voxelize(x, y, z), VOXEL_SIZE, ORIGIN, GROUND_HEIGHT)
        ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=float(speeds.get(ts, 0.0)))
        frames.append(Frame(grid, ego, time=ts / 1e9, objects=tuple(boxes.get(ts, ()))))
    return Scene(tuple(frames), scene_name)
