# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Rerun adapter -- log a PRISM Scene IR to Rerun for visualization. ONE adapter, not the core.

Rerun is an OPTIONAL extra. Everything here imports `rerun` LAZILY (inside the functions, never at
module top level), so `import prism` and `import prism.adapters.rerun_adapter` both work in a venv
WITHOUT rerun-sdk installed -- invariant 3 (core works without Rerun) + invariant 1 (the PRISM API
never hands back a rerun object; `to_rerun` returns the written `.rrd` path, not a stream).

What gets logged, per the S4 contract:
  - occupancy / LiDAR obstacle voxels  -> `rr.Points3D`  (entity path `world/ego/occupancy`)
  - tracked object boxes               -> `rr.Boxes3D`   (entity path `world/ego/objects`)
  - coordinate frames (calibration)    -> `rr.Transform3D` (one per CoordinateFrame, parent-chained)
  - timeline                           -> `frame` time index = `Frame.time` seconds

The occupancy points are pulled with the SAME `OccupancyGrid.obstacle_centers` the predicates use,
so what you SEE is what the predicates REASON OVER -- not a separate render path that could drift
from the query semantics. The render is a derived VIEW; it is never the source of truth (a query's
answer comes from Parquet/IR, never from the `.rrd`).

`pip install rerun-sdk` -> a `.rrd` is produced from real on-disk data. No rerun -> `to_rerun`
raises `RerunNotInstalled` with the exact install hint, and the adapter test skips. Nothing is faked.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np

from probe.grid import UnknownPolicy
from prism.ir import CoordinateFrame, SceneIR

__all__ = ["to_rerun", "RerunWriter", "RerunNotInstalled"]

_INSTALL_HINT = "rerun not installed: pip install 'spatial-probe[rerun]' (or: pip install rerun-sdk)"


class RerunNotInstalled(RuntimeError):
    """Raised when the Rerun adapter is invoked but `rerun-sdk` is not importable.

    A clear, actionable failure (with the install command) -- never a silent no-op or a fake render
    (repo rule: explicit failure over silent fallback).
    """


def _import_rerun():
    """Import `rerun` lazily, translating ImportError into the actionable RerunNotInstalled.

    This is the ONLY place `rerun` is imported in PRISM. Keeping it here is what makes the optional
    dependency optional: the import cost (and the absence) is paid only when someone actually renders.
    """
    try:
        import rerun as rr  # noqa: PLC0415 (lazy by design -- invariant 3)
    except ImportError as e:  # pragma: no cover - exercised only in a rerun-free venv
        raise RerunNotInstalled(_INSTALL_HINT) from e
    return rr


def _entity_path(cf: CoordinateFrame, by_name: dict[str, CoordinateFrame]) -> str:
    """The parent-chained Rerun entity path for a CoordinateFrame.

    The minimal world/ego/lidar chain the adapters emit nests as world -> world/ego ->
    world/ego/lidar. A named root with no CoordinateFrame entry (e.g. 'world') is appended as the
    chain head so transforms hang off a consistent root.
    """
    chain = [cf.name]
    cur = cf
    seen = {cf.name}
    while cur.parent is not None and cur.parent in by_name and cur.parent not in seen:
        chain.append(cur.parent)
        seen.add(cur.parent)
        cur = by_name[cur.parent]
    if cur.parent is not None and cur.parent not in by_name:
        chain.append(cur.parent)
    return "/".join(reversed(chain))


def to_rerun(
    scene_ir: SceneIR,
    out_path: str | pathlib.Path,
    *,
    application_id: str = "prism",
    policy: UnknownPolicy = UnknownPolicy.FREE,
    max_points_per_frame: Optional[int] = None,
) -> pathlib.Path:
    """Log `scene_ir` to a Rerun recording and save it to `out_path` (a `.rrd` file).

    Returns the written `.rrd` path (a PRISM/pathlib type -- never a rerun stream object, invariant
    1). Raises `RerunNotInstalled` if rerun-sdk is absent (the call site / test then skips).

    Per frame: the timeline `frame` is set to `Frame.time` seconds; obstacle voxels are logged as
    `rr.Points3D` (via `OccupancyGrid.obstacle_centers`, the predicate-aligned extraction) and the
    tracked boxes as `rr.Boxes3D`. Coordinate frames are logged once as static `rr.Transform3D`.
    `max_points_per_frame`, if set, deterministically subsamples dense occupancy (stride decimation,
    seed-free) to keep a `.rrd` small -- it changes the VIEW only, never any query result.
    """
    rr = _import_rerun()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Log against an EXPLICIT recording handle (never the implicit global stream), so the function
    # is import-safe and test-isolated.
    rec = rr.RecordingStream(application_id)

    # coordinate frames (static calibration) -> rr.Transform3D, parent-chained
    by_name = {cf.name: cf for cf in scene_ir.coordinate_frames}
    for cf in scene_ir.coordinate_frames:
        tx, ty, tz = cf.pose.translation
        qw, qx, qy, qz = cf.pose.quaternion
        rec.log(
            _entity_path(cf, by_name),
            rr.Transform3D(
                translation=[float(tx), float(ty), float(tz)],
                quaternion=rr.Quaternion(xyzw=[float(qx), float(qy), float(qz), float(qw)]),
            ),
            static=True,
        )

    for fr in scene_ir.scene.frames:
        rec.set_time("frame", timestamp=float(fr.time))

        pts = fr.grid.obstacle_centers(unknown_policy=policy)
        if max_points_per_frame is not None and len(pts) > max_points_per_frame > 0:
            stride = int(np.ceil(len(pts) / max_points_per_frame))
            pts = pts[::stride]
        rec.log("world/ego/occupancy", rr.Points3D(np.asarray(pts, dtype=np.float32)))

        boxes = fr.objects
        if boxes:
            centers = np.array([b.center for b in boxes], dtype=np.float32)
            half_sizes = np.array([(b.size[0] / 2.0, b.size[1] / 2.0, b.size[2] / 2.0) for b in boxes], dtype=np.float32)
            quats = []
            for b in boxes:
                cyaw, syaw = np.cos(b.yaw / 2.0), np.sin(b.yaw / 2.0)
                quats.append([0.0, 0.0, float(syaw), float(cyaw)])  # xyzw, yaw about +z
            labels = [b.label for b in boxes]
            rec.log(
                "world/ego/objects",
                rr.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    quaternions=[rr.Quaternion(xyzw=q) for q in quats],
                    labels=labels,
                ),
            )
        else:
            rec.log("world/ego/objects", rr.Clear(recursive=True))

    rec.save(str(out))
    return out


class RerunWriter:
    """`SceneWriter` adapter: `write(scene_ir, path)` -> a `.rrd`. Lazy-imports rerun (invariant 3).

    Registered behind `prism.adapters.get_writer('rerun')`. Satisfies the SceneWriter protocol
    structurally; it is intentionally thin -- all the logging lives in `to_rerun`.
    """

    name = "rerun"

    def write(self, scene_ir: SceneIR, path: str | pathlib.Path) -> pathlib.Path:
        return to_rerun(scene_ir, path)
