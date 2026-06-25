# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Lower a real on-disk dataset into the PRISM Scene IR (the ingest path).

The probe adapters (`probe.adapters.occ3d`, `probe.adapters.av2_sensor`) already produce a
`probe.Scene` from real data. PRISM REUSES them verbatim and layers the IR on top: it lifts each
box into an `Entity`, groups boxes into `Track`s by id, builds the `CoordinateFrame`s (the
calibration), records `Observation`s and `GroundTruth`, and stamps `Provenance`. Nothing here
re-reads occupancy -- it wraps what the probe adapter already loaded.

`ingest(path)` autodetects the dataset from the directory layout (an AV2-Sensor log dir vs an
Occ3D-nuScenes data root + scene name) and returns a validated `SceneIR`.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pyarrow as pa

from probe.scene import Scene
from prism.ir import (
    CoordinateFrame,
    Entity,
    GroundTruth,
    Observation,
    Pose,
    Provenance,
    SceneIR,
    Track,
)

__all__ = ["ingest", "scene_to_ir", "av2_coordinate_frames", "AdapterError"]


class AdapterError(ValueError):
    """Raised when an ingest path cannot be resolved to a known dataset layout."""


def _tracks_from_scene(scene: Scene, frame_name: str) -> tuple[Track, ...]:
    """Group per-frame boxes into Tracks. The probe Scene does not carry box ids, so within a frame
    boxes get a stable synthetic id (category + index); a Track here is therefore the per-(category,
    slot) series -- enough to exercise the Track type losslessly. (A real id join lives in a later
    stage; this keeps the IR honest about what the source provides.)"""
    series: dict[str, list[tuple[float, Entity]]] = {}
    for fr in scene.frames:
        per_cat: dict[str, int] = {}
        for box in fr.objects:
            slot = per_cat.get(box.label, 0)
            per_cat[box.label] = slot + 1
            eid = f"{box.label}#{slot}"
            ent = Entity.from_tracked_box(box, entity_id=eid, frame=frame_name)
            series.setdefault(eid, []).append((fr.time, ent))
    tracks = []
    for eid, pairs in series.items():
        pairs.sort(key=lambda p: p[0])
        tracks.append(
            Track(
                entity_id=eid,
                category=pairs[0][1].category,
                timestamps=tuple(t for t, _ in pairs),
                states=tuple(e for _, e in pairs),
            )
        )
    return tuple(tracks)


def _ground_truth_from_scene(scene: Scene, frame_name: str) -> tuple[GroundTruth, ...]:
    out = []
    for i, fr in enumerate(scene.frames):
        ents = tuple(
            Entity.from_tracked_box(box, entity_id=f"{box.label}#{j}", frame=frame_name)
            for j, box in enumerate(fr.objects)
        )
        out.append(GroundTruth(frame_index=i, entities=ents))
    return tuple(out)


def av2_coordinate_frames(log_dir: pathlib.Path) -> tuple[CoordinateFrame, ...]:
    """Build the calibration frames for an AV2-Sensor log from `city_SE3_egovehicle.feather`.

    That feather is the REAL calibration on disk: per-timestamp ego->city (world) SE3 as a
    (w,x,y,z) quaternion + translation. We take the FIRST stamp's transform as the representative
    ego->world frame (the per-frame series is recoverable from the same feather; one frame here is
    enough to make the calibration a typed, composable object instead of an assumption). The 'lidar'
    frame is identity-to-ego (the av2 adapter voxelizes in the ego frame). No per-sensor intrinsics
    feather exists in this release on disk, so camera intrinsics stay None -- explicitly absent, not
    faked.
    """
    se3 = log_dir / "city_SE3_egovehicle.feather"
    world = CoordinateFrame(name="world", parent=None, pose=Pose((0.0, 0.0, 0.0)))
    lidar = CoordinateFrame(name="lidar", parent="ego", pose=Pose((0.0, 0.0, 0.0)))
    if not se3.exists():
        ego = CoordinateFrame(name="ego", parent="world", pose=Pose((0.0, 0.0, 0.0)))
        return (world, ego, lidar)
    t = pa.ipc.open_file(pa.memory_map(str(se3), "r")).read_all()
    ts = np.asarray(t.column("timestamp_ns").to_pylist(), dtype=np.int64)
    order = int(np.argmin(ts))
    pose = Pose(
        translation=(
            float(t.column("tx_m")[order].as_py()),
            float(t.column("ty_m")[order].as_py()),
            float(t.column("tz_m")[order].as_py()),
        ),
        quaternion=(
            float(t.column("qw")[order].as_py()),
            float(t.column("qx")[order].as_py()),
            float(t.column("qy")[order].as_py()),
            float(t.column("qz")[order].as_py()),
        ),
    )
    ego = CoordinateFrame(name="ego", parent="world", pose=pose)
    return (world, ego, lidar)


def scene_to_ir(
    scene: Scene,
    *,
    provenance: Provenance,
    coordinate_frames: tuple[CoordinateFrame, ...] = (),
    sensor: str = "lidar",
) -> SceneIR:
    """Wrap a loaded `probe.Scene` into a `SceneIR`, lifting boxes into entities/tracks/GT and
    recording one observation per frame. `coordinate_frames` is the calibration (pass the av2
    frames; default = a minimal world/ego/lidar identity chain)."""
    frame_name = "ego"
    if not coordinate_frames:
        coordinate_frames = (
            CoordinateFrame(name="world", parent=None, pose=Pose((0.0, 0.0, 0.0))),
            CoordinateFrame(name="ego", parent="world", pose=Pose((0.0, 0.0, 0.0))),
            CoordinateFrame(name=sensor, parent="ego", pose=Pose((0.0, 0.0, 0.0))),
        )
    observations = tuple(
        Observation(sensor=sensor, timestamp=fr.time, frame_index=i, modality="lidar")
        for i, fr in enumerate(scene.frames)
    )
    return SceneIR(
        scene=scene,
        coordinate_frames=coordinate_frames,
        tracks=_tracks_from_scene(scene, frame_name),
        observations=observations,
        ground_truth=_ground_truth_from_scene(scene, frame_name),
        provenance=provenance,
    )


def _is_av2_log(path: pathlib.Path) -> bool:
    return (path / "sensors" / "lidar").is_dir() and (path / "city_SE3_egovehicle.feather").exists()


def ingest(path: str | pathlib.Path, *, scene: str | None = None) -> SceneIR:
    """Autodetect the dataset at `path` and return a validated SceneIR.

    - An AV2-Sensor log dir (has sensors/lidar/ + city_SE3_egovehicle.feather) -> av2 adapter,
      real calibration frames from the SE3 feather.
    - An Occ3D-nuScenes data root (has annotations.json + gts/) -> occ3d adapter; requires
      `scene` (e.g. 'scene-0001'), or defaults to the first scene.
    """
    p = pathlib.Path(path)
    if _is_av2_log(p):
        from probe.adapters.av2_sensor import load_scene

        sc = load_scene(p.name, p.parent, with_boxes=True)
        prov = Provenance(dataset="av2_sensor", log_id=p.name, adapter="probe.adapters.av2_sensor")
        return scene_to_ir(sc, provenance=prov, coordinate_frames=av2_coordinate_frames(p))
    if (p / "annotations.json").exists() and (p / "gts").is_dir():
        import json

        from probe.adapters.occ3d import load_scene

        if scene is None:
            scenes = sorted(json.loads((p / "annotations.json").read_text())["scene_infos"].keys())
            if not scenes:
                raise AdapterError(f"{p}: Occ3D annotations.json has no scenes")
            scene = scenes[0]
        sc = load_scene(scene, p, mask="none", with_boxes=False)
        prov = Provenance(dataset="occ3d_nuscenes", log_id=scene, adapter="probe.adapters.occ3d")
        return scene_to_ir(sc, provenance=prov, sensor="lidar")
    raise AdapterError(
        f"{p}: not a recognized dataset (need an AV2-Sensor log dir or an Occ3D data root with annotations.json + gts/)"
    )
