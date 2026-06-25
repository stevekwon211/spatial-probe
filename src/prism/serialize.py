# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Lossless (de)serialization of the PRISM Scene IR.

`to_parquet` / `from_parquet` is the round-trip contract: `from_parquet(to_parquet(s))` reproduces
`s` byte-for-byte at the content level (`content_hash(a) == content_hash(b)`). The IR mixes dense
numpy occupancy grids with small relational tables, so the serialization splits them:

- the dense occupancy of every frame is stacked into ONE Arrow column (raw little-endian int64
  bytes + a shape sidecar), so the grid survives exactly;
- everything else (ego poses, entities, tracks, coordinate frames, provenance, ...) is a JSON
  blob stored in the file-level Arrow metadata.

A single Parquet file therefore holds the whole IR. The JSON-in-metadata path keeps the relational
layers human-inspectable and trivially lossless (numbers round-trip as JSON; NaN is preserved via a
sentinel). `content_hash` is a sha256 over a canonical JSON of the IR plus the raw grid bytes -- the
load-bearing equality the tests assert on.

`to_openlabel_json` emits an OpenLABEL-style frame/object JSON (the interchange export). `to_jsonl`
emits one JSON object per frame (a streaming-friendly dump). Both are derived views, not the
round-trip source of truth -- only Parquet is lossless.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pathlib
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from probe.grid import EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox
from prism.ir import (
    SCHEMA_VERSION,
    CoordinateFrame,
    DatasetSlice,
    Entity,
    Event,
    Failure,
    GroundTruth,
    Observation,
    Pose,
    Prediction,
    Provenance,
    Relation,
    SceneIR,
    Track,
)

__all__ = [
    "to_parquet",
    "from_parquet",
    "to_openlabel_json",
    "to_jsonl",
    "content_hash",
    "scene_ir_to_dict",
    "scene_ir_from_dict",
]

_NAN = "__nan__"
_INF = "__inf__"
_NEG_INF = "__-inf__"
_META_KEY = b"prism_ir"


# --- NaN/inf-safe float encoding (JSON has no NaN; preserve it exactly) ---
def _enc_float(x: float) -> Any:
    x = float(x)
    if math.isnan(x):
        return _NAN
    if x == math.inf:
        return _INF
    if x == -math.inf:
        return _NEG_INF
    return x


def _dec_float(x: Any) -> float:
    if x == _NAN:
        return float("nan")
    if x == _INF:
        return math.inf
    if x == _NEG_INF:
        return -math.inf
    return float(x)


def _enc_tuple(t) -> list:
    return [_enc_float(v) for v in t]


def _dec_tuple(lst) -> tuple:
    return tuple(_dec_float(v) for v in lst)


# --- IR <-> plain-dict (the canonical, JSON-able form) ---
def _pose_to_dict(p: Pose) -> dict:
    return {"translation": _enc_tuple(p.translation), "quaternion": _enc_tuple(p.quaternion)}


def _pose_from_dict(d: dict) -> Pose:
    return Pose(translation=_dec_tuple(d["translation"]), quaternion=_dec_tuple(d["quaternion"]))


def _entity_to_dict(e: Entity) -> dict:
    return {
        "entity_id": e.entity_id,
        "category": e.category,
        "pose": _pose_to_dict(e.pose),
        "size": _enc_tuple(e.size),
        "velocity": _enc_tuple(e.velocity),
        "frame": e.frame,
    }


def _entity_from_dict(d: dict) -> Entity:
    return Entity(
        entity_id=d["entity_id"],
        category=d["category"],
        pose=_pose_from_dict(d["pose"]),
        size=_dec_tuple(d["size"]),
        velocity=_dec_tuple(d["velocity"]),
        frame=d["frame"],
    )


def _box_to_dict(b: TrackedBox) -> dict:
    return {
        "center": _enc_tuple(b.center),
        "size": _enc_tuple(b.size),
        "yaw": _enc_float(b.yaw),
        "label": b.label,
        "velocity": _enc_tuple(b.velocity),
    }


def _box_from_dict(d: dict) -> TrackedBox:
    return TrackedBox(
        center=_dec_tuple(d["center"]),
        size=_dec_tuple(d["size"]),
        yaw=_dec_float(d["yaw"]),
        label=d["label"],
        velocity=_dec_tuple(d["velocity"]),
    )


def _ego_to_dict(e: EgoPose) -> dict:
    return {
        "position": _enc_tuple(e.position),
        "heading": _enc_float(e.heading),
        "speed": _enc_float(e.speed),
        "width": _enc_float(e.width),
        "length": _enc_float(e.length),
        "height": _enc_float(e.height),
    }


def _ego_from_dict(d: dict) -> EgoPose:
    return EgoPose(
        position=_dec_tuple(d["position"]),
        heading=_dec_float(d["heading"]),
        speed=_dec_float(d["speed"]),
        width=_dec_float(d["width"]),
        length=_dec_float(d["length"]),
        height=_dec_float(d["height"]),
    )


def _grid_meta(g: OccupancyGrid) -> dict:
    return {
        "shape": list(g.occupancy.shape),
        "dtype": str(g.occupancy.dtype),
        "voxel_size": _enc_float(g.voxel_size),
        "origin": _enc_tuple(g.origin),
        "ground_height": _enc_float(g.ground_height),
    }


def _cf_to_dict(cf: CoordinateFrame) -> dict:
    return {
        "name": cf.name,
        "parent": cf.parent,
        "pose": _pose_to_dict(cf.pose),
        "intrinsics": None if cf.intrinsics is None else [[_enc_float(v) for v in row] for row in np.asarray(cf.intrinsics)],
    }


def _cf_from_dict(d: dict) -> CoordinateFrame:
    intr = d["intrinsics"]
    return CoordinateFrame(
        name=d["name"],
        parent=d["parent"],
        pose=_pose_from_dict(d["pose"]),
        intrinsics=None if intr is None else np.array([[_dec_float(v) for v in row] for row in intr], dtype=float),
    )


def scene_ir_to_dict(s: SceneIR) -> dict:
    """The full IR as a JSON-able dict (the canonical form; grids are described by metadata only --
    their bytes ride alongside in Parquet / the hash)."""
    frames = [
        {
            "grid": _grid_meta(fr.grid),
            "ego": _ego_to_dict(fr.ego),
            "time": _enc_float(fr.time),
            "objects": [_box_to_dict(b) for b in fr.objects],
        }
        for fr in s.scene.frames
    ]
    prov = s.provenance
    return {
        "schema_version": SCHEMA_VERSION,
        "name": s.scene.name,
        "frames": frames,
        "coordinate_frames": [_cf_to_dict(cf) for cf in s.coordinate_frames],
        "tracks": [
            {
                "entity_id": tk.entity_id,
                "category": tk.category,
                "timestamps": [_enc_float(t) for t in tk.timestamps],
                "states": [_entity_to_dict(e) for e in tk.states],
            }
            for tk in s.tracks
        ],
        "observations": [
            {"sensor": o.sensor, "timestamp": _enc_float(o.timestamp), "frame_index": o.frame_index, "modality": o.modality}
            for o in s.observations
        ],
        "relations": [
            {"subject_id": r.subject_id, "relation": r.relation, "object_id": r.object_id, "frame_index": r.frame_index}
            for r in s.relations
        ],
        "events": [
            {"label": e.label, "start_frame": e.start_frame, "end_frame": e.end_frame, "entity_id": e.entity_id}
            for e in s.events
        ],
        "predictions": [
            {"frame_index": p.frame_index, "entities": [_entity_to_dict(e) for e in p.entities], "source": p.source}
            for p in s.predictions
        ],
        "ground_truth": [
            {"frame_index": gt.frame_index, "entities": [_entity_to_dict(e) for e in gt.entities]}
            for gt in s.ground_truth
        ],
        "failures": [
            {"frame_index": f.frame_index, "kind": f.kind, "note": f.note, "entity_id": f.entity_id}
            for f in s.failures
        ],
        "slices": [{"name": sl.name, "frame_indices": list(sl.frame_indices)} for sl in s.slices],
        "provenance": None
        if prov is None
        else {
            "dataset": prov.dataset,
            "log_id": prov.log_id,
            "adapter": prov.adapter,
            "schema_version": prov.schema_version,
            "commit": prov.commit,
        },
    }


def scene_ir_from_dict(d: dict, grids: list[np.ndarray]) -> SceneIR:
    """Rebuild a SceneIR from the canonical dict + the dense grids (one per frame, in order)."""
    if len(grids) != len(d["frames"]):
        raise ValueError(f"{len(grids)} grids != {len(d['frames'])} frames")
    frames = []
    for fr, occ in zip(d["frames"], grids):
        gm = fr["grid"]
        grid = OccupancyGrid(
            occupancy=occ,
            voxel_size=_dec_float(gm["voxel_size"]),
            origin=_dec_tuple(gm["origin"]),
            ground_height=_dec_float(gm["ground_height"]),
        )
        frames.append(
            Frame(
                grid=grid,
                ego=_ego_from_dict(fr["ego"]),
                time=_dec_float(fr["time"]),
                objects=tuple(_box_from_dict(b) for b in fr["objects"]),
            )
        )
    scene = Scene(tuple(frames), d["name"])
    prov = d.get("provenance")
    return SceneIR(
        scene=scene,
        coordinate_frames=tuple(_cf_from_dict(c) for c in d.get("coordinate_frames", [])),
        tracks=tuple(
            Track(
                entity_id=tk["entity_id"],
                category=tk["category"],
                timestamps=tuple(_dec_float(t) for t in tk["timestamps"]),
                states=tuple(_entity_from_dict(e) for e in tk["states"]),
            )
            for tk in d.get("tracks", [])
        ),
        observations=tuple(
            Observation(sensor=o["sensor"], timestamp=_dec_float(o["timestamp"]), frame_index=o["frame_index"], modality=o["modality"])
            for o in d.get("observations", [])
        ),
        relations=tuple(
            Relation(subject_id=r["subject_id"], relation=r["relation"], object_id=r["object_id"], frame_index=r["frame_index"])
            for r in d.get("relations", [])
        ),
        events=tuple(
            Event(label=e["label"], start_frame=e["start_frame"], end_frame=e["end_frame"], entity_id=e["entity_id"])
            for e in d.get("events", [])
        ),
        predictions=tuple(
            Prediction(frame_index=p["frame_index"], entities=tuple(_entity_from_dict(e) for e in p["entities"]), source=p["source"])
            for p in d.get("predictions", [])
        ),
        ground_truth=tuple(
            GroundTruth(frame_index=gt["frame_index"], entities=tuple(_entity_from_dict(e) for e in gt["entities"]))
            for gt in d.get("ground_truth", [])
        ),
        failures=tuple(
            Failure(frame_index=f["frame_index"], kind=f["kind"], note=f["note"], entity_id=f["entity_id"])
            for f in d.get("failures", [])
        ),
        slices=tuple(DatasetSlice(name=sl["name"], frame_indices=tuple(sl["frame_indices"])) for sl in d.get("slices", [])),
        provenance=None
        if prov is None
        else Provenance(
            dataset=prov["dataset"],
            log_id=prov["log_id"],
            adapter=prov["adapter"],
            schema_version=prov["schema_version"],
            commit=prov["commit"],
        ),
    )


# --- grid byte (de)serialization: exact, dtype + shape preserved ---
def _grid_bytes(g: OccupancyGrid) -> bytes:
    return np.ascontiguousarray(g.occupancy).tobytes()


def _grid_from_bytes(buf: bytes, shape: list[int], dtype: str) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(tuple(shape)).copy()


def to_parquet(scene_ir: SceneIR, path: str | pathlib.Path) -> pathlib.Path:
    """Write the IR to ONE Parquet file, losslessly. The relational layers go in file metadata as
    JSON; each frame's dense grid is one row of raw bytes."""
    d = scene_ir_to_dict(scene_ir)
    grid_buffers = [_grid_bytes(fr.grid) for fr in scene_ir.scene.frames]
    table = pa.table(
        {"frame_index": pa.array(range(len(grid_buffers)), type=pa.int64()),
         "grid_bytes": pa.array(grid_buffers, type=pa.large_binary())},
        metadata={_META_KEY: json.dumps(d, sort_keys=True).encode("utf-8")},
    )
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out


def from_parquet(path: str | pathlib.Path) -> SceneIR:
    """Read back a SceneIR written by `to_parquet`, reconstructing grids exactly."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    blob = meta.get(_META_KEY)
    if blob is None:
        raise ValueError(f"{path}: not a prism IR parquet (missing {_META_KEY!r} metadata)")
    d = json.loads(blob.decode("utf-8"))
    order = table.column("frame_index").to_pylist()
    raw = table.column("grid_bytes").to_pylist()
    by_index = dict(zip(order, raw))
    grids = []
    for i, fr in enumerate(d["frames"]):
        gm = fr["grid"]
        grids.append(_grid_from_bytes(by_index[i], gm["shape"], gm["dtype"]))
    return scene_ir_from_dict(d, grids)


def content_hash(scene_ir: SceneIR) -> str:
    """A sha256 over the canonical IR dict + every grid's raw bytes -- the load-bearing content
    identity. Two SceneIRs hash equal iff every field AND every voxel match (NaN-exact)."""
    h = hashlib.sha256()
    h.update(json.dumps(scene_ir_to_dict(scene_ir), sort_keys=True).encode("utf-8"))
    for fr in scene_ir.scene.frames:
        h.update(str(fr.grid.occupancy.dtype).encode("utf-8"))
        h.update(_grid_bytes(fr.grid))
    return h.hexdigest()


def to_openlabel_json(scene_ir: SceneIR) -> dict:
    """An OpenLABEL-style export: a top-level `openlabel` with metadata, a `frames` map keyed by
    frame index (each frame lists its object ids), and an `objects` map (id -> type + per-frame
    cuboid). Derived view for interchange, not the lossless source (only Parquet is)."""
    objects: dict[str, dict] = {}
    frames_out: dict[str, dict] = {}
    for i, fr in enumerate(scene_ir.scene.frames):
        frame_objs: dict[str, dict] = {}
        for j, box in enumerate(fr.objects):
            oid = f"{i}:{j}"
            objects.setdefault(oid, {"name": oid, "type": box.label})
            cx, cy, cz = box.center
            l, w, hgt = box.size
            qw, qx, qy, qz = math.cos(box.yaw / 2.0), 0.0, 0.0, math.sin(box.yaw / 2.0)
            frame_objs[oid] = {
                "object_data": {
                    "cuboid": [
                        {
                            "name": "shape",
                            "val": [
                                _enc_float(cx), _enc_float(cy), _enc_float(cz),
                                qx, qy, qz, qw,
                                _enc_float(l), _enc_float(w), _enc_float(hgt),
                            ],
                        }
                    ]
                }
            }
        frames_out[str(i)] = {
            "frame_properties": {"timestamp": _enc_float(fr.time)},
            "objects": frame_objs,
        }
    prov = scene_ir.provenance
    return {
        "openlabel": {
            "metadata": {
                "schema_version": "1.0.0",
                "prism_schema_version": SCHEMA_VERSION,
                "name": scene_ir.scene.name,
                "dataset": None if prov is None else prov.dataset,
                "log_id": None if prov is None else prov.log_id,
            },
            "frames": frames_out,
            "objects": objects,
            "coordinate_systems": {
                cf.name: {"type": "sensor" if cf.parent else "scene", "parent": cf.parent or ""}
                for cf in scene_ir.coordinate_frames
            },
        }
    }


def to_jsonl(scene_ir: SceneIR) -> str:
    """One JSON object per frame (timestamp + ego + objects), newline-separated -- a streaming dump."""
    d = scene_ir_to_dict(scene_ir)
    lines = []
    for i, fr in enumerate(d["frames"]):
        lines.append(json.dumps({"frame_index": i, "time": fr["time"], "ego": fr["ego"], "objects": fr["objects"]}, sort_keys=True))
    return "\n".join(lines)
