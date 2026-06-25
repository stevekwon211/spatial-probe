# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""PRISM Scene IR -- the sensor-agnostic source-of-truth intermediate representation.

`probe.scene` (Scene / Frame / EgoPose / TrackedBox) is the IR EMBRYO: a time sequence of
occupancy + ego + boxes. PRISM generalizes it ADDITIVELY into a representation that any spatial
dataset (Occ3D-nuScenes, AV2-Sensor, ...) lowers into, and that the query / serialize / validate
layers all read. Nothing here renames or breaks `probe`; the probe types stay importable and
unchanged, and `SceneIR` wraps a `probe.Scene` so every existing predicate keeps running.

The pieces, in order of leverage:

- `CoordinateFrame` -- THE calibration/projection object. A named frame (sensor / ego / world) with
  the SE3 transform that maps a point IN this frame TO its parent, plus optional camera intrinsics.
  Compose transforms to project across frames. This is what makes "where is this, in which frame,
  under which calibration" answerable instead of assumed. Built from the AV2 city_SE3_egovehicle
  feather (ego->world per timestamp).
- `Entity` -- generalizes `TrackedBox`: an identified object (id, category, pose, size, velocity)
  expressed in a named frame. `pose` is a full SE3 (position + quaternion), not just a yaw.
- `Track` -- one entity observed over time (an id with per-timestamp `Entity` states).
- `Observation` -- a raw sensor reading reference (a measurement, a frame token, a sensor name).
- `Relation` -- a typed relation between two entities (e.g. "a behind b").
- `Event` -- a temporal occurrence (a window [t0, t1] with a label).
- `Prediction` -- a model output (may be empty -- no predictions on disk yet).
- `GroundTruth` -- the curated truth boxes for a frame (wraps GT entities).
- `Failure` -- a recorded model/system failure tied to a frame/entity.
- `DatasetSlice` -- a named subset of frames (e.g. the REFERRED danger window).
- `Provenance` -- where this IR came from (dataset, log, commit, adapter, schema version).
- `SceneIR` -- the container: a `probe.Scene` (frames + occupancy + ego + boxes) PLUS the above.

Every dataclass is frozen + type-hinted. Equality is structural (the default for frozen
dataclasses) EXCEPT where a field holds a numpy array (`CoordinateFrame` intrinsics, `SceneIR`'s
embedded grids), where `eq=False` is used and content equality is delegated to `serialize`'s hash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from probe.scene import Frame, Scene, TrackedBox

__all__ = [
    "Pose",
    "CoordinateFrame",
    "Entity",
    "Track",
    "Observation",
    "Relation",
    "Event",
    "Prediction",
    "GroundTruth",
    "Failure",
    "DatasetSlice",
    "Provenance",
    "SceneIR",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = "prism-ir/1"


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """BEV yaw (rad about +z) from a (w, x, y, z) quaternion -- same convention as the av2 adapter."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion for a pure yaw rotation about +z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """3x3 rotation matrix from a (w, x, y, z) quaternion. Normalizes first (defensive)."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Pose:
    """A rigid 6-DoF pose: translation (x, y, z) + (w, x, y, z) unit quaternion, in a named frame.

    Generalizes the yaw-only pose `TrackedBox`/`EgoPose` carry. `from_yaw` builds one from a BEV
    yaw so an Occ3D/AV2 box (yaw-only) lifts losslessly; `yaw` reads the BEV heading back.
    """

    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_yaw(translation: tuple[float, float, float], yaw: float) -> "Pose":
        return Pose(translation=translation, quaternion=_yaw_to_quat(yaw))

    @property
    def yaw(self) -> float:
        w, x, y, z = self.quaternion
        return _quat_to_yaw(w, x, y, z)

    def rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return _quat_to_matrix(w, x, y, z)

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous transform: maps a point in this pose's local frame to its parent frame."""
        m = np.eye(4)
        m[:3, :3] = self.rotation_matrix()
        m[:3, 3] = np.asarray(self.translation, dtype=float)
        return m


@dataclass(frozen=True, eq=False)
class CoordinateFrame:
    """A named coordinate frame and the SE3 that maps a point IN this frame TO its `parent`.

    THE highest-value piece: it turns "which frame, under which calibration" from an assumption
    into a typed, composable object. `name` is e.g. 'lidar', 'ego', 'world'. `parent` is the frame
    `pose` maps into (None for a root frame like 'world'). `pose.matrix()` is the 4x4 transform.
    `intrinsics` is an optional 3x3 camera matrix (None for lidar/ego/world frames; AV2-Sensor
    here ships no per-sensor intrinsics on disk, so it stays None -- explicitly optional, never
    faked).

    `transform_points(pts)` applies the SE3 (this-frame -> parent). `inverse()` gives the reverse
    frame. `project(pts)` requires `intrinsics` and raises if absent (no silent fallback).
    """

    name: str
    parent: Optional[str] = None
    pose: Pose = field(default_factory=lambda: Pose((0.0, 0.0, 0.0)))
    intrinsics: Optional[np.ndarray] = None

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) points from THIS frame into `parent`: p_parent = R @ p + t."""
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        return p @ self.pose.rotation_matrix().T + np.asarray(self.pose.translation, dtype=float)

    def inverse(self) -> "CoordinateFrame":
        """The frame that maps parent -> this (swapped name/parent, inverted SE3)."""
        rot = self.pose.rotation_matrix()
        inv_t = -rot.T @ np.asarray(self.pose.translation, dtype=float)
        w, x, y, z = self.pose.quaternion
        inv_q = (w, -x, -y, -z)  # conjugate of a unit quaternion = inverse rotation
        return CoordinateFrame(
            name=self.parent or f"{self.name}_parent",
            parent=self.name,
            pose=Pose(translation=tuple(float(v) for v in inv_t), quaternion=inv_q),
            intrinsics=self.intrinsics,
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project (N, 3) camera-frame points to (N, 2) pixels via `intrinsics`. Raises if no
        intrinsics -- a frame with no camera matrix CANNOT project, and saying so loudly beats a
        silent wrong answer (repo rule: explicit failure over silent fallback)."""
        if self.intrinsics is None:
            raise ValueError(f"frame {self.name!r} has no intrinsics; cannot project to pixels")
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        uvw = p @ np.asarray(self.intrinsics, dtype=float).T
        z = uvw[:, 2:3]
        if np.any(z == 0.0):
            raise ValueError("cannot project a point with z=0 in the camera frame")
        return uvw[:, :2] / z


@dataclass(frozen=True)
class Entity:
    """An identified object -- the generalization of `TrackedBox`.

    `entity_id` is the persistent track id (a uuid in AV2, an instance token in nuScenes).
    `category` is the coarse class. `pose` is a full SE3 (position + orientation). `size` is
    (length, width, height) meters. `velocity` is (vx, vy) m/s (may be (nan, nan) when unknown --
    NEVER silently 0, matching the adapters). `frame` names the coordinate frame `pose`/`velocity`
    are expressed in. `to_tracked_box()` lowers back to the probe type (yaw-only) so the existing
    predicates run; `from_tracked_box()` lifts a probe box up.
    """

    entity_id: str
    category: str
    pose: Pose
    size: tuple[float, float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    frame: str = "ego"

    @staticmethod
    def from_tracked_box(box: TrackedBox, entity_id: str, frame: str = "ego") -> "Entity":
        return Entity(
            entity_id=entity_id,
            category=box.label,
            pose=Pose.from_yaw(box.center, box.yaw),
            size=box.size,
            velocity=box.velocity,
            frame=frame,
        )

    def to_tracked_box(self) -> TrackedBox:
        return TrackedBox(
            center=self.pose.translation,
            size=self.size,
            yaw=self.pose.yaw,
            label=self.category,
            velocity=self.velocity,
        )


@dataclass(frozen=True)
class Track:
    """One entity observed over time: an id + per-timestamp states (parallel lists, equal length)."""

    entity_id: str
    category: str
    timestamps: tuple[float, ...]
    states: tuple[Entity, ...]

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.states):
            raise ValueError(
                f"track {self.entity_id}: {len(self.timestamps)} timestamps != {len(self.states)} states"
            )


@dataclass(frozen=True)
class Observation:
    """A reference to a raw sensor reading: which sensor, when, and the frame index it backs."""

    sensor: str
    timestamp: float
    frame_index: int
    modality: str = "lidar"


@dataclass(frozen=True)
class Relation:
    """A typed relation between two entities, optionally scoped to a frame index."""

    subject_id: str
    relation: str
    object_id: str
    frame_index: Optional[int] = None


@dataclass(frozen=True)
class Event:
    """A temporal occurrence over a frame window [start_frame, end_frame], with a label."""

    label: str
    start_frame: int
    end_frame: int
    entity_id: Optional[str] = None


@dataclass(frozen=True)
class Prediction:
    """A model output for one frame: predicted entities + a free-form source tag.

    Typically empty here -- there are no predictions on disk -- but the slot exists so a model's
    output lowers into the SAME IR as ground truth and the two are comparable by construction.
    """

    frame_index: int
    entities: tuple[Entity, ...] = ()
    source: str = ""


@dataclass(frozen=True)
class GroundTruth:
    """The curated truth entities for one frame -- the GT boxes, wrapped as Entities."""

    frame_index: int
    entities: tuple[Entity, ...] = ()


@dataclass(frozen=True)
class Failure:
    """A recorded model/system failure: a frame, an optional entity, a kind, and a note."""

    frame_index: int
    kind: str
    note: str = ""
    entity_id: Optional[str] = None


@dataclass(frozen=True)
class DatasetSlice:
    """A named subset of frame indices (e.g. the REFERRED danger window)."""

    name: str
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class Provenance:
    """Where this IR came from -- dataset + log + adapter + the schema version + an optional commit.

    Makes a serialized IR self-describing: who produced it, from what, under which schema. This is
    the 'honest instrument' audit trail at the data level.
    """

    dataset: str
    log_id: str
    adapter: str = ""
    schema_version: str = SCHEMA_VERSION
    commit: str = ""


@dataclass(frozen=True, eq=False)
class SceneIR:
    """The PRISM container: a `probe.Scene` PLUS the sensor-agnostic IR layers.

    `scene` carries the frames (occupancy + ego + the lowered boxes), so EVERY existing probe
    predicate / retrieval / metric runs on `scene_ir.scene` with no change. The added layers
    (`coordinate_frames`, `tracks`, `observations`, ground truth, predictions, ...) are the
    generalization. `eq=False` because the embedded numpy occupancy grids make structural equality
    ill-defined; content equality is tested via `serialize.content_hash`.
    """

    scene: Scene
    coordinate_frames: tuple[CoordinateFrame, ...] = ()
    tracks: tuple[Track, ...] = ()
    observations: tuple[Observation, ...] = ()
    relations: tuple[Relation, ...] = ()
    events: tuple[Event, ...] = ()
    predictions: tuple[Prediction, ...] = ()
    ground_truth: tuple[GroundTruth, ...] = ()
    failures: tuple[Failure, ...] = ()
    slices: tuple[DatasetSlice, ...] = ()
    provenance: Optional[Provenance] = None

    @property
    def name(self) -> str:
        return self.scene.name

    @property
    def frames(self) -> tuple[Frame, ...]:
        return self.scene.frames

    def __len__(self) -> int:
        return len(self.scene.frames)

    def frame_for(self, name: str) -> Optional[CoordinateFrame]:
        """The coordinate frame with this name, or None."""
        for cf in self.coordinate_frames:
            if cf.name == name:
                return cf
        return None

    def with_scene(self, scene: Scene) -> "SceneIR":
        return replace(self, scene=scene)
