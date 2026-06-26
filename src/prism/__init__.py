# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""PRISM -- a sensor-agnostic Scene IR + CLI layered ADDITIVELY on the `probe` core.

`probe` (occupancy grid + falsifiable predicates) stays the substrate; PRISM adds the
representation that any dataset lowers into (`ir`), lossless (de)serialization (`serialize`), the
honest-instrument calibration/timestamp validator (`validate`), the ingest path that reuses the
probe adapters (`adapt`), and the `prism` CLI (`cli`).
"""
from __future__ import annotations

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
from prism.detect import (
    COCO_NAMES,
    COCO_TO_AV2,
    Detection,
    detect_image,
    detect_log,
    detections_to_prediction,
)
from prism.failure import (
    SIGNATURES,
    AV2CameraCalib,
    FailureCandidate,
    FailureCluster,
    Signature,
    cluster,
    find,
    load_av2_camera,
    match_detections,
    mine,
    project_ego_boxes,
    resolve_signature,
    signature_for_query,
    similar_frames,
)
from prism.predicates import object_speed, occluded, ttc, velocity
from prism.query import QueryResult, query
from prism.serialize import (
    content_hash,
    from_parquet,
    to_jsonl,
    to_openlabel_json,
    to_parquet,
)
from prism.validate import ValidationError, validate_scene

__all__ = [
    "SCHEMA_VERSION",
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
    "to_parquet",
    "from_parquet",
    "to_openlabel_json",
    "to_jsonl",
    "content_hash",
    "validate_scene",
    "ValidationError",
    "query",
    "QueryResult",
    "occluded",
    "object_speed",
    "ttc",
    "velocity",
    "SIGNATURES",
    "Signature",
    "FailureCandidate",
    "FailureCluster",
    "mine",
    "cluster",
    "similar_frames",
    "find",
    "resolve_signature",
    "signature_for_query",
    "Detection",
    "COCO_NAMES",
    "COCO_TO_AV2",
    "detect_image",
    "detect_log",
    "detections_to_prediction",
    "AV2CameraCalib",
    "load_av2_camera",
    "project_ego_boxes",
    "match_detections",
]
