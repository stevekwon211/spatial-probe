# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S5b -- failure-taxonomy mining over a Scene IR (declarative signatures, registry, clustering).

THE honest framing (do not re-inflate; this matches the repo CLAUDE.md H1/H3 result discipline):
there are NO model predictions on disk, so model-eval failures (missed-by-model, TTC danger) are
NOT buildable here. The one REAL, prediction-free failure signal is the H1 occupancy-vs-box SET
DIFFERENCE -- two complementary "the two backends disagree" signatures, each carrying the oracle
that (does or does NOT) externally validate it:

- `path_blocked_no_box` -- occupancy blocks the ego in-path band AND no tracked box within R meters
  explains it. Either occupancy caught a REAL unboxed obstacle (the H1 expressivity win a box-only
  language is structurally blind to) or it is an occupancy false positive. The FP DIRECTION is the
  one externally anchored signal here: the ego-trajectory traversal oracle (v0.1, sealed) is
  RELIABLE -- occupancy does NOT hallucinate obstacles on the physically-driven path (true FP 0.000
  vs shuffled 0.036 on held-out free-driving). So a block-with-no-box is much more likely a real
  unboxed obstacle than a hallucination. honesty tag: external-fp.

- `box_in_free` -- a tracked box that LiDAR actually saw (num_interior_pts >= N) whose footprint
  occupancy marks FREE. A box-recall miss CANDIDATE. honesty tag: consistency-only. The box-recall
  oracle is RECALL-SUPPORTED but SAME-MODALITY (it gates on num_interior_pts, the SAME LiDAR the
  voxelizer reads), and the externally-independent recall route is honestly CLOSED on this substrate
  (classical stereo died on density AUC 0.259; frozen DAv2-VKITTI died on metric scale > 9 m). So
  this is consistency, NOT external truth -- and the absolute level is inflated by the `_ROAD_Z=0.3`
  floor-straddle confound (boxes whose returns sit on wheels/lower body below the road slab read as
  misses). The RELATIVE box-recall gap survives that confound; an absolute miss count does not.

A signature is a declarative, registered object (name + aliases + honesty + an `evaluate` over a
SceneIR frame), so adding a third is a registry entry, not a new code path. `mine` runs a signature
over logs; `cluster` groups the resulting candidates by a numpy feature vector into `DatasetSlice`s;
`similar_frames` ranks candidates by feature-vector distance to a cluster centroid. ALL similarity
here is FEATURE-DISTANCE, never a learned/semantic embedding -- said plainly so it is not oversold.

Everything reuses the existing predicates verbatim: `free_along_ego_path` (occupancy blockage),
`distance_to_nearest_object` (box explanation), `OccupancyGrid.world_to_voxel` + the AV2 voxelizer's
`_ROAD_Z`/`_Z1` slab (box footprint coverage). Nothing re-reads occupancy from raw sensors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from probe.grid import OCCUPIED, OccupancyGrid
from probe.predicates.freepath import free_along_ego_path
from probe.predicates.objects import distance_to_nearest_object
from prism.ir import DatasetSlice, Failure, SceneIR

__all__ = [
    "FailureCandidate",
    "Signature",
    "FailureCluster",
    "RankedFrame",
    "SIGNATURES",
    "register_signature",
    "resolve_signature",
    "signature_for_query",
    "mine",
    "cluster",
    "similar_frames",
    "find",
    "AV2CameraCalib",
    "load_av2_camera",
    "project_ego_boxes",
    "match_detections",
]


# === candidate + signature types =================================================================


@dataclass(frozen=True)
class FailureCandidate:
    """One mined failure: where it is, which signature flagged it, how honest the label is, and the
    numpy-friendly feature vector clustering/similarity read.

    `features` is a flat dict of floats (forward_range_m, clearance_m, category_code, signature_code,
    ...). `honesty` names the oracle (if any) that externally validates this signature's DIRECTION --
    NOT a per-instance verification (each instance is a candidate, not a confirmed failure).
    `entity_slot` is the per-frame box index for box-side signatures (None for path-side)."""

    log_id: str
    frame_index: int
    signature: str
    honesty: str
    note: str
    features: dict[str, float]
    entity_slot: Optional[int] = None

    def to_ir_failure(self) -> Failure:
        """Lower to the frozen IR `Failure` so a mined candidate can be stamped onto a SceneIR.
        The IR type is intentionally lean (frame + kind + note); the rich features live here."""
        eid = None if self.entity_slot is None else f"slot#{self.entity_slot}"
        return Failure(frame_index=self.frame_index, kind=self.signature, note=self.note, entity_id=eid)


# The feature keys every signature emits, in a FIXED order, so feature vectors are comparable across
# signatures (a missing key for a given signature is filled with 0.0 -- explicit, not silent).
_FEATURE_KEYS = ("signature_code", "forward_range_m", "clearance_m", "category_code", "frame_index")


def _feature_vector(features: dict[str, float]) -> np.ndarray:
    """Flat numpy vector in the fixed _FEATURE_KEYS order. inf/nan are mapped to a large finite
    sentinel so distances stay finite and deterministic (clustering must not choke on an open range)."""
    vals = []
    for k in _FEATURE_KEYS:
        v = float(features.get(k, 0.0))
        if not math.isfinite(v):
            v = 1e6 if v > 0 else -1e6
        vals.append(v)
    return np.asarray(vals, dtype=float)


EvaluateFn = Callable[[SceneIR, int, str, dict], list[FailureCandidate]]


@dataclass(frozen=True)
class Signature:
    """A declarative failure signature: a name, search aliases, an honesty tag, and an `evaluate`
    that returns the candidates a single frame yields. Registered in SIGNATURES; never hardcoded
    into the miner -- `mine` only knows `Signature`, so a new signature is a registry entry."""

    name: str
    aliases: tuple[str, ...]
    honesty: str
    code: float  # a stable numeric id for the feature vector (so signatures separate in cluster space)
    evaluate: EvaluateFn
    description: str = ""


SIGNATURES: dict[str, Signature] = {}


def register_signature(sig: Signature) -> Signature:
    if sig.name in SIGNATURES:
        raise ValueError(f"signature {sig.name!r} already registered")
    SIGNATURES[sig.name] = sig
    return sig


def resolve_signature(key: str) -> Signature:
    """A signature by exact name OR by any registered alias (case/space/dash-insensitive)."""
    if key in SIGNATURES:
        return SIGNATURES[key]
    norm = _norm(key)
    for sig in SIGNATURES.values():
        if norm == _norm(sig.name) or any(norm == _norm(a) for a in sig.aliases):
            return sig
    raise ValueError(f"unknown signature {key!r}; known: {sorted(SIGNATURES)}")


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# === the natural-ish query -> signature keyword map (S6) =========================================
# A small, explicit keyword/alias map (NOT an LLM). A query string scores against each signature's
# keyword bag; the top score wins. Honest: this is keyword routing, not language understanding.

_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "path_blocked_no_box": (
        "path", "blocked", "block", "no", "tracked", "object", "box", "explain", "unboxed",
        "obstacle", "nothing", "without",
    ),
    "box_in_free": (
        "recall", "box", "free", "occupancy", "lidar", "lost",
    ),
    "missed_detection": (
        "missed", "miss", "detector", "detection", "model", "detect", "see", "seen", "saw",
        "recall", "vehicle", "pedestrian", "car", "person", "object",
    ),
}


def signature_for_query(query: str) -> Signature:
    """Map a natural-ish query string to a signature by keyword overlap. Raises if nothing scores
    (a query the corpus cannot answer fails loudly, never silently picks a default)."""
    q = set(_norm_tokens(query))
    best_name, best_score = None, 0
    for name, kws in _QUERY_KEYWORDS.items():
        # weight discriminative tokens: "blocked"/"unboxed" -> path side, "missed"/"recall" -> box side
        score = sum(1 for kw in kws if kw in q)
        if score > best_score:
            best_name, best_score = name, score
    if best_name is None or best_score == 0:
        raise ValueError(
            f"no signature matches query {query!r}; try 'path blocked but no tracked object' "
            f"(path_blocked_no_box) or 'a box the occupancy missed' (box_in_free)"
        )
    # tie-break toward the more specific path side when both fire equally on generic 'box/object'
    return SIGNATURES[best_name]


def _norm_tokens(s: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in s).split() if t]


# === signature implementations ===================================================================

_CATEGORY_CODES = {
    "vehicle": 1.0, "pedestrian": 2.0, "bicycle": 3.0, "motorcycle": 4.0, "other": 5.0,
}


def _category_code(label: str) -> float:
    return _CATEGORY_CODES.get(label, 5.0)


def _eval_path_blocked_no_box(ir: SceneIR, t: int, honesty: str, params: dict) -> list[FailureCandidate]:
    """Occupancy blocks the ego in-path band AND no tracked box within `box_radius_m` explains it.

    Reuses `free_along_ego_path` (the C-space corridor blockage predicate, min_cluster_voxels=2 so a
    lone noise voxel never reads as a block) for the occupancy side, and `distance_to_nearest_object`
    (the box-only baseline) for the explanation side. The "forward range" feature is the forward
    distance to the nearest in-corridor obstacle surface (the block depth)."""
    sc = ir.scene
    horizon = float(params.get("horizon", 1.0))
    box_radius = float(params.get("box_radius_m", 5.0))
    grid = sc.grid_at(t)
    ego = sc.ego_at(t)
    blocked = not free_along_ego_path(grid, ego, horizon, min_cluster_voxels=2)
    if not blocked:
        return []
    # the nearest tracked box (any class) -- if one is within box_radius, the block is EXPLAINED.
    nearest_box = distance_to_nearest_object(sc, t)
    if nearest_box <= box_radius:
        return []
    fwd_range = _forward_block_range(grid, ego)
    feats = {
        "signature_code": SIGNATURES["path_blocked_no_box"].code,
        "forward_range_m": fwd_range,
        "clearance_m": fwd_range,  # the block depth doubles as the "how close" feature on this side
        "category_code": 0.0,  # no box involved
        "frame_index": float(t),
    }
    note = (
        f"occupancy blocks the in-path band at ~{fwd_range:.1f} m forward; nearest tracked box is "
        f"{nearest_box:.1f} m away (> {box_radius:.1f} m) -> unboxed-obstacle / FP candidate"
    )
    return [FailureCandidate(
        log_id=_log_id(ir), frame_index=t, signature="path_blocked_no_box",
        honesty=honesty, note=note, features=feats,
    )]


def _forward_block_range(grid: OccupancyGrid, ego) -> float:
    """Forward distance (m) from the ego to the nearest obstacle voxel center inside the ego-width
    corridor ahead. inf if the corridor scan finds none (defensive; blocked implies one exists, but
    the inflated C-space block can sit just off the raw centerline)."""
    centers = grid.obstacle_centers(max_height_agl=ego.height)
    if len(centers) == 0:
        return math.inf
    fwd, lat = ego.to_ego_frame(centers[:, :2])
    half = ego.width / 2.0 + grid.voxel_size
    band = (fwd > 0.0) & (np.abs(lat) <= half)
    if not band.any():
        return math.inf
    return float(np.min(fwd[band]))


def _eval_box_in_free(ir: SceneIR, t: int, honesty: str, params: dict) -> list[FailureCandidate]:
    """A LiDAR-seen (num_interior_pts >= N) tracked box whose above-road footprint occupancy marks
    FREE -> a recall-miss CONSISTENCY candidate.

    The interior-pts gate is read from an OPTIONAL side channel `params['interior_pts']` keyed by
    (frame_index, box_slot) -- because the probe `TrackedBox` does not carry num_interior_pts and the
    IR is left UNTOUCHED. On a real AV2 corpus `mine` fills that side channel from the annotations
    feather; on a synthetic SceneIR the test passes it directly. If NO side channel is given, the
    gate is treated as permissive (every box passes) so the FOOTPRINT mechanics are still testable;
    a real recall claim must supply interior_pts (else it is not gated to LiDAR-seen objects)."""
    sc = ir.scene
    n_min = int(params.get("n_interior_min", 5))
    pts_lookup = params.get("interior_pts")  # {(frame_index, slot): num_interior_pts} or None
    grid = sc.grid_at(t)
    out: list[FailureCandidate] = []
    for slot, box in enumerate(sc.objects_at(t)):
        if pts_lookup is not None:
            pts = pts_lookup.get((t, slot))
            if pts is None or pts < n_min:
                continue  # below the LiDAR-seen gate -> occupancy can't be blamed for not marking it
        covered = _box_footprint_covered(grid, box)
        if covered:
            continue  # occupancy puts mass on the box -> it recalls the object, NOT a miss
        ex, ey = sc.ego_at(t).position[0], sc.ego_at(t).position[1]
        rng = math.hypot(box.center[0] - ex, box.center[1] - ey)
        feats = {
            "signature_code": SIGNATURES["box_in_free"].code,
            "forward_range_m": rng,
            "clearance_m": 0.0,
            "category_code": _category_code(box.label),
            "frame_index": float(t),
        }
        note = (
            f"{box.label} box at {rng:.1f} m (LiDAR-seen, >= {n_min} pts) sits in occupancy-FREE "
            f"space -> recall-miss CONSISTENCY candidate (same-modality; floor-straddle inflates this)"
        )
        out.append(FailureCandidate(
            log_id=_log_id(ir), frame_index=t, signature="box_in_free",
            honesty=honesty, note=note, features=feats, entity_slot=slot,
        ))
    return out


# Box-footprint coverage reuses the AV2 voxelizer's above-road slab bounds so "covered" means the
# SAME above-road sub-volume the box-recall oracle credits -- not a different, drifting definition.
def _box_footprint_covered(grid: OccupancyGrid, box) -> bool:
    """True iff ANY occupied voxel lies inside the box's above-road footprint (a BEV oriented-rect x
    a z-slab). Mirrors the box-recall oracle's admissibility: a voxel center inside the rect, above
    the road floor, below the grid ceiling."""
    from probe.adapters.av2_sensor import _ROAD_Z  # the exact above-road floor the voxelizer uses

    cx, cy, cz = box.center
    length, width, height = box.size
    occ = grid.occupancy
    nx, ny, nz = occ.shape
    res = grid.voxel_size
    ox, oy, oz = grid.origin
    # candidate voxel index window around the box center (a generous AABB, then the oriented-rect test)
    reach = max(length, width) / 2.0 + res
    i0 = max(0, int(math.floor((cx - reach - ox) / res)))
    i1 = min(nx - 1, int(math.ceil((cx + reach - ox) / res)))
    j0 = max(0, int(math.floor((cy - reach - oy) / res)))
    j1 = min(ny - 1, int(math.ceil((cy + reach - oy) / res)))
    if i0 > i1 or j0 > j1:
        return False
    # z-slab: voxel centers in [max(cz-h/2, road), min(cz+h/2, ceil)] (above-road admissibility)
    z_lo = max(cz - height / 2.0, _ROAD_Z)
    z_hi = min(cz + height / 2.0, oz + (nz - 1) * res)
    cosy, siny = math.cos(box.yaw), math.sin(box.yaw)
    hl, hw = length / 2.0, width / 2.0
    for i in range(i0, i1 + 1):
        wx = ox + i * res
        for j in range(j0, j1 + 1):
            wy = oy + j * res
            # rotate (wx,wy) into the box frame; inside iff |u|<=hl and |v|<=hw
            dx, dy = wx - cx, wy - cy
            u = dx * cosy + dy * siny
            v = -dx * siny + dy * cosy
            if abs(u) > hl or abs(v) > hw:
                continue
            for k in range(nz):
                wz = oz + k * res
                if wz < z_lo or wz > z_hi or wz <= _ROAD_Z:
                    continue
                if occ[i, j, k] == OCCUPIED:
                    return True
    return False


def _log_id(ir: SceneIR) -> str:
    if ir.provenance is not None and ir.provenance.log_id:
        return ir.provenance.log_id
    return ir.name


# === missed_detection: 2D-detector recall vs AV2 GT boxes (REAL model-eval) ======================
# The first prediction-backed signature. A GT 3D box is projected into the camera image; if it is
# visible (in front + in FOV) AND the detector output NO matching detection, the detector failed to
# see a labeled object = a missed detection (Ramanagopal-style). This is detector-eval recall, NOT an
# occupancy claim and NOT in-domain (the detector is COCO-pretrained YOLOv8n, not trained on AV2).
#
# Calibration + projection reuse the MATH proven in experiments/occquery_v0 (projection.quat_to_rotmat
# self-check + oracle_stereo_recall.project_ego_to_left with the AV2 Su 3-coeff radial model). The
# code is copied here (not imported) so src/prism stays independent of experiments/. AV2 boxes are
# already ego-frame, so projection is ego->camera (sensor extrinsic) -> distort -> pixel; no ego pose.


@dataclass(frozen=True)
class AV2CameraCalib:
    """One AV2 camera's calibration (intrinsics + Su radial distortion + sensor->ego extrinsic).

    Camera frame is x-right, y-down, z-forward (optical axis): a point is visible iff z>0; pixel =
    distort(x/z, y/z) then *f + c. `R_cam2ego` maps camera axes -> ego axes; `t_cam_in_ego` is the
    camera origin in ego. p_cam = R_cam2ego.T @ (p_ego - t_cam_in_ego) -- the exact inverse used in
    the experiments' projection self-check."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    width: int
    height: int
    R_cam2ego: np.ndarray
    t_cam_in_ego: np.ndarray


def _quat_to_rotmat(q) -> np.ndarray:
    """(w,x,y,z) unit quaternion -> 3x3 rotation matrix (full 3D; cameras pitch/roll)."""
    w, x, y, z = np.asarray(q, dtype=float)
    n = np.linalg.norm([w, x, y, z])
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = np.array([w, x, y, z]) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_av2_camera(log_root, camera: str = "ring_front_center") -> Optional[AV2CameraCalib]:
    """Read one AV2 camera's calibration from a log's calibration/ feathers. Returns None if the log
    has no calibration dir (so a non-camera log degrades to 'no missed-detection signal', honestly)."""
    import pathlib

    root = pathlib.Path(log_root) / "calibration"
    intr_p, ext_p = root / "intrinsics.feather", root / "egovehicle_SE3_sensor.feather"
    if not intr_p.exists() or not ext_p.exists():
        return None
    import pyarrow as pa

    def _read(p):
        return pa.ipc.open_file(pa.memory_map(str(p), "r")).read_all().to_pydict()

    intr, ext = _read(intr_p), _read(ext_p)
    if camera not in intr["sensor_name"] or camera not in ext["sensor_name"]:
        return None
    ii = intr["sensor_name"].index(camera)
    ei = ext["sensor_name"].index(camera)
    q = (ext["qw"][ei], ext["qx"][ei], ext["qy"][ei], ext["qz"][ei])
    return AV2CameraCalib(
        name=camera,
        fx=float(intr["fx_px"][ii]), fy=float(intr["fy_px"][ii]),
        cx=float(intr["cx_px"][ii]), cy=float(intr["cy_px"][ii]),
        k1=float(intr["k1"][ii]), k2=float(intr["k2"][ii]), k3=float(intr["k3"][ii]),
        width=int(intr["width_px"][ii]), height=int(intr["height_px"][ii]),
        R_cam2ego=_quat_to_rotmat(q),
        t_cam_in_ego=np.array([ext["tx_m"][ei], ext["ty_m"][ei], ext["tz_m"][ei]], dtype=float),
    )


def _project_ego_points(points_ego: np.ndarray, cam: AV2CameraCalib):
    """Ego points (N,3) -> (uv pixels, depth, visible). Su 3-coeff radial distortion forward model.
    visible iff in front (depth>0) AND inside the raw image. Mirrors oracle_stereo_recall.project_ego_to_left."""
    p = np.atleast_2d(np.asarray(points_ego, dtype=float))
    p_cam = (cam.R_cam2ego.T @ (p - cam.t_cam_in_ego).T).T
    depth = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = p_cam[:, 0] / p_cam[:, 2]
        yn = p_cam[:, 1] / p_cam[:, 2]
        r2 = xn * xn + yn * yn
        s = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2 + cam.k3 * r2 * r2 * r2
        u = xn * s * cam.fx + cam.cx
        v = yn * s * cam.fy + cam.cy
    front = depth > 1e-6
    inframe = front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    uv = np.stack([u, v], axis=-1)
    uv = np.where(front[:, None], uv, np.nan)
    return uv, depth, inframe


def _box_corners_ego(box) -> np.ndarray:
    """The 8 ego-frame corners of a TrackedBox (BEV-yawed cuboid). (8,3)."""
    cx, cy, cz = box.center
    length, width, height = box.size
    hl, hw, hh = length / 2.0, width / 2.0, height / 2.0
    cosy, siny = math.cos(box.yaw), math.sin(box.yaw)
    corners = []
    for sx in (-hl, hl):
        for sy in (-hw, hw):
            for sz in (-hh, hh):
                wx = cx + sx * cosy - sy * siny
                wy = cy + sx * siny + sy * cosy
                corners.append((wx, wy, cz + sz))
    return np.asarray(corners, dtype=float)


def project_ego_boxes(boxes, cam: AV2CameraCalib, *, min_corners: int = 4):
    """Project each ego-frame box's 8 corners; return per-box (visible, box2d_xyxy, depth).

    A box is `visible` iff >= `min_corners` of its 8 corners are in front AND in frame (so a box must
    be substantially inside the image, not clipped by a single grazing corner). box2d_xyxy is the
    axis-aligned hull of the in-front projected corners (clipped to the image); depth is the box
    center's camera-frame depth. Returns a list aligned to `boxes`."""
    out = []
    for box in boxes:
        corners = _box_corners_ego(box)
        uv, depth, vis = _project_ego_points(corners, cam)
        n_vis = int(vis.sum())
        cuv, cdepth, _ = _project_ego_points(np.asarray([box.center], dtype=float), cam)
        center_depth = float(cdepth[0])
        if n_vis < min_corners:
            out.append((False, None, center_depth))
            continue
        front = depth > 1e-6
        fu = uv[front]
        x0 = float(np.clip(fu[:, 0].min(), 0, cam.width))
        y0 = float(np.clip(fu[:, 1].min(), 0, cam.height))
        x1 = float(np.clip(fu[:, 0].max(), 0, cam.width))
        y1 = float(np.clip(fu[:, 1].max(), 0, cam.height))
        out.append((True, (x0, y0, x1, y1), center_depth))
    return out


def _iou_xyxy(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / (area_a + area_b - inter + 1e-9)


def _center_in_box(center, box2d) -> bool:
    u, v = center
    x0, y0, x1, y1 = box2d
    return bool(x0 <= u <= x1 and y0 <= v <= y1)


def match_detections(gt_box2d: tuple, gt_label: str, detections, *, iou_thr: float = 0.3,
                     score_thr: float = 0.25, class_agnostic: bool = False) -> bool:
    """True iff SOME detection matches this GT 2D box: (IoU > iou_thr OR detection center inside the
    GT box) AND (class-agnostic OR the detection's mapped AV2 label == GT label) AND score >= thr.

    `detections` are `prism.detect.Detection` (duck-typed: needs .box_xyxy, .center, .av2_label,
    .score). Center-in-box is included alongside IoU because a 2D detector box and a 3D-box hull
    rarely overlap tightly, so center-containment catches a correct detection IoU alone would miss."""
    for d in detections:
        if d.score < score_thr:
            continue
        if not class_agnostic and d.av2_label != gt_label:
            continue
        if _iou_xyxy(gt_box2d, d.box_xyxy) > iou_thr or _center_in_box(d.center, gt_box2d):
            return True
    return False


def _eval_missed_detection(ir: SceneIR, t: int, honesty: str, params: dict) -> list[FailureCandidate]:
    """A GT box visible in the camera with NO matching detection above threshold = a missed detection.

    Detections are supplied per-frame via `params['detections']` keyed by frame_index:
    {frame_index: [Detection, ...]}. `mine` auto-fills this for a real AV2 log by running the detector
    on the nearest camera image per LiDAR frame (lazy, optional [detect] extra); a synthetic test
    passes it directly. The camera calibration rides in `params['camera_calib']` (an AV2CameraCalib);
    `mine` builds it from the log. If no calibration is available the signature is SILENT (no camera
    -> no detector-eval signal -- honest, never a faked miss).

    `min_range_m` / `max_range_m` gate by box depth (a box at 0.5 m or 120 m is not a fair recall
    target). Only GT boxes whose coarse label has a COCO counterpart are considered (a 'BOLLARD' the
    detector was never trained to find is not a model failure)."""
    cam = params.get("camera_calib")
    if cam is None:
        return []
    sc = ir.scene
    dets_by_frame = params.get("detections") or {}
    detections = dets_by_frame.get(t, [])
    iou_thr = float(params.get("iou_thr", 0.3))
    score_thr = float(params.get("score_thr", 0.25))
    class_agnostic = bool(params.get("class_agnostic", False))
    min_range = float(params.get("min_range_m", 2.0))
    max_range = float(params.get("max_range_m", 80.0))
    min_corners = int(params.get("min_corners", 4))
    target_labels = params.get("target_labels", _DETECTABLE_LABELS)
    occluded_lookup = params.get("occluded")  # optional {(frame, slot): bool} (compose w/ occluded predicate)

    boxes = sc.objects_at(t)
    projected = project_ego_boxes(boxes, cam, min_corners=min_corners)
    out: list[FailureCandidate] = []
    for slot, (box, (visible, box2d, depth)) in enumerate(zip(boxes, projected)):
        if box.label not in target_labels:
            continue  # the detector was never trained to find this class -> not its failure
        if not visible or box2d is None:
            continue  # the camera could not see it -> a miss here would be a projection artifact
        if not (min_range <= depth <= max_range):
            continue
        if match_detections(box2d, box.label, detections, iou_thr=iou_thr, score_thr=score_thr,
                            class_agnostic=class_agnostic):
            continue  # the detector saw it -> NOT a miss
        is_occ = None if occluded_lookup is None else bool(occluded_lookup.get((t, slot), False))
        feats = {
            "signature_code": SIGNATURES["missed_detection"].code,
            "forward_range_m": float(depth),
            "clearance_m": float(depth),
            "category_code": _category_code(box.label),
            "frame_index": float(t),
        }
        occ_tag = "" if is_occ is None else (" [occluded]" if is_occ else " [visible/unoccluded]")
        note = (
            f"{box.label} GT box at {depth:.1f} m projects into the camera (2D box "
            f"[{box2d[0]:.0f},{box2d[1]:.0f},{box2d[2]:.0f},{box2d[3]:.0f}]) but the detector output "
            f"no matching detection (>= {score_thr:.2f}){occ_tag} -> MISSED DETECTION (detector recall failure)"
        )
        out.append(FailureCandidate(
            log_id=_log_id(ir), frame_index=t, signature="missed_detection",
            honesty=honesty, note=note, features=feats, entity_slot=slot,
        ))
    return out


# AV2 coarse labels the COCO YOLOv8n CAN in principle detect (have a COCO counterpart). A class the
# detector was never trained on (BOLLARD, SIGN, CONSTRUCTION_CONE -> 'other') is excluded so a miss is
# a real recall failure, not a missing-class artifact.
_DETECTABLE_LABELS = frozenset({"pedestrian", "bicycle", "vehicle", "motorcycle"})


# Register the built-in signatures (declarative; the miner never names them directly).
register_signature(Signature(
    name="path_blocked_no_box",
    aliases=("path-blocked", "blocked-no-box", "unboxed-obstacle", "fp-candidate", "path blocked no object"),
    honesty="external-fp: the traversal-v0.1 oracle (sealed, RELIABLE) found occupancy does NOT "
            "hallucinate obstacles on the driven path -> a block-with-no-box is most likely a REAL "
            "unboxed obstacle a box-only language is blind to (H1 win), not a hallucination.",
    code=1.0,
    evaluate=lambda ir, t, h, p: _eval_path_blocked_no_box(ir, t, h, p),
    description="Occupancy blocks the ego in-path band where no tracked box within R m explains it.",
))
register_signature(Signature(
    name="box_in_free",
    aliases=("recall-miss", "box-recall", "missed-box", "box in free", "occupancy missed"),
    honesty="consistency-only: the box-recall oracle is RECALL-SUPPORTED but SAME-MODALITY (gates on "
            "num_interior_pts, the same LiDAR the voxelizer reads); externally-independent recall is "
            "honestly CLOSED on this substrate (stereo AUC 0.259, DAv2 scale > 9 m). NOT external "
            "truth; the absolute count is inflated by the _ROAD_Z floor-straddle confound.",
    code=2.0,
    evaluate=lambda ir, t, h, p: _eval_box_in_free(ir, t, h, p),
    description="A LiDAR-seen (>=N interior pts) box whose above-road footprint occupancy marks FREE.",
))
register_signature(Signature(
    name="missed_detection",
    aliases=(
        "missed-detection", "missed by the model", "missed by the detector", "detector miss",
        "model miss", "detection recall", "missed object", "detector recall",
    ),
    honesty="model-eval: COCO-YOLOv8n detector recall vs AV2 GT boxes; a miss = the detector failed "
            "to see a labeled object (Ramanagopal-style missed-detection); NOT an occupancy claim. "
            "Caveats: the detector is COCO-pretrained (NOT trained on AV2) so this is "
            "cross-distribution recall, not an in-domain benchmark; the COCO->AV2 class map is coarse "
            "(car/bus/truck -> vehicle); only camera-visible, in-range, COCO-detectable GT classes "
            "are scored; the ego-hood detector FP is filtered.",
    code=3.0,
    evaluate=lambda ir, t, h, p: _eval_missed_detection(ir, t, h, p),
    description="A camera-visible GT box the 2D detector failed to output a matching detection for.",
))


# === mining ======================================================================================


def mine(logs, signature, *, params: Optional[dict] = None) -> list[FailureCandidate]:
    """Run `signature` over every frame of every SceneIR in `logs`, returning all candidates.

    `signature` is a name/alias or a `Signature`. `params` tunes the signature (horizon, box_radius_m,
    n_interior_min, interior_pts, limit_frames). For the box side on a REAL AV2 log, `mine` auto-fills
    the `interior_pts` side channel from the annotations feather (so the LiDAR-seen gate is real); a
    caller-supplied `interior_pts` wins. Deterministic: a plain frame scan in index order."""
    sig = signature if isinstance(signature, Signature) else resolve_signature(signature)
    params = dict(params or {})
    limit = params.get("limit_frames")
    out: list[FailureCandidate] = []
    for ir in logs:
        frame_params = dict(params)
        if sig.name == "box_in_free" and "interior_pts" not in frame_params:
            lookup = _av2_interior_pts(ir)
            if lookup is not None:
                frame_params["interior_pts"] = lookup
        if sig.name == "missed_detection":
            _fill_missed_detection_inputs(ir, frame_params, limit)
        n = len(ir.scene.frames)
        rng = range(n) if limit is None else range(min(n, int(limit)))
        for t in rng:
            out.extend(sig.evaluate(ir, t, sig.honesty, frame_params))
    return out


def _av2_log_root(ir: SceneIR):
    """The on-disk AV2 log dir for a SceneIR, or None if not an AV2 log on disk. Mirrors the path
    reconstruction in `_av2_interior_pts` (the IR does not carry the data root, so reconstruct it from
    the standard danger-corpus layout)."""
    import pathlib

    prov = ir.provenance
    if prov is None or prov.dataset != "av2_sensor":
        return None
    candidates = [
        pathlib.Path("data/danger/av2_sensor") / prov.log_id,
        pathlib.Path(prov.log_id),
    ]
    return next((c for c in candidates if (c / "calibration").is_dir()), None)


def _fill_missed_detection_inputs(ir: SceneIR, frame_params: dict, limit) -> None:
    """For a real AV2 log, auto-fill the missed_detection inputs IN PLACE: the camera calibration and
    per-LiDAR-frame detections (run the detector on the nearest camera image per frame). Caller-
    supplied `camera_calib`/`detections` win (so tests inject synthetic ones). Lazy-imports detect so
    `import prism` / `mine` of the other signatures never touch onnxruntime."""
    if "camera_calib" in frame_params and "detections" in frame_params:
        return
    root = _av2_log_root(ir)
    if root is None:
        return
    camera = str(frame_params.get("camera", "ring_front_center"))
    if "camera_calib" not in frame_params:
        cam = load_av2_camera(root, camera)
        if cam is None:
            return
        frame_params["camera_calib"] = cam
    if "detections" not in frame_params:
        frame_params["detections"] = _detect_per_frame(ir, root, camera, frame_params, limit)


def _detect_per_frame(ir: SceneIR, root, camera: str, frame_params: dict, limit) -> dict:
    """Run the detector on the nearest camera image to each LiDAR frame -> {frame_index: [Detection]}.

    Matches each LiDAR sweep timestamp to the nearest camera-image timestamp (cameras and LiDAR are
    not synchronized 1:1). Deterministic; only the in-range frames (respecting `limit_frames`) are
    detected, so a capped scan does not pay for every image. Lazy import of `prism.detect`."""
    import pathlib

    from prism.detect import detect_image  # lazy: optional [detect] extra

    cam_dir = pathlib.Path(root) / "sensors" / "cameras" / camera
    cam_ts = sorted(int(p.stem) for p in cam_dir.glob("*.jpg"))
    if not cam_ts:
        return {}
    cam_arr = np.asarray(cam_ts, dtype=np.int64)
    conf_thr = float(frame_params.get("score_thr", 0.25))
    model_path = str(frame_params.get("model_path", "data/models/yolov8n.onnx"))
    n = len(ir.scene.frames)
    rng = range(n) if limit is None else range(min(n, int(limit)))
    cache: dict[int, list] = {}  # cam_ts -> detections (so re-used nearest image is not re-run)
    out: dict[int, list] = {}
    for t in rng:
        fr = ir.scene.frames[t]
        lidar_ts = int(round(fr.time * 1e9))
        ci = int(np.argmin(np.abs(cam_arr - lidar_ts)))
        nearest = int(cam_arr[ci])
        if nearest not in cache:
            cache[nearest] = detect_image(
                cam_dir / f"{nearest}.jpg", model_path=model_path, conf_thr=conf_thr,
                camera=camera, timestamp_ns=nearest,
            )
        out[t] = cache[nearest]
    return out


def _av2_interior_pts(ir: SceneIR) -> Optional[dict[tuple[int, int], int]]:
    """Build {(frame_index, box_slot): num_interior_pts} for a real AV2 SceneIR by re-reading the
    annotations feather in the SAME per-timestamp box order the av2 adapter built `frame.objects` in.

    Returns None if this is not an AV2 log on disk (synthetic / non-AV2 -> the gate is supplied by the
    caller or treated permissively). This re-read is the ONLY way to recover num_interior_pts: the
    probe TrackedBox drops it and the IR is left untouched (additive constraint)."""
    prov = ir.provenance
    if prov is None or prov.dataset != "av2_sensor":
        return None
    import pathlib

    # the ingest path is data/.../av2_sensor/<log_id>/annotations.feather; reconstruct via the corpus.
    # `mine` cannot know the data root, so look it up relative to the standard danger corpus location.
    candidates = [
        pathlib.Path("data/danger/av2_sensor") / prov.log_id / "annotations.feather",
        pathlib.Path(prov.log_id) / "annotations.feather",
    ]
    ann = next((c for c in candidates if c.exists()), None)
    if ann is None:
        return None
    import pyarrow as pa

    t = pa.ipc.open_file(pa.memory_map(str(ann), "r")).read_all()
    ts = np.asarray(t.column("timestamp_ns").to_pylist(), dtype=np.int64)
    pts = np.asarray(t.column("num_interior_pts").to_pylist(), dtype=np.int64)
    # group rows by timestamp_ns in the SAME insertion order the av2 adapter used (annotation file order)
    by_ts: dict[int, list[int]] = {}
    for i in range(len(ts)):
        by_ts.setdefault(int(ts[i]), []).append(int(pts[i]))
    lookup: dict[tuple[int, int], int] = {}
    for fi, fr in enumerate(ir.scene.frames):
        ts_ns = int(round(fr.time * 1e9))
        rows = by_ts.get(ts_ns)
        if rows is None:
            continue
        for slot, p in enumerate(rows):
            lookup[(fi, slot)] = p
    return lookup


# === clustering ==================================================================================


@dataclass
class FailureCluster:
    """A group of same-signature failures close in feature space, plus its DatasetSlice over frames.

    `centroid` is the mean feature dict; `candidates` are the members; `slice` is the IR DatasetSlice
    over the member frame indices (the reusable "this is the cluster of frames" handle)."""

    signature: str
    centroid: dict[str, float]
    candidates: list[FailureCandidate] = field(default_factory=list)
    slice: DatasetSlice = field(default_factory=lambda: DatasetSlice("", ()))


def cluster(candidates, *, range_bin_m: float = 8.0) -> list[FailureCluster]:
    """Group candidates into clusters by feature bin (signature + forward-range bin + category), the
    same coarse feature buckets the box-recall oracle stratifies by. A bin is a cluster; the centroid
    is the mean feature vector over its members. Simple + deterministic (numpy, no learned model).

    `range_bin_m` is the forward-range bucket width. Candidates of DIFFERENT signatures never share a
    cluster (signature_code is part of the key)."""
    buckets: dict[tuple, list[FailureCandidate]] = {}
    for c in candidates:
        key = (
            c.signature,
            int(_safe(c.features.get("forward_range_m", 0.0)) // range_bin_m),
            int(c.features.get("category_code", 0.0)),
        )
        buckets.setdefault(key, []).append(c)
    clusters: list[FailureCluster] = []
    for (sig_name, _, _), members in buckets.items():
        centroid = _centroid(members)
        frames = tuple(sorted({m.frame_index for m in members}))
        name = f"{sig_name}@range_bin~{centroid['forward_range_m']:.0f}m_n{len(members)}"
        clusters.append(FailureCluster(
            signature=sig_name, centroid=centroid, candidates=list(members),
            slice=DatasetSlice(name=name, frame_indices=frames),
        ))
    # largest cluster first -- the dominant failure mode leads the summary
    clusters.sort(key=lambda cl: len(cl.candidates), reverse=True)
    return clusters


def _safe(v: float) -> float:
    return 1e6 if (isinstance(v, float) and not math.isfinite(v) and v > 0) else (v if math.isfinite(v) else 0.0)


def _centroid(members) -> dict[str, float]:
    vecs = np.stack([_feature_vector(m.features) for m in members], axis=0)
    mean = vecs.mean(axis=0)
    return {k: float(mean[i]) for i, k in enumerate(_FEATURE_KEYS)}


# === similarity (feature-distance, NOT semantic) =================================================


@dataclass(frozen=True)
class RankedFrame:
    """A candidate ranked by feature-vector distance to a cluster centroid. `distance` is the plain
    Euclidean distance in the (signature-shared) feature space -- explicitly NOT a semantic score."""

    candidate: FailureCandidate
    distance: float


def similar_frames(target, pool, *, k: int = 5) -> list[RankedFrame]:
    """The `k` candidates in `pool` most similar to `target` by FEATURE-VECTOR distance.

    `target` is a FailureCluster (uses its centroid) or a FailureCandidate (uses its features).
    Similarity is Euclidean distance in the fixed feature space -- a simple numpy distance, NOT a
    learned/semantic embedding (said plainly so it is never oversold as 'scenes that MEAN the same')."""
    if isinstance(target, FailureCluster):
        ref = _feature_vector(target.centroid)
    elif isinstance(target, FailureCandidate):
        ref = _feature_vector(target.features)
    else:
        raise TypeError(f"target must be a FailureCluster or FailureCandidate, got {type(target)}")
    ranked = [
        RankedFrame(candidate=c, distance=float(np.linalg.norm(_feature_vector(c.features) - ref)))
        for c in pool
    ]
    ranked.sort(key=lambda r: (r.distance, r.candidate.frame_index))
    return ranked[: max(0, k)]


# === S6 find (the wow) ===========================================================================


def find(query: str, logs, *, params: Optional[dict] = None, k_similar: int = 5) -> dict:
    """Map a natural-ish `query` to a failure signature, mine it over `logs`, cluster the hits, and
    return a summary dict (the wow): the signature, honest match counts, per-cluster stats, the top
    similar frames, AND a clean human-readable `human_summary` string. Pure data in/out -- the CLI
    prints it; the API returns it.

    Honest by construction: `n_matches` may be 0 (a corpus where the two backends agree is a real,
    reportable negative -- never inflated), and `honesty` carries the per-signature oracle caveat."""
    sig = signature_for_query(query)
    params = dict(params or {})
    candidates = mine(logs, sig, params=params)
    clusters = cluster(candidates, range_bin_m=float(params.get("range_bin_m", 8.0)))
    n_frames = sum(
        (len(ir.scene.frames) if params.get("limit_frames") is None
         else min(len(ir.scene.frames), int(params["limit_frames"])))
        for ir in logs
    )
    ranges = [_safe(c.features.get("forward_range_m", 0.0)) for c in candidates]
    cats = _top_categories(candidates)
    similar = []
    if clusters:
        top = clusters[0]
        similar = [
            {"log_id": r.candidate.log_id, "frame_index": r.candidate.frame_index,
             "distance": round(r.distance, 3),
             "forward_range_m": round(_safe(r.candidate.features.get("forward_range_m", 0.0)), 2)}
            for r in similar_frames(top, candidates, k=k_similar)
        ]
    summary = {
        "query": query,
        "signature": sig.name,
        "signature_description": sig.description,
        "honesty": sig.honesty,
        "n_logs": len(logs),
        "n_frames_scanned": n_frames,
        "n_matches": len(candidates),
        "n_clusters": len(clusters),
        "mean_forward_range_m": round(float(np.mean(ranges)), 2) if ranges else None,
        "top_categories": cats,
        "clusters": [
            {"name": cl.slice.name, "signature": cl.signature, "size": len(cl.candidates),
             "centroid_forward_range_m": round(cl.centroid["forward_range_m"], 2),
             "frame_indices": list(cl.slice.frame_indices)[:20]}
            for cl in clusters[:10]
        ],
        "similar_frames": similar,
    }
    summary["human_summary"] = _human_summary(summary)
    return summary


def _top_categories(candidates, k: int = 5) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    inv = {v: name for name, v in _CATEGORY_CODES.items()}
    for c in candidates:
        code = c.features.get("category_code", 0.0)
        label = inv.get(code, "n/a" if code == 0.0 else "other")
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _human_summary(s: dict) -> str:
    """A clean shell-printable block -- the wow moment. Leads with the count, the signature, the
    honest oracle caveat, then the dominant cluster + similar frames."""
    lines = []
    lines.append(f'query: "{s["query"]}"')
    lines.append(f'  -> signature: {s["signature"]}  ({s["signature_description"]})')
    lines.append(
        f'  found {s["n_matches"]} matching frame-intervals across {s["n_logs"]} log(s) '
        f'({s["n_frames_scanned"]} frames scanned), in {s["n_clusters"]} cluster(s)'
    )
    if s["mean_forward_range_m"] is not None:
        lines.append(f'  mean forward range to the flagged region: {s["mean_forward_range_m"]} m')
    if s["top_categories"]:
        cats = ", ".join(f"{lab}:{n}" for lab, n in s["top_categories"])
        lines.append(f"  top categories: {cats}")
    if s["clusters"]:
        c0 = s["clusters"][0]
        lines.append(
            f'  dominant cluster: {c0["name"]} (size {c0["size"]}, '
            f'~{c0["centroid_forward_range_m"]} m forward)'
        )
    if s["similar_frames"]:
        sims = ", ".join(f'f{r["frame_index"]}(d={r["distance"]})' for r in s["similar_frames"])
        lines.append(f"  most-similar frames (FEATURE-distance, not semantic): {sims}")
    lines.append(f'  honesty: {s["honesty"]}')
    if s["n_matches"] == 0:
        lines.append("  NOTE: zero matches is an honest negative -- the two backends agreed on this corpus.")
    return "\n".join(lines)
