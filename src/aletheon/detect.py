# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Run a COCO-pretrained 2D detector (YOLOv8n ONNX) over AV2 camera images and lower its outputs
into the Aletheon IR `Prediction`.

WHY this exists: the rest of Aletheon had NO model predictions on disk, so the only buildable failure
signal was the H1 occupancy-vs-box set difference (consistency, not model-eval). This module adds a
REAL 2D detector so `failure.missed_detection` can ask a genuine model-eval question: did the
detector SEE the labeled object? A GT box that projects into the image with no matching detection is
a missed detection -- the detector failed (Ramanagopal-style), independent of whatever produced the
occupancy grid.

HONEST framing (do not re-inflate):
- This is a COCO-pretrained YOLOv8n, NOT trained on AV2/Argoverse. Its recall vs AV2 GT is a
  cross-distribution detector-eval signal, NOT an in-domain benchmark number.
- The COCO->AV2 category map is imperfect (COCO has no LARGE_VEHICLE/BOX_TRUCK split; AV2 has no
  generic "vehicle"). The map is coarse and explicit (`COCO_TO_AV2`), never silently exact.
- A miss = the detector did not output a matching box for a GT object the camera could see. It is
  NOT an occupancy claim and NOT a statement about the AV2 labeler.

REUSE: the decode (`letterbox`, `nms`, `_decode`) is the pure-numpy logic from
`data/models/verify_yolov8_detect.py`, copied in verbatim (incl. the verified normalized-coords
guard `max<=1.5 -> *640`). onnxruntime + PIL are LAZY-imported INSIDE the functions: onnxruntime is
an OPTIONAL `[detect]` extra, exactly like rerun, so `import aletheon` stays onnxruntime-free.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from aletheon.ir import Entity, Pose, Prediction

__all__ = [
    "Detection",
    "COCO_NAMES",
    "COCO_TO_AV2",
    "DEFAULT_MODEL_PATH",
    "detect_image",
    "detect_log",
    "detections_to_prediction",
]

DEFAULT_MODEL_PATH = "data/models/yolov8n.onnx"

# Standard 80-class COCO order (YOLOv8 default data.yaml order). Copied from the verify script.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# COCO class id -> the COARSE AV2 label the probe uses (av2_sensor._coarse outputs these).
# Imperfect by construction: COCO has one "car"/"bus"/"truck"; AV2 splits REGULAR_VEHICLE /
# LARGE_VEHICLE / BUS / BOX_TRUCK etc. We map all of {car,bus,truck} to the coarse "vehicle" the
# adapter already collapses AV2 vehicle classes into, so detector class == GT coarse class is a fair
# comparison. (The miss matcher can also be run class-agnostically; see failure.missed_detection.)
COCO_TO_AV2: dict[int, str] = {
    0: "pedestrian",   # person
    1: "bicycle",
    2: "vehicle",      # car
    3: "motorcycle",
    5: "vehicle",      # bus
    7: "vehicle",      # truck
}


@dataclass(frozen=True)
class Detection:
    """One 2D detection: COCO class name + id, the mapped coarse AV2 label, score, xyxy pixel box,
    plus the camera + timestamp it came from. `av2_label` is None for a COCO class with no AV2 driving
    counterpart (e.g. 'umbrella') -- kept explicit, never silently dropped to 'other'."""

    coco_name: str
    coco_id: int
    av2_label: Optional[str]
    score: float
    box_xyxy: tuple[float, float, float, float]
    camera: str = ""
    timestamp_ns: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box_xyxy
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# === pure-numpy decode (copied verbatim from data/models/verify_yolov8_detect.py) ================
# Kept self-contained here so the [detect] path has no dependency on a script under data/.


def letterbox(img: np.ndarray, new_shape=(640, 640), color=114):
    """img: HWC uint8 RGB. Returns padded img, scale ratio r, (pad_w, pad_h). Resize via PIL (no cv2)."""
    from PIL import Image  # lazy: PIL only needed on the detect path

    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    pad_h = (new_shape[0] - nh) // 2
    pad_w = (new_shape[1] - nw) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return canvas, r, pad_w, pad_h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """boxes: [N,4] xyxy. Pure-numpy greedy NMS. Returns kept indices (class-agnostic)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def _decode(out: np.ndarray, H0: int, W0: int, r: float, pad_w: int, pad_h: int,
            conf_thr: float, iou_thr: float) -> list[tuple[int, float, np.ndarray]]:
    """output0 [1,84,8400] + letterbox params -> [(coco_id, score, xyxy_orig_px), ...].

    Mirrors the verify script's postprocess EXACTLY, incl. the verified normalized-coords guard:
    this export emits box coords normalized to [0,1] of the 640 canvas, so multiply by 640 when the
    max <= 1.5 (a stock pixel-space export skips the multiply -- the guard handles both)."""
    pred = out[0].T  # [8400, 84]
    boxes_xywh = pred[:, :4].copy()
    cls_scores = pred[:, 4:]
    if boxes_xywh[:, :4].max() <= 1.5:
        boxes_xywh *= 640.0
    cls_ids = cls_scores.argmax(1)
    confs = cls_scores.max(1)
    m = confs >= conf_thr
    boxes_xywh, confs, cls_ids = boxes_xywh[m], confs[m], cls_ids[m]
    if boxes_xywh.shape[0] == 0:
        return []
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    xyxy[:, [0, 2]] -= pad_w
    xyxy[:, [1, 3]] -= pad_h
    xyxy /= r
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W0)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H0)
    keep = nms(xyxy, confs, iou_thr)
    return [(int(cls_ids[i]), float(confs[i]), xyxy[i]) for i in keep]


# === inference session (cached per process) ======================================================

_SESSION_CACHE: dict[str, object] = {}


def _session(model_path: str):
    """Lazily build (and cache) an onnxruntime CPU InferenceSession. onnxruntime is the OPTIONAL
    `[detect]` extra -- imported HERE, never at module top, so `import aletheon` stays onnxruntime-free.
    Raises a clear install hint if the extra is missing (explicit failure over silent fallback)."""
    if model_path in _SESSION_CACHE:
        return _SESSION_CACHE[model_path]
    try:
        import onnxruntime as ort  # lazy: optional [detect] extra
    except ImportError as e:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "onnxruntime is required for the detector. Install the optional extra: "
            "pip install -e '.[detect]'"
        ) from e
    if not pathlib.Path(model_path).exists():
        raise FileNotFoundError(
            f"detector model not found: {model_path} (it is gitignored; see "
            f"data/models/verify_yolov8_detect.py for how it was produced)"
        )
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[model_path] = sess
    return sess


# === ego-hood false-positive filter ==============================================================
# The ring_front_center frame includes the ego hood at the very bottom; the detector fires a large
# spurious box there (verified: an 'umbrella' spanning the full image width at y ~ 0.88..1.0 H). A
# real obstacle never fills the bottom band edge-to-edge, so drop a detection whose box bottom sits in
# the bottom EGO_HOOD_FRAC of the image AND that spans most of the width.
_EGO_HOOD_FRAC = 0.86          # box must reach below this fraction of image height to be a hood candidate
_EGO_HOOD_WIDTH_FRAC = 0.7     # ... and span at least this fraction of image width


def _is_ego_hood(box_xyxy, H0: int, W0: int) -> bool:
    x0, y0, x1, y1 = box_xyxy
    bottom_band = y1 >= _EGO_HOOD_FRAC * H0
    wide = (x1 - x0) >= _EGO_HOOD_WIDTH_FRAC * W0
    return bool(bottom_band and wide)


# === public API ==================================================================================


def detect_image(img_path, *, model_path: str = DEFAULT_MODEL_PATH, conf_thr: float = 0.25,
                 iou_thr: float = 0.45, camera: str = "", timestamp_ns: int = 0,
                 filter_ego_hood: bool = True) -> list[Detection]:
    """Run the detector on one image file -> list[Detection] (xyxy in original-image pixels).

    Lazy-imports onnxruntime + PIL INSIDE the function (optional `[detect]` extra). The ego-hood FP
    (a wide box pinned to the image bottom) is filtered by default. `camera`/`timestamp_ns` are
    stamped onto each Detection for downstream IR lowering / GT matching."""
    from PIL import Image  # lazy: optional [detect] path

    sess = _session(model_path)
    img = np.asarray(Image.open(str(img_path)).convert("RGB"))
    H0, W0 = img.shape[:2]
    padded, r, pad_w, pad_h = letterbox(img, (640, 640))
    blob = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # [1,3,640,640]
    out = sess.run(["output0"], {"images": blob})[0]  # [1,84,8400]
    decoded = _decode(out, H0, W0, r, pad_w, pad_h, conf_thr, iou_thr)
    dets: list[Detection] = []
    for coco_id, score, xyxy in decoded:
        box = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        if filter_ego_hood and _is_ego_hood(box, H0, W0):
            continue
        dets.append(Detection(
            coco_name=COCO_NAMES[coco_id], coco_id=coco_id, av2_label=COCO_TO_AV2.get(coco_id),
            score=score, box_xyxy=box, camera=camera, timestamp_ns=timestamp_ns,
        ))
    dets.sort(key=lambda d: -d.score)
    return dets


def _camera_image_index(log_root: pathlib.Path, camera: str) -> list[tuple[int, pathlib.Path]]:
    """[(timestamp_ns, jpg_path), ...] for one camera, sorted by timestamp."""
    cam_dir = log_root / "sensors" / "cameras" / camera
    if not cam_dir.is_dir():
        return []
    out = []
    for p in cam_dir.glob("*.jpg"):
        try:
            out.append((int(p.stem), p))
        except ValueError:
            continue
    out.sort()
    return out


def detect_log(log_root, *, camera: str = "ring_front_center", model_path: str = DEFAULT_MODEL_PATH,
               conf_thr: float = 0.25, iou_thr: float = 0.45, limit: Optional[int] = None,
               filter_ego_hood: bool = True) -> dict[int, list[Detection]]:
    """Run the detector over every image of one camera in an AV2 log dir.

    Returns {camera_timestamp_ns: [Detection, ...]}. `limit` caps the number of images (fast scans).
    Deterministic: images are processed in timestamp order. Camera timestamps differ from LiDAR
    sweep timestamps; `failure.missed_detection` matches each LiDAR frame to its nearest camera
    image, so this returns the RAW per-camera-timestamp detections."""
    log_root = pathlib.Path(log_root)
    index = _camera_image_index(log_root, camera)
    if limit is not None:
        index = index[: int(limit)]
    out: dict[int, list[Detection]] = {}
    for ts_ns, path in index:
        out[ts_ns] = detect_image(
            path, model_path=model_path, conf_thr=conf_thr, iou_thr=iou_thr,
            camera=camera, timestamp_ns=ts_ns, filter_ego_hood=filter_ego_hood,
        )
    return out


def detections_to_prediction(detections: list[Detection], frame_index: int, *,
                             source: str = "yolov8n-coco") -> Prediction:
    """Lower a frame's Detections into the IR `Prediction` (predicted entities for one frame).

    Each Detection becomes an `Entity` in the named camera frame: `category` = the mapped AV2 label
    (or the COCO name when there is no AV2 counterpart, so nothing is silently lost), `pose` carries
    the 2D box CENTER as (u, v, 0) in PIXEL coordinates, `size` carries (box_w_px, box_h_px, score)
    -- a deliberate 2D-in-a-3D-slot encoding so a 2D detector's output lowers into the SAME IR shape
    as a 3D box without faking a depth. `frame` names the source camera; the score rides in size[2]
    so it survives serialization. This makes the detector output comparable-by-construction with GT
    in the IR, with the 2D-ness stated, not hidden."""
    entities = []
    for k, d in enumerate(detections):
        u, v = d.center
        x0, y0, x1, y1 = d.box_xyxy
        entities.append(Entity(
            entity_id=f"det#{frame_index}.{k}",
            category=d.av2_label if d.av2_label is not None else d.coco_name,
            pose=Pose(translation=(float(u), float(v), 0.0)),
            size=(float(x1 - x0), float(y1 - y0), float(d.score)),
            velocity=(math.nan, math.nan),  # 2D detection has no measured velocity -> NEVER silent 0
            frame=d.camera or "camera",
        ))
    return Prediction(frame_index=frame_index, entities=tuple(entities), source=source)
