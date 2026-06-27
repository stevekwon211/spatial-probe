# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""aletheon.detect -- the COCO-pretrained YOLOv8n 2D detector + IR Prediction lowering.

Two things are pinned here:
1. onnxruntime is LAZY: `import aletheon` (and importing aletheon.detect) must NOT pull onnxruntime into
   sys.modules. It is an optional [detect] extra, like rerun. This is asserted structurally.
2. On a real on-disk AV2 image (skip-if-model-missing), the detector returns >= 1 vehicle detection
   with a plausible box, and the detections lower into an IR Prediction.

The detector model (data/models/yolov8n.onnx) is gitignored, so the real-inference test skips when it
is absent -- the lazy-import test always runs (it needs no model, no onnxruntime).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_MODEL = pathlib.Path("data/models/yolov8n.onnx")
_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")
_CAM_LOG = _AV2_ROOT / "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c"
_CAM_DIR = _CAM_LOG / "sensors" / "cameras" / "ring_front_center"


def _first_image() -> pathlib.Path | None:
    if not _CAM_DIR.is_dir():
        return None
    imgs = sorted(_CAM_DIR.glob("*.jpg"))
    return imgs[0] if imgs else None


# === 1. lazy onnxruntime ==========================================================================


def test_importing_aletheon_does_not_import_onnxruntime():
    # If a previous test already imported it, this assertion is moot -- so import in a clean subprocess.
    import subprocess

    code = (
        "import sys; import aletheon; import aletheon.detect; "
        "assert 'onnxruntime' not in sys.modules, "
        "'onnxruntime leaked into sys.modules on import (must be lazy)'; "
        "print('OK')"
    )
    env_src = str(pathlib.Path("src").resolve())
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={"PYTHONPATH": env_src, "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert res.returncode == 0, f"subprocess failed: {res.stderr}"
    assert "OK" in res.stdout


def test_detect_module_imports_without_onnxruntime_at_top():
    # Importing the module in-process must also not require onnxruntime to be importable at module top.
    import importlib

    import aletheon.detect as d

    importlib.reload(d)
    # the module exposes its API regardless of onnxruntime availability
    assert hasattr(d, "detect_image")
    assert hasattr(d, "detect_log")
    assert d.COCO_TO_AV2[2] == "vehicle"  # COCO 'car' -> AV2 'vehicle'
    assert d.COCO_TO_AV2[0] == "pedestrian"


# === 2. real inference (skip if model or images missing) ==========================================


@pytest.mark.skipif(not _MODEL.exists() or _first_image() is None, reason="no detector model or AV2 image")
def test_detect_image_returns_vehicle_on_real_av2_image():
    from aletheon.detect import detect_image

    img = _first_image()
    dets = detect_image(img, model_path=str(_MODEL))
    assert len(dets) >= 1
    # at least one vehicle detection with a plausible, in-bounds, positive-area box
    vehicles = [d for d in dets if d.av2_label == "vehicle"]
    assert vehicles, f"expected >=1 vehicle, got labels {[d.coco_name for d in dets]}"
    v = max(vehicles, key=lambda d: d.score)
    x0, y0, x1, y1 = v.box_xyxy
    assert 0.0 <= x0 < x1 and 0.0 <= y0 < y1, f"implausible box {v.box_xyxy}"
    assert v.score >= 0.25
    # the box must lie within the image bounds (un-letterboxed back to original px)
    from PIL import Image

    W, H = Image.open(img).size
    assert x1 <= W + 1 and y1 <= H + 1, f"box {v.box_xyxy} outside image {W}x{H}"


@pytest.mark.skipif(not _MODEL.exists() or _first_image() is None, reason="no detector model or AV2 image")
def test_ego_hood_false_positive_is_filtered():
    from aletheon.detect import detect_image

    img = _first_image()
    kept = detect_image(img, model_path=str(_MODEL), filter_ego_hood=True)
    unfiltered = detect_image(img, model_path=str(_MODEL), filter_ego_hood=False)
    # the wide bottom-pinned ego-hood box is present unfiltered and dropped when filtering
    from PIL import Image

    W, H = Image.open(img).size

    def has_hood(ds):
        return any((d.box_xyxy[3] >= 0.86 * H and (d.box_xyxy[2] - d.box_xyxy[0]) >= 0.7 * W) for d in ds)

    assert has_hood(unfiltered), "expected an ego-hood FP in the unfiltered detections (image-specific)"
    assert not has_hood(kept), "ego-hood FP was not filtered"


@pytest.mark.skipif(not _MODEL.exists() or _first_image() is None, reason="no detector model or AV2 image")
def test_detections_lower_into_ir_prediction():
    from aletheon.detect import detect_image, detections_to_prediction
    from aletheon.ir import Prediction

    dets = detect_image(_first_image(), model_path=str(_MODEL))
    pred = detections_to_prediction(dets, frame_index=7, source="yolov8n-coco")
    assert isinstance(pred, Prediction)
    assert pred.frame_index == 7
    assert pred.source == "yolov8n-coco"
    assert len(pred.entities) == len(dets)
    if dets:
        e = pred.entities[0]
        # score rides in size[2]; box center in pose.translation (u, v, 0); category = mapped label
        assert e.size[2] == pytest.approx(dets[0].score)
        u, v, z = e.pose.translation
        assert z == 0.0 and u >= 0.0 and v >= 0.0
