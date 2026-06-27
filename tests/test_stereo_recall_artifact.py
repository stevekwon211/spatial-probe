# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Round-trip test for the learned-stereo (IGEV) disparity-artifact seam in oracle_stereo_recall.

The learned-stereo pre-reg (`oracle_stereo_recall_learned_preregistration.md`, git 1c8b357) changes
EXACTLY ONE thing vs the sealed classical run: the depth front-end. torch/IGEV run on an external GPU
pod that emits per-(log, frame, side) disparity `.npz` artifacts; the local numpy oracle grades them
via `--disparity-source artifact`. This test proves that seam is FAITHFUL (an artifact loaded through
`get_disparity` is byte-identical to what was saved, on the exact census grid) WITHOUT a GPU, and that
the census path stays the untouched default. It NEVER imports torch -- which it also asserts.
"""
import sys

import numpy as np
import pytest

from experiments.occquery_v0.oracle_stereo_recall import (
    Config,
    artifact_name,
    compute_disparity,
    compute_disparity_right,
    get_disparity,
)

_LOG = "201fe83b-7dd7-38f4-9d26-7b4a668638a9"   # a real following-substrate log UUID (full, not truncated)
_TS = 315967376899927209                         # a full-nanosecond camera-frame timestamp (jpg-stem shaped)
_SHAPE = (48, 72)                                # a small stand-in for the undistorted 2x-downsampled grid


def _artifact_cfg(artifact_dir) -> Config:
    """A Config wired to the artifact source (everything else at the sealed defaults)."""
    return Config(
        z_min=2.0, z_max=30.0, n_stereo_min=8, lr_consistency_px=1.0, edge_discontinuity_m=1.5,
        null="band-local", shuffles=10, seed=0, downsample=2,
        disparity_source="artifact", disparity_artifact_dir=artifact_dir,
    )


def _synth_disp(seed: int) -> np.ndarray:
    """A small disparity array in the contract layout: float32, NaN=invalid, finite positive elsewhere
    (positive = nearer, the census sign convention)."""
    rng = np.random.default_rng(seed)
    disp = np.full(_SHAPE, np.nan, dtype=np.float32)
    disp[8:40, 12:60] = rng.uniform(1.0, 60.0, size=(32, 48)).astype(np.float32)  # a valid region
    return disp


def _save_artifact(artifact_dir, log, ts, side, disp) -> None:
    (artifact_dir / artifact_name(log, ts, side)).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(artifact_dir / artifact_name(log, ts, side)), disp=disp)


@pytest.mark.parametrize("side", ["L", "R"])
def test_artifact_roundtrip_byte_identical(tmp_path, side):
    """Save a disparity artifact, load it back through the --disparity-source artifact code path
    (get_disparity), and assert it is byte-identical to what was saved -- the seam is lossless."""
    saved = _synth_disp(seed=1 if side == "L" else 2)
    _save_artifact(tmp_path, _LOG, _TS, side, saved)
    cfg = _artifact_cfg(tmp_path)
    grayL_s = np.zeros(_SHAPE)  # only the SHAPE is read in artifact mode (validation), not the pixels
    grayR_s = np.zeros(_SHAPE)
    loaded = get_disparity(grayL_s, grayR_s, 1, 50, log=_LOG, ts=_TS, side=side, cfg=cfg)

    assert loaded.shape == _SHAPE
    # value-identical (NaN-aware): float32 -> float64 widening is exact
    assert np.array_equal(loaded, saved, equal_nan=True)
    # AND byte-identical at float32 (the stored precision): no silent rescale/transpose/sign flip
    assert loaded.astype(np.float32).tobytes() == saved.tobytes()
    # invalidity is preserved exactly (NaN mask unchanged)
    assert np.array_equal(np.isnan(loaded), np.isnan(saved))


def test_artifact_filename_matches_nearest_convention():
    """The artifact key is the FULL camera-frame timestamp (the _nearest/_cam_timestamps result the
    oracle already uses to open the jpg) -- NOT a truncation, NOT a new key. The full log UUID is kept."""
    name = artifact_name(_LOG, _TS, "L")
    assert name == f"disp_{_LOG}_{_TS}_L.npz"
    assert str(_TS) in name and _LOG in name        # full ts + full log present, untruncated
    assert artifact_name(_LOG, _TS, "R").endswith("_R.npz")
    with pytest.raises(ValueError):
        artifact_name(_LOG, _TS, "X")               # only L/R are valid sides


def test_census_is_the_untouched_default(tmp_path):
    """Census stays the DEFAULT source and is byte-for-byte the sealed matcher: get_disparity in census
    mode returns exactly compute_disparity()[0] (L) and compute_disparity_right() (R)."""
    assert Config(z_min=2.0, z_max=30.0, n_stereo_min=8, lr_consistency_px=1.0,
                  edge_discontinuity_m=1.5, null="band-local", shuffles=10, seed=0,
                  downsample=2).disparity_source == "census"
    rng = np.random.default_rng(7)
    gL = rng.normal(120, 30, size=_SHAPE)
    gR = rng.normal(120, 30, size=_SHAPE)
    cfg = Config(z_min=2.0, z_max=30.0, n_stereo_min=8, lr_consistency_px=1.0, edge_discontinuity_m=1.5,
                 null="band-local", shuffles=10, seed=0, downsample=2)  # source defaults to census
    dL = get_disparity(gL, gR, 1, 10, log=_LOG, ts=_TS, side="L", cfg=cfg)
    dR = get_disparity(gL, gR, 1, 10, log=_LOG, ts=_TS, side="R", cfg=cfg)
    assert np.array_equal(dL, compute_disparity(gL, gR, 1, 10)[0], equal_nan=True)
    assert np.array_equal(dR, compute_disparity_right(gL, gR, 1, 10), equal_nan=True)


def test_artifact_shape_mismatch_raises(tmp_path):
    """A mis-geometried artifact (wrong grid) fails LOUDLY rather than silently mis-grading."""
    _save_artifact(tmp_path, _LOG, _TS, "L", np.zeros((10, 10), dtype=np.float32))
    cfg = _artifact_cfg(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        get_disparity(np.zeros(_SHAPE), np.zeros(_SHAPE), 1, 50, log=_LOG, ts=_TS, side="L", cfg=cfg)


def test_missing_artifact_raises(tmp_path):
    """A requested artifact that the pod never produced is a hard error (no silent empty disparity)."""
    cfg = _artifact_cfg(tmp_path)
    with pytest.raises(FileNotFoundError):
        get_disparity(np.zeros(_SHAPE), np.zeros(_SHAPE), 1, 50, log=_LOG, ts=_TS, side="L", cfg=cfg)


def test_artifact_dir_required_for_artifact_source():
    """Selecting the artifact source without a dir is a configuration error, not a fallback to census."""
    cfg = Config(z_min=2.0, z_max=30.0, n_stereo_min=8, lr_consistency_px=1.0, edge_discontinuity_m=1.5,
                 null="band-local", shuffles=10, seed=0, downsample=2,
                 disparity_source="artifact", disparity_artifact_dir=None)
    with pytest.raises(ValueError, match="disparity-artifact-dir"):
        get_disparity(np.zeros(_SHAPE), np.zeros(_SHAPE), 1, 50, log=_LOG, ts=_TS, side="L", cfg=cfg)


def test_no_torch_imported_by_the_oracle():
    """Hard constraint: importing the oracle (done at module top) must NOT pull in torch or cv2 -- they
    live only on the GPU pod. This guards the pure-numpy core invariant."""
    assert "torch" not in sys.modules
    assert "cv2" not in sys.modules
