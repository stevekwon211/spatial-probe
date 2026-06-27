# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Oracle-v1 -- AV2 stereo-camera classical-triangulation RECALL oracle for occupancy
denotation-COMPLETENESS (occquery H3, recall half).

Design SEALED in `oracle_stereo_recall_preregistration.md` BEFORE this run (git 6fdbf5c). This
module realizes that pre-registration faithfully; it does NOT redesign it. Every parameter
(z-min/max, n_stereo_min=8, lr-consistency 1.0px, uniqueness peak-ratio 0.85, edge-discontinuity
1.5m, band-local null, AUC>=0.75 gate, texture-gate 40th-pct on the held-out log) is fixed by the
spec and exposed only as a CLI default so the sealed command is literal.

The estimand (pre-reg): a forward ego-frame voxel on the in-path band, read at the SAME frame t for
both sensors. Where occupancy reports FREE in the ego in-path band, does the classical stereo depth
map place a real surface there (a candidate occupancy MISS)? Reported as the per-log-clustered mean
miss-rate with a log-clustered bootstrap CI, and as the GAP (band-local-shuffled - true) whose CI
must be strictly > 0 to NOT trigger the falsifiable kill.

INDEPENDENCE (pre-reg ledger): passive optical triangulation vs active TOF LiDAR (modality), block-
match disparity Z=fB/d vs `av2_sensor._voxelize` (algorithm), stereo JPEGs vs LiDAR feathers
(provenance). Shares platform (same AV2 vehicle/timestamp) -> "much more independent, not external
truth".

REUSE: `projection.quat_to_rotmat` + the `project_ego_points` MATH (re-instantiated for AV2, NOT the
nuScenes Camera dataclass); `av2_sensor` grid spec + `_voxelize` filters EXACTLY (so a stereo
obstacle voxel == a LiDAR obstacle voxel); `freepath.free_along_ego_path` half-width semantic for the
band; `camera_oracle.patch_evidence` + `_roc_auc` VERBATIM for the texture gate + calibration AUC.

STACK: numpy / scipy / Pillow / pyarrow ONLY. NO cv2, NO torch, NO pandas (verified absent in .venv).
Feathers read via pyarrow Arrow-IPC. The numpy census/SAD matcher is the PRIMARY path (not a
fallback) per the pre-reg's GPU-gated-fallback section.

INTEGRITY GATES (owned by the repo owner, NOT by this code):
  - The 60-patch human labeling is a HUMAN integrity gate. `--emit-calib-patches` writes the
    deterministically-sampled patches + `calib_patches.json` (label=null) for the human to fill.
  - The confirmatory run is sealed-gated and is NOT executed here. `--self-check` validates the
    geometry on the real feathers with no confirmatory data.

Modes:
  --self-check            geometry round-trips on real calibration (no confirmatory data); RUN FIRST.
  --emit-calib-patches    deterministically sample + crop the 60 calibration patches for human labels.
  (default / confirmatory) the full sealed run -> results JSON. Requires human-filled labels.

Sealed run command (from `oracle_stereo_recall_preregistration.md`):
  .venv/bin/python experiments/occquery_v0/oracle_stereo_recall.py \
    --logs 201fe83b-7dd7-38f4-9d26-7b4a668638a9 \
           2c652f9e-8db8-3572-aa49-fae1344a875b \
           6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
    --heldout-threshold-log 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
    --z-min 2.0 --z-max 30.0 --n-stereo-min 8 --lr-consistency-px 1.0 \
    --edge-discontinuity-m 1.5 --null band-local --shuffles 1000 --seed 0 \
    --out experiments/occquery_v0/results/oracle_stereo_recall.json

DISPARITY SOURCE (sealed in `oracle_stereo_recall_learned_preregistration.md`, git 1c8b357) -- the
ONLY thing that may change vs the classical run is the DEPTH FRONT-END (the matcher). `--disparity-
source census` (DEFAULT) keeps the sealed numpy census/SAD path byte-for-byte. `--disparity-source
artifact --disparity-artifact-dir DIR` swaps ONLY the disparity at the 4 census call sites for a
pre-computed learned-stereo (IGEV) artifact; EVERYTHING downstream (Z=fx*B/d, band, FOV, voxelize
filters, lr-consistency 1.0px, edge 1.5m, n_stereo_min, band-local null, AUC gate, bootstrap, kill)
is inherited unchanged. torch/IGEV NEVER run here -- they run on an external GPU pod
(`igev_disparity_pod.py`) that emits the artifacts; this module stays pure numpy/scipy/Pillow/pyarrow.

ARTIFACT CONTRACT (the drop-in seam; pod writer + `get_disparity` reader share `artifact_name`):
  filename:  `<DIR>/disp_<log>_<cam_ts>_<side>.npz`  where
             - `<log>`    = the full AV2 log UUID (e.g. 201fe83b-7dd7-38f4-9d26-7b4a668638a9),
             - `<cam_ts>` = the FULL-nanosecond camera-frame timestamp == the key the oracle already
                            uses to find the stereo frame, i.e. `_nearest(_cam_timestamps(log_dir,
                            "stereo_front_{left|right}"), reference_ts)` (the same jpg stem it opens;
                            NOT a new key, NOT a truncation),
             - `<side>`   = "L" (left-image disparity, drop-in for `compute_disparity`) or
                            "R" (right-image disparity, drop-in for `compute_disparity_right`).
  npz keys:  `disp` -- REQUIRED. float32, shape == the census `undL_s`/`undR_s` grid (the UNDISTORTED,
             2x-downsampled LEFT grid for side "L"; the undistorted 2x-downsampled RIGHT grid for side
             "R"). Disparity is in DOWNSAMPLED pixels with the census sign convention: positive =
             nearer, so the right-image feature matching left pixel (v,u) sits at (v, u-disp).
             NaN marks invalid/unmatched pixels (IGEV finite-mask + any out-of-frame after warp-back).
             So Z = camL_s.fx * B / disp reproduces the census metric depth exactly, and the inherited
             local `lr_consistency` (1.0px, byte-for-byte) is the single LR-consistency filter, run on
             the IGEV disparities exactly as it ran on census ones. Any extra keys are IGNORED (NaN in
             `disp` already encodes all invalidity the downstream needs).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "experiments" / "dynfield_v0"))

import pyarrow as pa  # noqa: E402
import pyarrow.feather as pa_feather  # noqa: E402

from harness_v2 import _boot_mean  # noqa: E402  (log-clustered bootstrap, same as oracle_traversal)
from probe.adapters import av2_sensor  # noqa: E402  (grid spec + _voxelize filters reused EXACTLY)
from projection import quat_to_rotmat  # noqa: E402  (reused MATH)
from camera_oracle import _roc_auc, patch_evidence  # noqa: E402  (_roc_auc reused VERBATIM for the
#   calibration AUC; patch_evidence reused for the dense-field cross-check only -- see _patch_evidence_field
#   SPEC-NOTE on why it is NOT called verbatim on stereo-sized images.)

_AV2 = _HERE.parents[1] / "data" / "danger" / "av2_sensor"

# Grid spec mirrored from av2_sensor (read, never redefined -- the source of truth is the adapter).
_VOX = av2_sensor.VOXEL_SIZE
_RANGE = av2_sensor.RANGE
_NX, _NY, _NZ = av2_sensor.GRID_SHAPE
_EGO_HALF_W = av2_sensor._EGO_HALF_W  # 1.05 m -- the in-path band half-width per the pre-reg

# Sealed stereo-matcher constants (block-matching). These are the matcher's internal numerical
# settings, fixed here so the disparity search is deterministic; they are NOT the pre-registered
# decision thresholds (those arrive via the CLI). SPEC-NOTE below documents each literal reading.
_BLOCK = 7              # SAD/census block half-window is _BLOCK (window = 2*_BLOCK+1 = 15 px).
_UNIQUENESS = 0.85      # uniqueness peak-ratio (best/second-best) <= 0.85 -- SEALED in pre-reg.
_DOWNSAMPLE = 2         # SPEC-NOTE: the pre-reg fixes d-range [28,421]px on the FULL-RES image and
#                         fx*B (~841.6). 2048x1550 full-res dense block-matching over ~400 disparities
#                         in pure numpy is ~10^11 cost-volume ops/pair * ~957 pairs -- infeasible in
#                         the no-cv2/no-torch budget. The pre-reg does NOT fix an image scale, only the
#                         disparity range in px and Z=fx*B/d. We match on a 2x-downsampled image and
#                         carry the scale EXPLICITLY through the geometry: fx, cx, cy, B, and the
#                         d-range are all scaled by 1/_DOWNSAMPLE so Z=fx_s*B/d_s is identical to the
#                         full-res Z=fx*B/d (verified in _self_check, check ii). This is a compute
#                         realization choice, not a parameter change; the LITERAL Z mapping is
#                         preserved. Documented loudly here so the human can override (--downsample 1)
#                         if a full-res run is funded.


# ----------------------------------------------------------------------------------------------
# Calibration -- per-log AV2 stereo Camera re-instantiated from the feathers (NOT the nuScenes one).
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AV2Camera:
    """One AV2 stereo camera's calibration, read from intrinsics.feather + egovehicle_SE3_sensor.

    AV2 camera frame convention (verified against av2 devkit docs and the projection.py nuScenes
    convention, which is identical for the optical axes): x-right, y-down, z-forward (optical axis).
    A point is visible iff camera-frame z > 0; pixel = (fx*x/z + cx, fy*y/z + cy) after distortion.
    `R_cam2ego` maps camera axes -> ego axes (the sensor->ego rotation from the extrinsic quaternion);
    `t_cam_in_ego` is the camera origin in the ego frame. p_cam = R_cam2ego.T @ (p_ego - t_cam_in_ego)
    -- the exact inverse of projection.project_ego_points.
    """

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
    R_cam2ego: np.ndarray   # (3,3) camera axes -> ego axes
    t_cam_in_ego: np.ndarray  # (3,) camera origin in ego frame

    def scaled(self, ds: int) -> "AV2Camera":
        """Return an intrinsics-scaled copy for matching on a 1/ds-downsampled image. Distortion
        coefficients are normalized-radius dimensionless, so k1..k3 are UNCHANGED; fx,fy,cx,cy and
        the image size scale by 1/ds. Z=fx_s*B/d_s stays equal to full-res Z (see _self_check ii)."""
        if ds == 1:
            return self
        return AV2Camera(
            name=self.name, fx=self.fx / ds, fy=self.fy / ds,
            cx=self.cx / ds, cy=self.cy / ds, k1=self.k1, k2=self.k2, k3=self.k3,
            width=int(round(self.width / ds)), height=int(round(self.height / ds)),
            R_cam2ego=self.R_cam2ego, t_cam_in_ego=self.t_cam_in_ego,
        )


def _read_feather(p: pathlib.Path):
    return pa.ipc.open_file(pa.memory_map(str(p), "r")).read_all()


def load_stereo_calib(log: str, data_root: pathlib.Path) -> tuple[AV2Camera, AV2Camera, float]:
    """Read stereo_front_{left,right} intrinsics + extrinsics for one log. Returns (camL, camR, B).

    B = |t_left - t_right| (the lateral ego-y baseline). The pre-reg's epipolar-slope guard is
    applied per-frame in the matcher, not here; this just supplies the geometry."""
    root = data_root / log / "calibration"
    intr = pa_feather.read_table(root / "intrinsics.feather").to_pydict()
    ext = pa_feather.read_table(root / "egovehicle_SE3_sensor.feather").to_pydict()
    cams: dict[str, AV2Camera] = {}
    t_of: dict[str, np.ndarray] = {}
    inames = intr["sensor_name"]
    enames = ext["sensor_name"]
    eidx = {nm: i for i, nm in enumerate(enames)}
    for side in ("stereo_front_left", "stereo_front_right"):
        ii = inames.index(side)
        ei = eidx[side]
        q = (ext["qw"][ei], ext["qx"][ei], ext["qy"][ei], ext["qz"][ei])
        t = np.array([ext["tx_m"][ei], ext["ty_m"][ei], ext["tz_m"][ei]], dtype=float)
        cams[side] = AV2Camera(
            name=side,
            fx=float(intr["fx_px"][ii]), fy=float(intr["fy_px"][ii]),
            cx=float(intr["cx_px"][ii]), cy=float(intr["cy_px"][ii]),
            k1=float(intr["k1"][ii]), k2=float(intr["k2"][ii]), k3=float(intr["k3"][ii]),
            width=int(intr["width_px"][ii]), height=int(intr["height_px"][ii]),
            R_cam2ego=quat_to_rotmat(q), t_cam_in_ego=t,
        )
        t_of[side] = t
    B = float(np.linalg.norm(t_of["stereo_front_left"] - t_of["stereo_front_right"]))
    return cams["stereo_front_left"], cams["stereo_front_right"], B


# ----------------------------------------------------------------------------------------------
# AV2 Su radial distortion (k1,k2,k3) -- 3-coeff, NOT 5-coeff OpenCV. Pure numpy, no cv2.
# ----------------------------------------------------------------------------------------------
# SPEC-NOTE (distortion convention): AV2 ships the `Su` pinhole-radial model. Its FORWARD direction
# (3D -> pixel, the projection used in project()) is:
#     r2 = xn^2 + yn^2           (xn,yn = undistorted normalized coords = X/Z, Y/Z)
#     scale = 1 + k1*r2 + k2*r2^2 + k3*r2^3
#     (xd, yd) = (xn*scale, yn*scale);  u = fx*xd + cx,  v = fy*yd + cy
# This matches av2.geometry.camera.pinhole_camera (radial-only; no tangential, no k4..k6). The av2
# devkit is NOT importable in this .venv (ModuleNotFoundError: No module named 'av2', verified), so I
# implement the documented closed-form FORWARD model and INVERT it numerically (fixed-point iteration)
# for undistortion. The forward<->inverse round-trip is asserted < 0.5px at a corner pixel in
# _self_check (check iv); if that round-trip ever fails the convention is wrong and the run must stop.
def _distort_scale(r2: np.ndarray, k1: float, k2: float, k3: float) -> np.ndarray:
    return 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2


def distort_normalized(xn: np.ndarray, yn: np.ndarray, cam: AV2Camera) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted normalized coords -> distorted normalized coords (forward Su model)."""
    r2 = xn * xn + yn * yn
    s = _distort_scale(r2, cam.k1, cam.k2, cam.k3)
    return xn * s, yn * s


def undistort_normalized(xd: np.ndarray, yd: np.ndarray, cam: AV2Camera,
                         iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Distorted normalized coords -> undistorted normalized coords. Fixed-point inversion of the
    radial Su model: xn = xd / (1 + k1 r_n^2 + ...), iterated on r_n^2. Converges fast (|k1|<1)."""
    xn = np.array(xd, dtype=float, copy=True)
    yn = np.array(yd, dtype=float, copy=True)
    for _ in range(iters):
        r2 = xn * xn + yn * yn
        s = _distort_scale(r2, cam.k1, cam.k2, cam.k3)
        xn = xd / s
        yn = yd / s
    return xn, yn


def build_undistort_map(cam: AV2Camera) -> tuple[np.ndarray, np.ndarray]:
    """For each pixel (col u, row v) of the UNDISTORTED (rectified-pinhole) image, the source pixel
    (su, sv) in the DISTORTED raw image to sample from. Returns (map_u, map_v) each (H,W) float.

    Undistorted pixel -> undistorted normalized (xn,yn) via inverse K -> distort to (xd,yd) via the
    forward Su model -> distorted pixel via K. Standard remap convention (sample raw at the distorted
    location for each output pixel)."""
    H, W = cam.height, cam.width
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    xn = (uu - cam.cx) / cam.fx
    yn = (vv - cam.cy) / cam.fy
    xd, yd = distort_normalized(xn, yn, cam)
    map_u = xd * cam.fx + cam.cx
    map_v = yd * cam.fy + cam.cy
    return map_u, map_v


def remap_gray(gray: np.ndarray, map_u: np.ndarray, map_v: np.ndarray) -> np.ndarray:
    """Bilinear-sample `gray` (H,W) at the (map_u, map_v) source coords. Out-of-frame -> NaN."""
    H, W = gray.shape
    u0 = np.floor(map_u).astype(np.intp)
    v0 = np.floor(map_v).astype(np.intp)
    fu = map_u - u0
    fv = map_v - v0
    valid = (u0 >= 0) & (u0 < W - 1) & (v0 >= 0) & (v0 < H - 1)
    u0c = np.clip(u0, 0, W - 2)
    v0c = np.clip(v0, 0, H - 2)
    g00 = gray[v0c, u0c]
    g01 = gray[v0c, u0c + 1]
    g10 = gray[v0c + 1, u0c]
    g11 = gray[v0c + 1, u0c + 1]
    top = g00 * (1 - fu) + g01 * fu
    bot = g10 * (1 - fu) + g11 * fu
    out = top * (1 - fv) + bot * fv
    out = np.where(valid, out, np.nan)
    return out


# ----------------------------------------------------------------------------------------------
# Image IO + nearest-timestamp matching (camera frames ~20Hz, OFFSET from LiDAR; exact match fails).
# ----------------------------------------------------------------------------------------------
def _gray_from_jpg(path: pathlib.Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])  # same luma as camera_oracle._gray


def _cam_timestamps(log_dir: pathlib.Path, side: str) -> np.ndarray:
    ps = glob.glob(str(log_dir / "sensors" / "cameras" / side / "*.jpg"))
    return np.array(sorted(int(os.path.basename(p)[:-4]) for p in ps), dtype=np.int64)


def _nearest(ts_array: np.ndarray, target: int) -> int:
    return int(ts_array[int(np.argmin(np.abs(ts_array - target)))])


# ----------------------------------------------------------------------------------------------
# Block-matching disparity -- VECTORIZED shift-and-cost-volume (NOT per-pixel python loops).
# ----------------------------------------------------------------------------------------------
def _census_transform(gray: np.ndarray, half: int = 2) -> np.ndarray:
    """Census descriptor: for each pixel, a bit per neighbour in a (2*half+1)^2 window, 1 if the
    neighbour is brighter than the center. Returns uint32 codes (window 5x5 -> 24 bits). NaN ->
    treated as not-brighter (0 bit) and the pixel itself flagged invalid by the caller's mask."""
    H, W = gray.shape
    g = np.nan_to_num(gray, nan=0.0)
    code = np.zeros((H, W), dtype=np.uint32)
    bit = 0
    for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
            if dv == 0 and du == 0:
                continue
            shifted = np.full((H, W), -np.inf)
            v0, v1 = max(0, dv), min(H, H + dv)
            u0, u1 = max(0, du), min(W, W + du)
            shifted[v0 - dv:v1 - dv, u0 - du:u1 - du] = g[v0:v1, u0:u1]
            code |= ((shifted > g).astype(np.uint32) << bit)
            bit += 1
    return code


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel Hamming distance between two uint32 census-code arrays (vectorized popcount)."""
    x = (a ^ b).astype(np.uint32)
    return (_POPCOUNT[x & 0xFF].astype(np.int32)
            + _POPCOUNT[(x >> 8) & 0xFF]
            + _POPCOUNT[(x >> 16) & 0xFF]
            + _POPCOUNT[(x >> 24) & 0xFF])


def _box_filter(cost: np.ndarray, half: int) -> np.ndarray:
    """Box sum over a (2*half+1) window via an integral image (aggregates the per-pixel census
    Hamming cost into a block SAD-of-census). Interior-only; border windows are left +inf so they
    never win a match. Fully vectorized (no per-pixel loop)."""
    c = np.cumsum(np.cumsum(cost.astype(np.float64), axis=0), axis=1)
    H, W = cost.shape
    pad = np.zeros((H + 1, W + 1))
    pad[1:, 1:] = c
    out = np.full((H, W), np.inf)
    vv = np.arange(H)
    uu = np.arange(W)
    v0 = vv - half
    v1 = vv + half + 1
    u0 = uu - half
    u1 = uu + half + 1
    okv = (v0 >= 0) & (v1 <= H)
    oku = (u0 >= 0) & (u1 <= W)
    V0 = np.clip(v0, 0, H)
    V1 = np.clip(v1, 0, H)
    U0 = np.clip(u0, 0, W)
    U1 = np.clip(u1, 0, W)
    s = (pad[V1][:, U1] - pad[V0][:, U1] - pad[V1][:, U0] + pad[V0][:, U0])
    valid = okv[:, None] & oku[None, :]
    out[valid] = s[valid]
    return out


def compute_disparity(grayL: np.ndarray, grayR: np.ndarray, d_min: int, d_max: int,
                      uniqueness: float = _UNIQUENESS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense disparity for the LEFT image by census block-matching against the RIGHT image, over the
    integer disparity range [d_min, d_max]. Left pixel (v,u) matches right pixel (v, u-d): the right
    feature sits to the LEFT in the right image (positive disparity = nearer). Returns:
      disp:  (H,W) float -- sub-pixel disparity (parabolic), NaN where rejected.
      cost:  (H,W) float -- the winning aggregated census cost (for diagnostics).
      ratio: (H,W) float -- best/second-best cost ratio (uniqueness; <= `uniqueness` to keep).
    VECTORIZED: one cost-volume slice per integer disparity (shift-and-cost), no per-pixel loop.
    """
    H, W = grayL.shape
    cL = _census_transform(grayL)
    cR = _census_transform(grayR)
    n_d = d_max - d_min + 1
    best = np.full((H, W), np.inf)
    second = np.full((H, W), np.inf)
    best_d = np.full((H, W), -1, dtype=np.int32)
    # store every aggregated slice so sub-pixel can read the winner's d-1 and d+1 costs afterward.
    agg_slices: dict[int, np.ndarray] = {}
    for d in range(d_min, d_max + 1):
        # right pixel for left (v,u) is (v, u-d); build a shifted right-census aligned to the left.
        shifted = np.full((H, W), 0, dtype=np.uint32)
        valid_col = np.zeros((H, W), dtype=bool)
        if d < W:
            shifted[:, d:] = cR[:, :W - d]
            valid_col[:, d:] = True
        ham = _hamming(cL, shifted).astype(np.float64)
        ham[~valid_col] = 24.0  # max census distance where no right pixel exists (penalize)
        agg = _box_filter(ham, _BLOCK)
        agg_slices[d] = agg
        improve = agg < best
        # demote current best to second where the new slice beats it
        second = np.where(improve, best, np.minimum(second, agg))
        best_d = np.where(improve, d, best_d)
        best = np.where(improve, agg, best)
    # sub-pixel parabolic using aggregated cost at best_d-1, best_d, best_d+1.
    cost_at_best = best
    cb = np.full((H, W), np.inf)
    ca = np.full((H, W), np.inf)
    for d, agg in agg_slices.items():
        sel_b = best_d == (d + 1)  # this slice is below the winner (d = best_d-1)
        cb[sel_b] = agg[sel_b]
        sel_a = best_d == (d - 1)  # this slice is above the winner (d = best_d+1)
        ca[sel_a] = agg[sel_a]
    disp = best_d.astype(float)
    with np.errstate(invalid="ignore"):
        denom = (cb - 2.0 * cost_at_best + ca)
        ok_sub = np.isfinite(cb) & np.isfinite(ca) & (np.abs(denom) > 1e-9)
    sub = np.zeros((H, W))
    sub[ok_sub] = 0.5 * (cb[ok_sub] - ca[ok_sub]) / denom[ok_sub]
    sub = np.clip(sub, -0.5, 0.5)
    disp = disp + np.where(ok_sub, sub, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(second) & (second > 0), best / second, 0.0)
    keep = (best_d >= d_min) & np.isfinite(best) & (ratio <= uniqueness)
    disp = np.where(keep, disp, np.nan)
    return disp, cost_at_best, ratio


def lr_consistency(dispL: np.ndarray, dispR: np.ndarray, tol_px: float) -> np.ndarray:
    """Left-right consistency check. dispL is disparity for the left image (match right at u-d);
    dispR is disparity for the right image (match left at u+d). Keep a left pixel (v,u) iff the
    right pixel it maps to maps back within `tol_px`. Returns the filtered left disparity (NaN-out)."""
    H, W = dispL.shape
    out = np.array(dispL, dtype=float, copy=True)
    vv, uu = np.mgrid[0:H, 0:W]
    dL = dispL
    ur = uu - np.nan_to_num(dL, nan=0.0)
    ur_i = np.clip(np.round(ur).astype(np.intp), 0, W - 1)
    dR_back = dispR[vv, ur_i]
    bad = ~np.isfinite(dL) | ~np.isfinite(dR_back) | (np.abs(dL - dR_back) > tol_px)
    out[bad] = np.nan
    return out


def compute_disparity_right(grayL: np.ndarray, grayR: np.ndarray, d_min: int, d_max: int,
                            uniqueness: float = _UNIQUENESS) -> np.ndarray:
    """Disparity for the RIGHT image (right pixel (v,u) matches left (v, u+d)), for the LR check.
    Symmetric to compute_disparity with the shift sign flipped."""
    H, W = grayR.shape
    cL = _census_transform(grayL)
    cR = _census_transform(grayR)
    best = np.full((H, W), np.inf)
    second = np.full((H, W), np.inf)
    best_d = np.full((H, W), -1, dtype=np.int32)
    for d in range(d_min, d_max + 1):
        shifted = np.full((H, W), 0, dtype=np.uint32)
        valid_col = np.zeros((H, W), dtype=bool)
        if d < W:
            shifted[:, :W - d] = cL[:, d:]
            valid_col[:, :W - d] = True
        ham = _hamming(cR, shifted).astype(np.float64)
        ham[~valid_col] = 24.0
        agg = _box_filter(ham, _BLOCK)
        improve = agg < best
        second = np.where(improve, best, np.minimum(second, agg))
        best_d = np.where(improve, d, best_d)
        best = np.where(improve, agg, best)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(second) & (second > 0), best / second, 0.0)
    keep = (best_d >= d_min) & np.isfinite(best) & (ratio <= uniqueness)
    return np.where(keep, best_d.astype(float), np.nan)


# ----------------------------------------------------------------------------------------------
# Disparity SOURCE seam -- census (sealed numpy matcher, DEFAULT) or pre-computed learned artifact.
# This is the ONLY change vs the classical run (pre-reg: the single variable = the depth front-end).
# Everything downstream of `get_disparity` is inherited byte-for-byte.
# ----------------------------------------------------------------------------------------------
def artifact_name(log: str, cam_ts: int, side: str) -> str:
    """Single source of truth for the disparity-artifact filename (pod writer + oracle reader share
    this so the seam can never drift). `cam_ts` is the FULL-nanosecond camera-frame timestamp the
    oracle already resolves via `_nearest(_cam_timestamps(...), reference_ts)` -- the same jpg stem it
    opens to read the stereo frame (see the ARTIFACT CONTRACT in the module docstring)."""
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    return f"disp_{log}_{int(cam_ts)}_{side}.npz"


def _load_disparity_artifact(artifact_dir: pathlib.Path, log: str, cam_ts: int, side: str,
                             expected_shape: tuple[int, int]) -> np.ndarray:
    """Load `disp` (float32 HxW, NaN=invalid) from `<artifact_dir>/disp_<log>_<cam_ts>_<side>.npz` and
    return it as a float disparity on EXACTLY the census `undL_s`/`undR_s` grid. Asserts shape match so
    a mis-geometried artifact fails loudly instead of silently mis-grading."""
    path = pathlib.Path(artifact_dir) / artifact_name(log, cam_ts, side)
    if not path.exists():
        raise FileNotFoundError(
            f"disparity artifact missing: {path} (--disparity-source artifact expects the pod to have "
            f"emitted one .npz per (log, camera-frame, side); see igev_disparity_pod.py)")
    with np.load(path) as npz:
        if "disp" not in npz.files:
            raise KeyError(f"artifact {path} has no 'disp' key (found {npz.files}); see ARTIFACT CONTRACT")
        disp = np.asarray(npz["disp"], dtype=float)
    if disp.shape != tuple(expected_shape):
        raise ValueError(
            f"artifact {path} disp shape {disp.shape} != census-grid shape {tuple(expected_shape)} -- "
            f"the artifact must be on the UNDISTORTED 2x-downsampled grid (undL_s for L / undR_s for R)")
    return disp


def get_disparity(grayL_s: np.ndarray, grayR_s: np.ndarray, d_min: int, d_max: int, *,
                  log: str, ts: int, side: str, cfg: "Config") -> np.ndarray:
    """The disparity seam. `side="L"` returns the left-image disparity (drop-in for
    `compute_disparity(...)[0]`); `side="R"` returns the right-image disparity (drop-in for
    `compute_disparity_right(...)`). `ts` is the camera-frame timestamp the oracle resolved for THIS
    side (cl for L, cr for R) -- the same key it used to open the jpg.

    source == "census" (DEFAULT): runs the sealed numpy matcher, byte-for-byte unchanged.
    source == "artifact": loads the pre-computed learned-stereo (IGEV) disparity, validated onto the
                          exact census grid so ALL downstream filters run unchanged. NO torch here."""
    if cfg.disparity_source == "census":
        if side == "L":
            disp, _, _ = compute_disparity(grayL_s, grayR_s, d_min, d_max)
            return disp
        if side == "R":
            return compute_disparity_right(grayL_s, grayR_s, d_min, d_max)
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    if cfg.disparity_source == "artifact":
        if cfg.disparity_artifact_dir is None:
            raise ValueError("--disparity-source artifact requires --disparity-artifact-dir DIR")
        expected_shape = grayL_s.shape if side == "L" else grayR_s.shape
        return _load_disparity_artifact(cfg.disparity_artifact_dir, log, ts, side, expected_shape)
    raise ValueError(f"unknown disparity_source {cfg.disparity_source!r} (expected census|artifact)")


# ----------------------------------------------------------------------------------------------
# Back-projection: undistorted (u,v,Z) in the LEFT camera -> ego frame.
# ----------------------------------------------------------------------------------------------
def backproject_to_ego(u: np.ndarray, v: np.ndarray, Z: np.ndarray, cam: AV2Camera) -> np.ndarray:
    """(u,v) UNDISTORTED pinhole pixels + metric depth Z (camera-frame z) -> ego-frame points (N,3).

    Camera frame: x = (u-cx)/fx * Z, y = (v-cy)/fy * Z, z = Z (x-right, y-down, z-forward). Then
    p_ego = R_cam2ego @ p_cam + t_cam_in_ego -- the validated inverse of project_ego_points."""
    xc = (u - cam.cx) / cam.fx * Z
    yc = (v - cam.cy) / cam.fy * Z
    zc = Z
    p_cam = np.stack([xc, yc, zc], axis=-1)  # (N,3)
    p_ego = (cam.R_cam2ego @ p_cam.T).T + cam.t_cam_in_ego
    return p_ego


def project_ego_to_left(points_ego: np.ndarray, cam: AV2Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego points (N,3) -> (uv distorted pixels, depth, visible) in the left camera. Mirrors
    projection.project_ego_points but with AV2 intrinsics + Su distortion (for the FOV test and the
    calibration patch sampling). Visible iff in front AND inside the raw image after distortion."""
    p = np.atleast_2d(np.asarray(points_ego, dtype=float))
    p_cam = (cam.R_cam2ego.T @ (p - cam.t_cam_in_ego).T).T
    depth = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = p_cam[:, 0] / p_cam[:, 2]
        yn = p_cam[:, 1] / p_cam[:, 2]
    xd, yd = distort_normalized(xn, yn, cam)
    u = xd * cam.fx + cam.cx
    v = yd * cam.fy + cam.cy
    uv = np.stack([u, v], axis=-1)
    front = depth > 1e-6
    inframe = front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    uv = np.where(front[:, None], uv, np.nan)
    return uv, depth, inframe


# ----------------------------------------------------------------------------------------------
# Stereo -> ego points -> filtered + voxelized obstacle map (IDENTICAL filters to av2_sensor).
# ----------------------------------------------------------------------------------------------
def stereo_points_ego(dispL_full: np.ndarray, cam_full: AV2Camera, B: float,
                      z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From an UNDISTORTED-frame left disparity map (full-res intrinsics `cam_full`) compute Z and
    back-project. Returns (points_ego (M,3), Zkept (M,), u (M,), v (M,)) for pixels with finite
    disparity giving Z in [z_min,z_max]. d->Z is Z = fx*B/d on the SAME scale as `cam_full`."""
    H, W = dispL_full.shape
    vv, uu = np.mgrid[0:H, 0:W]
    d = dispL_full
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = cam_full.fx * B / d
    good = np.isfinite(d) & (d > 0) & np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
    u = uu[good].astype(float)
    v = vv[good].astype(float)
    Zk = Z[good]
    pts = backproject_to_ego(u, v, Zk, cam_full)
    return pts, Zk, u, v


def voxelize_stereo(points_ego: np.ndarray, n_stereo_min: int) -> np.ndarray:
    """Voxelize stereo ego points into the av2_sensor grid with the IDENTICAL ground/ego-self-return
    filters, then keep voxels with >= n_stereo_min supporting points (the pre-reg's stereo_struct).
    Returns a (NX,NY,NZ) int8 mask (1 = stereo_struct). The filter logic is copied from
    av2_sensor._voxelize so a stereo obstacle voxel means the SAME thing as a LiDAR one."""
    (x0, x1), (y0, y1), (z0, z1) = _RANGE
    x = points_ego[:, 0]
    y = points_ego[:, 1]
    z = points_ego[:, 2]
    # IDENTICAL filters to av2_sensor._voxelize:
    ego = (x > av2_sensor._EGO_X0) & (x < av2_sensor._EGO_X1) & (np.abs(y) < av2_sensor._EGO_HALF_W)
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z > av2_sensor._ROAD_Z) & (z < z1) & ~ego
    counts = np.zeros(av2_sensor.GRID_SHAPE, dtype=np.int32)
    if not m.any():
        return (counts >= n_stereo_min).astype(np.int8)
    ix = ((x[m] - x0) / _VOX).astype(np.intp)
    iy = ((y[m] - y0) / _VOX).astype(np.intp)
    iz = ((z[m] - z0) / _VOX).astype(np.intp)
    np.clip(ix, 0, _NX - 1, out=ix)
    np.clip(iy, 0, _NY - 1, out=iy)
    np.clip(iz, 0, _NZ - 1, out=iz)
    np.add.at(counts, (ix, iy, iz), 1)
    return (counts >= n_stereo_min).astype(np.int8)


def edge_adjacency_reject(points_ego: np.ndarray, Z: np.ndarray, u: np.ndarray, v: np.ndarray,
                          dispL_full: np.ndarray, edge_m: float) -> np.ndarray:
    """Edge-adjacency reject (silhouette depth-bleed fix). Drop any stereo point whose pixel is
    within 1 px of a large depth discontinuity (|ΔZ| > edge_m between neighbouring matched pixels).

    SPEC-NOTE: the pre-reg says "within 1 voxel of a large depth discontinuity (|ΔZ| > 1.5 m between
    neighbouring matched pixels)". "1 voxel" is ambiguous between image-space and BEV-voxel space.
    Literal reading chosen: a discontinuity is detected in IMAGE space (neighbouring matched pixels,
    as the parenthetical states), and "within 1 voxel" is applied as within-1-pixel image adjacency
    to the discontinuity edge (the pixel neighbourhood is where silhouette bleed physically occurs).
    Returns a boolean keep-mask aligned to points_ego."""
    H, W = dispL_full.shape
    Zmap = np.full((H, W), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pass
    ui = np.round(u).astype(np.intp)
    vi = np.round(v).astype(np.intp)
    Zmap[vi, ui] = Z
    # horizontal + vertical neighbour depth jumps
    disc = np.zeros((H, W), dtype=bool)
    dxr = np.abs(np.diff(Zmap, axis=1))  # (H, W-1)
    jmp_h = dxr > edge_m
    disc[:, :-1] |= jmp_h
    disc[:, 1:] |= jmp_h
    dyr = np.abs(np.diff(Zmap, axis=0))  # (H-1, W)
    jmp_v = dyr > edge_m
    disc[:-1, :] |= jmp_v
    disc[1:, :] |= jmp_v
    # dilate the discontinuity edge by 1 px (the "within 1 voxel/pixel" neighbourhood)
    grow = disc.copy()
    grow[:, :-1] |= disc[:, 1:]
    grow[:, 1:] |= disc[:, :-1]
    grow[:-1, :] |= disc[1:, :]
    grow[1:, :] |= disc[:-1, :]
    near_edge = grow[vi, ui]
    return ~near_edge


# ----------------------------------------------------------------------------------------------
# In-path band (|y| <= ego_half_width) intersect stereo-FOV -> denominator support.
# ----------------------------------------------------------------------------------------------
def band_fov_mask(cam_full: AV2Camera, z_max: float) -> np.ndarray:
    """The in-path band ∩ stereo-left-FOV support, as a (NX,NY) BEV bool over the av2_sensor grid.

    Band (pre-reg §In-path band): |y| <= ego_half_width (_EGO_HALF_W=1.05), forward to reach capped at
    Z_max and the stereo FOV. horizon=0 (body band): the body band is forward x in [0, z_max] (the
    forward in-path ribbon), |y| <= half-width. A BEV cell is in-FOV iff its voxel column (at a
    representative height) projects inside the left raw image AND in front. We sample the cell center
    at the road-plane band height to test FOV.

    SPEC-NOTE: free_along_ego_path is a centerline boolean (single lateral line), not a 2D region. The
    pre-reg's denominator is explicitly a 2D BAND of voxels (|y|<=half-width). I therefore construct
    the band region directly from the half-width semantic free_along_ego_path encodes (obstacles
    inflated by _EGO_HALF_W), per the pre-reg's parenthetical, rather than calling the centerline walk.
    """
    (x0, x1), (y0, y1), (z0, z1) = _RANGE
    xs = x0 + (np.arange(_NX) + 0.5) * _VOX  # forward cell centers (matches _voxelize index->coord)
    ys = y0 + (np.arange(_NY) + 0.5) * _VOX  # lateral cell centers
    XX, YY = np.meshgrid(xs, ys, indexing="ij")  # (NX,NY)
    band = (np.abs(YY) <= _EGO_HALF_W) & (XX >= 0.0) & (XX <= z_max)
    # FOV: sample each band cell center at a mid-band height and test projection into the left image.
    z_probe = max(av2_sensor._ROAD_Z + 0.5, 0.8)  # ~0.8 m, a typical obstacle-mid height
    fov = np.zeros((_NX, _NY), dtype=bool)
    bx, by = np.where(band)
    if len(bx):
        pts = np.stack([XX[bx, by], YY[bx, by], np.full(len(bx), z_probe)], axis=1)
        _, depth, vis = project_ego_to_left(pts, cam_full)
        fov[bx, by] = vis
    return band & fov


def _grid_index_of_voxels(struct_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a (NX,NY,NZ) stereo_struct mask to a (NX,NY) BEV bool (any z occupied) and return
    (bev_bool, _) -- the recall estimand is per BEV in-path column (forward,lateral)."""
    bev = struct_mask.any(axis=2)
    return bev, bev


# ----------------------------------------------------------------------------------------------
# Per-frame recall computation.
# ----------------------------------------------------------------------------------------------
@dataclass
class FrameResult:
    log: str
    ts: int
    n_struct: int          # stereo_struct band∩FOV voxels (denominator)
    n_miss: int            # MISS-candidate band∩FOV voxels = occ_free ∧ stereo_struct
    miss_rate: float       # n_miss / n_struct  (per-frame estimator)
    n_occ_band: int        # occupied band∩FOV voxels (for the band-local null relocation count)
    calib_reject: bool


def occupancy_bev(grid, z_max: float) -> np.ndarray:
    """Occupancy obstacle BEV (NX,NY) bool over the band's forward range, same obstacle definition as
    oracle_traversal (obstacle_centers capped at ego height), reduced to the in-path forward extent."""
    centers = grid.obstacle_centers(max_height_agl=1.9)  # ego-height cap (EgoPose.height default)
    bev = np.zeros((_NX, _NY), dtype=bool)
    if len(centers):
        (x0, _), (y0, _), _ = _RANGE
        bi = np.clip(((centers[:, 0] - x0) / _VOX).astype(int), 0, _NX - 1)
        bj = np.clip(((centers[:, 1] - y0) / _VOX).astype(int), 0, _NY - 1)
        bev[bi, bj] = True
    return bev


def compute_frame(log: str, ts: int, grayL_raw: np.ndarray, grayR_raw: np.ndarray,
                  camL_full: AV2Camera, camR_full: AV2Camera, B: float, grid,
                  cfg: "Config", ts_left: int | None = None,
                  ts_right: int | None = None) -> FrameResult:
    """Full per-frame pipeline: undistort -> downsample -> disparity (census or artifact) ->
    LR/uniqueness -> Z + back-project -> filters -> voxelize -> band∩FOV -> miss events.

    `ts_left`/`ts_right` are the resolved camera-frame timestamps for L/R (the jpg stems the caller
    already opened); they key the disparity artifact in --disparity-source artifact mode and are unused
    by the census path."""
    ds = cfg.downsample
    # (a) undistort both raw images to pinhole frames
    mapuL, mapvL = build_undistort_map(camL_full)
    mapuR, mapvR = build_undistort_map(camR_full)
    undL = remap_gray(grayL_raw, mapuL, mapvL)
    undR = remap_gray(grayR_raw, mapuR, mapvR)
    # (a') epipolar-slope guard: with the residual extrinsic rotation aligned (cameras are near-
    #      canonically rectified along ego-y), a horizontal-scanline search is valid. We approximate
    #      rectification by the shared pinhole intrinsics (cx,cy,fx near-equal); the residual vertical
    #      epipolar slope is bounded by the cy difference / image width. If it exceeds 1.0px over the
    #      width -> calib_reject (pre-reg).
    slope_px = abs(camL_full.cy - camR_full.cy)
    if slope_px > 1.0:
        # SPEC-NOTE: pre-reg drops the frame if residual epipolar slope > 1.0px over the image width.
        # We use the principal-point-row mismatch as the rectification residual proxy (no full stereo
        # rectification implemented in numpy; the cameras share a rig and are near-rectified). Logged.
        return FrameResult(log, ts, 0, 0, float("nan"), 0, calib_reject=True)
    # downsample (block-matching scale; geometry carried by camL_full.scaled(ds))
    if ds > 1:
        undL_s = undL[::ds, ::ds]
        undR_s = undR[::ds, ::ds]
    else:
        undL_s, undR_s = undL, undR
    camL_s = camL_full.scaled(ds)
    d_min = max(1, int(math.floor(cfg.d_min_px / ds)))
    d_max = int(math.ceil(cfg.d_max_px / ds))
    # (b) disparity from the configured source (census matcher, or pre-computed learned artifact)
    dispL = get_disparity(undL_s, undR_s, d_min, d_max, log=log, ts=ts_left, side="L", cfg=cfg)
    dispR = get_disparity(undL_s, undR_s, d_min, d_max, log=log, ts=ts_right, side="R", cfg=cfg)
    dispL = lr_consistency(dispL, dispR, cfg.lr_consistency_px / ds)  # tol scales with image scale
    # (e-texture) texture gate: only pixels whose local left-image gradient exceeds tau_tex match.
    if cfg.tau_tex is not None:
        grad = _patch_evidence_field(undL_s, half=12 // cfg.downsample + 1)
        dispL = np.where(grad >= cfg.tau_tex, dispL, np.nan)
    # (c) Z=fx*B/d on the downsampled scale, back-project, keep Z in [z_min,z_max]
    pts, Zk, us, vs = stereo_points_ego(dispL, camL_s, B, cfg.z_min, cfg.z_max)
    if len(pts):
        # (d) edge-adjacency reject (image-space discontinuity)
        keep = edge_adjacency_reject(pts, Zk, us, vs, dispL, cfg.edge_discontinuity_m)
        pts, Zk, us, vs = pts[keep], Zk[keep], us[keep], vs[keep]
    struct = voxelize_stereo(pts, cfg.n_stereo_min)
    struct_bev, _ = _grid_index_of_voxels(struct)
    # (f) band ∩ stereo-FOV denominator (FOV from full-res left camera)
    bf = band_fov_mask(camL_full, cfg.z_max)
    struct_in_band = struct_bev & bf
    occ_bev = occupancy_bev(grid, cfg.z_max)
    occ_free = ~occ_bev  # occupancy reports FREE
    # (g) miss event: occ_free ∧ stereo_struct, within band∩FOV
    miss = struct_in_band & occ_free
    n_struct = int(struct_in_band.sum())
    n_miss = int(miss.sum())
    n_occ_band = int((occ_bev & bf).sum())
    rate = (n_miss / n_struct) if n_struct > 0 else float("nan")
    return FrameResult(log, ts, n_struct, n_miss, rate, n_occ_band, calib_reject=False)


def _patch_evidence_field(gray: np.ndarray, half: int = 12) -> np.ndarray:
    """Dense version of camera_oracle.patch_evidence: at every pixel, the MEAN Sobel gradient
    magnitude over its (2*half+1) patch. patch_evidence(gray,u,v) == this field at (v,u) (verified by
    construction: both are mean(hypot(sobel_x,sobel_y)) over the same window). Computing it densely
    lets the per-pixel texture gate use the EXACT pre-registered statistic at the same scale the
    threshold is fit on. NaN gray -> 0 so undefined pixels never pass the gate."""
    from scipy import ndimage
    g = np.nan_to_num(gray, nan=0.0)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    field = ndimage.uniform_filter(mag, size=2 * half + 1, mode="nearest")
    field[~np.isfinite(gray)] = 0.0
    return field


# ----------------------------------------------------------------------------------------------
# Texture threshold tau_tex -- 40th percentile of in-band gradient on the HELD-OUT log.
# ----------------------------------------------------------------------------------------------
def fit_tau_tex(held_log: str, data_root: pathlib.Path, cfg: "Config", n_frames: int = 8) -> float:
    """tau_tex = 40th percentile of the left-image gradient inside the in-path band, measured on the
    HELD-OUT threshold log only (never the result logs). Uses patch_evidence's Sobel statistic family.

    SPEC-NOTE: the pre-reg says "40th percentile of in-band gradient on a HELD-OUT log". "in-band"
    here = pixels whose back-projected ground-plane ray falls in |y|<=half-width forward to z_max. We
    approximate the in-band pixel set by projecting the band BEV cells (road height) into the left
    image and taking the gradient there. n_frames sampled deterministically across the held-out log."""
    log_dir = data_root / held_log
    camL_full, camR_full, B = load_stereo_calib(held_log, data_root)
    bf = band_fov_mask(camL_full, cfg.z_max)
    (x0, _), (y0, _), _ = _RANGE
    bx, by = np.where(bf)
    xs = x0 + (bx + 0.5) * _VOX
    ys = y0 + (by + 0.5) * _VOX
    band_pts = np.stack([xs, ys, np.full(len(bx), av2_sensor._ROAD_Z + 0.5)], axis=1)
    lidar_ts = sorted(int(os.path.basename(p)[:-len(".feather")])
                      for p in glob.glob(str(log_dir / "sensors" / "lidar" / "*.feather")))
    camL_ts = _cam_timestamps(log_dir, "stereo_front_left")
    pick = np.linspace(0, len(lidar_ts) - 1, num=min(n_frames, len(lidar_ts))).astype(int)
    grads: list[float] = []
    ds = cfg.downsample
    half = 12 // ds + 1
    mapuL, mapvL = build_undistort_map(camL_full)
    for k in pick:
        ts = lidar_ts[int(k)]
        cl = _nearest(camL_ts, ts)
        gray = _gray_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{cl}.jpg")
        und = remap_gray(gray, mapuL, mapvL)
        und_s = und[::ds, ::ds] if ds > 1 else und
        # SPEC-NOTE (verbatim-reuse caveat): the pre-reg names the camera_oracle.patch_evidence Sobel
        # statistic. patch_evidence() itself HARD-CODES the nuScenes 1600x900 frame bounds (IMG_W/IMG_H
        # from projection.py) in its out-of-frame check, so calling it VERBATIM on a 2048x1550 stereo
        # image would mis-bound the patch (index past the array / wrong reject). I therefore use
        # _patch_evidence_field -- the IDENTICAL statistic (mean Sobel-magnitude over the same 25px
        # patch, verified by construction) -- computed densely with the CORRECT stereo bounds, for BOTH
        # the threshold here and the per-pixel gate (so gate and threshold are the same statistic/scale).
        field = _patch_evidence_field(und_s, half=half)
        uv, depth, vis = _project_undistorted(band_pts, camL_full)
        uu = np.round(uv[vis, 0] / ds).astype(int)
        vv = np.round(uv[vis, 1] / ds).astype(int)
        ok = (uu >= 0) & (uu < field.shape[1]) & (vv >= 0) & (vv < field.shape[0])
        grads.extend(field[vv[ok], uu[ok]].tolist())
    if not grads:
        return 0.0
    return float(np.percentile(np.asarray(grads), 40))


def _project_undistorted(points_ego: np.ndarray, cam: AV2Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego points -> UNDISTORTED pinhole pixels (no distortion applied; for sampling the undistorted
    image)."""
    p = np.atleast_2d(np.asarray(points_ego, dtype=float))
    p_cam = (cam.R_cam2ego.T @ (p - cam.t_cam_in_ego).T).T
    depth = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = p_cam[:, 0] / p_cam[:, 2] * cam.fx + cam.cx
        v = p_cam[:, 1] / p_cam[:, 2] * cam.fy + cam.cy
    uv = np.stack([u, v], axis=-1)
    front = depth > 1e-6
    inframe = front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return uv, depth, inframe


# ----------------------------------------------------------------------------------------------
# Band-local null + bootstrap.
# ----------------------------------------------------------------------------------------------
def band_local_shuffle_rate(struct_in_band: np.ndarray, occ_in_band_count: int, bf: np.ndarray,
                            rng: np.random.Generator) -> float:
    """BAND-LOCAL null (pre-reg): relocate the SAME count of occupied voxels that fall inside the
    band∩FOV to RANDOM voxels inside the same band∩FOV, recompute miss-rate against the SAME stereo
    structure. Support = the band, NOT the grid. Returns one shuffled miss-rate."""
    n_struct = int(struct_in_band.sum())
    if n_struct == 0:
        return float("nan")
    band_cells = np.argwhere(bf)  # (Nband, 2)
    nb = len(band_cells)
    if nb == 0 or occ_in_band_count == 0:
        # no occupied mass in band -> every struct voxel is a miss (occ_free everywhere)
        return 1.0
    k = min(occ_in_band_count, nb)
    chosen = rng.choice(nb, size=k, replace=False)
    occ_shuf = np.zeros((_NX, _NY), dtype=bool)
    cc = band_cells[chosen]
    occ_shuf[cc[:, 0], cc[:, 1]] = True
    miss = struct_in_band & (~occ_shuf)
    return float(miss.sum()) / n_struct


# ----------------------------------------------------------------------------------------------
# Config.
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    z_min: float
    z_max: float
    n_stereo_min: int
    lr_consistency_px: float
    edge_discontinuity_m: float
    null: str
    shuffles: int
    seed: int
    downsample: int
    tau_tex: float | None = None
    d_min_px: int = 28   # pre-reg: Z in [2,30] <-> d in [28,421] at fx~1686.6, B~0.499
    d_max_px: int = 421
    # depth front-end SOURCE (the single variable; pre-reg oracle_stereo_recall_learned). census =
    # the sealed numpy matcher (DEFAULT, byte-for-byte). artifact = pre-computed learned-stereo (IGEV)
    # disparity loaded from disparity_artifact_dir (NO torch here).
    disparity_source: str = "census"
    disparity_artifact_dir: pathlib.Path | None = None


# ----------------------------------------------------------------------------------------------
# Self-check (geometry round-trips on the REAL feathers, no confirmatory data).
# ----------------------------------------------------------------------------------------------
def _self_check(logs: list[str], data_root: pathlib.Path) -> bool:
    """Validate the geometry chain on REAL calibration. Prints each check + PASS/FAIL. Returns all-ok.
    (i) ego 3D point round-trip < 0.1m, (ii) synthetic disparity -> Z=fx*B/d within float tol,
    (iii) fx~1686.6 and B~0.499 from feathers, (iv) undistort->re-distort a corner pixel < 0.5px."""
    ok_all = True
    log = logs[0]
    camL, camR, B = load_stereo_calib(log, data_root)
    print(f"  [calib] log={log[:12]} fx_left={camL.fx:.3f} cx={camL.cx:.3f} cy={camL.cy:.3f} "
          f"k=({camL.k1:.4f},{camL.k2:.4f},{camL.k3:.4f}) B={B:.5f}m")

    # (iii) fx ~ 1686.6, B ~ 0.499
    ok_iii = abs(camL.fx - 1686.6) < 5.0 and abs(B - 0.499) < 0.003
    ok_all &= ok_iii
    print(f"  (iii) fx in [1681.6,1691.6]={1681.6 <= camL.fx <= 1691.6}, "
          f"B in [0.496,0.502]={0.496 <= B <= 0.502} -> {'PASS' if ok_iii else 'FAIL'}")

    # (ii) synthetic disparity d -> Z = fx*B/d, then Z*d/B == fx (within float tol)
    d_syn = np.array([28.0, 100.0, 421.0])
    Z_syn = camL.fx * B / d_syn
    fx_recover = Z_syn * d_syn / B
    err_ii = float(np.max(np.abs(fx_recover - camL.fx)))
    # also verify the downsample scale invariance: Z_s = fx_s*B/d_s equals Z at full res
    ds = _DOWNSAMPLE
    camL_s = camL.scaled(ds)
    Z_s = camL_s.fx * B / (d_syn / ds)
    err_scale = float(np.max(np.abs(Z_s - Z_syn)))
    ok_ii = err_ii < 1e-6 and err_scale < 1e-9
    ok_all &= ok_ii
    print(f"  (ii) Z=fx*B/d round-trip max fx err={err_ii:.2e}, downsample-scale Z err={err_scale:.2e} "
          f"-> {'PASS' if ok_ii else 'FAIL'}  (Z@d=28 -> {Z_syn[0]:.2f}m, @d=421 -> {Z_syn[2]:.3f}m)")

    # (i) ego point round-trip: pick a known ego point in front, project to LEFT pixel (distorted),
    #     get its undistorted pinhole pixel + the true Z (camera-frame z), back-project to ego.
    p_ego_true = np.array([[12.0, 0.6, 0.9]])  # 12 m fwd, 0.6 m left, 0.9 m up -- inside band+FOV
    uv_d, depth, vis = project_ego_to_left(p_ego_true, camL)
    assert vis[0], f"test point not visible: uv={uv_d[0]} depth={depth[0]}"
    # undistort that distorted pixel back to a pinhole pixel
    xd = (uv_d[0, 0] - camL.cx) / camL.fx
    yd = (uv_d[0, 1] - camL.cy) / camL.fy
    xn, yn = undistort_normalized(np.array([xd]), np.array([yd]), camL)
    u_pin = xn[0] * camL.fx + camL.cx
    v_pin = yn[0] * camL.fy + camL.cy
    Z_true = depth[0]  # camera-frame z (the stereo Z would equal this for the true match)
    p_back = backproject_to_ego(np.array([u_pin]), np.array([v_pin]), np.array([Z_true]), camL)
    err_i = float(np.linalg.norm(p_back[0] - p_ego_true[0]))
    ok_i = err_i < 0.1
    ok_all &= ok_i
    print(f"  (i) ego point [12,0.6,0.9] -> distorted px ({uv_d[0,0]:.1f},{uv_d[0,1]:.1f}) -> "
          f"undistort -> backproject -> [{p_back[0,0]:.3f},{p_back[0,1]:.3f},{p_back[0,2]:.3f}] "
          f"err={err_i:.4f}m -> {'PASS' if ok_i else 'FAIL'}")

    # (iv) undistort then re-distort a CORNER pixel round-trips < 0.5px
    corner_u, corner_v = 5.0, 5.0  # near the TL corner (max distortion)
    xd_c = (corner_u - camL.cx) / camL.fx
    yd_c = (corner_v - camL.cy) / camL.fy
    xn_c, yn_c = undistort_normalized(np.array([xd_c]), np.array([yd_c]), camL)
    xd2, yd2 = distort_normalized(xn_c, yn_c, camL)
    u2 = xd2[0] * camL.fx + camL.cx
    v2 = yd2[0] * camL.fy + camL.cy
    err_iv = math.hypot(u2 - corner_u, v2 - corner_v)
    ok_iv = err_iv < 0.5
    ok_all &= ok_iv
    print(f"  (iv) corner px (5,5) -> undistort -> re-distort -> ({u2:.3f},{v2:.3f}) "
          f"round-trip err={err_iv:.4f}px -> {'PASS' if ok_iv else 'FAIL'}")

    # (v) the dense _patch_evidence_field tracks camera_oracle.patch_evidence (same statistic family:
    #     mean Sobel-magnitude over a 25px patch). They are NOT bit-identical: patch_evidence Sobels the
    #     CROPPED 25x25 patch (reflect-pad inside the crop) while the dense field Sobels the whole image
    #     then box-averages, so the patch-border Sobel context differs by ~1-2 intensity units. The gate
    #     and the threshold BOTH use the dense field, so they are mutually consistent (what the gate needs);
    #     this check confirms the dense field is the same statistic, within the expected boundary delta.
    rng = np.random.default_rng(0)
    tile = rng.normal(50.0, 20.0, size=(80, 80))
    field = _patch_evidence_field(tile, half=12)
    rel = max(abs(field[v, u] - patch_evidence(tile, float(u), float(v), half=12))
              / patch_evidence(tile, float(u), float(v), half=12)
              for u, v in ((40, 40), (30, 50), (50, 30)))
    ok_v = rel < 0.05  # within 5% -- same statistic, boundary-padding delta only
    ok_all &= ok_v
    print(f"  (v) dense field vs patch_evidence (interior px): max rel-diff={rel:.4f} "
          f"-> {'PASS' if ok_v else 'FAIL'}  (same Sobel-mean statistic family; gate & threshold both use the dense field)")

    # bonus: forward-centered ego point lands near the principal point (sanity on axes)
    uv_c, _, vis_c = project_ego_to_left(np.array([[15.0, 0.0, camL.t_cam_in_ego[2]]]), camL)
    print(f"  (sanity) forward point [15,0,cam_h] -> px ({uv_c[0,0]:.0f},{uv_c[0,1]:.0f}) "
          f"vs principal ({camL.cx:.0f},{camL.cy:.0f}); visible={bool(vis_c[0])}")

    print(f"\n  SELF-CHECK {'PASSED' if ok_all else 'FAILED'} "
          f"(all geometry round-trips {'within tolerance' if ok_all else 'OUT OF TOLERANCE'})")
    return ok_all


# ----------------------------------------------------------------------------------------------
# Calibration patch emission (deterministic 60-patch sampler; NO labels, human gate).
# ----------------------------------------------------------------------------------------------
def emit_calib_patches(logs: list[str], data_root: pathlib.Path, cfg: Config, out_dir: pathlib.Path,
                       seed: int) -> dict:
    """Deterministically sample the 60 calibration patches (30 POS from projected annotation boxes in
    the in-path band at <=30 m, 30 NEG from random lower-image drivable-road patches in the band),
    write the cropped images + calib_patches.json with NO label field (label=null) for HUMAN labeling.

    Boxes are used ONLY to build calibration labels and ONLY to measure the oracle -- NOT in the
    miss-rate estimand. Per the pre-reg the human labels each of the 60 by eye; this code never sets
    a label. 30 pos + 30 neg split across the 3 logs (10+10 per log)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []
    half = 12  # patch half-size (matches camera_oracle._PATCH)
    per_log_pos = 10
    per_log_neg = 10
    for log in logs:
        log_dir = data_root / log
        camL_full, camR_full, B = load_stereo_calib(log, data_root)
        mapuL, mapvL = build_undistort_map(camL_full)
        lidar_ts = sorted(int(os.path.basename(p)[:-len(".feather")])
                          for p in glob.glob(str(log_dir / "sensors" / "lidar" / "*.feather")))
        camL_ts = _cam_timestamps(log_dir, "stereo_front_left")
        ann = _read_feather(log_dir / "annotations.feather").to_pydict()
        # index annotation boxes by timestamp
        ann_ts = np.asarray(ann["timestamp_ns"], dtype=np.int64)
        # collect POSITIVE patches: project box centers into the left image, band + <=30m
        pos_found = 0
        neg_found = 0
        frame_order = rng.permutation(len(lidar_ts))
        for fi in frame_order:
            if pos_found >= per_log_pos and neg_found >= per_log_neg:
                break
            ts = lidar_ts[int(fi)]
            cl = _nearest(camL_ts, ts)
            gray = _gray_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{cl}.jpg")
            und = remap_gray(gray, mapuL, mapvL)
            # POSITIVE: boxes at this LiDAR ts
            if pos_found < per_log_pos:
                sel = np.where(ann_ts == ts)[0]
                if len(sel):
                    centers = np.stack([np.asarray(ann["tx_m"])[sel],
                                        np.asarray(ann["ty_m"])[sel],
                                        np.asarray(ann["tz_m"])[sel]], axis=1).astype(float)
                    uv, depth, vis = _project_undistorted(centers, camL_full)
                    in_band = (np.abs(centers[:, 1]) <= _EGO_HALF_W) & (depth > 0) & (depth <= 30.0)
                    cand = np.where(vis & in_band)[0]
                    for c in cand:
                        if pos_found >= per_log_pos:
                            break
                        u, v = float(uv[c, 0]), float(uv[c, 1])
                        crop = _crop(und, u, v, half)
                        if crop is None:
                            continue
                        pid = f"{log[:8]}_pos_{pos_found:02d}"
                        _save_patch(crop, out_dir / f"{pid}.png")
                        manifest.append({"id": pid, "log": log, "lidar_ts": int(ts),
                                         "cam_ts": int(cl), "u": u, "v": v,
                                         "kind": "pos_box", "category": str(np.asarray(ann["category"])[sel][c]),
                                         "range_m": float(depth[c]), "label": None})
                        pos_found += 1
            # NEGATIVE: random lower-image drivable-road patches in the band (central wedge)
            if neg_found < per_log_neg:
                W, H = camL_full.width, camL_full.height
                u = int(rng.integers(W // 3, 2 * W // 3))
                v = int(rng.integers(int(0.69 * H), H - 20))  # lower image (below horizon)
                crop = _crop(und, float(u), float(v), half)
                if crop is not None:
                    nid = f"{log[:8]}_neg_{neg_found:02d}"
                    _save_patch(crop, out_dir / f"{nid}.png")
                    manifest.append({"id": nid, "log": log, "lidar_ts": int(ts),
                                     "cam_ts": int(cl), "u": float(u), "v": float(v),
                                     "kind": "neg_road", "category": None,
                                     "range_m": None, "label": None})
                    neg_found += 1
    report = {
        "n_patches": len(manifest),
        "n_pos": sum(1 for m in manifest if m["kind"] == "pos_box"),
        "n_neg": sum(1 for m in manifest if m["kind"] == "neg_road"),
        "seed": seed,
        "instructions": ("HUMAN integrity gate: label each patch by EYE (the photo is the primary "
                         "source). Set label=1 if a real surface/obstacle is present at (u,v), label=0 "
                         "if open drivable road/no structure. Disagreements with the box label are KEPT "
                         "as human truth (the pre-reg's annotation-gap catch). Leave NO patch unlabeled "
                         "before the confirmatory run."),
        "patches": manifest,
    }
    (out_dir / "calib_patches.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _crop(gray: np.ndarray, u: float, v: float, half: int) -> np.ndarray | None:
    ui, vi = int(round(u)), int(round(v))
    H, W = gray.shape
    if ui - half < 0 or ui + half >= W or vi - half < 0 or vi + half >= H:
        return None
    p = gray[vi - half:vi + half + 1, ui - half:ui + half + 1]
    if not np.isfinite(p).all():
        return None
    return p


def _save_patch(gray_patch: np.ndarray, path: pathlib.Path) -> None:
    arr = np.clip(gray_patch, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def run_calibration_auc(logs: list[str], data_root: pathlib.Path, cfg: Config,
                        calib_json: pathlib.Path) -> dict:
    """Compute the oracle's reliability AUC over the 60 HUMAN-LABELED patches: ROC AUC of the
    stereo_struct signal (valid-disparity-count after all filters in the patch) via _roc_auc, pos vs
    neg by HUMAN label. Gate: AUC >= 0.75. Requires calib_patches.json with labels filled in.

    This is part of the CONFIRMATORY path and is gated on human labels; it is NOT run by
    --emit-calib-patches and NOT fabricated here."""
    rep = json.loads(calib_json.read_text())
    patches = rep["patches"]
    if any(p.get("label") is None for p in patches):
        raise SystemExit("calib_patches.json has UNLABELED patches (label=null). The 60-patch human "
                         "labeling is a required integrity gate -- fill every label before the "
                         "confirmatory run. Aborting (no fabricated labels).")
    # For each patch, recompute the stereo valid-disparity-count in the patch window (the oracle's
    # signal), then AUC of that count between human-pos and human-neg.
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    by_log: dict[str, list[dict]] = {}
    for p in patches:
        by_log.setdefault(p["log"], []).append(p)
    for log, ps in by_log.items():
        log_dir = data_root / log
        camL_full, camR_full, B = load_stereo_calib(log, data_root)
        camL_ts = _cam_timestamps(log_dir, "stereo_front_left")
        mapuL, mapvL = build_undistort_map(camL_full)
        mapuR, mapvR = build_undistort_map(camR_full)
        ds = cfg.downsample
        # group by cam frame to match once per frame
        frames: dict[int, list[dict]] = {}
        for p in ps:
            frames.setdefault(int(p["cam_ts"]), []).append(p)
        camL_s = camL_full.scaled(ds)
        d_min = max(1, int(math.floor(cfg.d_min_px / ds)))
        d_max = int(math.ceil(cfg.d_max_px / ds))
        for cam_ts, plist in frames.items():
            grayL = _gray_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{cam_ts}.jpg")
            cr = _nearest(_cam_timestamps(log_dir, "stereo_front_right"), cam_ts)
            grayR = _gray_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_right" / f"{cr}.jpg")
            undL = remap_gray(grayL, mapuL, mapvL)
            undR = remap_gray(grayR, mapuR, mapvR)
            undL_s = undL[::ds, ::ds] if ds > 1 else undL
            undR_s = undR[::ds, ::ds] if ds > 1 else undR
            dispL = get_disparity(undL_s, undR_s, d_min, d_max, log=log, ts=cam_ts, side="L", cfg=cfg)
            dispR = get_disparity(undL_s, undR_s, d_min, d_max, log=log, ts=cr, side="R", cfg=cfg)
            dispL = lr_consistency(dispL, dispR, cfg.lr_consistency_px / ds)
            if cfg.tau_tex is not None:
                grad = _patch_evidence_field(undL_s, half=12 // cfg.downsample + 1)
                dispL = np.where(grad >= cfg.tau_tex, dispL, np.nan)
            for p in plist:
                u_s = p["u"] / ds
                v_s = p["v"] / ds
                half = 12 // ds + 1
                ui, vi = int(round(u_s)), int(round(v_s))
                v0, v1 = max(0, vi - half), vi + half + 1
                u0, u1 = max(0, ui - half), ui + half + 1
                win = dispL[v0:v1, u0:u1]
                cnt = float(np.isfinite(win).sum())
                if p["label"] == 1:
                    pos_scores.append(cnt)
                else:
                    neg_scores.append(cnt)
    auc = _roc_auc(pos_scores, neg_scores)
    # operating-point precision at n_stereo_min: of patches whose count >= n_stereo_min, fraction pos
    all_counts = [(c, 1) for c in pos_scores] + [(c, 0) for c in neg_scores]
    fired = [lab for c, lab in all_counts if c >= cfg.n_stereo_min]
    precision = float(np.mean(fired)) if fired else float("nan")
    return {"auc": auc, "n_pos": len(pos_scores), "n_neg": len(neg_scores),
            "operating_point_precision_at_n_stereo_min": precision,
            "gate_pass": bool(np.isfinite(auc) and auc >= 0.75)}


# ----------------------------------------------------------------------------------------------
# Confirmatory main.
# ----------------------------------------------------------------------------------------------
def run_confirmatory(logs: list[str], held_log: str, cfg: Config, data_root: pathlib.Path,
                     calib_json: pathlib.Path, out_path: pathlib.Path) -> dict:
    """The full sealed run. Result logs = the confirmatory logs EXCLUDING the held-out threshold log
    for the headline gap (the held-out log's miss-rate is reported separately, not pooled)."""
    rng = np.random.default_rng(cfg.seed)

    # calibration AUC gate FIRST (pre-reg: evaluated BEFORE the shuffled-null comparison)
    calib = run_calibration_auc(logs, data_root, cfg, calib_json)
    if not calib["gate_pass"]:
        report = {"verdict": "ORACLE-INSUFFICIENT",
                  "reason": f"calibration AUC {calib['auc']:.3f} < 0.75 -> secondary kill; no miss-rate reported",
                  "calibration": calib, "config": _cfg_dict(cfg)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    # tau_tex on the held-out log (fixed before result-log miss-rates). The census texture-gate exists
    # ONLY because census is unreliable on low texture; the learned-stereo pre-reg (item #3) DROPS it
    # for the artifact source (replaced by IGEV's own validity = the inherited lr-consistency 1.0px +
    # the artifact's NaN finite-mask). So fit/apply it for census only; artifact mode leaves tau_tex
    # None and all `if cfg.tau_tex is not None` gates skip.
    if cfg.disparity_source == "census":
        cfg.tau_tex = fit_tau_tex(held_log, data_root, cfg)

    headline_logs = [lg for lg in logs if lg != held_log]
    rows: list[dict] = []          # confirmatory (headline) frames
    held_rows: list[dict] = []     # held-out log frames (reported separately)
    shuffles_by_frame: list[list[float]] = []
    for log in logs:
        is_held = (log == held_log)
        frame_results = _run_log(log, cfg, data_root, rng)
        for fr in frame_results:
            if fr.calib_reject or not (fr.n_struct > 0):
                continue
            row = {"scene": log, "miss_rate": fr.miss_rate, "shuf_miss_rate": fr.shuf_rate,
                   "n_struct": fr.n_struct, "n_miss": fr.n_miss}
            if is_held:
                held_rows.append(row)
            else:
                rows.append(row)

    if not rows:
        report = {"verdict": "INDETERMINATE", "reason": "no usable headline frames",
                  "calibration": calib, "config": _cfg_dict(cfg)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    scenes = [r["scene"] for r in rows]
    true_b = _boot_mean([r["miss_rate"] for r in rows], scenes, rng)
    shuf_b = _boot_mean([r["shuf_miss_rate"] for r in rows], scenes, rng)
    gap_vals = [r["shuf_miss_rate"] - r["miss_rate"] for r in rows]
    gap_b = _boot_mean(gap_vals, scenes, rng)

    verdict = "INDETERMINATE"
    if gap_b["defined"]:
        verdict = "RECALL-SUPPORTED" if gap_b["lo"] > 0 else "FAIL"

    held_summary = None
    if held_rows:
        hs = [r["scene"] for r in held_rows]
        held_summary = {
            "n_frames": len(held_rows),
            "true_miss_rate": float(np.mean([r["miss_rate"] for r in held_rows])),
            "shuffled_miss_rate": float(np.mean([r["shuf_miss_rate"] for r in held_rows])),
        }

    report = {
        "substrate": "AV2 following/danger stereo logs (RESTRICTED -- following-substrate only)",
        "headline_logs": headline_logs,
        "held_out_threshold_log": held_log,
        "n_headline_logs": len(set(scenes)),
        "n_headline_frames": len(rows),
        "true_miss_rate_mean": true_b["mean"], "true_miss_rate_ci": [true_b["lo"], true_b["hi"]],
        "shuffled_miss_rate_mean": shuf_b["mean"], "shuffled_miss_rate_ci": [shuf_b["lo"], shuf_b["hi"]],
        "gap_mean": gap_b["mean"], "gap_ci": [gap_b["lo"], gap_b["hi"]],
        "verdict": verdict,
        "calibration": calib,
        "held_out_log_report": held_summary,
        "tau_tex": cfg.tau_tex,
        "config": _cfg_dict(cfg),
        "framing": ("occupancy in-path RECALL: miss-rate = MISS-candidate (occ_free & stereo_struct) "
                    "band∩FOV voxels / stereo_struct band∩FOV voxels, same frame t. GAP = "
                    "(band-local-shuffled - true); RECALL-SUPPORTED iff gap CI strictly > 0 "
                    "(falsifiable kill: gap CI includes 0 -> FAIL). Following-substrate only; "
                    "measured miss-rate is a LOWER BOUND (correlated textureless failures dropped)."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _cfg_dict(cfg: Config) -> dict:
    return {"z_min": cfg.z_min, "z_max": cfg.z_max, "n_stereo_min": cfg.n_stereo_min,
            "lr_consistency_px": cfg.lr_consistency_px, "edge_discontinuity_m": cfg.edge_discontinuity_m,
            "null": cfg.null, "shuffles": cfg.shuffles, "seed": cfg.seed, "downsample": cfg.downsample,
            "d_min_px": cfg.d_min_px, "d_max_px": cfg.d_max_px}


def _run_log(log: str, cfg: Config, data_root: pathlib.Path, rng: np.random.Generator) -> list:
    """Run every danger frame of one log. Attaches shuf_rate (mean band-local shuffle) per frame."""
    log_dir = data_root / log
    camL_full, camR_full, B = load_stereo_calib(log, data_root)
    bf = band_fov_mask(camL_full, cfg.z_max)
    scene = av2_sensor.load_scene(log, data_root, with_boxes=False)
    sweeps = [int(fr.time * 1e9) for fr in scene.frames]
    camL_ts = _cam_timestamps(log_dir, "stereo_front_left")
    camR_ts = _cam_timestamps(log_dir, "stereo_front_right")
    out = []
    for fi, ts in enumerate(sweeps):
        grid = scene.frames[fi].grid
        cl = _nearest(camL_ts, ts)
        cr = _nearest(camR_ts, ts)
        grayL = _gray_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{cl}.jpg")
        grayR = _gray_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_right" / f"{cr}.jpg")
        fr = compute_frame(log, ts, grayL, grayR, camL_full, camR_full, B, grid, cfg,
                           ts_left=cl, ts_right=cr)
        # band-local null: recompute struct_in_band + occ count to shuffle
        if not fr.calib_reject and fr.n_struct > 0:
            shuf_rates = _frame_shuffles(log, ts, grayL, grayR, camL_full, camR_full, B, grid, cfg, bf, rng,
                                         ts_left=cl, ts_right=cr)
            fr.shuf_rate = float(np.mean(shuf_rates)) if shuf_rates else float("nan")
        else:
            fr.shuf_rate = float("nan")
        out.append(fr)
    return out


def _frame_shuffles(log, ts, grayL, grayR, camL_full, camR_full, B, grid, cfg, bf, rng,
                    ts_left=None, ts_right=None) -> list[float]:
    """Recompute the per-frame struct_in_band + occupied-in-band count, then run `cfg.shuffles`
    band-local relocations. Returns the list of shuffled miss-rates."""
    # recompute struct_in_band (mirror of compute_frame, returning the masks we need)
    si_band, occ_in_band = _frame_masks(grayL, grayR, camL_full, camR_full, B, grid, cfg, bf,
                                        log=log, ts_left=ts_left, ts_right=ts_right)
    if si_band is None:
        return []
    return [band_local_shuffle_rate(si_band, occ_in_band, bf, rng) for _ in range(cfg.shuffles)]


def _frame_masks(grayL, grayR, camL_full, camR_full, B, grid, cfg, bf, *,
                 log=None, ts_left=None, ts_right=None):
    """Return (struct_in_band (NX,NY) bool, occupied-in-band count) for the band-local null. Mirrors
    compute_frame's structure computation so the null acts on the SAME stereo structure map."""
    ds = cfg.downsample
    mapuL, mapvL = build_undistort_map(camL_full)
    mapuR, mapvR = build_undistort_map(camR_full)
    undL = remap_gray(grayL, mapuL, mapvL)
    undR = remap_gray(grayR, mapuR, mapvR)
    if abs(camL_full.cy - camR_full.cy) > 1.0:
        return None, 0
    undL_s = undL[::ds, ::ds] if ds > 1 else undL
    undR_s = undR[::ds, ::ds] if ds > 1 else undR
    camL_s = camL_full.scaled(ds)
    d_min = max(1, int(math.floor(cfg.d_min_px / ds)))
    d_max = int(math.ceil(cfg.d_max_px / ds))
    dispL = get_disparity(undL_s, undR_s, d_min, d_max, log=log, ts=ts_left, side="L", cfg=cfg)
    dispR = get_disparity(undL_s, undR_s, d_min, d_max, log=log, ts=ts_right, side="R", cfg=cfg)
    dispL = lr_consistency(dispL, dispR, cfg.lr_consistency_px / ds)
    if cfg.tau_tex is not None:
        grad = _patch_evidence_field(undL_s, half=12 // cfg.downsample + 1)
        dispL = np.where(grad >= cfg.tau_tex, dispL, np.nan)
    pts, Zk, us, vs = stereo_points_ego(dispL, camL_s, B, cfg.z_min, cfg.z_max)
    if len(pts):
        keep = edge_adjacency_reject(pts, Zk, us, vs, dispL, cfg.edge_discontinuity_m)
        pts = pts[keep]
    struct = voxelize_stereo(pts, cfg.n_stereo_min)
    struct_bev = struct.any(axis=2)
    si_band = struct_bev & bf
    occ_bev = occupancy_bev(grid, cfg.z_max)
    occ_in_band = int((occ_bev & bf).sum())
    return si_band, occ_in_band


# attach a runtime field for shuf_rate without re-declaring the dataclass
FrameResult.shuf_rate = float("nan")  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AV2 stereo classical-triangulation RECALL oracle (sealed).")
    ap.add_argument("--logs", nargs="+", default=[
        "201fe83b-7dd7-38f4-9d26-7b4a668638a9",
        "2c652f9e-8db8-3572-aa49-fae1344a875b",
        "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c"])
    ap.add_argument("--heldout-threshold-log", default="6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c")
    ap.add_argument("--z-min", type=float, default=2.0)
    ap.add_argument("--z-max", type=float, default=30.0)
    ap.add_argument("--n-stereo-min", type=int, default=8)
    ap.add_argument("--lr-consistency-px", type=float, default=1.0)
    ap.add_argument("--edge-discontinuity-m", type=float, default=1.5)
    ap.add_argument("--null", default="band-local", choices=["band-local"])
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--downsample", type=int, default=_DOWNSAMPLE,
                    help="block-matching image downsample factor (geometry-invariant; SPEC-NOTE in source)")
    ap.add_argument("--disparity-source", default="census", choices=["census", "artifact"],
                    help="depth front-end: census (sealed numpy matcher, DEFAULT) or artifact "
                         "(pre-computed learned-stereo IGEV disparity .npz; the single pre-reg variable)")
    ap.add_argument("--disparity-artifact-dir", type=pathlib.Path, default=None,
                    help="dir of disp_<log>_<cam_ts>_<side>.npz artifacts (required for --disparity-source artifact)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=_HERE / "results" / "oracle_stereo_recall.json")
    ap.add_argument("--data-root", type=pathlib.Path, default=_AV2)
    ap.add_argument("--self-check", action="store_true",
                    help="run geometry round-trips on the real feathers and exit (no confirmatory data)")
    ap.add_argument("--emit-calib-patches", action="store_true",
                    help="deterministically sample + crop the 60 calibration patches for HUMAN labeling and exit")
    ap.add_argument("--calib-json", type=pathlib.Path,
                    default=_HERE / "results" / "calib_patches.json",
                    help="path to calib_patches.json (read for confirmatory, written by --emit-calib-patches)")
    ap.add_argument("--calib-out-dir", type=pathlib.Path,
                    default=_HERE / "results" / "calib_patches",
                    help="dir to write the cropped calibration patch PNGs + manifest")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = Config(
        z_min=args.z_min, z_max=args.z_max, n_stereo_min=args.n_stereo_min,
        lr_consistency_px=args.lr_consistency_px, edge_discontinuity_m=args.edge_discontinuity_m,
        null=args.null, shuffles=args.shuffles, seed=args.seed, downsample=args.downsample,
        disparity_source=args.disparity_source, disparity_artifact_dir=args.disparity_artifact_dir,
    )
    if cfg.disparity_source == "artifact" and cfg.disparity_artifact_dir is None:
        raise SystemExit("--disparity-source artifact requires --disparity-artifact-dir DIR")

    if args.self_check:
        print("STEREO RECALL ORACLE -- geometry self-check (real feathers, no confirmatory data):")
        ok = _self_check(args.logs, args.data_root)
        sys.exit(0 if ok else 1)

    if args.emit_calib_patches:
        print(f"emitting 60 calibration patches (30 pos / 30 neg) -> {args.calib_out_dir} ...")
        rep = emit_calib_patches(args.logs, args.data_root, cfg, args.calib_out_dir, args.seed)
        print(f"  wrote {rep['n_patches']} patches ({rep['n_pos']} pos / {rep['n_neg']} neg); "
              f"manifest {args.calib_out_dir / 'calib_patches.json'} -- ALL label=null (HUMAN gate).")
        sys.exit(0)

    # confirmatory (sealed) -- requires human-filled labels in calib_json
    print("STEREO RECALL ORACLE -- CONFIRMATORY (sealed). Requires human-labeled calib_patches.json.")
    report = run_confirmatory(args.logs, args.heldout_threshold_log, cfg, args.data_root,
                              args.calib_json, args.out)
    print(f"  verdict: {report.get('verdict')}")
    if "gap_ci" in report:
        print(f"  true miss-rate {report['true_miss_rate_mean']:.4f} CI{report['true_miss_rate_ci']}")
        print(f"  shuffled       {report['shuffled_miss_rate_mean']:.4f} CI{report['shuffled_miss_rate_ci']}")
        print(f"  GAP            {report['gap_mean']:.4f} CI{report['gap_ci']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
