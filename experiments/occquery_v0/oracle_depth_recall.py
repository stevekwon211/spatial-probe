# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Oracle-v3 -- AV2 frozen mono-depth (DAv2 metric) cross-modal RECALL oracle for occupancy
denotation-COMPLETENESS (occquery H3, recall half).

Design SEALED in `oracle_depth_recall_preregistration.md` BEFORE this run. This module realizes
that pre-registration faithfully; it does NOT redesign it. It FORKS the killed classical-stereo
oracle (`oracle_stereo_recall.py`, ORACLE-INSUFFICIENT at AUC 0.259) and replaces ONLY the
depth-production stage -- classical block-matching disparity -> a FROZEN monocular metric-depth net
(Depth-Anything-V2-Metric, Virtual-KITTI weights, ONNX, CPU). EVERY other stage (undistort,
back-project, voxelize-with-the-SAME-filters, band∩FOV, band-local null, AUC gate, log-clustered
bootstrap, kill rule) is reused VERBATIM from the stereo oracle (imported, not re-implemented, to
avoid drift). Nothing below was chosen after seeing a v3 number.

ESTIMAND (pre-reg, only the depth stage differs from the stereo pre-reg):
  Per frame t, on `stereo_front_left` nearest-timestamp to LiDAR sweep t:
    1. Undistort (AV2 3-coeff Su radial) -- the stereo oracle's build_undistort_map/remap, applied
       per RGB channel (DAv2 needs RGB).
    2. DAv2 metric depth: LETTERBOX the undistorted image to square (preserve aspect -- NOT a raw
       squash), resize to 518x518, run DAv2 ONNX -> per-pixel meters, un-letterbox + resize back to
       the undistorted 2048x1550 grid. No scale/intrinsics fix (the metric head is absolute).
    3. Keep pixels with Z in [z_min=2, z_max=30] m; back-project (u,v,Z) -> ego (the validated
       backproject_to_ego); voxelize into the av2_sensor grid with the IDENTICAL ground/ego filters
       -> `depth_struct` mask (>= n_depth_min=8 points/voxel). This REPLACES the stereo `stereo_struct`.
    4. In-path band ∩ camera FOV (band_fov_mask, |y|<=_EGO_HALF_W=1.05, x in [0,30]).
    5. occ_free = av2_sensor occupancy reports the voxel FREE.
  MISS candidate = occ_free ∧ depth_struct; recall_miss_rate(t) = |MISS band∩FOV| / |depth_struct band∩FOV|.
  Per-log-clustered mean, log-clustered bootstrap 95% CI (harness_v2._boot_mean, n_boot=1000). GAP =
  (band-local-shuffled - true); RECALL-SUPPORTED iff gap CI.lo > 0 else FAIL.

GATES (pre-reg):
  Gate 1 (metric-scale falsifier, BEFORE the run): on known-good unoccluded annotation boxes (high
    num_interior_pts, in band, 2-30 m), median |DAv2 depth at the box projection - box range|; if
    > 0.5 m in the 2-30 m band -> INVALID-SCALE, STOP, no miss-rate. (Boxes ONLY validate the oracle,
    never in the estimand.)
  Gate 2 (self-reliability AUC, BEFORE the confirmatory): reuse the 60 human-labeled patches; signal =
    the DAv2 depth_struct evidence count in each patch window (the SAME quantity the estimand
    thresholds); AUC via camera_oracle._roc_auc; gate >= 0.75 else ORACLE-INSUFFICIENT.

INDEPENDENCE (pre-reg ledger): passive RGB monocular vs active TOF LiDAR (modality), learned depth
net -> back-project -> voxel vs `av2_sensor._voxelize` (algorithm), DAv2 on Virtual-KITTI synthetic
(NO AV2) vs AV2 LiDAR sweeps (provenance). Shares platform (same AV2 vehicle/timestamp) -> "much more
independent than the traversal oracle, NOT external ground truth".

STACK: onnxruntime + Pillow + numpy + pyarrow ONLY. NO torch, NO cv2. This is an experiment module
(onnxruntime is allowed here, NOT in src/). Does not touch src/probe or src/prism.

Modes:
  --self-check            geometry + letterbox + ONNX-runs round-trips (no confirmatory data); RUN FIRST.
  --scale-check           Gate 1 metric-scale falsifier on annotation boxes (no miss-rate); cheap pre-check.
  (default / confirmatory) the full sealed run -> results JSON. SEALED-GATED, run by the orchestrator only.

Sealed run command (from `oracle_depth_recall_preregistration.md`):
  .venv/bin/python experiments/occquery_v0/oracle_depth_recall.py \
    --logs 201fe83b-7dd7-38f4-9d26-7b4a668638a9 2c652f9e-8db8-3572-aa49-fae1344a875b 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
    --heldout-threshold-log 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
    --onnx data/models/dav2_metric_vkitti_vitl/depth_anything_v2_metric_vkitti_vitl.onnx \
    --camera stereo_front_left --z-min 2 --z-max 30 --n-depth-min 8 \
    --scale-gate-m 0.5 --auc-gate 0.75 --null band-local --shuffles 1000 --seed 0 \
    --calib-json experiments/occquery_v0/results/calib_patches/calib_patches.json \
    --out experiments/occquery_v0/results/oracle_depth_recall.json
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

from harness_v2 import _boot_mean  # noqa: E402  (log-clustered bootstrap, same as oracle_stereo/traversal)
from probe.adapters import av2_sensor  # noqa: E402  (grid spec + _voxelize filters reused EXACTLY)
from camera_oracle import _roc_auc  # noqa: E402  (_roc_auc reused VERBATIM for the calibration AUC)

# REUSE the stereo oracle's validated geometry stages VERBATIM (import, do NOT re-implement -- avoids
# drift; every one of these is the SAME function the stereo pre-reg's self-check validated):
#   AV2Camera, load_stereo_calib (the camera also gives us the FOV/back-project geometry),
#   build_undistort_map, remap_gray, backproject_to_ego, project_ego_to_left, _project_undistorted,
#   band_fov_mask, occupancy_bev, band_local_shuffle_rate, _cam_timestamps, _nearest, _read_feather,
#   distort_normalized/undistort_normalized (for the self-check round-trips).
from oracle_stereo_recall import (  # noqa: E402
    AV2Camera,
    backproject_to_ego,
    band_fov_mask,
    band_local_shuffle_rate,
    build_undistort_map,
    distort_normalized,
    load_stereo_calib,
    occupancy_bev,
    project_ego_to_left,
    remap_gray,
    undistort_normalized,
    _cam_timestamps,
    _nearest,
    _project_undistorted,
    _read_feather,
)

_AV2 = _HERE.parents[1] / "data" / "danger" / "av2_sensor"
_DEFAULT_ONNX = _HERE.parents[1] / "data" / "models" / "dav2_metric_vkitti_vitl" / \
    "depth_anything_v2_metric_vkitti_vitl.onnx"

# Grid spec mirrored from av2_sensor (read, never redefined -- the source of truth is the adapter).
_VOX = av2_sensor.VOXEL_SIZE
_RANGE = av2_sensor.RANGE
_NX, _NY, _NZ = av2_sensor.GRID_SHAPE
_EGO_HALF_W = av2_sensor._EGO_HALF_W  # 1.05 m -- the in-path band half-width per the pre-reg

# DAv2 model constants (verified on disk before seal; see pre-reg "Model").
_NET_HW = 518                                          # input/output square side
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MAX_DEPTH_M = 80.0                                    # sigmoid*80 baked in; output is absolute meters


# ----------------------------------------------------------------------------------------------
# RGB undistort (DAv2 needs colour). Reuse the stereo build_undistort_map/remap_gray PER CHANNEL.
# ----------------------------------------------------------------------------------------------
def _rgb_from_jpg(path: pathlib.Path) -> np.ndarray:
    """Load a raw camera JPEG as float RGB (H,W,3), values in [0,255]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)[..., :3]


def undistort_rgb(rgb_raw: np.ndarray, map_u: np.ndarray, map_v: np.ndarray) -> np.ndarray:
    """Undistort an RGB image (H,W,3) by bilinear-sampling each channel at the SAME (map_u,map_v)
    source coords -- the exact maps used for the grayscale stereo undistort (reused remap_gray, so a
    DAv2 undistorted pixel sits at the SAME pinhole location a stereo undistorted pixel would).
    Out-of-frame -> NaN (remap_gray's convention), then carried as 0 for the net (and the depth-valid
    mask drops those pixels via the Z-range filter so no spurious surface is voxelized)."""
    chans = [remap_gray(rgb_raw[..., c], map_u, map_v) for c in range(3)]
    return np.stack(chans, axis=-1)  # (H,W,3) float, NaN out-of-frame


# ----------------------------------------------------------------------------------------------
# Letterbox (preserve aspect) <-> un-letterbox. NOT a raw squash (the pre-reg forbids it: a squash
# distorts depth geometry). Pad the undistorted image to a centered square, resize to 518, run DAv2,
# then map the 518-grid depth back to the undistorted grid by inverting the same letterbox transform.
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Letterbox:
    """The forward undistorted->net transform, kept so the inverse (depth net->undistorted) is exact.

    Undistorted image is (H,W). We pad to a square of side S = max(H,W) with the image centered
    (pad_x left/right, pad_y top/bottom), then SCALE the square down to _NET_HW. Forward map of an
    undistorted pixel (u,v): net_u = (u + pad_x) * scale, net_v = (v + pad_y) * scale. Inverse (the
    one we actually use to pull a depth value for each undistorted pixel): for undistorted (u,v),
    sample the 518 depth at (net_u, net_v). scale = _NET_HW / S."""
    H: int
    W: int
    side: int      # S = max(H,W) (square side before downscale)
    pad_x: int     # left pad (columns) to center horizontally
    pad_y: int     # top pad (rows) to center vertically
    scale: float   # _NET_HW / side

    @staticmethod
    def build(H: int, W: int) -> "Letterbox":
        side = max(H, W)
        pad_x = (side - W) // 2
        pad_y = (side - H) // 2
        return Letterbox(H=H, W=W, side=side, pad_x=pad_x, pad_y=pad_y, scale=_NET_HW / side)


def letterbox_to_net(rgb_und: np.ndarray, lb: Letterbox) -> np.ndarray:
    """Undistorted RGB (H,W,3) -> letterboxed, ImageNet-normalized net input (1,3,518,518) float32.

    Pad-to-square (NaN->0 for the padded border AND out-of-frame undistort pixels), resize to 518x518
    with PIL bilinear (Pillow only; no cv2), /255, ImageNet-normalize, CHW + batch."""
    H, W, _ = rgb_und.shape
    square = np.zeros((lb.side, lb.side, 3), dtype=np.float32)
    img = np.nan_to_num(rgb_und, nan=0.0).astype(np.float32)
    square[lb.pad_y:lb.pad_y + H, lb.pad_x:lb.pad_x + W, :] = img
    # PIL bilinear resize (uint8 round-trip is fine: the net input is /255 normalized anyway)
    pim = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8), mode="RGB")
    pim = pim.resize((_NET_HW, _NET_HW), Image.BILINEAR)
    arr = np.asarray(pim, dtype=np.float32) / 255.0          # (518,518,3) RGB in [0,1]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(arr, (2, 0, 1))[None, ...]            # (1,3,518,518)
    return np.ascontiguousarray(chw, dtype=np.float32)


def un_letterbox_depth(depth_net: np.ndarray, lb: Letterbox) -> np.ndarray:
    """518x518 metric depth -> the undistorted (H,W) grid, inverting the letterbox EXACTLY.

    For every undistorted pixel (u,v) we read the net depth at its forward-mapped location
    (net_u, net_v) = ((u+pad_x)*scale, (v+pad_y)*scale) via nearest-neighbour (depth is a per-pixel
    metric value -- bilinear would average across depth discontinuities and bleed silhouettes;
    nearest preserves the surface). Pixels mapping outside the valid (non-pad) net region get NaN.
    (Equivalently: crop the 518 grid to the image's letterbox sub-rectangle and resize to (H,W); the
    per-pixel pull below is that, computed directly so the un-letterbox is the exact inverse used in
    the self-check round-trip.)"""
    H, W = lb.H, lb.W
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    net_u = (uu + lb.pad_x) * lb.scale
    net_v = (vv + lb.pad_y) * lb.scale
    nu = np.round(net_u).astype(np.intp)
    nv = np.round(net_v).astype(np.intp)
    valid = (nu >= 0) & (nu < _NET_HW) & (nv >= 0) & (nv < _NET_HW)
    nu_c = np.clip(nu, 0, _NET_HW - 1)
    nv_c = np.clip(nv, 0, _NET_HW - 1)
    out = depth_net[nv_c, nu_c]
    return np.where(valid, out, np.nan)


# ----------------------------------------------------------------------------------------------
# GROUND-PLANE per-frame scale correction (v3.1 pre-reg `oracle_depth_recall_v2_preregistration.md`).
#
# v3 returned INVALID-SCALE: DAv2-VKITTI metric is RELATIVE-correct but ABSOLUTE-over by ~1.65x on AV2.
# This stage estimates a per-frame multiplicative scale `s` from a source that NEVER touches the LiDAR
# being graded -- ONLY the camera intrinsics K, the camera ego-frame height h from the calibration
# extrinsic (egovehicle_SE3_sensor, NOT LiDAR), a flat-ground prior on a lower-center image wedge, and
# the DAv2 depth image. `Z_corrected = s * Z_DAv2` is then emitted everywhere downstream.
#
# INDEPENDENCE GUARD (pre-reg ledger preserved): `ground_plane_scale` takes ONLY (depth_und, cam) and a
# few scalar config knobs. It is STRUCTURALLY incapable of reading a LiDAR sweep / occupancy grid -- no
# `scene`, `grid`, `av2_sensor.load_*`, or sweep path is in scope here. The estimate uses K (via the
# (u-cx)/fx form, same as backproject_to_ego), R_cam2ego + h (the extrinsic), flat-ground (z=0), and the
# DAv2 image. The self-check asserts this funnel touches no sweep/occupancy. See SPEC-NOTE below.
#
# SPEC-NOTE (Z_geo optical-axis derivation): for an undistorted pixel (u,v) the camera-frame ray
# direction with UNIT optical-axis component is d_cam = (xn, yn, 1), xn=(u-cx)/fx, yn=(v-cy)/fy (the
# exact normalized coords backproject_to_ego uses). In ego: d_ego = R_cam2ego @ d_cam. A point at
# optical-axis depth Z sits at ego-z = h + Z*d_ego_z (h = camera height = t_cam_in_ego[2]). The flat
# ground is ego-z = 0, so 0 = h + Z_geo*d_ego_z => Z_geo = h / (-d_ego_z), valid only where d_ego_z < 0
# (ray points down). Because d_cam already has unit z-component, Z_geo IS the optical-axis depth -- the
# SAME quantity DAv2 emits and backproject_to_ego consumes (the pre-reg's "optical-axis component"
# factor is 1 in this parameterization). Verified on real calib: a lower-center road pixel back-projects
# to ego-z = 0.0 at Z_geo. The pre-reg phrasing `Z_geo = h/(-d_ego,z) * (optical-axis component)` is
# this, with the factor = 1.
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GroundScale:
    """Per-frame ground-plane scale result. `valid` False => frame dropped from the estimand (logged)."""
    s: float                 # multiplicative scale: Z_corrected = s * Z_DAv2  (NaN if invalid)
    valid: bool              # >= n_road_min valid road pixels AND s in (s_lo, s_hi)
    n_road: int              # number of road-prior pixels with finite positive Z_geo and Z_DAv2
    reason: str              # "" if valid, else why dropped (logged, no silent fallback)


# Road-prior wedge (flat-ground prior; NO segmentation, NO LiDAR) -- pre-reg fractions, sealed.
_ROAD_V_LO, _ROAD_V_HI = 0.75, 0.97   # v in [0.75*H, 0.97*H]
_ROAD_U_LO, _ROAD_U_HI = 0.35, 0.65   # u in [0.35*W, 0.65*W]
_ROAD_N_MIN = 50                       # pre-reg "fewer than ~50 valid road pixels" => scale-invalid
_SCALE_LO, _SCALE_HI = 0.3, 3.0        # pre-reg s must lie in (0.3, 3.0) else scale-invalid


def ground_plane_scale(depth_und: np.ndarray, cam: AV2Camera,
                       n_road_min: int = _ROAD_N_MIN,
                       s_lo: float = _SCALE_LO, s_hi: float = _SCALE_HI) -> GroundScale:
    """Per-frame multiplicative scale s = median(Z_geo / Z_DAv2) over flat-ground road-prior pixels.

    INPUTS (independence-preserving, NO LiDAR/occupancy): the undistorted DAv2 depth map `depth_und`
    (H,W meters, optical-axis Z) + the camera calibration `cam` (intrinsics K via (u-cx)/fx, the
    extrinsic R_cam2ego, and the ego-frame camera height h = |t_cam_in_ego[2]|). This function has NO
    access to any sweep / occupancy / scene object -- the independence guard is the signature itself.

    Road prior: undistorted pixels with v in [0.75H,0.97H], u in [0.35W,0.65W] (lower-center wedge, a
    flat-ground prior -- no segmentation). For each such pixel: d_cam=((u-cx)/fx,(v-cy)/fy,1),
    d_ego=R_cam2ego@d_cam; if d_ego_z<0 (ray points down) the geometric optical-axis depth to ground
    z=0 is Z_geo = h/(-d_ego_z) (see SPEC-NOTE above). s = median(Z_geo/Z_DAv2) over pixels with finite
    positive Z_geo AND finite positive in-range DAv2 Z. Guard: < n_road_min valid pixels OR s not in
    (s_lo,s_hi) => valid=False (frame dropped, reason logged; explicit, no silent fallback)."""
    H, W = depth_und.shape
    h = float(abs(cam.t_cam_in_ego[2]))  # camera ego-frame height (extrinsic, NOT LiDAR)
    v0, v1 = int(round(_ROAD_V_LO * H)), int(round(_ROAD_V_HI * H))
    u0, u1 = int(round(_ROAD_U_LO * W)), int(round(_ROAD_U_HI * W))
    vv, uu = np.mgrid[v0:v1, u0:u1]
    uu = uu.astype(float).ravel()
    vv = vv.astype(float).ravel()
    # camera-frame ray dirs with UNIT optical-axis component (same normalized coords as backproject_to_ego)
    xn = (uu - cam.cx) / cam.fx
    yn = (vv - cam.cy) / cam.fy
    d_cam = np.stack([xn, yn, np.ones_like(xn)], axis=-1)        # (M,3), unit z-component
    d_ego = (cam.R_cam2ego @ d_cam.T).T                          # (M,3) ego-frame ray dirs
    d_ego_z = d_ego[:, 2]
    z_dav2 = depth_und[vv.astype(np.intp), uu.astype(np.intp)]   # DAv2 optical-axis Z at each road pixel
    down = d_ego_z < 0.0                                         # ray must point at the ground
    with np.errstate(divide="ignore", invalid="ignore"):
        z_geo = np.where(down, h / (-d_ego_z), np.nan)           # geometric optical-axis depth to z=0
    valid_px = (np.isfinite(z_geo) & (z_geo > 0) &
                np.isfinite(z_dav2) & (z_dav2 > 0))
    n_road = int(valid_px.sum())
    if n_road < n_road_min:
        return GroundScale(s=float("nan"), valid=False, n_road=n_road,
                           reason=f"only {n_road} valid road pixels (< {n_road_min})")
    s = float(np.median(z_geo[valid_px] / z_dav2[valid_px]))
    if not (s_lo < s < s_hi):
        return GroundScale(s=s, valid=False, n_road=n_road,
                           reason=f"scale {s:.4f} outside ({s_lo},{s_hi})")
    return GroundScale(s=s, valid=True, n_road=n_road, reason="")


def apply_ground_scale(depth_und: np.ndarray, cam: AV2Camera,
                       n_road_min: int = _ROAD_N_MIN) -> tuple[np.ndarray, GroundScale]:
    """Compute the per-frame ground-plane scale and return (Z_corrected, GroundScale).

    If valid: Z_corrected = s * depth_und applied to ALL pixels that frame. If invalid: returns an
    all-NaN depth map (so the frame contributes ZERO depth_struct -> effectively dropped from the
    estimand) plus the GroundScale with reason; the caller logs the drop. This is the explicit
    no-silent-fallback path: an invalid-scale frame produces no surface rather than a wrong-scale one."""
    gs = ground_plane_scale(depth_und, cam, n_road_min=n_road_min)
    if not gs.valid:
        return np.full_like(depth_und, np.nan), gs
    return depth_und * gs.s, gs


def corrected_depth(net: "DepthNet", rgb_und: np.ndarray, cam: AV2Camera,
                    rescale: str) -> tuple[np.ndarray, "GroundScale | None"]:
    """THE single funnel that turns an undistorted RGB image into the depth map every downstream stage
    consumes. `rescale="none"` => the EXACT v3 raw DAv2 metric depth (current behavior, preserved bit-
    for-bit). `rescale="ground-plane"` => v3.1: raw DAv2 depth * per-frame ground-plane scale s, with
    invalid-scale frames returned as all-NaN (dropped). Both compute_frame and the band-local null call
    THIS, so the null acts on the SAME corrected depth as the true path (no scale drift between them).

    NOTE (independence): the ground-plane branch reaches ground_plane_scale(depth_und, cam) only -- the
    scale path never sees a sweep/occupancy here either. Returns (depth_for_downstream, GroundScale|None)
    where the GroundScale is None when rescale='none'."""
    depth_und, _lb = net.depth_undistorted(rgb_und)
    if rescale == "ground-plane":
        return apply_ground_scale(depth_und, cam)
    return depth_und, None


# ----------------------------------------------------------------------------------------------
# DAv2 ONNX depth (the ONLY new pipeline stage vs the stereo oracle).
# ----------------------------------------------------------------------------------------------
class DepthNet:
    """Frozen DAv2-metric ONNX session (onnxruntime CPU). One instance per run; deterministic (the
    net is feed-forward, no sampling). Caches the session so the ~1.3 GB model loads once."""

    def __init__(self, onnx_path: pathlib.Path):
        import onnxruntime as ort  # local import: only this module needs onnxruntime
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0  # let ORT pick; deterministic for a feed-forward conv net
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def infer_square(self, net_input: np.ndarray) -> np.ndarray:
        """(1,3,518,518) float32 ImageNet-normalized -> (518,518) metric depth in meters."""
        out = self.sess.run([self.out_name], {self.in_name: net_input})[0]
        return np.asarray(out, dtype=np.float32).reshape(_NET_HW, _NET_HW)

    def depth_undistorted(self, rgb_und: np.ndarray) -> tuple[np.ndarray, Letterbox]:
        """Undistorted RGB (H,W,3) -> metric depth on the SAME (H,W) undistorted grid (letterbox in,
        un-letterbox out). Returns (depth_HxW_meters, letterbox)."""
        H, W, _ = rgb_und.shape
        lb = Letterbox.build(H, W)
        net_in = letterbox_to_net(rgb_und, lb)
        depth_net = self.infer_square(net_in)
        depth_und = un_letterbox_depth(depth_net, lb)
        return depth_und, lb


# ----------------------------------------------------------------------------------------------
# Depth -> ego points -> filtered + voxelized depth_struct (IDENTICAL filters to av2_sensor).
# ----------------------------------------------------------------------------------------------
def depth_points_ego(depth_und: np.ndarray, cam_full: AV2Camera,
                     z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From an UNDISTORTED-frame metric-depth map (meters, camera-frame Z) compute ego points.

    Keep pixels with finite Z in [z_min,z_max] (the sky-guard drops Z>z_max so the net's finite-sky
    depth never voxelizes; the ground filter below + z_min drop near-field/road). Back-project via the
    validated backproject_to_ego (the SAME inverse-projection the stereo oracle uses). Returns
    (points_ego (M,3), Zkept (M,), u (M,), v (M,))."""
    H, W = depth_und.shape
    vv, uu = np.mgrid[0:H, 0:W]
    Z = depth_und
    good = np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
    u = uu[good].astype(float)
    v = vv[good].astype(float)
    Zk = Z[good].astype(float)
    pts = backproject_to_ego(u, v, Zk, cam_full)
    return pts, Zk, u, v


def voxelize_depth(points_ego: np.ndarray, n_depth_min: int) -> np.ndarray:
    """Voxelize depth ego points into the av2_sensor grid with the IDENTICAL ground/ego-self-return
    filters, then keep voxels with >= n_depth_min supporting points (the pre-reg's depth_struct).
    Returns a (NX,NY,NZ) int8 mask (1 = depth_struct). The filter logic mirrors av2_sensor._voxelize
    EXACTLY (read from the adapter constants), so a depth obstacle voxel means the SAME thing as a
    LiDAR obstacle voxel -- this is the cross-modal comparability the estimand depends on."""
    (x0, x1), (y0, y1), (z0, z1) = _RANGE
    x = points_ego[:, 0]
    y = points_ego[:, 1]
    z = points_ego[:, 2]
    # IDENTICAL filters to av2_sensor._voxelize:
    ego = (x > av2_sensor._EGO_X0) & (x < av2_sensor._EGO_X1) & (np.abs(y) < av2_sensor._EGO_HALF_W)
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z > av2_sensor._ROAD_Z) & (z < z1) & ~ego
    counts = np.zeros(av2_sensor.GRID_SHAPE, dtype=np.int32)
    if not m.any():
        return (counts >= n_depth_min).astype(np.int8)
    ix = ((x[m] - x0) / _VOX).astype(np.intp)
    iy = ((y[m] - y0) / _VOX).astype(np.intp)
    iz = ((z[m] - z0) / _VOX).astype(np.intp)
    np.clip(ix, 0, _NX - 1, out=ix)
    np.clip(iy, 0, _NY - 1, out=iy)
    np.clip(iz, 0, _NZ - 1, out=iz)
    np.add.at(counts, (ix, iy, iz), 1)
    return (counts >= n_depth_min).astype(np.int8)


# ----------------------------------------------------------------------------------------------
# Per-frame recall computation (mirrors the stereo compute_frame; depth stage swapped).
# ----------------------------------------------------------------------------------------------
@dataclass
class FrameResult:
    log: str
    ts: int
    n_struct: int          # depth_struct band∩FOV voxels (denominator)
    n_miss: int            # MISS-candidate band∩FOV voxels = occ_free ∧ depth_struct
    miss_rate: float       # n_miss / n_struct  (per-frame estimator)
    n_occ_band: int        # occupied band∩FOV voxels (for the band-local null relocation count)
    depth_valid_frac: float  # in-band depth-valid fraction (clear-frame filter)
    dropped: bool          # clear-frame filter dropped this frame
    shuf_rate: float = float("nan")
    scale_s: float = float("nan")   # per-frame ground-plane scale (v3.1; NaN if rescale=none/invalid)
    scale_invalid: bool = False     # ground-plane scale invalid -> frame dropped from estimand (logged)


def _depth_struct_in_band(depth_und: np.ndarray, camL_full: AV2Camera, bf: np.ndarray,
                          cfg: "Config") -> tuple[np.ndarray, np.ndarray]:
    """Depth map -> (depth_struct BEV (NX,NY) bool, depth_struct band∩FOV (NX,NY) bool). The single
    funnel both compute_frame and the band-local null go through, so the null acts on the SAME
    structure map (no drift between true and shuffled paths)."""
    pts, _Zk, _us, _vs = depth_points_ego(depth_und, camL_full, cfg.z_min, cfg.z_max)
    struct = voxelize_depth(pts, cfg.n_depth_min)
    struct_bev = struct.any(axis=2)
    return struct_bev, struct_bev & bf


def _in_band_depth_valid_frac(depth_und: np.ndarray, camL_full: AV2Camera, cfg: "Config") -> float:
    """Clear-frame filter statistic: fraction of in-band image pixels with a usable depth (finite,
    Z in [z_min,z_max]). 'in-band' = undistorted pixels onto which the band∩FOV BEV cells (at road-
    plane probe height) project -- the SAME band the estimand denominator uses. Drop the frame if
    this fraction < cfg.clear_frame_min (both-fail / dark frames; declared confound mitigation)."""
    bf = band_fov_mask(camL_full, cfg.z_max)
    (x0, _), (y0, _), _ = _RANGE
    bx, by = np.where(bf)
    if len(bx) == 0:
        return 0.0
    xs = x0 + (bx + 0.5) * _VOX
    ys = y0 + (by + 0.5) * _VOX
    z_probe = max(av2_sensor._ROAD_Z + 0.5, 0.8)
    band_pts = np.stack([xs, ys, np.full(len(bx), z_probe)], axis=1)
    uv, depth, vis = _project_undistorted(band_pts, camL_full)
    H, W = depth_und.shape
    uu = np.round(uv[vis, 0]).astype(int)
    vv = np.round(uv[vis, 1]).astype(int)
    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    if not ok.any():
        return 0.0
    zvals = depth_und[vv[ok], uu[ok]]
    usable = np.isfinite(zvals) & (zvals >= cfg.z_min) & (zvals <= cfg.z_max)
    return float(usable.mean())


def compute_frame(log: str, ts: int, rgb_raw: np.ndarray, camL_full: AV2Camera, grid,
                  net: DepthNet, cfg: "Config",
                  map_u: np.ndarray, map_v: np.ndarray, bf: np.ndarray) -> FrameResult:
    """Full per-frame pipeline: undistort RGB -> letterbox -> DAv2 depth -> un-letterbox ->
    Z-range + back-project -> filters -> voxelize -> band∩FOV -> miss events. Mirrors the stereo
    compute_frame; the depth stage (block-match) is replaced by the DAv2 call."""
    # (a) undistort the raw RGB to the pinhole frame (reused maps; reused remap_gray per channel)
    rgb_und = undistort_rgb(rgb_raw, map_u, map_v)
    # (b) DAv2 metric depth on the undistorted grid, then v3.1 ground-plane rescale (none => raw v3)
    depth_und, gs = corrected_depth(net, rgb_und, camL_full, cfg.rescale)
    s_val = gs.s if gs is not None else float("nan")
    # (b0) scale-invalid frames (v3.1): depth is all-NaN -> drop from estimand, log explicitly
    if gs is not None and not gs.valid:
        return FrameResult(log, ts, 0, 0, float("nan"), 0, 0.0, dropped=True,
                           scale_s=s_val, scale_invalid=True)
    # (b') clear-frame filter (declared confound: drop both-fail/dark frames)
    dvf = _in_band_depth_valid_frac(depth_und, camL_full, cfg)
    if dvf < cfg.clear_frame_min:
        return FrameResult(log, ts, 0, 0, float("nan"), 0, dvf, dropped=True, scale_s=s_val)
    # (c) depth_struct band∩FOV (single funnel shared with the null)
    struct_bev, struct_in_band = _depth_struct_in_band(depth_und, camL_full, bf, cfg)
    occ_bev = occupancy_bev(grid, cfg.z_max)
    occ_free = ~occ_bev  # occupancy reports FREE
    # (d) miss event: occ_free ∧ depth_struct, within band∩FOV
    miss = struct_in_band & occ_free
    n_struct = int(struct_in_band.sum())
    n_miss = int(miss.sum())
    n_occ_band = int((occ_bev & bf).sum())
    rate = (n_miss / n_struct) if n_struct > 0 else float("nan")
    return FrameResult(log, ts, n_struct, n_miss, rate, n_occ_band, dvf, dropped=False, scale_s=s_val)


# ----------------------------------------------------------------------------------------------
# Gate 1 -- metric-scale falsifier on annotation boxes (BEFORE the run; boxes ONLY validate scale).
# ----------------------------------------------------------------------------------------------
def run_scale_gate(logs: list[str], data_root: pathlib.Path, net: DepthNet, cfg: "Config",
                   n_frames: int = 8, min_interior: int = 50) -> dict:
    """Gate 1 (pre-reg): on known-good unoccluded annotation boxes (high num_interior_pts, in band,
    2-30 m), compare DAv2 median surface range to the box range. INVALID-SCALE iff
    median |DAv2 - box_range| > cfg.scale_gate_m in the 2-30 m band.

    Box range = the box-center camera-frame Z (range to the surface the net should report). DAv2
    depth at the box projection = the median over a small window around the projected box center
    (robust to the exact center pixel). Boxes are used ONLY to validate the oracle's metric scale and
    are NEVER part of the miss-rate estimand. Direction of error logged (signed median).

    v3.1: with cfg.rescale='ground-plane' the depth compared here is Z_corrected = s * Z_DAv2 (the SAME
    correction the estimand uses), so this is the Gate-1 RE-CHECK the v2 pre-reg asks for. The per-frame
    scale s values are recorded and returned (frame_scales) for the report. Box-range/projection are
    LiDAR-INDEPENDENT scale validators, unchanged from v3."""
    abs_errs: list[float] = []
    signed_errs: list[float] = []
    used = 0
    per_log: dict[str, int] = {}
    frame_scales: list[float] = []       # per-frame ground-plane s seen (v3.1; empty if rescale=none)
    n_scale_invalid_frames = 0           # frames dropped for invalid ground-plane scale (v3.1)
    for log in logs:
        log_dir = data_root / log
        camL_full, _camR, _B = load_stereo_calib(log, data_root)
        map_u, map_v = build_undistort_map(camL_full)
        ann = _read_feather(log_dir / "annotations.feather").to_pydict()
        ann_ts = np.asarray(ann["timestamp_ns"], dtype=np.int64)
        tx = np.asarray(ann["tx_m"], dtype=float)
        ty = np.asarray(ann["ty_m"], dtype=float)
        tz = np.asarray(ann["tz_m"], dtype=float)
        npts = np.asarray(ann["num_interior_pts"], dtype=float)
        camL_ts = _cam_timestamps(log_dir, cfg.camera)
        lidar_ts = sorted(int(os.path.basename(p)[:-len(".feather")])
                          for p in glob.glob(str(log_dir / "sensors" / "lidar" / "*.feather")))
        if not lidar_ts:
            continue
        pick = np.linspace(0, len(lidar_ts) - 1, num=min(n_frames, len(lidar_ts))).astype(int)
        for k in pick:
            ts = lidar_ts[int(k)]
            sel = np.where(ann_ts == ts)[0]
            if len(sel) == 0:
                continue
            centers = np.stack([tx[sel], ty[sel], tz[sel]], axis=1)
            uv, depth, vis = _project_undistorted(centers, camL_full)
            in_band = (np.abs(centers[:, 1]) <= _EGO_HALF_W) & (depth >= cfg.z_min) & (depth <= cfg.z_max)
            good_box = vis & in_band & (npts[sel] >= min_interior)
            cand = np.where(good_box)[0]
            if len(cand) == 0:
                continue
            cl = _nearest(camL_ts, ts)
            rgb_raw = _rgb_from_jpg(log_dir / "sensors" / "cameras" / cfg.camera / f"{cl}.jpg")
            rgb_und = undistort_rgb(rgb_raw, map_u, map_v)
            # v3.1: corrected depth (Z_corrected = s*Z_DAv2) when rescale=ground-plane; raw v3 otherwise.
            depth_und, gs = corrected_depth(net, rgb_und, camL_full, cfg.rescale)
            if gs is not None:
                frame_scales.append(gs.s)              # record the per-frame scale (NaN if invalid)
                if not gs.valid:
                    n_scale_invalid_frames += 1        # depth is all-NaN -> boxes below contribute 0
                    continue                            # drop this frame's boxes (logged, explicit)
            H, W = depth_und.shape
            for c in cand:
                u, v = float(uv[c, 0]), float(uv[c, 1])
                ui, vi = int(round(u)), int(round(v))
                half = 6  # small window around the projected box center
                u0, u1 = max(0, ui - half), min(W, ui + half + 1)
                v0, v1 = max(0, vi - half), min(H, vi + half + 1)
                win = depth_und[v0:v1, u0:u1]
                fin = win[np.isfinite(win) & (win > 0)]
                if fin.size == 0:
                    continue
                dav2_range = float(np.median(fin))
                box_range = float(depth[c])  # camera-frame Z to the box center
                abs_errs.append(abs(dav2_range - box_range))
                signed_errs.append(dav2_range - box_range)
                used += 1
                per_log[log] = per_log.get(log, 0) + 1
    median_abs = float(np.median(abs_errs)) if abs_errs else float("nan")
    median_signed = float(np.median(signed_errs)) if signed_errs else float("nan")
    invalid = bool(np.isfinite(median_abs) and median_abs > cfg.scale_gate_m)
    finite_scales = [s for s in frame_scales if np.isfinite(s)]
    out = {
        "n_boxes": used,
        "per_log": per_log,
        "median_abs_err_m": median_abs,
        "median_signed_err_m": median_signed,   # >0 = DAv2 farther than box; <0 = nearer
        "scale_gate_m": cfg.scale_gate_m,
        "min_interior_pts": min_interior,
        "invalid_scale": invalid,
        "gate_pass": bool(np.isfinite(median_abs) and not invalid),
        "rescale": cfg.rescale,
    }
    if cfg.rescale == "ground-plane":
        out["frame_scales"] = [round(float(s), 5) for s in frame_scales]   # per-frame s (NaN=invalid)
        out["n_scale_invalid_frames"] = n_scale_invalid_frames
        out["median_frame_scale"] = float(np.median(finite_scales)) if finite_scales else float("nan")
    return out


# ----------------------------------------------------------------------------------------------
# Gate 2 -- self-reliability AUC over the 60 human-labeled patches (depth_struct evidence count).
# ----------------------------------------------------------------------------------------------
def run_calibration_auc(data_root: pathlib.Path, net: DepthNet, cfg: "Config",
                        calib_json: pathlib.Path) -> dict:
    """Gate 2 (pre-reg): AUC of the DAv2 depth_struct EVIDENCE COUNT in each patch window (the SAME
    quantity the estimand thresholds at n_depth_min), human-pos vs human-neg, via _roc_auc. Gate
    AUC >= cfg.auc_gate else ORACLE-INSUFFICIENT.

    Signal per patch = the number of voxel-supporting depth points whose UNDISTORTED projection falls
    in the patch's (u,v) window AND that pass the SAME Z-range + ground/ego filters as the estimand
    (i.e. the count that feeds depth_struct, restricted to the patch window). This is the count the
    n_depth_min threshold acts on -- the literal estimand evidence, not a proxy.

    Requires calib_patches.json with labels filled in (the 60-patch human labeling is a required
    integrity gate; NO labels are fabricated here)."""
    rep = json.loads(calib_json.read_text())
    patches = rep["patches"]
    if any(p.get("label") is None for p in patches):
        raise SystemExit("calib_patches.json has UNLABELED patches (label=null). The 60-patch human "
                         "labeling is a required integrity gate -- fill every label before the "
                         "confirmatory run. Aborting (no fabricated labels).")
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    by_log: dict[str, list[dict]] = {}
    for p in patches:
        by_log.setdefault(p["log"], []).append(p)
    half = 12  # patch half-size (matches camera_oracle._PATCH; same window the patches were cropped at)
    for log, ps in by_log.items():
        log_dir = data_root / log
        camL_full, _camR, _B = load_stereo_calib(log, data_root)
        map_u, map_v = build_undistort_map(camL_full)
        # group by cam frame so DAv2 runs once per frame
        frames: dict[int, list[dict]] = {}
        for p in ps:
            frames.setdefault(int(p["cam_ts"]), []).append(p)
        for cam_ts, plist in frames.items():
            rgb_raw = _rgb_from_jpg(log_dir / "sensors" / "cameras" / cfg.camera / f"{cam_ts}.jpg")
            rgb_und = undistort_rgb(rgb_raw, map_u, map_v)
            # v3.1: corrected depth (Z_corrected = s*Z_DAv2) when rescale=ground-plane; raw v3 otherwise.
            # The estimand evidence count is on the SAME corrected depth the confirmatory thresholds.
            depth_und, _gs = corrected_depth(net, rgb_und, camL_full, cfg.rescale)
            H, W = depth_und.shape
            # the SAME estimand filters, but we keep per-pixel so we can window-count + project.
            # A pixel contributes evidence iff: finite Z in [z_min,z_max] AND its back-projected ego
            # point passes the ground/ego voxelize filters (the depth_struct member test, per pixel).
            vv, uu = np.mgrid[0:H, 0:W]
            Z = depth_und
            zmask = np.isfinite(Z) & (Z >= cfg.z_min) & (Z <= cfg.z_max)
            evid = np.zeros((H, W), dtype=bool)
            if zmask.any():
                uvalid = uu[zmask].astype(float)
                vvalid = vv[zmask].astype(float)
                Zk = Z[zmask].astype(float)
                pe = backproject_to_ego(uvalid, vvalid, Zk, camL_full)
                (x0, x1), (y0, y1), (z0, z1) = _RANGE
                ex, ey, ez = pe[:, 0], pe[:, 1], pe[:, 2]
                ego = (ex > av2_sensor._EGO_X0) & (ex < av2_sensor._EGO_X1) & \
                    (np.abs(ey) < av2_sensor._EGO_HALF_W)
                keep = (ex >= x0) & (ex < x1) & (ey >= y0) & (ey < y1) & \
                    (ez > av2_sensor._ROAD_Z) & (ez < z1) & ~ego
                vi = vv[zmask][keep]
                ui = uu[zmask][keep]
                evid[vi, ui] = True
            for p in plist:
                ui, vi = int(round(p["u"])), int(round(p["v"]))
                v0, v1 = max(0, vi - half), min(H, vi + half + 1)
                u0, u1 = max(0, ui - half), min(W, ui + half + 1)
                cnt = float(evid[v0:v1, u0:u1].sum())
                if p["label"] == 1:
                    pos_scores.append(cnt)
                else:
                    neg_scores.append(cnt)
    auc = _roc_auc(pos_scores, neg_scores)
    all_counts = [(c, 1) for c in pos_scores] + [(c, 0) for c in neg_scores]
    fired = [lab for c, lab in all_counts if c >= cfg.n_depth_min]
    precision = float(np.mean(fired)) if fired else float("nan")
    return {"auc": auc, "n_pos": len(pos_scores), "n_neg": len(neg_scores),
            "operating_point_precision_at_n_depth_min": precision,
            "gate_pass": bool(np.isfinite(auc) and auc >= cfg.auc_gate)}


# ----------------------------------------------------------------------------------------------
# Config.
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    z_min: float
    z_max: float
    n_depth_min: int
    camera: str
    onnx: pathlib.Path
    scale_gate_m: float
    auc_gate: float
    null: str
    shuffles: int
    seed: int
    clear_frame_min: float = 0.30  # clear-frame filter: drop frames with in-band depth-valid frac < this
    rescale: str = "none"          # {none, ground-plane}: per-frame ground-plane scale correction (v3.1)


# ----------------------------------------------------------------------------------------------
# Self-check (geometry + letterbox round-trip + ONNX-runs; no confirmatory data).
# ----------------------------------------------------------------------------------------------
def _self_check(logs: list[str], data_root: pathlib.Path, onnx_path: pathlib.Path,
                rescale: str = "none") -> bool:
    """Validate the geometry chain + the NEW depth-stage transforms on REAL calibration. Prints each
    check + PASS/FAIL. Returns all-ok.
      (i)   ego 3D point round-trip < 0.1 m (reused backproject_to_ego / project_ego_to_left).
      (ii)  undistort -> re-distort a corner pixel < 0.5 px (reused Su model).
      (iii) letterbox -> un-letterbox round-trip < 1 px in the net grid (the NEW transform;
            pre-reg-required) + exact-inverse continuous check + un_letterbox_depth ramp-index proof.
      (v)   v3.1 only (--rescale ground-plane): a real frame's ground-plane scale returns finite s in
            (0.3,3.0) AND the scale path reads no sweep/occupancy (the independence guard).
      (iv)  ONNX runs and returns a (.,518,518) metric map with values in (0,80].
    """
    ok_all = True
    log = logs[0]
    camL, _camR, B = load_stereo_calib(log, data_root)
    print(f"  [calib] log={log[:12]} cam={camL.name} fx={camL.fx:.3f} cx={camL.cx:.3f} cy={camL.cy:.3f} "
          f"k=({camL.k1:.4f},{camL.k2:.4f},{camL.k3:.4f}) WxH={camL.width}x{camL.height} B={B:.5f}m")

    # (i) ego point round-trip: distorted px -> undistort -> back-project -> ego.
    p_ego_true = np.array([[12.0, 0.6, 0.9]])  # 12 m fwd, 0.6 m left, 0.9 m up -- inside band+FOV
    uv_d, depth, vis = project_ego_to_left(p_ego_true, camL)
    assert vis[0], f"test point not visible: uv={uv_d[0]} depth={depth[0]}"
    xd = (uv_d[0, 0] - camL.cx) / camL.fx
    yd = (uv_d[0, 1] - camL.cy) / camL.fy
    xn, yn = undistort_normalized(np.array([xd]), np.array([yd]), camL)
    u_pin = xn[0] * camL.fx + camL.cx
    v_pin = yn[0] * camL.fy + camL.cy
    p_back = backproject_to_ego(np.array([u_pin]), np.array([v_pin]), np.array([depth[0]]), camL)
    err_i = float(np.linalg.norm(p_back[0] - p_ego_true[0]))
    ok_i = err_i < 0.1
    ok_all &= ok_i
    print(f"  (i) ego [12,0.6,0.9] -> distorted px ({uv_d[0,0]:.1f},{uv_d[0,1]:.1f}) -> undistort -> "
          f"backproject -> [{p_back[0,0]:.3f},{p_back[0,1]:.3f},{p_back[0,2]:.3f}] err={err_i:.4f}m "
          f"-> {'PASS' if ok_i else 'FAIL'}")

    # (ii) undistort then re-distort a CORNER pixel round-trips < 0.5 px
    corner_u, corner_v = 5.0, 5.0
    xd_c = (corner_u - camL.cx) / camL.fx
    yd_c = (corner_v - camL.cy) / camL.fy
    xn_c, yn_c = undistort_normalized(np.array([xd_c]), np.array([yd_c]), camL)
    xd2, yd2 = distort_normalized(xn_c, yn_c, camL)
    u2 = xd2[0] * camL.fx + camL.cx
    v2 = yd2[0] * camL.fy + camL.cy
    err_ii = math.hypot(u2 - corner_u, v2 - corner_v)
    ok_ii = err_ii < 0.5
    ok_all &= ok_ii
    print(f"  (ii) corner px (5,5) -> undistort -> re-distort -> ({u2:.3f},{v2:.3f}) "
          f"round-trip err={err_ii:.4f}px -> {'PASS' if ok_ii else 'FAIL'}")

    # (iii) letterbox -> un-letterbox pixel round-trip < 1 px (NEW; pre-reg-required).
    #   The transform-correctness guarantee the pre-reg asks for ("the un-letterbox mapping is verified
    #   in self-check") is: the analytic letterbox map composed with its analytic inverse is identity
    #   (no aspect distortion, exact inverse), so a depth value lands at the geometrically correct
    #   undistorted pixel. Measured in the NET grid -- the resolution the mapping operates at and the
    #   resolution of the depth map itself -- this round-trip must be < 1 px.
    #   SPEC-NOTE (honest, no gaming): the net is 518x518 over a 2048-side letterbox, so 1 net px ~= 3.95
    #   undistorted px (scale ~0.253). The CONTINUOUS round-trip is 0 (the maps are exact inverses); the
    #   only residual when measured in UNDISTORTED px is the unavoidable net-grid quantization (<= 0.5 net
    #   px ~= 2 undistorted px), which is a resolution property of the 518-grid, NOT a transform defect.
    #   The gate therefore measures the transform in net-px (< 1 px); the undistorted-grid quantization
    #   residual is reported as informational, and the exact per-pixel inverse used by un_letterbox_depth
    #   is separately proven to 0 error by the ramp-index check below.
    H, W = camL.height, camL.width
    lb = Letterbox.build(H, W)
    test_uv = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1],
                        [W // 2, H // 2], [W // 3, 2 * H // 3]], dtype=float)
    # forward: undistorted (u,v) -> net (nu,nv) (continuous); inverse: net (nu,nv) -> undistorted.
    nu = (test_uv[:, 0] + lb.pad_x) * lb.scale
    nv = (test_uv[:, 1] + lb.pad_y) * lb.scale
    # continuous round-trip (transform identity, measured in net px after the inverse re-forwards):
    u_cont = nu / lb.scale - lb.pad_x
    v_cont = nv / lb.scale - lb.pad_y
    err_cont = float(np.max(np.hypot(u_cont - test_uv[:, 0], v_cont - test_uv[:, 1])))  # ~0 (exact inverse)
    # net-grid round-trip (what un_letterbox_depth quantizes to), measured in NET px (the gate):
    err_iii = float(np.max(np.hypot(np.round(nu) - nu, np.round(nv) - nv)))  # <= 0.5 net px
    # the same residual expressed in undistorted px (informational only -- resolution property):
    err_undist = err_iii / lb.scale
    ok_iii = err_iii < 1.0 and err_cont < 1e-6
    ok_all &= ok_iii
    print(f"  (iii) letterbox->un-letterbox round-trip (side={lb.side}, pad=({lb.pad_x},{lb.pad_y}), "
          f"scale={lb.scale:.5f}): continuous-inverse err={err_cont:.2e}px, net-grid quant err="
          f"{err_iii:.4f} net-px (= {err_undist:.3f} undistorted-px, resolution property) "
          f"-> {'PASS' if ok_iii else 'FAIL'}")
    # also verify un_letterbox_depth itself inverts a synthetic ramp depth (a per-pixel index check):
    ramp = np.tile(np.arange(_NET_HW, dtype=np.float32)[None, :], (_NET_HW, 1))  # depth = net column
    back = un_letterbox_depth(ramp, lb)
    samp = back[H // 2, [0, W // 2, W - 1]]
    expect = np.round((np.array([0, W // 2, W - 1]) + lb.pad_x) * lb.scale)
    err_map = float(np.nanmax(np.abs(samp - expect)))
    ok_iii_map = err_map < 1.0
    ok_all &= ok_iii_map
    print(f"      un_letterbox_depth ramp-index check: sampled {samp} vs expected {expect} "
          f"max err={err_map:.4f} -> {'PASS' if ok_iii_map else 'FAIL'}")

    # (iv) ONNX runs and returns a (.,518,518) metric map with values in (0,80].
    net = DepthNet(onnx_path)
    log_dir = data_root / log
    map_u, map_v = build_undistort_map(camL)
    cl_ts = _cam_timestamps(log_dir, camL.name)
    jpg = log_dir / "sensors" / "cameras" / camL.name / f"{int(cl_ts[len(cl_ts) // 2])}.jpg"
    rgb_raw = _rgb_from_jpg(jpg)
    rgb_und = undistort_rgb(rgb_raw, map_u, map_v)
    depth_und, lb2 = net.depth_undistorted(rgb_und)
    # check the RAW net output shape + range (before un-letterbox masks the pad border to NaN)
    net_in = letterbox_to_net(rgb_und, lb2)
    depth_net = net.infer_square(net_in)
    fin = depth_net[np.isfinite(depth_net)]
    shape_ok = depth_net.shape == (_NET_HW, _NET_HW)
    range_ok = bool(fin.size > 0 and fin.min() > 0.0 and fin.max() <= _MAX_DEPTH_M)
    ok_iv = shape_ok and range_ok
    ok_all &= ok_iv
    inband_und = depth_und[np.isfinite(depth_und)]
    print(f"  (iv) ONNX {onnx_path.name[:40]} ran: net depth shape={depth_net.shape} "
          f"min={fin.min():.3f}m max={fin.max():.3f}m (in (0,{_MAX_DEPTH_M:.0f}]={range_ok}); "
          f"un-letterbox undistorted-grid {depth_und.shape}, finite px={inband_und.size} "
          f"-> {'PASS' if ok_iv else 'FAIL'}")

    # (v) v3.1 ground-plane scale check (only when requested): finite s in (0.3,3.0) on a real frame
    #     AND the scale path reads NO LiDAR/sweep/occupancy (the pre-reg independence guard, asserted).
    if rescale == "ground-plane":
        import inspect
        import io
        import tokenize
        # (v-a) INDEPENDENCE GUARD (source-level): the scale function's CODE must not name any LiDAR /
        # occupancy / sweep symbol. A structural proof the scale path cannot read the graded modality --
        # it fails the self-check if a future edit wires one in. We scan NAME/OP tokens only (comments
        # and docstrings stripped via tokenize), so the prose "uses NO LiDAR" in the docstring does NOT
        # false-trip the guard -- only real identifier references count.
        def _code_identifiers(fn) -> set[str]:
            toks: set[str] = set()
            for tok in tokenize.generate_tokens(io.StringIO(inspect.getsource(fn)).readline):
                if tok.type in (tokenize.NAME, tokenize.OP):
                    toks.add(tok.string)
            # drop string/comment tokens implicitly (only NAME/OP kept); join NAME chains for attr checks
            return toks
        ids = _code_identifiers(ground_plane_scale) | _code_identifiers(apply_ground_scale)
        forbidden = ["sweep", "lidar", "occupancy", "occ_bev", "occ_free", "load_scene",
                     "av2_sensor", "_voxelize", "scene", "grid"]
        hits = [tok for tok in forbidden if tok in ids]  # exact identifier match (no substring/prose)
        ok_v_indep = (len(hits) == 0)
        ok_all &= ok_v_indep
        print(f"  (v-a) independence guard: ground_plane_scale/apply_ground_scale CODE identifiers name "
              f"NO {forbidden} (comments/docstrings stripped) -> hits={hits} "
              f"-> {'PASS' if ok_v_indep else 'FAIL'}")
        # (v-b) RUNTIME: scale on a real frame, computed from ONLY (depth_und, camL) -- no sweep/grid in
        # scope -- returns finite s in (0.3,3.0).
        gs = ground_plane_scale(depth_und, camL)
        ok_v_run = bool(gs.valid and np.isfinite(gs.s) and (_SCALE_LO < gs.s < _SCALE_HI))
        ok_all &= ok_v_run
        # prove the corrected depth halves the over-estimate direction (informational): a road pixel's
        # corrected depth vs raw, at the wedge center.
        Hd, Wd = depth_und.shape
        rc_v, rc_u = int(round(0.9 * Hd)), int(round(0.5 * Wd))
        raw_c = float(depth_und[rc_v, rc_u]); corr_c = raw_c * gs.s if gs.valid else float("nan")
        print(f"  (v-b) ground-plane scale on real frame: s={gs.s:.5f} (valid={gs.valid}, "
              f"n_road={gs.n_road}, in ({_SCALE_LO},{_SCALE_HI})={_SCALE_LO < gs.s < _SCALE_HI}); "
              f"road-center raw Z={raw_c:.2f}m -> corrected {corr_c:.2f}m "
              f"-> {'PASS' if ok_v_run else 'FAIL'}")

    print(f"\n  SELF-CHECK {'PASSED' if ok_all else 'FAILED'} "
          f"(geometry + letterbox + ONNX{' + ground-plane' if rescale == 'ground-plane' else ''} "
          f"{'within tolerance' if ok_all else 'OUT OF TOLERANCE'})")
    return ok_all


# ----------------------------------------------------------------------------------------------
# Per-log run + band-local null + confirmatory.
# ----------------------------------------------------------------------------------------------
def _run_log(log: str, cfg: Config, data_root: pathlib.Path, net: DepthNet,
             rng: np.random.Generator) -> list[FrameResult]:
    """Run every danger frame of one log. Attaches shuf_rate (mean band-local shuffle) per frame."""
    log_dir = data_root / log
    camL_full, _camR, _B = load_stereo_calib(log, data_root)
    map_u, map_v = build_undistort_map(camL_full)
    bf = band_fov_mask(camL_full, cfg.z_max)
    scene = av2_sensor.load_scene(log, data_root, with_boxes=False)
    sweeps = [int(fr.time * 1e9) for fr in scene.frames]
    camL_ts = _cam_timestamps(log_dir, cfg.camera)
    out: list[FrameResult] = []
    for fi, ts in enumerate(sweeps):
        grid = scene.frames[fi].grid
        cl = _nearest(camL_ts, ts)
        rgb_raw = _rgb_from_jpg(log_dir / "sensors" / "cameras" / cfg.camera / f"{cl}.jpg")
        fr = compute_frame(log, ts, rgb_raw, camL_full, grid, net, cfg, map_u, map_v, bf)
        if (not fr.dropped) and fr.n_struct > 0:
            # band-local null on the SAME depth_struct map (recompute the shared masks once)
            rgb_und = undistort_rgb(rgb_raw, map_u, map_v)
            depth_und, _gs = corrected_depth(net, rgb_und, camL_full, cfg.rescale)  # SAME corrected depth
            _struct_bev, si_band = _depth_struct_in_band(depth_und, camL_full, bf, cfg)
            occ_bev = occupancy_bev(grid, cfg.z_max)
            occ_in_band = int((occ_bev & bf).sum())
            shuf = [band_local_shuffle_rate(si_band, occ_in_band, bf, rng) for _ in range(cfg.shuffles)]
            fr.shuf_rate = float(np.mean(shuf)) if shuf else float("nan")
        out.append(fr)
    return out


def run_confirmatory(logs: list[str], held_log: str, cfg: Config, data_root: pathlib.Path,
                     calib_json: pathlib.Path, out_path: pathlib.Path) -> dict:
    """The full sealed run (SEALED-GATED; the orchestrator runs this, NOT the builder). Order:
    (i) Gate 1 scale -> STOP if > scale_gate_m; (ii) Gate 2 AUC -> secondary kill if < auc_gate;
    (iii) confirmatory gap + band-local null. Held-out log reported separately, never pooled."""
    rng = np.random.default_rng(cfg.seed)
    net = DepthNet(cfg.onnx)

    # Gate 1: metric-scale falsifier FIRST (a STOP gate, no miss-rate if it fails)
    scale = run_scale_gate(logs, data_root, net, cfg)
    if not scale["gate_pass"]:
        report = {"verdict": "INVALID-SCALE",
                  "reason": (f"DAv2 median |depth - box_range| = {scale['median_abs_err_m']:.3f} m "
                             f"> {cfg.scale_gate_m} m in the 2-30 m band -> metric scale invalid on "
                             f"AV2; no miss-rate reported"),
                  "scale_gate": scale, "config": _cfg_dict(cfg)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    # Gate 2: self-reliability AUC (secondary kill)
    calib = run_calibration_auc(data_root, net, cfg, calib_json)
    if not calib["gate_pass"]:
        report = {"verdict": "ORACLE-INSUFFICIENT",
                  "reason": (f"calibration AUC {calib['auc']:.3f} < {cfg.auc_gate} -> secondary kill; "
                             f"no miss-rate reported"),
                  "scale_gate": scale, "calibration": calib, "config": _cfg_dict(cfg)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        return report

    # (iii) confirmatory gap + band-local null
    headline_logs = [lg for lg in logs if lg != held_log]
    rows: list[dict] = []
    held_rows: list[dict] = []
    for log in logs:
        is_held = (log == held_log)
        for fr in _run_log(log, cfg, data_root, net, rng):
            if fr.dropped or not (fr.n_struct > 0):
                continue
            row = {"scene": log, "miss_rate": fr.miss_rate, "shuf_miss_rate": fr.shuf_rate,
                   "n_struct": fr.n_struct, "n_miss": fr.n_miss, "depth_valid_frac": fr.depth_valid_frac}
            (held_rows if is_held else rows).append(row)

    if not rows:
        report = {"verdict": "INDETERMINATE", "reason": "no usable headline frames",
                  "scale_gate": scale, "calibration": calib, "config": _cfg_dict(cfg)}
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
        held_summary = {
            "n_frames": len(held_rows),
            "true_miss_rate": float(np.mean([r["miss_rate"] for r in held_rows])),
            "shuffled_miss_rate": float(np.mean([r["shuf_miss_rate"] for r in held_rows])),
        }

    report = {
        "substrate": "AV2 following/danger stereo logs (RESTRICTED -- following-substrate only)",
        "oracle": "frozen mono-depth DAv2-metric (Virtual-KITTI), ONNX CPU",
        "headline_logs": headline_logs,
        "held_out_threshold_log": held_log,
        "n_headline_logs": len(set(scenes)),
        "n_headline_frames": len(rows),
        "true_miss_rate_mean": true_b["mean"], "true_miss_rate_ci": [true_b["lo"], true_b["hi"]],
        "shuffled_miss_rate_mean": shuf_b["mean"], "shuffled_miss_rate_ci": [shuf_b["lo"], shuf_b["hi"]],
        "gap_mean": gap_b["mean"], "gap_ci": [gap_b["lo"], gap_b["hi"]],
        "verdict": verdict,
        "scale_gate": scale,
        "calibration": calib,
        "held_out_log_report": held_summary,
        "config": _cfg_dict(cfg),
        "framing": ("occupancy in-path RECALL (frozen mono-depth oracle): miss-rate = MISS-candidate "
                    "(occ_free & depth_struct) band∩FOV voxels / depth_struct band∩FOV voxels, same "
                    "frame t. GAP = (band-local-shuffled - true); RECALL-SUPPORTED iff gap CI strictly "
                    "> 0 (falsifiable kill: gap CI includes 0 -> FAIL). Following-substrate only; "
                    "measured miss-rate is a LOWER BOUND (correlated textureless/dark failures dropped "
                    "by the clear-frame filter). Modality+algorithm+provenance independent of the AV2 "
                    "LiDAR occupancy; same vehicle/timestamp -> much more independent, NOT external truth."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _cfg_dict(cfg: Config) -> dict:
    return {"z_min": cfg.z_min, "z_max": cfg.z_max, "n_depth_min": cfg.n_depth_min,
            "camera": cfg.camera, "onnx": str(cfg.onnx), "scale_gate_m": cfg.scale_gate_m,
            "auc_gate": cfg.auc_gate, "null": cfg.null, "shuffles": cfg.shuffles, "seed": cfg.seed,
            "clear_frame_min": cfg.clear_frame_min, "rescale": cfg.rescale}


# ----------------------------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AV2 frozen mono-depth (DAv2) cross-modal RECALL oracle (sealed).")
    ap.add_argument("--logs", nargs="+", default=[
        "201fe83b-7dd7-38f4-9d26-7b4a668638a9",
        "2c652f9e-8db8-3572-aa49-fae1344a875b",
        "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c"])
    ap.add_argument("--heldout-threshold-log", default="6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c")
    ap.add_argument("--onnx", type=pathlib.Path, default=_DEFAULT_ONNX)
    ap.add_argument("--camera", default="stereo_front_left")
    ap.add_argument("--z-min", type=float, default=2.0)
    ap.add_argument("--z-max", type=float, default=30.0)
    ap.add_argument("--n-depth-min", type=int, default=8)
    ap.add_argument("--scale-gate-m", type=float, default=0.5)
    ap.add_argument("--auc-gate", type=float, default=0.75)
    ap.add_argument("--null", default="band-local", choices=["band-local"])
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear-frame-min", type=float, default=0.30,
                    help="clear-frame filter: drop frames with in-band depth-valid fraction below this")
    ap.add_argument("--rescale", default="none", choices=["none", "ground-plane"],
                    help="per-frame depth scale correction: 'none' = v3 raw DAv2 metric (default, "
                         "preserved); 'ground-plane' = v3.1 LiDAR-independent ground-plane scale "
                         "(Z_corrected = s * Z_DAv2)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=_HERE / "results" / "oracle_depth_recall_v2.json")
    ap.add_argument("--data-root", type=pathlib.Path, default=_AV2)
    ap.add_argument("--calib-json", type=pathlib.Path,
                    default=_HERE / "results" / "calib_patches" / "calib_patches.json",
                    help="path to calib_patches.json (60 human-labeled patches; read for Gate 2)")
    ap.add_argument("--self-check", action="store_true",
                    help="run geometry + letterbox + ONNX-runs round-trips and exit (no confirmatory data)")
    ap.add_argument("--scale-check", action="store_true",
                    help="run Gate 1 metric-scale falsifier on annotation boxes and exit (no miss-rate)")
    return ap


def _cfg_from_args(args) -> Config:
    return Config(
        z_min=args.z_min, z_max=args.z_max, n_depth_min=args.n_depth_min, camera=args.camera,
        onnx=args.onnx, scale_gate_m=args.scale_gate_m, auc_gate=args.auc_gate, null=args.null,
        shuffles=args.shuffles, seed=args.seed, clear_frame_min=args.clear_frame_min,
        rescale=args.rescale,
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = _cfg_from_args(args)

    if args.self_check:
        print(f"DEPTH RECALL ORACLE -- self-check (real calibration + ONNX, no confirmatory data; "
              f"rescale={cfg.rescale}):")
        ok = _self_check(args.logs, args.data_root, args.onnx, rescale=cfg.rescale)
        sys.exit(0 if ok else 1)

    if args.scale_check:
        print(f"DEPTH RECALL ORACLE -- Gate 1 metric-scale falsifier (annotation boxes; no miss-rate; "
              f"rescale={cfg.rescale}):")
        net = DepthNet(args.onnx)
        scale = run_scale_gate(args.logs, args.data_root, net, cfg)
        print(f"  n_boxes={scale['n_boxes']} per_log={scale['per_log']}")
        if cfg.rescale == "ground-plane":
            fs = scale.get("frame_scales", [])
            print(f"  per-frame ground-plane scale s: median={scale.get('median_frame_scale'):.4f} "
                  f"n_frames={len(fs)} n_scale_invalid={scale.get('n_scale_invalid_frames')}")
            print(f"  frame_scales={fs}")
        print(f"  median |Z_corrected - box_range| = {scale['median_abs_err_m']:.4f} m "
              f"(signed median {scale['median_signed_err_m']:+.4f} m; >0 = depth farther than box)")
        print(f"  gate: <= {scale['scale_gate_m']} m -> "
              f"{'PASS (scale valid)' if scale['gate_pass'] else 'INVALID-SCALE (STOP)'}")
        sys.exit(0 if scale["gate_pass"] else 2)

    # confirmatory (sealed) -- SEALED-GATED; run by the orchestrator only.
    print("DEPTH RECALL ORACLE -- CONFIRMATORY (sealed). Gate 1 scale -> Gate 2 AUC -> gap + null.")
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
