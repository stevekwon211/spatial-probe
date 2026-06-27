# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
#
# ==============================================================================================
# RUNS ON THE RUNPOD GPU POD ONLY -- IMPORTS torch + IGEV-Stereo (AND cv2 FOR RECTIFICATION);
# NOT IMPORTED BY THE REPO ORACLE OR ANY TEST. The repo .venv core stays pure numpy/scipy/Pillow/
# pyarrow (CLAUDE.md: no torch, no cv2 in core). This script is the EXTERNAL preprocessing front-end
# declared in `oracle_stereo_recall_learned_preregistration.md` (git 1c8b357): it consumes the AV2
# stereo JPEGs + per-log calibration and emits, per (log, camera-frame, side), a disparity artifact
# `disp_<log>_<cam_ts>_<side>.npz` that is a BYTE-FOR-BYTE drop-in for the sealed census matcher on the
# UNDISTORTED 2x-downsampled grid (see ARTIFACT CONTRACT in oracle_stereo_recall.py). The local numpy
# oracle then grades the artifacts with `--disparity-source artifact` -- NO GPU, fully reproducible.
#
# This file CANNOT be exercised in the repo .venv (no torch, no cv2, no GPU). It is written to the
# pre-reg + the artifact contract and is validated on the pod by `--self-check` (geometry sanity vs
# LiDAR) BEFORE the full pass. Do not import it from the oracle or tests.
# ==============================================================================================
"""IGEV-Stereo (Scene-Flow zero-shot) disparity front-end -> census-drop-in artifacts (POD ONLY).

Pipeline per stereo pair (mirrors the census front-end up to the matcher, then swaps in IGEV):
  raw L/R JPEG
    -> UNDISTORT with the oracle's Su-model maps (build_undistort_map/remap_gray)  [grid == census]
    -> RECTIFY the undistorted (distortion-free) pair via cv2.stereoRectify(R_rel,T_rel)  [cv2: pod-only]
    -> IGEV-Stereo (sceneflow.pth, zero-shot) on the rectified pair, both directions (L->R, R->L)
    -> WARP the rectified disparity BACK to the undistorted grid AND convert to the census disparity
       convention (positive = nearer, Z = cam.fx * B / d) so it is a literal drop-in
    -> 2x-downsample (d[::2,::2]/2)  [== census undL_s grid + downsampled-pixel disparity]
    -> save disp_<log>_<cam_ts>_<side>.npz  (key `disp`, float32, NaN = IGEV-invalid/out-of-frame)

SPEC-NOTEs (judgment calls, declared loudly):
  * cv2 IS needed on the pod (for stereoRectify / rectify maps). The pre-reg allowed numpy-OR-cv2 and
    said "cv2 OK on pod". Hand-rolling Fusiello rectification in numpy is feasible but a rectification
    sign error is a silent geometry bug (a one-way door for the result's validity), so the battle-
    tested cv2.stereoRectify is the lower-risk choice WHERE cv2 is allowed (pod). UNDISTORTION is NOT
    done by cv2 -- it reuses the oracle's Su-model maps so the undistorted grid matches census exactly
    (AV2 (k1,k2,k3) happen to map to cv2's radial coeffs, but reusing the oracle removes any doubt).
  * WARP-BACK + convention conversion is the crux. For each undistorted-LEFT pixel (u,v):
        ray_undL = K_undL^-1 [u,v,1];  ray_rect = R1 @ ray_undL
        (u_rect,v_rect) = project ray_rect with K_rect;  d_rect = bilinear(disp_rect, u_rect, v_rect)
        d_out = (camL.fx / K_rect.fx) * ray_rect_z * d_rect        # closed form (derivation in code)
    This returns the disparity in the census convention so downstream Z = camL_s.fx*B/d and
    backproject_to_ego(...,camL) reproduce the SAME ego point IGEV implies. (Right side: same with
    R2/K_rectR/camR.) Backward warp => no scatter holes.
  * LR-CONSISTENCY is the INHERITED LOCAL filter. The pre-reg's "matcher LR-consistency at tol=1.0px"
    == the oracle's `lr_consistency` run on the IGEV L+R artifacts (byte-for-byte the same code as on
    census). So the pod emits RAW IGEV disparities + the IGEV finite-mask (NaN); it does NOT pre-apply
    LR-consistency (that would double-apply and break "lr-consistency 1.0px inherited byte-for-byte").
    Both L and R artifacts are emitted precisely so the local lr_consistency has the R map to check.
  * The census texture-gate is DROPPED for IGEV (pre-reg item #3); nothing here re-introduces it.

Run (on the pod, AFTER pod_setup.sh):
  python experiments/occquery_v0/igev_disparity_pod.py \
    --logs 201fe83b-7dd7-38f4-9d26-7b4a668638a9 2c652f9e-8db8-3572-aa49-fae1344a875b \
           6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
    --data-root $DATA_ROOT --igev-repo /workspace/IGEV-Stereo \
    --checkpoint /workspace/IGEV-Stereo/pretrained_models/sceneflow/sceneflow.pth \
    --out-dir results/igev_disp --self-check   # run --self-check on ONE frame first
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

# Reuse the SEALED oracle's calibration + undistortion + artifact-name helpers so the artifact grid
# and filename are identical to what the census path / loader expect. The oracle is pure numpy -- it
# does NOT pull in torch; importing it here is one-directional (the oracle never imports this file).
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "experiments" / "dynfield_v0"))

from oracle_stereo_recall import (  # noqa: E402  (pure-numpy oracle helpers; NO torch)
    AV2Camera,
    _cam_timestamps,
    _nearest,
    _read_feather,
    artifact_name,
    backproject_to_ego,
    build_undistort_map,
    load_stereo_calib,
    remap_gray,
)

_DOWNSAMPLE = 2  # MUST equal oracle _DOWNSAMPLE so the artifact lands on the census undL_s grid.


# ----------------------------------------------------------------------------------------------
# Heavy, POD-ONLY deps are imported lazily so `python -c "import ..."` introspection on a torch-less
# box still parses the file. Calling any of these without torch/cv2/IGEV raises a clear message.
# ----------------------------------------------------------------------------------------------
def _import_torch():
    try:
        import torch  # noqa
        return torch
    except Exception as e:  # pragma: no cover - pod only
        raise SystemExit(f"torch is required ON THE POD and is intentionally absent in the repo .venv: {e}")


def _import_cv2():
    try:
        import cv2  # noqa
        return cv2
    except Exception as e:  # pragma: no cover - pod only
        raise SystemExit(f"cv2 is required ON THE POD (rectification) and is absent in the repo .venv: {e}")


def _load_igev(igev_repo: str, checkpoint: str):  # pragma: no cover - pod only
    """Load IGEV-Stereo (gangweiX/IGEV-Stereo) with the Scene-Flow checkpoint, eval mode, on CUDA.
    The IGEV repo is added to sys.path so `core.igev_stereo` / `core.utils.utils` import."""
    torch = _import_torch()
    sys.path.insert(0, igev_repo)
    sys.path.insert(0, str(pathlib.Path(igev_repo) / "core"))
    from core.igev_stereo import IGEVStereo  # type: ignore

    # IGEV's demo args (defaults match the official demo_imgs.py for sceneflow.pth zero-shot).
    args = argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg", shared_backbone=False,
        corr_levels=2, corr_radius=4, n_downsample=2, slow_fast_gru=False,
        n_gru_layers=3, max_disp=192, mixed_precision=True, valid_iters=32,
    )
    model = torch.nn.DataParallel(IGEVStereo(args))
    # sceneflow.pth is a plain state_dict (tensors only), so weights_only=True is safe AND avoids the
    # arbitrary-code-execution risk of the default unpickler. If a future checkpoint pickles non-tensor
    # objects this will raise -- inspect the file before relaxing it, never blindly flip to False.
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.module
    model.to("cuda").eval()
    return model, args


def _igev_disparity(model, args, rgbL: np.ndarray, rgbR: np.ndarray) -> np.ndarray:  # pragma: no cover
    """IGEV disparity for the LEFT image of a RECTIFIED pair (positive = right feature at u-d). Inputs
    are HxWx3 uint8 (RGB). Returns float32 HxW (rectified-left disparity)."""
    torch = _import_torch()
    from core.utils.utils import InputPadder  # type: ignore

    def _t(img):
        return torch.from_numpy(img).permute(2, 0, 1).float()[None].to("cuda")  # [1,3,H,W], [0,255]

    with torch.no_grad():
        i1, i2 = _t(rgbL), _t(rgbR)
        padder = InputPadder(i1.shape, divis_by=32)
        i1, i2 = padder.pad(i1, i2)
        disp = model(i1, i2, iters=args.valid_iters, test_mode=True)  # [1,1,H,W] or [1,H,W]
        disp = padder.unpad(disp).squeeze().detach().cpu().numpy().astype(np.float32)
    return disp


def _igev_disparity_right(model, args, rgbL: np.ndarray, rgbR: np.ndarray) -> np.ndarray:  # pragma: no cover
    """RIGHT-image disparity of a RECTIFIED pair via the standard horizontal-flip trick: run IGEV with
    flipped-right as 'left', flip the result back. Positive = nearer (left feature at u+d)."""
    fl = lambda a: np.ascontiguousarray(a[:, ::-1, :])  # noqa: E731
    disp_flipped = _igev_disparity(model, args, fl(rgbR), fl(rgbL))
    return np.ascontiguousarray(disp_flipped[:, ::-1])


# ----------------------------------------------------------------------------------------------
# Image IO (RGB, mirrors the oracle's _gray_from_jpg semantics but keeps 3 channels for IGEV).
# ----------------------------------------------------------------------------------------------
def _rgb_from_jpg(path: pathlib.Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _undistort_rgb(rgb: np.ndarray, cam: AV2Camera) -> np.ndarray:
    """Undistort an RGB image with the oracle's Su-model maps, per channel, onto the undistorted
    pinhole grid (== census). Out-of-frame -> 0 (so IGEV sees black border, masked out post-warp)."""
    mapu, mapv = build_undistort_map(cam)
    chans = [np.nan_to_num(remap_gray(rgb[..., c].astype(float), mapu, mapv), nan=0.0) for c in range(3)]
    return np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# Rectification (cv2, POD ONLY) from the AV2 per-log stereo extrinsics.
# ----------------------------------------------------------------------------------------------
def _K(cam: AV2Camera) -> np.ndarray:
    return np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _relative_extrinsics(camL: AV2Camera, camR: AV2Camera) -> tuple[np.ndarray, np.ndarray]:
    """R_rel, T_rel mapping LEFT-camera coords -> RIGHT-camera coords (p_R = R_rel p_L + T_rel), from
    the AV2 sensor->ego extrinsics: p_ego = R @ p_cam + t. So R_rel = R_R^T R_L, T_rel = R_R^T (t_L - t_R)."""
    R_L, t_L = camL.R_cam2ego, camL.t_cam_in_ego
    R_R, t_R = camR.R_cam2ego, camR.t_cam_in_ego
    R_rel = R_R.T @ R_L
    T_rel = R_R.T @ (t_L - t_R)
    return R_rel, T_rel


def _stereo_rectify(camL: AV2Camera, camR: AV2Camera):  # pragma: no cover - pod only (cv2)
    """cv2.stereoRectify on the UNDISTORTED pinhole pair (distCoeffs = 0; already undistorted).
    Returns (R1, R2, P1, P2, K_rectL, K_rectR, size). R1/R2 rotate undistorted-L/R coords -> rectified."""
    cv2 = _import_cv2()
    size = (camL.width, camL.height)  # (W, H)
    R_rel, T_rel = _relative_extrinsics(camL, camR)
    zero = np.zeros(5, dtype=np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        _K(camL), zero, _K(camR), zero, size, R_rel, T_rel,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    return R1, R2, P1[:3, :3], P2[:3, :3], R1, R2, P1, P2, size


def _rectify_image(cv2, rgb: np.ndarray, K: np.ndarray, R: np.ndarray, K_rect: np.ndarray,
                   size) -> np.ndarray:  # pragma: no cover - pod only
    P = np.hstack([K_rect, np.zeros((3, 1))])
    mapx, mapy = cv2.initUndistortRectifyMap(K, np.zeros(5), R, P, size, cv2.CV_32FC1)
    return cv2.remap(rgb, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderValue=0)


# ----------------------------------------------------------------------------------------------
# Warp the rectified disparity BACK to the undistorted grid + census convention (the crux).
# ----------------------------------------------------------------------------------------------
def _bilinear_sample(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample `field` (HxW, may contain NaN) at float coords (u,v); out-of-bounds -> NaN.
    Any NaN corner propagates to NaN (so IGEV-invalid pixels stay invalid through the warp)."""
    H, W = field.shape
    u0 = np.floor(u).astype(np.intp); v0 = np.floor(v).astype(np.intp)
    fu = u - u0; fv = v - v0
    ok = (u0 >= 0) & (u0 < W - 1) & (v0 >= 0) & (v0 < H - 1)
    u0c = np.clip(u0, 0, W - 2); v0c = np.clip(v0, 0, H - 2)
    f00 = field[v0c, u0c]; f01 = field[v0c, u0c + 1]
    f10 = field[v0c + 1, u0c]; f11 = field[v0c + 1, u0c + 1]
    top = f00 * (1 - fu) + f01 * fu
    bot = f10 * (1 - fu) + f11 * fu
    out = top * (1 - fv) + bot * fv
    out = np.where(ok, out, np.nan)
    return out


def _warp_rect_disp_to_undist(disp_rect: np.ndarray, cam: AV2Camera, R: np.ndarray,
                              K_rect: np.ndarray, B: float) -> np.ndarray:
    """Backward-warp a rectified-image disparity to the UNDISTORTED `cam` grid in the census convention.

    For each undistorted pixel (u,v): ray_undL = K_cam^-1[u,v,1]; ray_rect = R @ ray_undL; sample
    disp_rect at the projection of ray_rect; then (derivation in module SPEC-NOTE):
        d_out = (cam.fx / K_rect.fx) * ray_rect_z * d_rect
    so Z = cam.fx * B / d_out reproduces the depth along the undistorted optical axis. NaN where the
    rect disparity is invalid (incl. <= 0). `B` is unused in the closed form but kept for clarity/audit.
    """
    H, W = cam.height, cam.width
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xn = (uu - cam.cx) / cam.fx
    yn = (vv - cam.cy) / cam.fy
    # ray_rect = R @ [xn, yn, 1]
    rx = R[0, 0] * xn + R[0, 1] * yn + R[0, 2]
    ry = R[1, 0] * xn + R[1, 1] * yn + R[1, 2]
    rz = R[2, 0] * xn + R[2, 1] * yn + R[2, 2]
    fx_r, fy_r = K_rect[0, 0], K_rect[1, 1]
    cx_r, cy_r = K_rect[0, 2], K_rect[1, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u_rect = fx_r * (rx / rz) + cx_r
        v_rect = fy_r * (ry / rz) + cy_r
    d_rect = _bilinear_sample(disp_rect, u_rect, v_rect)
    with np.errstate(invalid="ignore"):
        d_out = (cam.fx / fx_r) * rz * d_rect
        d_out = np.where(np.isfinite(d_out) & (d_out > 0), d_out, np.nan)
    return d_out.astype(np.float32)


def _downsample_disp(disp_full: np.ndarray, ds: int) -> np.ndarray:
    """Match census: undL_s = undL[::ds, ::ds]; disparity in DOWNSAMPLED pixels => values / ds."""
    if ds == 1:
        return disp_full.astype(np.float32)
    return (disp_full[::ds, ::ds] / ds).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Per-pair artifact production.
# ----------------------------------------------------------------------------------------------
def _save_disp(out_dir: pathlib.Path, log: str, cam_ts: int, side: str, disp: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / artifact_name(log, cam_ts, side)), disp=disp.astype(np.float32))


def _process_log(model, args, cv2, log: str, data_root: pathlib.Path, out_dir: pathlib.Path,
                 ds: int) -> dict:  # pragma: no cover - pod only
    """Emit disp_<log>_<stem>_L.npz for EVERY left stem and disp_<log>_<stem>_R.npz for EVERY right
    stem. Coverage rationale: the oracle only ever requests cl in left-stems and cr in right-stems
    (cl/cr are _nearest outputs, i.e. real jpg stems), so covering all stems covers all requests."""
    log_dir = data_root / log
    camL, camR, B = load_stereo_calib(log, data_root)
    R1, R2, K_rectL, K_rectR, _, _, P1, P2, size = _stereo_rectify(camL, camR)
    camL_ts = _cam_timestamps(log_dir, "stereo_front_left")
    camR_ts = _cam_timestamps(log_dir, "stereo_front_right")

    n_L = n_R = 0
    # LEFT artifacts: for each left stem, pair with the nearest right stem, run IGEV L->R, warp back.
    for clts in camL_ts:
        crts = _nearest(camR_ts, int(clts))
        rgbL = _undistort_rgb(_rgb_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{clts}.jpg"), camL)
        rgbR = _undistort_rgb(_rgb_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_right" / f"{crts}.jpg"), camR)
        rectL = _rectify_image(cv2, rgbL, _K(camL), R1, K_rectL, size)
        rectR = _rectify_image(cv2, rgbR, _K(camR), R2, K_rectR, size)
        disp_rect = _igev_disparity(model, args, rectL, rectR)
        disp_rect = np.where(np.isfinite(disp_rect) & (disp_rect > 0), disp_rect, np.nan)
        disp_undL = _warp_rect_disp_to_undist(disp_rect, camL, R1, K_rectL, B)
        _save_disp(out_dir, log, int(clts), "L", _downsample_disp(disp_undL, ds))
        n_L += 1

    # RIGHT artifacts: for each right stem, pair with nearest left, run IGEV R->L, warp back.
    for crts in camR_ts:
        clts = _nearest(camL_ts, int(crts))
        rgbL = _undistort_rgb(_rgb_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{clts}.jpg"), camL)
        rgbR = _undistort_rgb(_rgb_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_right" / f"{crts}.jpg"), camR)
        rectL = _rectify_image(cv2, rgbL, _K(camL), R1, K_rectL, size)
        rectR = _rectify_image(cv2, rgbR, _K(camR), R2, K_rectR, size)
        disp_rect_R = _igev_disparity_right(model, args, rectL, rectR)
        disp_rect_R = np.where(np.isfinite(disp_rect_R) & (disp_rect_R > 0), disp_rect_R, np.nan)
        disp_undR = _warp_rect_disp_to_undist(disp_rect_R, camR, R2, K_rectR, B)
        _save_disp(out_dir, log, int(crts), "R", _downsample_disp(disp_undR, ds))
        n_R += 1

    meta = {
        "log": log, "baseline_m": B, "image_size_WH": list(size), "downsample": ds,
        "R_rel_T_rel": {"R_rel": _relative_extrinsics(camL, camR)[0].tolist(),
                        "T_rel": _relative_extrinsics(camL, camR)[1].tolist()},
        "R1_undL_to_rect": R1.tolist(), "R2_undR_to_rect": R2.tolist(),
        "P1": P1.tolist(), "P2": P2.tolist(),
        "K_undL": _K(camL).tolist(), "K_undR": _K(camR).tolist(),
        "K_rectL": K_rectL.tolist(), "K_rectR": K_rectR.tolist(),
        "n_left_artifacts": n_L, "n_right_artifacts": n_R,
        "convention": "disp float32 on undistorted 2x-downsampled grid, NaN=invalid, "
                      "positive=nearer, Z=cam.fx*B/disp (census drop-in)",
        "checkpoint": "sceneflow.pth (Scene-Flow zero-shot)",
    }
    (out_dir / f"rectify_meta_{log}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ----------------------------------------------------------------------------------------------
# Self-check: geometry sanity of ONE frame vs the LiDAR surface (run before the full pass).
# ----------------------------------------------------------------------------------------------
def _self_check(model, args, cv2, log: str, data_root: pathlib.Path, ds: int) -> float:  # pragma: no cover
    """Back-project IGEV's LEFT disparity (first frame) to the ego frame and print the median nearest-
    neighbour distance to the in-band, above-road LiDAR points. A small median (~< 1 voxel, 0.4 m)
    means the rectify->warp-back->backproject chain is geometrically sound BEFORE the full pass."""
    from scipy.spatial import cKDTree
    log_dir = data_root / log
    camL, camR, B = load_stereo_calib(log, data_root)
    R1, R2, K_rectL, K_rectR, _, _, P1, P2, size = _stereo_rectify(camL, camR)
    camL_ts = _cam_timestamps(log_dir, "stereo_front_left")
    camR_ts = _cam_timestamps(log_dir, "stereo_front_right")
    lidar = sorted(int(pathlib.Path(p).stem) for p in glob.glob(str(log_dir / "sensors" / "lidar" / "*.feather")))
    if not lidar or len(camL_ts) == 0:
        raise SystemExit("self-check needs lidar + stereo frames present on the pod")
    ts = lidar[len(lidar) // 2]
    clts = _nearest(camL_ts, ts); crts = _nearest(camR_ts, ts)
    rgbL = _undistort_rgb(_rgb_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_left" / f"{clts}.jpg"), camL)
    rgbR = _undistort_rgb(_rgb_from_jpg(log_dir / "sensors" / "cameras" / "stereo_front_right" / f"{crts}.jpg"), camR)
    rectL = _rectify_image(cv2, rgbL, _K(camL), R1, K_rectL, size)
    rectR = _rectify_image(cv2, rgbR, _K(camR), R2, K_rectR, size)
    disp_rect = _igev_disparity(model, args, rectL, rectR)
    disp_rect = np.where(np.isfinite(disp_rect) & (disp_rect > 0), disp_rect, np.nan)
    disp_undL = _warp_rect_disp_to_undist(disp_rect, camL, R1, K_rectL, B)   # full-res undistorted-left

    H, W = disp_undL.shape
    vv, uu = np.mgrid[0:H, 0:W]
    d = disp_undL
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = camL.fx * B / d
    good = np.isfinite(d) & (d > 0) & np.isfinite(Z) & (Z >= 2.0) & (Z <= 30.0)
    pts = backproject_to_ego(uu[good].astype(float), vv[good].astype(float), Z[good], camL)
    inband = (np.abs(pts[:, 1]) < 1.05) & (pts[:, 0] > 0) & (pts[:, 2] > 0.3)
    pts = pts[inband]

    lt = _read_feather(log_dir / "sensors" / "lidar" / f"{ts}.feather")
    lx = np.asarray(lt.column("x").to_pylist(), dtype=float)
    ly = np.asarray(lt.column("y").to_pylist(), dtype=float)
    lz = np.asarray(lt.column("z").to_pylist(), dtype=float)
    lm = (np.abs(ly) < 1.05) & (lx > 0) & (lz > 0.3) & (lx < 30.0)
    lidar_pts = np.stack([lx[lm], ly[lm], lz[lm]], axis=1)
    if len(pts) == 0 or len(lidar_pts) == 0:
        print(f"  [self-check] log={log[:8]} ts={ts}: insufficient in-band points (igev={len(pts)}, lidar={len(lidar_pts)})")
        return float("nan")
    dists, _ = cKDTree(lidar_pts).query(pts, k=1)
    med = float(np.median(dists))
    print(f"  [self-check] log={log[:8]} ts={ts}: IGEV in-band pts={len(pts)} vs LiDAR={len(lidar_pts)}; "
          f"median reproj error = {med:.3f} m  (sanity: expect ~< 0.4 m = 1 voxel on real obstacle backs)")
    return med


# ----------------------------------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IGEV-Stereo disparity front-end -> census-drop-in artifacts (POD ONLY).")
    ap.add_argument("--logs", nargs="+", default=[
        "201fe83b-7dd7-38f4-9d26-7b4a668638a9",
        "2c652f9e-8db8-3572-aa49-fae1344a875b",
        "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c"])
    ap.add_argument("--data-root", type=pathlib.Path, required=True,
                    help="$DATA_ROOT holding <log>/sensors/cameras/stereo_front_{left,right} + <log>/calibration")
    ap.add_argument("--igev-repo", type=str, required=True, help="path to the cloned gangweiX/IGEV-Stereo repo")
    ap.add_argument("--checkpoint", type=str, required=True, help="path to sceneflow.pth (Scene-Flow zero-shot)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=_HERE / "results" / "igev_disp")
    ap.add_argument("--downsample", type=int, default=_DOWNSAMPLE,
                    help="MUST equal the oracle downsample (2) so the artifact lands on the census grid")
    ap.add_argument("--self-check", action="store_true",
                    help="geometry sanity on ONE frame vs LiDAR, then exit (run this BEFORE the full pass)")
    return ap


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - pod only
    args = _build_argparser().parse_args(argv)
    if args.downsample != _DOWNSAMPLE:
        raise SystemExit(f"--downsample must be {_DOWNSAMPLE} to match the oracle grid (got {args.downsample})")
    cv2 = _import_cv2()
    model, igev_args = _load_igev(args.igev_repo, args.checkpoint)

    if args.self_check:
        print("IGEV DISPARITY POD -- self-check (geometry sanity vs LiDAR, ONE frame):")
        _self_check(model, igev_args, cv2, args.logs[0], args.data_root, args.downsample)
        sys.exit(0)

    for log in args.logs:
        print(f"IGEV DISPARITY POD -- processing log {log} ...")
        meta = _process_log(model, igev_args, cv2, log, args.data_root, args.out_dir, args.downsample)
        print(f"  wrote {meta['n_left_artifacts']} L + {meta['n_right_artifacts']} R artifacts to "
              f"{args.out_dir} (+ rectify_meta_{log}.json)")
    print(f"DONE. Re-grade locally (no GPU): oracle_stereo_recall.py --disparity-source artifact {args.out_dir}")


if __name__ == "__main__":
    main()
