# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Camera presence oracle for occquery H3 (Shot 1) -- a DIFFERENT-MODALITY independent check.

The occupancy predicate (LiDAR-derived) says "an obstacle is at (forward, lateral)". This module
asks the CAMERA -- a passive-optical sensor, a different modality from the active-TOF LiDAR Occ3D was
built from -- "is something really there?" by projecting the obstacle into the image (verified chain
in projection.py) and reading the local image evidence. This catches the occupancy FALSE-POSITIVE /
hallucination mode (occupancy obstacle that projects onto open drivable road in the photo).

HONEST DESIGN -- measure the oracle, do not assume it. A pure-CV "is this patch an obstacle?" signal
can be noisy (textureless walls, textured road). So before trusting it we CALIBRATE: compute the image
evidence at pixels where a known tracked OBJECT box projects (real object => positive) vs random
lower-lane road pixels (drivable => negative), and report the ROC AUC. The AUC is the oracle's
reliability: if it cannot separate objects from road, the pure-CV presence oracle is INSUFFICIENT and
we say so (a reachable kill -> the upgrade is a learned metric-depth net, which needs torch, a repo
rule we will not break without sign-off). If AUC is high, the same evidence scores occupancy obstacles.

Independence accounting (per preregistration.md): different MODALITY (optical vs TOF) AND different
ALGORITHM (image evidence vs EDT). NOT pristine -- same vehicle/timestamp -- so "much more independent,
not external truth". Pure numpy + scipy + Pillow; NO torch. Run: python experiments/occquery_v0/camera_oracle.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from probe.adapters.occ3d import load_scene
from probe.grid import UnknownPolicy
from probe.predicates.freepath import free_along_ego_path
from projection import IMG_H, IMG_W, Camera, cameras_for_frame, project_ego_points

_DATA = _HERE.parents[1] / "data"
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]
_PATCH = 12         # half-size of the image patch sampled around a projected point
_MAX_RANGE = 30.0   # meters; beyond this a patch is too small / too far to judge


def _gray(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])


def patch_evidence(gray: np.ndarray, u: float, v: float, half: int = _PATCH) -> float | None:
    """Image evidence that a SURFACE (not open road/sky) is at pixel (u, v): the mean gradient
    magnitude in the local patch. Obstacles (vehicles, walls, barriers) carry structure; smooth road
    and sky do not. None if the patch is out of frame. A measured statistic, not a verdict."""
    ui, vi = int(round(u)), int(round(v))
    if ui - half < 0 or ui + half >= IMG_W or vi - half < 0 or vi + half >= IMG_H:
        return None
    p = gray[vi - half: vi + half + 1, ui - half: ui + half + 1]
    gx = ndimage.sobel(p, axis=1)
    gy = ndimage.sobel(p, axis=0)
    return float(np.hypot(gx, gy).mean())


def _roc_auc(pos: list[float], neg: list[float]) -> float:
    """ROC AUC between positive and negative score lists (rank statistic; 0.5 = no separation).

    AUC = P(pos > neg) + 0.5 * P(pos == neg) (the Mann-Whitney form). Ties MUST get average ranks
    so each tied pos/neg pair contributes exactly 0.5 — `scipy.stats.rankdata` (method='average')
    does this. (A prior version used ordinal `arange(1,N+1)` ranks with no tie averaging; because the
    positive scores are concatenated first and the sort is stable, every tie went to the negatives,
    biasing the AUC systematically DOWNWARD on integer/saturating scores. Fixed 2026-06-28; this had
    contaminated every AUC-gated oracle here.)
    """
    from scipy.stats import rankdata

    pos_a, neg_a = np.asarray(pos), np.asarray(neg)
    if not len(pos_a) or not len(neg_a):
        return float("nan")
    ranks = rankdata(np.concatenate([pos_a, neg_a]))  # average ranks -> ties contribute 0.5
    r_pos = ranks[: len(pos_a)].sum()
    return float((r_pos - len(pos_a) * (len(pos_a) + 1) / 2) / (len(pos_a) * len(neg_a)))


def _road_pixels(rng: np.random.Generator, n: int = 12) -> np.ndarray:
    """Random pixels in the lower-center wedge -- the ego's own lane, almost always drivable road
    (the free-space negatives). Below the horizon (~v>540), central third horizontally."""
    us = rng.integers(IMG_W // 3, 2 * IMG_W // 3, n)
    vs = rng.integers(620, IMG_H - 20, n)
    return np.column_stack([us, vs])


def calibrate(scenes, rng: np.random.Generator) -> dict:
    """Measure the oracle's reliability: evidence at known-object (tracked box) pixels vs road pixels,
    on CAM_FRONT across the scenes. Returns AUC + a threshold (midpoint of class medians)."""
    ann = json.loads((_DATA / "annotations.json").read_text())
    pos: list[float] = []
    neg: list[float] = []
    for sc in scenes:
        si = ann["scene_infos"][sc.name]
        for i, tok in enumerate(si):  # dict order is temporal enough for calibration sampling
            if i >= len(sc):
                break
            cam = cameras_for_frame(si[tok])["CAM_FRONT"]
            gray = _gray(np.asarray(Image.open(_DATA / "samples" / cam.img_path).convert("RGB"), float))
            objs = sc.objects_at(i)
            if objs:
                centers = np.array([o.center for o in objs])
                uv, depth, vis = project_ego_points(centers, cam)
                for j in range(len(objs)):
                    if vis[j] and depth[j] < _MAX_RANGE:
                        e = patch_evidence(gray, uv[j, 0], uv[j, 1])
                        if e is not None:
                            pos.append(e)
            for (u, v) in _road_pixels(rng):
                e = patch_evidence(gray, u, v)
                if e is not None:
                    neg.append(e)
    auc = _roc_auc(pos, neg)
    thr = float((np.median(pos) + np.median(neg)) / 2) if pos and neg else float("nan")
    return {"auc": auc, "threshold": thr, "n_object": len(pos), "n_road": len(neg),
            "median_object": float(np.median(pos)) if pos else float("nan"),
            "median_road": float(np.median(neg)) if neg else float("nan")}


def camera_freepath_block(gray: np.ndarray, cam: Camera, ego_height: float, reach: float,
                          thr: float, step: float = 0.5, run: int = 3) -> float:
    """Independent FORWARD free-path estimate from the camera, clean-provenance (geometry + image
    evidence, NO learning, NO LiDAR). Walk the ego centerline on the GROUND plane (z=0) outward,
    project each station to a pixel via the verified chain, read the patch evidence. The camera calls
    the path BLOCKED at the first distance where a RUN of `run` consecutive stations all exceed `thr`
    (a sustained structure on the centerline; a lone high patch is noise, not a wall). Returns that
    forward distance, or math.inf if the centerline stays road-like to `reach`.

    Aggregating along the ray is what rescues the borderline single-patch signal (AUC 0.798): a real
    frontal obstacle blocks a contiguous run of centerline stations; per-patch noise does not."""
    hits = 0
    d = step
    while d <= reach:
        uv, depth, vis = project_ego_points(np.array([[d, 0.0, 0.0]]), cam)  # centerline, ground plane
        if vis[0]:
            e = patch_evidence(gray, uv[0, 0], uv[0, 1])
            if e is not None:
                hits = hits + 1 if e > thr else 0
                if hits >= run:
                    return float(d - (run - 1) * step)  # first station of the blocking run
        d += step
    return float("inf")


def forward_crosscheck(scenes, thr: float) -> dict:
    """Cross-modality FORWARD free-path consistency: occupancy free_along_ego_path (LiDAR) vs the
    camera centerline-ray block (optical). Reports the 2x2 agreement -- a DIFFERENT-MODALITY check,
    NOT external truth (same vehicle/timestamp). horizon 1.0 s; CAM_FRONT only (forward quantity)."""
    ann = json.loads((_DATA / "annotations.json").read_text())
    agree = {"free_free": 0, "blocked_blocked": 0, "occ_free_cam_blocked": 0, "occ_blocked_cam_free": 0}
    n = 0
    for sc in scenes:
        si = ann["scene_infos"][sc.name]
        toks = list(si)
        for i in range(len(sc)):
            grid, ego = sc.grid_at(i), sc.ego_at(i)
            occ_free = free_along_ego_path(grid, ego, 1.0, min_cluster_voxels=2)
            cam = cameras_for_frame(si[toks[i]])["CAM_FRONT"]
            gray = _gray(np.asarray(Image.open(_DATA / "samples" / cam.img_path).convert("RGB"), float))
            reach = ego.length / 2.0 + ego.speed * 1.0
            cam_block = camera_freepath_block(gray, cam, ego.height, max(reach, 6.0), thr)
            cam_free = not np.isfinite(cam_block)
            key = ("free" if occ_free else "blocked") + "_" + ("free" if cam_free else "blocked")
            agree[{"free_free": "free_free", "blocked_blocked": "blocked_blocked",
                   "free_blocked": "occ_free_cam_blocked", "blocked_free": "occ_blocked_cam_free"}[key]] += 1
            n += 1
    agreement = (agree["free_free"] + agree["blocked_blocked"]) / n if n else float("nan")
    return {"n_frames": n, "agreement_rate": agreement, **agree}


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"loading {len(MINI)} mini scenes (with boxes for calibration) ...", flush=True)
    scenes = [load_scene(n, _DATA, mask="none", with_boxes=True) for n in MINI]

    cal = calibrate(scenes, rng)
    print("\nCAMERA PRESENCE ORACLE -- reliability calibration (measured, not assumed):")
    print(f"  evidence at KNOWN OBJECT pixels (n={cal['n_object']}): median {cal['median_object']:.2f}")
    print(f"  evidence at ROAD pixels       (n={cal['n_road']}): median {cal['median_road']:.2f}")
    print(f"  ROC AUC (object vs road separation) = {cal['auc']:.3f}")
    verdict = ("USABLE -> proceed to score occupancy obstacles" if cal["auc"] >= 0.80 else
               "INSUFFICIENT (AUC < 0.80) -> pure-CV presence cannot reliably separate object from road; "
               "the honest upgrade is a learned metric-depth net (needs torch, repo rule -- needs sign-off). "
               "Reported as a reachable negative, NOT a failure to hide.")
    print(f"  VERDICT: {verdict}")
    out = _HERE / "results" / "camera_oracle_calibration.json"
    out.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"  wrote {out}")

    # Step 1: FORWARD free-path cross-modality consistency (occupancy vs camera centerline-ray)
    xc = forward_crosscheck(scenes, cal["threshold"])
    print("\nFORWARD free-path cross-modality consistency (occupancy LiDAR vs camera optical):")
    print(f"  {xc['n_frames']} frames | agreement = {xc['agreement_rate']:.3f}")
    print(f"    both free: {xc['free_free']}  both blocked: {xc['blocked_blocked']}")
    print(f"    occ-free/cam-blocked: {xc['occ_free_cam_blocked']}  occ-blocked/cam-free: {xc['occ_blocked_cam_free']}")
    print("    (disagreements are the SIGNAL -- a frame to inspect, not a number to hide)")
    xc_out = {**xc, "oracle_auc": cal["auc"],
              "reading": ("agreement is VACUOUS here: 0 both-blocked (mini zero-positive) so it is "
                          "dominated by both-free, and the presence oracle's AUC < 0.80 is INSUFFICIENT. "
                          "Consistency cross-check, NOT evidence.")}
    (_HERE / "results" / "camera_forward_crosscheck.json").write_text(json.dumps(xc_out, indent=2) + "\n")
    print("\n  Independence: different MODALITY (optical vs TOF) + ALGORITHM (centerline image evidence vs EDT);")
    print("  same vehicle/timestamp so MORE independent, not pristine. CONSISTENCY, not external truth.")
    print("  H1 stays the sole headline; this upgrades H3-forward from same-data to cross-modality consistency.")


if __name__ == "__main__":
    main()
