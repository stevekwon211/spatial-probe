# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Ego-frame -> camera-image projection for the camera oracle (occquery H3, Shot 0).

The camera oracle needs to take a point occquery measured in the ego frame (an obstacle the
occupancy predicate found) and ask the photo "is something really there?". That requires the
world->ego->camera->pixel chain. world->ego is already done in occ3d.py; this module adds the two
missing links -- ego->camera (extrinsic) and camera->pixel (intrinsic K) -- in pure numpy (no torch).

nuScenes calibration (in annotations.json per frame, per camera):
- extrinsic = the camera's POSE IN THE EGO FRAME: translation t (camera origin in ego) + rotation R
  (maps camera axes -> ego axes; verified: R @ [0,0,1] = ego-forward for CAM_FRONT). So to express an
  ego point in the camera frame: p_cam = R^T @ (p_ego - t).
- intrinsics = 3x3 K. nuScenes camera frame is x-right, y-down, z-forward, so a point is visible iff
  z > 0, and its pixel is (u, v) = (fx*x/z + cx, fy*y/z + cy), in-frame iff 0<=u<W and 0<=v<H.

This is Shot 0 of the camera-oracle plan: prove the chain is correct (reprojection self-check on real
calibration) BEFORE building the oracle on top. Run: python experiments/occquery_v0/projection.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMG_W, IMG_H = 1600, 900  # nuScenes camera resolution


def quat_to_rotmat(q) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix. Full 3D (not just yaw), needed because the
    cameras pitch/roll, not only rotate about up."""
    w, x, y, z = np.asarray(q, dtype=float)
    n = np.array([w, x, y, z])
    n = n / np.linalg.norm(n)
    w, x, y, z = n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class Camera:
    """One camera's calibration, parsed from annotations.json camera_sensor[name]."""

    name: str
    K: np.ndarray              # (3,3) intrinsics
    R_cam2ego: np.ndarray      # (3,3) maps camera axes -> ego axes
    t_cam_in_ego: np.ndarray   # (3,) camera origin in the ego frame
    img_path: str = ""

    @staticmethod
    def from_sensor(name: str, info: dict) -> "Camera":
        ext = info["extrinsic"]
        K = np.asarray(info["intrinsics"], dtype=float).reshape(3, 3)
        return Camera(
            name=name,
            K=K,
            R_cam2ego=quat_to_rotmat(ext["rotation"]),
            t_cam_in_ego=np.asarray(ext["translation"], dtype=float),
            img_path=info.get("img_path", ""),
        )


def project_ego_points(points_ego: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ego-frame points (N,3) into camera `cam`.

    Returns (uv, depth, visible): uv is (N,2) pixel coords, depth is (N,) metric depth along the camera
    optical axis (camera-frame z, meters), visible is (N,) bool = in front of the camera AND inside the
    image. Points behind the camera (depth<=0) get uv=nan and visible=False."""
    p = np.atleast_2d(np.asarray(points_ego, dtype=float))
    p_cam = (cam.R_cam2ego.T @ (p - cam.t_cam_in_ego).T).T  # ego -> camera frame
    depth = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (cam.K @ p_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    front = depth > 1e-6
    inframe = front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    uv = np.where(front[:, None], uv, np.nan)
    return uv, depth, inframe


def cameras_for_frame(frame: dict) -> dict[str, Camera]:
    """Parse all cameras from one annotations.json frame's camera_sensor field."""
    return {name: Camera.from_sensor(name, info) for name, info in frame["camera_sensor"].items()}


def _self_check() -> None:
    """Validate the chain on REAL calibration (no JPG needed): a forward-centered point lands near the
    image center of CAM_FRONT; a behind point is rejected; a left point lands left-of-center."""
    import json
    import pathlib

    data = pathlib.Path(__file__).resolve().parents[2] / "data"
    ann = json.loads((data / "annotations.json").read_text())
    si = ann["scene_infos"]["scene-0061"]
    frame = si[next(iter(si))]
    cams = cameras_for_frame(frame)
    front = cams["CAM_FRONT"]
    cx, cy = front.K[0, 2], front.K[1, 2]

    # a point 10 m straight ahead, at camera height -> should land at ~the principal point (image center)
    uv, depth, vis = project_ego_points(np.array([[10.0, 0.0, front.t_cam_in_ego[2]]]), front)
    assert vis[0], f"forward point not visible: uv={uv[0]} depth={depth[0]}"
    assert abs(uv[0, 0] - cx) < 30 and abs(uv[0, 1] - cy) < 30, f"forward point not centered: {uv[0]} vs ({cx:.0f},{cy:.0f})"
    assert abs(depth[0] - (10.0 - front.t_cam_in_ego[0])) < 0.2, f"depth wrong: {depth[0]}"

    # a point 10 m behind -> rejected (depth <= 0)
    _, depth_b, vis_b = project_ego_points(np.array([[-10.0, 0.0, 1.5]]), front)
    assert not vis_b[0] and depth_b[0] < 0, f"behind point not rejected: depth={depth_b[0]}"

    # a point forward-and-left -> lands left of center (u < cx) in CAM_FRONT (image x is right)
    uv_l, _, vis_l = project_ego_points(np.array([[10.0, 3.0, 1.5]]), front)
    assert vis_l[0] and uv_l[0, 0] < cx, f"left point not left-of-center: u={uv_l[0,0]} cx={cx:.0f}"

    # the side cameras should see a point beside the ego (the clearance zone) that CAM_FRONT cannot
    uv_side, _, vis_front_side = project_ego_points(np.array([[1.0, 4.0, 1.0]]), front)
    fl = cams["CAM_FRONT_LEFT"]
    _, _, vis_fl = project_ego_points(np.array([[1.0, 4.0, 1.0]]), fl)
    print(f"  CAM_FRONT principal point ({cx:.0f},{cy:.0f}); forward point -> {np.round(uv[0],0)} depth {depth[0]:.1f}m OK")
    print(f"  side point [fwd1,left4]: CAM_FRONT sees={bool(vis_front_side[0])}  CAM_FRONT_LEFT sees={bool(vis_fl[0])}")
    print("  self-check PASSED: ego->camera->pixel chain correct on real calibration")


if __name__ == "__main__":
    _self_check()
