# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Occ3D-nuScenes -> probe.Scene adapter (M2).

Loads occupancy scenes from the Occ3D-nuScenes release (annotations.json + gts/*/labels.npz) into
the dataset-agnostic probe.Scene type, so the predicates / retrieval / metrics run on real data
unchanged. The Occ3D voxel grid is EGO-CENTRIC, so the ego is the origin with heading 0 and only
its speed comes from the world ego pose -- the predicates then reason in the same ego frame they
were written for.

Needs no nuScenes-devkit: ego pose, timestamps, and the occupancy path are all in annotations.json.
Object boxes (for the tracking baseline) require nuScenes and are out of scope here -- objects=();
the occupancy predicates do not use them. Grid spec verified 2026-06-20 against labels.npz.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox

# Occ3D-nuScenes voxel-grid spec (verified against labels.npz: semantics (200,200,16) uint8).
VOXEL_SIZE = 0.4
GRID_SHAPE = (200, 200, 16)
RANGE = ((-40.0, 40.0), (-40.0, 40.0), (-1.0, 5.4))
# world (= ego-frame) coordinate of the center of voxel (0, 0, 0)
ORIGIN = (
    RANGE[0][0] + VOXEL_SIZE / 2.0,
    RANGE[1][0] + VOXEL_SIZE / 2.0,
    RANGE[2][0] + VOXEL_SIZE / 2.0,
)
GROUND_HEIGHT = RANGE[2][0]  # -1.0; ground-surface classes are ALSO mapped to FREE below

# Occ3D semantic class ids -> probe encoding.
FREE_CLASS = 17
GROUND_CLASSES = frozenset({11, 12, 13, 14})  # driveable_surface, other_flat, sidewalk, terrain
# every other class (0-10 vehicles/ped/cone/barrier/others, 15 manmade, 16 vegetation) -> OCCUPIED


def map_occupancy(semantics: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Map Occ3D (semantics, optional visibility mask) to a probe OCCUPIED / FREE / UNKNOWN grid.

    free + ground-surface classes -> FREE; every other semantic class -> OCCUPIED.

    `mask=None` returns the DENSE accumulated GT (the H3 denotation oracle): the Occ3D semantics are
    a multi-sweep accumulated label, so they are nearly fully observed. This is what the occupancy
    predicates should run on.

    Passing a per-frame visibility mask marks unobserved voxels (mask == 0) as UNKNOWN -- the
    single-frame OBSERVED view (~88% unknown on one lidar sweep). That view is the conditioning
    variable for gt-distrust / vis-calibration and for a predicted-occupancy robustness arm (v1); it
    is NOT the occquery oracle, because masking the dense GT makes the denotation depend almost
    entirely on the unknown policy.
    """
    occ = np.full(semantics.shape, OCCUPIED, dtype=int)
    occ[semantics == FREE_CLASS] = FREE
    for c in GROUND_CLASSES:
        occ[semantics == c] = FREE
    if mask is not None:
        occ[mask == 0] = UNKNOWN
    return occ


# --- nuScenes object boxes (the tracking baseline) -- ego-frame, to match the ego-centric occupancy ---
# The Occ3D frame token IS the nuScenes sample_token (verified 39/39 on scene-0061), so boxes link by
# that key. nuScenes annotations are GLOBAL; the occupancy is ego-centric with the ego at the origin,
# so a box must be rotated/translated into the ego frame or distance_to_nearest_object (ego.position ->
# box.center) would explode. Requires the nuScenes v1.0-trainval metadata under data/nuscenes/.

_COARSE_CLASS = (  # (category-name prefix, coarse label the queries use); first match wins, order matters
    ("vehicle.bicycle", "bicycle"),
    ("vehicle.motorcycle", "motorcycle"),
    ("human.pedestrian", "pedestrian"),
    ("vehicle.", "vehicle"),  # car / truck / bus / trailer / construction -> vehicle (after bike/moto)
)


def _coarse_label(category_name: str) -> str:
    for prefix, label in _COARSE_CLASS:
        if category_name.startswith(prefix):
            return label
    return "other"


def _quaternion_yaw(q: list[float]) -> float:
    """Heading (rad) from a nuScenes (w, x, y, z) ego->global quaternion (BEV yaw about +z)."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _box_index(nusc_root: pathlib.Path) -> dict[str, list[dict]]:
    """Lazily build + cache {sample_token: [annotation, ...]} joined to a coarse class label.

    Reads the big nuScenes tables ONCE (sample_annotation is ~1.2M rows). Cached on the resolved path
    so repeated load_scene calls in one run pay it a single time."""
    key = str(nusc_root)
    cached = _BOX_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    cat_name = {c["token"]: c["name"] for c in json.loads((nusc_root / "category.json").read_text())}
    inst_cat = {i["token"]: cat_name.get(i["category_token"], "") for i in json.loads((nusc_root / "instance.json").read_text())}
    index: dict[str, list[dict]] = {}
    for a in json.loads((nusc_root / "sample_annotation.json").read_text()):
        a["_label"] = _coarse_label(inst_cat.get(a["instance_token"], ""))
        index.setdefault(a["sample_token"], []).append(a)
    _BOX_INDEX_CACHE[key] = index
    return index


_BOX_INDEX_CACHE: dict[str, dict[str, list[dict]]] = {}


def _ego_frame_boxes(sample_token: str, ego_pose: dict, box_index: dict[str, list[dict]]) -> tuple[TrackedBox, ...]:
    """Transform this sample's global nuScenes boxes into the ego frame (forward, left, up), so they
    sit in the SAME frame as the ego-centric occupancy and the ego at the origin."""
    anns = box_index.get(sample_token, [])
    if not anns:
        return ()
    tx, ty, tz = ego_pose["translation"]
    yaw = _quaternion_yaw(ego_pose["rotation"])
    c, s = math.cos(yaw), math.sin(yaw)
    boxes: list[TrackedBox] = []
    for a in anns:
        gx, gy, gz = a["translation"]
        dx, dy = gx - tx, gy - ty
        fwd = c * dx + s * dy          # ego-frame forward (+x), heading-aligned with the occupancy
        left = -s * dx + c * dy        # ego-frame left (+y)
        w, l, h = a["size"]            # nuScenes size = (width, length, height); TrackedBox wants (length, width, height)
        box_yaw = _quaternion_yaw(a["rotation"]) - yaw
        boxes.append(TrackedBox(center=(fwd, left, gz - tz), size=(l, w, h), yaw=box_yaw, label=a["_label"]))
    return tuple(boxes)


def _ordered_tokens(scene_infos: dict) -> list[str]:
    """Frame tokens in temporal order via the prev/next linked list (fallback: dict order)."""
    head = None
    for tok, fr in scene_infos.items():
        if fr.get("prev") in (None, "", "EOF"):
            head = tok
            break
    if head is None:
        return list(scene_infos)
    order: list[str] = []
    tok: str | None = head
    seen: set[str] = set()
    while tok and tok not in seen and tok in scene_infos:
        order.append(tok)
        seen.add(tok)
        nxt = scene_infos[tok].get("next")
        tok = nxt if nxt not in (None, "", "EOF") else None
    return order or list(scene_infos)


def _speed(scene_infos: dict, tokens: list[str], i: int) -> float:
    """Ego speed (m/s) from consecutive world ego translations and timestamps."""
    if len(tokens) < 2:
        return 0.0
    j = i + 1 if i + 1 < len(tokens) else i - 1
    a, b = scene_infos[tokens[i]], scene_infos[tokens[j]]
    pa = np.asarray(a["ego_pose"]["translation"][:2], dtype=float)
    pb = np.asarray(b["ego_pose"]["translation"][:2], dtype=float)
    dt = abs(b["timestamp"] - a["timestamp"]) / 1e6
    return float(np.linalg.norm(pb - pa) / dt) if dt > 0 else 0.0


def load_scene(
    scene_name: str, data_root: pathlib.Path | str, *, mask: str = "lidar", with_boxes: bool = False
) -> Scene:
    """Load one Occ3D-nuScenes scene as a probe.Scene (ego-centric occupancy, one Frame per sample).

    `data_root` must contain `annotations.json` and `gts/`. `mask` is 'none' (dense accumulated GT,
    the occquery H3 oracle -- recommended), or 'lidar' / 'camera' (mark unobserved voxels UNKNOWN,
    the single-frame observed view used by gt-distrust / vis-calibration). Frames are returned in
    temporal order.

    `with_boxes=False` keeps `objects=()` (the occupancy predicates never use boxes, so the default
    path stays nuScenes-free and fast). `with_boxes=True` loads the nuScenes v1.0-trainval object
    annotations from `data_root/nuscenes/` into `Frame.objects`, transformed into the ego frame -- the
    tracking baseline (`distance_to_nearest_object`) needs them to score the relative gap vs box-only.
    Velocity is left (0,0): nuScenes gives no per-annotation velocity and the baseline queries are
    distance-only. Raises FileNotFoundError if the nuScenes metadata is absent.
    """
    root = pathlib.Path(data_root)
    annotations = json.loads((root / "annotations.json").read_text())
    scene_infos = annotations["scene_infos"]
    if scene_name not in scene_infos:
        raise KeyError(f"{scene_name!r} not in annotations ({len(scene_infos)} scenes available)")
    si = scene_infos[scene_name]
    tokens = _ordered_tokens(si)
    use_mask = mask in ("lidar", "camera")
    mask_key = "mask_lidar" if mask == "lidar" else "mask_camera"
    box_index = _box_index(root / "nuscenes" / "v1.0-trainval") if with_boxes else None
    frames: list[Frame] = []
    for i, tok in enumerate(tokens):
        fr = si[tok]
        labels = np.load(root / fr["gt_path"])
        occupancy = map_occupancy(labels["semantics"], labels[mask_key] if use_mask else None)
        grid = OccupancyGrid(occupancy, VOXEL_SIZE, ORIGIN, GROUND_HEIGHT)
        ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=_speed(si, tokens, i))
        objects = _ego_frame_boxes(tok, fr["ego_pose"], box_index) if box_index is not None else ()
        frames.append(Frame(grid, ego, time=fr["timestamp"] / 1e6, objects=objects))
    return Scene(tuple(frames), scene_name)
