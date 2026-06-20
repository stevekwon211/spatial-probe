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
import pathlib

import numpy as np

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene

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


def load_scene(scene_name: str, data_root: pathlib.Path | str, *, mask: str = "lidar") -> Scene:
    """Load one Occ3D-nuScenes scene as a probe.Scene (ego-centric occupancy, one Frame per sample).

    `data_root` must contain `annotations.json` and `gts/`. `mask` is 'none' (dense accumulated GT,
    the occquery H3 oracle -- recommended), or 'lidar' / 'camera' (mark unobserved voxels UNKNOWN,
    the single-frame observed view used by gt-distrust / vis-calibration). Frames are returned in
    temporal order; objects=() because boxes require nuScenes and the occupancy predicates do not
    use them.
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
    frames: list[Frame] = []
    for i, tok in enumerate(tokens):
        fr = si[tok]
        labels = np.load(root / fr["gt_path"])
        occupancy = map_occupancy(labels["semantics"], labels[mask_key] if use_mask else None)
        grid = OccupancyGrid(occupancy, VOXEL_SIZE, ORIGIN, GROUND_HEIGHT)
        ego = EgoPose((0.0, 0.0, 0.0), 0.0, speed=_speed(si, tokens, i))
        frames.append(Frame(grid, ego, time=fr["timestamp"] / 1e6))
    return Scene(tuple(frames), scene_name)
