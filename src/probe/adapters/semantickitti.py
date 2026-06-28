# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""SemanticKITTI Semantic Scene Completion (SSC) -> probe.Scene adapter (3rd dataset).

The SSC voxel release ships, per keyframe, three bit-/byte-packed 256x256x32 grids:
  * `<id>.bin`     -- bit-packed SINGLE-SCAN occupancy (sparse, occluded): the OBSERVED input.
  * `<id>.invalid` -- bit-packed undeterminable mask: voxels the dataset could not label.
  * `<id>.label`   -- uint16 semantic-completion GT: 0 = empty, !=0 = an occupied class (DENSE).

`.bin` (observed) and `.label` (completed) genuinely differ by occlusion/sparsity (the single scan is
essentially a SUBSET of the completion), which is exactly the non-degenerate denotation substrate the
Occ3D adapter could not provide. This adapter maps both into the dataset-agnostic
`probe.grid.OccupancyGrid` (m2-adapter-contract schema) so the sealed predicates run UNCHANGED.

Geometry (verified 2026-06-28 against the raw files): axis0 = forward x in [0, 51.2] m, axis1 = lateral
y in [-25.6, 25.6] m (ego centered), axis2 = up z in [-2.0, 4.4] m; 0.2 m voxels; ego at the grid
origin (x=0, y=0) heading +x. No per-frame ego pose ships with the SSC voxels, so the ego is the origin
with heading 0 (the predicates already reason in this ego frame).
"""
from __future__ import annotations

import pathlib

import numpy as np

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene

# SemanticKITTI SSC voxel-grid spec (verified against the raw .bin/.label byte sizes + occupied cluster).
VOXEL_SIZE = 0.2
GRID_SHAPE = (256, 256, 32)
RANGE = ((0.0, 51.2), (-25.6, 25.6), (-2.0, 4.4))
# world (= ego-frame) coordinate of the center of voxel (0, 0, 0)
ORIGIN = (
    RANGE[0][0] + VOXEL_SIZE / 2.0,   # 0.1
    RANGE[1][0] + VOXEL_SIZE / 2.0,   # -25.5
    RANGE[2][0] + VOXEL_SIZE / 2.0,   # -1.9
)
# road-class (40) median world z ~ -1.5 m (voxel z-idx 2); -1.4 excludes the ground band (z-idx 0-2)
# and keeps obstacle voxels (z-idx 3+), applied IDENTICALLY to observed and dense-GT grids.
GROUND_HEIGHT = -1.4

# standard car envelope (shared with the other adapters; no ego dims ship with the SSC voxels)
EGO_WIDTH, EGO_LENGTH, EGO_HEIGHT = 1.85, 4.6, 1.9

_N_VOXELS = GRID_SHAPE[0] * GRID_SHAPE[1] * GRID_SHAPE[2]
_PACKED_BYTES = _N_VOXELS // 8  # 262144


def _unpack_bits(path: pathlib.Path) -> np.ndarray:
    """Read a bit-packed 256x256x32 grid (.bin / .invalid) -> bool (256,256,32)."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != _PACKED_BYTES:
        raise ValueError(f"{path}: expected {_PACKED_BYTES} bytes, got {raw.size}")
    return np.unpackbits(raw).reshape(GRID_SHAPE).astype(bool)


def _read_label(path: pathlib.Path) -> np.ndarray:
    """Read the uint16 semantic-completion GT -> (256,256,32) uint16 (0 = empty)."""
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != _N_VOXELS:
        raise ValueError(f"{path}: expected {_N_VOXELS} uint16, got {raw.size}")
    return raw.reshape(GRID_SHAPE)


def map_observed(occ_bits: np.ndarray, invalid_bits: np.ndarray) -> np.ndarray:
    """OBSERVED occupancy from the single-scan `.bin` + `.invalid` masks.

    OCCUPIED (1) where the `.bin` bit is set; else UNKNOWN (-1) where the `.invalid` bit is set; else
    FREE (0). OCCUPIED takes precedence over UNKNOWN (a measured return is occupied regardless of the
    invalid flag) -- the single most important validity knob for the unknown-policy sensitivity.
    """
    occ = np.full(GRID_SHAPE, FREE, dtype=int)
    occ[np.asarray(invalid_bits, dtype=bool)] = UNKNOWN
    occ[np.asarray(occ_bits, dtype=bool)] = OCCUPIED
    return occ


def map_dense_gt(label: np.ndarray) -> np.ndarray:
    """DENSE-GT occupancy from `.label`: OCCUPIED (1) where label != 0, else FREE (0)."""
    occ = np.full(GRID_SHAPE, FREE, dtype=int)
    occ[np.asarray(label) != 0] = OCCUPIED
    return occ


def _ego() -> EgoPose:
    return EgoPose(
        position=(0.0, 0.0, 0.0), heading=0.0, speed=0.0,
        width=EGO_WIDTH, length=EGO_LENGTH, height=EGO_HEIGHT,
    )


def load_observed_grid(bin_path: pathlib.Path | str, invalid_path: pathlib.Path | str) -> OccupancyGrid:
    """Load one frame's OBSERVED (single-scan) OccupancyGrid from `.bin` + `.invalid`."""
    occ = map_observed(_unpack_bits(pathlib.Path(bin_path)), _unpack_bits(pathlib.Path(invalid_path)))
    return OccupancyGrid(occ, VOXEL_SIZE, ORIGIN, GROUND_HEIGHT)


def load_dense_gt_grid(label_path: pathlib.Path | str) -> OccupancyGrid:
    """Load one frame's DENSE-GT (completed) OccupancyGrid from `.label`."""
    occ = map_dense_gt(_read_label(pathlib.Path(label_path)))
    return OccupancyGrid(occ, VOXEL_SIZE, ORIGIN, GROUND_HEIGHT)


def _frame_ids(voxel_dir: pathlib.Path) -> list[str]:
    return sorted(p.stem for p in voxel_dir.glob("*.bin"))


def load_scene(
    sequence: str, data_root: pathlib.Path | str, *, variant: str = "observed", frame_ids: list[str] | None = None,
) -> Scene:
    """Load a SemanticKITTI sequence as a probe.Scene (one Frame per keyframe), ego-centric.

    `data_root` must contain `sequences/<sequence>/voxels/`. `variant` is 'observed' (single-scan
    `.bin` occupancy, the OBSERVED view) or 'dense_gt' (the completed `.label`, the REFERENCE).
    `frame_ids` selects/sub-samples frames (default: ALL `.bin` frames, sorted). Object boxes are out
    of scope (the occupancy predicates never use them), so objects=().
    """
    if variant not in ("observed", "dense_gt"):
        raise ValueError(f"variant must be 'observed' or 'dense_gt', got {variant!r}")
    vdir = pathlib.Path(data_root) / "sequences" / sequence / "voxels"
    if not vdir.is_dir():
        raise FileNotFoundError(f"voxel dir not found: {vdir}")
    ids = frame_ids if frame_ids is not None else _frame_ids(vdir)
    frames: list[Frame] = []
    for fid in ids:
        if variant == "observed":
            grid = load_observed_grid(vdir / f"{fid}.bin", vdir / f"{fid}.invalid")
        else:
            grid = load_dense_gt_grid(vdir / f"{fid}.label")
        frames.append(Frame(grid, _ego(), time=float(int(fid)), objects=()))
    return Scene(tuple(frames), f"{sequence}:{variant}")
