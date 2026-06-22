# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Internal-consistency probe for the occquery geometric measurements (occquery H3, DEMOTED arm).

WHAT THIS IS (and is NOT). Per `preregistration.md` (2026-06-21), H1/expressivity is the sole
headline and H3/denotation is DEMOTED to an internal-consistency check until an independent oracle
is earned. An independent oracle needs a DIFFERENT MODALITY (stereo/MVS/ruler) AND a measured
independence test; neither is feasible here (no raw LiDAR sweeps on disk, no second sensor on a Mac).
So this script does NOT compute denotation correctness. It re-derives each predicate quantity by a
second computation over the SAME Occ3D voxel data and reports the residual -- a consistency signal,
labeled as such, never external evidence.

PER-QUANTITY HONESTY (grounded in the predicate code, not asserted):
- lateral_clearance / centerline_lateral_distance are ALREADY exact point computations:
  `np.min(|lateral|)` over `ego.to_ego_frame(obstacle_centers)` (clearance.py:41-45, 68-74). A
  point-to-point reconstruction is the SAME algorithm over the SAME points -> residual is identically
  0 BY CONSTRUCTION. That is a TAUTOLOGY, not cross-validation. We compute it only to DEMONSTRATE the
  circularity empirically (the residual must be ~0), and we report it as a negative, never as agreement
  that validates anything.
- min_free_width_along_path is the ONE quantity with a genuinely different computational family for the
  width: the predicate reads the width off a RASTERISED obstacle row by index arithmetic
  (`(left_li - right_li)*res - res`, freepath.py:94-99); this script reads it off the CONTINUOUS
  obstacle coordinates by nearest-neighbour. We reuse the predicate's OWN reachable field for the walk
  (reachability is shared, so it is NOT being cross-validated -- stated plainly), so the residual
  isolates exactly ONE thing: the lateral RASTERISATION ROUNDING of the corridor-width measurement on a
  0.4 m grid. Same data source; bounds re-voxelisation rounding only, not physical correctness.

Output: residual distributions (median / p95 / max), with the tautological clearance residual reported
alongside as the documented negative. No pass/fail, no movable cutoff. Run: `python consistency.py`.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from probe.grid import FREE, OCCUPIED, EgoPose, OccupancyGrid, UnknownPolicy
from probe.predicates.clearance import lateral_clearance
from probe.predicates.freepath import min_free_width_along_path
from probe.predicates.reachable import reachable_free_field
from probe.scene import Scene

_DATA = _HERE.parents[1] / "data"
_MIN_CLUSTER = 2  # the production noise gate (retrieval.py uses 2 for real data); BOTH the predicate
# and the reconstruction use it, so the residual isolates rounding on the SAME filtered obstacle set.
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]


def reconstruct_lateral_clearance(grid: OccupancyGrid, ego: EgoPose, policy: UnknownPolicy) -> float:
    """Re-derive lateral_clearance by exact point-NN. This is INTENTIONALLY the same computation the
    predicate already does (clearance.py:65-74), so the residual is ~0 by construction -- the probe
    that demonstrates the clearance consistency check is a tautology, reported as a negative."""
    centers = grid.obstacle_centers(unknown_policy=policy, max_height_agl=ego.height)
    if len(centers) == 0:
        return math.inf
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    half = ego.width / 2.0 + grid.voxel_size / 2.0
    beside = (np.abs(forward) <= ego.length / 2.0) & (np.abs(lateral) > half)
    if not beside.any():
        return math.inf
    return float(np.min(np.abs(lateral[beside]) - half))


def reconstruct_min_free_width(
    grid: OccupancyGrid, ego: EgoPose, horizon: float, policy: UnknownPolicy, min_cluster_voxels: int
) -> float:
    """Re-derive the corridor width by CONTINUOUS nearest-neighbour, reusing the predicate's OWN
    reachable field for the station walk (reachability shared, not cross-validated). The only
    difference from the predicate is that the width at each station comes from continuous obstacle
    coordinates, not rasterised row indices -- so the residual isolates lateral rasterisation rounding.

    CRITICAL for apples-to-apples: the continuous points must be the SAME obstacle set the predicate
    uses. The predicate rasterises + drops clusters < `min_cluster_voxels` (freepath -> reachable.py),
    so a continuous point is kept ONLY if its cell survives in the predicate's filtered `f.obstacle`.
    Without this, a lone noise voxel the predicate dropped but the reconstruction kept produces a huge
    spurious residual (the probe's own artifact, not a property of the predicate).

    Mirrors freepath.py:79-104: same forward stations, same frontal-block stop, same surface offset
    (- voxel), same min-over-two-sided-stations, same clamp."""
    f = reachable_free_field(grid, ego, horizon, unknown_policy=policy, min_cluster_voxels=min_cluster_voxels)
    res = f.resolution
    fi0, li0 = f.ego_cell
    reach = ego.length / 2.0 + ego.speed * horizon
    reach_fi = int(round((reach - f.forward_min) / res))
    centers = grid.obstacle_centers(unknown_policy=policy, max_height_agl=ego.height)
    if len(centers) == 0:
        return math.inf
    fwd, lat = ego.to_ego_frame(centers[:, :2])
    fi_pt = np.round((fwd - f.forward_min) / res).astype(int)  # which raster row each point falls in
    li_pt = np.round((lat - f.lateral_min) / res).astype(int)  # and column
    inb = (fi_pt >= 0) & (fi_pt < f.obstacle.shape[0]) & (li_pt >= 0) & (li_pt < f.obstacle.shape[1])
    survives = np.zeros(len(centers), dtype=bool)  # keep only points whose cell survived cluster-drop
    survives[inb] = f.obstacle[fi_pt[inb], li_pt[inb]]
    fwd, lat, fi_pt = fwd[survives], lat[survives], fi_pt[survives]
    widths: list[float] = []
    for fi in range(fi0, min(reach_fi, f.obstacle.shape[0] - 1) + 1):
        if f.obstacle[fi, li0]:
            break  # raw obstacle on the centerline -> frontal blockage, same stop as the predicate
        here = fi_pt == fi
        if not here.any():
            continue
        lat_here = lat[here]
        left = lat_here[lat_here > 0.0]   # left of centerline (lateral > 0)
        right = lat_here[lat_here < 0.0]  # right of centerline (lateral < 0)
        if left.size and right.size:
            width = float(np.min(left) - np.max(right)) - res  # surface-to-surface, continuous
            widths.append(max(0.0, width))
    if not widths:
        return math.inf
    narrowest = min(widths)
    return 0.0 if narrowest < res else float(narrowest)


def _residual(pred: float, recon: float) -> float | None:
    """Signed (recon - pred) where BOTH are finite & positive; None otherwise (an inf/0 disagreement
    is a finiteness mismatch counted separately, never folded into the residual distribution)."""
    if math.isfinite(pred) and math.isfinite(recon) and pred > 0.0 and recon > 0.0:
        return recon - pred
    return None


def _summarize(name: str, residuals: list[float], finite_mismatch: int, n_frames: int) -> dict:
    arr = np.array(residuals, dtype=float)
    return {
        "quantity": name,
        "n_frames": n_frames,
        "n_residual_pairs": len(residuals),
        "n_finiteness_mismatch": finite_mismatch,
        "median_abs_m": float(np.median(np.abs(arr))) if arr.size else math.nan,
        "p95_abs_m": float(np.percentile(np.abs(arr), 95)) if arr.size else math.nan,
        "max_abs_m": float(np.max(np.abs(arr))) if arr.size else math.nan,
        "signed_mean_m": float(np.mean(arr)) if arr.size else math.nan,
    }


def _self_check() -> None:
    """Synthetic ground-truth: the reconstruction must MATCH the predicate on a clean two-sided
    corridor (residual ~0 within rounding) and the clearance reconstruction must be EXACTLY the
    predicate (tautology). Runs on every invocation; fails loudly if the reconstruction drifts."""
    voxel, n, ground = 0.2, 120, 0.5
    occ = np.full((n, n, n), FREE, dtype=int)
    ego = EgoPose((10.0, 10.0, 0.0), 0.0, speed=10.0)
    # clean corridor: walls at +/-1.0 m on clean voxel centers, forward 0..~11 m
    for i in range(0, 60):
        x = 10.0 + 0.2 * i
        occ[round(x / voxel), round(11.0 / voxel), round(1.0 / voxel)] = OCCUPIED
        occ[round(x / voxel), round(9.0 / voxel), round(1.0 / voxel)] = OCCUPIED
    grid = OccupancyGrid(occ, voxel, (0.0, 0.0, 0.0), ground)
    pred_w = min_free_width_along_path(grid, ego, 2.0, min_cluster_voxels=1)
    recon_w = reconstruct_min_free_width(grid, ego, 2.0, UnknownPolicy.FREE, min_cluster_voxels=1)
    assert math.isfinite(pred_w) and math.isfinite(recon_w), (pred_w, recon_w)
    assert abs(recon_w - pred_w) <= voxel + 1e-9, f"corridor reconstruction drifted: {recon_w} vs {pred_w}"
    # clearance tautology: identical computation -> residual exactly 0
    occ2 = np.full((n, n, n), FREE, dtype=int)
    occ2[round(11.5 / voxel), round(10.6 / voxel), round(1.0 / voxel)] = OCCUPIED  # an abeam obstacle
    g2 = OccupancyGrid(occ2, voxel, (0.0, 0.0, 0.0), ground)
    p = lateral_clearance(g2, ego)
    r = reconstruct_lateral_clearance(g2, ego, UnknownPolicy.FREE)
    assert (math.isinf(p) and math.isinf(r)) or abs(p - r) < 1e-12, f"clearance not a tautology: {p} vs {r}"


def main() -> None:
    from probe.adapters.occ3d import load_scene  # local import: needs data/, unlike the self-check

    _self_check()
    print("self-check passed (corridor reconstruction matches predicate; clearance residual == 0)\n")
    print(f"loading {len(MINI)} Occ3D-nuScenes mini scenes (mask=lidar) ...")
    scenes: list[Scene] = [load_scene(name, _DATA, mask="lidar") for name in MINI]
    policy = UnknownPolicy.FREE

    corridor_res: list[float] = []
    corridor_mismatch = 0
    clear_res: list[float] = []
    clear_mismatch = 0
    n_frames = 0
    for sc in scenes:
        for t in range(len(sc)):
            n_frames += 1
            grid, ego = sc.grid_at(t), sc.ego_at(t)
            pw = min_free_width_along_path(grid, ego, 2.0, min_cluster_voxels=_MIN_CLUSTER)
            rw = reconstruct_min_free_width(grid, ego, 2.0, policy, _MIN_CLUSTER)
            r = _residual(pw, rw)
            if r is None:
                if math.isfinite(pw) != math.isfinite(rw):
                    corridor_mismatch += 1
            else:
                corridor_res.append(r)
            pc = lateral_clearance(grid, ego)
            rc = reconstruct_lateral_clearance(grid, ego, policy)
            rcr = _residual(pc, rc)
            if rcr is None:
                if math.isfinite(pc) != math.isfinite(rc):
                    clear_mismatch += 1
            else:
                clear_res.append(rcr)

    corridor = _summarize("min_free_width (genuine: raster-vs-continuous rounding)", corridor_res, corridor_mismatch, n_frames)
    clear = _summarize("lateral_clearance (TAUTOLOGY: same algorithm)", clear_res, clear_mismatch, n_frames)

    print(f"  {len(scenes)} scenes, {n_frames} frames\n")
    print("INTERNAL-CONSISTENCY residuals (same Occ3D data source -- NOT denotation correctness):\n")
    for s in (corridor, clear):
        print(f"  {s['quantity']}")
        print(f"    pairs={s['n_residual_pairs']}  finiteness-mismatch={s['n_finiteness_mismatch']}")
        print(f"    |residual| median={s['median_abs_m']:.4f} m  p95={s['p95_abs_m']:.4f} m  max={s['max_abs_m']:.4f} m")
        print(f"    signed mean={s['signed_mean_m']:+.4f} m\n")
    print("READING: the clearance residual is ~0 BY CONSTRUCTION (same point-NN algorithm) -- a")
    print("tautology, reported as a negative, NOT evidence. The min_free_width residual bounds lateral")
    print("rasterisation rounding only (~half-voxel on a 0.4 m grid), same data source. Denotation")
    print("correctness needs val data with positives + an independent-modality oracle: neither exists yet.")


if __name__ == "__main__":
    main()
