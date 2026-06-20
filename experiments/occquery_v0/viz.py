# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Top-down occupancy renderer for hand-labeling occquery H3 (denotation correctness).

Renders one frame as an ego-centric top-down occupancy map so a human can judge a query's
retrieval VISUALLY -- "is something right next to the ego / is the corridor visibly narrow / is the
path blocked" -- independently of the predicate's exact formula. That visual call is the v0 GT;
comparing it to the predicate retrieval gives denotation P/R/F1. (A fully independent raw-LiDAR
oracle is a v1 step.)

Usage: python experiments/occquery_v0/viz.py   # renders the min-clearance frame of each mini scene
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe.adapters.occ3d import load_scene  # noqa: E402
from probe.grid import OCCUPIED  # noqa: E402
from probe.predicates.clearance import lateral_clearance  # noqa: E402
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path  # noqa: E402

_DATA = _HERE.parents[1] / "data"
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]


def top_down_obstacles(grid, ego) -> np.ndarray:
    """(X, Y) boolean: an obstacle voxel exists in the ego vertical band at that column."""
    occ = grid.occupancy
    zc = grid.origin[2] + np.arange(occ.shape[2]) * grid.voxel_size
    band = (zc > grid.ground_height) & (zc <= grid.ground_height + ego.height)
    return (occ[:, :, band] == OCCUPIED).any(axis=2)


def render(scene, t, out_path: pathlib.Path) -> None:
    grid = scene.grid_at(t)
    ego = scene.ego_at(t)
    obstacle = top_down_obstacles(grid, ego)  # [i=forward(x), j=lateral(y)]
    nx, ny = obstacle.shape
    ox, oy, _ = grid.origin
    extent = [ox, ox + nx * grid.voxel_size, oy, oy + ny * grid.voxel_size]  # x: forward, y: lateral

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # imshow wants [row=y, col=x]; transpose so forward is the horizontal axis, lateral vertical
    ax.imshow(obstacle.T, origin="lower", extent=extent, cmap="Greys", interpolation="nearest", aspect="equal")
    ax.plot(0, 0, "^", color="red", markersize=13, label="ego")
    ax.arrow(0, 0, 6, 0, color="red", width=0.15, head_width=1.2, length_includes_head=True)

    clearance = lateral_clearance(grid, ego)
    width = min_free_width_along_path(grid, ego, 2.0)
    blocked = not free_along_ego_path(grid, ego, 0.0)
    cl_s = "inf" if clearance == float("inf") else f"{clearance:.2f}m"
    fw_s = "inf" if width == float("inf") else f"{width:.2f}m"
    ax.set_title(
        f"{scene.name}  frame {t}/{len(scene) - 1}\n"
        f"speed {ego.speed:.1f} m/s  |  clearance {cl_s}  |  min_free_width {fw_s}  |  blocked_now {blocked}",
        fontsize=10,
    )
    ax.set_xlabel("forward (m)  ->  driving direction")
    ax.set_ylabel("lateral (m)  +left")
    ax.set_xlim(-8, 40)
    ax.set_ylim(-20, 20)
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = _HERE / "results" / "viz"
    out.mkdir(parents=True, exist_ok=True)
    for name in MINI:
        scene = load_scene(name, _DATA, mask="none")
        best_t, best_cl = 0, float("inf")
        for t in scene.times():
            cl = lateral_clearance(scene.grid_at(t), scene.ego_at(t))
            if cl < best_cl:
                best_cl, best_t = cl, t
        render(scene, best_t, out / f"{name}.png")
        cl_s = "inf" if best_cl == float("inf") else f"{best_cl:.2f}m"
        print(f"{name}: min-clearance frame {best_t}, clearance {cl_s}, speed {scene.ego_speed(best_t):.1f} m/s")


if __name__ == "__main__":
    main()
