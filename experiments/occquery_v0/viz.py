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


def _mark_measurement(ax, grid, ego, query: str | None) -> None:
    """Overlay WHERE the predicate took its measurement, so a human can verify the match."""
    centers = grid.obstacle_centers(max_height_agl=ego.height)
    if not len(centers):
        return
    fwd, lat = ego.to_ego_frame(centers[:, :2])
    if query == "corridor":
        reach = ego.length / 2.0 + ego.speed * 2.0
        hv = grid.voxel_size / 2.0
        best = (float("inf"), None)
        k = 0
        while k * grid.voxel_size <= reach + 1e-9:
            sft = k * grid.voxel_size
            k += 1
            near = np.abs(fwd - sft) <= hv
            if not near.any():
                continue
            side = lat[near]
            lft = side[side > 0]
            rgt = side[side < 0]
            if lft.size and rgt.size:
                top = float(lft.min()) - hv
                bot = float(rgt.max()) + hv
                w = max(0.0, top) + max(0.0, -bot)
                if w < best[0]:
                    best = (w, (sft, top, bot))
        if best[1] is not None:
            sft, top, bot = best[1]
            ax.plot([sft, sft], [bot, top], color="#2563eb", lw=3, solid_capstyle="round", zorder=5)
            ax.annotate(f"{best[0]:.2f} m gap here", (sft + 0.6, (top + bot) / 2),
                        color="#2563eb", fontsize=9, va="center", zorder=6)
    elif query == "tight":
        half = ego.width / 2.0 + grid.voxel_size / 2.0
        beside = (fwd >= 0.0) & (fwd <= 20.0) & (np.abs(lat) > half)
        if not beside.any():
            return
        i = int(np.argmin(np.abs(lat[beside]) - half))
        bx, by = float(fwd[beside][i]), float(lat[beside][i])
        ax.plot([bx], [by], "x", color="#2563eb", markersize=13, markeredgewidth=3, zorder=6)
        ax.annotate("nearest side obstacle", (bx + 0.6, by), color="#2563eb", fontsize=9, va="center", zorder=6)


def render(scene, t, out_path: pathlib.Path, query: str | None = None) -> None:
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
    _mark_measurement(ax, grid, ego, query)
    fig.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)


def _pick_frame(scene, query: str) -> int:
    """The most query-relevant frame to render for hand-labeling."""
    ts = scene.times()
    if query == "corridor":  # narrowest frame that actually MATCHES 0 < width < ego_width
        ego_w = scene.ego_width()
        cand = [(min_free_width_along_path(scene.grid_at(t), scene.ego_at(t), 2.0), t) for t in ts]
        matching = [(w, t) for w, t in cand if 0.0 < w < ego_w]
        return min(matching)[1] if matching else min(cand)[1]
    if query == "blocked":  # the first frame the swept path is blocked (start of a transition)
        for t in ts:
            if not free_along_ego_path(scene.grid_at(t), scene.ego_at(t), 1.0):
                return t
        return 0
    if query == "tight":  # smallest side clearance among above-threshold-speed frames
        fast = [(lateral_clearance(scene.grid_at(t), scene.ego_at(t)), t) for t in ts if scene.ego_speed(t) > 8.33]
        pool = fast or [(lateral_clearance(scene.grid_at(t), scene.ego_at(t)), t) for t in ts]
        return min(pool)[1]
    return min((lateral_clearance(scene.grid_at(t), scene.ego_at(t)), t) for t in ts)[1]


def main(query: str = "corridor") -> None:
    out = _HERE / "results" / "viz" / query
    out.mkdir(parents=True, exist_ok=True)
    for name in MINI:
        scene = load_scene(name, _DATA, mask="none")
        t = _pick_frame(scene, query)
        render(scene, t, out / f"{name}.png", query=query)
        g, ego = scene.grid_at(t), scene.ego_at(t)
        w = min_free_width_along_path(g, ego, 2.0)
        cl = lateral_clearance(g, ego)
        ws = "inf" if w == float("inf") else f"{w:.2f}m"
        cls = "inf" if cl == float("inf") else f"{cl:.2f}m"
        print(f"{name}: frame {t}/{len(scene) - 1} | min_free_width {ws} | side_clearance {cls} | speed {ego.speed:.1f} m/s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corridor")
