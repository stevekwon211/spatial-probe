# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""spatial-probe MCP server — the research instrument as LLM-callable tools.

The program's thesis is "3D is queryable, updatable STATE, not the render." This server makes that
literal: an agent can call a falsifiable physical predicate on a real Occ3D-nuScenes scene and get the
MEASUREMENT back, list scenes, and read the committed findings -- without writing a script each time.

Tools:
  list_scenes()                                  -> available scene names (needs local Occ3D data)
  scene_info(scene)                              -> n_frames + per-frame ego speed
  list_predicates()                              -> the callable predicates + what they measure
  probe_scene(scene, frame, predicate, ...)      -> run one predicate on one frame, return the value
  get_findings(experiment)                       -> the honest results summary (works with no data)

Run (stdio, for an MCP client like Claude Code):  python mcp_server/server.py
Register: see mcp_server/README.md (a project-scoped .mcp.json is provided).
"""
from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data"
sys.path.insert(0, str(_ROOT / "src"))

from mcp.server.fastmcp import FastMCP

from probe.grid import UnknownPolicy
from probe.predicates import (centerline_lateral_distance, distance_to_nearest_object,
                              free_along_ego_path, lateral_clearance, min_free_width_along_path)

mcp = FastMCP("spatial-probe")

# predicate name -> (callable-kind, one-line meaning). kind drives the dispatch in probe_scene.
_PREDICATES = {
    "lateral_clearance": "free-space distance to the nearest obstacle beside the ego centerline (m)",
    "min_free_width": "narrowest free corridor width along the ego path over the horizon (m)",
    "free_along_path": "is the ego forward path clear over the horizon (bool)",
    "centerline_lateral": "lateral distance from centerline to nearest obstacle over a lookahead (m)",
    "box_distance": "BOX-ONLY baseline: distance to the nearest tracked-object box (m) -- box-blind to "
                    "structure with no box, for contrast with the occupancy predicates",
}
_POLICY = {"free": UnknownPolicy.FREE, "occupied": UnknownPolicy.OCCUPIED, "ignored": UnknownPolicy.IGNORED}


def _scene_names() -> list[str]:
    import json
    ann = _DATA / "annotations.json"
    if ann.exists():
        return sorted(json.loads(ann.read_text())["scene_infos"].keys())
    gts = _DATA / "gts"
    return sorted(p.name for p in gts.glob("scene-*")) if gts.exists() else []


@mcp.tool()
def list_scenes() -> dict:
    """List the Occ3D-nuScenes scenes available locally (needs the gated dataset on disk)."""
    names = _scene_names()
    if not names:
        return {"count": 0, "scenes": [], "note": "no local Occ3D data (data/ is gitignored, account-gated)"}
    return {"count": len(names), "scenes": names[:200], "truncated": len(names) > 200}


@mcp.tool()
def list_predicates() -> dict:
    """The falsifiable physical predicates this server can run, and what each measures."""
    return {"predicates": _PREDICATES,
            "unknown_policy": list(_POLICY),
            "note": "occupancy predicates run on the dense Occ3D GT; box_distance is the box-only baseline"}


@mcp.tool()
def scene_info(scene: str) -> dict:
    """Frame count and per-frame ego speed (m/s) for a scene."""
    from probe.adapters.occ3d import load_scene
    try:
        sc = load_scene(scene, _DATA, mask="none")
    except (KeyError, FileNotFoundError) as e:
        return {"error": f"{type(e).__name__}: {e}", "hint": "call list_scenes(); ensure local Occ3D data"}
    speeds = [round(fr.ego.speed, 2) for fr in sc.frames]
    return {"scene": scene, "n_frames": len(sc.frames), "ego_speed_mps": speeds,
            "ego_speed_range": [min(speeds), max(speeds)] if speeds else None}


@mcp.tool()
def probe_scene(scene: str, frame: int, predicate: str, horizon: float = 3.0,
                unknown_policy: str = "free") -> dict:
    """Run one occupancy/box predicate on one frame of one scene and return the measurement.

    scene: e.g. 'scene-0061' (see list_scenes). frame: 0-based index (see scene_info).
    predicate: one of list_predicates(). horizon: forward look-ahead seconds (path predicates).
    unknown_policy: how UNOBSERVED voxels are treated -- 'free' | 'occupied' | 'ignored'.
    """
    if predicate not in _PREDICATES:
        return {"error": f"unknown predicate {predicate!r}", "available": list(_PREDICATES)}
    if unknown_policy not in _POLICY:
        return {"error": f"unknown policy {unknown_policy!r}", "available": list(_POLICY)}
    from probe.adapters.occ3d import load_scene
    up = _POLICY[unknown_policy]
    try:
        sc = load_scene(scene, _DATA, mask="none", with_boxes=(predicate == "box_distance"))
    except (KeyError, FileNotFoundError) as e:
        return {"error": f"{type(e).__name__}: {e}", "hint": "call list_scenes(); ensure local Occ3D data"}
    if not (0 <= frame < len(sc.frames)):
        return {"error": f"frame {frame} out of range", "n_frames": len(sc.frames)}
    fr = sc.frames[frame]
    g, ego = fr.grid, fr.ego
    if predicate == "lateral_clearance":
        value, unit = lateral_clearance(g, ego, unknown_policy=up), "m"
    elif predicate == "min_free_width":
        value, unit = min_free_width_along_path(g, ego, horizon, unknown_policy=up), "m"
    elif predicate == "free_along_path":
        value, unit = free_along_ego_path(g, ego, horizon, unknown_policy=up), "bool"
    elif predicate == "centerline_lateral":
        value, unit = centerline_lateral_distance(g, ego, unknown_policy=up), "m"
    else:  # box_distance
        value, unit = distance_to_nearest_object(sc, frame), "m"
    return {"scene": scene, "frame": frame, "predicate": predicate, "value": value, "unit": unit,
            "meaning": _PREDICATES[predicate], "horizon_s": horizon, "unknown_policy": unknown_policy,
            "ego_speed_mps": round(ego.speed, 2)}


@mcp.tool()
def get_findings(experiment: str = "occquery_v0") -> dict:
    """The committed, honest results summary for an experiment (works with no dataset on disk).

    experiment: a dir under experiments/ (e.g. 'occquery_v0', 'dynfield_v0')."""
    p = _ROOT / "experiments" / experiment / "results" / "summary.md"
    if not p.exists():
        avail = sorted(d.name for d in (_ROOT / "experiments").glob("*") if (d / "results" / "summary.md").exists())
        return {"error": f"no summary for {experiment!r}", "available": avail}
    return {"experiment": experiment, "summary_md": p.read_text()}


if __name__ == "__main__":
    mcp.run()
