# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Export a SEALED labeling pool to the web, structurally BLIND to the predicate's verdict.

The web labeler must judge the raw occupancy geometry WITHOUT seeing whether the predicate fired or
what value it computed (else confirmation bias). So this split:

  occquery/label/pool.json          -- the queries (nl + rationale + scope) + the SHUFFLED task list
  occquery/label/<scene>.json       -- scene meta: ego + per-frame speed/obstacle-count. NO predicates.
  occquery/label/<scene>/f<t>.json  -- just that frame's obstacle voxels (the geometry to judge). BLIND.
  occquery/label/answers/<scene>.json -- the predicate verdict + reachable field, REVEAL-ONLY. The
                                         labeler fetches this ONLY AFTER locking a verdict (QA spot-check),
                                         so the verdict is not even in the browser during judgment.

This keeps the bias-blind invariant a property of WHAT IS ON DISK for the blind path, not a UI promise.
Run AFTER build_pool.py has sealed the pool. Usage: python experiments/occquery_v0/export_label.py --pool pilot
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

from export_web import _band_obstacles, _finite  # noqa: E402  (reuse the exact obstacle extraction)
from probe.adapters.occ3d import _ordered_tokens, load_scene  # noqa: E402
from probe.grid import UnknownPolicy  # noqa: E402
from probe.predicates.clearance import lateral_clearance  # noqa: E402
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path  # noqa: E402
from probe.predicates.reachable import reachable_free_field  # noqa: E402
from probe.query_spec import load_queries  # noqa: E402
from probe.retrieval import frame_true  # noqa: E402

_DATA = _HERE.parents[1] / "data"
_OUT = _HERE.parents[1] / "web" / "public" / "data" / "occquery" / "label"


def _scene_meta_and_frames(name: str, out_scene_dir: pathlib.Path, si: dict, tokens: list[str], scene) -> dict:
    """Write the BLIND per-frame obstacle files; return the scene meta (NO predicate values)."""
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, t in enumerate(scene.times()):
        semantics = np.load(_DATA / si[tokens[i]]["gt_path"])["semantics"]
        _, ego, obstacles, n_band, _ = _band_obstacles(scene, t, semantics)
        (out_scene_dir / f"f{t}.json").write_text(json.dumps({"t": t, "obstacles": obstacles}))
        metas.append({"t": t, "speed": round(float(ego.speed), 2), "n_obstacles_band": n_band})
    e0, g0 = scene.ego_at(0), scene.grid_at(0)
    return {
        "scene": name,
        "voxel_size": g0.voxel_size,
        "ground_height": g0.ground_height,
        "ego": {"width": round(e0.width, 2), "length": round(e0.length, 2), "height": round(e0.height, 2)},
        "n_frames": len(metas),
        "frames": metas,  # speed + obstacle counts only -- NO min_free_width / clearance / verdict
    }


def _answers(name: str, scene, pool_queries: dict) -> dict:
    """The REVEAL-ONLY data (fetched after a verdict locks): per-frame reachable field + per-query
    predicate verdict (retrieved? which frames matched?). NEVER loaded during blind judgment."""
    frames = []
    for t in scene.times():
        grid, ego = scene.grid_at(t), scene.ego_at(t)
        rf = reachable_free_field(grid, ego, 2.0, min_cluster_voxels=2)
        frames.append({
            "t": t,
            "reachable": {
                "forward_min": round(float(rf.forward_min), 3),
                "lateral_min": round(float(rf.lateral_min), 3),
                "resolution": round(float(rf.resolution), 3),
                "ego_cell": [int(rf.ego_cell[0]), int(rf.ego_cell[1])],
                "shape": [int(rf.reachable.shape[0]), int(rf.reachable.shape[1])],
                "mask": rf.reachable.astype(np.uint8).flatten().tolist(),
            },
            "predicates": {
                "min_free_width": _finite(min_free_width_along_path(grid, ego, 2.0, min_cluster_voxels=2)),
                "lateral_clearance": _finite(lateral_clearance(grid, ego)),
                "free_path_blocked": not free_along_ego_path(grid, ego, 1.0, min_cluster_voxels=2),
            },
        })
    verdicts = {}
    for qid, q in pool_queries.items():
        matching = [int(t) for t in scene.times() if frame_true(scene, t, q.predicate, UnknownPolicy.FREE)]
        verdicts[qid] = {"retrieved": len(matching) > 0, "matching_frames": matching}
    return {"scene": name, "frames": frames, "query_verdicts": verdicts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, default="pilot")
    args = ap.parse_args()

    pool_meta = json.loads((_HERE / f"pool-{args.pool}.json").read_text())
    query_ids = pool_meta["query_ids"]
    all_q = {q.id: q for q in load_queries(_HERE / "queries.yaml")}
    pool_queries = {qid: all_q[qid] for qid in query_ids}
    scene_ids = sorted({task["scene_id"] for task in pool_meta["tasks"]})

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "answers").mkdir(exist_ok=True)
    annotations = json.loads((_DATA / "annotations.json").read_text())

    for name in scene_ids:
        scene = load_scene(name, _DATA, mask="none")  # dense GT (scoring policy = FREE)
        si = annotations["scene_infos"][name]
        tokens = _ordered_tokens(si)
        meta = _scene_meta_and_frames(name, _OUT / name, si, tokens, scene)
        (_OUT / f"{name}.json").write_text(json.dumps(meta))
        (_OUT / "answers" / f"{name}.json").write_text(json.dumps(_answers(name, scene, pool_queries)))
        print(f"  {name}: {meta['n_frames']} frames (blind) + answers", flush=True)

    # pool.json for the web: queries (nl + rationale + scope, NO predicate string shown) + shuffled tasks
    (_OUT / "pool.json").write_text(json.dumps({
        "pool_id": pool_meta["pool_id"],
        "sealed_at": pool_meta["sealed_at"],
        "is_pilot": pool_meta["is_pilot"],
        "scoring_policy": pool_meta["scoring_policy"],
        "honest_scope": pool_meta["honest_scope"],
        "queries": [{"id": qid, "nl": pool_queries[qid].nl, "rationale": pool_queries[qid].rationale,
                     "scope": pool_queries[qid].scope} for qid in query_ids],
        "tasks": pool_meta["tasks"],
    }, indent=2))
    print(f"\nwrote BLIND labeling data to {_OUT}")
    print(f"  pool.json: {len(query_ids)} queries, {len(pool_meta['tasks'])} tasks, {len(scene_ids)} scenes")
    print("  answers/ is reveal-only (fetched AFTER a verdict locks). Blind path carries no verdict.")


if __name__ == "__main__":
    main()
