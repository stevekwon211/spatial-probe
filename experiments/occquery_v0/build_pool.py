# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Build + SEAL a labeling pool for occquery H3 ground-truth (the pre-registration step).

The honest answer-key for H3 needs a scene pool committed BEFORE the human judges it, and that pool
must let RECALL be measured -- so it includes scenes the predicate did NOT retrieve (sampled
negatives), not only its hits. This script:

  1. scans a candidate scene set (Occ3D-nuScenes, excluding the v0 mini), running each pool query
     under unknown=FREE to find positives (scenes the predicate retrieves);
  2. for each query, takes up to N_POS positives + a random N_NEG sample of negatives;
  3. SHUFFLES the (query, scene) tasks with a recorded seed so the labeler cannot infer which scenes
     are predicted-positive (blind-to-verdict at the task level too);
  4. writes the sealed pool: `held-out-<pool>.txt` (scene ids) + `pool-<pool>.json` (tasks + meta +
     a sha256 of queries.yaml, so a post-hoc query/threshold change shows in the hash).

PILOT vs FULL: `--pilot` builds a small system-validation pool (3 queries) over a candidate sample --
NOT the sealed scientific run (it selects scenes by predicate output, which is a forking path,
acceptable ONLY because the pilot validates the labeling FLOW, never a result). The FULL run uses the
official nuScenes val split (all scenes, no selection) -- pass `--scenes val_scenes.txt`.

Usage:
  python experiments/occquery_v0/build_pool.py --pilot --sealed-at 2026-06-22T11:00:00Z
  python experiments/occquery_v0/build_pool.py --scenes val_scenes.txt --pool val --sealed-at <iso>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from probe.adapters.occ3d import load_scene
from probe.grid import UnknownPolicy
from probe.query_spec import load_queries
from probe.retrieval import retrieved

_DATA = _HERE.parents[1] / "data"
_QUERIES = _HERE / "queries.yaml"
MINI = frozenset({
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
})
PILOT_QUERIES = ["tight_clearance_at_speed", "free_path_is_blocked", "corridor_narrows_below_vehicle_width"]
_SEED = 0  # recorded; the task shuffle is deterministic given (pool, seed)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_scenes(limit: int) -> list[str]:
    """A deterministic sample of non-mini Occ3D scenes (sorted by name, first `limit`)."""
    annotations = json.loads((_DATA / "annotations.json").read_text())
    names = sorted(n for n in annotations["scene_infos"] if n not in MINI)
    return names[:limit]


def _scan(scene_names: list[str], query_ids: list[str]) -> dict[str, set[str]]:
    """Return {query_id: set(scenes that match under FREE)} by running retrieval on each scene.

    One scene at a time (load -> retrieve -> drop) so memory stays flat over hundreds of scenes."""
    queries = {q.id: q for q in load_queries(_QUERIES) if q.id in query_ids}
    positives: dict[str, set[str]] = {qid: set() for qid in query_ids}
    for i, name in enumerate(scene_names):
        scene = load_scene(name, _DATA, mask="none")  # dense GT (the H3 oracle view)
        for qid in query_ids:
            if retrieved([scene], queries[qid], UnknownPolicy.FREE):
                positives[qid].add(name)
        if (i + 1) % 10 == 0:
            print(f"  scanned {i + 1}/{len(scene_names)} ...", flush=True)
    return positives


def build(pool: str, scene_names: list[str], query_ids: list[str], sealed_at: str,
          n_pos: int, n_neg: int) -> None:
    rng = np.random.default_rng(_SEED)
    print(f"scanning {len(scene_names)} candidate scenes for {len(query_ids)} queries ...", flush=True)
    positives = _scan(scene_names, query_ids)

    tasks: list[dict] = []
    pool_scenes: set[str] = set()
    for qid in query_ids:
        pos = sorted(positives[qid])
        neg = sorted(set(scene_names) - positives[qid])
        chosen_pos = pos[:n_pos]
        chosen_neg = [neg[k] for k in rng.choice(len(neg), size=min(n_neg, len(neg)), replace=False)] if neg else []
        for sid in chosen_pos + chosen_neg:
            tasks.append({"query_id": qid, "scene_id": sid})
            pool_scenes.add(sid)
        print(f"  {qid}: {len(chosen_pos)} positives + {len(chosen_neg)} sampled negatives", flush=True)

    # shuffle so the labeler cannot infer positive-vs-negative from task order (blind at task level)
    order = rng.permutation(len(tasks))
    tasks = [{**tasks[k], "task_id": j} for j, k in enumerate(order)]

    out_dir = _HERE
    held = out_dir / f"held-out-{pool}.txt"
    held.write_text("\n".join(sorted(pool_scenes)) + "\n")
    meta = {
        "pool_id": pool,
        "sealed_at": sealed_at,                       # caller-supplied ISO timestamp (git commit = the real seal)
        "is_pilot": pool == "pilot",
        "queries_sha256": _sha256(_QUERIES),          # a post-hoc query/threshold edit changes this
        "occ3d_note": "Occ3D-nuScenes dense GT (mask=none); unknown~0. Scoring policy = FREE.",
        "scoring_policy": "free",
        "query_ids": query_ids,
        "n_scenes": len(pool_scenes),
        "n_tasks": len(tasks),
        "shuffle_seed": _SEED,
        "tasks": tasks,
        "honest_scope": ("Human-vs-code consistency on the SAME Occ3D data (different algorithm, same "
                         "data source) -- NOT a different-modality external oracle. H1 stays the sole headline."),
    }
    (out_dir / f"pool-{pool}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nSEALED pool '{pool}': {len(pool_scenes)} scenes, {len(tasks)} tasks")
    print(f"  {held}")
    print(f"  {out_dir / f'pool-{pool}.json'}")
    print("  COMMIT these to git BEFORE labeling -- the commit timestamp is the pre-registration seal.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="small system-validation pool (3 queries, NOT a result)")
    ap.add_argument("--scenes", type=str, default=None, help="a file of scene ids (one per line); else a candidate sample")
    ap.add_argument("--pool", type=str, default="pilot")
    ap.add_argument("--sealed-at", type=str, required=True, help="ISO-8601 timestamp (git commit is the real seal)")
    ap.add_argument("--candidates", type=int, default=40, help="how many non-mini scenes to scan (pilot)")
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--n-neg", type=int, default=4)
    args = ap.parse_args()

    if args.scenes:
        scene_names = [s.strip() for s in pathlib.Path(args.scenes).read_text().splitlines() if s.strip()]
        query_ids = [q.id for q in load_queries(_QUERIES) if q.is_occupancy]
        pool = args.pool
    else:  # pilot
        scene_names = _candidate_scenes(args.candidates)
        query_ids = PILOT_QUERIES
        pool = "pilot"
    build(pool, scene_names, query_ids, args.sealed_at, args.n_pos, args.n_neg)


if __name__ == "__main__":
    main()
