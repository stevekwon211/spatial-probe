# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Score occquery H3 from the human labels (the DEMOTED internal-consistency arm).

Consumes the sealed pool (`pool-<id>.json`) + the human verdicts (`labels/<id>.jsonl`) and reports,
per occupancy query: the occupancy predicate's denotation P/R/F1 vs the human ground-truth, a
cluster-bootstrap CI, and the relative gap over the best box-only approximation (a distance-to-
nearest-object threshold swept as a CURVE -- the load-bearing >=20-F1 claim, never a single cutoff).

HONEST SCOPE (stated up front, per preregistration.md): this is human-vs-code consistency on the SAME
Occ3D data -- a different ALGORITHM (human spatial judgment) but the SAME DATA SOURCE -- so it is an
INTERNAL-CONSISTENCY check, NOT external denotation correctness. The human catches gross semantic/logic
errors the automated point-NN consistency probe cannot, and yields retrieval P/R/F1; it does NOT earn
independence (that needs a different MODALITY, e.g. the camera-projection oracle). H1 stays the sole
headline. A pilot pool selects scenes by predicate output (a forking path) and is a SYSTEM-VALIDATION
run, never a result -- flagged from pool-meta `is_pilot`.

Under-power is reported honestly: a query with < MIN_POSITIVE human-positive scenes yields an
undefined CI (never a vacuous F1=1.0). Skipped verdicts are excluded from scoring, counted separately.

Usage: python experiments/occquery_v0/score_h3.py --pool pilot
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from probe.adapters.occ3d import load_scene
from probe.grid import UnknownPolicy
from probe.metrics import bootstrap_f1, prf1
from probe.predicates.objects import distance_to_nearest_object
from probe.query_spec import load_queries
from probe.retrieval import scene_matches

_DATA = _HERE.parents[1] / "data"
_MIN_POSITIVE = 2  # below this a bootstrap CI is undefined (matches metrics.bootstrap_f1 default)
_BOX_TAUS = np.round(np.arange(0.5, 10.01, 0.5), 2)  # pre-registered box-distance sweep (the curve)


def _load_verdicts(pool: str) -> dict[tuple[str, str], str]:
    """{(query_id, scene_id): verdict} from labels/<pool>.jsonl (last write wins per task)."""
    path = _HERE / "labels" / f"{pool}.jsonl"
    if not path.exists():
        sys.exit(f"no labels at {path} -- label the pool in the web /occquery/label view and Save first.")
    verdicts: dict[tuple[str, str], str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        verdicts[(r["query_id"], r["scene_id"])] = r["verdict"]
    return verdicts


def _box_only_curve(scenes, query, human_yes: set[str], human_labeled: set[str]) -> list[dict]:
    """Sweep a box-only proxy (nearest-object distance < tau, any class) and score F1 vs the human GT,
    over the labeled scenes only. The best point is the 'best box-only approximation' -- reported as a
    full curve so no tau is cherry-picked."""
    by_name = {s.name: s for s in scenes}
    curve = []
    for tau in _BOX_TAUS:
        retrieved = {
            name for name in human_labeled
            if any(distance_to_nearest_object(by_name[name], t) < tau for t in by_name[name].times())
        }
        m = prf1(retrieved & human_labeled, human_yes)
        curve.append({"tau": float(tau), "f1": m["f1"]})
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=str, default="pilot")
    args = ap.parse_args()

    pool_meta = json.loads((_HERE / f"pool-{args.pool}.json").read_text())
    verdicts = _load_verdicts(args.pool)
    queries = {q.id: q for q in load_queries(_HERE / "queries.yaml")}
    scene_ids = sorted({t["scene_id"] for t in pool_meta["tasks"]})
    print(f"loading {len(scene_ids)} pool scenes (with boxes for the box-only baseline) ...", flush=True)
    scenes = [load_scene(name, _DATA, mask="none", with_boxes=True) for name in scene_ids]
    by_name = {s.name: s for s in scenes}

    per_query = {}
    for qid in pool_meta["query_ids"]:
        q = queries[qid]
        labeled = {sid for (qq, sid) in verdicts if qq == qid and verdicts[(qq, sid)] != "skip"}
        human_yes = {sid for (qq, sid) in verdicts if qq == qid and verdicts[(qq, sid)] == "yes"}
        n_skip = sum(1 for (qq, sid) in verdicts if qq == qid and verdicts[(qq, sid)] == "skip")
        if not labeled:
            per_query[qid] = {"status": "unlabeled"}
            continue
        # occupancy predicate retrieval over the labeled scenes (scoring policy = FREE, sealed)
        occ_ret = {name for name in labeled if scene_matches(by_name[name], q, UnknownPolicy.FREE)}
        occ = prf1(occ_ret, human_yes)
        per_scene = {name: (name in occ_ret, name in human_yes) for name in labeled}
        ci = bootstrap_f1(per_scene, n_boot=1000, min_positive=_MIN_POSITIVE)
        box_curve = _box_only_curve(scenes, q, human_yes, labeled)
        best_box = max((p["f1"] for p in box_curve), default=0.0)
        per_query[qid] = {
            "status": "scored",
            "n_labeled": len(labeled), "n_human_yes": len(human_yes), "n_skip": n_skip,
            "occupancy_f1": occ["f1"], "precision": occ["precision"], "recall": occ["recall"],
            "occupancy_f1_ci": {"defined": ci.defined, "lo": ci.lo, "hi": ci.hi, "n_effective": ci.n_effective, "reason": ci.reason},
            "best_box_only_f1": round(best_box, 3),
            "relative_gap": round(occ["f1"] - best_box, 3),
            "box_curve": box_curve,
            "false_positives": occ["false_positives"], "false_negatives": occ["false_negatives"],
            "underpowered": len(human_yes) < _MIN_POSITIVE,
        }

    report = {
        "pool_id": pool_meta["pool_id"], "is_pilot": pool_meta["is_pilot"],
        "scoring_policy": "free", "min_positive": _MIN_POSITIVE,
        "honest_scope": ("INTERNAL-CONSISTENCY (human-vs-code on the same Occ3D data, different "
                         "algorithm/same source) -- NOT external denotation correctness. H1 is the sole "
                         "headline. The relative gap is a consistency comparison, reported as a curve."),
        "per_query": per_query,
    }
    out = _HERE / "results" / f"h3-{args.pool}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nH3 internal-consistency ({'PILOT system-validation, NOT a result' if pool_meta['is_pilot'] else 'val'}):\n")
    for qid, r in per_query.items():
        if r["status"] != "scored":
            print(f"  {qid}: {r['status']}")
            continue
        gap = r["relative_gap"]
        ci = r["occupancy_f1_ci"]
        cistr = f"CI[{ci['lo']:.2f},{ci['hi']:.2f}]" if ci["defined"] else f"CI undefined ({ci['reason']})"
        flag = "  [UNDER-POWERED]" if r["underpowered"] else ""
        print(f"  {qid}: occ F1={r['occupancy_f1']:.2f} (P={r['precision']:.2f} R={r['recall']:.2f}) {cistr}")
        print(f"      best box-only F1={r['best_box_only_f1']:.2f}  relative gap={gap:+.2f}"
              f"  | {r['n_human_yes']}/{r['n_labeled']} human-yes, {r['n_skip']} skip{flag}")
        if r["false_positives"]:
            print(f"      predicate FP (fired, human said no): {r['false_positives']}")
        if r["false_negatives"]:
            print(f"      predicate FN (missed, human said yes): {r['false_negatives']}")
    print(f"\n  wrote {out}")
    print("  READING: same-data consistency, not external truth. Gap is descriptive. Independence needs")
    print("  a different modality (camera-projection oracle), not more same-data labels.")


if __name__ == "__main__":
    main()
