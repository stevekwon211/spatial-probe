# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Run occquery_v0 over the synthetic scenes and write denotation metrics.

Loads the schema-validated queries, evaluates each occupancy query over every scene under the
three unknown policies (free / occupied / ignored), scores the retrieved set against the
hand-labeled ground truth, and reports the tracking-baseline query separately. Writes a
machine-readable JSON and a human Markdown summary that keeps the result classes distinct
(unit test vs synthetic smoke vs externally validated -- the last is NONE yet).

No dataset: `synthetic.SCENES` is hand-built. At M2, swap in adapter-loaded scenes and the same
engine yields the real numbers. Usage: `python experiments/occquery_v0/run.py`.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))  # make `probe` importable as a script
sys.path.insert(0, str(_HERE))                      # make `synthetic` importable

from probe.grid import UnknownPolicy
from probe.query_spec import load_queries
from probe.retrieval import retrieved
from synthetic import GROUND_TRUTH, SCENES

_SEED = 0  # synthetic geometry is deterministic; recorded for provenance


def _prf1(ret: set[str], truth: set[str]) -> dict:
    ret, truth = set(ret), set(truth)
    tp = len(ret & truth)
    precision = tp / len(ret) if ret else 1.0
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": sorted(ret - truth),
        "false_negatives": sorted(truth - ret),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only; never fail the run over it
        return "unknown"


def _evaluate_occupancy(query, gt: set[str]) -> dict:
    free = retrieved(SCENES, query, UnknownPolicy.FREE)
    occ = retrieved(SCENES, query, UnknownPolicy.OCCUPIED)
    undetermined = free ^ occ  # scenes whose membership flips with the unknown policy
    ignored = free & occ       # determined matches only
    return {
        "scope": query.scope,
        "refav_expressible": query.refav_expressible,
        "gt": sorted(gt),
        "unknown_stable": free == occ,
        "per_policy": {
            "free": {"retrieved": sorted(free), **_prf1(free, gt)},
            "occupied": {"retrieved": sorted(occ), **_prf1(occ, gt)},
            "ignored": {"retrieved": sorted(ignored), "undetermined": sorted(undetermined), **_prf1(ignored, gt)},
        },
    }


def _evaluate_baseline(query, gt: set[str]) -> dict:
    ret = retrieved(SCENES, query, UnknownPolicy.FREE)  # unknown policy is irrelevant for tracking
    return {
        "backend": query.backend,
        "status": query.status,
        "scope": query.scope,
        "refav_expressible": query.refav_expressible,
        "gt": sorted(gt),
        "retrieved": sorted(ret),
        **_prf1(ret, gt),
    }


def build_report() -> dict:
    queries = load_queries(_HERE / "queries.yaml")
    occupancy: dict = {}
    baseline: dict = {}
    for q in queries:
        gt = set(GROUND_TRUTH.get(q.id, set()))
        if q.is_occupancy:
            occupancy[q.id] = _evaluate_occupancy(q, gt)
        else:
            baseline[q.id] = _evaluate_baseline(q, gt)
    n = len(queries)
    n_refav = sum(1 for q in queries if q.refav_expressible)
    return {
        "experiment": "occquery_v0",
        "result_class": "synthetic-smoke",
        "disclaimer": (
            "Hand-built scenes with constructed ground truth. Verifies the retrieval loop and "
            "predicate behavior on known geometry; NOT an externally validated scientific result. "
            "External numbers require the M2 nuScenes/Occ3D adapter."
        ),
        "commit": _git_commit(),
        "seed": _SEED,
        "n_scenes": len(SCENES),
        "scenes": [s.name for s in SCENES],
        "expressibility_coverage": {"occupancy": f"{n}/{n}", "refav": f"{n_refav}/{n}"},
        "occupancy_queries": occupancy,
        "baseline_queries": baseline,
    }


def _render_md(rep: dict) -> str:
    lines = [
        "# occquery_v0 -- synthetic smoke-test results",
        "",
        f"**Result class: {rep['result_class'].upper()}.** {rep['disclaimer']}",
        "",
        f"- seed: {rep['seed']}  (full commit hash + per-policy FP/FN are in results.json, gitignored)",
        f"- scenes ({rep['n_scenes']}): {', '.join(rep['scenes'])}",
        f"- expressibility coverage: occupancy {rep['expressibility_coverage']['occupancy']}, "
        f"RefAV {rep['expressibility_coverage']['refav']}",
        "",
        "## Occupancy queries (denotation vs constructed GT)",
        "",
        "| query | scope | F1 (free) | F1 (occupied) | unknown-stable | GT |",
        "|---|---|---|---|---|---|",
    ]
    for qid, r in rep["occupancy_queries"].items():
        lines.append(
            f"| `{qid}` | {r['scope']} | {r['per_policy']['free']['f1']} | "
            f"{r['per_policy']['occupied']['f1']} | {r['unknown_stable']} | {', '.join(r['gt'])} |"
        )
    lines += ["", "## Baseline (tracking backend; box-only, NOT occupancy retrieval)", ""]
    for qid, r in rep["baseline_queries"].items():
        lines.append(f"- `{qid}` ({r['backend']}/{r['status']}): F1={r['f1']}, retrieved={r['retrieved']}, GT={r['gt']}")
    lines += [
        "",
        "## Unknown-policy sensitivity",
        "",
        "Retrieval under unknown=free vs unknown=occupied. A query whose retrieved set changes is "
        "unknown-SENSITIVE; under the IGNORED policy the flipping scenes are excluded as undetermined.",
    ]
    sensitive = [qid for qid, r in rep["occupancy_queries"].items() if not r["unknown_stable"]]
    if sensitive:
        for qid in sensitive:
            und = rep["occupancy_queries"][qid]["per_policy"]["ignored"]["undetermined"]
            lines.append(f"- `{qid}`: SENSITIVE -- undetermined under IGNORED: {und}")
    else:
        lines.append("- all occupancy queries stable across the unknown policies on these scenes.")
    lines += [
        "",
        "## Result categories (do not conflate)",
        "- unit / integration tests: `pytest` (instrument + predicates), see CI/local run.",
        "- synthetic smoke (this file): constructed GT, not external.",
        "- externally validated benchmark: NONE yet -- requires the M2 Occ3D-nuScenes adapter.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rep = build_report()
    results = _HERE / "results"
    results.mkdir(exist_ok=True)
    (results / "results.json").write_text(json.dumps(rep, indent=2) + "\n")
    # synthetic smoke goes to smoke.md; the committed narrative is the hand-written summary.md
    (results / "smoke.md").write_text(_render_md(rep))

    print(f"occquery_v0 SYNTHETIC smoke -- {rep['n_scenes']} scenes, commit {rep['commit'][:8]}")
    for qid, r in rep["occupancy_queries"].items():
        free = r["per_policy"]["free"]
        flag = "" if r["unknown_stable"] else "   [unknown-SENSITIVE]"
        print(f"  [occ] {qid} ({r['scope']}): free F1={free['f1']} retrieved={free['retrieved']} GT={r['gt']}{flag}")
    for qid, r in rep["baseline_queries"].items():
        print(f"  [baseline:{r['backend']}] {qid}: F1={r['f1']} retrieved={r['retrieved']} GT={r['gt']}")
    cov = rep["expressibility_coverage"]
    print(f"  expressibility coverage: occupancy {cov['occupancy']}, RefAV {cov['refav']}")
    print(f"  wrote {results / 'results.json'} and {results / 'smoke.md'}")


if __name__ == "__main__":
    main()
