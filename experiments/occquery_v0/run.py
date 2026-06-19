# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Run occquery_v0 over the synthetic scenes and report denotation metrics.

Each query in queries.yaml is a small expression over the predicate functions. We
evaluate it on every (scene, frame); a scene is *retrieved* if any frame satisfies it.
The retrieved set is scored against the hand-labeled ground truth (denotation P/R/F1).
Queries whose predicates are not yet in the v0 core are reported as SKIP.

No dataset: `synthetic.SCENES` is hand-built. At M2, swap in adapter-loaded scenes and the
same loop yields the real numbers. Usage: `python experiments/occquery_v0/run.py`.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))  # make `probe` importable when run as a script
sys.path.insert(0, str(_HERE))                      # make `synthetic` importable

import yaml

from probe.predicates.clearance import lateral_clearance
from probe.predicates.freepath import free_along_ego_path
from probe.query_dsl import UnsafeExpression, safe_eval
from probe.scene import Scene
from synthetic import GROUND_TRUTH, SCENES


def _namespace(scene: Scene) -> dict:
    """The only identifiers a query may reference. The expression is evaluated by
    probe.query_dsl.safe_eval (a whitelist AST walker), never by Python eval()."""
    return {
        "scene": scene,
        "lateral_clearance": lambda sc, t: lateral_clearance(sc.grid_at(t), sc.ego_at(t)),
        "free_along_ego_path": lambda sc, t, h: free_along_ego_path(sc.grid_at(t), sc.ego_at(t), h),
        "ego_speed": lambda sc, t: sc.ego_speed(t),
        "ego_width": lambda sc: sc.ego_width(),
    }


def _scene_matches(scene: Scene, predicate: str) -> bool:
    for t in scene.times():
        names = _namespace(scene)
        names["t"] = t
        if bool(safe_eval(predicate, names)):
            return True
    return False


def _prf1(retrieved: set[str], truth: set[str]) -> tuple[float, float, float]:
    tp = len(retrieved & truth)
    p = tp / len(retrieved) if retrieved else 1.0
    r = tp / len(truth) if truth else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main() -> None:
    queries = yaml.safe_load((_HERE / "queries.yaml").read_text())["queries"]
    print(f"{len(SCENES)} synthetic scenes, {len(queries)} queries\n")

    for q in queries:
        qid, predicate = q["id"], q["predicate"]
        try:
            retrieved = {s.name for s in SCENES if _scene_matches(s, predicate)}
        except NameError as exc:
            print(f"- {qid}: SKIP (v0 core lacks {exc})")
            continue
        except (SyntaxError, UnsafeExpression) as exc:
            print(f"- {qid}: SKIP (not v0-evaluable: {exc})")
            continue
        truth = GROUND_TRUTH.get(qid)
        if truth is None:
            print(f"- {qid}: retrieved={sorted(retrieved)}  (no GT labeled in v0)")
            continue
        p, r, f1 = _prf1(retrieved, truth)
        print(f"- {qid}: retrieved={sorted(retrieved)} GT={sorted(truth)} | P={p:.2f} R={r:.2f} F1={f1:.2f}")

    n = len(queries)
    n_refav = sum(1 for q in queries if q["refav_expressible"])
    print(f"\nexpressibility coverage: occupancy {n}/{n}, RefAV {n_refav}/{n}")


if __name__ == "__main__":
    main()
