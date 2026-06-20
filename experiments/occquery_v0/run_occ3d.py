# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Run the occquery_v0 occupancy predicates over REAL Occ3D-nuScenes mini scenes.

No ground truth yet (hand-labeling pending), so this reports the retrieved scene set per occupancy
query under the unknown=free vs unknown=occupied policies -- a first real-data signal: does each
query fire on plausible scenes, and how unknown-sensitive is it on real sensor coverage? Also a
control on the visibility mask (lidar vs camera). This is a SIGNAL, not a denotation result; P/R/F1
needs hand labels.

Usage: python experiments/occquery_v0/run_occ3d.py [lidar|camera]
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from probe.adapters.occ3d import load_scene
from probe.grid import UnknownPolicy
from probe.query_spec import load_queries
from probe.retrieval import retrieved

_DATA = _HERE.parents[1] / "data"
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]


def main(mask: str = "lidar") -> None:
    print(f"loading {len(MINI)} Occ3D-nuScenes mini scenes (mask={mask}) ...")
    scenes = [load_scene(name, _DATA, mask=mask) for name in MINI]
    nframes = sum(len(s) for s in scenes)
    print(f"  {len(scenes)} scenes, {nframes} frames total\n")

    queries = [q for q in load_queries(_HERE / "queries.yaml") if q.is_occupancy]
    for q in queries:
        free = retrieved(scenes, q, UnknownPolicy.FREE)
        occ = retrieved(scenes, q, UnknownPolicy.OCCUPIED)
        flag = "" if free == occ else "   [unknown-SENSITIVE]"
        print(f"[{q.scope}] {q.id}")
        print(f"   unknown=free      -> {sorted(free)}")
        print(f"   unknown=occupied  -> {sorted(occ)}{flag}")

    print("\nNOTE: no hand-labeled GT yet -> retrieved sets only, NOT P/R/F1. A real-data signal,")
    print("not a denotation result. Hand-label scene/frame membership to score H3.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
