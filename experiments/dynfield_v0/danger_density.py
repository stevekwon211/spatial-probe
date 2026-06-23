# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Resource-zero feasibility probe: how much genuine DANGER is in real nuScenes val?

Before building any closed-loop sim (R1), this counts how many lead-frames are actually
near-miss-like on the held-out val split, by several velocity-INDEPENDENT and velocity-dependent
danger definitions. It answers one decision input for the R1 pre-registration: can a necessity
measurement stand on REAL danger frames, or must danger be GENERATED (which raises the
self-built-sim circularity concern)? This is a power/feasibility check, NOT a hypothesis test --
the dynfield hypothesis (velocity necessary in danger) is already pre-registered; this only sizes
the regime cells. Read-only, pure numpy, no GPU, no new data.

Run: python experiments/dynfield_v0/danger_density.py [--limit N]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from harness import _CORRIDOR, _LEAD_RANGE, _agent_context
from probe.adapters.occ3d import load_scene

_DATA = _HERE.parents[1] / "data"


def _lead(scene, t):
    """Nearest in-corridor lead ahead -> (gap, ego_speed, lead_fwd, closing, label, yaw) or None."""
    best = None
    for o in scene.objects_at(t):
        if 0.5 < o.center[0] <= _LEAD_RANGE and abs(o.center[1]) < _CORRIDOR:
            if (best is None or o.center[0] < best.center[0]) and not math.isnan(o.velocity[0]):
                best = o
    if best is None:
        return None
    ego = scene.ego_at(t)
    return best.center[0], ego.speed, best.velocity[0], ego.speed - best.velocity[0], best.label, best.yaw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()
    names = [s.strip() for s in (_HERE / "held-out-val.txt").read_text().splitlines() if s.strip()][: args.limit]

    rows = []
    print(f"scanning {len(names)} held-out val scenes for danger density ...", flush=True)
    for i, name in enumerate(names):
        try:
            scene = load_scene(name, _DATA, mask="none", with_boxes=True)
        except (KeyError, FileNotFoundError):
            continue
        for t in scene.times():
            lr = _lead(scene, t)
            if lr is None:
                continue
            gap, ego, lead_fwd, closing, label, yaw = lr
            urgency = ego * ego / (2.0 * gap)               # velocity-INDEPENDENT static danger (m/s^2)
            ttc = gap / closing if closing > 0.1 else float("inf")  # velocity-dependent
            rows.append({"scene": name, "gap": gap, "ego": ego, "closing": closing,
                         "urgency": urgency, "ttc": ttc, "ctx": _agent_context(label, yaw)})
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(names)} scenes, {len(rows)} lead-frames", flush=True)
    if not rows:
        sys.exit("no lead-frames.")

    n = len(rows)
    n_scenes = len({r["scene"] for r in rows})
    urg = np.array([r["urgency"] for r in rows])
    ttc = np.array([r["ttc"] for r in rows])
    gap = np.array([r["gap"] for r in rows])
    clos = np.array([r["closing"] for r in rows])

    def frac(mask):
        m = np.asarray(mask)
        sc = len({rows[j]["scene"] for j in np.where(m)[0]})
        return int(m.sum()), 100.0 * m.sum() / n, sc

    print(f"\nDANGER DENSITY on real nuScenes val ({n_scenes} scenes, {n} lead-frames):\n")
    print(f"  static-urgency ego^2/2gap (m/s^2):  median {np.median(urg):.2f}  p90 {np.percentile(urg,90):.2f}  p99 {np.percentile(urg,99):.2f}  max {urg.max():.2f}")
    print(f"  gap (m):                            median {np.median(gap):.1f}   p10 {np.percentile(gap,10):.1f}   min {gap.min():.1f}")
    print(f"  closing speed (m/s):                median {np.median(clos):+.2f}  p90 {np.percentile(clos,90):+.2f}  max {clos.max():+.2f}")
    print(f"\n  velocity-INDEPENDENT danger bands (the non-circular regime cut):")
    for thr in (1.0, 1.5, 2.5, 4.0):
        c, p, sc = frac(urg >= thr)
        print(f"    urgency >= {thr:>4}:  {c:5d} frames ({p:5.1f}%)  across {sc:3d} scenes")
    print(f"\n  velocity-DEPENDENT near-miss (genuinely closing & near):")
    for g, cs in ((10.0, 2.0), (7.0, 3.0), (5.0, 4.0)):
        c, p, sc = frac((gap < g) & (clos > cs))
        print(f"    gap<{g:>4}m & closing>{cs:>3}m/s:  {c:5d} frames ({p:5.1f}%)  across {sc:3d} scenes")
    for tthr in (3.0, 2.0, 1.5):
        c, p, sc = frac(ttc < tthr)
        print(f"    TTC < {tthr}s:                   {c:5d} frames ({p:5.1f}%)  across {sc:3d} scenes")
    print(f"\n  READING: if the velocity-INDEPENDENT danger bands have enough frames across enough")
    print(f"  scenes, R1 can measure necessity on REAL danger. If near-empty, danger must be GENERATED")
    print(f"  (perturbation) -> stronger self-built-sim circularity caveat -> R2/R3 matters more.")


if __name__ == "__main__":
    main()
