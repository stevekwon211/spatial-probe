# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""dynfield Tier-1 v2 — graded IDM surrogate, danger-stratified (pre-registered 2026-06-23).

v1's binary brake/proceed surrogate was too coarse (velocity flipped 3.3% of decisions, below the
8.5% shuffled floor). v2 (design sealed in preregistration.md BEFORE this run): the action is a
CONTINUOUS IDM deceleration, the ablated velocity enters only through IDM's closing-gap term, and the
effect is the decel-delta (a magnitude, not a binary flip). Regimes are cut NON-circularly on
agent-context × a STATIC-URGENCY danger band (ego_speed²/2·gap -- ego state + gap only, NOT the ablated
velocity; standard TTC is circular here). Primary = decel-delta with a shuffled-velocity control.
"necessary" stays reserved for Tier-2; this is action-sensitivity on a deterministic surrogate.

Run: python experiments/dynfield_v0/harness_v2.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

from harness import _CORRIDOR, _LEAD_RANGE, _agent_context  # reuse lead/corridor + class regime
from probe.adapters.occ3d import load_scene
from surrogate import plan_idm_motion, plan_idm_static

_DATA = _HERE.parents[1] / "data"
_URGENCY_HI = 1.5  # ego_speed^2/(2*gap) >= this m/s^2 required-decel = high-urgency (near-miss-ish)


def _lead(scene, t):
    """Nearest in-corridor lead box ahead -> (gap, ego_speed, lead_fwd_speed, label, yaw) or None."""
    ego = scene.ego_at(t)
    best = None
    for o in scene.objects_at(t):
        if 0.5 < o.center[0] <= _LEAD_RANGE and abs(o.center[1]) < _CORRIDOR:
            if (best is None or o.center[0] < best.center[0]) and not math.isnan(o.velocity[0]):
                best = o
    if best is None:
        return None
    return best.center[0], ego.speed, best.velocity[0], best.label, best.yaw


def _decel_delta(gap, ego_speed, lead_fwd):
    """|IDM(with velocity) - IDM(without)| -- the ablated motion field's effect on commanded accel."""
    return abs(plan_idm_motion(ego_speed, gap, lead_fwd) - plan_idm_static(ego_speed, gap))


def _boot_mean(vals, scene_ids, rng, n_boot=1000, level=0.95):
    if len(vals) < 4:
        return {"defined": False, "mean": float(np.mean(vals)) if vals else float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(vals)}
    v = np.array(vals, float); u = np.array(scene_ids); uniq = np.unique(u)
    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        pool = np.concatenate([v[u == s] for s in drawn])
        if len(pool):
            samples.append(pool.mean())
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(samples, [a, 100 - a])
    return {"defined": True, "mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": len(vals), "n_scenes": len(uniq)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    names = [s.strip() for s in (_HERE / "held-out-val.txt").read_text().splitlines() if s.strip()][: args.limit]
    rng = np.random.default_rng(0)

    rows = []
    print(f"loading {len(names)} held-out val scenes ...", flush=True)
    for i, name in enumerate(names):
        try:
            scene = load_scene(name, _DATA, mask="none", with_boxes=True)
        except (KeyError, FileNotFoundError):
            continue
        for t in scene.times():
            lr = _lead(scene, t)
            if lr is None:
                continue
            gap, ego_speed, lead_fwd, label, yaw = lr
            urgency = ego_speed * ego_speed / (2.0 * gap)             # static danger (no lead velocity)
            rows.append({"scene": name, "gap": gap, "ego": ego_speed, "lead_fwd": lead_fwd,
                         "ctx": _agent_context(label, yaw), "urgency": urgency,
                         "dd": _decel_delta(gap, ego_speed, lead_fwd),
                         "closing": ego_speed - lead_fwd})
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(names)} scenes, {len(rows)} lead-frames", flush=True)
    if not rows:
        sys.exit("no lead-frames.")

    # ---- GATE: surrogate-validity (graded) -- IDM brakes MORE as closing rises (monotone response) ----
    closing = np.array([r["closing"] for r in rows]); accel_m = np.array([plan_idm_motion(r["ego"], r["gap"], r["lead_fwd"]) for r in rows])
    valid_corr = float(np.corrcoef(closing, accel_m)[0, 1]) if len(rows) > 2 else 0.0  # closing up -> accel down
    valid_pass = valid_corr < -0.1

    # ---- shuffled-velocity control: true decel-delta must beat permuted-velocity decel-delta ----
    shuf_lead = rng.permutation([r["lead_fwd"] for r in rows])
    shuf_dd = [_decel_delta(r["gap"], r["ego"], float(sl)) for r, sl in zip(rows, shuf_lead)]

    # ---- the matrix: decel-delta by (agent-context x static-urgency), true vs shuffled, bootstrap CI ----
    def cell(sub, sub_shuf):
        t = _boot_mean([r["dd"] for r in sub], [r["scene"] for r in sub], rng)
        s = _boot_mean(sub_shuf, [r["scene"] for r in sub], rng)
        verdict = "INDETERMINATE"
        if t["defined"] and s["defined"]:
            verdict = "CHANGED" if t["lo"] > s["hi"] else ("EQUIVALENT" if t["hi"] <= s["hi"] else "INDETERMINATE")
        return {"true": t, "shuffled": s, "verdict": verdict}

    matrix = {}
    for ctx in ["vehicle_following", "vehicle_crossing", "vru", "other"]:
        for band, lo, hi in [("low_urgency", -1e9, _URGENCY_HI), ("high_urgency", _URGENCY_HI, 1e9)]:
            idx = [j for j, r in enumerate(rows) if r["ctx"] == ctx and lo <= r["urgency"] < hi]
            if len(idx) < 4:
                continue
            matrix[f"{ctx}|{band}"] = cell([rows[j] for j in idx], [shuf_dd[j] for j in idx])

    report = {
        "n_scenes": len({r["scene"] for r in rows}), "n_lead_frames": len(rows),
        "gates": {"surrogate_validity": {"pass": valid_pass, "closing_accel_corr": valid_corr},
                  "shuffled_global": {"true_dd": float(np.mean([r["dd"] for r in rows])), "shuffled_dd": float(np.mean(shuf_dd))}},
        "decel_delta_matrix": {k: {"true_mean": v["true"]["mean"], "true_ci": [v["true"]["lo"], v["true"]["hi"]],
                                   "shuffled_ci": [v["shuffled"]["lo"], v["shuffled"]["hi"], ] if v["shuffled"]["defined"] else None,
                                   "n": v["true"]["n"], "verdict": v["verdict"]} for k, v in matrix.items()},
        "framing": "ACTION-SENSITIVITY (decel-delta), graded IDM, NOT necessity (Tier-2/GPU).",
    }
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "tier1_v2_matrix.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\ndynfield Tier-1 v2 (graded IDM, {report['n_scenes']} scenes, {len(rows)} lead-frames):\n")
    g = report["gates"]
    print(f"  GATE surrogate-validity: {'PASS' if valid_pass else 'FAIL'} (closing↑→accel↓ corr {valid_corr:+.2f}, need <-0.1)")
    print(f"  shuffled (global): true decel-delta {g['shuffled_global']['true_dd']:.3f} vs shuffled {g['shuffled_global']['shuffled_dd']:.3f}")
    print(f"\n  decel-delta by (agent-context × static-urgency), true vs shuffled CI:")
    for k, v in matrix.items():
        t, s = v["true"], v["shuffled"]
        sci = f"shuf[{s['lo']:.2f},{s['hi']:.2f}]" if s["defined"] else "shuf n/a"
        print(f"    {k:30} {v['verdict']:13} true {t['mean']:.2f} CI[{t['lo']:.2f},{t['hi']:.2f}]  {sci}  (n={t['n']})")
    print(f"\n  wrote {_HERE / 'results' / 'tier1_v2_matrix.json'}")
    print("  CHANGED = true decel-delta CI clears the shuffled band (velocity moves the action beyond noise).")
    print("  Non-uniform verdicts across regimes = the by-regime action-sensitivity result. Uniform = a negative.")


if __name__ == "__main__":
    main()
