# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""R0 -- occupancy-vs-box action-sensitivity (design sealed in r0_preregistration.md BEFORE this run).

Q1 (oracle-free): does swapping the spatial REPRESENTATION fed to a fixed IDM -- dense occupancy
free-distance (occ_gap) vs the object-box lead distance (box_gap) -- CHANGE the commanded action,
beyond a shuffled-occupancy null, by a pre-named regime set? Same fixed IDM transfer function; ONLY
the gap input swaps. This is the action-sensitivity analogue of occquery H1 (expressivity).

Gates run FIRST: (1) surrogate-validity (IDM brakes more as closing rises); (2) predicate-correctness
(occ_gap agrees with box_gap within one voxel on CLEAN frames, else the occ predicate's false positives
fake the delta -> INVALID, not a result). Ceiling: Q1 only; no outcome/better claim (see preregistration).

Run: python experiments/dynfield_v0/r0_action_sensitivity.py [--limit N] [--horizon S]
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

from harness import _CORRIDOR, _LEAD_RANGE, _agent_context  # reuse lead window/corridor + class regime
from harness_v2 import _URGENCY_HI, _boot_mean                    # reuse v2 bootstrap + urgency cut
from probe.adapters.occ3d import load_scene
from surrogate import plan_idm_motion, plan_idm_static

_DATA = _HERE.parents[1] / "data"
_VOXEL = 0.4  # predicate-correctness tolerance: one Occ3D voxel (m)


def _occ_forward_gap(grid, ego) -> float:
    """v2 (r0v2_preregistration.md): nearest occupancy obstacle SURFACE ahead in the ego corridor up to
    _LEAD_RANGE -- the speed-independent, same-window analogue of the box lead distance. No
    reachable_free_field, no speed-capped BEV horizon. Clear-to-_LEAD_RANGE if the corridor is empty."""
    centers = grid.obstacle_centers(max_height_agl=ego.height)
    if not len(centers):
        return _LEAD_RANGE
    fwd, lat = ego.to_ego_frame(centers[:, :2])
    m = (fwd > 0.5) & (fwd <= _LEAD_RANGE) & (np.abs(lat) < ego.width / 2.0)  # v3: ego in-path strip, not box corridor
    return float(fwd[m].min()) if m.any() else _LEAD_RANGE


def _lead_box(scene, t):
    """Nearest in-corridor lead box ahead -> (box, ego) or None. Same window as harness._lead, but
    returns the box object so box_gap can use the FRONT SURFACE (center - length/2) to match occ surface."""
    ego = scene.ego_at(t)
    best = None
    for o in scene.objects_at(t):
        if 0.5 < o.center[0] <= _LEAD_RANGE and abs(o.center[1]) < _CORRIDOR and not math.isnan(o.velocity[0]):
            if best is None or o.center[0] < best.center[0]:
                best = o
    return (best, ego) if best is not None else None


def _delta(occ_gap: float, box_gap: float, ego_speed: float) -> float:
    """|IDM(occ_gap) - IDM(box_gap)| -- the representation swap's effect on the commanded accel.
    v3: clamp each IDM action to the physical [-9, +3] m/s^2 band before differencing (an action, not a
    sub-meter-gap blowup)."""
    def a(g: float) -> float:
        return max(-9.0, min(3.0, plan_idm_static(ego_speed, g)))
    return abs(a(occ_gap) - a(box_gap))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--horizon", type=float, default=3.0)
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
            lb = _lead_box(scene, t)
            if lb is None:
                continue
            box, ego = lb
            ego_speed = ego.speed
            box_gap = box.center[0] - box.size[0] / 2.0   # FRONT surface -- matches the occ obstacle surface
            lead_fwd = box.velocity[0]
            try:
                occ_gap = _occ_forward_gap(scene.grid_at(t), ego)
            except Exception:
                continue
            urgency = ego_speed * ego_speed / (2.0 * max(box_gap, 0.1))  # static danger (ego + box gap; non-circular)
            rows.append({"scene": name, "box_gap": box_gap, "occ_gap": occ_gap, "ego": ego_speed,
                         "lead_fwd": lead_fwd, "ctx": _agent_context(box.label, box.yaw), "urgency": urgency,
                         "dd": _delta(occ_gap, box_gap, ego_speed), "closing": ego_speed - lead_fwd})
        if (i + 1) % 15 == 0:
            print(f"  {i + 1}/{len(names)} scenes, {len(rows)} lead-frames", flush=True)
    if not rows:
        sys.exit("no lead-frames.")

    # ---- GATE 1: surrogate-validity -- IDM brakes MORE as closing rises ----
    closing = np.array([r["closing"] for r in rows]); accel_m = np.array([plan_idm_motion(r["ego"], r["box_gap"], r["lead_fwd"]) for r in rows])
    g1_corr = float(np.corrcoef(closing, accel_m)[0, 1]) if len(rows) > 2 else 0.0
    g1_pass = g1_corr < -0.1

    # ---- GATE 2: predicate-correctness -- on CLEAN frames occ_gap must match box_gap within one voxel ----
    # clean = a box lead within range AND the occ centerline blocks near it (both see the same lead);
    # exclude clear-to-horizon occ frames (no centerline obstacle -> not a clean comparison).
    clean = [r for r in rows if r["occ_gap"] < _LEAD_RANGE - 0.5]  # occ found an obstacle in the corridor
    agree = [r for r in clean if abs(r["occ_gap"] - r["box_gap"]) <= _VOXEL]
    g2_rate = len(agree) / len(clean) if clean else 0.0
    g2_pass = g2_rate >= 0.70

    # ---- shuffled-occupancy null: true dd must beat permuted-occ-gap dd ----
    shuf_occ = rng.permutation([r["occ_gap"] for r in rows])
    shuf_dd = [_delta(float(so), r["box_gap"], r["ego"]) for r, so in zip(rows, shuf_occ)]

    # ---- the matrix: action-delta by (agent-context x static-urgency), true vs shuffled, bootstrap CI ----
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
        "gates": {
            "surrogate_validity": {"pass": g1_pass, "closing_accel_corr": g1_corr},
            "predicate_correctness": {"pass": g2_pass, "agree_rate": g2_rate, "n_clean": len(clean),
                                      "note": "occ_gap vs box_gap within 0.4m on clean lead-frames; <0.70 => occ predicate fakes the delta => INVALID"},
            "shuffled_global": {"true_dd": float(np.mean([r["dd"] for r in rows])), "shuffled_dd": float(np.mean(shuf_dd))},
        },
        "valid": bool(g1_pass and g2_pass),
        "action_delta_matrix": {k: {"true_mean": v["true"]["mean"], "true_ci": [v["true"]["lo"], v["true"]["hi"]],
                                    "shuffled_ci": [v["shuffled"]["lo"], v["shuffled"]["hi"]] if v["shuffled"]["defined"] else None,
                                    "n": v["true"]["n"], "verdict": v["verdict"]} for k, v in matrix.items()},
        "framing": "Q1 ACTION-SENSITIVITY (occ vs box gap through fixed IDM), oracle-free. NOT Q2 outcome / Q3 better.",
    }
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "r0_action_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nR0 occ-vs-box action-sensitivity ({report['n_scenes']} scenes, {len(rows)} lead-frames):\n")
    print(f"  GATE-1 surrogate-validity:   {'PASS' if g1_pass else 'FAIL'} (closing↑→accel↓ corr {g1_corr:+.2f}, need <-0.1)")
    print(f"  GATE-2 predicate-correctness:{'PASS' if g2_pass else 'FAIL'} (occ≈box within 0.4m on {g2_rate:.0%} of {len(clean)} clean frames, need ≥70%)")
    if not (g1_pass and g2_pass):
        print("  -> a gate FAILED: this run is INVALID, not a result (see r0_preregistration.md).")
    print(f"  shuffled (global): true action-delta {report['gates']['shuffled_global']['true_dd']:.3f} vs shuffled {report['gates']['shuffled_global']['shuffled_dd']:.3f}")
    print(f"\n  action-delta by (agent-context × static-urgency), true vs shuffled CI:")
    for k, v in matrix.items():
        t, s = v["true"], v["shuffled"]
        sci = f"shuf[{s['lo']:.2f},{s['hi']:.2f}]" if s["defined"] else "shuf n/a"
        print(f"    {k:30} {v['verdict']:13} true {t['mean']:.2f} CI[{t['lo']:.2f},{t['hi']:.2f}]  {sci}  (n={t['n']})")
    print(f"\n  wrote {_HERE / 'results' / 'r0_action_sensitivity.json'}")
    print("  CHANGED = true CI clears shuffled band (the representation moves the action beyond noise).")
    print("  EQUIVALENT-everywhere = the pre-registered NEGATIVE headline. Q1 only; no outcome/better claim.")


if __name__ == "__main__":
    main()
