# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""R0-danger -- occupancy-vs-box action-sensitivity on the AV2 DANGER substrate (design sealed in
r0_danger_preregistration.md BEFORE this run).

R0-v3 returned a pre-registered NEGATIVE on SAFE nuScenes following (every regime EQUIVALENT;
high_urgency cell n=4, under-powered). This runs the IDENTICAL R0 estimand + gates + matrix on
AV2-Sensor val danger frames (REFERRED vehicle-longitudinal windows; the sealed selection lives in
av2_danger_logs.json). The estimand functions (_occ_forward_gap, _lead_box, _delta) are IMPORTED from
r0_action_sensitivity so they cannot drift; only the substrate (av2_sensor.load_scene + the sealed
danger json) changes. Same gates, shuffled null, (agent-context x static-urgency) matrix, and
scene-clustered (= per-log) bootstrap CI.

Adversarial-review hardening (two independent reviewers, neither flipped the negative; both flagged one
honest weakening each -- folded in as REPORTING, the sealed estimand/verdict unchanged):
  - effective-N: each cell now reports n_logs (distinct clusters) + cluster_thin (<3 logs). A cell can be
    "defined" on >=4 FRAMES from 1-2 logs; that is suggestive, not decisive. Power = independent logs, not
    frames. This makes the R0-v3 "n=4" / any "n=78" over-power claim structurally visible.
  - lateral-window fairness: occ_gap scans the ego in-path strip (|lat| < ego.width/2 ~ 0.925 m) while
    box_gap draws its lead from the wider _CORRIDOR (1.5 m). The run now ALSO computes occ at the box
    corridor (dd_wide) and reports the verdict under both footprints, so "the occ strip was narrowed to
    fake agreement" is answerable from the output (the verdict is invariant).

Run: python experiments/dynfield_v0/r0_danger.py [--limit N]
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

from harness import _CORRIDOR, _LEAD_RANGE, _agent_context
from harness_v2 import _URGENCY_HI, _boot_mean
from probe.adapters.av2_sensor import load_scene
from r0_action_sensitivity import _VOXEL, _delta, _lead_box, _occ_forward_gap
from surrogate import plan_idm_motion

_AV2 = _HERE.parents[1] / "data" / "danger" / "av2_sensor"
_LOGS = _HERE / "av2_danger_logs.json"
_MIN_LOGS = 3  # below this a cell's bootstrap CI is cluster-thin (suggestive, not decisive)


def _occ_gap_w(grid, ego, half_width: float) -> float:
    """occ_gap at a configurable lateral half-width -- the robustness knob for the strip-vs-corridor
    fairness point. The sealed predicate uses ego.width/2; passing _CORRIDOR measures occ over the SAME
    lateral footprint box_gap selects its lead from, so the two distances are comparable like-for-like."""
    centers = grid.obstacle_centers(max_height_agl=ego.height)
    if not len(centers):
        return _LEAD_RANGE
    fwd, lat = ego.to_ego_frame(centers[:, :2])
    m = (fwd > 0.5) & (fwd <= _LEAD_RANGE) & (np.abs(lat) < half_width)
    return float(fwd[m].min()) if m.any() else _LEAD_RANGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=18)
    args = ap.parse_args()
    danger = json.loads(_LOGS.read_text())
    names = list(danger)[: args.limit]
    rng = np.random.default_rng(0)

    rows = []
    print(f"loading {len(names)} AV2 danger logs ...", flush=True)
    for i, name in enumerate(names):
        ts = [int(t) for t in danger[name]]
        try:
            scene = load_scene(name, _AV2, timestamps=ts)
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
            grid = scene.grid_at(t)
            try:
                occ_gap = _occ_forward_gap(grid, ego)        # sealed: ego in-path strip |lat| < ego.width/2
            except Exception:
                continue
            occ_gap_wide = _occ_gap_w(grid, ego, _CORRIDOR)  # robustness: occ over the SAME footprint as box_gap
            urgency = ego_speed * ego_speed / (2.0 * max(box_gap, 0.1))  # static danger (ego + box gap; non-circular)
            rows.append({"scene": name, "box_gap": box_gap, "occ_gap": occ_gap, "occ_gap_wide": occ_gap_wide,
                         "ego": ego_speed, "lead_fwd": lead_fwd, "ctx": _agent_context(box.label, box.yaw),
                         "urgency": urgency, "dd": _delta(occ_gap, box_gap, ego_speed),
                         "dd_wide": _delta(occ_gap_wide, box_gap, ego_speed), "closing": ego_speed - lead_fwd})
        print(f"  {i + 1}/{len(names)} {name[:12]} -> {len(rows)} lead-frames", flush=True)
    if not rows:
        sys.exit("no lead-frames.")

    # ---- GATE 1: surrogate-validity -- IDM brakes MORE as closing rises ----
    closing = np.array([r["closing"] for r in rows]); accel_m = np.array([plan_idm_motion(r["ego"], r["box_gap"], r["lead_fwd"]) for r in rows])
    g1_corr = float(np.corrcoef(closing, accel_m)[0, 1]) if len(rows) > 2 else 0.0
    g1_pass = g1_corr < -0.1

    # ---- GATE 2: predicate-correctness -- on CLEAN frames occ_gap must match box_gap within one voxel ----
    def _g2(key):
        clean = [r for r in rows if r[key] < _LEAD_RANGE - 0.5]
        rate = sum(abs(r[key] - r["box_gap"]) <= _VOXEL for r in clean) / len(clean) if clean else 0.0
        return rate, len(clean)
    g2_rate, g2_n = _g2("occ_gap")
    g2_pass = g2_rate >= 0.70
    g2w_rate, g2w_n = _g2("occ_gap_wide")  # same gate under the box-corridor footprint

    # ---- shuffled-occupancy null (both footprints): true dd must beat permuted-occ-gap dd ----
    shuf_dd = [_delta(float(so), r["box_gap"], r["ego"]) for r, so in zip(rows, rng.permutation([r["occ_gap"] for r in rows]))]
    shuf_dd_wide = [_delta(float(so), r["box_gap"], r["ego"]) for r, so in zip(rows, rng.permutation([r["occ_gap_wide"] for r in rows]))]

    # ---- the matrix: action-delta by (agent-context x static-urgency), true vs shuffled, per-log bootstrap CI ----
    def _verdict(dd_list, shuf_list, scenes):
        t = _boot_mean(dd_list, scenes, rng); s = _boot_mean(shuf_list, scenes, rng)
        v = "INDETERMINATE"
        if t["defined"] and s["defined"]:
            v = "CHANGED" if t["lo"] > s["hi"] else ("EQUIVALENT" if t["hi"] <= s["hi"] else "INDETERMINATE")
        return t, s, v

    matrix = {}
    for ctx in ["vehicle_following", "vehicle_crossing", "vru", "other"]:
        for band, lo, hi in [("low_urgency", -1e9, _URGENCY_HI), ("high_urgency", _URGENCY_HI, 1e9)]:
            idx = [j for j, r in enumerate(rows) if r["ctx"] == ctx and lo <= r["urgency"] < hi]
            if len(idx) < 4:
                continue
            sub = [rows[j] for j in idx]; scenes = [r["scene"] for r in sub]
            n_logs = len(set(scenes))
            t, s, v = _verdict([r["dd"] for r in sub], [shuf_dd[j] for j in idx], scenes)
            _, _, v_wide = _verdict([r["dd_wide"] for r in sub], [shuf_dd_wide[j] for j in idx], scenes)
            matrix[f"{ctx}|{band}"] = {"true": t, "shuffled": s, "verdict": v, "verdict_wide": v_wide,
                                       "n_logs": n_logs, "cluster_thin": n_logs < _MIN_LOGS}

    report = {
        "substrate": "AV2-Sensor val, REFERRED vehicle-longitudinal danger windows (av2_danger_logs.json)",
        "n_scenes": len({r["scene"] for r in rows}), "n_lead_frames": len(rows),
        "gates": {
            "surrogate_validity": {"pass": g1_pass, "closing_accel_corr": g1_corr},
            "predicate_correctness": {"pass": g2_pass, "agree_rate": g2_rate, "n_clean": g2_n,
                                      "agree_rate_wide": g2w_rate, "n_clean_wide": g2w_n,
                                      "note": "occ_gap vs box_gap within 0.4m on clean lead-frames; <0.70 => occ predicate fakes the delta => INVALID. _wide = same gate at the box-corridor footprint."},
            "shuffled_global": {"true_dd": float(np.mean([r["dd"] for r in rows])), "shuffled_dd": float(np.mean(shuf_dd)),
                                "true_dd_wide": float(np.mean([r["dd_wide"] for r in rows])), "shuffled_dd_wide": float(np.mean(shuf_dd_wide))},
        },
        "valid": bool(g1_pass and g2_pass),
        "action_delta_matrix": {k: {"true_mean": v["true"]["mean"], "true_ci": [v["true"]["lo"], v["true"]["hi"]],
                                    "shuffled_ci": [v["shuffled"]["lo"], v["shuffled"]["hi"]] if v["shuffled"]["defined"] else None,
                                    "n": v["true"]["n"], "n_logs": v["n_logs"], "cluster_thin": v["cluster_thin"],
                                    "verdict": v["verdict"], "verdict_wide_footprint": v["verdict_wide"]} for k, v in matrix.items()},
        "framing": "Q1 ACTION-SENSITIVITY (occ vs box gap through fixed IDM), oracle-free, on DANGER. NOT Q2 outcome / Q3 better.",
        "effective_power_note": "n is FRAMES; n_logs is independent clusters. cluster_thin (n_logs<3) cells are suggestive, not decisive -- frames within a log are autocorrelated. Decisive cells have the most logs.",
    }
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "r0_danger.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nR0-danger occ-vs-box action-sensitivity ({report['n_scenes']} logs, {len(rows)} lead-frames):\n")
    print(f"  GATE-1 surrogate-validity:   {'PASS' if g1_pass else 'FAIL'} (closing↑→accel↓ corr {g1_corr:+.2f}, need <-0.1)")
    print(f"  GATE-2 predicate-correctness:{'PASS' if g2_pass else 'FAIL'} (occ≈box within 0.4m on {g2_rate:.0%} of {g2_n} clean frames, need ≥70%; wide-footprint {g2w_rate:.0%} of {g2w_n})")
    if not (g1_pass and g2_pass):
        print("  -> a gate FAILED: this run is INVALID, not a result (see r0_danger_preregistration.md).")
    sg = report["gates"]["shuffled_global"]
    print(f"  shuffled (global): true {sg['true_dd']:.3f} vs shuffled {sg['shuffled_dd']:.3f}  | wide-footprint true {sg['true_dd_wide']:.3f} vs shuffled {sg['shuffled_dd_wide']:.3f}")
    print(f"\n  action-delta by (agent-context × static-urgency), true vs shuffled CI [n=frames, L=logs]:")
    for k, v in matrix.items():
        t, s = v["true"], v["shuffled"]
        sci = f"shuf[{s['lo']:.2f},{s['hi']:.2f}]" if s["defined"] else "shuf n/a"
        thin = "  CLUSTER-THIN(suggestive)" if v["cluster_thin"] else ""
        wide = "" if v["verdict_wide"] == v["verdict"] else f"  [wide-footprint: {v['verdict_wide']}]"
        print(f"    {k:30} {v['verdict']:13} true {t['mean']:.2f} CI[{t['lo']:.2f},{t['hi']:.2f}]  {sci}  (n={t['n']},L={v['n_logs']}){thin}{wide}")
    print(f"\n  wrote {_HERE / 'results' / 'r0_danger.json'}")
    print("  CHANGED = true CI clears shuffled band (the FIRST positive if a danger cell shows it). verdict invariant across both lateral footprints.")
    print("  EQUIVALENT in the well-clustered cells = the pre-registered NEGATIVE extends to danger. cluster-thin cells are suggestive. Q1 only; no outcome/better claim.")


if __name__ == "__main__":
    main()
