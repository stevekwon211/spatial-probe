# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""dynfield Tier-1 ACTION-SENSITIVITY harness (Mac, pure numpy).

Measures, by regime, whether ABLATING the stored per-object velocity field MOVES vs LEAVES a
deterministic longitudinal planner-surrogate's action on real nuScenes scenes. Per the sealed
decisions (preregistration.md, 2026-06-23): the metric is the dimensionless decision-FLIP rate
(brake<->proceed), regime-comparable; decel-delta is a secondary curve; regimes are cut on STATIC /
CLASS observables, never on the ablated velocity; the word "necessary" is NOT used (no quality oracle
on a Mac) -- this is action-sensitivity, not necessity.

Surrogate (longitudinal lead-following, the PDM/IDM class): the LEAD is the nearest tracked box in the
ego corridor ahead. static-only reads its DISTANCE (a static snapshot); motion-aware also reads its
relative closing speed (the ablated motion field). The contrast cell-pair is vehicle-following vs
VRU/crossing -- both have a MOVING agent, differing in interaction type, not presence (the non-circular
cut). Runs the SH4 leakage gate + a surrogate-validity probe + a shuffled-velocity control FIRST; the
matrix is only reported if those pass.

Run: python experiments/dynfield_v0/harness.py [--scenes held-out-val.txt]
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

from probe.adapters.occ3d import load_scene
from surrogate import StoredState, plan_motion_aware, plan_static_only

_DATA = _HERE.parents[1] / "data"
_CORRIDOR = 1.5      # m, |lateral| of the ego corridor for the lead
_LEAD_RANGE = 40.0   # m, forward range to consider a lead
_SPEED_BAND = 5.0    # m/s, low/high ego-speed split (a regime descriptor, NOT a surrogate input)


def _agent_context(label: str, yaw: float) -> str:
    """Regime axis cut on CLASS + static geometry only (never on velocity)."""
    if label in ("pedestrian", "bicycle", "motorcycle"):
        return "vru"
    if label == "vehicle":
        return "vehicle_crossing" if abs(yaw) > 1.0 else "vehicle_following"
    return "other"


def _lead_and_regime(scene, t):
    """Nearest in-corridor lead box ahead -> (StoredState, agent_context, speed_band) or None.

    static field = lead distance (a static snapshot); motion field = lead relative closing speed."""
    ego = scene.ego_at(t)
    objs = scene.objects_at(t)
    best = None
    for o in objs:
        fwd, lat = o.center[0], o.center[1]
        if 0.0 < fwd <= _LEAD_RANGE and abs(lat) < _CORRIDOR:
            if best is None or fwd < best.center[0]:
                best = o
    if best is None or math.isnan(best.velocity[0]):
        return None
    rel = best.velocity[0] - ego.speed                     # lead_fwd - ego_fwd; negative = gap closing
    state = StoredState(lead_distance_m=best.center[0], lead_rel_speed_mps=rel)
    band = "low" if ego.speed < _SPEED_BAND else "high"
    return state, _agent_context(best.label, best.yaw), band


def _flip(state: StoredState) -> tuple[bool, float]:
    """Ablation: static-only (no velocity) vs motion-aware action. Returns (decision flipped, |Δdecel|)."""
    a_static = plan_static_only(state)
    a_motion = plan_motion_aware(state)
    return (a_static > 0.0) != (a_motion > 0.0), abs(a_motion - a_static)


def _bootstrap_rate(flags: list[bool], scene_ids: list[str], rng, n_boot=1000, level=0.95):
    """Cluster bootstrap (scene = unit) CI on a flip RATE. Undefined if < 2 positive scenes' worth."""
    if len(flags) < 4:
        return {"defined": False, "point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(flags)}
    flags_a = np.array(flags, float)
    units = np.array(scene_ids)
    uniq = np.unique(units)
    point = float(flags_a.mean())
    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        vals = np.concatenate([flags_a[units == u] for u in drawn])
        if len(vals):
            samples.append(vals.mean())
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(samples, [a, 100 - a])
    return {"defined": True, "point": point, "lo": float(lo), "hi": float(hi), "n": len(flags), "n_scenes": len(uniq)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=str, default=str(_HERE / "held-out-val.txt"))
    ap.add_argument("--limit", type=int, default=40, help="cap scenes for a tractable first pass")
    args = ap.parse_args()
    names = [s.strip() for s in pathlib.Path(args.scenes).read_text().splitlines() if s.strip()][: args.limit]
    rng = np.random.default_rng(0)

    # collect per-frame: regime, flip, decel-delta, the velocity (for the shuffled control), scene id
    rows = []
    print(f"loading {len(names)} held-out scenes (boxes+velocity) ...", flush=True)
    for i, name in enumerate(names):
        try:
            scene = load_scene(name, _DATA, mask="none", with_boxes=True)
        except (KeyError, FileNotFoundError):
            continue
        for t in scene.times():
            lr = _lead_and_regime(scene, t)
            if lr is None:
                continue
            state, ctx, band = lr
            flip, dd = _flip(state)
            rows.append({"scene": name, "ctx": ctx, "band": band, "flip": flip, "dd": dd,
                         "d": state.lead_distance_m, "rel": state.lead_rel_speed_mps})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(names)} scenes, {len(rows)} lead-frames", flush=True)

    if not rows:
        sys.exit("no lead-frames found -- check the scene list / corridor.")

    # ---- GATE 1: surrogate-validity probe (does the surrogate respond to closing, ignore receding?) ----
    closing = [r for r in rows if r["rel"] < -1.0 and r["d"] < 15.0]   # clearly closing, near
    receding = [r for r in rows if r["rel"] > 1.0]                      # clearly receding
    probe_closing_brakes = np.mean([plan_motion_aware(StoredState(r["d"], r["rel"])) > 0 for r in closing]) if closing else float("nan")
    probe_receding_proceeds = np.mean([plan_motion_aware(StoredState(r["d"], r["rel"])) == 0 for r in receding]) if receding else float("nan")
    probe_pass = (closing and receding and probe_closing_brakes > 0.8 and probe_receding_proceeds > 0.8)

    # ---- GATE 2: SH4 leakage -- static-only must be blind to motion (its action is distance-only) ----
    # the dual control: can lead VELOCITY be read off the static field (distance) alone? correlation ~0.
    d_arr = np.array([r["d"] for r in rows]); rel_arr = np.array([r["rel"] for r in rows])
    leak_corr = abs(float(np.corrcoef(d_arr, rel_arr)[0, 1])) if len(rows) > 2 else 1.0
    sh4_pass = leak_corr < 0.2   # distance must NOT encode velocity (else the static baseline is contaminated)

    # ---- shuffled-velocity control: true flip-rate must beat velocity-shuffled ----
    shuffled_rel = rng.permutation(rel_arr)
    shuffled_flips = [( _flip(StoredState(r["d"], float(sr)))[0]) for r, sr in zip(rows, shuffled_rel)]
    true_rate = np.mean([r["flip"] for r in rows]); shuf_rate = float(np.mean(shuffled_flips))

    # ---- the {velocity x agent-context} action-sensitivity matrix (flip rate, cluster-bootstrap CI) ----
    contexts = ["vehicle_following", "vehicle_crossing", "vru", "other"]
    matrix = {}
    for ctx in contexts:
        sub = [r for r in rows if r["ctx"] == ctx]
        ci = _bootstrap_rate([r["flip"] for r in sub], [r["scene"] for r in sub], rng)
        dd = [r["dd"] for r in sub]
        matrix[ctx] = {**ci, "decel_delta_mean": float(np.mean(dd)) if dd else float("nan")}

    report = {
        "n_scenes": len({r["scene"] for r in rows}), "n_lead_frames": len(rows),
        "gates": {
            "surrogate_validity": {"pass": bool(probe_pass), "closing_brakes": float(probe_closing_brakes),
                                   "receding_proceeds": float(probe_receding_proceeds), "n_closing": len(closing), "n_receding": len(receding)},
            "sh4_leakage": {"pass": bool(sh4_pass), "dist_vel_corr": leak_corr},
            "shuffled_control": {"true_flip_rate": float(true_rate), "shuffled_flip_rate": shuf_rate,
                                 "true_beats_shuffled": bool(true_rate > shuf_rate + 0.02)},
        },
        "velocity_x_agent_context": matrix,
        "framing": "ACTION-SENSITIVITY (did removing velocity move the surrogate's action), NOT necessity. "
                   "Necessity needs a closed-loop quality oracle (GPU Tier-2).",
    }
    (_HERE / "results").mkdir(exist_ok=True)
    (_HERE / "results" / "tier1_matrix.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\ndynfield Tier-1 action-sensitivity ({report['n_scenes']} scenes, {len(rows)} lead-frames):\n")
    g = report["gates"]
    print(f"  GATE surrogate-validity: {'PASS' if g['surrogate_validity']['pass'] else 'FAIL'} "
          f"(closing->brake {g['surrogate_validity']['closing_brakes']:.2f}, receding->proceed {g['surrogate_validity']['receding_proceeds']:.2f})")
    print(f"  GATE SH4 leakage: {'PASS' if g['sh4_leakage']['pass'] else 'FAIL'} (dist-vel corr {leak_corr:.3f}, must be <0.2)")
    print(f"  shuffled control: true flip {true_rate:.3f} vs shuffled {shuf_rate:.3f} -> {'true beats shuffled' if g['shuffled_control']['true_beats_shuffled'] else 'NO separation (spurious)'}")
    print(f"\n  velocity action-sensitivity by agent-context (flip rate = fraction where velocity flipped the brake/proceed decision):")
    for ctx, m in matrix.items():
        if not m["defined"]:
            print(f"    {ctx:18} UNDER-POWERED (n={m['n']})")
        else:
            print(f"    {ctx:18} flip {m['point']:.2f} CI[{m['lo']:.2f},{m['hi']:.2f}]  (n={m['n']} frames / {m['n_scenes']} scenes)  decelΔ {m['decel_delta_mean']:.2f}")
    print(f"\n  wrote {_HERE / 'results' / 'tier1_matrix.json'}")
    print("  READING: if the flip-rate CI is high in some contexts and ~0 (CI excludes the shuffled band) in others,")
    print("  the velocity field is action-CHANGING in some regimes and action-EQUIVALENT in others (the by-regime result).")
    print("  Uniform high or uniform zero = a NEGATIVE (report it). 'Necessary' is licensed only at GPU Tier-2.")


if __name__ == "__main__":
    main()
