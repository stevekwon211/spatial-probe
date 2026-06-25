# R0 — occupancy-vs-box action-sensitivity: honest result (2026-06-25)

Pre-registrations: `r0_preregistration.md` (58e189a), `r0v2_preregistration.md` (8ca2530) — both sealed
before the corresponding run. This summary is the honest outcome; nothing below was tuned to pass.

## Verdict: INCONCLUSIVE on Q1 by this design — and the pre-registered gate exposed WHY (a real result).

R0 asked the oracle-free Q1: does swapping the spatial representation fed to a fixed IDM (occupancy
forward free-distance vs box lead distance) CHANGE the commanded action beyond a shuffled-occupancy null?

### What happened, in order
1. **R0-v1 — INVALID (gate caught a bug).** GATE-2 (predicate-correctness: occ_gap ≈ box_gap within one
   voxel on clean frames) = **0/145**. Debug-from-evidence (read the actual values, not a guess):
   `reachable_free_field`'s BEV forward extent `reach = ego.length/2 + ego.speed·horizon` collapses to
   ~2.4 m when the ego is stopped (most lead-frames have ego_speed ≈ 0), so occ_gap was clamped below the
   7–15 m lead. An instrument bug, not a finding. The pre-registered gate did its job.
2. **R0-v2 — instrument fixed, still INVALID, but the gate now exposes a DESIGN flaw, not a bug.** occ_gap
   rebuilt as the nearest occupancy obstacle SURFACE in the corridor up to `_LEAD_RANGE` (speed-independent);
   box_gap rebuilt as the box FRONT surface. GATE-2 rose 0% → **56%** (still < 70% → INVALID per the literal
   sealed gate). The diagnostic of the 44% disagreement is decisive:
   - occ FARTHER than box (occ misses the lead = a bug): **0%**.
   - occ NEARER than box (occ sees structure the box pipeline has no object for): **100%** of disagreements.
   - median (occ − box) = −0.20 m, mean −3.16 m, |Δ| p90 = 13.5 m.

### The real finding the gate exposed
Occupancy's forward free-distance is ≤ the box lead distance in **100%** of clean lead-frames (it NEVER
misses the tracked lead), and strictly nearer in 44% of them — occupancy denotes nearer free-space
structure the box pipeline cannot express. That is occquery **H1 (expressivity) shown on the
action-relevant gap quantity**, on real nuScenes-mini data, oracle-free.

But this is exactly why the action-sensitivity verdict is **inconclusive by this design**: the
predicate-correctness gate required occ ≈ box to validate the instrument, while the hypothesis (and the
data) is that occupancy sees MORE. The gate and the question are in tension — requiring agreement filters
out the very signal R0 wanted to test. The gap-swap-through-a-fixed-IDM-with-an-agreement-gate design is
structurally self-contradictory on a substrate where the two representations genuinely differ.

Separately, the by-regime matrix (set aside since the gate failed) trended EQUIVALENT/INDETERMINATE: even
the large occ-vs-box gap differences mostly did not clear the shuffled-occupancy band — because shuffling
occ_gaps across frames also produces large random deltas. So the pre-registered prior (likely
action-equivalent in the abundant safe regime; danger cell under-powered on nuScenes) is NOT contradicted.

### Honest standing
- Q1 (does the representation change the action) on this substrate via this design: **INCONCLUSIVE** —
  not because of a bug (v2 occ never misses the lead) but because the agreement-gated gap-swap cannot
  isolate the action effect when occupancy legitimately sees more.
- H1-consistent side-observation (real, oracle-free): occupancy's action-relevant forward free-distance
  is never farther than, and often nearer than, the box lead — it sees structure boxes do not.
- Ceiling unchanged: this is Q1-only; no Q2 (outcome) / Q3 (better) claim. "occupancy helps the planner"
  is prior art (OccNet, PKL); only the falsifiable shuffled-controlled framing was ours, and it returned
  inconclusive here.

### A v3 would need a NEW sealed pre-registration (do not edit v1/v2 to pass)
The corrected gate is "occ never MISSES the box lead (occ-farther rate ≈ 0 = no bug)" — which the v2 data
passes — and DROP the within-0.4 m agreement requirement (it contradicts the premise). Proposing+running
that now, after seeing it passes, is a forking path — so it is left as a sealed v3 step, not run here.
The deeper fix is to test action-sensitivity where occupancy-sees-more actually matters (a danger/cut-in
substrate), not on safe nuScenes following where dynfield already found the action redundant.
