# R0 — occupancy-vs-box action-sensitivity (PRE-REGISTRATION, sealed before data 2026-06-25)

Sealed BEFORE implementing the experiment or looking at any R0 output. Changing anything below after
seeing data is HARKing and shows in the git diff. This is the laptop, oracle-free **Q1** rung of the
action-sensitivity ladder (see the reframe: Q1 action-change = oracle-free; Q2 outcome / Q3 better =
closed-loop + GPU + danger substrate, NOT in scope here).

## Question (Q1)
Does swapping the spatial REPRESENTATION fed to a fixed planner — dense occupancy free-space vs the
object-box distance — **change the planner's action**, in at least one pre-named regime, beyond
shuffled-occupancy noise? This is the action-sensitivity analogue of occquery H1 (expressivity): H1
says occupancy *denotes* free-space a box language cannot; R0 asks whether that distinction *moves the
action*. It is oracle-free: it compares two actions, it does not consult a ground-truth answer key.

## Estimand (frozen)
Per in-corridor lead-frame: `action_delta = | plan_idm_static(ego_speed, occ_gap) − plan_idm_static(ego_speed, box_gap) |`.
- Same fixed IDM transfer function for both arms (`surrogate.plan_idm_static`); ONLY the gap input swaps. The IDM is a transfer function, not a judge.
- `box_gap` = nearest in-corridor lead-box forward distance — the existing `harness._lead` / harness_v2 `_lead()[0]` (= `best.center[0]`).
- `occ_gap` = occupancy forward free-distance along the ego centerline: from `reachable_free_field(grid_at(t), ego_at(t), horizon)`, scan lateral≈0 outward from forward=0 and take the forward distance at which `reachable` first becomes False (the centerline leaving the reachable region = blocked ahead), capped at the BEV horizon. Horizon = the dynfield default used for the lead window.
- Unit / scene resampling unit = the scene (scene-clustered).

## Baseline / control (frozen)
Shuffled-occupancy null: permute `occ_gap` across all lead-frames (`rng.permutation`), recompute
`action_delta` with the permuted occ-gap. The TRUE action-delta must beat the shuffled-occupancy
action-delta. The headline is the RELATIVE gap (true CI vs shuffled CI), never an absolute action magnitude.

## Regimes (named NOW — anti-HARK)
`agent-context ∈ {vehicle_following, vehicle_crossing, vru, other}` (via `harness._agent_context`)
× `static-urgency ∈ {low, high}` cut at `ego_speed² / (2·gap) = 1.5 m/s²` (ego state + gap only, NOT
the swapped representation — velocity-independent, non-circular; identical to dynfield v2). The regime
set, the urgency cut (1.5), and the corridor/lead window are FROZEN — they will NOT be tuned to
manufacture a non-equivalent cell.

## Gates (run FIRST, before any verdict)
1. **Surrogate-validity** (same as v2): the IDM must brake harder as closing speed rises —
   `corr(closing, plan_idm_motion) < −0.1`. If FAIL, the surrogate is invalid and there is no result.
2. **Predicate-correctness** (the load-bearing integrity gate): on CLEAN lead-frames (a box lead present
   AND the occ centerline unobstructed up to that lead), `occ_gap` must agree with `box_gap` within one
   voxel (≤ 0.4 m) for ≥ 70% of clean frames. If it does not, the occupancy predicate's known
   false-positive modes (reachable.py docstring: frontal-as-corridor, far-wall-as-clearance,
   lone-voxel-as-blockage) are manufacturing a spurious delta → the run is INVALID, not a result.
   Report the agreement rate regardless.

## Statistic & verdict (frozen)
Scene-clustered paired bootstrap, 95% CI (n_boot=1000), reusing harness_v2 `_boot_mean`. Per regime cell:
`CHANGED` if true-CI.lo > shuffled-CI.hi; `EQUIVALENT` if true-CI.hi ≤ shuffled-CI.hi; else `INDETERMINATE`.

## Falsifiable kill criterion (reachable)
If EVERY regime cell with n ≥ 4 returns `EQUIVALENT` (true action-delta CI inside the shuffled band
everywhere), the "occupancy changes the action vs boxes (through this IDM)" thesis is **FALSIFIED on
this substrate** — and that NEGATIVE is the headline, not a footnote. Prior expectation (stated before
data, not a prediction to confirm): dynfield already found the velocity field action-REDUNDANT in safe
following (n=443 EQUIVALENT); occupancy may likewise be action-equivalent to boxes in the abundant safe
regime, with the high-urgency/danger cell under-powered on nuScenes (~22/2114 frames at TTC<2s). A
non-uniform matrix (CHANGED in some named regime) would be the positive result.

## Ceiling (state explicitly in any report)
R0 measures **Q1 only** (does the representation change the ACTION) — oracle-free. It does NOT measure
Q2 (does it change the OUTCOME) or Q3 (does it change it for the BETTER), which require a closed-loop
collision/progress oracle on a danger-bearing substrate (nuPlan/PDM-Closed, GPU) that this experiment
lacks. No "necessary" / "better" / "occupancy beats boxes" claim is licensed by R0. Prior art note:
"occupancy helps the planner" is already shown empirically (OccNet Table 7, PKL); R0's only unoccupied
contribution is the falsifiable, shuffled-controlled, pre-registered action-sensitivity framing — and a
pre-registered NEGATIVE if that is what the data gives.

## Analysis path (frozen)
load held-out-val scenes (`held-out-val.txt`, --limit default 60) → per lead-frame compute box_gap,
occ_gap, action_delta, regime, urgency, closing → run both gates → shuffled-occupancy control →
by-regime matrix with bootstrap CI → verdicts → write `results/r0_action_sensitivity.json` + print.
No step is added or removed after seeing output.
