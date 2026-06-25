# R0-v3 — root-cause instrument fix (PRE-REGISTRATION, sealed before re-run 2026-06-25)

An adversarial review of R0-v2 (committed 6949166) found my v2 INTERPRETATION overclaimed, and identified
the ROOT cause of GATE-2's 56% — not an irreducible occ-vs-box tension, but a fixable instrument bug:
- **corridor-width artifact.** `_CORRIDOR = 1.5 m` (a 3 m strip) was inherited from `harness._lead`, where
  it selects sparse tracked-vehicle leads. Applied to DENSE occupancy as a `min`-over-corridor, it latches
  the nearest roadside/SHOULDER voxel, not a dead-ahead lead: 82.8% of the v2 "occ-nearer" hits sit at
  |lateral| > 1.0 m (median 1.40 m). The wide strip STRUCTURALLY forces occ ≤ box (you cannot have the
  min-over-corridor be farther than a lead also inside the strip), which manufactured the "0% farther."
  With a physically-correct in-path strip (|lateral| < ego.width/2 ≈ 0.92 m) the review measured GATE-2 →
  76% (PASSES) and occ-farther → 8.5% (a normal two-sided disagreement).
- **unclamped IDM.** `surrogate.plan_idm_static` is not clamped to physical limits; at a sub-meter shoulder
  "gap" it returns ~-174 m/s² (≈18 g), which dominates the action-delta matrix with unphysical blowups.

This v3 fixes the INSTRUMENT at the root (no-band-aid). Hypothesis (Q1), regimes, shuffled-occupancy
control, statistic, falsifiable kill, and the Q1-only ceiling are UNCHANGED from r0_preregistration.md.
Disclosure (forking-path honesty): the review already showed |lat| < 0.95 m passes the gate, so the gate
PASSING is expected; the NEW information v3 yields is the by-regime action verdicts (CHANGED/EQUIVALENT),
which the corridor choice does not pre-determine. The corridor `ego.width/2` is the ego's own collision
strip — the uniquely physical choice for a longitudinal lead gap, not a value tuned until it passed.

## CHANGE 1 — occ_gap scans the EGO IN-PATH strip, not the box corridor
In `_occ_forward_gap`, the lateral bound becomes `|lateral| < ego.width / 2.0` (the strip the ego itself
sweeps going straight) instead of `_CORRIDOR (1.5)`. box_gap (nearest tracked lead front-surface in the
1.5 m lane corridor) is unchanged; the residual ~8.5% occ-farther = laterally-offset leads outside the
ego's exact path, an honest two-sided disagreement.

## CHANGE 2 — clamp the IDM action to physical limits before differencing
`action = clamp(plan_idm_static(ego_speed, gap), -9.0, +3.0)` m/s² (standard AV comfort/emergency band)
in BOTH arms before `|occ − box|`. Removes the sub-meter-gap blowups so the action-delta is an action,
not an artifact.

## GATE-2 (unchanged definition; now expected to pass on the corrected strip)
occ_gap ≈ box_gap front-surface within one voxel (0.4 m) on clean frames (occ found an in-path obstacle),
≥ 70%. If it still fails, occ_gap is still not a valid comparison → INVALID.

## Everything else: UNCHANGED from r0_preregistration.md
Estimand `|clamp(IDM(occ_gap)) − clamp(IDM(box_gap))|`; shuffled-occupancy null; regimes (agent-context ×
static-urgency = ego²/2·box_gap cut at 1.5); scene-clustered paired bootstrap CI; CHANGED/EQUIVALENT/
INDETERMINATE; falsifiable kill (EQUIVALENT-everywhere = the pre-registered NEGATIVE headline; prior
expectation = likely action-equivalent in the abundant safe regime, danger cell under-powered on nuScenes);
Q1-only ceiling (no outcome/better; "occupancy helps the planner" is prior art — only the falsifiable
shuffled-controlled framing is ours). The corrected side-finding scope: occupancy's in-path forward
free-distance vs the box lead — NOT "structure boxes cannot express" globally (the review showed ~31% of
v2 disagreements were boxed-elsewhere); report only what the in-path comparison supports.
