# R0-v2 — occupancy-vs-box action-sensitivity, instrument fix (PRE-REGISTRATION, sealed before re-run 2026-06-25)

R0-v1 (r0_preregistration.md, committed 58e189a) RAN and its pre-registered GATE-2 (predicate-correctness)
FAILED: occ_gap agreed with box_gap within one voxel on **0 of 145** clean frames -> the run was declared
INVALID, not a result (as the protocol required). Debug-from-evidence (read the actual values, not a guess)
found the cause and it is recorded in `results/r0_action_sensitivity.json` (v1): the occupancy BEV forward
extent in `reachable_free_field` is `reach = ego.length/2 + ego.speed*horizon`, which **collapses to ~2.4 m
when the ego is stopped/slow** (most lead-frames have ego_speed ≈ 0). So occ_gap was clamped to the BEV
edge (1.7–4.9 m) and could never reach a lead at 7–15 m. That is an instrument bug baked into the v1
occ_gap definition, NOT a finding that occupancy disagrees with boxes.

This v2 fixes the INSTRUMENT only. The hypothesis (Q1), regime set, shuffled-occupancy control, statistic,
falsifiable kill criterion, and the Q1-only ceiling are **UNCHANGED from r0_preregistration.md** — re-read
that file; only the two clauses below change. Changing the instrument after an INVALID (gate-caught,
never-a-result) run is legitimate; the hypothesis is not touched, and this is committed before the re-run.

## CHANGE 1 — occ_gap (speed-independent corridor scan, surface reference)
`occ_gap` = the nearest occupancy obstacle SURFACE ahead of the ego, in the ego corridor, up to the lead
window — the direct, speed-independent analogue of the box lead distance:
- from `grid.obstacle_centers(max_height_agl=ego.height)` (the same occupancy obstacles the predicates use),
  projected to the ego frame (`ego.to_ego_frame`);
- keep obstacles with `0.5 < forward ≤ _LEAD_RANGE (40 m)` and `|lateral| < _CORRIDOR (1.5 m)` — the IDENTICAL
  window `harness._lead` uses for the box lead;
- `occ_gap = min(forward)` over those; if none, `occ_gap = _LEAD_RANGE` (clear to the lead window).
No `reachable_free_field`, no speed-dependent horizon. Horizon is no longer a parameter of occ_gap.

## CHANGE 2 — box_gap (front-surface, to match the reference)
For a fair comparison the two gaps must measure the same reference point. The box lead distance
`best.center[0]` is to the box CENTER; the occ_gap above is to the obstacle SURFACE. So box_gap becomes the
box FRONT-SURFACE distance: `box_gap = best.center[0] − best.size_length/2`. Both arms feed the IDM a
bumper-to-front surface gap. (The IDM's own gap semantics are surface/bumper anyway.)

## GATE-2 (now achievable, unchanged in spirit)
On CLEAN lead-frames (a box lead present AND occ finds an obstacle in the corridor near it, i.e. occ_gap <
_LEAD_RANGE), `occ_gap` must agree with the box FRONT-SURFACE `box_gap` within one voxel (≤ 0.4 m) for ≥ 70%
of clean frames. This validates that on frames where the lead IS a tracked box, the two representations
measure the same gap — so any action-delta elsewhere is a real representation difference (occupancy sees
structure with no box), not a reference/calibration offset. If it still fails, occ_gap is still not a valid
comparison and the run is INVALID, not a result.

## Everything else: UNCHANGED from r0_preregistration.md
Estimand `|IDM(occ_gap) − IDM(box_gap)|` via `plan_idm_static`; shuffled-occupancy null; regimes
(agent-context × static-urgency, urgency = ego²/2·box_gap cut at 1.5); scene-clustered paired bootstrap CI;
verdicts CHANGED/EQUIVALENT/INDETERMINATE; the falsifiable kill (EQUIVALENT-everywhere = the pre-registered
NEGATIVE headline; prior expectation = likely action-equivalent in the abundant safe regime, danger cell
under-powered on nuScenes); and the Q1-only ceiling (no outcome/better claim; "occupancy helps the planner"
is already prior art — only the falsifiable shuffled-controlled framing + a pre-registered negative is ours).
