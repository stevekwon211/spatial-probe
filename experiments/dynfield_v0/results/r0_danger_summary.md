# R0-danger — occupancy-vs-box action-sensitivity on a DANGER substrate: final result (2026-06-25)

Pre-registration sealed before the run: `r0_danger_preregistration.md` (commit 7022383, with the
estimand imported verbatim from the sealed `r0_action_sensitivity.py` so it cannot drift). Same Q1
(oracle-free): does swapping the spatial representation fed to a fixed IDM (occupancy in-path forward
free-distance vs box lead distance) CHANGE the commanded action, beyond a shuffled-occupancy null, by a
pre-named regime set? Substrate = AV2-Sensor val REFERRED vehicle-longitudinal danger windows. The
danger-frame selection (`av2_danger_logs.json`) was sealed before the result. Nothing was tuned to pass.

## Verdict: the pre-registered NEGATIVE EXTENDS to danger — occupancy is action-EQUIVALENT to boxes (Q1), now with the high-urgency cell POWERED.

R0-v3 returned EQUIVALENT-everywhere on safe nuScenes following but the high_urgency cell was n=4
(under-powered), so the sealed conclusion was "the action-change signal, if any, lives in danger." This
runs the IDENTICAL protocol on real AV2 danger frames where that cell finally has power. It does not
appear. Both pre-registered gates pass; EVERY regime cell — including `vehicle_following|high_urgency`
at **n=78** — returns EQUIVALENT.

```
substrate : AV2-Sensor val, REFERRED vehicle-longitudinal danger windows — 14/18 sealed logs
            contributed an in-corridor lead (4 logs had no strict in-corridor lead; data property,
            not a bug), 1347 in-corridor lead-frames
GATE-1 surrogate-validity  : PASS (closing↑→accel↓ corr −0.73)
GATE-2 predicate-correctness: PASS (occ ≈ box within 0.4 m on 81% of 1301 clean frames, need ≥70%)
shuffled (global)          : true action-delta 0.022  vs shuffled 0.384  m/s²
  vehicle_following|low_urgency   EQUIVALENT  true 0.01 CI[0.01,0.03]  shuf[0.21,0.40]  n=1046
  vehicle_following|high_urgency  EQUIVALENT  true 0.02 CI[0.00,0.04]  shuf[0.72,2.01]  n=78   (POWERED)
  vehicle_crossing|low_urgency    EQUIVALENT  true 0.04 CI[0.00,0.11]  shuf[0.04,0.36]  n=14
  vru|low_urgency                 EQUIVALENT  true 0.09 CI[0.00,0.24]  shuf[0.22,3.00]  n=29
  other|low_urgency               EQUIVALENT  true 0.04 CI[0.01,0.09]  shuf[0.19,0.87]  n=158
  other|high_urgency              EQUIVALENT  true 0.09 CI[0.02,0.50]  shuf[1.05,1.06]  n=20
```
The sealed kill criterion was: a danger cell returning CHANGED (true CI clears shuffled) where the safe
cells did not = the first positive. None did. EQUIVALENT-everywhere on the powered danger regime is the
pre-registered NEGATIVE, extended — the headline, not a footnote. The "signal lives in danger"
hypothesis is FALSIFIED for the longitudinal lead-following action.

## Why it is a real negative, not a broken predicate (the adversary is me)
A RANDOM occ_gap (shuffled) moves the action a lot (delta 0.2–2.0 m/s²); the TRUE occ_gap moves it ~0.02.
So occupancy is NOT noise — it is informative AND it agrees with the box on the action. Gate-2 (81%
within one voxel) shows occ_gap genuinely tracks box_gap, so the small delta is agreement, not a dead
predicate. The IDM clamp [−9,+3] is not hiding a difference either: the shuffled null is computed under
the SAME clamp and is large, so it is the occ≈box agreement — not clamping — that zeroes the true delta.

## The integrity machinery fired again (debug-from-evidence, before the full run)
1. **Voxelizer ego self-return — caught by reading the actual points.** A first end-to-end check showed
   occ_gap = 0.6 m on no-lead frames. Reading the points: exactly 2 returns at fwd≈0.52 m, z≈1.0 m (roof
   LiDAR seeing the ego body). Fixed with the standard AV2 ego-cuboid removal (a real obstacle sits beyond
   the front bumper, so the removal cannot drop one) BEFORE any result — not after seeing the verdict.
2. **Regime mismatch — caught and corrected to the seal.** The first auto-picked log was a
   pedestrian-crosswalk (cross-path) scene; the pre-registration explicitly excludes cross-path (the IDM is
   a longitudinal lead model). Re-selected to vehicle-longitudinal REFERRED prompts — applying the sealed
   restriction, not tuning to a result.
3. **Frame-sharing verified by content, not assumed.** AV2 LiDAR could be in the sensor or the ego frame;
   a mismatch would offset occ_gap from box_gap. Confirmed ego-frame by content (occ_gap tracks box_gap to
   <0.4 m, ground peak at z≈−0.35) — no calibration transform needed.

## What this does and does not say
- **Says (Q1, oracle-free):** on real AV2 danger (vehicle-longitudinal conflict, including high-urgency
  near-misses, n=78), occupancy's in-path forward free-distance agrees with the box lead distance (81%
  within one voxel) and, where it differs, the difference does not change the clamped IDM action beyond a
  shuffled null. The representation swap is action-equivalent — even in danger.
- **The bounded ceiling (state plainly):**
  - AV2 boxes are NEAR-EXHAUSTIVE — every object is labeled — so this tests "when the box layer is
    complete, does occupancy add longitudinal action-value?" (no). It does NOT test "occupancy sees
    obstacles the box layer misses," because this substrate has no unlabeled obstacles by construction. The
    occupancy-sees-more advantage is structurally absent here, not measured-and-refuted.
  - The IDM is a longitudinal lead-follower. It does not probe LATERAL / free-space maneuvers, where
    occupancy's C-space reachability — not lead distance — is the thing that would bear on the action.
  - Q1 only. No Q2 (outcome) / Q3 (better): those need a closed-loop collision/progress oracle.
- **Where the signal could still live (the two clean next tests):**
  1. **Unlabeled-obstacle injection.** Drop a fraction of boxes (simulate detection misses / the long
     tail) and re-run the SAME protocol — occupancy should then change the action exactly where the box
     layer goes blind. This directly probes occupancy's real value proposition, which AV2's complete labels
     hide.
  2. **Lateral free-space action.** Swap the longitudinal IDM for a steering / path-feasibility action and
     test occ-vs-box there — occupancy's advantage is free-space geometry, not the lead gap an IDM reads.

Together with R0-v3 (safe nuScenes, EQUIVALENT, high_urgency n=4) this is a clean, pre-registered,
falsifiable NEGATIVE across both the abundant-safe and the powered-danger longitudinal regimes — and it
sharpens the next question rather than ending it.
