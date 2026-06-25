# R0-danger — occupancy-vs-box action-sensitivity on a DANGER substrate: final result (2026-06-25)

Pre-registration sealed before the run: `r0_danger_preregistration.md` (commit 7022383), estimand
imported verbatim from the sealed `r0_action_sensitivity.py` so it cannot drift. Same Q1 (oracle-free):
does swapping the spatial representation fed to a fixed IDM (occupancy in-path forward free-distance vs
box lead distance) CHANGE the commanded action, beyond a shuffled-occupancy null, by a pre-named regime
set? Substrate = AV2-Sensor val REFERRED vehicle-longitudinal danger windows; the danger-frame selection
(`av2_danger_logs.json`) was sealed before the result. Nothing was tuned to pass. This result was then
attacked by two independent adversarial reviewers (voxelizer lens + statistics lens); neither could flip
it, both found one honest weakening, and both are folded in below.

## Verdict: the pre-registered NEGATIVE holds on danger — occupancy is action-EQUIVALENT to boxes (Q1), decisively in the well-clustered regimes, suggestively in the cluster-thin high-urgency danger cell.

Both pre-registered gates pass; EVERY regime cell returns EQUIVALENT — and the verdict is INVARIANT
across both lateral footprints (the ego in-path strip and the wider box corridor). The honest power
caveat (from adversarial review) is now explicit per cell: `n` is FRAMES, `L` is independent logs, and
frames within a log are autocorrelated, so a cell's real power is `L`, not `n`.

```
substrate : AV2-Sensor val, REFERRED vehicle-longitudinal danger windows — 14/18 sealed logs
            contributed an in-corridor lead (4 logs had no strict in-corridor lead; data property,
            not a bug), 1347 in-corridor lead-frames
GATE-1 surrogate-validity  : PASS (closing↑→accel↓ corr −0.73)
GATE-2 predicate-correctness: PASS (occ≈box within 0.4 m on 81% of 1301 clean frames; 76% under the
                              wider box-corridor footprint — both ≥70%)
shuffled (global)          : true 0.022 vs shuffled 0.384 m/s² | box-corridor footprint: true 0.066 vs 0.469
  cell                            verdict      true dd CI            shuffled CI       n     L   power
  vehicle_following|low_urgency   EQUIVALENT   0.01 [0.01,0.03]      [0.21,0.39]       1046  14  decisive
  vehicle_following|high_urgency  EQUIVALENT   0.02 [0.00,0.04]      [0.72,2.01]       78    3   suggestive*
  vehicle_crossing|low_urgency    EQUIVALENT   0.04 [0.00,0.11]      [0.04,0.36]       14    3   thin
  vru|low_urgency                 EQUIVALENT   0.09 [0.00,0.21]      [0.22,3.01]       29    4   moderate
  other|low_urgency               EQUIVALENT   0.04 [0.01,0.09]      [0.20,0.84]       158   5   decisive
  other|high_urgency              EQUIVALENT   0.09 [0.02,0.50]      [1.05,1.06]       20    2   thin
```
\* `vehicle_following|high_urgency`: n=78 frames but only L=3 logs, ~62% from a single log — better-powered
than R0-v3's n=4 (the cell that motivated this whole substrate) but NOT decisive. The three log-means are
{0.019, 0.004, 0.040} — all ~18× below the CHANGED bar (~0.72) and mutually consistent, so it is a
suggestive EQUIVALENT, not a powered one. The decisive evidence is `vehicle_following|low_urgency` (L=14)
and `other|low_urgency` (L=5).

Per the sealed kill criterion, a danger cell returning CHANGED where the safe cells did not would be the
first positive. None did, under either footprint. The "signal lives in danger" hypothesis is NOT rescued
by this substrate: the danger cells powered enough to read are EQUIVALENT; the highest-urgency cell is
suggestively EQUIVALENT and needs more independent logs to be decisive.

## Why it is a real negative, not a broken predicate or a verdict artifact (two adversaries, neither flipped it)
A RANDOM occ_gap (shuffled) moves the action a lot (0.2–2.0 m/s²); the TRUE occ_gap moves it ~0.02 — and
an HONEST small null (occ_gap jittered ±1 voxel) gives 0.0245, with the true delta 0.0216 sitting at
0.88× it. So occupancy is informative AND agrees with the box on the action; the small delta is agreement,
not a dead predicate (corr(occ,box)=0.93, slope 0.91, residual std 2.7 m, disagreement in BOTH directions).
The IDM clamp [−9,+3] is not hiding a difference: it bites the true pair in 0/1347 frames; removing it
leaves the global delta identical to 4 decimals. The verdict is seed-stable (EQUIVALENT in all 6 cells
across 20 RNG seeds; leave-one-log-out keeps the global delta in [0.017,0.024]). The honest masking test —
the 58 frames where occ disagrees with box (>0.4 m) AND the IDM is gap-sensitive (>0.05 m/s² per m), the
only place a hidden CHANGED could live — is itself EQUIVALENT (true 0.117 CI[0.067,0.208] vs shuffled
[0.284,0.908]).

## The integrity machinery fired again (debug-from-evidence + adversarial review)
1. **Voxelizer ego self-return — caught by reading the actual points before the run.** A first end-to-end
   check showed occ_gap = 0.6 m on no-lead frames; reading the points found exactly 2 returns at fwd≈0.52 m,
   z≈1.0 m (roof LiDAR seeing the ego body). Fixed with the standard AV2 ego-cuboid removal — verified safe
   here: min lead front-surface is 6.66 m, 0/1347 leads fall inside the removed cuboid.
2. **Regime mismatch — corrected to the seal, not to a result.** The first auto-picked log was a
   pedestrian-crosswalk (cross-path) scene; the pre-registration excludes cross-path (the IDM is a
   longitudinal lead model). Re-selected to vehicle-longitudinal REFERRED prompts — applying the sealed
   restriction.
3. **Frame-sharing verified by content.** AV2 LiDAR could be sensor- or ego-frame; a mismatch would offset
   occ_gap from box_gap. Confirmed ego-frame by content (residual mean −0.18 m, std 2.7 m — not a constant
   offset; ground peak z≈−0.38) — no calibration transform needed.
4. **Two adversarial weakenings folded in (post-result, neither flipped the verdict):**
   - *Effective N.* The bootstrap guards on ≥4 frames, not ≥k logs, so a cell can look "defined" on 1–2
     logs. Now every cell reports `L` (logs) + a `cluster_thin` flag, so the power claim is structurally
     visible — and the "POWERED" wording on the high-urgency cell was retracted (it is L=3, suggestive).
   - *Lateral-window fairness.* occ_gap scanned the narrow ego strip (|lat|<0.925 m) while box_gap drew
     its lead from the 1.5 m corridor. The run now computes occ at BOTH footprints; the verdict is
     EQUIVALENT under both and Gate-2 passes under both (81% / 76%), so "the occ strip was narrowed to
     fake agreement" is answered from the output.

## What this does and does not say
- **Says (Q1, oracle-free):** on real AV2 danger (vehicle-longitudinal conflict), occupancy's in-path
  forward free-distance agrees with the box lead distance (81%, or 76% at the wider footprint, within one
  voxel) and where it differs the difference does not change the clamped IDM action beyond a shuffled null
  — decisively in the 14-log and 5-log regimes, suggestively in the 3-log high-urgency danger cell.
- **The bounded ceiling (state plainly):**
  - AV2 boxes are NEAR-EXHAUSTIVE — every object is labeled — so this tests "when the box layer is
    complete, does occupancy add longitudinal action-value?" (no). It does NOT test "occupancy sees
    obstacles the box layer misses": this substrate has no unlabeled obstacles by construction, so the
    occupancy-sees-more advantage is structurally absent here, not measured-and-refuted.
  - The IDM is a longitudinal lead-follower; it does not probe LATERAL / free-space maneuvers, where
    occupancy's C-space reachability — not lead distance — would bear on the action.
  - Q1 only. No Q2 (outcome) / Q3 (better): those need a closed-loop collision/progress oracle.
- **Where the signal could still live (the clean next tests, strongest first):**
  1. **Unlabeled-obstacle injection.** Drop a fraction of boxes (simulate detection misses / the long
     tail) and re-run the SAME protocol — occupancy should then change the action exactly where the box
     layer goes blind. This directly probes occupancy's real value proposition, which AV2's complete labels
     hide. (Caveat to seal: GT-occ vs holed-GT-boxes is an UPPER BOUND — a fair head-to-head needs a
     predicted-occupancy model, which is the unbuilt CUDA/flow stage.)
  2. **Power top-up for the high-urgency cell.** Pull more REFERRED longitudinal danger logs landing in
     the high-urgency band to reach ≥10 independent logs there, making that cell decisive rather than
     suggestive. Sealed by the existing pre-registration (it cannot change the estimand, only tighten one cell).
  3. **Lateral free-space action.** Swap the longitudinal IDM for a steering / path-feasibility action —
     occupancy's advantage is free-space geometry, not the lead gap an IDM reads.

Together with R0-v3 (safe nuScenes, EQUIVALENT, high_urgency n=4) this is a clean, pre-registered,
adversarially-survived NEGATIVE across the abundant-safe and the readable-danger longitudinal regimes —
honest about where it is decisive vs suggestive, and it sharpens the next question rather than ending it.
