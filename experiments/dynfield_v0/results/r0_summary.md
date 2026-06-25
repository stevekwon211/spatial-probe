# R0 — occupancy-vs-box action-sensitivity: final result (2026-06-25)

Pre-registrations, each sealed before its run: `r0_preregistration.md` (58e189a), `r0v2_preregistration.md`
(8ca2530), `r0v3_preregistration.md` (commit before this run). Q1 (oracle-free): does swapping the spatial
representation fed to a fixed IDM (occupancy forward free-distance vs box lead distance) CHANGE the
commanded action, beyond a shuffled-occupancy null, by a pre-named regime set? Nothing below was tuned to pass.

## Verdict: pre-registered NEGATIVE — occupancy is action-EQUIVALENT to boxes here (Q1, safe nuScenes following).

R0-v3 is VALID (both pre-registered gates pass) and EVERY regime cell returns EQUIVALENT: the occ-vs-box
representation swap does not move the clamped IDM action beyond shuffled-occupancy noise, in any tested
regime, on the held-out nuScenes val set (50 scenes, 672 in-corridor lead-frames).

```
GATE-1 surrogate-validity  : PASS (closing↑→accel↓ corr −0.51)
GATE-2 predicate-correctness: PASS (occ ≈ box within 0.4 m on 80% of 655 clean frames, need ≥70%)
shuffled (global)          : true action-delta 0.06  vs shuffled 0.58  m/s²
  vehicle_following|low_urgency  EQUIVALENT  true 0.01 CI[0.00,0.01]  shuf[0.36,0.83]  n=440
  vehicle_crossing|low_urgency   EQUIVALENT  true 0.29 CI[0.02,0.72]  shuf[0.23,0.87]  n=60
  vru|low_urgency                EQUIVALENT  true 0.08 CI[0.02,0.20]  shuf[0.21,0.68]  n=85
  other|low_urgency              EQUIVALENT  true 0.10 CI[0.02,0.19]  shuf[0.39,0.91]  n=78
  other|high_urgency             EQUIVALENT  true 0.61 CI[0.02,0.81]  shuf[1.61,5.34]  n=4  (under-powered)
```
Per the sealed kill criterion, EQUIVALENT-everywhere is the pre-registered NEGATIVE headline, not a footnote.

## How we got here — the integrity machinery fired three times
1. **v1 INVALID — a gate caught a bug.** GATE-2 = 0/145: occ_gap was clamped to ~2.4 m because
   `reachable_free_field`'s BEV forward extent collapses when the ego is stopped. Debug-from-evidence (read
   the values) found it. The pre-registered gate did its job.
2. **v2 INVALID — adversarial review caught my OVERCLAIM.** With occ_gap rebuilt (surface, corridor scan)
   GATE-2 rose to 56%. I wrote that the gate "contradicts the hypothesis" (a structural occ-vs-box tension).
   An adversarial re-review showed that was wrong: the 56% was a **corridor-width artifact** — `_CORRIDOR =
   1.5 m` (inherited from the sparse-box lead selector) applied to a dense-occupancy `min`-scan latched
   roadside SHOULDER voxels (82.8% of "occ-nearer" hits at |lateral| > 1.0 m). The wide strip structurally
   forces occ ≤ box, manufacturing the "0% farther / sees structure boxes can't express" story. Root cause:
   reused a box corridor for an occupancy scan, plus an unclamped IDM (sub-meter "gaps" → ~−174 m/s²). Not a
   structural tension — a fixable instrument bug. "occupancy never misses the lead" and "structure boxes
   cannot express" were over-stated (the genuine no-box share was ~69%, not 100%).
3. **v3 VALID — root-cause fix, no tuning.** occ_gap scans the ego in-path strip (|lateral| < ego.width/2,
   the ego's own collision width), and the IDM action is clamped to [−9,+3] m/s² before differencing. Both
   sealed before the run. GATE-2 → 80% (PASS), occ-farther → a normal two-sided rate. Result above: a clean
   EQUIVALENT-everywhere negative.

## What this does and does not say
- **Says (Q1, oracle-free):** on safe nuScenes vehicle-following, occupancy's in-path forward free-distance
  agrees with the box lead distance (80% within one voxel) and, where it differs, the difference does not
  change the IDM action beyond noise. The representation swap is action-equivalent here.
- **Consistent with:** dynfield (velocity action-redundant when safe), and the reframe — the action-change
  signal, if any, lives in DANGER (cut-ins, near-miss), which nuScenes barely contains (high_urgency n=4,
  under-powered). This negative is on the abundant safe regime, exactly where redundancy is expected.
- **Does NOT say:** anything about Q2 (outcome) or Q3 (better) — those need a closed-loop collision/progress
  oracle on a danger substrate (GPU + nuPlan/PDM-Closed + a danger-bearing dataset), which this lacks.
  "occupancy helps the planner" is prior art (OccNet, PKL); our only contribution is the falsifiable,
  shuffled-controlled, pre-registered framing — which returned a clean, honest negative here.
- **Next (where the signal could live):** re-run on a danger-bearing 3D substrate (Argoverse-2
  scenario-mining cut-ins / near-collisions — labels already downloaded) where occupancy-sees-more actually
  bears on the action, with the same sealed protocol.
