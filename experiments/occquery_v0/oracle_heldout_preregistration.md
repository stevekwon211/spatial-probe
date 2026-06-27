# Oracle-v0.1 — clean held-out confirmatory of the ego-traversal oracle (substrate decision, sealed before re-running the oracle, 2026-06-25)

## Why this exists
`results/oracle_traversal_summary.md` reported the ego-trajectory traversal oracle as a real,
independent, label-free signal (occupancy puts an obstacle in the physically-driven road ≤1.4% of the
time vs 3.3% random) — but BOTH sealed window choices were confounded on the stop-and-go FOLLOWING
substrate (the 18 logs in `dynfield_v0/av2_danger_logs.json`):
- W small (5 frames = 0.5 s): ego creeps ~0.9 m, the swept ribbon sits ENTIRELY inside the ego
  self-return-removal zone (x < 3.9 m) where occupancy is 0 by construction → the 0.00 FP is vacuous.
- W large (40 frames = 4 s): the lead vehicle MOVES, the ego drives into space the lead VACATED → that
  correctly-occupied-at-t cell is counted as an apparent false positive → inflates FP against occupancy.

W=40 / far=4 were chosen AFTER seeing the result. This note seals the clean confirmatory's substrate
and parameters BEFORE re-running, so the choice cannot be HARKed.

## The decision: substrate = (b) general AV2 val FREE-DRIVING logs, NOT (a) held-out following logs 19–30
Decided from a discriminating measurement on all 150 AV2-Sensor val logs (the RefAV mining val feather
gives ego-frame boxes; `city_SE3_egovehicle.feather` poses give ego motion — both downloaded for all 150,
~29 MB, scratch `data/danger/_poses_val/`). Two confound metrics at the principled W=10 frames (1.0 s,
the minimum to clear the ego self-return zone given the substrate's creep speed):
- `escape_frac` = fraction of W-windows whose ego displacement > 3.9 m (ribbon reaches real road).
- `close_lead_frac` = fraction of frames with an in-corridor vehicle (|lat|<2.5 m) within 15 m ahead of
  the front bumper (the moving-lead-that-vacates-space confound).

| substrate | escape_frac (median) | close_lead_frac (median) | ego disp/1 s |
|---|---|---|---|
| 18 used following | 0.65 | 0.05 — but 8/18 logs ≥ 0.24, up to **0.87** (`0b5142c1`) | 5.65 m |
| free-driving (option b) | **1.00** | **0.00** | 7–15 m |

Option (a) only moves to FRESH logs of the SAME confounded type: held-out following logs inherit the
mix where some logs are trapped behind a close lead 24–87 % of frames, and at small W the ribbon is still
trapped in the near field. Option (b) removes BOTH confounds at once and by construction: a free-driving
ego ALWAYS clears the self-return zone (escape=1.00) and has NO close lead to vacate occupied space
(close_lead=0.00). It is the cleanest fix the original summary itself named (line 28). The metric was
validated against the downloaded ground-truth ego-frame `annotations.feather` (890/900 shared tracks
matched within 0.5 m) after a first apparatus bug (poses are 197 Hz, not 10 Hz; and the mining-feather
boxes are already ego-frame — a city-frame transform was wrong) was caught and fixed.

## Sealed parameters for the re-run (principled, not result-selected)
- **W = 10 frames (1.0 s @ 10 Hz)** — derived as the minimum lookahead for the free-driving ego to clear
  the ego self-return zone (x > 3.9 m), NOT the post-hoc W=40.
- **far = ego front bumper, x > 3.9 m** (`_EGO_X1` in `av2_sensor.py:41`) — mechanism-derived (the zone
  occupancy zeroes out by construction), NOT the post-hoc far=4.
- **Declared confound (up front):** even on free-driving, a departed dynamic CROSS-traffic object at the
  exact voxel within 1 s is possible but rare and inflates `true_fp` CONSERVATIVELY (against occupancy),
  so a passing result is a conservative upper bound, never gamed by it.
- **Estimand / null / verdict UNCHANGED** from `oracle_traversal_preregistration.md`: occupancy
  false-positive rate inside the ego's physically-driven ribbon vs a shuffled-occupancy null;
  per-log-clustered bootstrap 95 % CI; RELIABLE = true CI strictly below shuffled CI; one-sided (FP only,
  recall = oracle-v1). Kill: true_fp not clearly below shuffled.

## Held-out logs (8, speed-stratified 7.3–15.0 m/s, NONE among the 18)
Written to `oracle_heldout_logs.json`. Selected from the 18 cleanest free-driving logs (escape ≥ 0.99,
close_lead = 0, mid_lead ≤ 0.10, not in the 18) by an even spread across ego speed for motion diversity.
This is FRESH data (different logs) AND a cleaner substrate — both integrity asks (re-test exploratory
findings on held-out data; switch off the confounded substrate) satisfied at once.

## Ceiling (restate in any report)
Tests occupancy FALSE POSITIVES only; not RECALL (oracle-v1: camera cross-modal + multi-sweep). The ego
ribbon is a thin slice (only where the ego drove); it cannot certify occupancy off-path. Genuinely
label-free: boxes/annotations are NOT read by the oracle (annotations.feather is downloaded only for the
voxelizer's box layer and was used here ONLY to validate the confound metric and pick the substrate, not
in the FP estimand). Occquery H3-family denotation-correctness, not a danger/safety claim.
