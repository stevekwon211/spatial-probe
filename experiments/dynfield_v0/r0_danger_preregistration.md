# R0-danger — action-sensitivity on a DANGER substrate (PRE-REGISTRATION, sealed before adapter results 2026-06-25)

R0-v3 returned a clean, valid, pre-registered NEGATIVE on SAFE nuScenes following (every regime
EQUIVALENT; high_urgency cell n=4, under-powered). The sealed conclusion: the action-change signal, if
any, lives in DANGER. This pre-registers the SAME R0 protocol on a danger substrate that actually has it.

Sealed BEFORE building the adapter produces any R0-danger result. The danger-density go/no-go gate
(zero-download, on-disk) already PASSED and is recorded: 22,846 distinct REFERRED danger frames across
150 AV2-Sensor val logs (~1038x nuScenes's ~22). Substrate choice is fixed below; results are not yet seen.

## Substrate (fixed)
Argoverse-2 SENSOR val split (150 logs), with the on-disk RefAV scenario-mining val feather
(`data/danger/av2_scenario_mining/scenario_mining_val_annotations.feather`, verified valid) as the
danger answer-key. DeepAccident (CARLA) is a SMOKE/STRESS test only — synthetic ≠ science, a positive
there is affirming-the-consequent (IDM and occupancy read the same simulator state), never the headline.
nuScenes-danger is rejected (relaxing the TTC cut on a benign-by-selection distribution is a forking path).

## Danger-frame selection (fixed)
From the val feather, filter `mining_category == 'REFERRED_OBJECT'`, group by `(log_id, prompt,
track_uuid)` into contiguous `timestamp_ns` windows. Restrict to LONGITUDINAL-conflict prompts (the lead
the IDM models): braking / closing / in-front / merge-cut-in / getting-closer. The REFERRED track is the
ego's lead for that window. (Cross-path/turning prompts are excluded — the IDM is a longitudinal lead model.)
The prompt taxonomy is named NOW from the 403 distinct prompts; do not re-pick prompts after seeing results.

## Estimand, gates, statistic, verdict — UNCHANGED from r0v3_preregistration.md
- `action_delta = | clamp(IDM(occ_gap), -9,+3) − clamp(IDM(box_gap), -9,+3) |`, `plan_idm_static`.
- `occ_gap` = nearest occupancy obstacle surface in the ego in-path strip (|lateral| < ego.width/2) up to
  `_LEAD_RANGE`, from the AV2 occupancy grid (voxelized from raw LiDAR by the new av2_sensor adapter).
- `box_gap` = the REFERRED lead box FRONT surface (center_x − length/2), ego frame (AV2 boxes are ego-frame).
- lead velocity = finite-difference of the REFERRED track's (tx,ty) across its frames.
- Shuffled-occupancy null; surrogate-validity (closing↑→accel↓ corr<−0.1) + predicate-correctness
  (occ≈box within one voxel ≥70% on clean frames) gates run FIRST; INVALID if either fails.
- Regimes: agent-context × static-urgency (ego²/2·box_gap, cut 1.5). The DANGER windows give the
  high_urgency cell real n (the whole point — vs nuScenes n=4). Scene-clustered (= per-log) bootstrap 95% CI.
- Verdict per cell: CHANGED (true CI clears shuffled) / EQUIVALENT / INDETERMINATE.

## Falsifiable kill + the headline this substrate can finally test
- If the danger-regime cells (now powered) return EQUIVALENT, the pre-registered NEGATIVE extends to
  danger: occupancy is action-equivalent to boxes even in conflict — a strong, falsifiable negative.
- If a danger cell returns CHANGED (true action-delta clears the shuffled band) where the safe cells did
  not, that is the FIRST positive: the representation changes the action specifically when it is dangerous
  — the result R0 was built to reach, on real sensor data.

## Ceiling (unchanged, state in any report)
Q1 only (does the representation change the ACTION), oracle-free. NOT Q2 (outcome) / Q3 (better) — those
need closed-loop sim + a collision/progress oracle. AV2 occupancy is voxelized from a single 10 Hz sweep;
if Gate-2 fails on single sweeps, aggregate k≈5 sweeps (a sealed fallback, ~20 LOC) and re-run — but that
is an instrument fix, sealed here, not a result change. "occupancy helps the planner" is prior art; only
the falsifiable, shuffled-controlled, pre-registered danger-vs-safe contrast is ours.
