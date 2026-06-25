# Oracle-v0 — ego-trajectory traversal oracle for occupancy denotation-correctness (PRE-REGISTRATION, sealed before data 2026-06-25)

## Why
The premise verification (`docs/premise-verification-2026-06-25.md`) found the program's single biggest
exposure is the L5 "collapse" attack: *"a clearance predicate over an occupancy grid is a half-day
script; where is the contribution?"* The durable answer is denotation-CORRECTNESS — proving the
occupancy predicate returns the RIGHT answer — which requires an INDEPENDENT oracle (different data
source AND algorithm), the open "oracle problem." This seals the FIRST, cleanest such oracle before any
result is seen.

## The oracle (independence by construction)
The ego's RECORDED future trajectory is ground truth for "free": a vehicle cannot drive through a real
obstacle. So the ego's own swept volume over the next W frames marks space that was physically FREE at
frame t. This is independent of the occupancy perception in BOTH source (recorded motion / `city_SE3_
egovehicle` poses, not LiDAR returns) and algorithm (rigid-body sweep, not voxelization). No labels, no
boxes, no HD-map needed — the only inputs are the occupancy grid and the recorded ego poses.

## Estimand (sealed)
For each frame t in the AV2 danger logs (`av2_danger_logs.json`, the already-sealed substrate):
- Build the ego future swept volume: transform ego poses p(t+1 … t+W) into frame-t ego coordinates,
  sweep the ego footprint (length × width) along that path → the set of frame-t voxels the ego
  physically occupies within W frames.
- `traversal_fp_rate(t)` = (# occupancy-OCCUPIED voxels inside that swept volume) / (# voxels in the
  swept volume). A voxel occupied at t that the ego drives through within W frames is a FALSE POSITIVE
  (occupancy hallucinated an obstacle in the physically-driven path).
- Report the per-log-clustered mean FP rate with a scene-clustered bootstrap 95% CI.

W = 5 frames (0.5 s at 10 Hz) — short, to minimise the departed-dynamic-object confound (a voxel
occupied at t and driven through 0.5 s later is overwhelmingly a static false positive, not an object
that physically vacated that exact voxel in half a second).

## Pre-registered comparison (RELATIVE, not absolute)
Shuffled-occupancy null: randomly relocate the same count of occupied voxels within the grid extent;
recompute its `traversal_fp_rate`. The load-bearing claim is the RELATIVE gap: `true_fp << shuffled_fp`
means occupancy reliably keeps the physically-driven path clear (its obstacles sit where vehicles do
NOT go), validated with ZERO labels and independent of the box layer.

## Falsifiable kill (declared before data)
If `true_fp` is NOT clearly below the shuffled null (bootstrap CIs overlap), occupancy's "occupied"
verdicts are no better than random at avoiding the driven path → occupancy hallucinates obstacles → its
"blocked" denotation is untrustworthy → FAIL, and the collapse-attack rebuttal fails with it. A clean
negative is the headline, not a footnote.

## Honest scope / ceiling (declared)
- ONE-SIDED: measures occupancy FALSE POSITIVES (hallucinated obstacles) in the ego's driven ribbon
  only. It does NOT measure RECALL (real obstacles occupancy MISSES) — the harder half, deferred to
  oracle-v1 (camera cross-modal + multi-sweep).
- The ego ribbon is a thin slice of the scene (only where the ego actually drove); it cannot certify
  occupancy off-path.
- Genuinely label-free: boxes / annotations are NOT read. Inputs = the `av2_sensor` occupancy grid +
  recorded ego poses only.
- Static-world-over-W assumption; W kept small. A departed dynamic object at the exact voxel within
  0.5 s is possible but rare and inflates `true_fp` CONSERVATIVELY (against the hypothesis), so a passing
  result is not gamed by it.
- This is occupancy denotation-correctness (occquery H3-family), oracle-free of boxes; it is NOT a
  danger / safety claim.

## Next signals (named, not run)
oracle-v1 adds the camera cross-modal confirm (real AV2 ring cameras — independent SENSOR) and
multi-sweep persistence (independent OBSERVATION geometry), each with its OWN measured reliability
against a small human-labeled calibration set, combined only as far as their measured
failure-independence licenses. The contribution is the HONEST ACCOUNTING of how much a label-free
verdict can be trusted — not any single oracle.
