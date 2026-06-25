# Oracle-v3 — frozen mono-depth (DAv2) cross-modal RECALL oracle (PRE-REGISTRATION, seal before data 2026-06-26)

The decisive experiment of the PRISM+alpha re-plan: can an EXTERNAL, label-free, genuinely-independent
cross-modal recall oracle be built solo/CPU? It forks the killed stereo oracle
(`oracle_stereo_recall_preregistration.md`, ORACLE-INSUFFICIENT at AUC 0.259) and replaces ONLY the
depth-production stage — classical block-matching → a FROZEN monocular metric-depth net — keeping every
other stage (undistort, back-project, voxelize-with-same-filters, band∩FOV, band-local null, AUC gate,
log-clustered bootstrap, kill rule) identical. Nothing below was chosen after seeing a v3 number.

## Why this is the linchpin
The traversal oracle gives a defensible FALSE-POSITIVE check (RELIABLE). The recall half — real obstacles
occupancy MISSES — has no external oracle: box-recall is same-modality (consistency), stereo died at its
gate. Recall is the safety-load-bearing half AND the moat half PRISM/Visionary doesn't publish. If a
frozen-depth recall oracle clears its gates here, denotation-honesty becomes a two-sided, externally-
validated artifact on data already on disk. If it fails, recall stays consistency-only and we consolidate
to FP-only — a legitimate, pre-registered outcome (the A-vs-consolidate fork).

## Model (frozen, verified on disk before seal)
`data/models/dav2_metric_vkitti_vitl/depth_anything_v2_metric_vkitti_vitl.onnx` (Depth-Anything-V2-Metric,
Virtual-KITTI weights, exported to ONNX). Verified: onnxruntime CPU, input `[1,3,518,518]` ImageNet-norm,
output `[1,518,518]` in METERS (max_depth=80 m, sigmoid·80 baked in — NO external scale/intrinsics).
~1.7 s/image CPU. **Provenance: Virtual-KITTI (synthetic), NOT Argoverse2/nuScenes/real-KITTI** → clean
for grading AV2 LiDAR (verified against the official metric_depth README: outdoor metric = Virtual-KITTI).
(Metric3Dv2/UniDepth were rejected — they train on Argoverse2 = circular.)

## Independence ledger (vs the av2_sensor LiDAR voxelization being graded)
| axis | occupancy (graded) | depth oracle (grader) | independent? |
|---|---|---|---|
| modality | active TOF LiDAR | passive RGB monocular | YES |
| algorithm | point→voxel (`_voxelize`) | learned depth net → back-project → voxel | YES |
| provenance | AV2 LiDAR sweeps | DAv2 weights on Virtual-KITTI synthetic (no AV2) | YES |
| platform | same AV2 vehicle/timestamp | same AV2 vehicle/timestamp | NO (declared) |
Verdict: differs on modality AND algorithm AND provenance → clears the `docs/research-integrity.md:37`
bar; shares platform → "much more independent than the traversal oracle, NOT external ground truth."
This is the FIRST oracle in the program to clear all three of modality+algorithm+provenance against AV2.

## Substrate (declared)
The 3 AV2 logs with stereo cameras on disk (`201fe83b`, `2c652f9e`, `6aaf5b08`) — the stop-and-go
FOLLOWING substrate. Following-only is declared; a free-driving generalization needs a camera download
(deferred). Camera = `stereo_front_left` (so the EXISTING 60 human-labeled calibration patches —
owner-labeled for the stereo attempt, 34 structure / 26 empty — are reused VERBATIM: a clean head-to-head
"same patches where classical stereo scored AUC 0.259").

## Estimand (sealed; only the depth stage differs from the stereo pre-reg)
Per frame t, on `stereo_front_left` nearest-timestamp to LiDAR sweep t:
1. Undistort (AV2 3-coeff Su radial, the stereo oracle's `build_undistort_map`/`remap`).
2. **DAv2 metric depth**: letterbox the undistorted image to square (preserve aspect — NOT a raw squash),
   run DAv2 ONNX → per-pixel meters, un-letterbox back to the undistorted grid. No scale/intrinsics fix
   (metric head is absolute).
3. Keep pixels with Z ∈ [z_min=2, z_max=30] m; back-project (u,v,Z) → ego (the validated
   `backproject_to_ego`); voxelize into the av2_sensor grid with the IDENTICAL ground/ego filters
   (`_ROAD_Z=0.3`, ego cuboid, RANGE, VOXEL_SIZE=0.4) → `depth_struct` mask (≥ n_depth_min=8 points/voxel).
4. In-path band ∩ camera FOV (`band_fov_mask`, |y|≤_EGO_HALF_W=1.05, x∈[0,30]).
5. `occ_free` = av2_sensor occupancy reports the voxel FREE.
- **MISS candidate = occ_free ∧ depth_struct** (a real surface the camera sees where occupancy says free).
- `recall_miss_rate(t) = |MISS band∩FOV voxels| / |depth_struct band∩FOV voxels|`. Per-log-clustered mean,
  log-clustered bootstrap 95% CI (`harness_v2._boot_mean`, n_boot=1000). Only 3 logs → 1 held out for
  thresholds → 2 headline clusters → WIDE CI (declared power limit).

## Gate 1 — metric-scale falsifier (BEFORE the run; pre-condition)
On known-good unoccluded annotation boxes (high `num_interior_pts`, in band, 2–30 m), compare DAv2 median
surface range to the LiDAR/box range. **If median |DAv2 − LiDAR_range| > 0.5 m in the 2–30 m band → metric
scale is invalid on AV2 → STOP, report INVALID-SCALE, no miss-rate.** (Boxes used only to validate the
oracle, never in the estimand.) Direction of error logged.

## Gate 2 — self-reliability AUC (BEFORE the confirmatory; the stereo lesson)
Reuse the existing 60 human-labeled patches (`results/calib_patches/calib_patches.json`). Signal =
the DAv2 `depth_struct` evidence count in each patch window. **AUC = ROC of that signal, human-pos vs
human-neg, via `camera_oracle._roc_auc`. Gate: AUC ≥ 0.75 → proceed; AUC < 0.75 → ORACLE-INSUFFICIENT,
no miss-rate.** Pre-registered prediction (part of the falsifier set): a learned net predicts depth from
priors/context, not local texture, so it should clear 0.75 on the SAME dark-vehicle-back patches that sank
classical stereo (0.259). If it ALSO scores < 0.75, the "learned priors rescue textureless backs" story is
wrong and the oracle is killed at the same gate — a reachable, non-rigged outcome.

## Null (band-local, reachable) + kill
Band-local null (`band_local_shuffle_rate`): relocate the occupied band∩FOV voxels to random band∩FOV
voxels, recompute miss-rate vs the SAME depth_struct, 1000 shuffles, seed 0. Claim = the GAP
`(shuffled_miss − true_miss)` log-clustered bootstrap CI strictly > 0.
- **RECALL-SUPPORTED** iff gap CI.lo > 0 (occupancy puts band-mass on the right band voxels better than
  random → it recalls structure the camera independently sees).
- **FAIL** iff gap CI includes 0 (occupancy no better than random within the band → it misses structure an
  independent sensor sees → recall denotation unsafe). Reported as the headline.
- **ORACLE-INSUFFICIENT** iff Gate 2 AUC < 0.75. **INVALID-SCALE** iff Gate 1 fails. **INDETERMINATE** iff
  < 4 usable frames or CI straddles 0.
**"This observation means I am wrong":** gap ≤ 0 on the oracle-clean subset refutes "occupancy recalls
in-path structure" on this substrate.

## Confounds + bias direction (declared)
- Correlated failure (both DAv2 and LiDAR fail on matte-black/low-light) → measured miss-rate is a LOWER
  BOUND (both-fail cases dropped). Mitigation: a clear-frame filter (drop frames with in-band depth-valid
  fraction < 0.30). Stated, not erased. NOTE: the net's textureless-prior strength should make this bound
  tighter than stereo's would have been.
- Aspect-squash error → letterbox (above) controls it; the un-letterbox mapping is verified in self-check.
- Sky/ground leakage (net assigns finite depth to sky/road) → killed by the reused Z∈[2,30] + `_ROAD_Z`
  ground filter + band∩FOV; a depth-specific guard drops Z>z_max (sky) before voxelizing.
- Same-platform/timestamp → "much more independent, not external truth" (declared).

## Self-check (run FIRST, no confirmatory data)
Reuse the stereo `_self_check`: ego-point round-trip < 0.1 m; undistort↔redistort < 0.5 px; PLUS a
DAv2-specific check: letterbox→un-letterbox round-trips pixel coords to < 1 px; the ONNX runs and returns
a [.,518,518] metric map with values in (0, 80].

## Sealed run (after this doc is committed; run ONCE)
```sh
.venv/bin/python experiments/occquery_v0/oracle_depth_recall.py --self-check
.venv/bin/python experiments/occquery_v0/oracle_depth_recall.py \
  --logs 201fe83b-7dd7-38f4-9d26-7b4a668638a9 2c652f9e-8db8-3572-aa49-fae1344a875b 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
  --heldout-threshold-log 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
  --onnx data/models/dav2_metric_vkitti_vitl/depth_anything_v2_metric_vkitti_vitl.onnx \
  --camera stereo_front_left --z-min 2 --z-max 30 --n-depth-min 8 \
  --scale-gate-m 0.5 --auc-gate 0.75 --null band-local --shuffles 1000 --seed 0 \
  --calib-json experiments/occquery_v0/results/calib_patches/calib_patches.json \
  --out experiments/occquery_v0/results/oracle_depth_recall.json
```
Order: (i) self-check; (ii) Gate 1 metric-scale → STOP if >0.5 m; (iii) Gate 2 AUC → secondary kill if
<0.75; (iv) confirmatory gap + band-local null. Held-out log reported separately, never pooled.

## Honest scope (for the summary; do NOT re-inflate)
- RECALL, one-sided; following-substrate only; 2 headline clusters → wide CI, suggestive not definitive.
- Cross-modal external check (modality+algorithm+provenance independent), but same vehicle/timestamp →
  "much more independent, not external ground truth."
- Measured miss-rate is a LOWER BOUND (correlated failures dropped).
- A clean ORACLE-INSUFFICIENT / INVALID-SCALE / FAIL is a legitimate sealed outcome → then recall stays
  consistency-only (box-recall) + FP-only external (traversal), and external recall routes to a future
  GPU/free-driving step.

## Seal checklist
- [x] DAv2 model on disk, metric, provenance = Virtual-KITTI (no AV2), verified by inference before seal.
- [x] Estimand/null/kill mirror the stereo pre-reg; only depth stage swapped; 60 existing labels reused (head-to-head).
- [x] Gate 1 (scale ≤0.5 m) + Gate 2 (AUC ≥0.75) + band-local null + kill rule declared before data.
- [ ] `--self-check` passes. **(required before the run)**
- [ ] This doc committed (git timestamp) BEFORE the confirmatory run; run EXACTLY once.
