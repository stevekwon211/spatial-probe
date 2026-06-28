# Synthetic denotation-MECHANISM validation for occquery (PRE-REG, seal before data 2026-06-28)

The ONE place denotation-correctness can be EXTERNALLY validated. Every real AV occupancy GT (Occ3D /
nuScenes) is LiDAR-derived, so grading our LiDAR-voxelized occupancy against it is CONSISTENCY, not truth
(that is why H3 is demoted in `preregistration.md` / `CLAUDE.md`, and why `l1_denotation_occ3d` came back
DEGENERATE: the Occ3D lidar mask hides ~0% of OCCUPIED voxels, so observed-obstacles == dense-GT-obstacles
by construction). Here WE construct the ground truth, so the reference is genuinely independent of the
predicate's input — and we inject REAL occlusion with the repo's own raycaster, so the observed grid
genuinely differs from the truth. Nothing below is chosen after seeing a number.

**HONEST SCOPE — read first.** This is a SYNTHETIC MECHANISM validation. Per research-integrity, a
by-construction result is valid for MECHANISM / EDGE / NUMERICAL correctness ONLY — it establishes that the
free-space predicate denotation LOGIC is sound and degrades gracefully under occlusion. It is NEVER field
evidence and NEVER re-inflates H3 (real-world denotation stays gated/demoted). H1 (expressivity, oracle-free)
remains the SOLE field headline.

## Construct (independent-by-construction GT)
N = 120 procedural single-frame scenes, `numpy.default_rng(0)`. Each scene is a dense voxel grid (voxel
0.5 m, ground_height 0.25 m, ego at world (5, 10, 1.0), heading 0 = +x, width 1.85 / length 4.6 / height
1.9, speed in {8, 10, 12} m/s) whose every voxel is KNOWN FREE or OCCUPIED — this is the TRUE GT. Obstacles
are vertical columns (z 0.5..2.0 m, the ego height band) placed at CONTROLLED ego-frame (forward, lateral):
- ~30 scenes with **0 in-path blockers** (free controls; may still carry off-corridor roadside walls);
- ~50 scenes with **exactly 1 in-path blocker** at forward U[5,15] m, near the centerline, varied width;
- ~40 scenes with **several (2–4) in-path blockers** at varied range/lateral + side walls.
Roughly half the obstacle-bearing scenes also place a SECOND obstacle directly BEHIND a visible one (so the
rear one is genuinely occluded) — this is what makes the occlusion test non-trivial. NON-VACUITY is reported
as the band GT blocked-fraction on the obstacle-bearing subset (the `l1_denotation_occ3d` band was 99.5%
free → vacuous; this design controls it and reports the realized number).

## Simulate the OBSERVED grid (real single-sensor occlusion via the repo raycaster)
From the ego voxel, every voxel in the forward height-band region is tested with `probe.raycast.line_of_sight`
(3D DDA). A voxel keeps its TRUE value iff line-of-sight from the ego is clear; if an OCCUPIED voxel lies
strictly between the ego and it, it becomes UNKNOWN. So the first hit along each ray stays visible (OCCUPIED)
and everything behind it is hidden (UNKNOWN) — a realistic single-sensor occluded view. (Confirmed this
session: `line_of_sight(ego, wall) is True`, `line_of_sight(ego, behind-wall) is False`.)

## Estimand (PRIMARY — cell-level free/blocked denotation, with CI vs baselines)
Over the ego in-path band (forward 0..reach = length/2 + speed·horizon with **horizon = 1.0 s**, |lateral| ≤
width/2 + **1.0 m**, ego-height band — the EXACT substrate `free_along_ego_path` / `min_free_width_along_path`
/ `lateral_clearance` read, reused verbatim as `band_blocked_bev` from `l1_denotation_occ3d.py`), classify
each band cell FREE/BLOCKED. Positive class = FREE.
- **Reference (TRUE GT):** `band_blocked_bev` on the fully-known constructed grid — independent of the
  predicate's occluded input.
- **Under test:** `band_blocked_bev` on the OCCLUDED observed grid, under **unknown_policy = FREE (SEALED)**
  and **unknown_policy = OCCUPIED (sensitivity)**.
- **Metrics:** FREE-class IoU / precision / recall / F1, false_block_rate (= FN/(TP+FN): observed BLOCKED
  where GT FREE), miss_rate (= FP/(FP+TN): observed FREE where GT OCCUPIED), with a **scene-clustered
  bootstrap CI (1000 resamples, seed 0)** on the pooled confusion. Metric machinery is the
  already-unit-tested L1 numpy set-metrics (`confusion_from_masks`, `free_set_metrics`, `_boot_metric`) — no
  sklearn/torch (repo rule; the `_roc_auc` lesson). The metrics are reported on the FULL set AND on the
  **obstacle-bearing subset** (scenes whose TRUE band has ≥1 blocked cell) — the verdict uses the subset.

## Estimand (SECONDARY — predicate-verdict denotation, mechanism)
Per scene, `free_along_ego_path` (horizon 1.0) BLOCKED verdict on the TRUE grid and on the OBSERVED grid,
graded against an INDEPENDENT constructed label `path_truly_blocked` = (a TRUE OCCUPIED voxel sits in the
ego corridor |lateral| ≤ width/2, forward in [0, reach], height band). On the TRUE grid this validates the
predicate LOGIC on perfect input (must be near-identity); on the OBSERVED grid it shows degradation.

## Trivial baselines (the relative gap, not an absolute cutoff)
- **all-free:** predict every band cell FREE (`_allfree_conf`).
- **random@free-rate:** predict FREE with prob = band GT free-rate (`_random_conf`, expected confusion).
Box-only is INAPPLICABLE (coverage 0 — cannot express free-space; stated, never fabricated).

## KILL (declared before data — reachable, falsifiable)
- **Killed** iff, on the OBSTACLE-BEARING subset, the predicate (unknown = FREE, sealed) FREE-class **IoU
  bootstrap CI.lo ≤ max(all-free IoU mean, random@free-rate IoU mean)** — the predicate denotation carries
  NO signal beyond the trivial prior even with a perfect independent GT.
- **"This observation means I am wrong":** (a) false_block_rate (unknown = FREE) is materially > 0 on
  truly-free in-path space (the predicate falsely blocks free space → logic broken — expected EXACTLY 0,
  since occlusion only turns OCCUPIED→UNKNOWN→free, never free→blocked); (b) the SECONDARY predicate-verdict
  on the TRUE (perfectly observed) grid does NOT match `path_truly_blocked` (IoU/accuracy < ~0.99 → the
  predicate logic itself is broken, independent of occlusion).
- A high miss_rate under unknown = FREE that is concentrated in OCCLUDED (hidden-behind) obstacles is NOT a
  kill — it is correct, expected graceful degradation (you cannot see through an obstacle). It is quantified
  and reported as the occlusion cost, and the unknown = OCCUPIED arm shows the conservative trade (miss_rate
  down, false_block_rate up).

## Expected (stated before data, so drift shows)
Predicate (unknown = FREE) beats both baselines on the obstacle-bearing subset (visible obstacles are
recovered → fewer FP than all-free); false_block_rate ≈ 0; miss_rate > 0 driven by occluded rear obstacles;
unknown = OCCUPIED raises false_block_rate and lowers miss_rate; predicate-verdict on the TRUE grid ≈ perfect.
A NEGATIVE (CI.lo ≤ baseline) would be the headline.

## Run (after this doc is written; once)
`experiments/occquery_v0/synth_denotation.py` → constructs scenes (seed 0), simulates observation, grades,
writes `results/synth_denotation.json` + `results/synth_denotation_summary.md` (only `*_summary.md` tracked).
`tests/test_synth_denotation.py` pins the constructed-GT + occlusion logic and a hand-checked single-blocker
scene (one blocker at 10 m → predicate denotes BLOCKED there, FREE elsewhere).

## Note on the seal
Repo instruction for THIS task forbids `git commit`. The pre-registration is therefore WRITTEN before any
grading code or data this session (temporal seal preserved); its sha256 is recorded in the result JSON for a
tamper-evident link. Committing is deferred to the user. This deviation from the usual git-timestamp seal is
disclosed in the report.
