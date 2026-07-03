# Pre-registration — occupancy-to-box AUTO-LABEL recovery + the human residual (P3 mini-flywheel)

Committed BEFORE any corpus run of this experiment. Post-hoc edits show in git history.
Author: Doeon Kwon + Claude. Date: 2026-07-03. Experiment: `experiments/autolabel_v0`.

## Motivation (the product question this measures)

A data engine auto-labels, then routes what it can't trust to humans; a Data PM exists because
that residual is large, structured, and expensive. This experiment measures the residual on a
REAL, CPU-runnable auto-labeler — connected-component clustering of the Occ3D-nuScenes occupancy
field into proposal boxes — scored against the real nuScenes GT boxes with the **nuScenes
detection matching rule** (BEV center-distance, standard-over-homegrown). It answers: *how much
of the box-labeling can occupancy alone automate, and where does it structurally fail?*

Honest scope up front: the occupancy is itself nuScenes-LiDAR-derived, so this auto-labeler is a
**same-modality proposal**, NOT an independent learned detector — it is the floor that P2's
camera-LiDAR fusion detector is meant to raise. The HARNESS (QA gate + automation-rate curve +
failure slices) is built to accept P2's mmdet3d predictions unchanged; running it on the
occupancy proposer now gives a real, pre-registered baseline and the residual's SHAPE.

## The auto-labeler (fixed before the run)

Per frame, dense arm (`mask='none'`, ego-height band via `obstacle_centers(max_height_agl=
ego.height)`):
1. Take the occupied-voxel mask above ground within the ego-height band.
2. 3D connected components (`scipy.ndimage.label`, 26-connectivity) → clusters.
3. Each cluster → a BEV proposal: axis-aligned centroid + a "confidence" = cluster voxel count.
4. A cluster is a candidate object proposal iff its voxel count ∈ [τ, τ_max]: τ gates out
   sensor-noise specks, τ_max = 2000 voxels gates out static structure (buildings/walls span far
   more voxels than any vehicle at 0.4 m resolution) — both pre-registered knobs, swept/reported.

## Matching (nuScenes rule, fixed)

Greedy nuScenes-style: for match distance d, each GT box is matched to the nearest unmatched
proposal whose BEV center distance ≤ d (GT sorted by range, nearest-first). TP = matched GT,
FN = unmatched GT, FP = unmatched proposal. Report the full curve over the nuScenes distance set
d ∈ {0.5, 1, 2, 4} m; the **verdict operating distance = d = 2.0 m** (the nuScenes default).
Recall = TP/(TP+FN), precision = TP/(TP+FP). "other"-class GT (barriers/cones/debris — Occ3D
maps them to generic obstacle) is EXCLUDED from GT for scoring (no cuboid-object identity);
proposals matching them are neither TP nor penalized (removed before FP counting).

## Claims (falsifiable, reachable kills)

**C3-A — an automation ceiling exists (occupancy alone cannot do the labeling).**
Sweep τ ∈ {2, 5, 10, 20, 40} (τ_max fixed). Over the τ sweep, NO operating point reaches BOTH
precision ≥ 0.90 AND recall ≥ 0.90 at d = 2.0 m on the headline split.
- **C3-A HOLDS** iff max over τ of min(precision, recall) < 0.90 — i.e. every τ leaves a
  nonzero residual human queue at any usable quality.
- **C3-A KILLED** iff some τ gives precision ≥ 0.90 AND recall ≥ 0.90 — occupancy nearly
  automates the labeling; the automation-ceiling case is not supported on this axis.

**C3-B — the failure is STRUCTURED (so human effort can be targeted).**
At the τ maximizing F1 (d = 2.0 m, headline), recall is ordered by class and by range:
- vehicle-recall − pedestrian-recall ≥ 0.20, AND
- near-recall (GT range < 20 m) − far-recall (GT range > 35 m) ≥ 0.20.
- **C3-B HOLDS** iff BOTH gaps ≥ 0.20.
- **C3-B KILLED (unstructured)** iff BOTH gaps < 0.10 — misses are spread evenly, so there is
  no cheap slice to hand a human; the targeted-QA / Data-PM playbook weakens on this mechanism.
- Mixed (one gap in [0.10, 0.20)): NO adjective claim; report the slice table only.

## Metric implementation

- Precision/recall/F1 = plain TP/FP/FN counts (pure numpy; the `_roc_auc` lesson — no homegrown
  ranking metric, no sklearn). Scene-clustered bootstrap CI (1000 resamples, seed 0), the L1
  machinery. Unit-tested against a hand-computed toy (matcher + rates) BEFORE the run
  (`tests/test_autolabel.py`).
- The greedy matcher is unit-tested on a fixed 3-GT / 3-proposal layout with a hand-derived TP set.
- Range = BEV distance of the GT box center from the ego origin (ego-centric frame; ego at origin).

## Data (sealed)

- Corpus + split identical to occquery: every scene in `data/annotations.json`; first 20% by
  sorted id = dev, rest (~680) = headline. No parameter is tuned on headline (τ/τ_max/d all
  pre-registered here; F1-max τ for C3-B is selected on headline but is a REPORTED operating
  point, not a fitted free parameter — the claim gaps are pre-registered).
- `load_scene(name, 'data', mask='none', with_boxes=True)`; `unknown_policy=FREE` (= occquery).

## Independence ledger / honest scope

- Auto-labeler and GT share the nuScenes LiDAR modality → this is a same-modality RECOVERY /
  consistency study, not detector field-eval. Stated, not hidden. C3 claims are about the
  occupancy proposer's ceiling, explicitly the floor P2's fusion detector should beat.
- The honesty-link to occquery P1 (single-frame UNRESOLVED → must route to human) is REPORTED as
  context, not a C3 claim: fraction of FN GT boxes whose ego-frame BEV cell is UNKNOWN on the
  single-sweep observed grid (how many misses are "unobserved", not "mis-labeled").

## Run (post-seal, once)

```
python experiments/autolabel_v0/run.py            # full sealed run
python experiments/autolabel_v0/run.py --limit 5  # smoke only, never reported
```
Outputs `results/summary.md` (git-tracked) + `results/autolabel_v0.json` + per-scene JSONL
checkpoint (gitignored). Seed 0, commit recorded. Negatives are headlines.
