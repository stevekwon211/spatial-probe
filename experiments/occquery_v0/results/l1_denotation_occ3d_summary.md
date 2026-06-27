# L1 denotation -- occupancy free-space vs Occ3D dense GT (CONSISTENCY)

- Pre-reg: `l1_denotation_occ3d_preregistration.md (SEALED, commit 6475448)`
- Commit: `7296cd071d04cb2a222da8ee12360d4ca3a34bcf`  seed 0  horizon 1.0s
- Result class: CONSISTENCY (observed single-frame occupancy vs dense aggregated Occ3D GT, both LiDAR-derived) -- NOT external truth. No H3 re-inflation.
- Band: ego in-path corridor: forward 0..(length/2 + speed*1.0s), |lateral| <= width/2 + 1.0 m; obstacle BEV in the ego-height band (the substrate the sealed free-space predicates read).

## Leg 1 -- EXPRESSIVITY (SOLE headline, oracle-free)
Free-space families: occupancy **100.0%** vs box-only **0.0%** -> **100.0-pt** expressivity gap (H1 falsified = False). Source: results/h3b_expressivity.json (sealed h3b_expressivity_preregistration.md, a47b500).

## Leg 2 -- DENOTATION CONSISTENCY (NOT external truth)
Headline split: 680 scenes / 27406 frames (dev 170 scenes). Band GT free-rate 0.9953.

| metric (FREE class) | predicate (unknown=FREE, sealed) | all-free | random@free-rate |
|---|---|---|---|
| IOU | 1.0000 CI[1.0000, 1.0000] | 0.9953 CI[0.9945, 0.9960] | 0.9906 CI[0.9898, 0.9913] |
| F1 | 1.0000 CI[1.0000, 1.0000] | 0.9976 CI[0.9972, 0.9980] | 0.9953 CI[0.9949, 0.9956] |

Predicate (unknown=FREE, sealed) full denotation:
- precision: 1.0000 CI[1.0000, 1.0000]
- recall: 1.0000 CI[1.0000, 1.0000]
- false_block_rate: 0.0000 CI[0.0000, 0.0000]
- miss_rate: 0.0001 CI[0.0000, 0.0003]

Box-only baseline: INAPPLICABLE (coverage 0 -- box-only cannot express free-space; no number fabricated)

### Sensitivity -- unknown_policy = OCCUPIED (conservative reading of the unobserved)
- iou: 0.1390 CI[0.1377, 0.1407]
- f1: 0.2441 CI[0.2419, 0.2464]
- false_block_rate: 0.8610 CI[0.8595, 0.8624]
- miss_rate: 0.0000 CI[0.0000, 0.0000]

## Ground-truth degeneracy (the headline negative for Leg 2)
Occ3D's mask_lidar marks ~100% of OCCUPIED voxels visible (verified: only ~0.008% of occupied voxels differ obs-vs-dense), so under the sealed unknown_policy=FREE the observed and dense-GT OBSTACLE sets are identical and the FREE denotation is ~1.0 BY CONSTRUCTION -- a synthetic-class identity, NOT an occlusion-robustness test. The pre-reg premise 'the two differ by occlusion/sparsity' is FALSIFIED for the obstacle class; the difference lives entirely in FREE->UNKNOWN voxels, governed by the unknown policy (see the OCCUPIED sensitivity arm). The adapter docstring (occ3d.py) warns this.

**Observed obstacles == dense-GT obstacles by construction: True**

## Verdict (per the sealed kill criteria)
- IoU CI.lo beats all-free mean: True
- Leg 2 is a result: False
- NOT A RESULT (degenerate): any apparent win over the trivial baseline is an artifact of observed-OBSTACLES == dense-GT-OBSTACLES by construction, not evidence of denotation robustness. Reported as consistency only; the SOLE headline remains Leg 1 (expressivity).
