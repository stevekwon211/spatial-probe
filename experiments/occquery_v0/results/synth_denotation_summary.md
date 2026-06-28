# Synthetic denotation-MECHANISM validation -- occquery free-space predicates

- Pre-reg: `synth_denotation_preregistration.md (SEALED, written before grading code/data)` (sha256 `619219be2c48d25b...`)
- Commit: `589e7279d41d2d426495fb4a3ee52b4abe33b987`  seed 0  voxel 0.5 m  horizon 1.0s
- **Result class: SYNTHETIC MECHANISM validation -- by-construction GT, independent of the predicate input; real raycast occlusion. Valid for MECHANISM/EDGE/NUMERICAL correctness ONLY, NEVER field evidence. Does NOT re-inflate H3. H1 stays the sole field headline.**
- Scenes: 120 total, 82 obstacle-bearing
- Band: ego in-path corridor: forward 0..(length/2 + speed*1.0s), |lateral| <= width/2 + 1.0 m; obstacle BEV in the ego-height band (the EXACT substrate the sealed free-space predicates read).
- Occlusion: single-sensor: voxel -> UNKNOWN iff probe.raycast.line_of_sight from ego is blocked

## SYNTHETIC label (loud)
This is a **by-construction MECHANISM check**: it validates the predicate denotation LOGIC and its graceful degradation under occlusion. It is **NOT real-world field denotation** -- the real-world denotation stays gated (H3 demoted). **H1 (expressivity) remains the field headline.**

## Non-vacuity
Obstacle-bearing subset band GT **blocked-rate = 0.1140** (free-rate 0.8860). The `l1_denotation_occ3d` band was 99.5% free (vacuous); this design controls obstacle density so the FREE/BLOCKED classes are both substantial.

## Denotation metrics -- obstacle-bearing subset (FREE class, scene-clustered bootstrap CI 1000)
| metric | predicate (unknown=FREE, sealed) | all-free | random@free-rate |
|---|---|---|---|
| IOU | 0.9056 CI[0.8914, 0.9185] | 0.8860 CI[0.8720, 0.8988] | 0.7953 CI[0.7836, 0.8059] |
| F1 | 0.9504 CI[0.9431, 0.9570] | 0.9396 CI[0.9318, 0.9464] | 0.8860 CI[0.8791, 0.8922] |

Predicate (unknown=FREE, sealed) full denotation:
- precision: 0.9056 CI[0.8917, 0.9181]
- recall: 1.0000 CI[1.0000, 1.0000]
- false_block_rate: 0.0000 CI[0.0000, 0.0000]
- miss_rate: 0.8106 CI[0.7888, 0.8302]

Box-only baseline: INAPPLICABLE (coverage 0 -- box-only cannot express free-space; no number fabricated)

### Sensitivity -- unknown_policy = OCCUPIED (conservative reading of the unobserved)
- iou: 0.8613 CI[0.8379, 0.8868]
- f1: 0.9255 CI[0.9097, 0.9409]
- false_block_rate: 0.1387 CI[0.1119, 0.1655]
- miss_rate: 0.0000 CI[0.0000, 0.0000]

## Secondary -- predicate-verdict `free_along_ego_path` vs INDEPENDENT constructed label
- on TRUE grid (perfect input): accuracy **0.9917** (tp 81, fp 1, fn 0, tn 38) -- validates the LOGIC
- on OBSERVED grid (occluded): accuracy 0.9917 (tp 81, fp 1, fn 0, tn 38) -- graceful degradation

## Full-set denotation (incl. free controls, for completeness)
- predicate(unknown=FREE) IoU 0.9345 CI[0.9216, 0.9466], band blocked-rate 0.0796

## Verdict (per the sealed kill criteria)
- kill metric: predicate(unknown=FREE) FREE-IoU CI.lo > max(all-free, random) IoU mean, on obstacle-bearing subset
- predicate IoU CI.lo = 0.8914; baseline max IoU mean = 0.8860
- predicate beats trivial baseline (CI.lo > baseline mean): **True**
- HONEST caveat -- predicate IoU CI overlaps all-free CI: **True** (thin cell-IoU margin; see verdict)
- predicate-VERDICT accuracy vs constructed label: true grid 0.9917, observed (occluded) 0.9917 (the decisive occlusion-robust line)
- false_block_rate(unknown=FREE) ~ 0: True
- predicate logic on perfect input OK (>=0.99): True
- **KILLED: False**
- NOT KILLED (per the sealed kill: cell FREE-IoU CI.lo beats the trivial baseline mean). HONEST caveat: on cell-IoU the margin is THIN and the bootstrap CIs OVERLAP the all-free baseline -- because the in-path band is majority-free and most blocked VOLUME is genuinely occluded under a single sensor (miss_rate high, false_block_rate exactly 0 -> correct graceful degradation, no logic bug). The DECISIVE, occlusion-robust evidence is the predicate VERDICT (free_along_ego_path) vs the independent constructed label: accuracy 0.9917 on perfect input AND 0.9917 under occlusion (the visible front face suffices to denote BLOCKED). SYNTHETIC MECHANISM result ONLY -- not field evidence; H3 stays gated; H1 remains the field headline.
