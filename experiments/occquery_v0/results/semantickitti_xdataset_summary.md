# SemanticKITTI cross-dataset -- H1 (3rd dataset) + non-degenerate denotation

- Pre-reg: `semantickitti_xdataset_preregistration.md (SEALED before the confirmatory verdict; NOT git-committed per task instruction)`
- Commit: `1f9ae79396c61ca7330cde9632dcf80eedf014a5`  seed 0
- Sample: 11 labeled sequences ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 440 frames (<= 40/seq, evenly spaced)
- Result class: Leg 1 = real-data EXPRESSIVITY (oracle-free, the headline). Leg 2 = CONSISTENCY (observed single-scan vs completed dense GT, both LiDAR-derived) -- NON-degenerate + NON-vacuous, but NOT external truth.

## Leg 1 -- H1 EXPRESSIVITY on a THIRD dataset (SOLE field headline, oracle-free)
Free-space families on real SemanticKITTI grids: occupancy **100.0%** vs box-only **0.0%** -> **100.0-pt** gap. H1 falsified (box expresses free-space) = **False**. Probe scene: `00:observed`.
The expressivity gap now holds on AV2 (h3b) + Occ3D-nuScenes + SemanticKITTI -- 3 structurally-different datasets.

## Non-degeneracy + non-vacuity (the two L1 failure modes, checked)
- **Non-degeneracy**: XOR = 4.440% of all voxels differ obs-vs-dense (L1/Occ3D was 0.008%). Dense GT has 8.6x the observed occupied voxels; only 1938 observed-occupied voxels are absent from the dense GT (.bin ~ a subset of .label). NON-DEGENERATE.
- **Non-vacuity**: Headline forward ego-height BEV field: GT blocked-rate 50.6% (NON-vacuous). Thin in-path corridor (L1-style): GT blocked-rate 5.7% (VACUOUS, the corridor ahead is free by construction -- why the headline domain was widened).

## Leg 2 -- DENOTATION CONSISTENCY (non-degenerate, non-vacuous; NOT external truth)
Headline domain: forward ego-height BEV field (x in [0,51.2] m, ego-height z-band voxel-idx 3-11), determinable columns only (not all-invalid). FREE = positive; pred = observed .bin, ref = dense .label.
GT blocked-rate 50.6% (free-rate 49.4%).

| metric (FREE class) | predicate (observed .bin) | all-free | random@free-rate |
|---|---|---|---|
| IOU | 0.5304 CI[0.4865, 0.5731] | 0.4939 CI[0.4495, 0.5387] | 0.3280 CI[0.3085, 0.3472] |
| F1 | 0.6932 CI[0.6555, 0.7313] | 0.6613 CI[0.6225, 0.6982] | 0.4939 CI[0.4717, 0.5153] |

Predicate full denotation (observed vs dense GT):
- precision: 0.5304 CI[0.4873, 0.5754]
- recall: 1.0000 CI[1.0000, 1.0000]
- false_block_rate: 0.0000 CI[0.0000, 0.0000]
- miss_rate: 0.8640 CI[0.8546, 0.8738]

Box-only baseline: INAPPLICABLE (RefAV box+map has no free-space primitive; no number fabricated)

Contrast -- thin in-path corridor (VACUOUS, GT blocked-rate 5.7%): predicate IoU 0.9477 CI[0.9102, 0.9730] vs all-free IoU 0.9434 CI[0.9105, 0.9676].

## Verdict (per the sealed kill criteria)
- Leg 1 H1 holds on KITTI: **True**
- Leg 2 predicate IoU CI.lo > all-free IoU mean (a relative result): **False**
- Leg 2 label: CONSISTENCY (non-degenerate, non-vacuous occlusion-robustness), NOT external truth
