# Auto-label recovery + the human residual — Occ3D-nuScenes (P3)

- Pre-reg: `experiments/autolabel_v0/preregistration.md (SEALED, commit 9b34df9)`
- Commit: `f7864a0be728967e2bbb4d7e8a5218c218c19071`  seed 0
- Result class: SAME-MODALITY recovery/consistency (occupancy auto-labeler vs nuScenes GT, both LiDAR-derived) — the floor P2's camera-LiDAR fusion detector is meant to beat, NOT independent detector field-eval.
- Headline 680 scenes / dev 170; τ_max 2000, verdict d=2.0 m

## Verdicts: C3-A HOLDS (automation ceiling) / C3-B HOLDS (structured)

## C3-A — precision/recall vs τ (d=2.0 m, headline)
best over τ of min(precision, recall) = **0.341** (< 0.9 → ceiling holds).

| τ | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 2 | 0.126 CI[0.120, 0.132] | 0.499 CI[0.490, 0.509] | 0.201 | 361402 | 2506280 | 362180 |
| 5 | 0.202 CI[0.194, 0.210] | 0.452 CI[0.442, 0.461] | 0.279 | 326711 | 1292021 | 396871 |
| 10 | 0.286 CI[0.275, 0.296] | 0.408 CI[0.398, 0.417] | 0.336 | 294889 | 736172 | 428693 |
| 20 | 0.372 CI[0.360, 0.385] | 0.341 CI[0.332, 0.351] | 0.356 | 246850 | 416129 | 476732 |
| 40 | 0.431 CI[0.415, 0.446] | 0.264 CI[0.255, 0.273] | 0.327 | 190930 | 252363 | 532652 |

## C3-B — recall by slice (τ*=20, F1-max)
class gap vehicle−pedestrian = **0.205**, range gap near−far = **0.469** (both ≥ 0.2 → structured).

| slice | recall |
|---|---|
| vehicle | 0.391 |
| pedestrian | 0.186 |
| bicycle | 0.366 |
| motorcycle | 0.427 |
| near | 0.589 |
| mid | 0.474 |
| far | 0.120 |

## Reading
The occupancy-only auto-labeler is the FLOOR: a same-modality proposer with no semantics. The residual it leaves (low precision from static structure, recall collapse on small/distant objects) is exactly the human queue a Data PM owns — and what P2's camera-LiDAR fusion detector is meant to shrink. Numbers are recovery/consistency, not detector field-eval.
