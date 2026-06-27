# L1 — occupancy free-space denotation vs the Occ3D-nuScenes dense GT, field-standard (PRE-REG, seal before data 2026-06-28)

The forward-research replacement for the homegrown stereo recall oracle (which a tie bug in `_roc_auc`
deflated, verdict retracted, git e8e96aa). Instead of chasing a MORE-independent custom oracle than the
field has, this adopts the FIELD-STANDARD reference (the dense, temporally-aggregated Occ3D-nuScenes
occupancy GT, already local at `data/gts/`) and a tested metric machinery, exactly as occupancy-prediction
benchmarks do — honestly labelled CONSISTENCY vs the dense reference, not external truth. Nothing below was
chosen after seeing a number.

## Claim (two legs; H1 is the headline, oracle-free)
**Leg 1 — EXPRESSIVITY (already established, re-stated as the load-bearing headline).** Box-only (RefAV's
released 32-function set: tracked boxes + HD-map polygons) CANNOT express the free-space query family
(clearance / centerline / free_path / corridor) — no dense-occupancy / free-space primitive. Established
oracle-free in `h3b_expressivity` (sealed a47b500, no `_roc_auc` dependency → uncontaminated). This is the
sole headline.

**Leg 2 — DENOTATION CORRECTNESS vs the dense GT (the new, field-standard, CI'd measurement).** For the
free-space queries occupancy CAN express, is the predicate's denotation on a REALISTIC (single-frame
OBSERVED) occupancy correct against the DENSE aggregated reference?
- **Under test:** the occupancy free-space predicates (`free_along_ego_path`, `min_free_width_along_path`,
  `lateral_clearance`) evaluated on the SINGLE-FRAME OBSERVED occupancy (`load_scene(..., mask='lidar')` —
  unobserved voxels UNKNOWN).
- **Reference (GT):** the SAME predicates' answer on the DENSE accumulated Occ3D GT (`mask='none'`) — the
  field-standard temporally-aggregated occupancy. The two differ by occlusion/sparsity, so this is a real
  test, not the identity.
- **Estimand:** per-frame, over the in-path band, the free/blocked denotation of each predicate on the
  observed grid vs the dense-GT grid. Reported as standard **IoU (free-space class), precision, recall, F1**
  of the "free" denotation, plus the false-block rate (observed says BLOCKED where dense GT is FREE) and the
  miss rate (observed says FREE where dense GT is OCCUPIED) — the recall half the stereo oracle never reached.

## Metric implementation (the _roc_auc lesson, applied)
IoU / precision / recall / F1 are SET operations (TP/FP/FN over voxels) with NO rank-tie ambiguity, so they
are computed in numpy and **unit-tested against hand-computed values on a fixed toy case** in the same
commit (and spot-checked against `scipy`); a homegrown rank-AUC is explicitly avoided. The decision is made
on a **bootstrap confidence interval** (scene-clustered, 1000 resamples, seed 0), NOT a bare point estimate
(the second lesson). No fixed absolute cutoff is load-bearing.

## Independence ledger (HONEST — this is consistency, not external truth)
Both the observed occupancy and the dense GT are LiDAR-derived (single-frame vs temporally-aggregated +
annotation-cleaned). So Leg 2 is **same-modality CONSISTENCY between the observed occupancy and the dense
field-standard reference**, NOT external ground truth — the same honest status the whole occupancy field
operates under (mIoU-vs-LiDAR-GT). H1 (Leg 1, expressivity) needs NO oracle and remains the headline. No
H3 re-inflation: a passing Leg 2 is reported as "observed-vs-dense consistency, CI'd," never as externally
validated denotation F1.

## Load-bearing comparison = RELATIVE, not absolute
Box-only's denotation of the SAME free-space queries = INAPPLICABLE (coverage 0, it cannot express them) →
the gap is STRUCTURAL (H1). On the queries occupancy expresses, Leg 2 reports the denotation IoU/F1 with a
CI as a secondary internal check. The headline number is the EXPRESSIVITY gap (Leg 1), not an absolute F1.

## Data (sealed)
Occ3D-nuScenes local `data/gts/` + `annotations.json`. Scenes: ALL scenes present locally, split into a
threshold/dev set (first 20% by sorted scene id) and a held-out headline set (remaining 80%); the dev set
is for nothing that touches the estimand (there are no free parameters to tune — the predicates and band are
the sealed ones from `queries.yaml`). Per-frame in-path band identical to the sealed occquery band.

## Kill (reachable, declared before data)
- **H1 falsified** iff box-only CAN express the free-space family (coverage gap collapses) — would sink the
  occupancy angle. Not expected; the run reports the real RefAV flags, not assumed.
- **Leg-2 denotation FAILS** iff the observed-occupancy free-space predicate is NO BETTER than a trivial
  baseline (all-free, or random at the band's free-rate) on IoU/F1 with the bootstrap CI — i.e. observed
  occupancy's free-space denotation does not track the dense GT. A real negative, reported as the headline.
- **"This observation means I am wrong":** if the false-block rate's CI includes the all-free baseline, the
  predicate carries no denotation signal beyond the prior; if IoU CI.lo ≤ trivial, Leg 2 is dead.

## Run (after THIS doc is committed; once)
`l1_denotation_occ3d.py` → loads each scene twice (mask='lidar' under-test, mask='none' GT), evaluates the
sealed free-space predicates per frame, computes the set metrics + scene-clustered bootstrap CI + the
trivial baselines, writes `results/l1_denotation_occ3d.json` + a summary. A unit test
`tests/test_l1_denotation_metrics.py` pins the numpy set-metrics against hand-computed values.

## Honest scope
Leg 1 (expressivity) is the rigorous oracle-free headline. Leg 2 is a CONSISTENCY measurement of observed
vs dense-aggregated occupancy with tested set-metrics + a CI — stronger and more standard than the retracted
stereo oracle, but still NOT external ground truth (both sides LiDAR-derived). Reported with that label, no
inflation.
