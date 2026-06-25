# Oracle-v2 — GT-box RECALL oracle for occupancy denotation-COMPLETENESS (PRE-REGISTRATION, seal before data 2026-06-25)

Mirrors the methodology of the PASSED FP oracle (`oracle_traversal_v0_1_preregistration.md`): per-frame
estimand, a RELATIVE region-local null, scene-clustered bootstrap (`harness_v2._boot_mean`), a reachable
CI kill. Absorbs the FAILED stereo oracle's lessons (`oracle_stereo_recall_summary.md`): a
relocate-ANYWHERE null is unreachable → the null is region-local; and a noisy oracle signal needs a
self-reliability gate → here that gate is a DETERMINISTIC data filter (`num_interior_pts`), not an AUC.
Nothing below was chosen after seeing a recall number.

## Why
The traversal oracle measures occupancy FALSE POSITIVES only and explicitly defers RECALL (real obstacles
occupancy MISSES) — the safety-load-bearing half. This oracle measures recall: does the occupancy map
mark FREE where a real, LiDAR-observed object is? A missed obstacle is the collision-relevant failure.

## Thesis — what is and is NOT claimed (read first)
The graded artifact is `av2_sensor._voxelize(raw LiDAR sweep t)`. A human-annotated tracked box with
`num_interior_pts ≥ N` is a location where the **same LiDAR sweep physically returned ≥N points on a real
object**. If the voxelization marks that object's above-ground/in-range/non-ego sub-volume FREE, the
pipeline **lost returns it provably had** — an *internal-completeness miss*. The estimand is the rate of
such misses; the load-bearing claim is the RELATIVE gap between the miss-rate at real boxes and at
size/range-matched random on-road relocations.
**Scope ceiling, stated up front (per repo CLAUDE.md H3 demotion):** gating on `num_interior_pts` ties
this to the **same LiDAR modality** as the voxelizer, so this is a **same-modality internal-consistency
check of the voxelization/threshold/filter chain, NOT external truth.** Box *provenance* (human
annotation) is independent of the voxelization *algorithm* — that is the only independence axis it earns
(see ledger). The honest contribution is **diagnosing where the pipeline drops returns it had** (the
`_ROAD_Z` floor, the one-point-per-voxel rule, range/ego clipping), with per-stratum attribution — not a
verified recall P/R/F1.

## Verified data facts (computed before seal, on disk; NOT the estimand)
- `annotations.feather` present for all 27 logs; 93,529 boxes; `num_interior_pts` min 0 / max 41,464 /
  mean 188.9. **pts==0: 18.6%** (sensor-blind, excluded); **pts≥5: 57.1%** (the sealed inclusion set);
  pts≥20: 35.5%.
- **Annotation timestamps == LiDAR sweep timestamps EXACTLY** (156/156 … per log, ann_ts ⊆ sweeps for
  all logs). So the box and the graded sweep are the SAME instant → the dynamic-temporal-mismatch
  confound (C1) is **zero** at the graded frame.
- **88.1% of box bottoms (`tz − height/2`) lie below `_ROAD_Z = 0.3 m`** → the floor-straddle confound
  (C2) is real and large → rasterization MUST be restricted to the above-`_ROAD_Z` slab (sealed below).
- ~34% of boxes are in-range (|x|,|y| < 40 m).

## Sealed grid (must equal the graded voxelizer exactly — `av2_sensor.py:32-41,73-75`)
`VOXEL_SIZE=0.4`, `GRID_SHAPE=(200,200,16)`, `RANGE=((-40,40),(-40,40),(-1,5.4))`, `ORIGIN`,
admissibility filter (identical to `_voxelize`): a voxel is admissible iff `x∈[-40,40) ∧ y∈[-40,40) ∧
z∈(_ROAD_Z=0.3, 5.4) ∧ NOT in ego cuboid (x∈(-1.1,3.9) ∧ |y|<1.05)`.

## Estimand (sealed)
For each tracked box `b` at frame `t` (annotations are AV2 ego-frame, ego at origin, +x fwd / +y left):
1. **Box footprint → admissible voxels.** BEV oriented-rectangle mask via reuse of
   `oracle_traversal._rect_mask(XX,YY, tx, ty, yaw, length, width)` over `_bev_centers()`, with
   `yaw = _quat_yaw(qw,qx,qy,qz)` (`av2_sensor.py:64`; sealed: yaw-only, boxes near-upright). Vertical
   extent = voxel-z centers in `[max(tz−H/2, _ROAD_Z), min(tz+H/2, 5.4)]` — the **above-road slab only**.
   Apply the full admissibility filter. → `box_voxels(b,t)`.
2. **Same-frame occupancy** `occ = _voxelize(sweep t)` (`fr.grid`). Per box:
   ```
   covered(b,t)  = |box_voxels(b,t) ∧ (occ == OCCUPIED)|
   miss(b,t)     = 1 if covered(b,t) == 0 else 0          # the box is ENTIRELY FREE in occupancy
   ```
   **Sealed primary = per-box binary MISS** (does the map see the object AT ALL — the safety-relevant
   recall failure). Per-voxel coverage is reported descriptively only, NOT as the kill metric (a real
   object returns points on its near face, not its whole volume — confound C5 — so per-voxel coverage
   reads spuriously low).
3. **Inclusion set (the `num_interior_pts` gate, the crux):**
   - `pts==0` → LiDAR returned nothing → occupancy CANNOT mark it → **EXCLUDED** (reported as a separate
     sensor-blind stratum). This is the single most important inclusion rule (controls confound C6).
   - `1 ≤ pts < N` → sparsely seen → **EXCLUDED** from primary (sensitivity stratum).
   - `pts ≥ N` → LiDAR returned ≥N points → **INCLUDED**. A FREE verdict here = the pipeline dropped
     returns it had.
   **Sealed N = 5** (mechanism: the voxelizer marks OCCUPIED on ≥1 admissible return, so a box with ≥5
   above-road returns must light ≥1 admissible voxel unless the pipeline drops it; N=5 matches the
   repo's existing `_MIN_SWEEP_CELLS=5` structural threshold — a within-repo precedent, not a tuned
   value). Report the full `N`-curve over {1,3,5,10,20} (CLAUDE.md: a curve, not a movable cutoff); the
   load-bearing claim is the gap at N=5, with the curve showing the verdict is not knife-edge on N.
4. **Aggregation:** per-frame estimand = mean `miss(b,t)` over included boxes; drop frames with
   `< _MIN_BOXES = 3` included boxes. Bootstrap over per-box rows tagged `scene = log_uuid`, clustered on
   scene, via `harness_v2._boot_mean(vals, scene_ids, rng, n_boot=1000)`; `defined=False` if <4 usable
   frames → INDETERMINATE.

## The null (region-local, REACHABLE)
For each included box, relocate its footprint (keeping length/width/yaw/vertical-slab) to a uniformly
random admissible column in a **matched region**: (i) forward range `x` within ±1 range-bin (sealed bin
= 8 m) of the real box, (ii) on the **empirical on-road support** = the union of BEV columns occupied by
any included box across the log (needs no map download; the within-substrate analogue of the stereo
oracle's band-local support). Compute `miss(b',t)` against the SAME `occ`; bootstrap identically.
- **Why reachable:** a box relocated ANYWHERE lands in FREE cells ~always (occupancy is sparse), so
  relocate-anywhere `miss(b')≈1` and the kill is unreachable (the stereo lesson). The range/on-road
  match controls for occupancy's base occupied-density at that annulus; the only difference is whether a
  real object is there. RELIABLE ⟺ occupancy is FREE at real boxes LESS than at matched random.
- **Sealed null-reachability pre-condition** (computed at the run, before reading the verdict):
  `0.3 < null_miss < 0.97`. If `null_miss` saturates outside this, the null is degenerate →
  INDETERMINATE-BY-NULL (a sealed apparatus outcome, not a pass).

## Falsifiable kill (declared before data)
Both bootstrap CIs defined, one-sided (recall):
- **RECALL-SUPPORTED** iff `true_miss.hi < null_miss.lo` (real boxes missed strictly less than matched random).
- **FAIL** iff `true_miss.lo ≥ null_miss.lo` (occupancy's FREE verdict at real LiDAR-seen obstacles is no
  better than at random matched on-road locations → the recall half of the denotation story fails).
  Reported as the headline (a negative is a headline).
- **INDETERMINATE** otherwise (partial overlap; <4 usable frames; or the null-reachability pre-condition fails).
**"This observation means I am wrong":** if `true_miss` stays high even at `pts ≥ 20` (densely-seen
boxes), occupancy does not recall LiDAR-confirmed structure (or the apparatus is broken — disambiguated
by the N-curve + self-check, never by moving N). The kill is reachable because the matched null is a
genuine non-degenerate baseline a real recall failure would push `true_miss` up into.

## Confounds + bias direction (declared; verified numbers)
| # | Confound | Bias on `true_miss` | Control |
|---|---|---|---|
| C1 | Box/sweep temporal mismatch | INFLATES | **Zero** at the graded frame — ann_ts == sweep_ts exactly (verified). |
| C2 | `_ROAD_Z=0.3` floor straddle (88.1% of bottoms below floor) | INFLATES | Rasterize only the above-`_ROAD_Z` slab on BOTH arms; height-stratify; the floor-loss rate is itself the diagnostic. |
| C3 | `z=5.4` height-cap straddle (tall classes) | INFLATES (marginal) | Rasterize only `[_ROAD_Z,5.4)`; BUS/LARGE_VEHICLE stratum. |
| C4 | Partial occlusion (`1≤pts<N`) | INFLATES | Excluded by the N gate; miss-rate must DECREASE monotonically with N (sealed sanity check). |
| C5 | Box footprint ⊋ return locations (near-face only) | DEFLATES per-voxel coverage | Sealed primary is BINARY per-box (`covered≥1`); per-voxel coverage descriptive only. |
| C6 | Annotation beyond sensor (`pts==0`, 18.6%) | massively INFLATES | EXCLUDED by the `pts==0` gate — the key inclusion rule. |
| C7 | Null box lands on a real object | DEFLATES the gap (conservative) | Allowed (keeps the null simple + conservative against a spurious pass). |
**Net:** the dominant confounds (C2, C4, C6) all INFLATE `true_miss`; any measured `true_miss` is an
**UPPER BOUND**, so a RECALL-SUPPORTED verdict is conservative. C5/C7 bias against a spurious pass.

## Independence ledger (vs the occupancy voxelization)
| axis | this oracle (box-recall) | independent? |
|---|---|---|
| provenance | human-annotated cuboids | YES (annotation ≠ `_voxelize`) |
| algorithm | box→voxel rasterization + set intersection | YES (different code path) |
| modality | `num_interior_pts` from the SAME LiDAR sweep | NO (same active-TOF LiDAR, same vehicle/timestamp) |
Verdict: **same-modality internal-consistency check** (CLAUDE.md H3) — earns provenance + algorithm
independence, NOT modality. The cross-modality upgrade (restrict to camera-unoccluded boxes) is named as
a future variant needing a camera download + re-pre-registration.

## Self-check (run FIRST; no confirmatory data)
`--self-check` must pass before any confirmatory run: (a) a synthetic box at a known ego location
rasterizes to exactly the voxel indices `_voxelize` would assign for a point at its center (0-voxel
round-trip error); (b) a box centered in the ego cuboid → 0 admissible voxels; (c) a box entirely below
`_ROAD_Z` → 0 admissible voxels. If any fails, the rasterizer disagrees with `_voxelize` → no miss-rate reported.

## Oracle-validity falsifiers (kill the ORACLE, not occupancy — declared before building)
1. Self-check fails → no result.
2. `N`-monotonicity violated (miss-rate not decreasing across {1,3,5,10,20}) → the gate isn't isolating
   sensor-coverage from pipeline-loss → report the broken curve, not a verdict.
3. Null unreachable (`null_miss` ∉ (0.3,0.97)) → INDETERMINATE-BY-NULL.
4. `pts==0` stratum miss-rate ≈ `pts≥5` stratum → the gate does nothing → invalid premise.
5. Any log's ann_ts ≠ sweep_ts → drop that log (C1 reactivates).

## Sealed run (after this doc is committed; run ONCE)
```sh
source .venv/bin/activate
python experiments/occquery_v0/oracle_box_recall.py --self-check      # geometry; no confirmatory data
python experiments/occquery_v0/oracle_box_recall.py \
    --logs ALL --n-interior-min 5 --min-boxes 3 --range-bin-m 8 \
    --null on-road-matched --n-curve 1,3,5,10,20 --seed 0 \
    --out experiments/occquery_v0/results/oracle_box_recall.json
```
Report JSON: `true_miss_mean/ci`, `null_miss_mean/ci`, `verdict`, the N-curve, the `pts==0` /
`1≤pts<5` / height / class strata, and the thesis framing string.

## Honest scope (goes in the summary; do NOT re-inflate)
- RECALL, one-sided (real LiDAR-seen obstacles occupancy marks FREE) — complement of the PASSED FP oracle.
- SAME-MODALITY internal-consistency check, NOT external truth. Earns provenance + algorithm independence only.
- `true_miss` is an UPPER BOUND (C2/C4/C6 inflate it) → RECALL-SUPPORTED is conservative.
- Contribution = diagnosing voxelization/threshold/filter losses with per-stratum attribution; not a verified recall P/R/F1.
- The cross-modality (camera-visibility) upgrade requires a download + fresh pre-registration.

## Seal checklist (verify BEFORE the confirmatory run)
- [x] Grid spec + admissibility copied to equal `_voxelize` exactly.
- [x] N=5 mechanism-derived; N-curve {1,3,5,10,20} sealed; ann_ts==sweep_ts verified (C1=0); 88.1% floor-straddle → above-road slab sealed.
- [x] Region-local matched null + reachability pre-condition (0.3,0.97) sealed.
- [x] Kill rule (RECALL-SUPPORTED/FAIL/INDETERMINATE) + oracle-validity falsifiers declared.
- [ ] `--self-check` passes (0-voxel round-trip). **(required before the run)**
- [ ] This doc committed (git + timestamp) BEFORE the run; confirmatory executed EXACTLY once.
