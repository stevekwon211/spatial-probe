# occquery H3 via future-reveal — pre-registration (2026-06-23)

Committed BEFORE building the grader or looking at any agreement/F1 number. [research-integrity.md](../../docs/research-integrity.md)
applies: changing the hypothesis, the metric, or the kill criterion after seeing data is HARKing and
shows in this file's git history. This seals the design choice the owner approved ("(b) single-frame
view") for occquery's H3 denotation claim, which the sealed [preregistration.md](preregistration.md)
demoted to internal-consistency for lack of an INDEPENDENT oracle. Future-reveal supplies that oracle.

## The question (H3, finally with an independent oracle)

Does the occupancy free-space predicate DENOTE free-space correctly on regions it could not directly
observe — i.e., when it extrapolates free/occupied into space the frame-t sensor did not see, does a
later DIRECT observation confirm it — and does it beat a box-only baseline at this? The feasibility
probe (`future_reveal_density.py`, committed a041aeb) confirmed the precondition: 3,000-4,700 near,
static voxels per frame are unobserved at t but directly observed at t+k (both FREE ~3-4k and OCCUPIED
~600), verified three ways (k=0 identity = 0; count scales with ego motion).

## The independent oracle (the grader)

For a voxel V that is UNOBSERVED at frame t (`mask_lidar_t[V] == 0`) and DIRECTLY OBSERVED at frame
t+k, the **reveal-truth** of V comes from the RAW t+k LiDAR sweep (`data/samples/LIDAR_TOP/*.pcd.bin`),
carved independently by our own raycaster (`probe.raycast`):
- a return lands inside V → **OCCUPIED**;
- a ray passes through V to a return beyond it → **FREE**;
- no ray reaches V (occluded even at t+k) → **unobserved**, EXCLUDED.

Independence level, stated honestly: this is **temporal / later-direct-observation** independence —
a DIFFERENT sweep at a different time and vantage, carved by a DIFFERENT algorithm (direct
point-carving) than the predicate (reachable-free flood-fill). It is NOT cross-modality (both are the
same nuScenes LiDAR sensor type). It is valid for STATIC structure only (a static voxel's free/occupied
state is the same at t and t+k; a dynamic actor's is not — hence the static-only scope below). Crucially,
the predicate at t is forbidden from having seen the t+k sweep (guaranteed by design (b): the predicate
reads only the frame-t observed view), so the grader is not one of the predicate's own inputs — the
circularity that sank the dense-GT oracle (same data + same EDT, `preregistration.md` L16-21) does not
recur here. The grader is **NOT** the Occ3D accumulated `semantics` at t+k (that is multi-sweep
accumulated and shares the predicate's data lineage); it is the raw single t+k sweep.

## The instrument under test (design (b), owner-approved)

The occupancy free-space primitive `reachable_free_field` computed on the **single-frame-t OBSERVED
view** (`load_scene(..., mask='lidar')`): only what the t-sweep saw is FREE/OCCUPIED; everything else is
UNKNOWN. The predicate must EXTRAPOLATE reachable-free space across UNKNOWN voxels — those extrapolations
are exactly what the reveal grades. The extrapolation is governed by the unknown policy:
- **PRIMARY: unknown→free** (the predicate claims reachable-free space into unobserved regions — the
  case the reveal can actually falsify, and the occupancy-native claim worth testing).
- Reported as a curve alongside: unknown→occupied (conservative; claims nothing free in unobserved, so
  it can only be graded on revealed-occupied) and unknown→ignored.

## The baseline (the relative-gap anchor)

Box-only free-space: a voxel is FREE iff no nuScenes 3D box contains it at frame t (`with_boxes=True`,
the same boxes the RefAV-style baseline uses). Box-only is blind to box-less structure (walls, curbs,
vegetation, construction) — occquery's whole thesis. The H3 claim is the RELATIVE gap of the occupancy
predicate OVER box-only on the reveal oracle, never an absolute F1.

## Test set

Revealed voxels: UNOBSERVED@t, DIRECTLY OBSERVED@t+k (raw sweep), STATIC (semantics≥11), NEAR
(|x,y|≤15 m in ego-t frame), over k∈{1,3,5} (k=10 reported as a robustness point). The LOAD-BEARING
discriminator is the revealed-**OCCUPIED** subset (~600/frame): box-less structure that box-only calls
free and the occupancy predicate should call occupied. Revealed-FREE (~3-4k/frame) is the
specificity check (both should call it free; an occupancy predicate that over-claims occupied loses
here).

## Primary metric + analysis path (fixed now)

- Per revealed voxel: `occ_pred(V)` ∈ {free, occupied} (reachable-free or not, single-frame-t view,
  unknown→free) vs `box_pred(V)` vs `reveal_truth(V)`.
- **F1 of occ_pred against reveal_truth, treating OCCUPIED as the positive class** (catching box-less
  structure is the skill under test), and the same F1 for box_pred.
- **Primary statistic = the GAP `F1(occ) − F1(box)`**, with a scene-clustered bootstrap 95% CI
  (scene = resampling unit, as in `metrics.py`). Reported as a curve over k and over the unknown policy,
  never one cutoff.
- Secondary: precision/recall split (is the gap from catching more structure = recall, or fewer false
  occupied = precision), and the revealed-FREE specificity (occ_pred must not collapse to "occupied
  everywhere").

## Pre-registered kill criteria (reachable, stated before data)

- **KILL (falsified):** if the gap `F1(occ) − F1(box)` CI INCLUDES 0 across all k under the primary
  (unknown→free) policy — i.e., the occupancy predicate does NOT beat box-only at denoting box-less
  structure on the independent reveal — then H3-via-reveal is FALSIFIED. Report the negative as the
  headline (the occupancy predicate's free-space extrapolation is no better than box-only when graded by
  an independent later observation), and do NOT tune the unknown policy or near-zone to manufacture a
  gap.
- **HOLDS:** gap CI lower bound > 0 across k → the occupancy predicate denotes box-less free/occupied
  structure better than box-only, verified by an independent oracle. This is the H3 result the sealed
  prereg said did not yet exist; it would promote H3 from internal-consistency to externally-graded.
- **INDETERMINATE:** CIs straddle 0 for some k but not others → report the curve, claim nothing beyond
  where the lower bound clears 0.

## Controls + leak channels (each enumerated, none hand-waved)

1. **Registration error (ego_pose).** The t↔t+k transform has finite error; a mis-mapped voxel could
   falsely read unobserved@t. CONTROL: k=0 identity returns exactly 0 (verified, transform not broken);
   report sensitivity to a ±1-voxel (0.4 m) registration dilation as a tolerance band on the gap curve.
2. **Dynamic actors (temporal leak).** A moving object differs between t and t+k. CONTROL: static-only
   (semantics≥11) excludes dynamic classes 0-10; additionally drop any revealed voxel whose t+k carve
   conflicts with a t+k nuScenes box (a dynamic object that re-entered).
3. **Grader independence.** Use the RAW t+k sweep carved by our raycaster, NOT Occ3D accumulated
   semantics (shared lineage). The predicate (single-frame-t) definitionally never saw the t+k sweep.
4. **Same-sensor-type honesty.** Independence is temporal, not cross-modal; stated in every result. A
   cross-modal corroboration (V2V4Real, a different vehicle's LiDAR) remains a GATED follow-on, not part
   of this claim (`docs/research-program/substrate-and-oracle.md`).
5. **Unknown-policy forking-path guard.** unknown→free is pre-registered as PRIMARY here; the other
   policies are reported as a curve, and the kill criterion is evaluated on the primary only — the
   policy is not selected using the F1 output.

## Scope / honest ceiling

- **Static free-space only.** Says nothing about dynamic obstacles (that is dynfield's axis).
- **Cannot grade permanently-occluded voxels** (occluded at t AND t+k) — a structural blind spot, not a
  coverage gap one more scene closes.
- **Resolution floor 0.4 m** (Occ3D voxel) — sub-voxel clearance unresolvable, same ceiling as
  `h3-real-data-findings.md`.
- **Temporal, not cross-modal, independence** — the strongest real-data oracle available with zero new
  download, weaker than a second modality; the claim is scoped to "later direct observation," not
  "ground truth."
- **Audited subset, relative gap** — the claim is the occ−box gap on the revealed subset with bootstrap
  CIs, never an absolute benchmark F1.

## Build order (after this file is committed)

1. Raw-sweep carver: load `LIDAR_TOP/*.pcd.bin` for frame t+k, raycast into the ego-t+k grid → per-voxel
   reveal-truth (reuse `probe.raycast`). Verify on one frame three ways (counts + a known free corridor +
   a rendered BEV).
2. Grader harness: for each revealed voxel, collect (occ_pred, box_pred, reveal_truth); compute the
   gap + bootstrap CI per k and per unknown policy.
3. Run on a pre-registered scene budget (start ~40 scenes, extend only if CIs need it — the budget, not
   the result, sets the extension), verify three ways, write `results/h3_future_reveal.md`.

## Addendum (2026-06-23, data-availability fact discovered AFTER sealing — design unchanged)

The raw-sweep grader needs `LIDAR_TOP/*.pcd.bin` at both t and t+k. Local raw sweeps cover exactly the
**10 nuScenes-mini scenes** (scene-0061/0103/0553/0655/0757/0796/0916/1077/1094/1100), ~40 consecutive
key-frames each, all with Occ3D GT — so the oracle IS buildable (t and t+k sweeps both present), but the
audited subset is **10 scenes = 10 scene-clustered bootstrap units**, not the "~40" the build-order line
hoped. This changes only the achievable N (wider CIs, lower power), NOT the hypothesis, metric, or kill
criterion (all still sealed above). The feasibility probe's 850-scene geometry count stands (it needs
only mask_lidar, not raw sweeps). Extending N requires downloading more raw nuScenes sweeps
(account-gated). Honest consequence: a 10-scene relative-gap claim with bootstrap CIs is a small-but-hard
audited subset (the framing the program already adopted), reported as such; if the gap CI is too wide to
clear 0, that is an under-power outcome to report, not a kill to hide.
