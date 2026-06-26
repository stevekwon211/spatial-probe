# H3b — occupancy-native vs box-only denotation, the SOLO/CPU honest version (PRE-REGISTRATION, seal before data 2026-06-26)

The program's load-bearing claim is "occupancy-native free-space predicates beat box-only (RefAV) — the
≥20-F1 denotation gap." This session established (3 pre-registered kills: stereo-following, DAv2-scale,
stereo-free-driving) that a NON-CIRCULAR, EXTERNAL, BOTH-SIDED denotation oracle is NOT solo/CPU-achievable
on AV2 — exactly the "ORACLE CIRCULAR → report only H1" branch occquery.md:27 pre-registered. So the full
both-sided ≥20-F1 empirical gap is NOT claimable solo. This pre-reg claims the part that IS rigorous + solo/
CPU + non-circular, and bounds the rest to GPU. Nothing below chosen after seeing a number.

## Claim (two legs, both solo/CPU)
**Leg 1 — EXPRESSIVITY (oracle-free, structural).** Box-only (RefAV's released 32-function set: tracked
boxes + HD-map polygons only) CANNOT express free-space queries (no dense-occupancy / free-space
primitive). On the SEALED query set (`queries.yaml`, the `refav_expressible` flags verified against
RefAV's `atomic_functions.py`), report the COVERAGE: fraction each backend can express, run over REAL AV2
logs (not synthetic). Expected: occupancy expresses the free-space family, box-only ~0 of it. This is a
structural proof made into a real coverage number, not a graded score.

**Leg 2 — FP-SIDE DENOTATION (empirical, NON-CIRCULAR oracle).** For the free-space queries occupancy CAN
express, is its denotation CORRECT? Graded against the INDEPENDENT traversal oracle (ego/agent future
trajectory = physically-FREE ground truth; recorded poses ≠ LiDAR voxelization, rigid sweep ≠ voxelize —
the oracle that is RELIABLE, sealed `oracle_traversal_v0_1`). On the driven-free ribbon: occupancy's
free-space predicate denotation correctness (it should match — traversal RELIABLE means occupancy ~never
false-blocks the driven path). Box-only has NO free-space denotation to grade (can't express) → on these
queries box-only is INAPPLICABLE, not merely worse. The empirical anchor: occupancy free-space denotation
is traversal-validated; box-only cannot produce one.

## Honest bound (declared, the wall)
The full both-sided ≥20-F1 gap additionally needs BLOCKED-side truth (does each language correctly denote
real obstacles, incl. UNBOXED ones?). Unboxed-obstacle ground truth = a cross-modal recall oracle =
empirically CLOSED solo/CPU this session (stereo AUC 0.259 / DAv2 scale >9 m / free-driving vacuity),
GPU-gated. So the BLOCKED-side denotation gap is NOT claimed here; it is named GPU-future. We claim:
expressivity dominance (Leg 1) + FP-side denotation correctness with box-only inapplicability (Leg 2).

## Independence ledger
- Leg 1: no oracle (structural — RefAV's function signatures vs the query semantics). Non-circular by construction.
- Leg 2: traversal oracle — modality (recorded poses, GPS/IMU-derived city_SE3) ≠ active LiDAR; algorithm
  (rigid-body sweep) ≠ voxelization. RELIABLE + sealed. Genuinely independent of the occupancy it grades.

## Kill (reachable, declared before data)
- **H1 falsified** iff box-only CAN express the free-space family (coverage gap collapses) — would sink the
  whole occupancy angle. (Structural; not expected, but the run reports the real flags, not assumed.)
- **Leg-2 fails** iff occupancy's free-space denotation is NOT correct on the traversal-FREE truth (it
  false-blocks the driven path) — i.e. traversal were UNRELIABLE. (Already RELIABLE; re-confirmed here on
  the query predicates specifically.)
- **BOX-MATCHES (the program's primary kill)** would fire only with the both-sided oracle we cannot build
  solo → that kill is HELD, explicitly deferred to GPU, NOT silently passed.

## Run (after commit; real AV2)
Run the sealed query set over real AV2 logs: per query, backend = occupancy vs box-only(tracking); report
(a) expressivity coverage per family, (b) for occupancy free-space queries, denotation agreement with the
traversal-FREE truth on the driven ribbon. Output `results/h3b_expressivity.json` + a summary.

## Honest scope
H1 expressivity is the rigorous headline (oracle-free). Leg 2 is a ONE-SIDED (FP/free) empirical denotation
validation via an independent oracle. The both-sided ≥20-F1 gap stays GPU-gated. No inflation: we do not
claim "occupancy beats box-only on denotation F1 both-sided" — we claim it EXPRESSES what box-only can't
AND its free-side denotation is independently validated, with the blocked-side honestly open.
