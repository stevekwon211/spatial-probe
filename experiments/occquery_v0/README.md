# occquery_v0 — occupancy-native physical-predicate retrieval

First falsifiable experiment. Full rationale in [`../../PLAN.md`](../../PLAN.md) §1–§7;
this is the runnable protocol.

## Claim

Spatial queries that matter for safety (tight clearance, blocked/narrowing free path)
are **expressible as predicates over the occupancy field** but **inexpressible** with an
object-box query language (RefAV's 28 cuboid+velocity functions) — and executing them on
occupancy is **denotation-correct** against a GT-occupancy oracle.

## Predicates (v0)

- `lateral_clearance(scene, t)` → meters. Min horizontal distance from the ego lane
  corridor to the nearest occupied, non-ground voxel.
- `free_along_ego_path(scene, t, horizon)` → bool. Ego swept-footprint over
  `[t, t+horizon]` stays collision-free in the occupancy grid.

## Protocol

1. Hand-compile ~20 NL queries → these predicates (see `queries.yaml`), each tagged with
   whether RefAV's function set can express it.
2. Oracle = GT occupancy (Occ3D) + GT ego pose/box → "true" predicate values.
3. Run on nuScenes-**mini**, then **val**.
4. **Unobserved-voxel sensitivity:** evaluate under 3 rules (unobserved = free / occupied
   / excluded) and report denotation spread across them.

## Metrics

- expressibility coverage: # expressible by occupancy vs by RefAV.
- denotation precision / recall / **F1** vs hand-labeled GT set.
- clearance MAE + tolerance-accuracy ([75%,125%] of GT).

## Decision thresholds

- **Success:** expressivity separation is clean (occupancy ≈ N, RefAV ≈ 0 on the spatial
  subset) AND denotation F1 > 0.9 with low MAE, **stable across the 3 unobserved rules.**
- **Kill / pivot:** denotation flips across the 3 rules on GT occupancy → the
  predicate-on-Occ3D premise is shaky → pivot to dense-LiDAR free-space (accumulated
  sweeps) and re-test. (An honest, acceptable outcome — record it.)

Write the outcome to `results/summary.md`.
