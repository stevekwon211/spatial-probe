# occquery_v0 -- synthetic smoke-test results

**Result class: SYNTHETIC-SMOKE.** Hand-built scenes with constructed ground truth. Verifies the retrieval loop and predicate behavior on known geometry; NOT an externally validated scientific result. External numbers require the M2 nuScenes/Occ3D adapter.

- seed: 0  (full commit hash + per-policy FP/FN are in results.json, gitignored)
- scenes (7): tight_pass, slow_near, open_road, narrowing_corridor, blocked_then_clears, unknown_side, near_vehicle
- expressibility coverage: occupancy 4/4, RefAV 1/4

## Occupancy queries (denotation vs constructed GT)

| query | scope | F1 (free) | F1 (occupied) | unknown-stable | GT |
|---|---|---|---|---|---|
| `tight_clearance_at_speed` | any | 1.0 | 0.667 | False | tight_pass |
| `corridor_narrows_below_vehicle_width` | any | 1.0 | 1.0 | True | narrowing_corridor |
| `blocked_then_clears` | transition | 1.0 | 1.0 | True | blocked_then_clears |

## Baseline (tracking backend; box-only, NOT occupancy retrieval)

- `near_a_tracked_vehicle` (tracking/baseline_only): F1=1.0, retrieved=['near_vehicle'], GT=['near_vehicle']

## Unknown-policy sensitivity

Retrieval under unknown=free vs unknown=occupied. A query whose retrieved set changes is unknown-SENSITIVE; under the IGNORED policy the flipping scenes are excluded as undetermined.
- `tight_clearance_at_speed`: SENSITIVE -- undetermined under IGNORED: ['unknown_side']

## Result categories (do not conflate)
- unit / integration tests: `pytest` (instrument + predicates), see CI/local run.
- synthetic smoke (this file): constructed GT, not external.
- externally validated benchmark: NONE yet -- requires the M2 Occ3D-nuScenes adapter.
