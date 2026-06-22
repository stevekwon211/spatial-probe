# Expressivity: occupancy predicates vs RefAV's box-only function set

H1 (PLAN §4) claims OccQuery's occupancy predicates express safety queries that RefAV's box-only
function language cannot. This is the oracle-free headline. Here it is checked against the RELEASED
source so anyone can reproduce it — no model, no dataset, no oracle.

## What was checked

RefAV's complete function set is the public `refAV/atomic_functions.py`
([github.com/CainanD/RefAV](https://github.com/CainanD/RefAV), `main`) — the 32 functions the LLM is
allowed to call to translate a natural-language query into executable code. Verified **2026-06-20**,
re-verified **2026-06-22** (32 public functions, 0 free-space primitives).

## The 32 functions, categorized

- **Object attributes**: `get_objects_of_category`, `is_category`, `is_color`, `within_camera_view`
- **Motion**: `turning`, `changing_lanes`, `has_lateral_acceleration`, `accelerating`,
  `has_velocity`, `stationary`, `facing_toward`, `heading_toward`, `heading_in_relative_direction_to`
- **Object / ego spatial relations**: `has_objects_in_relative_direction`,
  `get_objects_in_relative_direction`, `near_objects`, `being_crossed_by`, `following`,
  `on_relative_side_of_road`, `reverse_relationship`
- **HD-map relations**: `at_pedestrian_crossing`, `on_lane_type`, `near_intersection`,
  `on_intersection`, `in_drivable_area`, `on_road`, `in_same_lane`, `at_stop_sign`
- **Combinators**: `scenario_and`, `scenario_or`, `scenario_not`, `output_scenario`

Every function takes **object tracks** (`track_uuid: dict`, `candidate_uuids: dict`) and/or the **HD
map** (`log_dir`). No argument is a dense occupancy field or free space.

## The separation

`grep -iE "occupancy|voxel|free.?space|clearance|corridor|swept" atomic_functions.py` returns
**nothing** — zero free-space / occupancy primitives. The closest spatial functions are structurally
blind to unboxed geometry:

- `near_objects(track_uuid, candidate_uuids, log_dir, distance_thresh, ...)` — distance between
  *tracked object boxes*. An unboxed wall / debris / curb / protruding load has no `uuid`, so it is
  invisible to this function.
- `in_drivable_area(track_candidates, log_dir)` — a *static HD-map polygon*, not the scene's dynamic
  occupancy. It cannot tell that the drivable area is currently blocked by something with no box, nor
  measure how much free width remains.

So our three occupancy predicates are inexpressible in this set:

| our predicate | needs | the RefAV set has |
|---|---|---|
| `lateral_clearance` | distance to nearest *occupied space*, boxed or not | only distance to *boxed objects* (`near_objects`) |
| `free_along_ego_path` | dynamic *free space* along a swept path | only static drivable-area map polygons |
| `min_free_width_along_path` | width of *empty space* between obstacles | nothing about empty-space extent |

## Honesty about the claim

This is a **syntactic / observational** separation: the set has no free-space primitive, and its
object+map inputs do not observe unboxed occupancy, so no *composition* of these functions can denote
a property that depends on unboxed geometry. That is precisely what the executable
non-identifiability witness (`tests/test_expressivity.py`) shows — two scenes with identical box+map
observables but different occupancy: RefAV's functions receive identical inputs, so they MUST return
identical answers, while the occupancy predicate distinguishes them.

It is NOT a claim that RefAV "scores worse" on a shared metric (it cannot run on the query at all),
nor that *no conceivable* box-language could ever approximate it with extra engineering. The
defensible statement: **on the released RefAV function set, free-space predicates are not
expressible, and the witness proves a box+map observation is insufficient to denote them.**

## Reproduce

```sh
curl -sL https://raw.githubusercontent.com/CainanD/RefAV/main/refAV/atomic_functions.py -o atomic_functions.py
grep -iE "occupancy|voxel|free.?space|clearance|corridor|swept" atomic_functions.py   # -> nothing
python -m pytest tests/test_expressivity.py
```

(We do not vendor `atomic_functions.py` — it is RefAV's under their license; this file cites the
function names and signatures as fact and links to the source.)
