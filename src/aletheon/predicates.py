# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S3 physical predicates for SceneQL -- the occupancy-native edge over a box-only language.

These extend the box/free-space predicates already in `probe.predicates` with three quantities a
cuboid query language structurally cannot express well:

- `occluded` -- line-of-sight from the ego to a target POINT, decided by the dense occupancy
  between them (reuses `probe.raycast.line_of_sight`, the load-bearing DDA primitive). A box tool
  can test "is there a box on the segment" but not "does unboxed occupancy (a wall, debris) block
  the view"; this does. THE edge predicate.
- `velocity` -- the speed of an `Entity`/`TrackedBox` from its stored 2D velocity, explicit about
  an unknown (None, never a silent 0). `object_speed` is the SceneQL-callable per-frame form.
- `ttc` -- time-to-collision along the ego heading from relative position + velocity. SHIPPED
  FLAGGED: the dynfield pre-registered NEGATIVE found velocity action-EQUIVALENT to boxes in the
  dominant safe regime (see `ttc`'s docstring). It is a primitive, not a validated danger signal.

All values are Aletheon/python types (floats, bools, None). Targets are world-XY points in the same
frame the predicates already reason in (the probe ego/world frame the grid is calibrated to).
"""
from __future__ import annotations

import math
from typing import Optional, Union

from probe.grid import OccupancyGrid
from probe.raycast import line_of_sight
from probe.scene import Scene, TrackedBox
from aletheon.ir import Entity, SceneIR

__all__ = ["occluded", "velocity", "object_speed", "ttc"]

# A SceneQL predicate may be called with either the IR or the bare probe.Scene (the query engine
# passes whatever the public call received). Both expose grids/ego/objects via the Scene API.
SceneLike = Union[SceneIR, Scene]


def _scene(scene_like: SceneLike) -> Scene:
    """The underlying probe.Scene, whether given a SceneIR wrapper or a bare Scene."""
    return scene_like.scene if isinstance(scene_like, SceneIR) else scene_like


def occluded(
    scene: SceneLike,
    t: int,
    target_x: float,
    target_y: float,
    *,
    target_z: Optional[float] = None,
    unknown_blocks: bool = False,
) -> bool:
    """True iff the ego's line of sight to (`target_x`, `target_y`) is blocked by occupancy at `t`.

    This is the occupancy-native occlusion a box-only language cannot express: it marches a ray
    through the DENSE occupancy grid (via `probe.raycast.line_of_sight`) from the ego voxel to the
    target voxel and reports whether any solid voxel lies strictly between. A box query can only
    ever ask about object boxes on the path; unboxed obstacles (a wall, a barrier, debris) are
    invisible to it but block sight here.

    TARGET-POINT CONVENTION (a choice, stated): the target is a world (x, y) point in the same
    frame the grid is calibrated to (the probe world frame), passed as two SCALAR args -- not a
    tuple -- because the SceneQL AST whitelist (`probe.query_dsl`) admits constants but not tuple
    literals, and that whitelist is the untouched untrusted-input boundary. `target_z` defaults to
    the ego's z (eye-level, a horizontal sight line) when omitted -- never silently 0, which would
    aim the ray at the ground. `unknown_blocks` (default False) decides whether UNKNOWN voxels also
    block, the same conservative/optimistic knob `line_of_sight` exposes (kept independent of
    UnknownPolicy so occlusion's reading is explicit at the call site).
    """
    sc = _scene(scene)
    grid: OccupancyGrid = sc.grid_at(t)
    ego = sc.ego_at(t)
    ez = ego.position[2]
    tz = ez if target_z is None else float(target_z)
    src = grid.world_to_voxel((ego.position[0], ego.position[1], ez))
    dst = grid.world_to_voxel((float(target_x), float(target_y), tz))
    nx, ny, nz = grid.shape
    for cx, cy, cz in (src, dst):
        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz):
            raise ValueError(
                f"occluded: voxel {(cx, cy, cz)} is outside grid shape {grid.shape}; "
                "ego or target_xy lies off the calibrated grid"
            )
    return not line_of_sight(grid.occupancy, src, dst, unknown_blocks=unknown_blocks)


def velocity(entity: Union[Entity, TrackedBox]) -> Optional[float]:
    """Speed (m/s) of an `Entity` or `TrackedBox` from its 2D velocity, or None if unknown.

    Returns hypot(vx, vy). A NaN component means the adapter could not difference a velocity for
    this object -- that is reported as None (explicit unknown), NEVER a silent 0. A measured (0, 0)
    is a real 0.0 (a stationary object), distinct from None. (Repo rule: explicit failure over
    silent fallback; matches `Entity.velocity` carrying (nan, nan) for unknown.)
    """
    vx, vy = entity.velocity
    if math.isnan(vx) or math.isnan(vy):
        return None
    return math.hypot(float(vx), float(vy))


def object_speed(
    scene: SceneLike,
    t: int,
    *,
    object_class: Optional[str] = None,
    aggregate: str = "max",
) -> Optional[float]:
    """SceneQL-callable per-frame object speed: aggregate the per-object `velocity` over the boxes
    at frame `t` (optionally filtered by class).

    `aggregate` is 'max' (default) or 'min' over the objects that HAVE a known velocity. Returns
    None if there is no object with a known velocity (explicit -- not 0). Objects with unknown
    velocity are skipped, not counted as 0 (that would understate a 'max' and fake a 'min').
    """
    sc = _scene(scene)
    objects = sc.objects_at(t)
    if object_class is not None:
        objects = tuple(o for o in objects if o.label == object_class)
    speeds = [s for s in (velocity(o) for o in objects) if s is not None]
    if not speeds:
        return None
    if aggregate == "max":
        return max(speeds)
    if aggregate == "min":
        return min(speeds)
    raise ValueError(f"object_speed: aggregate must be 'max' or 'min', got {aggregate!r}")


def ttc(
    scene: SceneLike,
    t: int,
    *,
    object_class: Optional[str] = None,
    half_angle_deg: float = 30.0,
) -> float:
    """Time-to-collision (s) along the ego heading to the nearest CLOSING object at frame `t`.

    *** dynfield pre-registered NEGATIVE *** -- occupancy/velocity is action-EQUIVALENT to boxes
    in the dominant safe-following regime (experiments/dynfield_v0: on 443 held-out val lead-frames
    the TRUE velocity's effect on the planner-surrogate's action sat ENTIRELY BELOW the
    shuffled-velocity band). `ttc` is provided as a PRIMITIVE, NOT a validated headline signal. DO
    NOT build a danger claim on it -- the "necessary when dangerous" half is untestable on nuScenes
    (almost no danger frames), so ttc has no externally-validated denotation here.

    Mechanics (a primitive, by-construction correct): for each object box whose center is ahead of
    the ego (within `half_angle_deg` of the heading), the closing speed is the component of the
    relative velocity (object minus ego) along the ego->object direction, taken POSITIVE toward the
    ego. If closing, candidate ttc = forward range / closing speed. Returns the minimum over
    closing objects; math.inf when nothing is closing or no object has a known velocity. Objects
    with unknown (NaN) velocity are skipped -- never treated as stationary.
    """
    sc = _scene(scene)
    ego = sc.ego_at(t)
    ex, ey = ego.position[0], ego.position[1]
    fx, fy = ego.forward
    evx = ego.speed * fx
    evy = ego.speed * fy
    objects = sc.objects_at(t)
    if object_class is not None:
        objects = tuple(o for o in objects if o.label == object_class)
    cos_thresh = math.cos(math.radians(half_angle_deg))
    best = math.inf
    for o in objects:
        ovx, ovy = o.velocity
        if math.isnan(ovx) or math.isnan(ovy):
            continue  # unknown velocity -> cannot compute closing speed; skip, never assume 0
        dx = o.center[0] - ex
        dy = o.center[1] - ey
        rng = math.hypot(dx, dy)
        if rng == 0.0:
            return 0.0  # already coincident
        # forward gate: object must be roughly ahead of the ego heading.
        if (dx * fx + dy * fy) / rng < cos_thresh:
            continue
        ux, uy = dx / rng, dy / rng  # ego -> object unit vector
        rel_vx, rel_vy = ovx - evx, ovy - evy
        closing = -(rel_vx * ux + rel_vy * uy)  # positive when the gap shrinks
        if closing <= 0.0:
            continue
        best = min(best, rng / closing)
    return best
