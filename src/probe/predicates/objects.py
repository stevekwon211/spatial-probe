# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Tracking-backend baseline predicate: `distance_to_nearest_object`.

This is what a box-only query language (RefAV's cuboid functions) can compute -- it sees object
boxes, not free space. It lives apart from the occupancy predicates to keep the fairness
boundary explicit: the occupancy core never calls it, the fairness-control query uses it, and
the expressivity witness uses it to show that a box-only backend is structurally blind to
unboxed obstacles.
"""
from __future__ import annotations

import math

from probe.scene import Scene


def distance_to_nearest_object(scene: Scene, t: int, *, object_class: str | None = None) -> float:
    """Min planar (x, y) distance in meters from the ego to the nearest tracked object box at
    frame `t`, optionally filtered by class label. math.inf if there is no such object. Ignores
    occupancy entirely -- it can only see boxes.
    """
    ego = scene.ego_at(t)
    objects = scene.objects_at(t)
    if object_class is not None:
        objects = tuple(o for o in objects if o.label == object_class)
    if not objects:
        return math.inf
    ex, ey = ego.position[0], ego.position[1]
    return min(math.hypot(o.center[0] - ex, o.center[1] - ey) for o in objects)
