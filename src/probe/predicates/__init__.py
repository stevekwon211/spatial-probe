# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Physical predicates over a scene.

The occupancy-native predicates (clearance, free-path, free-width) are the OccQuery core: pure
geometric functions a box-only query language cannot express. `distance_to_nearest_object` is the
box-only BASELINE, kept here for one import surface but deliberately separate in semantics -- it
sees object boxes, not free space.
"""
from probe.predicates.clearance import centerline_lateral_distance, lateral_clearance
from probe.predicates.freepath import free_along_ego_path, min_free_width_along_path
from probe.predicates.objects import distance_to_nearest_object

__all__ = [
    "centerline_lateral_distance",
    "lateral_clearance",
    "free_along_ego_path",
    "min_free_width_along_path",
    "distance_to_nearest_object",
]
