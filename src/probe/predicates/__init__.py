# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Falsifiable physical predicates over an occupancy field.

Each predicate is a pure geometric function of (OccupancyGrid, EgoPose, ...) that a
box-only query language cannot express. The set grows by accretion; v0 ships two.
"""
from probe.predicates.clearance import lateral_clearance
from probe.predicates.freepath import free_along_ego_path

__all__ = ["lateral_clearance", "free_along_ego_path"]
