# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""free_along_ego_path: boolean companion to lateral_clearance.

Does the ego's swept footprint stay collision-free over a time horizon? v0 models the
sweep as a straight, constant-velocity extrusion of the ego box along its heading (no
curvature); the real logged-trajectory arm arrives with the dataset adapter (M2).
"""
from __future__ import annotations

import numpy as np

from probe.grid import EgoPose, OccupancyGrid


def free_along_ego_path(grid: OccupancyGrid, ego: EgoPose, horizon: float) -> bool:
    """True iff no non-ground obstacle lies in the ego's swept footprint over `horizon`.

    Footprint = forward in [0, length/2 + speed*horizon], |lateral| <= width/2 — the ego
    body plus a straight constant-velocity reach. horizon=0 tests the ego body alone (is it
    blocked *right now*); a larger horizon extends the corridor by speed*horizon meters.
    """
    centers = grid.nonground_occupied_centers()
    if len(centers) == 0:
        return True
    forward, lateral = ego.to_ego_frame(centers[:, :2])
    reach = ego.length / 2.0 + ego.speed * horizon
    hit = (forward >= 0.0) & (forward <= reach) & (np.abs(lateral) <= ego.width / 2.0)
    return not bool(hit.any())
