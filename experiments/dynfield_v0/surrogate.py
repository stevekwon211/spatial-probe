# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Analytic planner-surrogate for dynfield (Stage 2, dynamics/time axis).

dynfield asks: WHICH stored motion field does a planner actually need, and in WHICH regime? occquery
(Stage 1) proved an occupancy predicate MEASURES a static fact (free-width = 0.8 m beside something)
but, by its own scene-0061 lead-car finding, cannot tell a passable gap beside a FOLLOWING lead car
from an impassable gap beside a WALL -- that verdict needs relative motion over time. dynfield is the
necessity question one layer up.

This module is the deterministic, analytic planner-surrogate the necessity measurement ablates around
-- pure numpy, no learned weights, no GPU (the Mac-feasible Tier-1 core; the nuPlan closed-loop
reproduction is GPU-gated Tier-2, deferred). A surrogate reads a STORED STATE and emits a longitudinal
ACTION (the deceleration it would command to stay safe). Two configurations differ in EXACTLY which
stored field they may read:

- `plan_static_only` reads ONLY the static occupancy summary (distance to the nearest in-path
  obstacle). It cannot see motion, so two scenes with identical occupancy get the identical action.
- `plan_motion_aware` also reads the stored per-object MOTION field (relative closing speed), so it
  separates a closing obstacle (brake) from a receding one (proceed).

The gap between the two, by regime, is the necessity matrix (SH2/SH3). The headline SH1 claim needs
no oracle: it is a non-identifiability CONSTRUCTION (identical static input MUST yield identical
action), the dynamics analogue of occquery's expressivity witness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_A_MAX = 4.0          # m/s^2, comfortable max decel the surrogate is willing to command
_D_SAFE = 5.0         # m, static standoff: brake if the in-path obstacle is closer than this
_TTC_SAFE = 3.0       # s, motion gate: brake if time-to-contact is under this


@dataclass(frozen=True)
class StoredState:
    """The stored spatial state at one decision frame, as a planner would read it.

    `lead_distance_m` is the static-occupancy summary: forward distance to the nearest in-path
    occupied space (what occquery's reachable field already gives). `lead_rel_speed_mps` is the
    STORED MOTION field under test: closing speed of that obstacle relative to the ego (NEGATIVE =
    closing/approaching, POSITIVE = separating). It comes from the dataset's tracked-box velocity, NOT
    re-derived from the occupancy the surrogate also reads (the circular-oracle guard)."""

    lead_distance_m: float
    lead_rel_speed_mps: float


def plan_static_only(state: StoredState) -> float:
    """Longitudinal action (commanded decel, m/s^2) from the STATIC occupancy summary alone.

    Sees only distance; brakes when the obstacle is inside the static standoff. Two states with the
    same `lead_distance_m` get the SAME action regardless of motion -- that forced tie is the SH1
    witness's load-bearing property."""
    return _A_MAX if state.lead_distance_m < _D_SAFE else 0.0


def plan_motion_aware(state: StoredState) -> float:
    """Longitudinal action (commanded decel, m/s^2) using the stored MOTION field too.

    A separating obstacle (rel speed >= 0) is never an in-path threat -> proceed. A closing obstacle
    brakes when time-to-contact (distance / closing speed) drops under the gate. This separates the
    scene-0061 lead-car case (gap beside a following/receding object = benign) from an identical gap
    beside a closing object."""
    closing = -state.lead_rel_speed_mps  # positive when approaching
    if closing <= 0.0:
        return 0.0
    ttc = state.lead_distance_m / closing
    return _A_MAX if ttc < _TTC_SAFE else 0.0


# ---- v2: graded IDM longitudinal acceleration (continuous; the model PDM-Closed uses) ----
# Action = commanded acceleration (negative = braking). The ablated MOTION field enters ONLY through
# the closing-gap term v*dv/(2*sqrt(a*b)); static-only zeroes it. So decel-delta = the field's effect.
_IDM = dict(v0=13.9, s0=2.0, T=1.5, a=1.5, b=2.0, delta=4.0)  # urban defaults; pre-registered config


def _idm_accel(v_ego: float, gap: float, dv: float) -> float:
    """IDM acceleration (m/s^2; negative = braking). dv = closing speed (ego - lead; >0 approaching)."""
    p = _IDM
    s_star = p["s0"] + max(0.0, v_ego * p["T"] + v_ego * dv / (2.0 * math.sqrt(p["a"] * p["b"])))
    interaction = (s_star / gap) ** 2 if gap > 0.1 else 1e6
    return p["a"] * (1.0 - (v_ego / p["v0"]) ** p["delta"] - interaction)


def plan_idm_static(ego_speed: float, gap: float) -> float:
    """Graded action WITHOUT the motion field: IDM with the closing term dropped (gap only)."""
    return _idm_accel(ego_speed, gap, 0.0)


def plan_idm_motion(ego_speed: float, gap: float, lead_fwd_speed: float) -> float:
    """Graded action WITH the motion field: full IDM, closing = ego_speed - lead's forward speed."""
    return _idm_accel(ego_speed, gap, ego_speed - lead_fwd_speed)
