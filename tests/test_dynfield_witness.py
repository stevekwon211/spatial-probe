# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""dynfield SH1 non-identifiability witness + SH4 leakage control (Tier-0, oracle-free).

The dynamics analogue of occquery's expressivity witness. Two decision frames have IDENTICAL static
occupancy (same in-path obstacle at the same distance) but DIFFERENT stored motion (one obstacle
closing, one receding). A planner-surrogate that reads only the static occupancy MUST emit the
identical action on both (its input is identical) -- so a static-only stored state is INSUFFICIENT to
determine the action in this regime. A motion-aware surrogate distinguishes them. This proves, by
construction with no oracle, that SOME stored motion field is necessary SOMEWHERE -- the load-bearing,
dispute-proof dynfield headline (SH1). It directly formalizes occquery's scene-0061 lead-car finding.

Scope (same honesty bound as occquery H1): this is a non-identifiability result under the surrogate's
STATIC-OCCUPANCY observable set, not a claim about every planner. And it is Tier-0 SYNTHETIC SMOKE --
a by-construction existence proof, never a scientific necessity number. The real {field x regime}
matrix on Occ3D-nuScenes (Tier-1) and the closed-loop reproduction (Tier-2, GPU) are separate.
"""
from experiments.dynfield_v0.surrogate import StoredState, plan_motion_aware, plan_static_only

_D = 8.0           # m, identical in-path obstacle distance in BOTH scenes (static occupancy identical)
_REL_SPEED = 3.0   # m/s, magnitude of the relative motion that is the ONLY difference


def _witness_pair() -> tuple[StoredState, StoredState]:
    """(closing, receding): identical static occupancy (same lead_distance), differing ONLY in the
    stored motion field -- one obstacle closing at -3 m/s, one receding at +3 m/s."""
    closing = StoredState(lead_distance_m=_D, lead_rel_speed_mps=-_REL_SPEED)
    receding = StoredState(lead_distance_m=_D, lead_rel_speed_mps=+_REL_SPEED)
    return closing, receding


def test_witness_inputs_differ_only_in_motion():
    closing, receding = _witness_pair()
    # the static-occupancy observable (distance) is identical; only the stored motion field differs
    assert closing.lead_distance_m == receding.lead_distance_m
    assert closing.lead_rel_speed_mps != receding.lead_rel_speed_mps


def test_static_only_surrogate_cannot_distinguish():
    closing, receding = _witness_pair()
    # identical static input -> a static-only planner MUST return the identical action (forced tie)
    assert plan_static_only(closing) == plan_static_only(receding)


def test_motion_aware_surrogate_distinguishes():
    closing, receding = _witness_pair()
    # the motion field separates them: brake on the closing obstacle, proceed on the receding one
    assert plan_motion_aware(closing) > 0.0   # closing within TTC gate -> command decel
    assert plan_motion_aware(receding) == 0.0  # receding -> no in-path threat
    assert plan_motion_aware(closing) != plan_motion_aware(receding)


def test_sh4_static_only_collapses_where_dynamics_required():
    """SH4 leakage gate: on a frame where dynamics are definitionally required (a closing obstacle the
    ego must brake for), the no-dynamics control (static-only) gives the WRONG action relative to the
    motion-aware reference -- confirming the static-only surrogate truly cannot see motion (no
    privileged motion state is leaking through the static channel). At _D=8 m > _D_SAFE=5 m the
    static-only surrogate proceeds (0.0) while the closing case actually requires braking."""
    closing, _ = _witness_pair()
    assert plan_static_only(closing) == 0.0       # static-only: distance is 'safe', proceeds
    assert plan_motion_aware(closing) > 0.0        # motion-aware: closing -> must brake
    # the control collapses exactly where it must: static-only is blind to the required brake
    assert plan_static_only(closing) != plan_motion_aware(closing)
