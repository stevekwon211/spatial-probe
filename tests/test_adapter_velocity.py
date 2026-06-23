# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""dynfield step 1: per-object velocity is populated in the Occ3D adapter (box_velocity differencing).

Guards the first hard dependency for dynfield -- TrackedBox.velocity must be a real finite-difference
of box translations across prev/next keyframe annotations, transformed into the ego frame, NOT silently
zeroed. Skips if the nuScenes box metadata is not on disk (CI without data)."""
import math
import pathlib

import pytest

from probe.adapters.occ3d import _global_velocity

_NUSC = pathlib.Path(__file__).resolve().parents[1] / "data" / "nuscenes" / "v1.0-trainval"


def test_global_velocity_central_difference():
    # central difference of two known translations over a known dt
    cur = {"prev": "p", "next": "n", "sample_token": "s1"}
    by_token = {
        "p": {"translation": [10.0, 0.0, 0.0], "sample_token": "s0"},
        "n": {"translation": [14.0, 2.0, 0.0], "sample_token": "s2"},
    }
    ts = {"s0": 0, "s2": 2_000_000}  # microseconds -> dt = 2.0 s
    vx, vy = _global_velocity(cur, by_token, ts)
    assert math.isclose(vx, 2.0) and math.isclose(vy, 1.0)  # (14-10)/2, (2-0)/2


def test_isolated_annotation_is_nan_not_zero():
    # no prev and no next -> NaN, never a silent 0 (a missing motion field must be visible)
    cur = {"prev": "", "next": "", "sample_token": "s1"}
    vx, vy = _global_velocity(cur, {}, {})
    assert math.isnan(vx) and math.isnan(vy)


@pytest.mark.skipif(not _NUSC.exists(), reason="nuScenes box metadata not on disk")
def test_real_scene_velocity_populated_and_sane():
    import numpy as np

    from probe.adapters.occ3d import load_scene

    data = _NUSC.parents[1]
    sc = load_scene("scene-0061", data, mask="none", with_boxes=True)
    objs = sc.objects_at(0)
    speeds = [math.hypot(*o.velocity) for o in objs if not math.isnan(o.velocity[0])]
    assert objs, "no boxes loaded"
    assert speeds, "no finite velocities populated -- velocity still (0,0)?"
    # plausible urban speeds: most finite-velocity objects under ~30 m/s, not all identically zero
    assert max(speeds) < 30.0
    assert any(s > 0.1 for s in speeds), "all velocities ~0 -- differencing likely not wired"
