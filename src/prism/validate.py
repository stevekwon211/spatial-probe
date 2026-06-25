# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""The 'honest instrument' validator -- reject silently-corrupt IR before any query runs on it.

A spatial query is only as trustworthy as its calibration and timing. This module makes a class of
silent-wrong-answer bugs IMPOSSIBLE to carry forward: it refuses an IR whose timestamps are
inconsistent, whose poses are NaN/inf, whose coordinate frames are missing or form a broken
parent chain, or whose tracks are internally inconsistent. Every failure is an explicit raise with
a precise message -- never a silent fallback (repo rule: explicit failure over silent fallback).

`validate_scene(scene_ir)` raises `ValidationError` on the first problem (or all of them via
`collect=True`). It is the gate the CLI ingest path and any experiment runner should call before
trusting an IR.
"""
from __future__ import annotations

import math

from prism.ir import SceneIR

__all__ = ["validate_scene", "ValidationError"]


class ValidationError(ValueError):
    """Raised when an IR fails a calibration / timestamp / integrity check."""


def _finite(values) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _check_frames(s: SceneIR, errors: list[str]) -> None:
    frames = s.scene.frames
    if not frames:
        errors.append("scene has no frames")
        return
    # poses must be finite (NaN/inf ego position or heading = a broken calibration, not a value)
    for i, fr in enumerate(frames):
        if not _finite(fr.ego.position):
            errors.append(f"frame {i}: ego.position is non-finite {fr.ego.position}")
        if not _finite([fr.ego.heading]):
            errors.append(f"frame {i}: ego.heading is non-finite ({fr.ego.heading})")
        if not _finite([fr.time]):
            errors.append(f"frame {i}: timestamp is non-finite ({fr.time})")
        for j, box in enumerate(fr.objects):
            if not _finite(box.center):
                errors.append(f"frame {i} object {j}: box.center is non-finite {box.center}")


def _check_timestamps(s: SceneIR, errors: list[str]) -> None:
    """Frame timestamps must be strictly increasing -- a non-monotonic or duplicate stamp means the
    sweeps are misordered or two sensors disagree on time, exactly the mismatch that makes a
    multi-sensor query wrong."""
    times = [fr.time for fr in s.scene.frames]
    for i in range(1, len(times)):
        if not (times[i] > times[i - 1]):
            errors.append(
                f"timestamps not strictly increasing: frame {i - 1} t={times[i - 1]} >= frame {i} t={times[i]}"
            )
    # observations (if any) must reference an existing frame and agree with that frame's timestamp
    for o in s.observations:
        if not (0 <= o.frame_index < len(times)):
            errors.append(f"observation on sensor {o.sensor!r} references frame {o.frame_index} out of range")
            continue
        ft = times[o.frame_index]
        if math.isfinite(o.timestamp) and math.isfinite(ft) and o.timestamp != ft:
            errors.append(
                f"sensor {o.sensor!r} timestamp {o.timestamp} mismatches frame {o.frame_index} timestamp {ft}"
            )


def _check_coordinate_frames(s: SceneIR, errors: list[str]) -> None:
    """If coordinate frames are declared they must be well-formed: unique names, every non-root
    parent must resolve, no cycles, and any pose/intrinsics must be finite. An IR that declares an
    object 'in frame X' but has no frame X is rejected (a missing frame = an unprojectable point)."""
    cfs = s.coordinate_frames
    names = [cf.name for cf in cfs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        errors.append(f"duplicate coordinate frame names: {dupes}")
    nameset = set(names)
    for cf in cfs:
        if cf.parent is not None and cf.parent not in nameset:
            errors.append(f"coordinate frame {cf.name!r} has unknown parent {cf.parent!r}")
        if not _finite(cf.pose.translation) or not _finite(cf.pose.quaternion):
            errors.append(f"coordinate frame {cf.name!r} has a non-finite pose")
        if cf.intrinsics is not None and not _finite(cf.intrinsics.ravel()):
            errors.append(f"coordinate frame {cf.name!r} has non-finite intrinsics")
    # parent-chain cycle detection
    parent_of = {cf.name: cf.parent for cf in cfs}
    for start in names:
        seen = set()
        cur = start
        while cur is not None:
            if cur in seen:
                errors.append(f"coordinate frame parent chain has a cycle at {cur!r}")
                break
            seen.add(cur)
            cur = parent_of.get(cur)
    # every entity/track/box frame referenced must exist (only when frames are declared at all)
    if nameset:
        referenced: set[str] = set()
        for tk in s.tracks:
            referenced.update(e.frame for e in tk.states)
        for gt in s.ground_truth:
            referenced.update(e.frame for e in gt.entities)
        for p in s.predictions:
            referenced.update(e.frame for e in p.entities)
        for fr in referenced - nameset:
            errors.append(f"an entity references coordinate frame {fr!r} which is not declared")


def _check_tracks(s: SceneIR, errors: list[str]) -> None:
    for tk in s.tracks:
        if len(tk.timestamps) != len(tk.states):
            errors.append(f"track {tk.entity_id!r}: {len(tk.timestamps)} timestamps != {len(tk.states)} states")
        for e in tk.states:
            if not _finite(e.pose.translation):
                errors.append(f"track {tk.entity_id!r}: a state has a non-finite position")


def validate_scene(scene_ir: SceneIR, *, collect: bool = False) -> SceneIR:
    """Validate an IR. Raises `ValidationError` on the first failure; with `collect=True`, gathers
    every failure into one error. Returns the IR unchanged on success (so it composes in a pipeline).

    Checks: non-empty; finite ego poses, headings, timestamps, and box centers; strictly increasing
    frame timestamps; sensor-vs-frame timestamp agreement; well-formed coordinate frames (unique,
    resolvable parents, no cycles, finite, every referenced frame declared); consistent tracks.
    """
    errors: list[str] = []
    _check_frames(scene_ir, errors)
    _check_timestamps(scene_ir, errors)
    _check_coordinate_frames(scene_ir, errors)
    _check_tracks(scene_ir, errors)
    if errors:
        if collect:
            raise ValidationError("; ".join(errors))
        raise ValidationError(errors[0])
    return scene_ir
