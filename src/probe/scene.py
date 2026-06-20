# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Scene + Frame + TrackedBox: a time sequence of world state.

A Frame holds the occupancy field (what OccQuery reasons over), the ego pose, and the tracked
object boxes (`TrackedBox`) -- the observable a box-only query language gets. Keeping both the
dense occupancy and the object boxes on the frame is what lets the expressivity witness compare
the two backends fairly on the SAME scene.

A predicate runs on one frame; a query runs over a scene's frames. `t` is a frame index
(0-based), not seconds; `Frame.time` carries the wall-clock stamp. The dataset adapter (M2)
builds a Scene from Occ3D-nuScenes; the synthetic generator builds one by hand. Same type either
way, so the retrieval loop is dataset-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

from probe.grid import EgoPose, OccupancyGrid

__all__ = ["Frame", "Scene", "TrackedBox"]


@dataclass(frozen=True, eq=False)
class TrackedBox:
    """A tracked object box -- the observable a box-only (RefAV-style) query language sees.

    center: world (x, y, z) in meters; size: (length, width, height); yaw: radians; label: class
    string; velocity: (vx, vy) m/s. A box-only language can only ever answer questions about
    these fields -- never about unboxed occupancy (walls, debris, free space).
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float
    label: str
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, eq=False)
class Frame:
    grid: OccupancyGrid
    ego: EgoPose
    time: float = 0.0
    objects: tuple[TrackedBox, ...] = ()


@dataclass(frozen=True, eq=False)
class Scene:
    frames: tuple[Frame, ...]
    name: str = ""

    def __len__(self) -> int:
        return len(self.frames)

    def times(self) -> tuple[int, ...]:
        """Frame indices, the `t` values a query iterates over."""
        return tuple(range(len(self.frames)))

    def grid_at(self, t: int) -> OccupancyGrid:
        return self.frames[t].grid

    def ego_at(self, t: int) -> EgoPose:
        return self.frames[t].ego

    def ego_speed(self, t: int) -> float:
        return self.frames[t].ego.speed

    def ego_width(self) -> float:
        return self.frames[0].ego.width

    def objects_at(self, t: int) -> tuple[TrackedBox, ...]:
        return self.frames[t].objects
