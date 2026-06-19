# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Scene: a time sequence of (OccupancyGrid, EgoPose) frames.

A predicate runs on one frame; a *query* runs over a scene's frames and the scene is
retrieved if any frame satisfies it. The dataset adapter (M2) builds a Scene from
Occ3D-nuScenes; the synthetic generator builds one by hand. Same type either way, so the
retrieval loop (`experiments/occquery_v0/run.py`) is dataset-agnostic.

`t` is a frame index (0-based), not seconds; `Frame.time` carries the wall-clock stamp.
"""
from __future__ import annotations

from dataclasses import dataclass

from probe.grid import EgoPose, OccupancyGrid

__all__ = ["Frame", "Scene"]


@dataclass(frozen=True, eq=False)
class Frame:
    grid: OccupancyGrid
    ego: EgoPose
    time: float = 0.0  # seconds


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
