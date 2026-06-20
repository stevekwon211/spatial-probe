# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Dataset adapters: load real datasets into the dataset-agnostic `probe.scene.Scene` type.

An adapter's only job is to produce the SAME Scene/Frame/OccupancyGrid/EgoPose/TrackedBox objects
the synthetic generator produces, so the predicates, retrieval engine, and metrics run unchanged on
real data. M2 = `occ3d` (Occ3D-nuScenes); see docs/m2-adapter-contract.md.
"""
