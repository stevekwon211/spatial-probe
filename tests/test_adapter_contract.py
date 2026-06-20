# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""The M2 adapter contract is present and honest: the interface exists, and it fails loudly (not
silently) until the gated dataset is wired in."""
import pathlib

import pytest

from probe.adapters import occ3d


def test_adapter_exposes_load_scene():
    assert hasattr(occ3d, "load_scene")


def test_adapter_raises_until_m2():
    with pytest.raises(NotImplementedError):
        occ3d.load_scene("scene-token", pathlib.Path("/nonexistent"))
