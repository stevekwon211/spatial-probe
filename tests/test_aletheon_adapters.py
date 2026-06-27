# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S4 adapters -- the SceneReader/SceneWriter protocol + the Rerun (and MCAP) adapters.

The load-bearing claims, in order of integrity weight:

- **Invariant 1 / 3 (critical):** `import aletheon` must NOT import `rerun`. The core
  (`aletheon query` / `aletheon export`) runs on pure numpy/scipy/pyarrow with rerun-sdk absent or
  unused. Asserted in a SUBPROCESS with a scrubbed environment so a leaked top-level `import rerun`
  anywhere in the import graph would show up as `'rerun' in sys.modules` and fail.
- **Invariant 2:** a query's answer comes from the IR (parquet), never from a `.rrd`. Export ->
  query -> delete every `.rrd` -> re-query yields a byte-identical result; the render is not the
  source of truth.
- **Protocol round-trip:** a writer -> reader cycle preserves the IR at the content-hash level
  (reuses the serialize round-trip the parquet adapter wraps).
- **Rerun adapter:** skip if rerun-sdk is unavailable; else assert a non-empty `.rrd` is produced
  from REAL on-disk AV2 data.
- **MCAP adapter:** skip if mcap is unavailable; else a lossless IR <-> MCAP round-trip.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from probe.grid import FREE, OCCUPIED, UNKNOWN, EgoPose, OccupancyGrid
from probe.scene import Frame, Scene, TrackedBox
from aletheon.adapters import (
    SceneReader,
    SceneWriter,
    available_readers,
    available_writers,
    get_reader,
    get_writer,
    parquet_reader,
    parquet_writer,
)
from aletheon.ir import (
    CoordinateFrame,
    Entity,
    GroundTruth,
    Observation,
    Pose,
    Provenance,
    SceneIR,
    Track,
)
from aletheon.serialize import content_hash

_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")
_HAS_RERUN = importlib.util.find_spec("rerun") is not None
_HAS_MCAP = importlib.util.find_spec("mcap") is not None


def _rich_ir() -> SceneIR:
    """A small IR exercising occupancy, boxes, coordinate frames, tracks, GT, NaN velocity."""
    rng = np.random.default_rng(0)
    occ0 = rng.integers(UNKNOWN, OCCUPIED + 1, size=(4, 4, 4)).astype(int)
    occ1 = np.full((4, 4, 4), FREE, dtype=int)
    occ1[0, 0, 0] = OCCUPIED
    box = TrackedBox(center=(3.0, -1.0, 0.5), size=(4.5, 2.0, 1.8), yaw=0.4, label="vehicle",
                     velocity=(float("nan"), float("nan")))
    scene = Scene(
        (
            Frame(OccupancyGrid(occ0, 0.4, (-40.0, -40.0, -1.0), -1.0), EgoPose((0, 0, 0), 0.1, speed=5.0), time=0.0, objects=(box,)),
            Frame(OccupancyGrid(occ1, 0.4, (-40.0, -40.0, -1.0), -1.0), EgoPose((1, 0, 0), 0.1, speed=6.0), time=0.5),
        ),
        name="rich",
    )
    ent = Entity.from_tracked_box(box, "vehicle#0", "ego")
    return SceneIR(
        scene=scene,
        coordinate_frames=(
            CoordinateFrame("world", None, Pose((0.0, 0.0, 0.0))),
            CoordinateFrame("ego", "world", Pose.from_yaw((5568.3, 2152.2, 74.4), 1.2)),
            CoordinateFrame("lidar", "ego", Pose((0.1, 0.0, 1.5))),
        ),
        tracks=(Track("vehicle#0", "vehicle", (0.0,), (ent,)),),
        observations=(Observation("lidar", 0.0, 0), Observation("lidar", 0.5, 1)),
        ground_truth=(GroundTruth(0, (ent,)), GroundTruth(1, ())),
        provenance=Provenance(dataset="synthetic", log_id="rich", adapter="test"),
    )


def _av2_log() -> pathlib.Path | None:
    if not _AV2_ROOT.is_dir():
        return None
    logs = sorted(p for p in _AV2_ROOT.iterdir() if p.is_dir())
    return logs[0] if logs else None


# --------------------------------------------------------------------------------------------------
# Invariant 1 / 3 (CRITICAL): import aletheon must not import rerun; core runs rerun-free.
# --------------------------------------------------------------------------------------------------
def _run_scrubbed(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with src/ on the path and NO inherited PYTHONPATH/state.

    A scrubbed env means a top-level `import rerun` anywhere in the aletheon import graph (a regression
    of invariant 3) would put 'rerun' in sys.modules and the assertion inside `code` would fail.
    """
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(pathlib.Path("src").resolve()),
        "HOME": "/tmp",
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(pathlib.Path.cwd()),
    )


def test_import_aletheon_does_not_import_rerun():
    """import aletheon (and aletheon.query/serialize/adapters) leaves 'rerun' out of sys.modules."""
    code = """
        import sys
        import aletheon
        from aletheon import query, to_parquet, from_parquet
        import aletheon.adapters
        assert "rerun" not in sys.modules, sorted(m for m in sys.modules if "rerun" in m)
        assert "mcap" not in sys.modules, sorted(m for m in sys.modules if "mcap" in m)
        print("OK no-rerun-import")
    """
    r = _run_scrubbed(code)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK no-rerun-import" in r.stdout


def test_core_query_runs_without_rerun(tmp_path):
    """A full export -> query cycle runs in a scrubbed interpreter with rerun never imported."""
    parquet = tmp_path / "ir.parquet"
    parquet_writer.write(_rich_ir(), parquet)
    code = f"""
        import sys
        import aletheon
        from aletheon import from_parquet, query
        ir = from_parquet({str(parquet)!r})
        res = query(ir, "ego_speed(scene, t) > 0.0", scope="any")
        assert res.matched is True, res
        assert "rerun" not in sys.modules, "rerun leaked into a pure query path"
        print("OK query-rerun-free", res.n_frames)
    """
    r = _run_scrubbed(code)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK query-rerun-free" in r.stdout


# --------------------------------------------------------------------------------------------------
# Invariant 2: the .rrd is never the source of truth -- a query is identical with no .rrd present.
# --------------------------------------------------------------------------------------------------
def test_query_identical_after_deleting_rrd(tmp_path):
    """Export -> query, then delete any .rrd -> re-query gives a byte-identical (JSON) result."""
    from aletheon.query import query

    parquet = tmp_path / "ir.parquet"
    parquet_writer.write(_rich_ir(), parquet)

    def run_query() -> str:
        ir = parquet_reader.read(parquet)
        res = query(ir, "ego_speed(scene, t) > 0.0", scope="any")
        return json.dumps(
            {"matched": res.matched, "matched_frames": res.matched_frames, "n_frames": res.n_frames, "scope": res.scope},
            sort_keys=True,
        )

    before = run_query()

    # write a .rrd alongside (if rerun present) then delete every .rrd; the query must not care.
    if _HAS_RERUN:
        from aletheon.adapters.rerun_adapter import to_rerun

        to_rerun(_rich_ir(), tmp_path / "scene.rrd")
    for rrd in tmp_path.glob("*.rrd"):
        rrd.unlink()
    assert not list(tmp_path.glob("*.rrd"))

    after = run_query()
    assert before == after, "query result changed when the .rrd was removed -- render leaked into truth"


# --------------------------------------------------------------------------------------------------
# Protocol: the registry + the parquet writer->reader cycle preserve the IR (content hash).
# --------------------------------------------------------------------------------------------------
def test_parquet_protocol_round_trip(tmp_path):
    s = _rich_ir()
    out = parquet_writer.write(s, tmp_path / "ir.parquet")
    back = parquet_reader.read(out)
    assert content_hash(back) == content_hash(s)


def test_registry_exposes_protocol_instances():
    assert isinstance(parquet_reader, SceneReader)
    assert isinstance(parquet_writer, SceneWriter)
    assert isinstance(get_reader("parquet"), SceneReader)
    assert isinstance(get_writer("parquet"), SceneWriter)
    assert "parquet" in available_readers()
    assert {"parquet", "rerun", "mcap"} <= set(available_writers())


def test_get_writer_does_not_import_optional_at_module_load():
    """available_writers() lists rerun/mcap but listing them must not import them."""
    code = """
        import sys
        import aletheon.adapters as a
        ws = a.available_writers()
        assert "rerun" in ws and "mcap" in ws, ws
        assert "rerun" not in sys.modules, "listing writers imported rerun"
        assert "mcap" not in sys.modules, "listing writers imported mcap"
        print("OK lazy-registry")
    """
    r = _run_scrubbed(code)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK lazy-registry" in r.stdout


def test_unknown_reader_writer_raise():
    with pytest.raises(KeyError):
        get_reader("nope")
    with pytest.raises(KeyError):
        get_writer("nope")


# --------------------------------------------------------------------------------------------------
# Rerun adapter: skip if unavailable; else a non-empty .rrd from real AV2 data + a synthetic .rrd.
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_RERUN, reason="rerun-sdk not installed (optional extra)")
def test_to_rerun_writes_non_empty_rrd_synthetic(tmp_path):
    from aletheon.adapters.rerun_adapter import to_rerun

    out = to_rerun(_rich_ir(), tmp_path / "synthetic.rrd")
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not _HAS_RERUN, reason="rerun-sdk not installed (optional extra)")
def test_to_rerun_returns_path_not_rerun_object(tmp_path):
    """Invariant 1: the adapter returns a pathlib.Path, never a rerun stream/object."""
    from aletheon.adapters.rerun_adapter import to_rerun

    out = to_rerun(_rich_ir(), tmp_path / "s.rrd")
    assert isinstance(out, pathlib.Path)


@pytest.mark.skipif(not _HAS_RERUN, reason="rerun-sdk not installed (optional extra)")
@pytest.mark.skipif(_av2_log() is None, reason="no AV2-Sensor logs on disk")
def test_to_rerun_from_real_av2_data(tmp_path):
    """The headline: a real .rrd from a real on-disk AV2 log, non-empty."""
    from aletheon.adapt import ingest
    from aletheon.adapters.rerun_adapter import to_rerun

    log = _av2_log()
    ir = ingest(log)
    out = to_rerun(ir, tmp_path / "av2.rrd", max_points_per_frame=2000)
    assert out.exists()
    assert out.stat().st_size > 1000, f".rrd suspiciously small: {out.stat().st_size} bytes"


@pytest.mark.skipif(_HAS_RERUN, reason="rerun IS installed; the not-installed path is unreachable here")
def test_to_rerun_raises_clearly_when_rerun_absent(tmp_path):
    """When rerun-sdk is absent, to_rerun raises RerunNotInstalled with the install hint."""
    from aletheon.adapters.rerun_adapter import RerunNotInstalled, to_rerun

    with pytest.raises(RerunNotInstalled) as ei:
        to_rerun(_rich_ir(), tmp_path / "x.rrd")
    assert "pip install" in str(ei.value)


# --------------------------------------------------------------------------------------------------
# MCAP adapter (optional, lower priority): lossless IR <-> MCAP round-trip when mcap is present.
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_MCAP, reason="mcap not installed (optional extra)")
def test_mcap_round_trip_content_hash(tmp_path):
    from aletheon.adapters.mcap_adapter import from_mcap, to_mcap

    s = _rich_ir()
    out = to_mcap(s, tmp_path / "ir.mcap")
    assert out.exists() and out.stat().st_size > 0
    back = from_mcap(out)
    assert content_hash(back) == content_hash(s)


@pytest.mark.skipif(not _HAS_MCAP, reason="mcap not installed (optional extra)")
def test_mcap_round_trip_preserves_nan_velocity(tmp_path):
    from aletheon.adapters.mcap_adapter import from_mcap, to_mcap

    back = from_mcap(to_mcap(_rich_ir(), tmp_path / "ir.mcap"))
    vx, vy = back.scene.frames[0].objects[0].velocity
    assert math.isnan(vx) and math.isnan(vy)


@pytest.mark.skipif(not _HAS_MCAP, reason="mcap not installed (optional extra)")
def test_mcap_via_registry(tmp_path):
    s = _rich_ir()
    out = get_writer("mcap").write(s, tmp_path / "ir.mcap")
    back = get_reader("mcap").read(out)
    assert content_hash(back) == content_hash(s)
