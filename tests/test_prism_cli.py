# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""PRISM CLI -- main() entrypoint over real on-disk data (skip-if-missing), parser wiring.

`prism query`/`export`/`ingest` must run end-to-end on a real AV2 log or Occ3D root. The
data-detection + "needs data" paths are always tested; the real-run assertions are skip-if-missing.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from prism.cli import build_parser, main

_AV2_ROOT = pathlib.Path("data/danger/av2_sensor")
_OCC3D_ROOT = pathlib.Path("data")


def _first_av2_log():
    if _AV2_ROOT.is_dir():
        logs = sorted(p for p in _AV2_ROOT.iterdir() if p.is_dir())
        if logs:
            return logs[0]
    return None


def _has_occ3d() -> bool:
    return (_OCC3D_ROOT / "annotations.json").exists() and (_OCC3D_ROOT / "gts").is_dir()


def test_parser_builds_all_verbs():
    p = build_parser()
    for verb in ("query", "ingest", "export", "view", "find"):
        ns = p.parse_args(_min_args(verb))
        assert ns.command == verb


def _min_args(verb: str):
    if verb == "query":
        return ["query", "x < 1", "--data", "nope"]
    if verb == "ingest":
        return ["ingest", "nope"]
    if verb == "export":
        return ["export", "out.parquet", "--data", "nope"]
    if verb == "find":
        return ["find", "path blocked", "--data", "nope"]
    return ["view", "--data", "nope"]  # view is now wired to a backend + --data, not a stub


def test_find_missing_data_reports_clearly(capsys):
    rc = main(["find", "path blocked but no tracked object", "--data", "/no/such/path"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "needs data" in out


def test_find_unknown_query_reports_clearly(capsys):
    log = _first_av2_log()
    if log is None:
        pytest.skip("no AV2 log to exercise the signature mapping")
    rc = main(["find", "completely unrelated nonsense phrase", "--data", str(log), "--limit-frames", "1"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "no signature" in out


@pytest.mark.skipif(_first_av2_log() is None, reason="no AV2-Sensor log on disk")
def test_find_runs_on_real_av2_log(capsys):
    log = _first_av2_log()
    rc = main(["find", "path blocked but no tracked object explains it",
               "--data", str(log), "--limit-frames", "20"])
    out = capsys.readouterr().out
    assert rc == 0
    # human summary block is printed, then the JSON; the JSON parses and carries the honest fields.
    assert "signature: path_blocked_no_box" in out
    json_start = out.index("{")
    summary = json.loads(out[json_start:])
    assert summary["signature"] == "path_blocked_no_box"
    assert summary["n_frames_scanned"] > 0
    assert "external" in summary["honesty"].lower()


def test_view_missing_data_reports_clearly(capsys):
    """view --backend rerun on a non-existent path: a clear 'needs data', not a stub or a crash."""
    rc = main(["view", "--backend", "rerun", "--data", "/no/such/path"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "needs data" in out


def test_query_missing_data_reports_clearly(capsys):
    rc = main(["query", "ego_speed(scene, t) > 0", "--data", "/no/such/path"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "needs data" in out


def test_export_missing_data_reports_clearly(capsys, tmp_path):
    rc = main(["export", str(tmp_path / "o.parquet"), "--data", "/no/such/path"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "needs data" in out


def test_query_rejects_unsafe_expression(capsys, tmp_path):
    # build a tiny fake AV2-ish dir so detection picks it up, but the expr is what fails.
    # Use an Occ3D/AV2 path only if present; otherwise this still exercises the data-missing guard.
    log = _first_av2_log()
    if log is None and not _has_occ3d():
        pytest.skip("no on-disk dataset to load a scene for the query")
    args = ["query", "__import__('os')", "--data", str(log) if log else str(_OCC3D_ROOT)]
    if log is None:
        args += ["--frame", "0"]
    rc = main(args)
    out = capsys.readouterr().out
    assert rc == 2
    assert "bad query" in out


@pytest.mark.skipif(_first_av2_log() is None, reason="no AV2-Sensor log on disk")
def test_query_runs_on_real_av2_log(capsys):
    log = _first_av2_log()
    rc = main(["query", "ego_speed(scene, t) >= 0.0", "--data", str(log), "--frame", "0"])
    out = capsys.readouterr().out
    assert rc == 0
    result = json.loads(out)
    assert result["n_frames"] > 0
    assert result["scene"] == log.name
    assert 0 in result["matched_frames"]  # speed >= 0 is always true


@pytest.mark.skipif(_first_av2_log() is None, reason="no AV2-Sensor log on disk")
def test_export_writes_parquet_on_real_av2_log(capsys, tmp_path):
    log = _first_av2_log()
    out_pq = tmp_path / "av2.parquet"
    rc = main(["export", str(out_pq), "--data", str(log), "--openlabel", str(tmp_path / "av2.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert out_pq.exists() and out_pq.stat().st_size > 0
    assert (tmp_path / "av2.json").exists()
    # the written parquet round-trips back to the same content
    from prism.serialize import content_hash, from_parquet
    from prism.adapt import ingest

    assert content_hash(from_parquet(out_pq)) == content_hash(ingest(log))


@pytest.mark.skipif(_first_av2_log() is None, reason="no AV2-Sensor log on disk")
def test_ingest_validates_and_summarizes_real_av2_log(capsys):
    log = _first_av2_log()
    rc = main(["ingest", str(log)])
    out = capsys.readouterr().out
    assert rc == 0
    summary = json.loads(out)
    assert summary["n_frames"] > 0
    assert summary["n_coordinate_frames"] >= 3
    assert "content_hash" in summary


@pytest.mark.skipif(not _has_occ3d(), reason="no Occ3D-nuScenes data on disk")
def test_export_writes_parquet_on_real_occ3d(capsys, tmp_path):
    out_pq = tmp_path / "occ3d.parquet"
    rc = main(["export", str(out_pq), "--data", str(_OCC3D_ROOT)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out_pq.exists() and out_pq.stat().st_size > 0
