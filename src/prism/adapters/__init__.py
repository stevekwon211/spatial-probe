# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""S4 adapters -- the SceneReader / SceneWriter protocol + a registry of read/write paths.

PRISM already had three concrete data paths before S4:
  - the lossless Parquet round-trip (`prism.serialize.from_parquet` / `to_parquet`),
  - the AV2-Sensor ingest path (`prism.adapt.ingest` over an AV2 log dir),
  - the Occ3D-nuScenes ingest path (`prism.adapt.ingest` over an Occ3D root).
S4 does NOT rewrite any of them. It FORMALIZES the shape they already have into two
`typing.Protocol`s and registers the existing callables behind a small registry, so a new source
plugs in by satisfying the protocol -- the "swap source with no other change" contract of
`docs/m2-adapter-contract.md`, generalized one level up.

Two invariants this module enforces, by construction:

- **Invariant 1 (PRISM API never exposes external types).** A `SceneReader` reads external bytes
  INTO a `SceneIR`; a `SceneWriter` writes a `SceneIR` OUT to external bytes. The protocol surface
  speaks ONLY `SceneIR` + `pathlib.Path` -- never a `rerun` / `mcap` object. The Rerun and MCAP
  adapters live in their own modules and are NOT imported here.
- **Invariant 3 (core works without Rerun).** This module imports nothing optional. `import prism`
  pulls in `serialize` / `adapt` (pure numpy/scipy/pyarrow) only. Rerun/MCAP are reached lazily,
  inside their own adapter modules, and only when a caller asks for them.

The registry is intentionally tiny: a name -> reader and a name -> writer. It is the seam, not a
framework. Reuse is the point -- `parquet_reader` / `parquet_writer` are thin wrappers over
`serialize`, and the AV2 / Occ3D readers are thin wrappers over `adapt.ingest`.
"""
from __future__ import annotations

import pathlib
from typing import Callable, Protocol, runtime_checkable

from prism.ir import SceneIR

__all__ = [
    "SceneReader",
    "SceneWriter",
    "parquet_reader",
    "parquet_writer",
    "av2_reader",
    "occ3d_reader",
    "ingest_reader",
    "get_reader",
    "get_writer",
    "available_readers",
    "available_writers",
]


@runtime_checkable
class SceneReader(Protocol):
    """Read an external on-disk source INTO a `SceneIR`.

    The single method `read(path)` takes a filesystem path (a Parquet file, an AV2 log dir, an
    Occ3D root, an MCAP file) and returns a fully-formed `SceneIR`. It NEVER returns an external
    library's object -- the boundary is `SceneIR` in PRISM types only (invariant 1).
    """

    def read(self, path: str | pathlib.Path) -> SceneIR: ...


@runtime_checkable
class SceneWriter(Protocol):
    """Write a `SceneIR` OUT to an external on-disk form.

    `write(scene_ir, path)` serializes the IR to `path` and returns the written path. The input is
    a `SceneIR` (PRISM types only); whatever external format the writer targets stays inside the
    writer (invariant 1).
    """

    def write(self, scene_ir: SceneIR, path: str | pathlib.Path) -> pathlib.Path: ...


# --- concrete adapters: thin wrappers over the existing, tested paths (reuse, not rewrite) ---
class _ParquetReader:
    """The lossless Parquet reader, behind the protocol. Wraps `prism.serialize.from_parquet`."""

    name = "parquet"

    def read(self, path: str | pathlib.Path) -> SceneIR:
        from prism.serialize import from_parquet

        return from_parquet(path)


class _ParquetWriter:
    """The lossless Parquet writer, behind the protocol. Wraps `prism.serialize.to_parquet`."""

    name = "parquet"

    def write(self, scene_ir: SceneIR, path: str | pathlib.Path) -> pathlib.Path:
        from prism.serialize import to_parquet

        return to_parquet(scene_ir, path)


class _IngestReader:
    """The autodetecting ingest reader (AV2 log dir OR Occ3D root), behind the protocol.

    Wraps `prism.adapt.ingest`, which already detects the dataset layout. `scene` is forwarded for
    the Occ3D case (default = first scene). One reader covers both real-data sources because
    `ingest` is the one place that knows how to tell them apart.
    """

    name = "ingest"

    def __init__(self, scene: str | None = None) -> None:
        self._scene = scene

    def read(self, path: str | pathlib.Path) -> SceneIR:
        from prism.adapt import ingest

        return ingest(path, scene=self._scene)


# instances (the protocol is satisfied structurally; these are the registered handlers)
parquet_reader: SceneReader = _ParquetReader()
parquet_writer: SceneWriter = _ParquetWriter()
ingest_reader: SceneReader = _IngestReader()
# AV2 and Occ3D both route through ingest's autodetect; expose named aliases for clarity.
av2_reader: SceneReader = ingest_reader
occ3d_reader: SceneReader = ingest_reader


_READERS: dict[str, SceneReader] = {
    "parquet": parquet_reader,
    "ingest": ingest_reader,
    "av2": av2_reader,
    "occ3d": occ3d_reader,
}


def _make_rerun_writer() -> SceneWriter:
    """Lazily build the Rerun writer (imports `rerun` only when this is called -- invariant 3)."""
    from prism.adapters.rerun_adapter import RerunWriter

    return RerunWriter()


def _make_mcap_writer() -> SceneWriter:
    """Lazily build the MCAP writer (imports `mcap` only when this is called -- invariant 3)."""
    from prism.adapters.mcap_adapter import McapWriter

    return McapWriter()


def _make_mcap_reader() -> SceneReader:
    from prism.adapters.mcap_adapter import McapReader

    return McapReader()


# writers are produced lazily so the optional ones (rerun/mcap) never import at module load.
_WRITERS: dict[str, Callable[[], SceneWriter]] = {
    "parquet": lambda: parquet_writer,
    "rerun": _make_rerun_writer,
    "mcap": _make_mcap_writer,
}
_LAZY_READERS: dict[str, Callable[[], SceneReader]] = {
    "mcap": _make_mcap_reader,
}


def get_reader(name: str) -> SceneReader:
    """The registered `SceneReader` for `name` (e.g. 'parquet', 'ingest', 'av2', 'occ3d', 'mcap').

    Optional-source readers (mcap) are built lazily so importing this module never imports them.
    Raises KeyError with the known names if `name` is unregistered.
    """
    if name in _READERS:
        return _READERS[name]
    if name in _LAZY_READERS:
        return _LAZY_READERS[name]()
    known = sorted({*_READERS, *_LAZY_READERS})
    raise KeyError(f"unknown reader {name!r}; known readers: {known}")


def get_writer(name: str) -> SceneWriter:
    """The registered `SceneWriter` for `name` (e.g. 'parquet', 'rerun', 'mcap').

    The optional writers (rerun/mcap) are built lazily -- importing this module imports neither
    `rerun` nor `mcap`; only `get_writer('rerun')` reaches the optional dependency (invariant 3).
    Raises KeyError with the known names if `name` is unregistered.
    """
    if name not in _WRITERS:
        raise KeyError(f"unknown writer {name!r}; known writers: {sorted(_WRITERS)}")
    return _WRITERS[name]()


def available_readers() -> list[str]:
    return sorted({*_READERS, *_LAZY_READERS})


def available_writers() -> list[str]:
    return sorted(_WRITERS)
