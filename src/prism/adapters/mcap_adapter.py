# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""MCAP adapter -- best-effort, lossless IR <-> MCAP round-trip. Optional extra, lazy import.

MCAP (https://mcap.dev) is the robotics log container (the format ros2 bags and many AV stacks
write). This adapter lets a PRISM `SceneIR` be written to / read from an `.mcap` file. Like the
Rerun adapter, `mcap` is imported LAZILY (inside the functions) so `import prism` works in a venv
WITHOUT `mcap` installed (invariant 3); and the surface speaks only `SceneIR` + path (invariant 1).

Design: rather than invent a new schema, this REUSES the serialize layer's canonical form. The IR
is split exactly as Parquet does it -- the relational layers as `serialize.scene_ir_to_dict` (one
JSON metadata record) and each frame's dense occupancy grid as raw bytes (one MCAP message per
frame, on the `occupancy_grid` channel, log_time = the frame's timestamp). `from_mcap` reverses it
and rebuilds the SceneIR via `serialize.scene_ir_from_dict`. The round-trip is therefore lossless
by the SAME mechanism Parquet uses (`content_hash` equal in, equal out), with no second encoder to
drift.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from prism.ir import SceneIR

__all__ = ["to_mcap", "from_mcap", "McapWriter", "McapReader", "McapNotInstalled"]

_INSTALL_HINT = "mcap not installed: pip install 'spatial-probe[mcap]' (or: pip install mcap)"
_SCHEMA_NAME = "prism_ir/grid_bytes"
_GRID_TOPIC = "occupancy_grid"
_META_TOPIC = "scene_ir_meta"


class McapNotInstalled(RuntimeError):
    """Raised when the MCAP adapter is invoked but `mcap` is not importable (with install hint)."""


def _import_mcap():
    """Import `mcap` lazily, translating ImportError into McapNotInstalled. Only place mcap loads."""
    try:
        from mcap.reader import make_reader  # noqa: PLC0415 (lazy by design -- invariant 3)
        from mcap.writer import Writer  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - exercised only in an mcap-free venv
        raise McapNotInstalled(_INSTALL_HINT) from e
    return Writer, make_reader


def to_mcap(scene_ir: SceneIR, out_path: str | pathlib.Path) -> pathlib.Path:
    """Write `scene_ir` to an `.mcap` file losslessly. Returns the path (PRISM/pathlib, invariant 1).

    One metadata message carries the canonical IR dict (`serialize.scene_ir_to_dict`); one message
    per frame carries that frame's raw occupancy bytes, log_time = the frame timestamp (ns). Raises
    `McapNotInstalled` if `mcap` is absent.
    """
    from prism.serialize import _grid_bytes, scene_ir_to_dict  # reuse the exact serialize encoding

    Writer, _ = _import_mcap()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    d = scene_ir_to_dict(scene_ir)
    with out.open("wb") as f:
        w = Writer(f)
        w.start(profile="prism", library="prism-mcap")
        schema_id = w.register_schema(name=_SCHEMA_NAME, encoding="json", data=b"{}")
        meta_chan = w.register_channel(topic=_META_TOPIC, message_encoding="json", schema_id=schema_id)
        grid_chan = w.register_channel(topic=_GRID_TOPIC, message_encoding="raw", schema_id=schema_id)
        w.add_message(
            channel_id=meta_chan,
            log_time=0,
            publish_time=0,
            data=json.dumps(d, sort_keys=True).encode("utf-8"),
        )
        for i, fr in enumerate(scene_ir.scene.frames):
            log_time = int(round(fr.time * 1e9))
            w.add_message(
                channel_id=grid_chan,
                log_time=max(log_time, 0),
                publish_time=max(log_time, 0),
                data=_grid_bytes(fr.grid),
                sequence=i,
            )
        w.finish()
    return out


def from_mcap(path: str | pathlib.Path) -> SceneIR:
    """Read back a SceneIR written by `to_mcap`, reconstructing grids exactly (lossless round-trip).

    Reverses `to_mcap`: the metadata message rebuilds the canonical dict, the per-frame grid
    messages rebuild each dense occupancy array (ordered by `sequence`), and
    `serialize.scene_ir_from_dict` stitches them back into a SceneIR.
    """
    from prism.serialize import _grid_from_bytes, scene_ir_from_dict

    _, make_reader = _import_mcap()
    p = pathlib.Path(path)
    meta: dict | None = None
    grid_msgs: dict[int, bytes] = {}
    with p.open("rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic == _META_TOPIC:
                meta = json.loads(message.data.decode("utf-8"))
            elif channel.topic == _GRID_TOPIC:
                grid_msgs[message.sequence] = message.data
    if meta is None:
        raise ValueError(f"{p}: not a prism IR mcap (no {_META_TOPIC!r} message)")
    grids = []
    for i, fr in enumerate(meta["frames"]):
        gm = fr["grid"]
        grids.append(_grid_from_bytes(grid_msgs[i], gm["shape"], gm["dtype"]))
    return scene_ir_from_dict(meta, grids)


class McapWriter:
    """`SceneWriter` adapter: `write(scene_ir, path)` -> an `.mcap`. Lazy-imports mcap (invariant 3)."""

    name = "mcap"

    def write(self, scene_ir: SceneIR, path: str | pathlib.Path) -> pathlib.Path:
        return to_mcap(scene_ir, path)


class McapReader:
    """`SceneReader` adapter: `read(path)` -> a `SceneIR` from an `.mcap`. Lazy-imports mcap."""

    name = "mcap"

    def read(self, path: str | pathlib.Path) -> SceneIR:
        return from_mcap(path)
