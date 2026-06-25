# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""`prism` CLI -- run the spatial-probe instrument from the shell.

Each verb runs end-to-end on real on-disk data (or prints a clear "needs data" message; never a
stub that pretends). The query verb reuses `probe.query_dsl.safe_eval` + the predicate namespace
(the SAME AST-whitelist the mcp_server and retrieval use -- a query expression is untrusted input).

Verbs:
  prism query "<expr>" --data <root> [--scene <name>] [--policy free|occupied|ignored] [--frame N]
      Evaluate a predicate DSL expression over a scene's frames; print per-frame truth + the
      matched frame indices. Reuses the occupancy/box predicates verbatim.
  prism ingest <path> [--scene <name>] [--out <ir.parquet>]
      Adapter -> Scene IR; validate it; print a summary (or write the IR parquet with --out).
  prism export <out.parquet> --data <path> [--scene <name>] [--openlabel <out.json>] [--jsonl <out.jsonl>]
      Ingest -> IR -> write a lossless parquet (and optional OpenLABEL / JSONL views).
  prism view
      Stub: prints that a viewer backend (rerun) is optional and not wired here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from probe.grid import UnknownPolicy
from probe.query_dsl import UnsafeExpression, safe_eval
from probe.retrieval import namespace

_POLICY = {"free": UnknownPolicy.FREE, "occupied": UnknownPolicy.OCCUPIED, "ignored": UnknownPolicy.IGNORED}


def _load_scene_for_query(data: pathlib.Path, scene: str | None, with_boxes: bool):
    """Load a probe.Scene from either dataset layout (mirrors prism.adapt.ingest detection but
    returns the raw probe.Scene the predicates run on)."""
    from prism.adapt import _is_av2_log

    if _is_av2_log(data):
        from probe.adapters.av2_sensor import load_scene

        return load_scene(data.name, data.parent, with_boxes=with_boxes)
    if (data / "annotations.json").exists() and (data / "gts").is_dir():
        from probe.adapters.occ3d import load_scene

        if scene is None:
            scene = sorted(json.loads((data / "annotations.json").read_text())["scene_infos"].keys())[0]
        return load_scene(scene, data, mask="none", with_boxes=with_boxes)
    return None


def _cmd_query(args: argparse.Namespace) -> int:
    data = pathlib.Path(args.data)
    if not data.exists():
        print(f"needs data: {data} does not exist (point --data at an AV2 log dir or an Occ3D root)")
        return 2
    needs_boxes = "distance_to_nearest_object" in args.expr
    sc = _load_scene_for_query(data, args.scene, with_boxes=needs_boxes)
    if sc is None:
        print(f"needs data: {data} is not a recognized dataset (AV2 log dir or Occ3D root)")
        return 2
    policy = _POLICY[args.policy]
    frames = range(len(sc.frames)) if args.frame is None else [args.frame]
    matched = []
    rows = []
    for t in frames:
        if not (0 <= t < len(sc.frames)):
            print(f"frame {t} out of range (n_frames={len(sc.frames)})")
            return 2
        ns = namespace(sc, policy)
        ns["t"] = t
        try:
            val = safe_eval(args.expr, ns)
        except (UnsafeExpression, NameError, SyntaxError) as e:
            print(f"bad query: {type(e).__name__}: {e}")
            return 2
        truth = bool(val)
        rows.append({"frame": t, "value": val if not isinstance(val, bool) else truth})
        if truth:
            matched.append(t)
    out = {
        "scene": sc.name,
        "expr": args.expr,
        "policy": args.policy,
        "n_frames": len(sc.frames),
        "matched_frames": matched,
        "n_matched": len(matched),
        "per_frame": rows,
    }
    print(json.dumps(out, indent=2, default=float))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from prism.adapt import AdapterError, ingest
    from prism.serialize import content_hash, to_parquet
    from prism.validate import ValidationError, validate_scene

    path = pathlib.Path(args.path)
    if not path.exists():
        print(f"needs data: {path} does not exist")
        return 2
    try:
        ir = ingest(path, scene=args.scene)
    except AdapterError as e:
        print(f"needs data: {e}")
        return 2
    try:
        validate_scene(ir, collect=True)
    except ValidationError as e:
        print(f"INVALID IR: {e}")
        return 1
    summary = {
        "name": ir.name,
        "n_frames": len(ir),
        "n_coordinate_frames": len(ir.coordinate_frames),
        "n_tracks": len(ir.tracks),
        "n_ground_truth_frames": len(ir.ground_truth),
        "n_observations": len(ir.observations),
        "content_hash": content_hash(ir),
        "provenance": None if ir.provenance is None else {"dataset": ir.provenance.dataset, "log_id": ir.provenance.log_id},
    }
    if args.out:
        to_parquet(ir, args.out)
        summary["written"] = str(pathlib.Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from prism.adapt import AdapterError, ingest
    from prism.serialize import to_jsonl, to_openlabel_json, to_parquet
    from prism.validate import ValidationError, validate_scene

    data = pathlib.Path(args.data)
    if not data.exists():
        print(f"needs data: {data} does not exist")
        return 2
    try:
        ir = ingest(data, scene=args.scene)
    except AdapterError as e:
        print(f"needs data: {e}")
        return 2
    try:
        validate_scene(ir, collect=True)
    except ValidationError as e:
        print(f"INVALID IR (refusing to export): {e}")
        return 1
    out = to_parquet(ir, args.out)
    written = {"parquet": str(out)}
    if args.openlabel:
        pathlib.Path(args.openlabel).write_text(json.dumps(to_openlabel_json(ir), indent=2))
        written["openlabel"] = args.openlabel
    if args.jsonl:
        pathlib.Path(args.jsonl).write_text(to_jsonl(ir))
        written["jsonl"] = args.jsonl
    print(json.dumps({"name": ir.name, "n_frames": len(ir), "written": written}, indent=2))
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    print("prism view: no viewer wired. use --backend rerun (optional) once a visualization backend is added.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prism", description="spatial-probe: query, ingest, export a Scene IR")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="evaluate a predicate DSL expression over a scene")
    q.add_argument("expr", help='e.g. "min_free_width_along_path(scene, t, 3.0) < ego_width(scene)"')
    q.add_argument("--data", required=True, help="AV2 log dir or Occ3D data root")
    q.add_argument("--scene", default=None, help="scene name (Occ3D); default = first scene")
    q.add_argument("--policy", default="free", choices=list(_POLICY), help="unknown-voxel policy")
    q.add_argument("--frame", type=int, default=None, help="single frame index; default = all frames")
    q.set_defaults(func=_cmd_query)

    ing = sub.add_parser("ingest", help="adapter -> Scene IR (validated); --out to write parquet")
    ing.add_argument("path", help="AV2 log dir or Occ3D data root")
    ing.add_argument("--scene", default=None, help="scene name (Occ3D); default = first scene")
    ing.add_argument("--out", default=None, help="optional: write the IR to this parquet")
    ing.set_defaults(func=_cmd_ingest)

    exp = sub.add_parser("export", help="ingest -> IR -> lossless parquet (+ optional openlabel/jsonl)")
    exp.add_argument("out", help="output parquet path")
    exp.add_argument("--data", required=True, help="AV2 log dir or Occ3D data root")
    exp.add_argument("--scene", default=None, help="scene name (Occ3D); default = first scene")
    exp.add_argument("--openlabel", default=None, help="optional: also write OpenLABEL JSON here")
    exp.add_argument("--jsonl", default=None, help="optional: also write per-frame JSONL here")
    exp.set_defaults(func=_cmd_export)

    v = sub.add_parser("view", help="(stub) viewer backend is optional / not wired")
    v.set_defaults(func=_cmd_view)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
