# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""`aletheon` CLI -- run the spatial-probe instrument from the shell.

Each verb runs end-to-end on real on-disk data (or prints a clear "needs data" message; never a
stub that pretends). The query verb routes through `aletheon.query.namespace` + `probe.query_dsl.safe_eval`
(the SAME AST-whitelist + ONE shared predicate namespace the programmatic `aletheon.query.query` API
uses -- a query expression is untrusted input), so the CLI and the API never drift apart. That
namespace carries the S3 physical predicates (occluded / ttc / object_speed / velocity) on top of
the existing occupancy/box predicates.

Verbs:
  aletheon query "<expr>" --data <root> [--scene <name>] [--policy free|occupied|ignored] [--frame N]
      Evaluate a predicate DSL expression over a scene's frames; print per-frame truth + the
      matched frame indices. Reuses the occupancy/box predicates verbatim.
  aletheon ingest <path> [--scene <name>] [--out <ir.parquet>]
      Adapter -> Scene IR; validate it; print a summary (or write the IR parquet with --out).
  aletheon export <out.parquet> --data <path> [--scene <name>] [--openlabel <out.json>] [--jsonl <out.jsonl>]
      Ingest -> IR -> write a lossless parquet (and optional OpenLABEL / JSONL views).
  aletheon view --backend rerun --data <path> [--scene <name>] [--out <log.rrd>] [--max-points N]
      Ingest -> IR -> render to a Rerun recording (.rrd). Rerun is an OPTIONAL extra; without
      rerun-sdk installed this prints the install hint and exits non-zero (never a faked render).
  aletheon find "<query>" --data <corpus-or-log> [--scene <name>] [--horizon S] [--box-radius M]
             [--n-interior-min N] [--camera C] [--score-thr T] [--limit-frames N]
             [--render rerun [--out <log.rrd>]] [--json]
      The S6 wow verb. Map a natural-ish query to a failure SIGNATURE, mine it over the corpus, and
      print a clean human summary + structured JSON: how many matching frame-intervals, the signature,
      mean range, top categories, the dominant cluster, and K most-similar frames (FEATURE-distance,
      not semantic). Honest counts -- zero is a real negative, never inflated. Signatures:
        - path_blocked_no_box / box_in_free: the H1 occupancy-vs-box set difference (prediction-free;
          consistency / FP-direction, NOT model-eval).
        - missed_detection ("vehicle missed by the detector"): REAL model-eval -- a COCO-YOLOv8n 2D
          detector run on the camera image vs the AV2 GT boxes; a GT box visible in-frame with no
          matching detection = the detector failed (Ramanagopal-style). Cross-distribution recall
          (COCO model, not trained on AV2), coarse class map -- detector-eval, NOT occupancy-eval.
          Needs the optional [detect] extra (onnxruntime+pillow) and the gitignored yolov8n.onnx.
      `--data` may be a single AV2 log dir OR the av2_sensor corpus root.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from probe.grid import UnknownPolicy
from probe.query_dsl import UnsafeExpression, safe_eval
from aletheon.query import namespace

_POLICY = {"free": UnknownPolicy.FREE, "occupied": UnknownPolicy.OCCUPIED, "ignored": UnknownPolicy.IGNORED}


def _load_scene_for_query(data: pathlib.Path, scene: str | None, with_boxes: bool):
    """Load a probe.Scene from either dataset layout (mirrors aletheon.adapt.ingest detection but
    returns the raw probe.Scene the predicates run on)."""
    from aletheon.adapt import _is_av2_log

    if _is_av2_log(data):
        from probe.adapters.av2_sensor import load_scene

        return load_scene(data.name, data.parent, with_boxes=with_boxes)
    if (data / "annotations.json").exists() and (data / "gts").is_dir():
        from probe.adapters.occ3d import load_scene

        if scene is None:
            scene = sorted(json.loads((data / "annotations.json").read_text())["scene_infos"].keys())[0]
        return load_scene(scene, data, mask="none", with_boxes=with_boxes)
    return None


def _query_all_scenes(args: argparse.Namespace, data: pathlib.Path) -> int:
    """Corpus-scale query over an Occ3D root: evaluate the expression on EVERY scene (sorted),
    print the matching scenes + per-scene matched frames — the measurement-search verb over a
    whole labeled corpus. Loader failures are reported to stderr, never silently dropped."""
    if not ((data / "annotations.json").exists() and (data / "gts").is_dir()):
        print(f"needs data: --all-scenes requires an Occ3D root (annotations.json + gts), got {data}")
        return 2
    from probe.adapters.occ3d import _annotations, load_scene

    names = sorted(_annotations(data)["scene_infos"].keys())
    if args.limit:
        names = names[: args.limit]
    needs_boxes = any(p in args.expr for p in ("distance_to_nearest_object", "ttc", "object_speed", "velocity"))
    policy = _POLICY[args.policy]
    scenes_out = []
    n_loaded = 0
    for name in names:
        try:
            sc = load_scene(name, data, mask="none", with_boxes=needs_boxes)
        except Exception as e:  # noqa: BLE001 - a corrupt scene is visible, not fatal to the scan
            print(f"[skip] {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        n_loaded += 1
        matched = []
        for t in range(len(sc.frames)):
            ns = namespace(sc, policy)
            ns["t"] = t
            try:
                val = safe_eval(args.expr, ns)
            except (UnsafeExpression, NameError, SyntaxError) as e:
                print(f"bad query: {type(e).__name__}: {e}")
                return 2
            if bool(val):
                matched.append(t)
        if matched:
            scenes_out.append({"scene": name, "matched_frames": matched, "n_matched": len(matched)})
    print(json.dumps({"expr": args.expr, "policy": args.policy, "n_scenes": n_loaded,
                      "n_scenes_matched": len(scenes_out), "scenes": scenes_out},
                     indent=2, default=float))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    data = pathlib.Path(args.data)
    if not data.exists():
        print(f"needs data: {data} does not exist (point --data at an AV2 log dir or an Occ3D root)")
        return 2
    if args.all_scenes:
        return _query_all_scenes(args, data)
    # box-reading predicates need the tracked boxes loaded; occupancy/occlusion predicates do not.
    needs_boxes = any(p in args.expr for p in ("distance_to_nearest_object", "ttc", "object_speed", "velocity"))
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
    from aletheon.adapt import AdapterError, ingest
    from aletheon.serialize import content_hash, to_parquet
    from aletheon.validate import ValidationError, validate_scene

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
    from aletheon.adapt import AdapterError, ingest
    from aletheon.serialize import to_jsonl, to_openlabel_json, to_parquet
    from aletheon.validate import ValidationError, validate_scene

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
    """Render a scene to a Rerun recording (.rrd). Rerun is an OPTIONAL backend -- this verb is the
    ONLY CLI path that can reach it, and only when --backend rerun is passed AND rerun-sdk is
    installed. Without rerun-sdk it prints the install hint and exits non-zero (never a fake render).
    """
    if args.backend != "rerun":
        print(f"aletheon view: unknown backend {args.backend!r} (only 'rerun' is wired)")
        return 2
    data = pathlib.Path(args.data)
    if not data.exists():
        print(f"needs data: {data} does not exist (point --data at an AV2 log dir or an Occ3D root)")
        return 2

    from aletheon.adapt import AdapterError, ingest

    try:
        ir = ingest(data, scene=args.scene)
    except AdapterError as e:
        print(f"needs data: {e}")
        return 2

    from aletheon.adapters.rerun_adapter import RerunNotInstalled, to_rerun

    out = args.out or f"{ir.name}.rrd"
    try:
        written = to_rerun(ir, out, max_points_per_frame=args.max_points)
    except RerunNotInstalled as e:
        print(str(e))
        return 2
    print(json.dumps({"name": ir.name, "n_frames": len(ir), "backend": "rerun", "written": str(written)}, indent=2))
    return 0


def _resolve_find_logs(data: pathlib.Path, scene: str | None):
    """Resolve --data to a list of SceneIRs. `data` may be a single recognized dataset (one AV2 log
    dir or an Occ3D root) OR a CORPUS ROOT holding many AV2 log subdirs (the data/danger/av2_sensor
    case). Returns (irs, label) or (None, reason) -- never a stub."""
    from aletheon.adapt import AdapterError, _is_av2_log, ingest

    if _is_av2_log(data) or ((data / "annotations.json").exists() and (data / "gts").is_dir()):
        try:
            return [ingest(data, scene=scene)], data.name
        except AdapterError as e:
            return None, str(e)
    # corpus root: every immediate subdir that is itself an AV2 log
    if data.is_dir():
        logs = sorted(p for p in data.iterdir() if p.is_dir() and _is_av2_log(p))
        if logs:
            return [ingest(p) for p in logs], f"{data.name} ({len(logs)} logs)"
    return None, f"{data} is not a recognized dataset or AV2 corpus root"


def _cmd_find(args: argparse.Namespace) -> int:
    from aletheon.failure import find

    data = pathlib.Path(args.data)
    if not data.exists():
        print(f"needs data: {data} does not exist (point --data at an AV2 log dir or the av2_sensor corpus root)")
        return 2
    irs, label = _resolve_find_logs(data, args.scene)
    if irs is None:
        print(f"needs data: {label}")
        return 2
    params = {
        "horizon": args.horizon,
        "box_radius_m": args.box_radius,
        "n_interior_min": args.n_interior_min,
        # missed_detection knobs (ignored by the other signatures):
        "camera": args.camera,
        "iou_thr": args.iou_thr,
        "score_thr": args.score_thr,
        "class_agnostic": args.class_agnostic,
        "min_range_m": args.min_range,
        "max_range_m": args.max_range,
    }
    if args.limit_frames is not None:
        params["limit_frames"] = args.limit_frames
    try:
        summary = find(args.query, irs, params=params)
    except ValueError as e:  # query maps to no signature -> fail loudly, never default silently
        print(f"no signature for query: {e}")
        return 2

    # the wow: human summary first, then structured JSON (unless --json-only).
    if not args.json_only:
        print(summary["human_summary"])
        print()
    if args.json_only or not args.no_json:
        print(json.dumps(summary, indent=2, default=float))

    # optional render of the dominant cluster's frames to a .rrd (reuses the rerun adapter; lazy).
    if args.render == "rerun" and summary["clusters"]:
        rc = _render_cluster(irs, summary, args.out)
        if rc != 0:
            return rc
    return 0


def _render_cluster(irs, summary, out: str | None) -> int:
    """Write a .rrd of the dominant cluster's source SceneIR (Rerun is optional/lazy -- no rerun-sdk
    -> print the install hint, non-zero, never a fake render). Renders the first log that owns a
    cluster frame; the cluster's frame_indices are noted in the summary for navigation."""
    from aletheon.adapters.rerun_adapter import RerunNotInstalled, to_rerun

    c0 = summary["clusters"][0]
    target_log = None
    for ir in irs:
        lid = ir.provenance.log_id if ir.provenance else ir.name
        if any(cand_log == lid for cand_log in [c0["name"].split("@")[0]]) or len(irs) == 1:
            target_log = ir
            break
    target_log = target_log or irs[0]
    path = out or f"{target_log.name}_{c0['signature']}.rrd"
    try:
        written = to_rerun(target_log, path)
    except RerunNotInstalled as e:
        print(str(e))
        return 2
    print(json.dumps({"render": "rerun", "cluster": c0["name"], "written": str(written)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aletheon", description="spatial-probe: query, ingest, export a Scene IR")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="evaluate a predicate DSL expression over a scene")
    q.add_argument("expr", help='e.g. "min_free_width_along_path(scene, t, 3.0) < ego_width(scene)"')
    q.add_argument("--data", required=True, help="AV2 log dir or Occ3D data root")
    q.add_argument("--scene", default=None, help="scene name (Occ3D); default = first scene")
    q.add_argument("--policy", default="free", choices=list(_POLICY), help="unknown-voxel policy")
    q.add_argument("--frame", type=int, default=None, help="single frame index; default = all frames")
    q.add_argument("--all-scenes", dest="all_scenes", action="store_true",
                   help="scan EVERY scene of an Occ3D root (corpus-scale measurement search)")
    q.add_argument("--limit", type=int, default=0, help="with --all-scenes: cap scanned scenes")
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

    v = sub.add_parser("view", help="render a scene to a viewer backend (rerun, optional extra)")
    v.add_argument("--backend", default="rerun", choices=["rerun"], help="visualization backend")
    v.add_argument("--data", required=True, help="AV2 log dir or Occ3D data root")
    v.add_argument("--scene", default=None, help="scene name (Occ3D); default = first scene")
    v.add_argument("--out", default=None, help="output .rrd path; default = <scene>.rrd")
    v.add_argument("--max-points", dest="max_points", type=int, default=None, help="cap occupancy points per frame (view-only decimation)")
    v.set_defaults(func=_cmd_view)

    f = sub.add_parser("find", help="S6: map a natural-ish query to a failure signature, mine, summarize")
    f.add_argument("query", help='e.g. "path blocked but no tracked object explains it"')
    f.add_argument("--data", required=True, help="AV2 log dir, Occ3D root, or the av2_sensor corpus root")
    f.add_argument("--scene", default=None, help="scene name (Occ3D); default = first scene")
    f.add_argument("--horizon", type=float, default=1.0, help="ego forward look-ahead seconds for the path block")
    f.add_argument("--box-radius", dest="box_radius", type=float, default=5.0, help="a box within this many m EXPLAINS a block")
    f.add_argument("--n-interior-min", dest="n_interior_min", type=int, default=5, help="LiDAR-seen gate for box_in_free")
    f.add_argument("--camera", default="ring_front_center", help="camera for missed_detection (AV2 ring/stereo name)")
    f.add_argument("--iou-thr", dest="iou_thr", type=float, default=0.3, help="missed_detection: GT/detection IoU match thresh")
    f.add_argument("--score-thr", dest="score_thr", type=float, default=0.25, help="missed_detection: detector confidence thresh")
    f.add_argument("--class-agnostic", dest="class_agnostic", action="store_true", help="missed_detection: match ignoring class")
    f.add_argument("--min-range", dest="min_range", type=float, default=2.0, help="missed_detection: min GT box depth (m)")
    f.add_argument("--max-range", dest="max_range", type=float, default=80.0, help="missed_detection: max GT box depth (m)")
    f.add_argument("--limit-frames", dest="limit_frames", type=int, default=None, help="cap frames per log (faster scan)")
    f.add_argument("--render", default=None, choices=["rerun"], help="optional: write a .rrd of the dominant cluster")
    f.add_argument("--out", default=None, help="output .rrd path when --render rerun")
    f.add_argument("--json", dest="json_only", action="store_true", help="print ONLY the structured JSON")
    f.add_argument("--no-json", dest="no_json", action="store_true", help="print ONLY the human summary")
    f.set_defaults(func=_cmd_find)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
