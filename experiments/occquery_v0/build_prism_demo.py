# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""PRISM demo DATA layer (B1) -- run the 3 failure signatures across a real AV2 demo log set,
aggregate a failure CATALOGUE, render annotated demo frames, and export everything the web demo reads.

READ-ONLY w.r.t. src/probe + src/prism: this is an experiments/ driver. It calls the public
`prism.failure` / `prism.adapt` / `prism.detect` API verbatim (no monkeypatching of the engine).

Honest scope (stated in the catalogue JSON, not just here):
  - missed_detection: 3 AV2 camera logs, FRAME STRIDE applied (detector is ~3.4 s/img CPU; a full
    157-frame log is ~9 min, so a stride bounds runtime). Stride is recorded per-signature.
  - path_blocked_no_box / box_in_free: 8 LiDAR-only AV2 logs, scanned FULL (no stride; ~11 s/log).
    The 8 include 78683234-... which carries the 1 known unboxed-obstacle at frame 66 (~4.2 m).

Outputs (all under web/public/data/):
  - failure_catalogue.json        per-signature aggregate (counts, clusters, honesty, scope)
  - frames/*.png + *.caption.txt  ~4-6 annotated demo frames (missed_detection + a BEV path_blocked)
  - h3b_expressivity.json         copied verbatim from results/ (the A-side headline)
  - oracle_status.json            the 3 oracle verdicts (the honesty-layer panel)

Determinism: signatures + clustering are deterministic; the detector (onnxruntime CPU) is
deterministic for a fixed image. The stride is fixed. No randomness is introduced here.

Run:  .venv/bin/python experiments/occquery_v0/build_prism_demo.py
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from prism.adapt import ingest  # noqa: E402
from prism.failure import (  # noqa: E402
    cluster,
    load_av2_camera,
    match_detections,
    mine,
    project_ego_boxes,
    resolve_signature,
)

DATA_ROOT = REPO / "data" / "danger" / "av2_sensor"
OUT_DIR = REPO / "web" / "public" / "data"
FRAMES_DIR = OUT_DIR / "frames"
RESULTS = REPO / "experiments" / "occquery_v0" / "results"

# === demo log set (honest, fixed) ================================================================
# missed_detection NEEDS cameras: the 3 logs that carry sensors/cameras + calibration.
CAMERA_LOGS = [
    "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c",
    "201fe83b-7dd7-38f4-9d26-7b4a668638a9",
    "2c652f9e-8db8-3572-aa49-fae1344a875b",
]
# path_blocked_no_box / box_in_free are LiDAR-only. 8 logs incl. the known-unboxed-obstacle log.
LIDAR_LOGS = [
    "78683234-e6f1-3e4e-af52-6f839254e4c0",  # frame 66 ~4.2 m known unboxed obstacle
    "070bbf42-31d3-3aa9-aca4-c262afc9077d",
    "0b5142c1-420b-3fea-9e98-b87327ae22c6",
    "0fb7276f-ecb5-3e5b-87a8-cc74c709c715",
    "19350c96-623d-4d77-af96-f8c23f00c358",
    "2ff4f798-78d9-3384-87e9-61928aa4cb6d",
    "51bbdd4d-3065-34ae-b369-b6e0444f34db",
    "7039e410-b5ab-35aa-96bc-2c4b89d3c5e3",
]
# Detector budget: a full 157-frame log is ~9 min CPU. A stride of 5 -> ~32 frames/log -> ~1.8 min/log.
MISSED_DETECTION_STRIDE = 5
MAX_IMG_PX = 1100  # web-friendly downsize cap for the annotated camera frames


def _stride_scene_ir(ir, stride: int):
    """Return a copy of `ir` whose Scene keeps every `stride`-th frame, re-indexed 0..k-1.

    The signature auto-fill (interior_pts / detections) re-derives its lookups from
    `ir.scene.frames` by enumeration, so a re-indexed subset stays self-consistent. Frozen
    dataclasses are copied via dataclasses.replace (no engine mutation)."""
    if stride <= 1:
        return ir, list(range(len(ir.scene.frames)))
    kept = list(range(0, len(ir.scene.frames), stride))
    new_frames = tuple(ir.scene.frames[i] for i in kept)
    new_scene = dataclasses.replace(ir.scene, frames=new_frames)
    new_ir = dataclasses.replace(ir, scene=new_scene)
    return new_ir, kept


def _category_label(code: float) -> str:
    inv = {1.0: "vehicle", 2.0: "pedestrian", 3.0: "bicycle", 4.0: "motorcycle", 5.0: "other", 0.0: "n/a"}
    return inv.get(round(code, 1), "other")


def _aggregate_signature(name: str, logs_ir, *, stride: int, params: dict, scope_note: str) -> dict:
    """Mine one signature over the given SceneIRs, cluster the hits, and build the catalogue entry.

    Records the VERBATIM honesty tag from the Signature, the per-signature scope (logs, stride,
    frames scanned), candidate count, cluster count, and the top clusters (range bin, size, category)."""
    sig = resolve_signature(name)
    t0 = time.time()
    candidates = mine(logs_ir, sig, params=params)
    clusters = cluster(candidates, range_bin_m=float(params.get("range_bin_m", 8.0)))
    elapsed = time.time() - t0

    n_frames = sum(len(ir.scene.frames) for ir in logs_ir)
    top_clusters = []
    for cl in clusters[:8]:
        cat = _category_label(cl.centroid.get("category_code", 0.0))
        top_clusters.append({
            "name": cl.slice.name,
            "size": len(cl.candidates),
            "range_bin_m": round(cl.centroid["forward_range_m"], 2),
            "category": cat,
            "example_frames": list(cl.slice.frame_indices)[:6],
            "example_logs": sorted({c.log_id[:8] for c in cl.candidates})[:6],
        })

    ranges = [c.features.get("forward_range_m", 0.0) for c in candidates]
    ranges = [r for r in ranges if np.isfinite(r)]
    return {
        "signature": sig.name,
        "description": sig.description,
        "honesty": sig.honesty,  # VERBATIM from failure.py
        "n_logs": len(logs_ir),
        "log_ids": [ir.provenance.log_id if ir.provenance else ir.name for ir in logs_ir],
        "stride": stride,
        "n_frames_scanned": n_frames,
        "n_candidates": len(candidates),
        "n_clusters": len(clusters),
        "mean_forward_range_m": round(float(np.mean(ranges)), 2) if ranges else None,
        "top_clusters": top_clusters,
        "scope_note": scope_note,
        "mine_seconds": round(elapsed, 1),
    }


# === annotated demo frames =======================================================================

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def _font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _missed_detection_frames(max_frames: int = 3) -> list[dict]:
    """Render annotated missed_detection frames: camera image + green detector boxes + red MISSED
    GT boxes. Picks frames with BOTH >=1 detection AND >=1 clear miss (the visceral contrast).

    Reuses the proven logic from /tmp/prism_demo.py (load_av2_camera, detect_image, project_ego_boxes,
    match_detections class_agnostic, PIL draw) -- run through the same public API the engine uses."""
    from prism.detect import detect_image

    from probe.adapters import av2_sensor

    written = []
    target_labels = frozenset({"pedestrian", "bicycle", "vehicle", "motorcycle"})
    for log in CAMERA_LOGS:
        if len(written) >= max_frames:
            break
        root = DATA_ROOT / log
        cam = load_av2_camera(root, "ring_front_center")
        if cam is None:
            continue
        cam_dir = root / "sensors" / "cameras" / "ring_front_center"
        cam_ts = sorted(int(p.stem) for p in cam_dir.glob("*.jpg"))
        if not cam_ts:
            continue
        cam_arr = np.asarray(cam_ts, dtype=np.int64)
        sweeps = sorted(int(p.stem) for p in (root / "sensors" / "lidar").glob("*.feather"))
        scene = av2_sensor.load_scene(log, str(DATA_ROOT), with_boxes=True, timestamps=sweeps)

        best = None  # (n_missed, frame_idx, cam_ts, jpg, missed, dets)
        # scan a strided set of frames so we don't run the detector on all 157
        for i in range(0, len(scene.frames), 4):
            boxes = list(scene.frames[i].objects)
            if not boxes:
                continue
            lidar_ts = sweeps[i]
            ci = int(np.argmin(np.abs(cam_arr - lidar_ts)))
            cts = int(cam_arr[ci])
            jpg = cam_dir / f"{cts}.jpg"
            if not jpg.exists():
                continue
            dets = detect_image(jpg, camera="ring_front_center", timestamp_ns=cts)
            if len(dets) < 2:
                continue
            proj = project_ego_boxes(boxes, cam)
            missed = []
            for b, (vis, b2d, depth) in zip(boxes, proj):
                if not vis or b2d is None or not (3.0 < depth < 60.0):
                    continue
                lab = str(b.label)
                if lab not in target_labels:
                    continue
                if not match_detections(b2d, lab, dets, class_agnostic=True):
                    missed.append((b2d, lab, depth))
            if missed and (best is None or len(missed) > best[0]):
                best = (len(missed), i, cts, jpg, missed, dets)
            if best and best[0] >= 2:
                break
        if best is None:
            continue
        nmiss, i, cts, jpg, missed, dets = best
        out_png, caption = _draw_missed(log, i, cts, jpg, missed, dets)
        written.append({"signature": "missed_detection", "log": log, "frame_index": i,
                        "cam_timestamp_ns": cts, "n_detected": len(dets), "n_missed": nmiss,
                        "png": out_png, "caption": caption})
    return written


def _draw_missed(log, frame_idx, cts, jpg, missed, dets) -> tuple[str, str]:
    img = Image.open(jpg).convert("RGB")
    d = ImageDraw.Draw(img)
    f = _font(30)
    fs = _font(24)
    for det in dets:
        x0, y0, x1, y1 = det.box_xyxy
        d.rectangle([x0, y0, x1, y1], outline=(40, 220, 80), width=4)
        d.text((x0 + 3, max(0, y0 - 26)), f"{det.av2_label} {det.score:.2f}", fill=(40, 220, 80), font=fs)

    def _plate(xy, text, font, fg):
        """Draw text on a dark plate so a red MISSED label stays legible over a busy background."""
        x, y = xy
        bb = d.textbbox((x, y), text, font=font)
        d.rectangle([bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2], fill=(20, 0, 0))
        d.text((x, y), text, fill=fg, font=font)

    # Stagger MISSED labels: distant misses cluster, so left-to-right order + a per-row vertical push
    # keeps overlapping labels readable instead of smearing into one blob.
    row_h = 32
    placed: list[tuple[float, float, float, float]] = []
    for b2d, lab, depth in sorted(missed, key=lambda m: m[0][0]):
        x0, y0, x1, y1 = b2d
        d.rectangle([x0, y0, x1, y1], outline=(240, 40, 40), width=5)
        lx, ly = x0, max(60, y0 - row_h)
        # if this label's x-range overlaps an already-placed one near the same y, push it up a row
        bumped = True
        while bumped:
            bumped = False
            for px0, py0, px1, py1 in placed:
                if not (lx > px1 + 6 or lx + 180 < px0 - 6) and abs(ly - py0) < row_h:
                    ly = max(60, ly - row_h)
                    bumped = True
                    break
        _plate((lx, ly), f"MISSED {lab} @{depth:.0f}m", f, (255, 70, 70))
        placed.append((lx, ly, lx + 180, ly + row_h))

    bar_h = 50
    d.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    d.text((12, 11), f"missed_detection  green=detector saw  red=model MISSED ({len(missed)})",
           fill=(255, 255, 255), font=f)
    # downsize to web-friendly width
    if img.width > MAX_IMG_PX:
        scale = MAX_IMG_PX / img.width
        img = img.resize((MAX_IMG_PX, int(img.height * scale)))
    name = f"missed_detection_{log[:8]}_f{frame_idx}"
    # photographic camera frame -> JPEG (PNG re-encode of a photo is ~10x larger for no gain).
    out_png = FRAMES_DIR / f"{name}.jpg"
    img.save(out_png, format="JPEG", quality=85)
    miss_desc = "; ".join(f"{lab}@{depth:.0f}m" for _, lab, depth in missed)
    caption = (
        f"missed_detection -- log {log[:8]} frame {frame_idx} (cam ts {cts}). "
        f"The COCO-YOLOv8n detector output {len(dets)} detections (green) but FAILED to see "
        f"{len(missed)} camera-visible GT object(s) (red): {miss_desc}. "
        f"Honest: cross-distribution recall (COCO model, not trained on AV2), coarse class map "
        f"(car/bus/truck -> vehicle), class-agnostic match; a miss = the detector's recall failure, "
        f"NOT an occupancy claim."
    )
    (FRAMES_DIR / f"{name}.caption.txt").write_text(caption)
    return str(out_png.relative_to(OUT_DIR)), caption


def _path_blocked_bev_frame() -> dict | None:
    """Render a top-down BEV PNG of the path_blocked_no_box hit at frame 66 of 78683234 (no camera):
    occupancy obstacle voxels (gray), ego in-path band (light), the BLOCKED location (red),
    tracked boxes (blue) -- 'occupancy blocks here, no box explains it'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    log = "78683234-e6f1-3e4e-af52-6f839254e4c0"
    frame_idx = 66
    ir = ingest(DATA_ROOT / log)
    sc = ir.scene
    t = frame_idx
    grid = sc.grid_at(t)
    ego = sc.ego_at(t)
    boxes = sc.objects_at(t)

    centers = grid.obstacle_centers(max_height_agl=ego.height)  # (N,3) world (ego-frame for AV2)
    # forward block range (reuse the same geometry the signature uses)
    fwd, lat = ego.to_ego_frame(centers[:, :2]) if len(centers) else (np.array([]), np.array([]))
    half = ego.width / 2.0 + grid.voxel_size

    # BEV window: ego-centric, +-25 m forward, +-15 m lateral. Plot in ego frame (forward = up).
    fig, ax = plt.subplots(figsize=(7.5, 9.0), dpi=120)
    win_fwd, win_lat = 28.0, 16.0

    if len(centers):
        m = (fwd > -8.0) & (fwd < win_fwd) & (np.abs(lat) < win_lat)
        ax.scatter(lat[m], fwd[m], s=6, c="#888888", marker="s", linewidths=0, label="occupancy obstacle voxels")

    # ego in-path band (the corridor the freepath predicate scans)
    band = Rectangle((-half, 0.0), 2 * half, win_fwd, facecolor="#cfe8ff", edgecolor="none", alpha=0.45,
                     zorder=0, label="ego in-path band")
    ax.add_patch(band)

    # tracked boxes (blue), in ego frame
    drew_box_label = False
    for b in boxes:
        bf, bl = ego.to_ego_frame(np.array([[b.center[0], b.center[1]]]))
        bf, bl = float(bf[0]), float(bl[0])
        if not (-8.0 < bf < win_fwd and abs(bl) < win_lat):
            continue
        # approximate footprint as an axis-aligned rect in ego frame (length along forward)
        L, W = b.size[0], b.size[1]
        rect = Rectangle((bl - W / 2, bf - L / 2), W, L, facecolor="none", edgecolor="#1f6fff",
                         linewidth=2.0, zorder=3, label=(None if drew_box_label else "tracked box"))
        ax.add_patch(rect)
        drew_box_label = True

    # the BLOCKED location: nearest in-band obstacle voxel ahead (red marker)
    block_fwd = None
    if len(centers):
        bandmask = (fwd > 0.0) & (np.abs(lat) <= half)
        if bandmask.any():
            j = int(np.argmin(fwd[bandmask]))
            block_fwd = float(fwd[bandmask][j])
            block_lat = float(lat[bandmask][j])
            ax.scatter([block_lat], [block_fwd], s=420, marker="*", c="#e10000",
                       edgecolors="black", linewidths=1.2, zorder=5,
                       label=f"BLOCKED @ {block_fwd:.1f} m (no box explains it)")

    # ego marker at origin
    ax.scatter([0], [0], s=160, marker="^", c="black", zorder=6, label="ego")
    ax.set_xlim(-win_lat, win_lat)
    ax.set_ylim(-8.0, win_fwd)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)  <- left   right ->")
    ax.set_ylabel("forward (m)")
    ax.set_title(f"path_blocked_no_box  --  {log[:8]} frame {frame_idx}\n"
                 f"occupancy blocks the in-path band; no tracked box within 5 m explains it", fontsize=11)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    name = f"path_blocked_no_box_{log[:8]}_f{frame_idx}_bev"
    out_png = FRAMES_DIR / f"{name}.png"
    fig.savefig(out_png)
    plt.close(fig)

    block_str = f"{block_fwd:.1f} m" if block_fwd is not None else "in-corridor"
    caption = (
        f"path_blocked_no_box -- log {log[:8]} frame {frame_idx} (LiDAR-only, no camera; BEV render). "
        f"Occupancy obstacle voxels (gray) block the ego in-path band (blue) at ~{block_str} forward "
        f"(red star), and NO tracked box (blue rect) within 5 m explains it. This is the H1 "
        f"expressivity win a box-only language is structurally blind to: either a REAL unboxed "
        f"obstacle or an occupancy FP -- and the traversal oracle (RELIABLE) says occupancy does NOT "
        f"hallucinate obstacles on the driven path, so a real unboxed obstacle is the likelier reading. "
        f"honesty tag: external-fp."
    )
    (FRAMES_DIR / f"{name}.caption.txt").write_text(caption)
    return {"signature": "path_blocked_no_box", "log": log, "frame_index": frame_idx,
            "block_forward_m": block_fwd, "png": str(out_png.relative_to(OUT_DIR)), "caption": caption}


# === oracle status (honesty-layer panel) =========================================================


def _oracle_status() -> dict:
    """Summarize the 3 oracle verdicts from the results JSONs (the honesty layer the demo headlines).
    Reads the PRIMARY result files -- not a re-derivation -- so the verdicts match the sealed runs."""
    trav = json.loads((RESULTS / "oracle_traversal.json").read_text())
    box = json.loads((RESULTS / "oracle_box_recall.json").read_text())
    depth = json.loads((RESULTS / "oracle_depth_recall.json").read_text())
    return {
        "note": "The three oracles that gate PRISM's failure claims. Honest: only the traversal "
                "(FP-direction) oracle is externally anchored; box-recall is same-modality "
                "consistency; depth is INVALID on this substrate (reported as a negative, not hidden).",
        "oracles": [
            {
                "name": "traversal (occupancy FP on the driven path)",
                "verdict": trav["verdict"],
                "anchors": "path_blocked_no_box (external-fp)",
                "headline": f"true FP {trav['true_fp_mean']:.3f} vs shuffled {trav['shuffled_fp_mean']:.3f} "
                            f"(n={trav['n_frames']} frames, {trav['n_logs']} held-out free-driving logs); "
                            f"RELIABLE = true CI below shuffled CI. Occupancy does NOT hallucinate "
                            f"obstacles on space the ego physically drove through.",
                "external": True,
            },
            {
                "name": "box-recall (LiDAR-seen boxes occupancy marks FREE)",
                "verdict": box["verdict"],
                "anchors": "box_in_free (consistency-only)",
                "headline": f"true miss {box['true_miss_mean']:.3f} vs size/range-matched null "
                            f"{box['null_miss_mean']:.3f} (n_interior>={box['n_interior_min']}, "
                            f"{box['n_logs']} logs). RECALL-SUPPORTED but SAME-MODALITY (gates on the "
                            f"same LiDAR the voxelizer reads) -> internal consistency, NOT external truth; "
                            f"absolute count inflated by the _ROAD_Z floor-straddle confound.",
                "external": False,
            },
            {
                "name": "depth-recall (frozen DAv2 metric depth as an independent recall route)",
                "verdict": depth["verdict"],
                "anchors": "would-have externally anchored box recall -- CLOSED",
                "headline": f"{depth['reason']}. The externally-independent recall route is honestly "
                            f"CLOSED on this substrate (also: classical stereo AUC 0.259). Reported as a "
                            f"NEGATIVE, not buried.",
                "external": False,
            },
        ],
    }


# === main ========================================================================================


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("== ingesting demo logs ==", flush=True)
    # LiDAR signatures: full-frame scan over the 8 LiDAR logs.
    lidar_irs = [ingest(DATA_ROOT / lg) for lg in LIDAR_LOGS]
    print(f"  ingested {len(lidar_irs)} LiDAR logs", flush=True)

    # missed_detection: 3 camera logs, strided to bound detector runtime.
    cam_irs_full = [ingest(DATA_ROOT / lg) for lg in CAMERA_LOGS]
    cam_irs = []
    for ir in cam_irs_full:
        sir, _ = _stride_scene_ir(ir, MISSED_DETECTION_STRIDE)
        cam_irs.append(sir)
    print(f"  ingested {len(cam_irs)} camera logs (stride {MISSED_DETECTION_STRIDE})", flush=True)

    catalogue = {
        "schema": "prism_failure_catalogue/v1",
        "generated_by": "experiments/occquery_v0/build_prism_demo.py",
        "data_root": str(DATA_ROOT),
        "demo_scope": {
            "missed_detection": {
                "logs": CAMERA_LOGS,
                "stride": MISSED_DETECTION_STRIDE,
                "note": "3 AV2 camera logs (the only logs with sensors/cameras + calibration); "
                        "frame stride to bound CPU detector runtime (~3.4 s/img).",
            },
            "path_blocked_no_box": {
                "logs": LIDAR_LOGS,
                "stride": 1,
                "note": "8 LiDAR-only AV2 logs, FULL scan (no stride). Incl. 78683234-... with the "
                        "1 known unboxed-obstacle at frame 66 (~4.2 m).",
            },
            "box_in_free": {
                "logs": LIDAR_LOGS,
                "stride": 1,
                "note": "same 8 LiDAR-only AV2 logs, FULL scan (no stride).",
            },
            "honest_caveat": "Demo scope is a BOUNDED SUBSET of the AV2 corpus (8 LiDAR logs + 3 camera "
                             "logs, missed_detection strided). NOT a full-corpus sweep. Per-signature "
                             "honesty tags are verbatim from src/prism/failure.py.",
        },
        "signatures": {},
    }

    print("== mining missed_detection (this runs the detector; slowest) ==", flush=True)
    catalogue["signatures"]["missed_detection"] = _aggregate_signature(
        "missed_detection", cam_irs,
        stride=MISSED_DETECTION_STRIDE,
        params={"class_agnostic": True, "score_thr": 0.25, "range_bin_m": 8.0},
        scope_note=f"3 camera logs, stride {MISSED_DETECTION_STRIDE}, class-agnostic match, score>=0.25.",
    )
    print("   ", catalogue["signatures"]["missed_detection"]["n_candidates"], "candidates", flush=True)

    print("== mining path_blocked_no_box ==", flush=True)
    catalogue["signatures"]["path_blocked_no_box"] = _aggregate_signature(
        "path_blocked_no_box", lidar_irs,
        stride=1,
        params={"horizon": 1.0, "box_radius_m": 5.0, "range_bin_m": 4.0},
        scope_note="8 LiDAR logs, full scan, horizon 1.0 s, box_radius 5 m.",
    )
    print("   ", catalogue["signatures"]["path_blocked_no_box"]["n_candidates"], "candidates", flush=True)

    print("== mining box_in_free ==", flush=True)
    catalogue["signatures"]["box_in_free"] = _aggregate_signature(
        "box_in_free", lidar_irs,
        stride=1,
        params={"n_interior_min": 5, "range_bin_m": 8.0},
        scope_note="8 LiDAR logs, full scan, LiDAR-seen gate n_interior>=5.",
    )
    print("   ", catalogue["signatures"]["box_in_free"]["n_candidates"], "candidates", flush=True)

    (OUT_DIR / "failure_catalogue.json").write_text(json.dumps(catalogue, indent=2, default=float))
    print("wrote failure_catalogue.json", flush=True)

    print("== rendering missed_detection frames ==", flush=True)
    frames = _missed_detection_frames(max_frames=3)
    print(f"   {len(frames)} missed_detection frames", flush=True)

    print("== rendering path_blocked BEV frame ==", flush=True)
    bev = _path_blocked_bev_frame()
    if bev is not None:
        frames.append(bev)
        print("   BEV frame written", flush=True)

    frames_manifest = {
        "schema": "prism_frames_manifest/v1",
        "frames": [{k: v for k, v in fr.items() if k != "caption"} | {"caption": fr["caption"]} for fr in frames],
    }
    (OUT_DIR / "frames_manifest.json").write_text(json.dumps(frames_manifest, indent=2, default=float))
    print("wrote frames_manifest.json", flush=True)

    print("== copying h3b_expressivity.json + writing oracle_status.json ==", flush=True)
    shutil.copyfile(RESULTS / "h3b_expressivity.json", OUT_DIR / "h3b_expressivity.json")
    (OUT_DIR / "oracle_status.json").write_text(json.dumps(_oracle_status(), indent=2, default=float))
    print("wrote h3b_expressivity.json, oracle_status.json", flush=True)

    print("\n== DONE ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
