# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Aletheon demo DATA layer (B1) -- run the 3 failure signatures across a real AV2 demo log set,
aggregate a failure CATALOGUE, render annotated demo frames, and export everything the web demo reads.

READ-ONLY w.r.t. src/probe + src/aletheon: this is an experiments/ driver. It calls the public
`aletheon.failure` / `aletheon.adapt` / `aletheon.detect` API verbatim (no monkeypatching of the engine).

Honest scope (stated in the catalogue JSON, not just here):
  - missed_detection: a BOUNDED subset of 3 AV2 camera logs, FRAME STRIDE applied (detector-bound),
    so it is NOT a full-corpus detector sweep. Stride is recorded per-signature.
  - path_blocked_no_box / box_in_free: the FULL AV2 corpus on disk -- EVERY log with annotations +
    lidar (derived from disk at runtime), scanned FULL (no stride; detector-free + fast). The corpus
    includes 78683234-... which carries the known unboxed-obstacle at frame 66 (~4.2 m).

Outputs (under web/public/data/ AND mirrored to web/app/aletheon/_data/ for the build-time bundle):
  - failure_catalogue.json        per-signature aggregate (counts, clusters, honesty, scope)
  - frames/*.png|jpg + *.caption.txt  annotated demo frames (up to ~12 missed_detection + up to 4 BEV)
  - h3b_expressivity.json         copied verbatim from results/ (the A-side headline)
  - oracle_status.json            the 3 oracle verdicts (the honesty-layer panel)
  The JSONs are copied into web/app/aletheon/_data/ (frames_manifest + catalogue + h3b + oracle); the
  frame images stay under web/public/data/frames/ (CDN static).

Determinism: signatures + clustering are deterministic; the detector (onnxruntime CPU) is
deterministic for a fixed image. The stride is fixed. No randomness is introduced here.

Run:  .venv/bin/python experiments/occquery_v0/build_aletheon_demo.py
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

from aletheon.adapt import ingest  # noqa: E402
from aletheon.failure import (  # noqa: E402
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
Aletheon_DATA_DIR = REPO / "web" / "app" / "aletheon" / "_data"  # the static-imported source the page bundles
RESULTS = REPO / "experiments" / "occquery_v0" / "results"

# === demo log set (honest, fixed) ================================================================
# missed_detection NEEDS cameras: 3 of the 7 camera+calibration logs (the detector is bounded by
# stride, so a 3-log subset is the honest scope; the corpus carries more camera logs).
CAMERA_LOGS = [
    "6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c",
    "201fe83b-7dd7-38f4-9d26-7b4a668638a9",
    "2c652f9e-8db8-3572-aa49-fae1344a875b",
]
# path_blocked_no_box / box_in_free are detector-free + fast (~0.5 s/log mine after ~9 s/log ingest),
# so they run over the FULL AV2 corpus: EVERY log with annotations + lidar. The list is derived at
# runtime (see `_full_lidar_corpus`) so it cannot silently drift from disk; the known-unboxed-obstacle
# log (78683234, frame 66 ~4.2 m) is part of the corpus and is sorted first so it leads the catalogue.
KNOWN_UNBOXED_LOG = "78683234-e6f1-3e4e-af52-6f839254e4c0"  # frame 66 ~4.2 m known unboxed obstacle
# Detector budget: YOLOv8n CPU measured ~0.05-0.22 s/img here (onnxruntime, warm). A stride of 3 ->
# ~52 frames/log -> ~5-10 s/log mine; 3 logs stays well within the ~10-15 min total budget. If the
# detector were slower (~3 s/img) this stride would be raised; it is recorded per-signature either way.
MISSED_DETECTION_STRIDE = 3
MAX_IMG_PX = 1100  # web-friendly downsize cap for the annotated camera frames
MAX_MISSED_FRAMES = 12  # up to ~10-12 annotated camera frames, spread across the 3 camera logs
MAX_BEV_FRAMES = 4      # render BEV for up to 4 path_blocked_no_box instances found in the full corpus


def _full_lidar_corpus() -> list[str]:
    """Every AV2 log under DATA_ROOT that has BOTH annotations.feather and a non-empty sensors/lidar/
    dir -> the full LiDAR corpus for the detector-free signatures. Derived from disk (never a frozen
    list) so the scope tracks the actual data. The known-unboxed-obstacle log is sorted first."""
    logs = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if (d / "annotations.feather").exists() and any((d / "sensors" / "lidar").glob("*.feather")):
            logs.append(d.name)
    # lead with the known-unboxed-obstacle log so it heads the catalogue/log_ids list
    if KNOWN_UNBOXED_LOG in logs:
        logs = [KNOWN_UNBOXED_LOG] + [lg for lg in logs if lg != KNOWN_UNBOXED_LOG]
    return logs


LIDAR_LOGS = _full_lidar_corpus()


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


def _aggregate_signature(name: str, logs_ir, *, stride: int, params: dict, scope_note: str):
    """Mine one signature over the given SceneIRs, cluster the hits, and build the catalogue entry.
    Returns (catalogue_entry, candidates) so the caller can render frames from the raw hits.

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
    entry = {
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
    return entry, candidates


# === annotated demo frames =======================================================================

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def _font(size: int):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _missed_detection_frames(max_frames: int = MAX_MISSED_FRAMES, per_log_max: int = 5) -> list[dict]:
    """Render annotated missed_detection frames: camera image + green detector boxes + red MISSED
    GT boxes. Picks frames with BOTH >=1 detection AND >=1 clear miss (the visceral contrast), spread
    across the 3 camera logs (up to `per_log_max` each, `max_frames` total). Within a log the picked
    frames are the strongest-contrast ones (most misses), spaced apart so they are not near-duplicates.

    Reuses the proven logic (load_av2_camera, detect_image, project_ego_boxes, match_detections
    class_agnostic, PIL draw) -- run through the same public API the engine uses."""
    from aletheon.detect import detect_image

    from probe.adapters import av2_sensor

    written: list[dict] = []
    target_labels = frozenset({"pedestrian", "bicycle", "vehicle", "motorcycle"})
    # distribute the per-log budget evenly so all 3 logs are represented even if one is miss-rich
    per_log_budget = max(1, min(per_log_max, -(-max_frames // max(1, len(CAMERA_LOGS)))))
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

        # collect ALL frames with both detections AND >=1 clear miss (strided so we don't run the
        # detector on every image), then pick the strongest, well-spaced subset for this log.
        candidates = []  # (n_missed, frame_idx, cam_ts, jpg, missed, dets)
        for i in range(0, len(scene.frames), MISSED_DETECTION_STRIDE):
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
            if missed:
                candidates.append((len(missed), i, cts, jpg, missed, dets))

        # rank by miss-count (visceral contrast), then greedily keep ones >= MIN_FRAME_GAP apart so
        # the rendered set spans the log rather than clustering on one stretch.
        MIN_FRAME_GAP = 12
        candidates.sort(key=lambda c: (-c[0], c[1]))
        picked: list[tuple] = []
        for c in candidates:
            if len(picked) >= per_log_budget or len(written) + len(picked) >= max_frames:
                break
            if all(abs(c[1] - p[1]) >= MIN_FRAME_GAP for p in picked):
                picked.append(c)
        for nmiss, i, cts, jpg, missed, dets in sorted(picked, key=lambda c: c[1]):
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


def _path_blocked_bev_frames(instances: list[tuple[str, int]], max_frames: int = MAX_BEV_FRAMES) -> list[dict]:
    """Render up to `max_frames` BEV PNGs for the given (log, frame_index) path_blocked_no_box hits
    found in the full-corpus mine. Re-ingests each distinct log ONCE (cached) so a multi-hit log does
    not pay ingest per frame. Returns a manifest entry per rendered frame (skips any that fail)."""
    out: list[dict] = []
    ir_cache: dict[str, object] = {}
    for log, frame_idx in instances[:max_frames]:
        if log not in ir_cache:
            ir_cache[log] = ingest(DATA_ROOT / log)
        entry = _path_blocked_bev_frame(log, frame_idx, ir_cache[log])
        if entry is not None:
            out.append(entry)
    return out


def _path_blocked_bev_frame(log: str, frame_idx: int, ir=None) -> dict | None:
    """Render a top-down BEV PNG of a path_blocked_no_box hit at (log, frame_idx) (no camera):
    occupancy obstacle voxels (gray), ego in-path band (light), the BLOCKED location (red),
    tracked boxes (blue) -- 'occupancy blocks here, no box explains it'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ir is None:
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
        "note": "The three oracles that gate Aletheon's failure claims. Honest: only the traversal "
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


def _path_blocked_instances(candidates) -> list[tuple[str, int]]:
    """Distinct (log_id, frame_index) pairs of path_blocked_no_box hits, nearest-block first (the
    most visceral 'something blocks the path right ahead' frames lead). De-duped per (log, frame)."""
    seen: set[tuple[str, int]] = set()
    rows: list[tuple[float, str, int]] = []
    for c in candidates:
        key = (c.log_id, c.frame_index)
        if key in seen:
            continue
        seen.add(key)
        rng = c.features.get("forward_range_m", 1e9)
        rng = rng if np.isfinite(rng) else 1e9
        rows.append((float(rng), c.log_id, c.frame_index))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))  # nearest block first, deterministic tie-break
    return [(lg, fi) for _, lg, fi in rows]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("== ingesting demo logs ==", flush=True)
    # LiDAR signatures: full-frame scan over the FULL corpus (every annotated+lidar log).
    print(f"  LiDAR corpus: {len(LIDAR_LOGS)} logs (full corpus, stride 1)", flush=True)
    lidar_irs = [ingest(DATA_ROOT / lg) for lg in LIDAR_LOGS]
    print(f"  ingested {len(lidar_irs)} LiDAR logs", flush=True)

    # missed_detection: 3 camera logs, strided to bound detector runtime.
    cam_irs_full = [ingest(DATA_ROOT / lg) for lg in CAMERA_LOGS]
    cam_irs = []
    for ir in cam_irs_full:
        sir, _ = _stride_scene_ir(ir, MISSED_DETECTION_STRIDE)
        cam_irs.append(sir)
    print(f"  ingested {len(cam_irs)} camera logs (stride {MISSED_DETECTION_STRIDE})", flush=True)

    n_lidar = len(LIDAR_LOGS)
    catalogue = {
        "schema": "aletheon_failure_catalogue/v1",
        "generated_by": "experiments/occquery_v0/build_aletheon_demo.py",
        "data_root": str(DATA_ROOT),
        "demo_scope": {
            "missed_detection": {
                "logs": CAMERA_LOGS,
                "stride": MISSED_DETECTION_STRIDE,
                "note": f"{len(CAMERA_LOGS)} AV2 camera logs (a bounded subset of the camera+calibration "
                        f"logs); frame stride {MISSED_DETECTION_STRIDE} to bound CPU detector runtime "
                        f"(YOLOv8n CPU ~0.05-0.2 s/img here).",
            },
            "path_blocked_no_box": {
                "logs": LIDAR_LOGS,
                "stride": 1,
                "note": f"FULL corpus: all {n_lidar} AV2 logs with annotations + lidar, FULL scan "
                        f"(no stride). Incl. 78683234-... with the known unboxed-obstacle at frame 66 "
                        f"(~4.2 m).",
            },
            "box_in_free": {
                "logs": LIDAR_LOGS,
                "stride": 1,
                "note": f"FULL corpus: same {n_lidar} AV2 annotated+lidar logs, FULL scan (no stride).",
            },
            "honest_caveat": f"Demo scope: the LiDAR-only signatures (path_blocked_no_box, box_in_free) "
                             f"run over the FULL AV2 corpus on disk ({n_lidar} annotated+lidar logs, "
                             f"stride 1); missed_detection runs over a BOUNDED {len(CAMERA_LOGS)}-camera-log "
                             f"subset, strided (stride {MISSED_DETECTION_STRIDE}) to cap CPU detector "
                             f"runtime, so it is NOT a full-corpus detector sweep. Per-signature honesty "
                             f"tags are verbatim from src/aletheon/failure.py.",
        },
        "signatures": {},
    }

    print("== mining missed_detection (this runs the detector; slowest) ==", flush=True)
    md_entry, _md_cands = _aggregate_signature(
        "missed_detection", cam_irs,
        stride=MISSED_DETECTION_STRIDE,
        params={"class_agnostic": True, "score_thr": 0.25, "range_bin_m": 8.0},
        scope_note=f"{len(CAMERA_LOGS)} camera logs, stride {MISSED_DETECTION_STRIDE}, "
                   f"class-agnostic match, score>=0.25.",
    )
    catalogue["signatures"]["missed_detection"] = md_entry
    print("   ", md_entry["n_candidates"], "candidates", flush=True)

    print("== mining path_blocked_no_box (full corpus) ==", flush=True)
    pb_entry, pb_cands = _aggregate_signature(
        "path_blocked_no_box", lidar_irs,
        stride=1,
        params={"horizon": 1.0, "box_radius_m": 5.0, "range_bin_m": 4.0},
        scope_note=f"{n_lidar} LiDAR logs (full corpus), full scan, horizon 1.0 s, box_radius 5 m.",
    )
    catalogue["signatures"]["path_blocked_no_box"] = pb_entry
    print("   ", pb_entry["n_candidates"], "candidates", flush=True)

    print("== mining box_in_free (full corpus) ==", flush=True)
    bf_entry, _bf_cands = _aggregate_signature(
        "box_in_free", lidar_irs,
        stride=1,
        params={"n_interior_min": 5, "range_bin_m": 8.0},
        scope_note=f"{n_lidar} LiDAR logs (full corpus), full scan, LiDAR-seen gate n_interior>=5.",
    )
    catalogue["signatures"]["box_in_free"] = bf_entry
    print("   ", bf_entry["n_candidates"], "candidates", flush=True)

    (OUT_DIR / "failure_catalogue.json").write_text(json.dumps(catalogue, indent=2, default=float))
    print("wrote failure_catalogue.json", flush=True)

    print("== rendering missed_detection frames ==", flush=True)
    frames = _missed_detection_frames(max_frames=MAX_MISSED_FRAMES)
    print(f"   {len(frames)} missed_detection frames", flush=True)

    print("== rendering path_blocked BEV frames ==", flush=True)
    pb_instances = _path_blocked_instances(pb_cands)
    print(f"   {len(pb_instances)} path_blocked instances found; rendering up to {MAX_BEV_FRAMES}", flush=True)
    bevs = _path_blocked_bev_frames(pb_instances, max_frames=MAX_BEV_FRAMES)
    frames.extend(bevs)
    print(f"   {len(bevs)} BEV frame(s) written", flush=True)

    frames_manifest = {
        "schema": "aletheon_frames_manifest/v1",
        "frames": [{k: v for k, v in fr.items() if k != "caption"} | {"caption": fr["caption"]} for fr in frames],
    }
    (OUT_DIR / "frames_manifest.json").write_text(json.dumps(frames_manifest, indent=2, default=float))
    print("wrote frames_manifest.json", flush=True)

    print("== copying h3b_expressivity.json + writing oracle_status.json ==", flush=True)
    shutil.copyfile(RESULTS / "h3b_expressivity.json", OUT_DIR / "h3b_expressivity.json")
    (OUT_DIR / "oracle_status.json").write_text(json.dumps(_oracle_status(), indent=2, default=float))
    print("wrote h3b_expressivity.json, oracle_status.json", flush=True)

    # === copy the JSONs the page statically imports (app/aletheon/_data/) ===========================
    # The page reads failure_catalogue.json + frames_manifest.json from _data/ (build-time bundle),
    # while the frame PNGs stay CDN-served under public/data/frames/. h3b/oracle JSONs are unchanged
    # by this regen but are re-copied so _data/ always mirrors the freshly written public/data/.
    print("== copying JSONs to web/app/aletheon/_data/ ==", flush=True)
    Aletheon_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("failure_catalogue.json", "frames_manifest.json", "h3b_expressivity.json", "oracle_status.json"):
        shutil.copyfile(OUT_DIR / name, Aletheon_DATA_DIR / name)
        print(f"   copied {name} -> app/aletheon/_data/", flush=True)

    print("\n== DONE ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
