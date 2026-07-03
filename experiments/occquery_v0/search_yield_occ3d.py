# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Corpus-scale physical-quantity SEARCH YIELD on Occ3D-nuScenes + tri-state honesty tags
(realizes the SEALED pre-reg `search_yield_occ3d_preregistration.md`; run AFTER the seal, nothing
here was chosen after seeing a corpus number).

Two arms per scene (both `unknown_policy=FREE`, sealed = L1):
  * DENSE   -- `load_scene(name, data, mask='none', with_boxes=True)`: the labeled corpus a data
               engine would search. All 20 sealed queries (queries.yaml) run per frame; a scene is
               RETRIEVED iff any frame matches (all sealed queries are scope `any`). C1 = per-query
               scene-yield table + measured-value distributions.
  * OBSERVED -- `load_scene(name, data, mask='lidar')`: single-sweep visibility. The 10 sealed
               free_path/corridor queries get a per-(frame, query) tri-state tag:
               CONFIRMED_HIT (hit; under FREE only observed occupied voxels can cause it),
               EXONERATED (no hit, band unknown-fraction <= eps), UNRESOLVED (no hit, > eps).
               C2 = tag informativeness (see the sealed criteria).

Queries are evaluated by the SEALED evaluator (`probe.retrieval.namespace` bindings via
`probe.query_dsl.safe_eval`) behind a memoization layer; `tests/test_search_yield.py` proves the
memoized evaluation equals `retrieval.frame_true` (semantics-preservation gate). The measurement
cache doubles as the physical-quantity index (dumped to a gitignored parquet — product artifact,
not a claim).

Honest scope: dense-arm yields DESCRIBE Occ3D's auto-labeled corpus (consistency class, not world
truth); tag validity vs dense GT is NOT checkable here (Occ3D obstacle-visibility degeneracy,
sealed 6475448) and rests on the sealed synthetic mechanism test (5844786).

Run:  python experiments/occquery_v0/search_yield_occ3d.py            # full sealed run
      python experiments/occquery_v0/search_yield_occ3d.py --limit 5  # smoke only, never reported
      python experiments/occquery_v0/search_yield_occ3d.py --report-only
Per-scene results checkpoint to results/search_yield_occ3d_scenes.jsonl (append; restart skips
completed scenes).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))  # script-mode import of probe.*

from probe import retrieval  # noqa: E402
from probe.adapters.occ3d import _annotations  # noqa: E402
from probe.adapters.occ3d import load_scene  # noqa: E402
from probe.grid import UNKNOWN, EgoPose, OccupancyGrid, UnknownPolicy  # noqa: E402
from probe.query_dsl import safe_eval  # noqa: E402
from probe.query_spec import Query, load_queries  # noqa: E402

_DATA = _REPO / "data"
_QUERIES = _HERE / "queries.yaml"

# --- SEALED constants (search_yield_occ3d_preregistration.md) — do not tune post hoc -----------
_BAND_MARGIN = 1.0            # widest sealed lateral tier (= L1)
_DEV_FRACTION = 0.20          # first 20% by sorted scene id = dev (hygiene, = L1)
_N_BOOT = 1000
_SEED = 0
_EPS_VERDICT = 0.05           # tag eps for the C2 verdict
_EPS_TIERS = (0.01, 0.05, 0.10, 0.20)   # reported as a curve
_LONGTAIL_HI = 0.20           # long-tail yield band = (0, 0.20]
_C1_HOLD_MIN = 10             # >= 10/20 occupancy queries in the band -> C1 holds (Amendment 1)
_C1_EMPTY_MIN = 16            # >= 16/20 yield exactly 0               -> C1 killed (empty)
_C1_SAT_MIN = 10              # >= 10/20 yield > 0.50                  -> C1 killed (saturated)
_C1_SAT_YIELD = 0.50
_C2_UNRESOLVED_MIN = 0.10     # holds: UNRESOLVED >= 10% of non-hits at eps=0.05 ...
_C2_EXONERATED_MIN = 0.01     # ... AND EXONERATED >= 1% of non-hits
_C2_DEGENERATE = 0.01         # kills: UNRESOLVED < 1%; or EXONERATED < 1% at EVERY eps tier

_NUMERIC_MEASUREMENTS = frozenset({
    "lateral_clearance", "centerline_lateral_distance", "min_free_width_along_path",
    "distance_to_nearest_object", "ego_speed",
})


# --------------------------------------------------------------------------------------------------
# Sealed-evaluator memoization (semantics proven equal to retrieval.frame_true in tests).
# --------------------------------------------------------------------------------------------------
def memoized_namespace(scene, policy: UnknownPolicy) -> tuple[dict, dict]:
    """The sealed `retrieval.namespace(scene, policy)` with every predicate call cached per
    (name, args, kwargs). The predicates are pure functions of (scene, t, knobs), so caching cannot
    change a verdict — enforced by the equality test, not assumed."""
    base = retrieval.namespace(scene, policy)
    cache: dict = {}
    ns: dict = {"scene": scene}

    def _wrap(name: str, fn):
        def g(sc, *args, **kwargs):
            key = (name, args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = fn(sc, *args, **kwargs)
            return cache[key]
        return g

    for name, fn in base.items():
        if name != "scene":
            ns[name] = _wrap(name, fn)
    return ns, cache


def horizon_of(query: Query) -> float:
    """The forward horizon a sealed free_path/corridor predicate reads (fails loudly if absent —
    never a silent default)."""
    m = re.search(r"horizon=([0-9.]+)", query.predicate or "")
    if not m:
        raise ValueError(f"query {query.id}: no horizon= in predicate {query.predicate!r}")
    return float(m.group(1))


# --------------------------------------------------------------------------------------------------
# Honesty tag (Leg 2). Band = the L1 in-path band at the query's horizon, volumetric.
# --------------------------------------------------------------------------------------------------
def band_unknown_fraction(grid: OccupancyGrid, ego: EgoPose, horizon: float,
                          band_margin: float = _BAND_MARGIN) -> float:
    """Fraction of UNKNOWN voxels in the in-path band volume: forward 0..(length/2 + speed*h),
    |lateral| <= width/2 + band_margin, ground < z <= ground + ego.height. NaN if the band holds no
    voxels (explicit failure, never a silent 0)."""
    occ = grid.occupancy
    origin = np.asarray(grid.origin, dtype=float)
    res = grid.voxel_size
    zc = origin[2] + np.arange(occ.shape[2]) * res
    zsel = (zc > grid.ground_height) & (zc <= grid.ground_height + ego.height)
    if not zsel.any():
        return float("nan")
    xs = origin[0] + np.arange(occ.shape[0]) * res
    ys = origin[1] + np.arange(occ.shape[1]) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    fwd, lat = ego.to_ego_frame(np.column_stack([gx.ravel(), gy.ravel()]))
    reach = ego.length / 2.0 + ego.speed * horizon
    half = ego.width / 2.0 + band_margin
    inband = (fwd >= 0.0) & (fwd <= reach) & (np.abs(lat) <= half)
    if not inband.any():
        return float("nan")
    cells = occ[:, :, zsel].reshape(-1, int(np.count_nonzero(zsel)))[inband]
    return float(np.count_nonzero(cells == UNKNOWN) / cells.size)


def tag_decision(*, hit: bool, unknown_fraction: float, eps: float) -> str:
    """The sealed tri-state honesty tag for one (frame, query) search decision."""
    if hit:
        return "CONFIRMED_HIT"
    return "EXONERATED" if unknown_fraction <= eps else "UNRESOLVED"


# --------------------------------------------------------------------------------------------------
# Yield statistics (scene-clustered bootstrap, pure numpy — the L1 pattern).
# --------------------------------------------------------------------------------------------------
def yield_ci(indicators: list[int], rng: np.random.Generator, n_boot: int = _N_BOOT) -> dict:
    arr = np.asarray(indicators, dtype=float)
    point = float(arr.mean()) if arr.size else float("nan")
    if arr.size < 2:
        return {"mean": point, "lo": point, "hi": point, "n_scenes": int(arr.size)}
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return {"mean": point, "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n_scenes": int(arr.size)}


def _rate_ci(per_scene: list[tuple[int, int]], rng: np.random.Generator,
             n_boot: int = _N_BOOT) -> dict:
    """Pooled-rate CI over scenes: per_scene = (numerator, denominator) pairs; resample scenes,
    pool, recompute numerator/denominator."""
    arr = np.asarray(per_scene, dtype=float)
    num, den = arr.sum(axis=0)
    point = num / den if den else float("nan")
    if arr.shape[0] < 2 or not den:
        return {"mean": point, "lo": float("nan"), "hi": float("nan"), "n_scenes": arr.shape[0]}
    samples = []
    for _ in range(n_boot):
        n, d = arr[rng.integers(0, arr.shape[0], size=arr.shape[0])].sum(axis=0)
        if d:
            samples.append(n / d)
    lo, hi = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))) \
        if samples else (float("nan"), float("nan"))
    return {"mean": float(point), "lo": lo, "hi": hi, "n_scenes": int(arr.shape[0])}


# --------------------------------------------------------------------------------------------------
# Per-scene work
# --------------------------------------------------------------------------------------------------
def _round(v: float) -> float | None:
    return round(float(v), 4) if math.isfinite(v) else None


def _measurement_label(key: tuple) -> str | None:
    name, _args, kwargs = key
    if name not in _NUMERIC_MEASUREMENTS:
        return None
    suffix = "".join(f"[{k}={v}]" for k, v in kwargs)
    return name + suffix


def process_scene(name: str, queries: list[Query], tag_queries: list[Query]) -> dict:
    policy = UnknownPolicy.FREE  # sealed
    rec: dict = {"scene": name, "dense": {}, "observed": {}, "measurements": {}}

    # DENSE arm — C1 yield + measurement index
    dense = load_scene(name, _DATA, mask="none", with_boxes=True)
    ns, cache = memoized_namespace(dense, policy)
    rec["n_frames"] = len(dense.frames)
    for q in queries:
        hits = 0
        for t in dense.times():
            ns["t"] = t
            if bool(safe_eval(q.predicate, ns)):
                hits += 1
        rec["dense"][q.id] = {"match": hits > 0, "n_frame_hits": hits}
    per_frame: dict[str, dict[int, float | None]] = {}
    for key, val in cache.items():
        label = _measurement_label(key)
        if label is None or not key[1]:  # ego_width has no t; ego_speed etc. have args=(t,)
            continue
        per_frame.setdefault(label, {})[key[1][0]] = _round(float(val))
    rec["measurements"] = {
        label: [vals.get(t) for t in range(len(dense.frames))] for label, vals in per_frame.items()
    }

    # OBSERVED arm — C2 tags (no boxes: the 10 tag queries are occupancy-only)
    observed = load_scene(name, _DATA, mask="lidar", with_boxes=False)
    ns2, _ = memoized_namespace(observed, policy)
    uf_cache: dict[tuple[int, float], float] = {}
    for q in tag_queries:
        h = horizon_of(q)
        n_hits = 0
        nonhit_uf: list[float] = []
        for t in observed.times():
            ns2["t"] = t
            hit = bool(safe_eval(q.predicate, ns2))
            if hit:
                n_hits += 1
                continue
            if (t, h) not in uf_cache:
                uf_cache[(t, h)] = band_unknown_fraction(observed.grid_at(t), observed.ego_at(t), h)
            nonhit_uf.append(round(uf_cache[(t, h)], 4))
        rec["observed"][q.id] = {"n_frames": len(observed.frames), "n_hits": n_hits,
                                 "nonhit_uf": nonhit_uf}
    return rec


# --------------------------------------------------------------------------------------------------
# Report (from the JSONL checkpoint — pure aggregation, no re-loading)
# --------------------------------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def build_report(records: list[dict], queries: list[Query], tag_queries: list[Query]) -> dict:
    rng = np.random.default_rng(_SEED)
    loaded = sorted(r["scene"] for r in records)
    n_dev = int(round(_DEV_FRACTION * len(loaded)))
    dev_ids = set(loaded[:n_dev])
    head = [r for r in records if r["scene"] not in dev_ids]
    dev = [r for r in records if r["scene"] in dev_ids]

    def _yields(group: list[dict]) -> dict:
        out = {}
        for q in queries:
            ind = [1 if r["dense"][q.id]["match"] else 0 for r in group]
            frame_hits = sum(r["dense"][q.id]["n_frame_hits"] for r in group)
            frames = sum(r["n_frames"] for r in group)
            out[q.id] = {
                "backend": q.backend, "is_occupancy": q.is_occupancy,
                "refav_expressible": q.refav_expressible,
                "scene_yield": yield_ci(ind, rng),
                "n_scenes_retrieved": int(sum(ind)),
                "frame_hit_rate": (frame_hits / frames) if frames else float("nan"),
                "n_frame_hits": int(frame_hits),
            }
        return out

    head_yields = _yields(head)
    occupancy_ids = [q.id for q in queries if q.is_occupancy]

    def _band(y: float) -> str:
        if y == 0.0:
            return "zero"
        if y <= _LONGTAIL_HI:
            return "long_tail"
        if y > _C1_SAT_YIELD:
            return "saturated"
        return "mid"

    bands = {qid: _band(head_yields[qid]["scene_yield"]["mean"]) for qid in occupancy_ids}
    n_longtail = sum(1 for b in bands.values() if b == "long_tail")
    n_zero = sum(1 for b in bands.values() if b == "zero")
    n_sat = sum(1 for b in bands.values() if b == "saturated")
    c1 = {
        "n_occupancy_queries": len(occupancy_ids),
        "n_long_tail": n_longtail, "n_zero_yield": n_zero, "n_saturated": n_sat,
        "holds": n_longtail >= _C1_HOLD_MIN,
        "killed_empty": n_zero >= _C1_EMPTY_MIN,
        "killed_saturated": n_sat >= _C1_SAT_MIN,
        "query_band": bands,
    }

    # C2 — pooled tag rates over the 10 tag queries, per eps tier, on the headline split.
    def _tag_counts(group: list[dict], eps: float) -> tuple[dict, list[tuple[int, int]], list[tuple[int, int]]]:
        pooled = {"CONFIRMED_HIT": 0, "EXONERATED": 0, "UNRESOLVED": 0}
        unres_pairs, exon_pairs = [], []  # per-scene (numerator, n_nonhit)
        for r in group:
            n_unres = n_exon = n_hit = 0
            for q in tag_queries:
                o = r["observed"][q.id]
                n_hit += o["n_hits"]
                for uf in o["nonhit_uf"]:
                    if uf <= eps:
                        n_exon += 1
                    else:
                        n_unres += 1
            pooled["CONFIRMED_HIT"] += n_hit
            pooled["EXONERATED"] += n_exon
            pooled["UNRESOLVED"] += n_unres
            nonhit = n_exon + n_unres
            unres_pairs.append((n_unres, nonhit))
            exon_pairs.append((n_exon, nonhit))
        return pooled, unres_pairs, exon_pairs

    eps_curve = {}
    for eps in _EPS_TIERS:
        pooled, unres_pairs, exon_pairs = _tag_counts(head, eps)
        nonhit = pooled["EXONERATED"] + pooled["UNRESOLVED"]
        eps_curve[str(eps)] = {
            "counts": pooled,
            "unresolved_rate_of_nonhits": _rate_ci(unres_pairs, rng),
            "exonerated_rate_of_nonhits": _rate_ci(exon_pairs, rng),
            "n_nonhit_decisions": int(nonhit),
        }
    v = eps_curve[str(_EPS_VERDICT)]
    unres = v["unresolved_rate_of_nonhits"]["mean"]
    exon = v["exonerated_rate_of_nonhits"]["mean"]
    exon_all_tiers = [eps_curve[str(e)]["exonerated_rate_of_nonhits"]["mean"] for e in _EPS_TIERS]
    c2 = {
        "eps_verdict": _EPS_VERDICT,
        "unresolved_rate": unres, "exonerated_rate": exon,
        "holds": (not math.isnan(unres)) and unres >= _C2_UNRESOLVED_MIN
                 and (not math.isnan(exon)) and exon >= _C2_EXONERATED_MIN,
        "killed_tag_unnecessary": (not math.isnan(unres)) and unres < _C2_DEGENERATE,
        "killed_exoneration_impossible": all(
            (not math.isnan(x)) and x < _C2_DEGENERATE for x in exon_all_tiers),
    }

    # measurement distributions (pooled headline, finite values only)
    dists = {}
    pool: dict[str, list[float]] = {}
    for r in head:
        for label, vals in r["measurements"].items():
            pool.setdefault(label, []).extend(v for v in vals if v is not None)
    for label, vals in sorted(pool.items()):
        if not vals:
            continue
        a = np.asarray(vals, dtype=float)
        dists[label] = {"n_finite": int(a.size),
                        "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
                        "p95": float(np.percentile(a, 95))}

    return {
        "experiment": "occquery_v0 / search yield + honesty tags (Occ3D-nuScenes corpus)",
        "preregistration": "search_yield_occ3d_preregistration.md (SEALED, commit 6f17a1c)",
        "result_class": ("corpus DESCRIPTION / consistency (dense arm = Occ3D auto-label; "
                         "observed arm = single-sweep visibility) — NOT external world-truth. "
                         "Tag validity vs dense GT NOT checkable here (Occ3D obstacle-visibility "
                         "degeneracy); mechanism validity = sealed synthetic test 5844786."),
        "commit": _git_commit(), "seed": _SEED,
        "split": {"dev_fraction": _DEV_FRACTION, "n_dev": len(dev), "n_headline": len(head)},
        "headline_yields": head_yields,
        "dev_yields": _yields(dev) if dev else None,
        "c1_verdict": c1,
        "c2_verdict": c2,
        "c2_eps_curve": eps_curve,
        "measurement_distributions": dists,
        "box_expressibility_restated": ("20/20 occupancy queries refav_expressible=false (sealed "
                                        "queries.yaml, verified vs RefAV's 32-function set) — "
                                        "restated, not recomputed."),
    }


def _fmt_ci(d: dict) -> str:
    return f"{d['mean']:.4f} CI[{d['lo']:.4f}, {d['hi']:.4f}]"


def write_summary(path: pathlib.Path, r: dict, queries: list[Query]) -> None:
    c1, c2 = r["c1_verdict"], r["c2_verdict"]
    L = ["# Search yield + honesty tags — Occ3D-nuScenes corpus\n"]
    L.append(f"- Pre-reg: `{r['preregistration']}`")
    L.append(f"- Commit: `{r['commit']}`  seed {r['seed']}")
    L.append(f"- Result class: {r['result_class']}")
    L.append(f"- Split: headline {r['split']['n_headline']} scenes / dev {r['split']['n_dev']}\n")

    verdict_bits = []
    verdict_bits.append("C1 HOLDS" if c1["holds"] else
                        ("C1 KILLED (empty)" if c1["killed_empty"] else
                         ("C1 KILLED (saturated)" if c1["killed_saturated"] else "C1 NO CLAIM (mixed)")))
    verdict_bits.append("C2 HOLDS" if c2["holds"] else
                        ("C2 KILLED (tag unnecessary)" if c2["killed_tag_unnecessary"] else
                         ("C2 KILLED (exoneration impossible)" if c2["killed_exoneration_impossible"]
                          else "C2 NO CLAIM (mixed)")))
    L.append(f"## Verdicts (per the sealed criteria): {' / '.join(verdict_bits)}\n")

    L.append(f"## C1 — scene yield (headline split; long-tail band = (0, {_LONGTAIL_HI:.0%}])")
    n_occ = c1["n_occupancy_queries"]
    L.append(f"{c1['n_long_tail']}/{n_occ} long-tail, {c1['n_zero_yield']}/{n_occ} zero, "
             f"{c1['n_saturated']}/{n_occ} saturated (>{_C1_SAT_YIELD:.0%}).\n")
    L.append("| query | backend | scene yield | scenes | frame hit rate | band |")
    L.append("|---|---|---|---|---|---|")
    for q in queries:
        y = r["headline_yields"][q.id]
        band = c1["query_band"].get(q.id, "baseline")
        L.append(f"| {q.id} | {q.backend} | {_fmt_ci(y['scene_yield'])} | "
                 f"{y['n_scenes_retrieved']} | {y['frame_hit_rate']:.4f} | {band} |")
    L.append("")
    L.append("## C2 — honesty tags (observed arm, 10 free_path/corridor queries)")
    L.append(f"At sealed eps={c2['eps_verdict']}: UNRESOLVED {c2['unresolved_rate']:.4f} of non-hits, "
             f"EXONERATED {c2['exonerated_rate']:.4f} of non-hits.\n")
    L.append("| eps | CONFIRMED_HIT | EXONERATED | UNRESOLVED | unresolved rate | exonerated rate |")
    L.append("|---|---|---|---|---|---|")
    for eps, row in r["c2_eps_curve"].items():
        cts = row["counts"]
        L.append(f"| {eps} | {cts['CONFIRMED_HIT']} | {cts['EXONERATED']} | {cts['UNRESOLVED']} | "
                 f"{_fmt_ci(row['unresolved_rate_of_nonhits'])} | "
                 f"{_fmt_ci(row['exonerated_rate_of_nonhits'])} |")
    L.append("")
    L.append("## Physical-quantity distributions (headline, finite values)")
    L.append("| measurement | n | P5 | P50 | P95 |")
    L.append("|---|---|---|---|---|")
    for label, d in r["measurement_distributions"].items():
        L.append(f"| {label} | {d['n_finite']} | {d['p5']:.3f} | {d['p50']:.3f} | {d['p95']:.3f} |")
    L.append("")
    L.append(f"- {r['box_expressibility_restated']}")
    path.write_text("\n".join(L) + "\n")


def _write_measurement_parquet(records: list[dict], path: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    scenes, frames, labels, values = [], [], [], []
    for r in records:
        for label, vals in r["measurements"].items():
            for t, v in enumerate(vals):
                if v is None:
                    continue
                scenes.append(r["scene"]); frames.append(t); labels.append(label); values.append(v)
    pq.write_table(pa.table({"scene": scenes, "frame": frames,
                             "measurement": labels, "value": values}), path)


# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = ALL scenes (the sealed run)")
    ap.add_argument("--report-only", action="store_true", help="aggregate the existing checkpoint")
    args = ap.parse_args()

    queries = load_queries(_QUERIES)
    occupancy = [q for q in queries if q.is_occupancy]
    if len(queries) != 24 or len(occupancy) != 20:
        raise SystemExit(f"sealed query set drifted: {len(queries)} queries / {len(occupancy)} occupancy")
    tag_queries = [q for q in occupancy
                   if "free_along_ego_path" in (q.predicate or "")
                   or "min_free_width_along_path" in (q.predicate or "")]
    if len(tag_queries) != 10:
        raise SystemExit(f"expected the 10 sealed free_path/corridor queries, got {len(tag_queries)}")

    out_dir = _HERE / "results"
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "search_yield_occ3d_scenes.jsonl"
    done: dict[str, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["scene"]] = rec

    if not args.report_only:
        all_scenes = sorted(_annotations(_DATA)["scene_infos"].keys())
        if args.limit:
            all_scenes = all_scenes[: args.limit]
        todo = [s for s in all_scenes if s not in done]
        print(f"search_yield: {len(all_scenes)} scenes ({len(done)} checkpointed, {len(todo)} to run)",
              flush=True)
        t0 = time.time()
        with ckpt.open("a") as fh:
            for i, name in enumerate(todo):
                try:
                    rec = process_scene(name, queries, tag_queries)
                except Exception as e:  # noqa: BLE001 - a missing/corrupt scene is skipped, reported
                    print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
                    continue
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[name] = rec
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    rate = (time.time() - t0) / (i + 1)
                    print(f"  {i + 1}/{len(todo)} scenes  ({rate:.1f}s/scene, "
                          f"~{rate * (len(todo) - i - 1) / 60:.0f} min left)", flush=True)

    records = list(done.values())
    if not records:
        raise SystemExit("no scenes processed")
    if args.limit and not args.report_only:
        print("\n--limit smoke run: no report written (never reported, per the pre-reg)")
        return

    report = build_report(records, queries, tag_queries)
    (out_dir / "search_yield_occ3d.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(out_dir / "search_yield_occ3d_summary.md", report, queries)
    try:
        _write_measurement_parquet(records, out_dir / "search_yield_occ3d_measurements.parquet")
    except ImportError:
        print("pyarrow unavailable — measurement parquet skipped (index only, not a claim)")
    print(f"\nwrote {out_dir / 'search_yield_occ3d.json'}")
    print(f"wrote {out_dir / 'search_yield_occ3d_summary.md'}")
    c1, c2 = report["c1_verdict"], report["c2_verdict"]
    print(f"\nC1: holds={c1['holds']} killed_empty={c1['killed_empty']} "
          f"killed_saturated={c1['killed_saturated']} "
          f"(long-tail {c1['n_long_tail']}/{c1['n_occupancy_queries']}, "
          f"zero {c1['n_zero_yield']}/{c1['n_occupancy_queries']}, "
          f"sat {c1['n_saturated']}/{c1['n_occupancy_queries']})")
    print(f"C2: holds={c2['holds']} killed_tag_unnecessary={c2['killed_tag_unnecessary']} "
          f"killed_exoneration_impossible={c2['killed_exoneration_impossible']} "
          f"(unresolved {c2['unresolved_rate']:.4f}, exonerated {c2['exonerated_rate']:.4f})")


if __name__ == "__main__":
    main()
