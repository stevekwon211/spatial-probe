# Aletheon (spatial-probe)

**Find the scenes where the model is wrong — with a label-free correctness check.**

Aletheon is an open, local spatial-failure-search engine. Point it at your autonomous-vehicle logs,
ask "where did the model miss something," and get back the frame-intervals that broke it — each
answer tagged with *how much you can trust it*. No infra, no server, no account: `pip install`, then
`aletheon find` on your own data.

> The engine reads your sensor logs into a sensor-agnostic **Scene IR** (the source of truth), runs
> falsifiable **physical predicates** over it through a safe query language (**SceneQL**), and mines
> for failure signatures. Visualization (Rerun) is an *adapter*, never the foundation.

This is the **open core**. The thesis behind it — 3D's essence is queryable *state* (geometry,
occupancy, dynamics), not the render — and the research protocol live in
**[docs/aletheon.md](docs/aletheon.md)** and **[PLAN.md](PLAN.md)**.

## Install

```sh
# from a clone (today):
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .                  # core: numpy / scipy / pyarrow / pyyaml — no torch, no GPU
pip install -e ".[all]"           # + optional features: rerun view, mcap, model-eval find
# future, once published:        pip install aletheon
```

Core is pure CPU Python. The heavy bits (Rerun renderer, ONNX detector) are **optional extras**,
lazy-imported — `import aletheon` never pulls in torch / onnxruntime / rerun.

## Quickstart — the 5 commands

```sh
aletheon ingest <av2_log_or_occ3d>                                          # adapter -> validated Scene IR
aletheon query  "min_free_width_along_path(scene, t, 3.0) < ego_width(scene)" --data <log>   # run a predicate
aletheon find   "pedestrian missed by the model" --data <log_or_corpus>     # the wow: search for failures
aletheon export out.parquet --data <log> --openlabel out.json               # lossless, no lock-in
aletheon view   --backend rerun --data <log>                                # optional: pip install -e ".[rerun]"
```

Real output from `aletheon find` on an Argoverse 2 sensor log (8 frames, CPU YOLOv8n detector):

```text
$ aletheon find "pedestrian missed by the model" \
      --data data/danger/av2_sensor/6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c --limit-frames 8

query: "pedestrian missed by the model"
  -> signature: missed_detection  (A camera-visible GT box the 2D detector failed to output a matching detection for.)
  found 34 matching frame-intervals across 1 log(s) (8 frames scanned), in 6 cluster(s)
  mean forward range to the flagged region: 47.86 m
  top categories: pedestrian:24, vehicle:10
  dominant cluster: missed_detection@range_bin~36m_n16 (size 16, ~35.54 m forward)
  most-similar frames (FEATURE-distance, not semantic): f3(d=0.683), f4(d=0.686), f3(d=0.687), ...
  honesty: model-eval: COCO-YOLOv8n detector recall vs AV2 GT boxes; a miss = the detector failed to
  see a labeled object (Ramanagopal-style missed-detection); NOT an occupancy claim. Caveats: the
  detector is COCO-pretrained (NOT trained on AV2) so this is cross-distribution recall, not an
  in-domain benchmark ...
```

(Structured JSON follows the human summary; add `--json` for JSON only, `--render rerun` to write a
`.rrd` of the dominant cluster.) Counts are honest — zero matches is a real negative, never inflated.

## Architecture (one line)

**Scene IR is the source of truth; SceneQL queries it; Rerun is an adapter.** Five invariants are
tested: the Aletheon API never exposes Rerun types; `.rrd` is cache/export only (`rm *.rrd` →
byte-identical query); the core works with nothing optional installed; export is lossless
(content-hash round-trip); the viewer is separated from compute. Full diagram in
[docs/aletheon.md](docs/aletheon.md).

## Honesty — what the correctness check does and does NOT prove

Every `aletheon find` result is tagged by which oracle backs it. We state this verbatim, no inflation:

- **False-positive side (does the map hallucinate obstacles on the driven path?) — EXTERNAL +
  RELIABLE.** The traversal oracle uses the ego's *recorded future trajectory* (poses, not LiDAR) ⇒
  that space was physically free. `true_fp 0.0000` vs `shuffled 0.0357`, held-out free-driving. This
  is the `path_blocked_no_box` signature: the H1 occupancy-vs-box expressivity win a box-only language
  is structurally blind to.
- **Recall side (does the map miss real obstacles?) — consistency-only externally; external recall is
  honestly CLOSED on this substrate.** Box-recall is same-modality (relative gap supported, absolute is
  floor-straddle-inflated). External cross-modal recall was attempted three times and killed honestly
  (classical stereo AUC 0.259; frozen mono-depth INVALID-SCALE >9 m; free-driving INDETERMINATE). It
  needs a resource the solo/CPU build lacks: a GPU-trained AV-domain metric-depth model. Named, not
  faked.
- **`missed_detection` ("missed by the model") — model-eval, cross-distribution.** A CPU COCO-YOLOv8n
  detector's recall vs AV2 GT boxes. Real "find what breaks your model," but honestly cross-distribution
  (COCO model, not trained on AV2), coarse class map. Not an in-domain benchmark number.

"Similar scenes" is feature-distance, not semantic. Synthetic runs are smoke tests, never evidence.
The sealed protocol and result framing are in
[experiments/occquery_v0/preregistration.md](experiments/occquery_v0/preregistration.md) and
[docs/aletheon.md](docs/aletheon.md).

## Layout

- `src/aletheon/` — the open-core engine (Scene IR, SceneQL, predicates, adapters, failure-mining, CLI)
- `src/probe/` — the underlying instrument (DDA raycast, occupancy grid, C-space predicates, metrics)
- `experiments/occquery_v0/` — the first falsifiable experiment (H1 expressivity holds; see prereg)
- `docs/` — [aletheon.md](docs/aletheon.md) (engine + honesty), [benchmark-anchors.md](docs/benchmark-anchors.md) (success/kill criteria)
- `tests/` — 202 unit/integration tests (pure numpy/scipy, no torch, no data needed)
- `data/` — datasets (gitignored; never committed)
- `web/` — build-in-public site (Next.js, deployed on Vercel); reads experiment results

## Dependencies

| Tier | Install | What it adds |
|---|---|---|
| **core** | `pip install -e .` | numpy, scipy, pyarrow, pyyaml — the full `aletheon` engine (CPU, pure Python) |
| `[rerun]` | `pip install -e ".[rerun]"` | `aletheon view` Rerun `.rrd` renderer |
| `[mcap]` | `pip install -e ".[mcap]"` | MCAP log-container adapter |
| `[detect]` | `pip install -e ".[detect]"` | onnxruntime + pillow — `aletheon find "... missed by the model"` (YOLOv8n model-eval) |
| `[all]` | `pip install -e ".[all]"` | rerun + mcap + detect together |
| `[data]` | `pip install -e ".[data]"` | nuscenes-devkit — Occ3D-nuScenes research substrate (not a tool dependency) |
| `[dev]` | `pip install -e ".[dev]"` | pytest |

`torch` and the experiment-only frozen-depth oracle are **not** package dependencies — they live in
`experiments/`, off the install path.

## Develop / test

```sh
pip install -e ".[dev]"
python -m pytest                  # 202 tests, ~CPU-only, no dataset needed
```

## License

Apache-2.0 © 2026 Doeon Kwon. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
