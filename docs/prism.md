# PRISM+α — an open, verifiable, physics-aware spatial-failure-search engine

PRISM is an independent **Spatial Semantic Engine**: a sensor-agnostic **Scene IR** (source of truth) with
a safe query language (**SceneQL**), physical predicates, lossless import/export adapters, and a
**failure-mining** layer whose answers come with a **label-free correctness check**. Visualization backends
(Rerun, Foxglove, web) are *adapters*, never the foundation.

One-line strategy: **Rerun-like open-core strategy, NOT a Rerun-like product scope.** Rerun's wow is "see
all your multimodal data in a few lines." PRISM's wow is **"find the scenes where the map is wrong in a few
lines — and know how much to trust the answer."**

## Install + the 5 commands (the "one thing")
```sh
pip install -e .            # installs the `prism` console script (pure numpy/scipy/pyarrow core)
prism ingest <av2_log_or_occ3d>          # adapter -> validated Scene IR
prism query  "min_free_width_along_path(scene, t, 3.0) < ego_width(scene)" --data <log>
prism find   "path blocked but no tracked object explains it" --data <log_or_corpus>   # the wow
prism export out.parquet --data <log> --openlabel out.json    # lossless, no lock-in
prism view --backend rerun --data <log>  # optional: pip install -e ".[rerun]"
```

## Architecture (layers; each backend is one adapter)
```
prism find / query / ingest / export / view        (CLI + Python API — PRISM types only)
        │
SceneQL  (probe.query_dsl safe_eval, AST-whitelist; no eval)   +  physical predicates
        │
Scene IR  (SOURCE OF TRUTH): Entity/Track/CoordinateFrame/Observation/Relation/
          Event/Prediction/GroundTruth/Failure/DatasetSlice/Provenance   + validate + lossless serialize
        │
Adapters (SceneReader/SceneWriter):  AV2 · Occ3D · Parquet · MCAP · Rerun(.rrd)
```
**5 enforced invariants** (tested): (1) the PRISM API never exposes Rerun types; (2) Scene IR is the source
of truth, `.rrd` is cache/export only (`rm *.rrd` → byte-identical query); (3) the core works with no Rerun
installed (`import prism` leaves rerun/mcap out of `sys.modules`); (4) lossless Parquet/OpenLABEL export
always (content-hash round-trip); (5) viewer separated from compute.

## The honesty layer — what makes this defensible (and its honest ceiling)
PRISM's differentiator is that a failure verdict comes with an **independent, label-free correctness check**
(the "oracle"). The check has two halves, and we are precise about which holds:

| half | question | status | how |
|---|---|---|---|
| **false-positive** | does the map HALLUCINATE obstacles on the driven path? | **EXTERNAL + RELIABLE** | traversal-v0.1 oracle: ego future trajectory (recorded poses, not LiDAR) ⇒ that space was physically free. true_fp 0.0000 vs shuffled 0.0357, held-out free-driving. |
| **recall** | does the map MISS real obstacles? | **consistency-only externally; external route honestly closed on this substrate** | box-recall (same-modality, RECALL-SUPPORTED relative, absolute is floor-straddle-inflated). External cross-modal recall was attempted twice and KILLED honestly: classical stereo (AUC 0.259, too sparse) and frozen DAv2 mono-depth (INVALID-SCALE, >9 m — VKITTI depth not metrically self-consistent on AV2). Deferred to a GPU/free-driving path. |

So `prism find` tags every result by which oracle backs it: `path_blocked_no_box` is **external-fp**
(traversal-RELIABLE ⇒ a block with no box is most likely a real unboxed obstacle a box-only language is
structurally blind to — the H1 expressivity win); `box_in_free` is **consistency-only** (and its absolute
count is inflated by the `_ROAD_Z` ground filter — only the relative gap survives); `missed_detection` is
**model-eval** (a CPU COCO-YOLOv8n detector's recall vs AV2 GT — `prism find "pedestrian missed by the
model"` returns the scenes the detector failed on, the real "find what breaks your model" wow; honestly
cross-distribution, not an in-domain number). "Similar scenes" is feature-distance, not semantic. `ttc`
ships as a primitive flagged with the dynfield pre-registered NEGATIVE.

## What is real today vs deferred (no inflation)
- **Real:** the installable engine (Scene IR + SceneQL + predicates + adapters + lossless export, 202 tests
  green); the FP-side external oracle (RELIABLE); `prism find` over real AV2 returning the H1 unboxed-
  obstacle win + the consistency recall signal + **model-eval missed-detections (CPU YOLOv8n detector
  recall vs GT)**, each honesty-tagged; a real Rerun `.rrd` render.
- **External recall — triangulated CLOSED for solo/CPU (3 pre-registered attempts, 3 honest negatives):**
  classical stereo on the following substrate (AUC 0.259, textureless dark backs); frozen DAv2 mono-depth
  (INVALID-SCALE >9 m, VKITTI not metrically self-consistent on AV2, even ground-plane-rescaled); classical
  stereo on free-driving (INDETERMINATE-VACUOUS — the clean path has no in-path obstacles to grade). The
  fix needs a resource the solo/CPU build lacks: a trained AV-domain metric-depth model (GPU) or a curated
  textured-AND-obstacle-dense substrate. Recall therefore ships consistency-only externally; named, not faked.
- **Deferred (needs GPU/users):** cross-sensor auto-structuring + the data-flywheel Hub. Funded-Visionary
  territory.

The defensible solo wedge: **an open, verifiable, physics-aware spatial-failure-search layer whose answers
carry a label-free correctness check — half of which (the FP side) is already external and RELIABLE.**
