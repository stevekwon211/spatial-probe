# spatial-probe — PLAN

_Status: DRAFT for review. No experiment logic is implemented yet — this plan + the
scaffold are the review checkpoint. Implementation (M1+) starts after sign-off._

> Provisional name. `spatial-probe` = an instrument that *probes* what a spatial
> representation actually stores (occupancy / geometry / dynamics) and whether that
> stored state is queryable and trustworthy enough to act on. Easy to rename
> (directory + `pyproject` name only).

---

## 0. North star

One thesis: **3D's essence is queryable/updatable STATE, not the render.** One method:
take a *falsifiable physical predicate* (e.g. "is X visible from here", "is there
clearance"), run it as a TEST, and measure whether a representation stores the signal
needed to answer it — always paired with a **fairness control** (an easy query the
"weaker" representation *should* win) so the instrument is not rigged.

`spatial-probe` is the reusable instrument (`src/probe/`). Each research question is an
experiment on top (`experiments/<name>/`). The instrument is the durable, public asset;
it grows by accretion (extract shared code into `core` only on the *second* use — do not
big-design it up front).

The six-paper program this repo is meant to hold (do NOT build all at once):

| axis | experiment | one-line |
|---|---|---|
| geometry/occupancy | **occquery** (FIRST) | occupancy-native predicates retrieve scenes box-only languages can't express |
| render→state | asof | does GS/NeRF→occupancy conversion preserve action-relevant signal? |
| visibility | gt-distrust | occlusion geometry predicts which occupancy GT labels are untrustworthy |
| uncertainty | vis-calibration | is occupancy confidence honest where the sensor can't see? |
| dynamics/time | dynfield | which stored dynamics fields a planner actually needs, by regime |
| valuation | value-of-correction | which label fixes actually move the model |

Program order (by external-anchor strength; rationale + per-topic success criteria in
[`docs/benchmark-anchors.md`](docs/benchmark-anchors.md)):
**occquery → dynfield → value-of-correction → asof → gt-distrust → vis-calibration.** Only occquery
rides a live third-party leaderboard (RefAV / EvalAI HOTA-Temporal); the other five define a new axis
on a public substrate, so their bar is "public data + a measurable gap vs a public baseline + a
released benchmark", not a leaderboard win.

---

## 1. Why OccQuery first (sequencing)

- **It is where the voxel/GS edge is load-bearing, not incidental.** The predicate
  executor over an occupancy grid *is* the voxel engine's job. (Honest: for gt-distrust /
  calibration the same DDA helps but isn't a moat — anyone can raycast a grid.)
- **It builds the shared `core` that gt-distrust and calibration reuse** (the
  `raycast` / line-of-sight primitive + the Occ3D adapter + metrics). Starting here is
  the most efficient first move for the whole program.
- **It plugs straight into PRISM's product** (the physical-quantity search layer) and is
  demoable: "queries box-only mining can't express, run on the occupancy asset PRISM
  already builds, with reproducible/denotation-correct answers (no VLM hallucination)."
- **Cheapest honest path to a falsifiable signal.** v0 runs on *GT* occupancy with
  mostly-CPU geometric ops; no training, no GPU.

Calibrated scope: this is **not** "nobody can do this." Scenario mining on geometric
predicates is standard AV practice (done on object tracks). The defensible sliver is
*occupancy-native* predicates (free-space / clearance / reachable-region that object
boxes can't express) **plus a denotation-correctness benchmark**. Lead with that; don't
overclaim a "first predicate algebra" (Scenic 3.0 owns typed geo predicates; RefAV owns
box-based scenario mining). See §6 prior-work.

---

## 2. Repo architecture

```
spatial-probe/
  PLAN.md                       # this file
  README.md
  pyproject.toml                # package `probe`, deps pinned minimal
  .gitignore                    # data/, checkpoints, caches
  src/probe/                    # THE instrument (durable, public)
    raycast.py                  # core primitive: DDA line-of-sight over a voxel grid
    grid.py                     # OccupancyGrid: load/index, world<->voxel, occ/free/unknown
    predicates/                 # falsifiable physical predicates (grow over time)
      clearance.py              # lateral_clearance
      freepath.py               # free_along_ego_path
    adapters/
      occ3d.py                  # Occ3D-nuScenes loader (occupancy GT + visibility mask + ego pose)
    oracle.py                   # ground-truth predicate values from GT occupancy
    metrics.py                  # denotation P/R/F1, tolerance-accuracy, MAE, expressibility coverage
  experiments/
    occquery_v0/
      README.md                 # the falsifiable protocol (success / kill thresholds)
      queries.yaml              # hand-compiled NL -> predicate queries
      run.py                    # execute predicates on a split, emit metrics
      results/                  # gitignored except a committed summary.md
  tests/
    test_raycast.py             # TDD: write first, M1 = make pass
    test_predicates.py
  data/                         # gitignored; Occ3D-nuScenes lives here
```

Principle: `src/probe/` only gains code that ≥2 experiments share. `experiments/<x>/`
holds anything specific to one paper and is independently releasable at submission.

---

## 3. v0 scope (OccQuery first falsifiable experiment)

**In scope (v0):** two predicates (`lateral_clearance`, `free_along_ego_path`), the DDA
core, an Occ3D-nuScenes adapter, a GT-occupancy oracle, ~20 hand-written queries, the
metrics, and the expressivity comparison vs RefAV's function set. Run on
nuScenes-**mini** first, then the **val** split.

**Explicitly OUT of v0 (defer):** predicted-occupancy arm (running FB-OCC etc. — that is
v1, needs a GPU), natural-language→predicate auto-compilation (v0 compiles queries by
hand), a typed predicate "algebra", any UI, the other five experiments.

---

## 4. The falsifiable protocol (external-anchor version)

Replaces a self-graded bar (denotation F1 > 0.9 vs our own oracle) with three layers, ordered by
how hard they are to dispute: an oracle-free expressivity separation, a third-party-scored
retrieval anchor, and a denotation check whose oracle is released so others can recompute it. The
full per-topic matrix and verification status is in [`docs/benchmark-anchors.md`](docs/benchmark-anchors.md).

**H1 — expressivity (oracle-free, the headline).** A set of N safety-relevant spatial queries
(tight clearance, blocked free-path, narrowing corridor) is expressible as occupancy predicates but
is *not* expressible in RefAV's released box-only function set (`refAV/atomic_functions.py`: cuboid
translation/size/yaw + velocity + HD-map polygons; no dense-occupancy or free-space primitive).
Checkable by anyone against the public source — no oracle. The strongest form is a
non-identifiability witness: two scenes with identical box observables that differ only by unboxed
occupancy; a box-only language MUST return the same answer, the occupancy predicate does not.
Implemented as an executable test (`tests/test_expressivity.py`).

**H2 — retrieval competitiveness (third-party scored).** On the subset of our queries expressible
in RefAV's language, the retrieval backbone is competitive on a leaderboard nobody controls:
HOTA-Temporal >= 50 on the RefAV Argoverse 2 Scenario Mining test split (EvalAI, opened 2025-05-07;
public SOTA ~53.12 SMc2f / 52.37 Gemini-2.5-Pro, verified 2026-06-20). PRE-REQ: confirm the
challenge rules permit a deterministic-predicate-over-occupancy submission; otherwise report this as
an offline reproduction of the public eval protocol on val, explicitly labeled (not a leaderboard
placement).

**H3 — denotation correctness (released oracle, secondary).** Run the predicates over a RELEASED
third-party occupancy field (not one we produce) on Occ3D-nuScenes; report denotation P/R/F1 of the
returned scene-frames against a LiDAR-derived geometric oracle, AND beat the best box-only RefAV
approximation of the same query by >= 20 absolute F1. The RELATIVE gap is the load-bearing claim;
absolute F1 (target >= 0.90) is an internal-validity check. (Occ3D occupancy SOTA is
modality-dependent and currently PROVISIONAL: pin camera-only vs multi-modal before citing a
substrate number.)

**Predicates (v0).** `lateral_clearance` (physical free gap, ego body side to obstacle surface),
`free_along_ego_path`, and `min_free_width_along_path`. All carry an explicit `UnknownPolicy`
(free / occupied / ignored).

**Oracle.** Occ3D-nuScenes dense GT occupancy -- the accumulated, near-fully-observed `semantics`
field -- with clearance / free-space computed geometrically from it. Verified 2026-06-20 on the 10
mini scenes: on the dense GT, denotation is STABLE across the three `UnknownPolicy` rules (the GT
has ~0 unknown), so the premise holds. The per-frame visibility mask (~88% unknown on a single
lidar sweep) is deliberately NOT applied to the oracle -- masking the dense GT makes denotation
depend on the unknown policy rather than the geometry (it produces the section-4 kill shape on real
data). That masked observed view is instead the conditioning variable for gt-distrust /
vis-calibration and the predicted-occupancy robustness arm (v1), where the 3-rule unknown
sensitivity is the real test. RELEASE the oracle-construction code + held-out scene IDs.

**Baseline.** RefAV's released function set — demonstrate the spatial subset is *inexpressible* there
(H1), and that the best box-only approximation loses to free-space predicates by >= 20 F1 (H3).

**Metrics.** expressibility coverage (occupancy ~N vs RefAV ~0); HOTA-Temporal on the EvalAI subset;
denotation P/R/F1 vs the released oracle; the >= 20-F1 gap vs box-only; clearance MAE +
tolerance-accuracy ([75%,125%]). Every self-chosen threshold is pre-registered and reported as a full
curve — a cutoff a reviewer can move is not a result.

**Success (validates the angle):** (1) clean expressivity separation (occupancy ~N, RefAV ~0); AND
(2) HOTA-Temporal >= 50 on the EvalAI test split (or a labeled offline protocol reproduction); AND
(3) free-space predicates beat the best box-only approximation by >= 20 F1, with absolute F1 >= 0.90
stable across the 3 unknown-voxel rules as a secondary check.

**Kill / pivot:** if a box-only language matches our denotation quality (the >= 20-F1 gap collapses),
free-space predicates add no retrieval power — the angle fails. If unobserved-voxel ambiguity flips
denotation across the 3 rules even on a released field, shrink scope to dense-LiDAR free-space
(accumulated sweeps) and re-test. (A real possible outcome and an acceptable, honest result.)

**Prior art (scope the novelty precisely).** nuPlan and Waymo Open Motion already do
predicate-over-OBJECT-TRACKS scenario mining; RefAV owns box scenario mining; Scenic owns typed geo
predicates. The novel sliver is the occupancy/free-space SUBSTRATE + a released
denotation-correctness benchmark — not "predicate retrieval" in general.

**Synthetic results are not scientific results.** `experiments/occquery_v0` currently runs on
hand-built scenes (F1 = 1.00 by construction); that is a smoke test of the loop, not evidence. Every
number above requires the M2 Occ3D-nuScenes adapter and the released oracle.

---

## 5. Milestones (each ends in a reviewable artifact)

- **M0 — scaffold + this plan.** (done) Acceptance: structure + PLAN reviewed/approved.
- **M1 — core `raycast` + `grid` + unit tests.** TDD: `tests/test_raycast.py` is written
  first (synthetic grids with *known* visibility) and M1 = make it pass. Acceptance:
  green tests on hand-built grids (clear path visible; wall blocks; adjacency). No data
  needed.
- **M2 — Occ3D-nuScenes adapter on `mini`.** Load occupancy GT + visibility mask + ego
  pose for a handful of scenes. Acceptance: load 1 scene, sanity-check occupied/free/
  unknown counts + a quick 2D slice render.
- **M3 — predicates + oracle.** Implement `lateral_clearance`, `free_along_ego_path`, and
  the GT oracle. Acceptance: predicate values on a few frames match hand-computed values.
- **M4 — queries + metrics on mini → val.** `queries.yaml` (~20), `run.py`, metrics.
  Acceptance: the §4 success/kill thresholds evaluated; `results/summary.md` written.
- **M5 — short writeup + decision.** 1-page result + one figure → pursue (predicted-occ
  arm, v1) or pivot (dense-LiDAR free-space) per the kill criterion.

Rough effort: M1–M2 ~ days; M3–M4 ~ 1–2 weeks part-time; all CPU/laptop-feasible on mini.

---

## 6. Prior work to thread between (cite, don't overclaim)

- **RefAV** (arXiv:2505.20981) — NL scenario mining over Argoverse2 with 28 cuboid+
  velocity functions; the box-only *expressivity* baseline (it cannot express
  occupancy/free-space). Our oracle/denotation idea must NOT reuse its "GT-track ceiling"
  framing (it already owns that).
- **Scenic 3.0** (arXiv:2307.03325) — typed, composable geometric+visibility predicates,
  but generative/simulation spec, not retrieval-over-recorded-occupancy with a
  correctness benchmark. ⇒ demote visibility, lead with clearance/free-corridor.
- **NuScenes-SpatialQA** (arXiv:2504.03164) — quantitative spatial GT but per-frame VQA,
  cuboid scene-graph, no clearance/free-space.
- **POP-3D** (arXiv:2401.09413) — semantic voxel retrieval via CLIP (different
  denotation). **OccLLaMA / SparseOccVLA** — occupancy→LLM, learned, not
  deterministic/reproducible.

The defensible residue (verified open): deterministic physical predicates executed
against a **dense occupancy/free-space field** to retrieve real logged scenes, with (a) a
denotation-correctness benchmark vs an occupancy oracle and (b) a predicted-occupancy
robustness curve (v1).

---

## 7. Datasets & infra (access — confirmed feasible)

- **nuScenes-mini** (~4 GB) + **Occ3D-nuScenes** labels (mini subset) for M1–M4
  prototyping. Free; needs a nuScenes research account + terms.
- **nuScenes val** + Occ3D-nuScenes val labels for the real numbers (tens of GB).
- Compute: v0 is geometric ops over *GT* occupancy = **CPU / laptop**. A GPU is only
  needed for the v1 predicted-occupancy arm (running an FB-OCC-class checkpoint).
- His own voxel engine / SplatCarve are NOT required for v0 (numpy grids suffice) but are
  the substrate for v1+ and for the `asof` experiment later.

`data/` is gitignored; the repo never commits datasets or checkpoints.

---

## 8. Tech decisions

- **Python 3.11**, `numpy` + `pyyaml` for v0; `nuscenes-devkit` added at M2; `torch` only
  at v1. (Deviation from his usual TS/Swift is deliberate — the occupancy/AV ecosystem is
  Python; modules use snake_case per PEP8, not kebab/camel.)
- **pytest**, tests-first for the core primitive.
- Immutable-by-default, small files (<500 lines), small functions (<50). No premature
  abstraction in `core` — extract on second use only.
- Validation kept light (dataclasses); no heavy framework.

### Performance & the Rust core (`core-rs`) — later, not now

v0–v1 stay pure Python + numpy: correctness and falsifiability matter, not speed (the DDA
primitive vectorizes; heavy lifting is PyTorch = C++/CUDA). Do NOT add Rust now — premature
optimization (a known over-engineering trap). Escalate only when profiling shows the
raycast/predicate loop is the real bottleneck at dataset scale: `numpy → numba → Rust`.

When Rust earns its place — expected around **M4** (the interactive web demo needs the same
query logic client-side), or earlier if full-val raycast is the bottleneck — extract the hot
kernel ONCE into a Rust crate and wire both runtimes to it. This is the "extract on second
use" rule applied at the language boundary:

```
core-rs  (Rust: voxel / DDA raycast / predicate kernel)        <- single implementation
   |-- PyO3 (maturin)      ->  probe (Python)   : research experiments call the fast kernel
   |-- WASM (wasm-bindgen) ->  web/             : the browser demo runs the SAME kernel client-side
```

The canonical "Python API + Rust core" pattern (polars, tokenizers, ruff) plus a WASM build
for the web = one kernel, multiple runtimes (the SPACE0 one-engine pattern). Reuses Doeon's
existing Rust voxel engine.

Web performance: TypeScript erases to JavaScript at runtime (no perf cost vs vanilla JS —
types are pure upside). Heavy work is never in JS: 3D occupancy rendering runs on the GPU
via three.js / WebGPU; heavy client-side compute (live predicate queries) runs in Rust→WASM.
JS/TS is the orchestration shell only.

---

## 9. Public / private + build-in-public

- `src/probe/` (the instrument) → **public**, build-in-public. It is the
  commoditize-your-complement artifact his thesis argues for and a tangible PRISM-usable
  tool. Low scoop risk (tooling, not a result).
- `experiments/<x>/` *results/writeups* → private branch until paper-ready, released with
  the arXiv drop.
- License: TBD (Doeon) — Apache-2.0 or MIT both fine for an open instrument.
- No commit is made until this plan is approved; nothing is pushed without explicit ask.

---

## 10. Risks

| risk | mitigation |
|---|---|
| unobserved-voxel ambiguity poisons the oracle | the 3-rule sensitivity report IS the test; pivot to dense-LiDAR free-space if it flips |
| "occupancy retrieval" reads as incremental vs RefAV/Scenic | lead with expressivity gap (box-only *can't express* it) + denotation benchmark; cite both |
| over-scaffolding / premature core | extract to `core` only on 2nd use; v0 stays 2 predicates |
| PRISM's ex-42dot team already does internal versions | frame as a rigorous *open instrument* + reproducible benchmark, not "I'll teach you the field" |

---

## 11. Definition of done (v0) + what it unlocks

**Done when:** §4 thresholds are evaluated on nuScenes val with the 3-rule sensitivity,
`results/summary.md` + one figure exist, and a pursue/pivot decision is recorded.

**Unlocks:** the same `core` (raycast + Occ3D adapter + metrics) is then reused by
`gt-distrust` (occlusion-depth from the same DDA) and `vis-calibration` — i.e. v0 pays
for ~3 of the 6 experiments' shared substrate.

---

## 12. Open decisions for Doeon (please confirm before M1)

1. **Name** — keep `spatial-probe`? (alt: `state-over-render` as a build-in-public brand.)
2. **License** for the public instrument — Apache-2.0 / MIT / hold?
3. **Public from day 1**, or private until M4 then open the `core`?
4. Proceed to **M1 (implement `raycast` + tests)** now, or adjust the protocol first?
