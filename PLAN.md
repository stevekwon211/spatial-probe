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

## 4. The falsifiable protocol (the rigorous part)

**H1 — expressivity.** A set of safety-relevant spatial queries (tight clearance,
blocked free-path, narrowing corridor) is expressible as occupancy predicates but
*cannot* be expressed with a box-only function set (RefAV's 28 cuboid+velocity functions).

**H2 — correctness.** Executing those predicates on occupancy retrieves a scene set that
matches a hand-verified ground-truth set with high denotation F1 *on GT occupancy*, and
the continuous predicate (clearance value) is accurate.

**Predicates (v0).**
- `lateral_clearance(scene, t)` — min horizontal distance from the ego lane corridor to
  the nearest occupied, non-ground voxel at time `t`. Example query: *clearance < 0.5 m
  while ego speed > 30 km/h.*
- `free_along_ego_path(scene, t, horizon)` — boolean: does the ego's swept footprint over
  `[t, t+horizon]` stay collision-free in the occupancy grid? Example query: *the only
  free path narrows below vehicle width.*

**Oracle.** GT occupancy (Occ3D) + GT ego pose + GT ego box → compute "true"
clearance / free-space directly. **Main validity threat = `unobserved` voxels.** Define
3 handling rules (treat unobserved as free / as occupied / as excluded) and **report
denotation sensitivity across all three.** Stability across rules = the premise holds;
instability = the premise is at risk.

**Baseline.** RefAV's 28 functions — demonstrate the spatial-geometry subset is
*inexpressible* there (the expressivity gap), not just "scores worse."

**Metrics.**
- expressibility coverage: of N queries, # occupancy can express vs # RefAV can express.
- denotation precision / recall / **F1** of the returned scene set vs hand-labeled GT.
- clearance MAE + tolerance-accuracy ([75%,125%] of GT).

**Success (validates the angle):** clean expressivity separation (occupancy ≈ N, RefAV ≈
0 on the spatial subset) AND denotation F1 > 0.9 on GT occupancy with low clearance MAE,
stable across the 3 unobserved-voxel rules.

**Kill / pivot:** even on GT occupancy, unobserved-voxel/boundary ambiguity makes the
oracle so fuzzy that denotation flips across the 3 rules → predicate-on-Occ3D premise is
shaky → shrink scope to dense-LiDAR free-space (accumulated sweeps) instead of Occ3D
voxels, and re-test. (This is a real possible outcome and an acceptable, honest result.)

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
