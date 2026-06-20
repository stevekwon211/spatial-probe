# Benchmark anchors & success criteria

How each of the six spatial-probe research programs connects to an **externally verifiable**
success criterion. The owner's question was: *are we at least matching or beating a public
benchmark, and if there is no benchmark, what makes a result "success"?* The honest answer per
topic is below.

## How to read this

Each topic is classified:

- **RIDE** — competes on an existing public leaderboard (bar = >= current SOTA).
- **DEFINE** — defines a new evaluation axis with no direct leaderboard (bar = public dataset +
  measurable gap vs a public baseline + a released, reproducible benchmark).
- **HYBRID** — both: rides a public substrate/floor and defines a new axis on top.

Each success criterion is given in three tiers:

1. **Smoke** — the implementation works (internal, synthetic; NOT a scientific result).
2. **Publishable** — supported by public data, a public baseline, and held-out evaluation.
3. **Strong** — competitive on a public leaderboard, or the new benchmark is actually adoptable.

### Cross-cutting rules (apply to every topic)

- **Self-oracle is never the headline.** Lead with the part that needs no oracle (expressivity,
  descriptive direction-of-effect, beat-random). Where an oracle is unavoidable, release its
  construction code + held-out IDs so anyone can recompute it.
- **Self-chosen thresholds are a vulnerability.** Every fixed cutoff (0.90 F1, 95%, 0.7, 2x,
  80%) must be pre-registered AND reported as a full continuous curve, so a result does not hinge
  on a movable number. Where a *relative* claim exists (>= 20 F1 over box-only, beat-random), that
  is the load-bearing claim.
- **No cross-protocol number comparison.** The same metric name across different dataset
  versions, splits, input modality (camera-only vs multi-modal), supervision, or compute is NOT
  comparable. Match all of them or label it "adjacent, not comparable".
- **Stub / unreleased baselines are not baselines.** If the public code is a stub or
  irreproducible, reimplement self-contained and label it a reimplementation.
- **Negative / falsification conditions are part of the criterion**, not an afterthought.

### Verification status legend

- **VERIFIED** — confirmed against a primary source (paper / leaderboard / repo) on the date shown.
- **PROVISIONAL** — from a 13-agent adversarial research+verification workflow (2026-06-20);
  benchmark *existence* cross-checked, but specific SOTA numbers are **not yet primary-confirmed**.
  Do not cite as fact until re-verified.
- **UNVERIFIED** — surfaced but not checked; treat as a lead only.

---

## Verification ledger

| topic | category | nearest public benchmark | SOTA (status) | oracle dependence | status |
|---|---|---|---|---|---|
| occquery | HYBRID | RefAV / AV2 Scenario Mining (HOTA-Temporal, EvalAI) | HOTA-T 53.12 SMc2f / 52.37 Gemini-2.5-Pro (**VERIFIED**) | H1/H2 none; H3 released oracle | partial **VERIFIED** |
| dynfield | HYBRID | nuPlan closed-loop / NAVSIM (planner oracles) | PDM-Closed CLS-R ~92; NAVSIM PDMS ~91 (PROVISIONAL) | rides public planner; matrix is new | PROVISIONAL |
| value-of-correction | DEFINE+anchor | Argoverse2 3D det + WACV-2026 correction code | beat-random at fixed budget (no leaderboard) | downstream metric (external) | PROVISIONAL |
| asof | HYBRID | Occ3D-nuScenes RayIoU (SparseOcc protocol) | RayIoU OPUS-V2 ~44 (PROVISIONAL) | floor external; preservation metric self-defined | PROVISIONAL |
| gt-distrust | DEFINE | none direct (Occ3D masks as substrate) | no leaderboard | injected-error (self-controlled) + downstream | PROVISIONAL (weakest) |
| vis-calibration | DEFINE | none direct (ReliOcc ECE protocol, Occ3D mask) | no leaderboard | descriptive (external mask) | PROVISIONAL (last) |

**Stale / mislabeled numbers flagged by verification — do NOT cite until re-checked:** Occ3D mIoU
"SOTA" is modality-dependent (camera-only ~34.5 vs multi-modal 50s; DAOcc 54.33 / Gau-Occ ~55.1
are PROVISIONAL); nuScenes-detection SOTA is ~FusionFormer 75.1 NDS, not MEFormer; PDM-Closed
CLS-R ~92 must not be conflated with PDM-Hybrid OLS ~42/84; GS-Occ3D / OccNL / ReliOcc public code
is stub-or-unverified.

---

## 1. occquery — occupancy-native predicate retrieval **[HYBRID]**

**Core claim.** Deterministic physical predicates (clearance, free-path, free-width) over a dense
occupancy field retrieve logged driving scenes that a box-only query language cannot express, and
they do so denotation-correctly.

**Scope (2026-06-21, PLAN s4).** occquery MEASURES box-blind static geometry (free-width, clearance,
free-path); it does NOT judge whether a measured situation is *dangerous* (a corridor the ego cannot
pass vs a lead car it follows, a near-miss vs a car parked beside it) — that requires relative motion
over time and is the **dynfield** topic. So the success below is **measurement accuracy + expressivity**,
never danger-retrieval F1. (A 0.8 m free-width beside a lead car is a CORRECT measurement, not a false
positive; "it's a lead car, not a wall" is a dynfield verdict.)

- **Public dataset / substrate.** Argoverse 2 Sensor (RefAV scenario mining); Occ3D-nuScenes
  (occupancy substrate for the denotation arm).
- **Public baseline.** RefAV's released function set (`refAV/atomic_functions.py`, ~33 cuboid +
  velocity + map functions; no dense-occupancy / free-space primitive).
- **External evaluator / protocol.** RefAV AV2 Scenario Mining on EvalAI, metric **HOTA-Temporal**
  (test split opened 2025-05-07).
- **Primary metric.** Expressivity coverage (oracle-free) + denotation F1 vs a released oracle.
  **Secondary.** HOTA-Temporal on the RefAV-expressible subset; clearance MAE.

**Success tiers**

- **Smoke (done).** Retrieval loop runs end-to-end on synthetic scenes; predicates verified on
  known geometry; the expressivity witness (identical box observables, occupancy distinguishes)
  holds as an executable test. 59 unit/integration tests green. *Not a scientific result.*
- **Publishable.** On public Occ3D-nuScenes val: (a) expressivity separation — N safety queries
  expressible as occupancy predicates, ~0 in RefAV's released function set, checkable by anyone
  against `atomic_functions.py` (**verified 2026-06-20: 0 free-space primitives in the 33-function
  set**, see [expressivity-vs-refav.md](expressivity-vs-refav.md); oracle-free, the headline); (b) denotation P/R/F1 over a
  **released** occupancy field vs a LiDAR-derived oracle whose construction code + held-out scene
  IDs are released; (c) free-space predicates beat the best box-only approximation by **>= 20
  absolute F1** (the relative gap is load-bearing; absolute F1 is a secondary internal check),
  stable across the 3 unknown-voxel policies.
- **Strong.** On the RefAV-expressible subset, **HOTA-Temporal >= 50** on the EvalAI test split
  (public SOTA ~53.12) — third-party scored. Pre-req: confirm the challenge permits a
  deterministic-predicate-over-occupancy submission; otherwise report as an offline reproduction
  of the public eval protocol on val (explicitly labeled, not a leaderboard placement).
- **Falsification.** If a box-only language matches our denotation quality (the >= 20 F1 gap
  collapses), free-space predicates add no retrieval power — the angle fails. If unobserved-voxel
  ambiguity flips denotation across the 3 policies even on a released field, shrink scope to
  dense-LiDAR free-space (accumulated sweeps) and re-test.

**Oracle dependence.** H1 (expressivity) and H2 (HOTA leaderboard) need NO oracle; H3 (denotation)
uses an author-reconstructed oracle that is released for external recompute.
**Data/code availability.** RefAV + EvalAI public; Occ3D-nuScenes public (gated by nuScenes terms).
**Main confounders.** unobserved voxels (3-policy report); RefAV task target vs our retrieval
target may differ (verify metric compatibility before claiming a leaderboard number).
**Verified source / date.** RefAV = AV2 Scenario Mining baseline, HOTA-Temporal metric, EvalAI test
split 2025-05-07 — arXiv:2505.20981 (**VERIFIED 2026-06-20**). SOTA HOTA-T: SMc2f 53.12
(arXiv:2601.12010), Gemini-2.5-Pro 52.37 (arXiv:2506.11124) (**VERIFIED 2026-06-20**). The
previously-cited "53.38 / Zeekr_UMCV / 2026-02-18" is **UNVERIFIED** — not used. Occ3D mIoU SOTA is
**PROVISIONAL** (modality not pinned).
**Status: partially VERIFIED (the RIDE anchor and its SOTA are primary-confirmed).**

---

## 2. dynfield — which dynamics field a planner needs, by regime **[HYBRID]**

**Core claim.** Not every dynamics field (occupancy-flow, scene-flow, instantaneous velocity)
helps a planner in every regime; necessity/sufficiency is regime-dependent and measurable.

- **Substrate / oracle.** nuPlan closed-loop (reactive) and NAVSIM — released reference planners.
- **Baseline.** PDM-Closed (released `autonomousvision/tuplan_garage`).
- **Metric.** Closed-loop score (CLS-R) / PDMS, sliced by nuPlan scenario-type tags.

**Success tiers**
- **Smoke.** Ablation harness runs; a no-dynamics control provably collapses closed-loop score in
  interactive regimes (proves the harness is not leaking privileged state).
- **Publishable.** On public splits, a {field x regime} necessity/sufficiency matrix with the full
  continuous delta-vs-S_full curve and seed-variance bands; pre-registered sufficient/necessary
  cutoffs.
- **Strong.** Reproduce a released planner's headline number (e.g. PDM-Closed CLS-R ~92, nuPlan
  Val14 reactive — PROVISIONAL) as the harness anchor, then show a field that is sufficient in some
  regimes and necessary in others.
- **Falsification.** Dies if some field is necessary in EVERY regime (kills the by-regime thesis)
  or if no field is ever sufficient anywhere (kills parsimony).

**Oracle dependence.** Rides external planners (low). **Confounder.** harness leakage (the
no-dynamics control gate is mandatory). **Status: PROVISIONAL** — metric names/splits
(CLS-R vs OLS, PDM-Closed vs Hybrid) must be primary-confirmed before any number is cited.

---

## 3. value-of-correction — which label fixes move the model **[DEFINE + external anchor]**

**Core claim.** A correction-prioritization ranking recovers more downstream performance per unit
correction budget than random/uniform selection.

- **Substrate.** Argoverse 2 3D detection (official devkit, AP/ATE/CDS).
- **Baseline.** Released WACV-2026 geometric box-correction code + a public VoxelNeXt checkpoint;
  random / uniform / uncertainty selection.
- **Metric.** Downstream CDS recovered per unit budget (a curve, not one cutoff).

**Success tiers**
- **Smoke.** OpenDataVal-style noisy-label-detection F1 / point-removal AUC validates the ranker
  mechanics on injected noise.
- **Publishable.** Retrain/fine-tune on corrected vs uncorrected TRAINING data; a value-ranked
  budget beats random AND beats geometric-correction-without-prioritization at equal budget, on the
  official AV2 val split, >= 3 seeds, released reproduction code, full utility-vs-budget curve.
- **Strong.** The released benchmark (budget x recovery curve + ranker) is adoptable by others.
- **Falsification.** Value-ranked corrections do not beat random at equal budget.

**Critical fix flagged by verification.** Do NOT inherit WACV Table-3's val-PERTURBATION numbers
(e.g. 18.7->23.0) as a *retrain* recovery ceiling — that experiment does not exist in the paper;
measure the real full-correction delta first. **Oracle dependence.** downstream metric is external
(low). **Status: PROVISIONAL.**

---

## 4. asof — does render→state preserve action-relevant signal **[HYBRID: RIDE floor + DEFINE]**

**Core claim.** Converting a render representation (3DGS/NeRF) to occupancy preserves the spatial
properties a planner needs (free space, clearance, line-of-sight), measurably.

- **Floor (external today).** Occ3D-nuScenes **RayIoU** under the public SparseOcc protocol
  (`MCG-NJU/SparseOcc`).
- **Contribution (new).** A render-to-state property-preservation metric (clearance-IoU,
  free-space connectivity, occlusion boundary), released as code + frozen config.

**Success tiers**
- **Smoke.** Property-preservation metric computes on a toy converted field.
- **Publishable.** render-derived RayIoU >= 0.90x the IDENTICAL pipeline fed LiDAR/GT occupancy
  (pre-registered denominator, run both); clearance/free-path denotation preserved within a
  released, pre-registered band through a public occupancy-consuming planner.
- **Strong.** Reproduce a closed-loop nav result (Splat-Nav / CATNIPS) on a shared scene set.
- **Falsification.** Conversion preserves mIoU/RayIoU but destroys clearance/free-path denotation
  (then "render preserves state for planning" is false) — itself a publishable negative.

**Oracle dependence.** RayIoU floor external; the preservation metric is self-defined (release the
code or it stays self-graded). **Status: PROVISIONAL** (RayIoU SOTA numbers and GS-Occ3D code
release unconfirmed).

---

## 5. gt-distrust — occlusion predicts untrustworthy GT labels **[DEFINE — weakest]**

**Core claim.** A per-voxel occlusion-depth score predicts which occupancy GT labels are
untrustworthy (labeled through an occluder), beyond a binary observed/unobserved mask.

- **Substrate.** Occ3D-nuScenes GT + released per-voxel LiDAR visibility mask; a released occupancy
  checkpoint as the downstream probe.
- **Self-contained oracle.** synthetic-occluder injection (known-error GT you control) — removes
  the dependency on OccNL, whose repo is a stub.

**Success tiers**
- **Smoke.** DDA occlusion-depth flag computes; agreement with the LiDAR visibility mask
  (NOT the camera mask — that shares the flag's own ray-cast, near-tautological).
- **Publishable.** (a) injected-error detection AP beating an occlusion-depth-shuffled control AND
  a distance-from-ego baseline; (b) excluding flagged GT changes a released checkpoint's mIoU AND
  RayIoU strictly more than excluding a range/class-matched control mask.
- **Strong.** The continuous per-voxel trust score + protocol is adopted as a GT-audit tool.
- **Falsification.** injected-error detection at chance, OR downstream delta indistinguishable from
  the matched control.

**Oracle dependence.** self-controlled injection (good) + external downstream (good); the original
camera-mask-agreement leg was circular and is dropped. **Status: PROVISIONAL, weakest of the six**
(no leaderboard; two original legs were broken and repaired).

---

## 6. vis-calibration — is confidence honest where the sensor cannot see **[DEFINE — last]**

**Core claim.** Occupancy models are less calibrated in unobserved regions than observed ones — an
untested hypothesis with no external precedent for the effect size.

- **Substrate.** Occ3D-nuScenes released per-voxel visibility mask (the conditioning variable);
  >= 2 public camera-only occupancy checkpoints; ReliOcc's ECE_geo/ECE_sem protocol (ECE is
  standard — reimplement if ReliOcc code is unreleased).

**Success tiers**
- **Smoke.** Per-stratum ECE computes on one public checkpoint.
- **Publishable (descriptive, the load-bearing part).** On >= 2 public checkpoints, partition
  scored voxels by the released mask into observed/unobserved, compute per-stratum
  ECE_sem/ECE_geo with per-scene bootstrap 95% CIs, and show the unobserved stratum is
  significantly worse on BOTH models; reproduce the direction on a second substrate. Report
  reliability diagrams + bin sensitivity, not a single ECE.
- **Strong.** A visibility-aware recalibration that reduces the unobserved-stratum gap (a SEPARATE,
  explicitly-labeled stretch target; any "<= X% ECE" figure is aspirational until measured).
- **Falsification.** observed/unobserved ECE statistically indistinguishable on the public mask, or
  the gap vanishes under the released mask vs a custom labeling.

**Oracle dependence.** descriptive direction-of-effect needs no oracle; the recalibration leg is a
stretch. **Status: PROVISIONAL, run last** (cheapest once the Occ3D visibility-mask plumbing from
occquery/gt-distrust exists).

---

## Sequencing

`occquery -> dynfield -> value-of-correction -> asof -> gt-distrust -> vis-calibration`

Rationale: front-load the topics whose bar is a third-party-scored leaderboard or a released
directly-comparable baseline (1–3), then the ride-floor-plus-define-contribution topic (4), and
defer the two pure new-axis topics with no external anchor (5–6) until the shared occquery core
(raycast + Occ3D adapter + visibility-mask handling + metrics) makes them cheap. This produces an
externally-verifiable result first and pushes the highest-dispute-risk, oracle-dependent work last.
