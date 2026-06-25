# Data substrate scarcity and the oracle problem — findings and a program reframe

Session note, 2026-06-23. Captures a chain of investigation that started as "where do we get danger
data for dynfield" and ended at "what is the oracle for this whole program, and is the framing right."
Three multi-agent web surveys (61-dataset sweep, AV2-conflict verification, A/B stress-test) plus the
existing dynfield runs ground it. Confidence is marked per claim; primary sources are cited inline in
the underlying workflow transcripts.

## TL;DR

1. **Danger is never a downloadable label.** Real crashes exist only as monocular dashcam RGB (no metric
   3D possible); real 3D/LiDAR datasets are crash-free by design. The two corpora are disjoint and cannot
   be merged. Confirmed across ~61 datasets.
2. **No public dataset gives {real LiDAR/occupancy + per-object velocity + elevated danger} in one split.**
   Argoverse-2 and Waymo both split these across non-overlapping scene pools. Confirmed from primary
   sources (incl. the Argoverse team).
3. **A surrogate must match the conflict geometry of its substrate, or the measurement is a category
   error.** Our longitudinal IDM cannot be run on the lateral intersection-crossing AV2 conflict subset.
   This killed the convenient path and is the methodological headline.
4. **The recurring blocker across 5 of 6 program stages is the same: the independent-oracle problem.**
   On one dataset, the only buildable measurement oracle shares the data source, so it is circular.
5. **The reframe (load-bearing):** the *real* oracle is human-defined and behavioral — "don't collide,
   obey the law, make progress, stay comfortable" — validated in the closed loop / real world, not
   against a pre-existing per-frame answer key. Per-frame ground-truth datasets are a *cheap proxy* for
   that outcome. The honest version of this program anchors on whether a representation **changes the
   action/outcome**, not on whether it matches a per-frame truth. Several stages already pivoted this
   way; the question validated those pivots and flagged the stages that did not.

## 1. What triggered this

dynfield's by-regime thesis came out **half-confirmed on real nuScenes**: a graded IDM surrogate shows
the stored velocity field is action-**redundant** in safe vehicle-following (n=443, clean, true
decel-delta CI below a shuffled-velocity control). The complementary "necessary when dangerous" half is
**untestable on nuScenes** because the data is benign — measured ~22 / 2114 lead-frames at TTC < 2s, and
~0 genuine fast-closing near-misses (lead closing-speed median +0.07 m/s; leads travel *with* the ego,
not at it). So the immediate question was: where does the danger data come from. (Confirmed; see
`experiments/dynfield_v0/results/summary.md` and `danger_density.py`.)

## 2. Finding 1 — danger is not downloadable

Why no public dataset carries real crashes in a usable 3D form. Structural, not a search failure:

- **Crash base-rate makes 3D capture irrational.** ~1 police-reported crash per ~500k miles ≈ ~8,750
  recording-hours per crash. SHRP2 (largest naturalistic study) drove 35M miles for ~1,500 crashes — and
  ran **no LiDAR**. Every LiDAR dataset (nuScenes, Waymo, AV2, KITTI) is therefore crash-free by design,
  curated for perceptual diversity.
- **Retro-instrumentation is impossible.** A dashcam records what happened; LiDAR must be present at
  capture. The millions of crash videos on the internet can never be upgraded to metric 3D. **The
  crash-video corpus and the 3D-sensor corpus are disjoint and cannot be merged.**
- **AV fleets avoid the event they would record** (safety driver intervenes), so naturalistic AV data has
  zero crash frames by selection.
- **Legal / privacy / liability + annotation cost** block release even where data exists.

Four buckets (confirmed): real-crash video (RGB-only, unusable for occupancy); real near-miss/naturalistic
(3D sometimes, danger or access the problem); simulation-generated (right modality, by-construction
danger → smoke test, not evidence); adversarial-generation-on-real-logs (real distribution, but
trajectory/BEV-only and often circular on a bicycle-model prior).

## 3. Finding 2 — the AV2 / Waymo substrate investigation

The one pre-mined real near-miss artifact found is the **TU-Delft Argoverse-2 conflict subset**
(arXiv:2308.13839): 21,431 near-miss scenarios (5,337 AV-involved), real, downloadable (4TU + GitHub,
~1.75 GB), velocity first-class. It looked like the way around nuScenes' benign data. The verification
killed the convenient path **two ways**:

1. **No LiDAR link (confirmed, primary source).** The conflict subset is built on AV2 *Motion
   Forecasting*, which has **no released LiDAR** (the Argoverse team: 250k scenarios ≈ 25 TB, never
   published — GitHub issue #4). AV2 *Sensor* has LiDAR but a separate, non-linkable ID namespace and no
   stored velocity. So the conflict scenes are **2D trajectories + velocity, no occupancy, ever.**
2. **Surrogate–substrate category error (confirmed).** The conflict subset is **100% lateral
   intersection-crossing**. Our IDM is a **longitudinal car-following** model — it has no lead vehicle in
   a crossing conflict, so running it there fabricates a fictitious lead and measures noise. This is the
   same shape as occquery's v0 0/5 error (a static instrument graded by a dynamic key).

**Waymo reproduces the same split**, it does not escape it: Occ3D-Waymo occupancy GT is on 2,030
Perception scenes; per-object velocity/trajectory is on 103k Motion scenes; WOMD-LiDAR covers only a
1-second history window. Danger is never a native label anywhere — TTC/near-miss is always a post-hoc
researcher derivation. (Confirmed.) Net: **no off-the-shelf {real LiDAR + occupancy GT + velocity +
higher-than-nuScenes danger} dataset exists.**

## 4. The principle that fell out — match the surrogate to the conflict geometry

Standard surrogate-safety practice: **crossing conflicts → PET (post-encroachment time) + gap-acceptance
/ time-to-conflict-point yield-go**; **following conflicts → TTC + a car-following model (IDM)**. The
surrogate and the substrate must agree on geometry. This makes "which dataset" downstream of "which
surrogate," not a free choice.

A consequence: the repo's v2 hint of decel-delta ≈ 0.88 at *crossing* came from running the longitudinal
IDM on crossing geometry — **the wrong instrument**. That number cannot be trusted until re-measured with
a proper crossing surrogate. (Correction; previously treated as a live hint.)

## 5. Corrected substrate menu

| Option | Surrogate fit | Gives | Honest gap |
|---|---|---|---|
| ~~AV2-conflict + our IDM~~ | category error | — | **dropped** (§3.2) |
| AV2-conflict + new yield-go surrogate | ✅ crossing | dense real crossing near-miss + velocity | 2D only (no occupancy, ever); new surrogate to build; a *separate* experiment |
| **HighD/NGSIM cut-in + our IDM** | ✅ following | dense labeled highway cut-ins, velocity first-class | 2D only (no occupancy); highway not urban; still action-sensitivity, not necessity |
| Self-mine AV2 Sensor LiDAR | mining target sets it | real LiDAR + self-gen occupancy + velocity | ~1 week occupancy build; danger density unmeasured (may be nuScenes-like) |
| Waymo Perception + Occ3D | mining target sets it | downloadable occupancy GT + LiDAR | reproduces split; no native danger; Mac arm64 path fragile |
| Closed-loop sim (PDM-Closed) — Tier-2 | IDM→planner | a real quality oracle (collision score) → *necessity* | GPU-gated; self-built-sim circularity if generator = grader |

**Recommended next (real-data, cheap, methodologically clean): HighD/NGSIM cut-ins + the existing IDM.**
It is the one pairing that is not a category error, and it directly tests the half nuScenes could not:
"does the velocity field change the IDM action in *dangerous* following?" — the falsifiable complement to
the clean "redundant in *safe* following" result. The `harness_v2.py` decel-delta + shuffled-control
machinery transfers almost unchanged. **Gate it first** on a one-recording danger-density count (cut-in
frames at TTC < 2s); if density is not materially above nuScenes' ~22/2114, HighD is also too benign and
the path escalates to sim. (Plan; HighD danger density is **inference, unverified** — the gate decides it.)

Occupancy-grounded and crossing-conflict versions are **separate, later, gated** experiments, not this
step.

## 6. Industry framing — why this is everyone's problem

The scarcity + oracle problem is industry-wide. Tesla's fleet-data moat is a real answer to the
**scarcity** half: millions of cars can mine the rare danger events public datasets lack, plus
auto-labeling and an occupancy network. But two honest qualifications:

- **"Closest to success" is contestable.** Waymo is operationally ahead — driverless robotaxis at scale
  in several cities today. Tesla FSD is still L2 (supervised). "Closest" depends on whether you weight
  *scalable approach* (Tesla) or *driver already removed* (Waymo).
- **A data moat solves scarcity, not the oracle.** Tesla does not grade its occupancy net against
  absolute 3D truth; it grades the *driving* against outcomes ("drives well," validated operationally).
  Auto-labeling is still self-supervision from the same sensor stream. The measurement-oracle epistemics
  are *sidestepped by a product bar*, not solved.

Implication for a solo researcher: the data-moat game is unwinnable solo. The **measurement-honesty /
independent-oracle** game is a different game that even the data giants only sidestep — and it is exactly
this program's wedge.

## 7. The reframe — there are two oracles, and the real one is human-defined

"Oracle" was doing too much work. Two distinct things:

- **(A) Outcome oracle — real, human-defined, behavioral.** "Don't collide, obey the law, make progress,
  stay comfortable." Not a pre-existing answer key — a *specification* humans wrote, validated in the
  world (did the car crash). Nobody in the field started with the answer key; Waymo/Tesla's real oracle
  is fleet outcomes. This is the foundational truth.
- **(B) Per-frame measurement oracle — a cheap proxy.** "Is this occupancy voxel actually free?" Used
  *because* the outcome oracle is expensive, rare (crash scarcity), and does not localize blame. (B) is a
  fast offline stand-in for (A), valuable only insofar as it predicts (A).

Both have a hard side, two faces of one coin: **(B) is circular** (graded by truth from the same data);
**(A) is sparse + expensive + does not isolate components + the human spec is itself fuzzy at the margin**
(a car that never moves never collides; the real spec is a contested safety/progress trade-off). Neither
is free. The field uses both: outcome as the north star, per-frame proxy as the cheap localizable
stand-in — and the honest research question is *keeping the proxy faithful to the outcome*.

**This validates pivots the program already made** and flags the ones it did not:

- occquery **H1 (expressivity)** is oracle-free — "this predicate expresses what box-only cannot" — and is
  the sound headline. occquery **H3 (per-frame denotation accuracy)** is the circular-proxy one, correctly
  **demoted**.
- dynfield was demoted from "denotation accuracy" to **action-sensitivity** — "does removing the field
  change the action" — which is the outcome side, not the per-frame side. The right oracle there is the
  *outcome* one (closed-loop), which is exactly why necessity needed sim (Tier-2).
- The program's **weakest stages (gt-distrust REJECT-as-unsound, vis-calibration)** are precisely the ones
  still anchored on a per-frame ground-truth oracle. The reframe explains *why* they are fragile.

## 8. So how do we proceed

1. **Anchor the program on action/outcome-sensitivity, not per-frame denotation correctness.** "Does the
   representation change the action a planner takes?" is oracle-cleaner than "is the representation
   per-frame correct?" This is the through-line: occquery H1 (expressivity), dynfield (action-sensitivity)
   already live here; re-state the program narrative around it.
2. **dynfield next step:** HighD/NGSIM cut-ins + the existing IDM, gated on a danger-density count
   (§5). Cheap, real, clean. It closes the open half ("velocity in dangerous following") on real data.
3. **Treat as separately-gated follow-ons:** occupancy-grounded danger (self-mine AV2 Sensor or Waymo
   Perception, gated on its own danger-density check), the crossing-conflict experiment (a yield-go
   surrogate, its own substrate), and necessity-via-sim (Tier-2, GPU, with an *independent* generator vs
   grader to avoid self-built-sim circularity).
4. **Send the open-problems research request** (drafted, in clipboard / to be filed as
   `open-problems-research-request.md`): ask the field how it manufactures independent oracles
   (cross-modality, multi-traversal, future-frame reveal) for each stage's blocker.
5. **Position the wedge honestly:** not "out-data the giants," but "measure, honestly, whether a spatial
   representation stores and changes the action — and keep the cheap proxy faithful to the human-defined
   outcome." That is the niche even Tesla/Waymo only sidestep.

## 9. Confidence ledger

- **Confirmed (primary source):** danger-corpus disjointness; AV2 MF-has-no-LiDAR + non-linkable to
  Sensor; AV2 conflict subset is 100% crossing; IDM-on-crossing is a category error; Waymo reproduces the
  split; nuScenes danger density ~22/2114 at TTC<2s (our own run).
- **Inference (unverified, has a named check):** HighD's dangerous-following density beats nuScenes (the
  §5 gate decides it); AV2 Sensor / Waymo Perception danger density (own gate); whether the crossing
  decel-delta survives a proper yield-go surrogate (re-measure required).
- **Reframe (argued, not a measurement):** the two-oracle distinction and the "anchor on action/outcome"
  recommendation are a position, defended above; they reorganize the program but are not themselves an
  empirical result.
