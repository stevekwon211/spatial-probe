# Research Request — Independent-Oracle and Substrate Solutions for the spatial-probe Six-Paper Program

I am a solo researcher running an empirical research program on **occupancy-native spatial
representations for autonomous driving**. I need help with one recurring, program-wide obstacle and
its six stage-specific instances. Please research current (2024–2026) methods, datasets, tools, and
papers that address the questions below, and flag anything I have missed. Be adversarial: if a
proposed oracle or substrate is circular or unsound, say so.

---

## 1. The thesis and the instrument

**One thesis:** a 3D scene is *queryable, updatable state*, not a render. Each of six papers takes one
falsifiable physical predicate, runs it over a representation, and measures whether that representation
*stores the signal needed to answer the predicate* — always with a fairness control so the instrument
is not rigged.

- **Instrument** (durable asset): a pure-CPU Python core — 3D DDA raycasting (line-of-sight,
  occlusion-depth), an occupancy grid model, falsifiable geometric predicates (lateral clearance,
  free-path / corridor width, reachable free-space via configuration-space flood-fill), a
  schema-validated natural-language → predicate query DSL, and an Occ3D-nuScenes adapter. numpy + scipy
  only; zero torch/CUDA.
- **Substrate in use:** Occ3D-nuScenes (voxel occupancy ground truth derived from nuScenes LiDAR +
  3D boxes), plus tracked-box motion.
- **Each paper** is an experiment on top of the shared instrument.

## 2. The hard constraints that SHAPE every blocker (please respect these — they are non-negotiable)

These come from a strict research-integrity rule set. They are *why* the naive solution to each
blocker does not count, so please do not propose solutions that violate them:

1. **Independent oracle.** Whatever GRADES a measurement must differ from the instrument in **both data
   source AND algorithm**. An oracle built from the same dataset (e.g. grading an Occ3D-derived
   predicate with an Occ3D-derived "truth") is a *consistency check*, not external evidence. This is
   the single biggest blocker — see §4.
2. **Pre-register before data; relative over absolute.** Hypotheses, analysis path, and success/kill
   criteria are committed (git-timestamped) before looking at outcomes. Claims must be a *relative gap
   vs a pre-registered public baseline*, never a movable absolute threshold.
3. **Synthetic is not science.** A by-construction result (e.g. F1=1.00 on hand-built scenes, or a
   simulator where the answer is baked into the scenario generator) is a smoke test, never evidence.
4. **Report negatives; reachable kills.** A design where every outcome is publishable is unfalsifiable
   and rejected.
5. **Hardware/operational reality:** solo researcher, Apple-Silicon Mac (no CUDA), RunPod available for
   GPU bursts (~$0.5–2/hr). Datasets behind free research-account gates are usable; I cannot use
   anything requiring institutional/NDA access (e.g. SHRP2) or proprietary fleet logs.

## 3. Program order, status, and the six axes

Order (by external-anchor strength): `occquery → dynfield → value-of-correction → asof → gt-distrust →
vis-calibration`. Only occquery rides a live leaderboard; the other five must DEFINE a new axis on
public data (public dataset + measurable gap vs a public baseline + a released benchmark). All six are
PLANS; only occquery has code. The partition is
`render→state→query→trust→curate→dynamics`, MECE.

---

## 4. THE CROSS-CUTTING PROBLEM (the most important question)

**Five of the six stages are blocked by the same thing: the independent-oracle / circular-ground-truth
problem on a single AV dataset.** Every "is this measurement *correct / necessary / trustworthy*" claim
needs a ground truth that differs in data source AND algorithm from the instrument — but on one dataset
(nuScenes/Occ3D), the only buildable oracle tends to share the data source, making it circular.

**General research questions:**
- How does the AV / robotics / ML-measurement literature obtain **independent ground truth for
  occupancy and free-space** when the only dense label is itself sensor-derived from the same logs?
- Are there standard **cross-modality** oracle constructions (e.g. LiDAR-derived predicate graded by a
  camera/radar-derived truth) that are genuinely independent in both data source and algorithm — and
  what are their known failure modes (calibration, FOV mismatch, shared upstream)?
- Is **multi-traversal / multi-sensor aggregation** (the same place driven multiple times, or a second
  vehicle) a recognized way to manufacture an independent oracle? Which public datasets support it?
- How do measurement papers in adjacent fields (medical imaging, remote sensing, SLAM benchmarking)
  break analogous circularity, and does any of it transfer?
- Secondary recurring blockers: (a) **GPU-gated publishable tiers** — what are minimal-GPU or no-GPU
  paths to a defensible result for each stage below? (b) **substrate availability** — which public
  datasets actually carry the needed property (danger events, paired render+GT, multi-traversal)?

---

## 5. Per-stage blockers and specific research questions

### Stage 1 — occquery (state→query, geometry) — ACTIVE, has code
- **Hypothesis:** occupancy predicates measure box-blind free-space that box-only query languages cannot
  express (H1, expressivity), and do so denotation-correctly (H3, accuracy).
- **Status:** H1 (expressivity) HOLDS and is oracle-free — it is a constructive witness (a scene where
  an occupancy predicate separates cases a box-only language provably cannot). This is the sound
  headline.
- **BLOCKER (H3):** a real denotation precision/recall/F1 against an **independent** oracle. The only
  oracle I can build (a human/visual call on the same dense Occ3D slice the predicate reads, or a
  LiDAR-derived free-space mask from the same nuScenes logs) **shares the predicate's data source → it
  is a consistency check, not external truth.** A camera-derived cross-check I tried gave AUC 0.798
  (insufficient) and was vacuous on a zero-positive mini-split.
- **Questions:** What is the state of the art for an **independent free-space / occupancy oracle** for a
  geometric predicate? Is there a public dataset with free-space ground truth from a *different* sensing
  modality or a *different* labeling algorithm than nuScenes/Occ3D? How do occupancy-prediction
  benchmarks (Occ3D, OpenOccupancy, SurroundOcc) define "ground truth," and is any of it independent
  enough to grade a predicate, or is it all self-referential LiDAR accumulation?

### Stage 2 — dynfield (dynamics) — PLAN + prototype
- **Hypothesis:** *which* stored motion field a planner needs is regime-dependent and identifiable
  (necessity ≠ "store everything").
- **Status / what I have measured on real nuScenes val:** an oracle-free non-identifiability witness
  holds; a graded IDM (Intelligent Driver Model) surrogate shows **velocity is action-REDUNDANT in safe
  vehicle-following (n=443, clean, CI below a shuffled-velocity control).** The complementary
  "necessary when dangerous" half is **untestable on nuScenes** because the dataset is benign — I
  measured only ~22 / 2114 lead-frames at time-to-collision < 2s, and ~0 genuine fast-closing
  near-misses (lead closing speed median +0.07 m/s; the leads travel *with* the ego, not at it).
- **BLOCKERS (two, both confirmed this week):**
  1. **Danger is never a downloadable label.** I surveyed ~61 datasets. Real-crash data exists only as
     monocular dashcam RGB (DAD/CCD/DoTA/DADA/MM-AU) which cannot yield a metric top-down occupancy +
     per-object velocity. Real 3D LiDAR datasets (nuScenes/Waymo/AV2/KITTI/ZOD) are crash-free *by
     design* (the AV fleet avoids collisions; perception sets are curated for diversity). The two
     corpora are disjoint and cannot be merged (you cannot retro-fit LiDAR onto a dashcam crash). Even
     Argoverse-2 and Waymo **replicate a split**: occupancy GT and per-object-velocity / trajectory GT
     live in *separate, non-overlapping* scene pools (AV2 Sensor has LiDAR+boxes but no velocity field;
     AV2 Motion-Forecasting has velocity but no released LiDAR — confirmed by the Argoverse team; Waymo
     Occ3D is on 2,030 Perception scenes while velocity is on 103k Motion scenes, and WOMD-LiDAR covers
     only a 1-second history window).
  2. **A surrogate must match the conflict GEOMETRY of its substrate, or the measurement is a category
     error.** My IDM surrogate is *longitudinal car-following*. The one pre-mined real near-miss artifact
     I found (a TU-Delft conflict subset of Argoverse-2, 21,431 scenarios) is **100% lateral
     intersection-crossing** conflicts — running a car-following IDM on crossing geometry fabricates a
     fictitious lead and measures noise. Crossing conflicts need a different surrogate
     (gap-acceptance / time-to-conflict-point yield-go), which I have not built.
  3. **"Necessity" (vs "action-sensitivity") needs a closed-loop quality oracle** = a real planner
     (e.g. PDM-Closed) whose collision score grades the ablation, which is GPU-gated; and if I *generate*
     the danger AND grade necessity on the *same* simulator, the oracle is not independent (self-built-sim
     circularity).
- **Current plan (please critique):** pair the existing IDM with a *following*-conflict substrate
  (HighD / NGSIM highway cut-ins, which have dense labeled cut-in events — the danger where lead velocity
  matters) to measure "does velocity change the action in *dangerous* following," gated on a cheap
  danger-density count first. Treat occupancy-grounded and crossing-conflict versions as separate, later,
  gated experiments.
- **Questions:** Is there a public dataset with **real LiDAR/occupancy + per-object velocity + elevated
  *following*-conflict (hard-braking / cut-in) density** in one coherent split? What is the cleanest
  **non-circular** closed-loop necessity oracle — which simulator + released planner pairing, and how do
  people keep the danger generator independent from the necessity grader? Are HighD/NGSIM the right
  longitudinal substrate, or is there something better with sensor data? Any published **near-miss /
  surrogate-safety (TTC/DRAC/PET) mining** on a 3D AV dataset that I can reuse rather than re-derive?

### Stage 3 — value-of-correction (curate / valuation) — PLAN
- **Hypothesis:** value-ranked label fixes beat error-count ranking per fixed budget (value ≠ error
  count) — i.e., fixing the *most decision-relevant* wrong labels first beats fixing the *most numerous*.
- **BLOCKER:** the publishable tier is **GPU-gated** — to show "these corrections were worth more" you
  must measure a **downstream task delta** (retrain/fine-tune a model on corrected vs uncorrected labels
  and compare), and the title claim ("value ≠ error count") is currently unsourced. I also need a
  principled, non-circular **label-error injection harness** and a **value oracle** (what makes a label
  "valuable" independent of how I rank it).
- **Questions:** What is the state of the art in **data-valuation / active-correction** (influence
  functions, Datamodels, Shapley-style data value, coreset/curation) that gives a *downstream-value*
  signal **without** full retraining, or with cheap proxy models? Is there a public AV labeling-error or
  label-quality benchmark? How do people define "label value" independently of the error count?

### Stage 4 — asof (render→state) — PLAN
- **Hypothesis:** converting a *render* representation (3D Gaussian Splatting / NeRF) into a queryable
  occupancy state **preserves the action-relevant spatial signal** (free space, lateral clearance,
  corridor width) — and a volumetric *overlap* score (mIoU / RayIoU) is NOT a sufficient proxy for that
  preservation.
- **BLOCKER:** the **render substrate is unconfirmed** — I need per-scene 3DGS/NeRF reconstructions of
  driving scenes to even have a "render" to convert, which is ~hours of GPU per scene
  (DriveStudio / StreetGaussians / OmniRe class methods). And the **field-preservation metrics** (how to
  quantify "the predicate's answer survived the render→state conversion, beyond RayIoU") are undefined.
- **Questions:** Are there **released, downloadable 3DGS/NeRF reconstructions of public driving scenes**
  (so I avoid the per-scene GPU reconstruction)? What is the cheapest reliable driving-scene
  reconstruction pipeline on a single GPU? How have people quantified **property/measurement preservation**
  (not overlap) across a representation conversion — any metrics beyond IoU-family?

### Stage 5 — gt-distrust (visibility) — PLAN, rated weakest, currently UNSOUND
- **Hypothesis:** occlusion geometry predicts *which* occupancy ground-truth labels are untrustworthy
  (beyond a binary visibility mask).
- **BLOCKER (sound-ness):** the proposed oracle is **affirming-the-consequent / circular** — it uses
  occlusion to *predict* untrustworthiness and then validates with an occlusion-derived notion of truth.
  I need an **independent** signal for "this GT label is actually wrong," not derived from the same
  occlusion geometry.
- **Questions:** How can one obtain an **independent label-trustworthiness signal** for occupancy GT —
  e.g. multi-traversal disagreement, future-frame reveal (a voxel later observed directly), a second
  modality, or human re-annotation? Is there prior work on **occlusion-conditioned label-noise** in 3D
  perception with a non-circular evaluation?

### Stage 6 — vis-calibration (uncertainty) — PLAN
- **Hypothesis:** an occupancy model's confidence is *more over-confident where the sensor never
  observed* (unobserved regions).
- **BLOCKER:** the unobserved-stratum ground truth is **circular by definition** — to check calibration
  in never-observed voxels you need truth there, but they are unobserved. I need an independent source of
  truth specifically for regions the sensor did not see at capture time.
- **Questions:** What is the standard way to obtain ground truth for **never-observed** regions —
  future-frame observation, multi-traversal accumulation, a second sensor with different FOV, or HD-map
  priors? Any work on **calibration stratified by observability** in 3D occupancy / depth?

---

## 6. What I want from you

For each stage and for the cross-cutting problem:
1. The **state-of-the-art method(s)** to obtain a genuinely independent (data-source + algorithm)
   oracle, or to break the circularity — with the key papers (2022–2026) and their evaluation design.
2. **Public datasets / released artifacts / tools I may have missed** that carry the needed property
   (independent free-space truth; LiDAR + velocity + following-conflict density; released driving 3DGS;
   multi-traversal / future-reveal for unobserved-region truth; AV label-quality benchmarks).
3. **Minimal-GPU or no-GPU paths** to the publishable tier of the GPU-gated stages (2 necessity, 3
   downstream-value, 4 render substrate).
4. Where any of my proposed solutions (esp. dynfield's HighD+IDM plan, and any oracle above) are
   **still circular or unsound**, say so and explain why.
5. Relevant **benchmarks, challenges, or research groups** working on measurement-honesty for 3D
   occupancy / free-space, whose protocols I could adopt.

Please cite primary sources (papers, dataset pages, code repos) and distinguish what you confirmed from
a source vs what is your inference.
