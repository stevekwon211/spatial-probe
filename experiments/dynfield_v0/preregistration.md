# dynfield v0 — pre-registration (2026-06-22)

Committed BEFORE any real-data necessity measurement. [research-integrity.md](../../docs/research-integrity.md)
applies: changing a hypothesis or a cutoff after seeing data is HARKing and shows in this file's git
history. Full plan: [docs/research-program/dynfield.md](../../docs/research-program/dynfield.md).

**The question.** Which stored motion field does a planner actually need, and in which regime?
occquery (Stage 1) proved an occupancy predicate MEASURES a static fact but, by its own scene-0061
lead-car finding, cannot tell a passable gap beside a following lead car from an impassable gap beside
a wall — that verdict needs relative motion over time. dynfield is the necessity question one layer up.
The headline contribution is a {field × regime} necessity/sufficiency MATRIX, not an accuracy number.

**Hardware honesty (shapes the whole plan).** Mac arm64, no CUDA. The PLAN's nuPlan/NAVSIM closed-loop
anchor is GPU-heavy and is NOT done from a Mac. The claim is split: a Mac-feasible core (counterfactual
necessity on a deterministic analytic planner-surrogate over Occ3D-nuScenes + tracked-box motion, pure
numpy) and a GPU-gated stretch (the nuPlan closed-loop reproduction), never presented as done here.

## Decisions sealed (2026-06-23) — supersede any looser wording below

1. **Field lattice (richer):** {static-occupancy-only} → {+per-object velocity} → {+occupancy-flow}
   (flow = per-voxel 2D motion from consecutive GT occupancy by a fixed numpy procedure, provenance
   argued non-circular). Velocity ships first; flow is the second field in the same matrix.
2. **Framing = ACTION-SENSITIVITY, not necessity.** On a Mac there is no planner-quality oracle, so
   Tier-1 measures whether removing a field MOVES vs LEAVES the deterministic surrogate's action
   (action-sensitivity / action-equivalence). The word "necessary/sufficient" is reserved for Tier-2
   (a real closed-loop planner whose score is a quality oracle). SH2/SH3 are action-sensitivity /
   action-equivalence below.
3. **Data = nuScenes val (already on disk).** Occ3D val occupancy is in `data/gts` (850 scenes =
   train 700 + val 150) and box metadata in `data/nuscenes/v1.0-trainval` — both present from the
   earlier extraction; NO new download for Tier-1 (raw camera/LiDAR val is absent but unused at
   Tier-1). The official 150-scene val split is the held-out set, sealed in `held-out.txt`.
4. **Primary metric = dimensionless decision-FLIP** (brake/proceed flip rate) — regime-comparable,
   defuses the cross-regime scale-confound. Continuous decel-delta is a SECONDARY internal-validity
   curve under a per-regime band, never the cross-regime headline.
5. **Tiers: ship Tier-1 (Mac) first**, then Tier-2 (GPU/RunPod, PDM-Closed anchor) only after Tier-1
   holds and the PDM metric/split is primary-confirmed.

## v2 — graded IDM surrogate, danger-stratified (sealed 2026-06-23, BEFORE its run)

v1's binary brake/proceed surrogate was too coarse (shuffled control failed: velocity flipped the
decision on 3.3% of frames, below the 8.5% shuffled floor) -- on real, mostly-safe nuScenes following,
distance dominates and a binary decision cannot surface velocity's effect. v2 fixes the SURROGATE and
the REGIME axis, committed before running so the change is not fit to the v1 hint:

- **Graded surrogate = IDM** (the planner PDM-Closed itself uses, so it ties to Tier-2). The action is
  a CONTINUOUS commanded deceleration. static-only = IDM with the closing-gap term Δv set to 0 (gap
  only); motion-aware = full IDM (the lead's relative velocity enters via the Δv interaction term).
  The ablated field's effect is the decel-delta = |a_motion − a_static|, graded, not a binary flip.
- **Primary metric = decel-delta** (continuous, regime-comparable as a magnitude) with a per-regime
  bootstrap 95% CI and a SHUFFLED-velocity control (true velocity's decel-delta must beat a permuted
  velocity's, else spurious). Decision-flip is demoted to a secondary descriptor (v1 showed it is too
  coarse here).
- **Regime axis = danger, cut NON-CIRCULARLY** (the ablated field must not define the regime):
  AGENT-CONTEXT (class/geometry: following / crossing / vru -- velocity-independent) × a STATIC-URGENCY
  band = ego_speed² / (2·gap) (a surrogate-safety danger proxy using EGO state + gap only, NOT the
  lead's velocity). Low urgency = safe, high urgency = near-miss-ish. Standard TTC/DRAC is NOT used as
  the cut because it is computed from the ablated velocity (circular).
- **Gates (re-specified for the graded action, run FIRST):** surrogate-validity = the IDM decel rises
  with closing speed and falls for a receding lead (monotone response to the field it should use); SH4
  leakage (unchanged, passed at 0.184); shuffled-velocity control on decel-delta.
- **Kills:** decel-delta EQUIVALENT (CI within the shuffled band) in EVERY regime → velocity is
  action-redundant for this surrogate on real data, report the negative. Non-uniform (changing in
  high-urgency/crossing, equivalent in low-urgency/following) → the by-regime action-sensitivity
  result. "Necessary" still reserved for Tier-2.

## Sub-hypotheses

- **SH1 — necessity EXISTS (oracle-free headline).** A non-identifiability witness: two frames with
  IDENTICAL static occupancy but DIFFERENT stored motion force a static-only surrogate to act
  identically, while a motion-aware surrogate differs → a static-only stored state is insufficient to
  determine the action in at least one regime. Oracle-free, by construction. **Tier-0 status: HOLDS**
  (`tests/test_dynfield_witness.py`, 4 tests). Scope bound: non-identifiability under the surrogate's
  static-occupancy observable set, not a claim about every planner.
- **SH2 — regime-dependence (the matrix).** Per pre-registered regime (lead-following vs free-flow;
  interactive-agent vs none; static-only vs dynamic; low vs high ego speed), the counterfactual action
  change from ablating each stored field is non-uniform: ≥1 field NECESSARY in ≥1 regime AND REDUNDANT
  in ≥1 other. A uniform matrix FALSIFIES the by-regime thesis.
- **SH3 — parsimony.** In ≥1 regime a strict motion-field subset is SUFFICIENT (reproduces the
  full-state action within a pre-registered no-effect band) while a smaller subset is not.
- **SH4 — harness non-leakage (gate, runs FIRST).** The no-dynamics control MUST collapse where
  dynamics are definitionally required; if it does not, motion state is leaking and SH1–SH3 are void.
  **Tier-0 status: HOLDS** (the static-only surrogate proceeds where the closing case requires braking).

## Tiers (pre-registered)

- **Tier 0 / SMOKE (Mac, NOT a result):** the SH1 witness + SH4 control hold as executable tests on
  hand-built motion scenes. **Done** — by construction, never quoted as evidence.
- **Tier 1 / publishable-on-Mac:** the {field × regime} matrix on Occ3D-nuScenes with a deterministic
  analytic surrogate (pure numpy), SH1 holding on ≥1 REAL mined scene-pair, per-cell counterfactual
  distributions with bootstrap 95% CIs, pre-registered cutoffs as curves, SH4 passing, released
  harness + held-out IDs. Load-bearing = the matrix is NON-UNIFORM (necessary-here vs redundant-there),
  never an absolute action-delta. **First build step + hard dependency:** populate per-object velocity
  in the Occ3D adapter from nuScenes `sample_annotation` (currently `TrackedBox.velocity` defaults to
  (0,0) — the box loader sets position/size/yaw but not velocity yet).
- **Tier 2 / STRONG (GPU-gated, deferred):** re-run the ablation against a real released closed-loop
  planner (PDM-Closed on nuPlan, metric/split primary-confirmed before citing). OUT of scope on a Mac.

## Controls (pre-registered)

- **static-occupancy-only baseline** — the load-bearing naive baseline (dynfield analogue of occquery's
  box-only RefAV baseline); necessity is the RELATIVE gap over it.
- **motion-field-shuffled control** — a surrogate fed a permuted motion field; the TRUE field must beat
  the shuffled one, else the "necessity" is spurious (affirming-the-consequent antidote).

## Circular-oracle guards

- Per-object velocity comes from the dataset's tracked-box motion, NOT re-derived from the occupancy the
  surrogate also reads.
- The ablated field must NOT be the signal the regime partition is cut on (else necessity is definitional).
- Full-state reference and ablated surrogate differ in EXACTLY the field under test (one-factor isolation).
