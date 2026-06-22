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
