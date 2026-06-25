# Premise Verification — box-query denotation gap + expressivity-not-accuracy (adversarial re-check, 2026-06-25)

Permanent record. Adversarial re-check of two load-bearing pitch premises plus one external
fact, after a hostile critic pass. Verdicts are scoped to what is **machine-verified or
content-verified against a primary source**; optimism is suppressed in favor of accuracy. Every
file path is absolute. arXiv ids and confidence are marked inline.

---

## Bottom line (honest)

Both premises survive **as scoped facts, not as the headline sentences currently written.** The
underlying work is real: the H1 expressivity witness is a valid by-construction non-identifiability
proof (8/8 tests pass, machine-reproducible), RefAV's released set genuinely has zero free-space
primitives (32 defs, grep-verified at source), the R0 action-equivalence result reproduces from raw
AV2 LiDAR to the decimal, and the IROS Track 5 / Visionary external fact is content-verified against
the primary arXiv source. Three independent prior-art finders converged that the *free-space-native
scenario-mining query* is genuinely open.

What does **not** survive is the unscoped phrasing the writeups currently use. Three corrections are
load-bearing for the meeting, and each is a place a reviewer can land a puncture **using facts this
dossier already verified**:

1. **Premise 1 must be scoped to "non-identifiability over a fixed observable signature," demonstrated on the BOX channel** — not "structural / permanent / no amount of data," not "box+map" (the map channel is equalized only vacuously — empty both sides), and **never** "because the box language lacks transitive closure."
2. **Premise 2 must be scoped to "action-equivalent on the ONE shared longitudinal query through a fixed IDM, no accuracy advantage relative to the box as reference"** — not "no accuracy advantage on shared queries generally."
3. **The single biggest exposure is the L5 *collapse* attack** ("a clearance predicate over a grid you already have is a half-day of engineering"). It is **not currently rebuttable from built artifacts.** The witness/proof is present; the durable answer — a released denotation-correctness **benchmark** — is H3, which the repo itself marks **DEMOTED / "not-a-result." (`/Users/doeonkwon/Projects/Personal/spatial-probe/CLAUDE.md`)** Split witness (present) from benchmark (unbuilt) everywhere, and state the collapse attack as an open conditional.

Walk in leading with the **proof + the measurement-honesty / benchmark-as-next-deliverable** framing.
Do **not** lean on H1's novelty, on any "structural/permanent/no-accuracy-advantage" wording, or on
the transitive-closure analogy — each can be corrected on the spot with facts already in this dossier.

---

## Premise 1 — "Box-query languages cannot express free-space hazards: a denotation gap, not a coverage/data gap"

**VERDICT: HOLDS-WITH-CAVEATS (high).** The gap is real and proven by construction; specific
headline words ("structural / permanent / no amount of data," "box+map," the transitive-closure
analogy) over-reach and must be scoped.

### Strongest surviving form (use this exact wording)

> "On RefAV's released function set (32 atomic functions over tracked cuboids + static HD-map
> polygons, arXiv:2505.20981), free-space / clearance / corridor predicates are **non-identifiable
> over the box observable channel**: we exhibit two scenes whose **box observables are byte-identical**
> (object center / size / yaw / class / velocity — the same shared Python objects), differing only by
> unboxed occupancy, so every function of that channel is forced to return the same answer while an
> occupancy predicate separates them. The **map channel is equalized only vacuously (empty on both
> sides)** — `Frame` carries no map field — so RefAV's 8 HD-map functions (`in_drivable_area`,
> `near_intersection`, …) are **not yet defeated**; that requires a mapped-curb witness. This is a
> denotation gap **relative to a fixed observable signature**, not a detection-difficulty gap, and no
> amount of data closes it **so long as the obstacle stays unboxed**."

Every clause is machine-verified:
- **8/8 witness tests pass** (`/Users/doeonkwon/Projects/Personal/spatial-probe/tests/test_expressivity.py`, run on Python 3.14.3, ~0.3s).
- The tie is **logical necessity, not empirical near-miss**: the `TrackedBox` and `EgoPose` are built once and shared into both scenes (`box is box` → True); the box function (`distance_to_nearest_object` in `/Users/doeonkwon/Projects/Personal/spatial-probe/src/probe/predicates/objects.py`) reads only `o.center` over the *identical* object, so referential transparency forces equality for ANY box-only function, not just the one the test asserts (I1, I4).
- **Zero free-space primitives** in the released RefAV set: `grep -cE '^def '` on the live `refAV/atomic_functions.py` (fetched from `main`) = **32**; `grep -iE "occupancy|voxel|free.?space|clearance|corridor|swept"` → nothing (I2, L1, L2, L3, L5 each independently re-ran this).

### The strawman rebuttal (this is the load-bearing move)

**The threat (L5):** "Production AV stacks (Tesla Occupancy Network, Mobileye REM, NVIDIA RadarNet)
already output free-space / occupancy alongside boxes — so 'box-only' is a strawman." The factual
premise is **TRUE and fully citable** — occupancy grids date to Moravec & Elfes 1985; Tesla's OccNet
exists specifically for unboxed long-tail obstacles; nuScenes / AV2 ship drivable-area layers.

**The rebuttal — do NOT defend the literal reading.** The objection equivocates between *computing*
free-space at runtime and *querying / composing* over a persisted free-space field for **scenario
mining / data retrieval**. Production free-space is a **runtime intermediate consumed immediately by
the planner's cost-map** (industry survey + ONRAP arXiv:2602.13577) — never exposed as an indexed,
composable predicate you can write `min_free_width < 0.5 m` against to mine a corpus. RefAV — the
actual scenario-mining query language — has **zero** free-space primitive. So the pitch is **not**
"we invented free-space perception" (fatal if stated that way); it is "the free-space the stack
already computes is **not queryable / composable for retrieval**, and box+map query languages are
non-identifiable over it." Pitched correctly, Tesla OccNet becomes **supporting evidence** (the
substrate exists and is valuable; the gap is the query layer).

**The deadlier follow-up — flag it for the meeting (see NET):** the **collapse attack** — "a
clearance predicate over a grid you already have is a half-day of engineering, not a contribution;
RefAV lacks it only because the academic benchmark chose AV2 boxes." This concedes every verified
fact and reframes the gap as a benchmark-scoping accident. It is the single biggest exposure in the
whole pitch and is **not currently rebuttable from built artifacts** (NET).

### What you must NOT say (caveats that demote it from clean HOLDS)

Drop the unconditional phrasings the writeups currently carry — "no composition of the 32 functions
can EVER," "Permanent," "structural fact, not an empirical score," and the B&W-camera analogy *as
literally what is happening*. Two defects (I4, L4; the repo's own review at
`/Users/doeonkwon/Projects/Personal/spatial-probe/docs/research-program/occquery.md` L100 makes the
same point):

1. **Fixed-function-set ≠ fixed-observable-ontology.** "What gets boxed" is a detector / training
   choice. Train a detector to box walls / curbs / debris and `free = drivable_polygon − union(boxes)`
   becomes composable — the gap shrinks on the coinciding-object subset. The writeup itself concedes
   the seam ("may NARROW where a boxed object coincides"). A gap that shrinks as coverage improves is,
   on that axis, partly a coverage gap. So the bulletproof claim is **non-identifiability over a frozen
   observable channel**, NOT camera-grade physical impossibility.
2. **Demonstrated scope is the BOX channel, only vacuously the MAP channel.** Confirmed at source:
   `Frame` carries only `grid / ego / time / objects`
   (`/Users/doeonkwon/Projects/Personal/spatial-probe/src/probe/scene.py` L40-45) — there is **no map
   field**. The "identical map" is vacuously true (empty both sides), asserted in a comment, never
   exercised (test docstrings concede this, L90-92). So the witness has not defeated RefAV's HD-map
   functions (`in_drivable_area`, `near_intersection`, `at_pedestrian_crossing`, `on_lane_type` — 8 of
   32). Say "box-channel non-identifiability," not "box+map," unless you add a mapped-curb witness.

### Novelty vs prior art (three independent finders converged)

**Genuinely new, survives prior-art search:** No released system queries a **scene corpus by a
free-space / clearance / corridor predicate over UNBOXED occupancy**. The discriminating axis is
**free-space-native ≠ voxel-native**:

- **L1 (HOLDS, high):** RefAV (arXiv:2505.20981) / Scenic-data-query (arXiv:2511.10627, arXiv:2112.00206) / nuPlan / Waymo are all object / box / map-bounded retrieval languages — exactly the gap.
- **L2 (REFUTED → favors the pitch, high):** OccNet (arXiv:2306.02851) / UniAD / OccWorld / QuAD (arXiv:2404.01486) / Implicit-Occ-Flow (arXiv:2308.01471) demonstrate "occupancy helps a *learned planner*" (proposition A) — a different, weaker, planner-coupled claim spatial-probe explicitly disavows. None exposes a falsifiable symbolic predicate or an expressivity proof.
- **L3 (HOLDS, high):** voxel-native NL systems (Talk2Occ / GroundingOcc arXiv:2508.01197, POP-3D arXiv:2401.09413, OccLLaMA arXiv:2409.03272) ground language to **occupied voxels of named entities** ("where is the object"), never "how much free width remains." Free-space-native NL query is open.
- **ESDF / costmap clearance (Voxblox / FIESTA arXiv:1903.02144 / nvblox arXiv:2311.00626) is the genuine prior art for the *quantity*** — but it is a per-pose planning lookup, never a corpus-retrieval language. It refutes a broad "first to query free space" claim, so **never say that**; scope to scenario-mining query languages.

### Formal foundation (L4 — WEAKENED, high) — what the document is allowed to claim

The **citable foundations are real and top-tier**, but the project must cite the **technique, not the
theorem-as-result**. L4's finding: the foundations exist, but the specific analogy the project wants
— "free-space inexpressibility ≈ connectivity-needs-transitive-closure" — is **WEAKENED because it is
analogy-of-method, not identity-of-result.**

- The classic results (Codd's theorem; transitive-closure-not-FO, Fagin 1974 / Aho-Ullman 1979; connectivity-not-FO, Gaifman-Vardi 1985; **Libkin, *Elements of Finite Model Theory*, Thm 13.25** "topological connectivity is not expressible in FO(R,σ)") are a **missing-operator** deficiency: the edge relation **is** in the schema; FO merely lacks recursion.
- The box-language gap is a **missing-signature / observability** deficiency: the obstacle is **absent from the schema**, so the box language ties **even with transitive closure added**. Different impossibility mechanism.
- The headline predicates (`lateral_clearance`, `min_free_width_along_path`, `free_along_ego_path`) are **local / metric, not connectivity-shaped.** The **only** genuinely TC-shaped predicate is `reachable_free_field`'s 8-connected flood-fill — source-confirmed: `ndimage.label(free, structure=np.ones((3, 3)))` at `/Users/doeonkwon/Projects/Personal/spatial-probe/src/probe/predicates/reachable.py` L131 — and the **H1 witnesses never exercise it** (they exercise the three metric predicates, test L21-23, L66-174).

**Therefore, the formal home for H1 is non-identifiability over a fixed observable signature
(querying-incomplete-information / certain-answers), proved by the EF-game / locality
indistinguishability technique — for which connectivity-not-FO is the textbook *exemplar of the
method*, NOT an identity of result.** Reserve the literal transitive-closure analogy for
`reachable_free_field` only. **Never say the box language fails "because it lacks transitive
closure."** Cite the four results as *the technique*, not as grounding-by-result.

---

## Premise 2 — "Occupancy expands the queryable space but is action-equivalent on shared queries; the value is expressivity, not accuracy"

**VERDICT: WEAKENED (high).** The within-experiment result is solid and reproduced; the *synthesized
causal sentence* over-generalizes one shared query through one lossy planner into "no accuracy
advantage on shared queries."

### Strongest surviving form (use this exact wording — it survives every attack)

> "On the **single shared longitudinal-lead query**, occupancy's in-path free-distance is
> **action-equivalent** to the box lead **through a fixed IDM** — 81% Gate-2 agreement, action-delta
> 0.0216 m/s², decisively beating a shuffled-occ null (17.8× larger) — and shows **no accuracy
> advantage relative to the box as reference**. So on THIS substrate occupancy's distinctive value is
> **expressivity** (queries boxes cannot pose), with **accuracy-vs-external-truth and all other shared
> queries (lateral clearance, time-headway, multi-object min-gap) untested**."

The numbers reproduce from raw AV2 LiDAR (1347 frames; `true_dd=0.0216`, Gate-2 = 81.0% — I3 re-ran
it and matched to the decimal). The integrity scaffolding (gates, shuffled null, per-log clustering,
two adversarial reviewers, v1/v2/v3 bug history) is real and the negative is honest.

### The overgeneralization caveat (I3 — three compounding ways)

1. **"Shared queries" = exactly ONE query, and it is the IDM's own scalar input.** "Action-equivalent"
   reduces to "occ_gap ≈ box_gap survives one saturating 1-D transfer function." R0-v3's own
   pre-registration disclaims generality ("NOT 'structure boxes cannot express' globally",
   `/Users/doeonkwon/Projects/Personal/spatial-probe/experiments/dynfield_v0/r0v3_preregistration.md`
   L43-45). Lateral clearance, time-headway, gap-to-crossing-box — all box-expressible, all untested.
2. **"EQUIVALENT" is action-invariance through a lossy planner, ≠ accuracy.** On the 291 disagreeing
   frames (21.6%) the mean geometric disagreement is 4.67 m yet the action-delta is only 0.080 m/s² —
   two representations can be action-equivalent while differing in accuracy whenever the gap lands in
   the planner's insensitive region. Action-equivalence is the *weaker* property; it cannot upgrade to
   "no accuracy advantage."
3. **"NOT accuracy" is absence-of-evidence.** The accuracy leg (H3 denotation P/R/F1 vs an external
   oracle) is **demoted and explicitly unmeasured** ("the 10-scene substrate has no power"). The one
   accuracy-flavored number (Gate-2, 81%) uses **the box as reference, not LiDAR truth** — it
   presupposes the box is correct and structurally cannot test "occupancy-sees-more" (no unlabeled
   obstacles on AV2 by construction).

**The counter-threat that partially rescues it (state it honestly):** the IDM is not an arbitrary
filter — it is the canonical longitudinal planner (PDM-Closed's transfer function), so "a geometric
error that never changes any action a real planner takes is operationally not an accuracy deficit that
matters." This **lands for the longitudinal action** and explains why the authors frame it as
action-sensitivity — but it (a) cannot extend to lateral / free-space / multi-object shared queries
the word "generally" sweeps in, and (b) the high-urgency danger cell that would stress it most is
L=3, "suggestive not decisive" by the authors' own retraction. So even the longitudinal claim isn't
powered where it bites hardest.

**Net for Premise 2:** the *expressivity-vs-accuracy split as a research frame* is clean and
well-grounded in the program's own scaffolding (`/Users/doeonkwon/Projects/Personal/spatial-probe/PLAN.md`
L128-160). What is WEAKENED is the specific sentence "R0 EQUIVALENT ⟹ value is expressivity not
accuracy on shared queries": R0 supplies action-equivalence on *one* shared query through a lossy
planner, which is *necessary but not sufficient* for the general claim, and the "not accuracy" half
rests on an unmeasured leg. Use the scoped restatement above and the premise is a fact.

---

## External fact — IROS / RoboSense Track 5 / youngseok_kim / 58.54% (E1, L6)

**CONFIRMED (high), verified by content against the primary source** (arXiv:2601.05014v1, *The
RoboSense Challenge*; de-tagged raw HTML grepped directly — the WebFetch summarizer was inconsistent
on the Visionary linkage, so it was bypassed).

Safe to state verbatim in the meeting:

- **Track 5 = Cross-Platform 3D Object Detection** (train on vehicle LiDAR, generalize unsupervised
  to drone + quadruped). Base detector PV-RCNN, baseline ST3D/++. **Strictly 3D bounding boxes** —
  metric 3D AP@0.5 (R40), mAP / NDS. **Zero occupancy / free-space component** in Track 5 (or any of
  the 5 tracks). Corroborates that the surrounding AV-perception ecosystem is box / detection-framed.
- **Handle `youngseok_kim` placed 1st, mAP 58.54** (2nd `castiel972` 55.66, 3rd `linyongchun` 55.61).
  Exact to the decimal (Table 5; restated in body: "phase-1 mAP 66.94, phase-2 mAP 58.54").
- **Attributable to Team Visionary** — confirmed via Appendix 7.4.5: Team [Visionary], member
  Youngseok Kim, affiliation Visionary Inc. + KAIST, email `youngseok.kim@visionary.run`, champion
  method "DataEngine." The bare leaderboard handle and the registered team are the same entity.

**Safe meeting wording (do not overstate):**

- Visionary **competed in and won** Track 5 (3D box detection). Cite the 58.54 champion row and the
  `visionary.run` attribution. **Do NOT claim Visionary *organized* the track** — E1 could not
  independently confirm that.
- Don't cite the 2nd-4th attributions: one internal cross-report labeling quirk exists among those
  teams (arXiv:2601.08174 self-labels 3rd-place with scores matching the runner-up / 4th). It does
  **not** touch the champion row; the 58.54 champion fact is clean.
- **Visionary / PRISM product framing (E1 + L6, both HOLD, high):** public materials describe PRISM as
  an AD **dataset-construction / scene-curation + labeling** pipeline (LiDAR + camera → training
  datasets, human-in-the-loop "data flywheel"); **zero** occurrence of occupancy / free-space /
  drivable-area across EN + KR sources. The "the denotation-gap pitch applies to them" point is
  **absence of published occupancy-query capability**, **not** proof they are internally box-only —
  marketing blurbs are not a spec sheet, and a "Spatial Data OS" for E2E / VLA could already emit
  occupancy labels. Frame it as *no published evidence of occupancy querying*; a reviewer who has seen
  their internal stack can puncture an "internally box-only" claim.

---

## What to change in the pitch (concrete)

1. **Premise 1 headline → scope to the box channel and the observable signature.** Replace any
   "box+map" / "structural" / "permanent" / "no amount of data" / "no composition ever" wording with
   the *Strongest surviving form* box above. The phrase "box+map" must not appear without the vacuity
   qualifier attached **in the same sentence**. Either add a mapped-curb witness or say "box-channel
   non-identifiability" in the headline itself.
2. **Drop the transitive-closure analogy from the headline.** Ground H1 on the **EF-game / locality
   indistinguishability technique** + querying-incomplete-information / certain-answers. Cite Codd /
   Fagin-Aho-Ullman / Gaifman-Vardi / Libkin Thm 13.25 as *the technique*, not as grounding-by-result.
   Reserve the literal TC analogy for `reachable_free_field` only. **Never** say the box language fails
   "because it lacks transitive closure."
3. **Premise 2 → scope to the one longitudinal query.** Use "action-equivalent on the *one* shared
   longitudinal query through a fixed IDM, no accuracy advantage *relative to the box as reference*."
   Never say "no accuracy advantage on shared queries generally." Name the untested shared queries
   (lateral clearance, time-headway, multi-object min-gap) and the unmeasured accuracy leg out loud.
4. **Split witness (present) from benchmark (unbuilt) everywhere.** The oracle-free **witness / proof**
   (3 families, 8/8, machine-reproducible) is a present asset and addresses non-identifiability. The
   **benchmark** (H3 denotation P/R/F1 vs an external oracle) is **unbuilt** — the repo marks it
   DEMOTED / "not-a-result"
   (`/Users/doeonkwon/Projects/Personal/spatial-probe/CLAUDE.md`). Never say "benchmark + witness" as
   one present asset.
5. **State the L5 collapse attack as an OPEN conditional, not an answered one.** Honest meeting
   position: "the proof is done; the benchmark that makes the predicate a contribution rather than a
   half-day script is the **stated next deliverable, not a present claim** — and depends on H3 being
   built and validated against a data-source-independent oracle." This is the single biggest exposure;
   lead with the proof + measurement-honesty framing, not with H1 novelty.
6. **External fact:** cite Track 5 = 3D box detection, the 58.54 champion row, the `visionary.run`
   attribution. Do not claim Visionary organized the track; frame "the pitch applies to them" as
   *absence of published occupancy-query capability*.
7. **Cosmetic (non-blocking) cleanup:** `queries.yaml` carries a stale "33-function" string (line 10 +
   rationale strings) where the released RefAV set is **32** (I2, grep-verified). 2-line doc-drift; fix
   `33 → 32` but it does not touch any headline claim.

---

## NET — honest bottom line

Both premises survive **as scoped facts, not as the headline sentences currently written.**

- **Premise 1 (denotation gap)** is real, machine-verified (8/8 witnesses; 32-def grep; three
  independent prior-art finders converging), genuinely novel as *free-space-native scenario-mining
  query* (not voxel-native, not planner-coupled) — **provided** it is scoped to "non-identifiability
  over a fixed observable signature, demonstrated on the box channel," and the unconditional words
  ("permanent / no amount of data / no composition ever / box+map") are dropped. On the formal
  foundation: the citable home for H1 is **non-identifiability over a fixed observable signature
  (querying-incomplete-information / certain-answers), proved by the EF-game / locality
  indistinguishability technique — for which connectivity-not-FO is the textbook exemplar of the
  *method*, NOT an identity of result.** The literal transitive-closure analogy applies to **only one
  predicate, `reachable_free_field`, which the H1 witnesses do not exercise.** **Never say the box
  language fails "because it lacks transitive closure."** Cite Codd / Fagin-Aho-Ullman / Gaifman-Vardi
  / Libkin 13.25 as the technique, not the theorem-as-result.

- **Premise 2 (expressivity-not-accuracy)** is reproduced exactly and the frame is clean — **provided**
  you say "action-equivalent on the *one* shared longitudinal query through a fixed IDM, no accuracy
  advantage *relative to the box as reference*," not "no accuracy advantage on shared queries
  generally."

- **E1** is solid to the decimal and safe to cite, **with its source caveats attached**: Visionary
  **competed in and won** Track 5 (do **not** claim they organized it); the "pitch applies to them"
  point is **absence of published occupancy-query capability**, not proof they are internally box-only.

**The single biggest risk is not novelty or correctness — it is the L5 *collapse* attack:** "a
clearance predicate over an occupancy grid is a half-day of engineering; the gap is a benchmark-scoping
accident." The verified facts **concede** this, and it is **not currently rebuttable from built
artifacts.** The witness / proof (oracle-free, 8/8, machine-reproducible) is real and addresses
non-identifiability; but the durable answer to *collapse* — a released **denotation-correctness
benchmark with external validation** — is **H3, which the repo marks DEMOTED / unbuilt /
"not-a-result"** (`/Users/doeonkwon/Projects/Personal/spatial-probe/CLAUDE.md`). **Until H3 is built
and validated against a data-source-independent oracle, the honest meeting position is: "the proof is
done; the benchmark that makes the predicate a contribution rather than a half-day script is the stated
next deliverable, not a present claim."** Everywhere the pitch says "benchmark + witness" as a present
asset, split them: witness = present, benchmark = unbuilt.

Walk in leading with the **proof + the benchmark / measurement-honesty framing**. Do **not** lean on
H1's novelty, on any unscoped "structural / permanent / no-accuracy-advantage" wording, or on the
transitive-closure analogy — or the meeting can puncture it with facts this dossier already verified.

---

## Sources

### Repository files consulted (all under `/Users/doeonkwon/Projects/Personal/spatial-probe/`)

- `tests/test_expressivity.py` — H1 witness, 3 families, **executed: 8 passed** (~0.3s). Box-channel equalization, empty-map / box-channel scope conceded in docstrings (L90-92); metric predicates exercised (L21-23, L66-174).
- `src/probe/scene.py` — `Frame` data model, L40-45: fields are **`grid / ego / time / objects` only; no map field** (confirms map equalization is vacuous).
- `src/probe/predicates/objects.py` — `distance_to_nearest_object` reads only `o.center` over the shared box (the referential-transparency tie).
- `src/probe/predicates/reachable.py` — `reachable_free_field`, L131 `ndimage.label(free, structure=np.ones((3, 3)))` = **8-connected flood-fill** (the one genuinely transitive-closure-shaped predicate; **not** exercised by the H1 witnesses).
- `src/probe/predicates/clearance.py`, `src/probe/predicates/freepath.py` — occupancy-side predicates; never read `TrackedBox` / `objects_at`.
- `CLAUDE.md` — result framing: **"H1 … is the SOLE headline and holds"; "H3 … is DEMOTED to an internal-consistency check"; "Treat any H3/H2 number as not-a-result."**
- `PLAN.md` (L124-193, esp. L128-160) — expressivity-vs-accuracy split as the program frame.
- `docs/expressivity-vs-refav.md` — 32-function claim, syntactic/observational scoping (L51-62).
- `docs/occquery-writeup.md` — H1 framing, "non-identifiability under the box+map observation set."
- `docs/research-program/occquery.md` — repo's own adversarial review; L100: valid deduction is "box+map OBSERVABLES insufficient," not "no composition of the 32 functions."
- `docs/research-integrity.md`, `docs/benchmark-anchors.md`, `README.md` — scope / anchors / verification ledger.
- `experiments/occquery_v0/preregistration.md` — sealed H1 pre-registration (non-identifiability under RefAV box+map observation set; reachable kill criteria).
- `experiments/occquery_v0/queries.yaml` — `refav_expressible: false` annotations; **stale "33-function" string at L10 + rationale strings** (released set is 32 — cosmetic doc-drift).
- `experiments/dynfield_v0/` — R0 reproduction: `r0_danger.py`, `r0_action_sensitivity.py`, `surrogate.py`, `results/r0_danger.json` + `results/r0_action_sensitivity.json` + `results/r0_danger_summary.md` + `results/r0_summary.md` + `results/summary.md`; `r0v3_preregistration.md` (L43-45 disclaims generality); `r0_danger_preregistration.md`. Reconstruction re-ran the danger estimand on `data/danger/av2_sensor` (1347 frames), reproduced `true_dd=0.0216` / Gate-2 = 81.0%; IDM-saturation / shuffled-fairness / speed-stress adversarial tests all passed.

### Papers (arXiv ids)

- **RefAV: Towards Planning-Centric Scenario Mining** — arXiv:2505.20981 (NeurIPS 2025). Live-fetched `refAV/atomic_functions.py` from `github.com/CainanD/RefAV` `main`: **32 top-level defs, 0 free-space primitives** (grep exit 1).
- **Querying Labeled Time Series Data with Scenario Programs (Scenic data-query)** — arXiv:2511.10627, arXiv:2112.00206. Object-level labels; closest retrieval-language analog, still box/track-bounded.
- **OccNet / Scene as Occupancy** — arXiv:2306.02851 (ICCV 2023). Learned BEV safety cost; collision 15-58% lower vs box planning; no symbolic query interface (the strongest "occupancy beats boxes" threat — wrong proposition).
- **QuAD: Query-based Interpretable Neural Motion Planning** — arXiv:2404.01486. "query" = learned occupancy-field lookup; "interpretable" = inspectable cost, not symbolic predicate.
- **Implicit Occupancy Flow Fields** — arXiv:2308.01471 (Waabi/Waymo). Continuous (x,y,t) query is a learned NN evaluation.
- **Occupancy Flow Fields for Motion Forecasting** — arXiv:2203.03875 (Waymo RA-L 2022). Learned forecasting grid; no retrieval language.
- **OccWorld** — arXiv:2311.16038; **Drive-OccWorld** — arXiv:2408.14197; **OccLLaMA** — arXiv:2409.03272; **Occ-LLM** — arXiv:2502.06419; **SparseOccVLA** — arXiv:2601.06474 — occupancy world / language-action models; learned planners, not falsifiable retrieval predicates.
- **QueryOcc** — arXiv:2511.17221; **SparseWorld** — arXiv:2510.17482 — "query" = training-time self-supervision / DETR decoder queries, not a user retrieval predicate.
- **Talk2Occ / GroundingOcc (Coarse-to-Fine 3D Occupancy Grounding)** — arXiv:2508.01197. Grounds language to occupied voxels of a **named** entity (the strongest voxel-native NL threat — object-referential, not free-space).
- **POP-3D** — arXiv:2401.09413 (NeurIPS 2023). Open-vocab retrieval over **occupied** voxels.
- **LangOcc** — arXiv:2407.17310; **Language-Driven Occupancy** — arXiv:2411.16072 — open-vocab semantic labeling of occupied voxels.
- **NuScenes-QA** — arXiv:2305.14836; **NuScenes-SpatialQA** — arXiv:2504.03164; **STRIDE-QA** — arXiv:2508.10427; **SpaceDrive** — arXiv:2512.10719 — object / scene-graph spatial QA; no free-space measurement.
- **Talk2Radar** — arXiv:2405.12821; **LidaRefer** — arXiv:2411.04351; **Language-Guided 3D Detection** — arXiv:2305.15765 — referring-expression → single box; object-referential.
- **BEV-TSR (Text-Scene Retrieval in BEV Space)** — arXiv:2401.01065. BEV features + scene-graph semantics; no geometric free-space predicate.
- **Declarative Scenario-based Testing with RoadLogic** — arXiv:2603.09455. Symbolic (ASP/clingo) but over OpenSCENARIO **objects**; generates + monitors sim, not occupancy retrieval.
- **Spatial Retrieval Augmented Autonomous Driving** — arXiv:2512.06865 (CVPR 2026). Retrieves geographic map images/priors, not a free-space query language.
- **ESDF / clearance prior art:** **FIESTA** — arXiv:1903.02144; **nvblox** — arXiv:2311.00626 (per-pose planning lookup, not corpus retrieval). Voxblox (Oleynikova et al.).
- **General-obstacle / occupancy (L5 strawman support):** vision-based 3D occupancy review arXiv:2405.02595; general-obstacle dataset arXiv:2408.12322; Tesla Occupancy Network (CVPR 2022 / AI Day 2022); Moravec & Elfes 1985 "High resolution maps from wide angle sonar"; ONRAP arXiv:2602.13577 (planners consume occupancy as soft costs/priors); Grid-Centric Traffic Scenario Perception review arXiv:2303.01212; Waymo Open Motion arXiv:2104.10133 (SQL agent-relationship scenario predicates).
- **External fact:** **The RoboSense Challenge** — arXiv:2601.05014v1 (Table 5 leaderboard; Sec 4.5.1; Appendix 7.4.5 — Team Visionary / Youngseok Kim / `youngseok.kim@visionary.run`). Cross-report quirk: arXiv:2601.08174 (self-described 3rd-place report, scores align with runner-up/4th — does not touch champion row).

### Formal-foundation references (cite as the *technique*, not theorem-as-result)

- **Leonid Libkin, *Elements of Finite Model Theory* (Springer 2004)** — Prop. 3.1 (connectivity not FO-definable), Cor. 6.9, **Thm 13.25 (topological connectivity not expressible in FO(R,σ))**, Ch. 13.6 (constraint / spatial databases), intro attribution of TC-not-FO to Fagin 1974 / Aho-Ullman 1979. `https://homepages.inf.ed.ac.uk/libkin/fmt/fmt.pdf`
- **Codd's theorem** (relational algebra ≡ domain-independent relational calculus; TC and aggregation inexpressible) — `https://en.wikipedia.org/wiki/Codd's_theorem`.
- **Transitive closure inexpressibility / FO(TC)** (Fagin 1974; Aho & Ullman 1979) — `https://en.wikipedia.org/wiki/Transitive_closure`.
- **Gaifman & Vardi**, "A simple proof that connectivity is not first-order definable," Bull. EATCS 26 (1985), 43-45.
- **Ehrenfeucht-Fraïssé game** (the FO-inexpressibility technique; connectivity as the classic exemplar) — `https://en.wikipedia.org/wiki/Ehrenfeucht%E2%80%93Fra%C3%AFss%C3%A9_game`.
- **Querying incomplete data / certain answers** (the correct formal home for non-identifiability) — Libkin, `https://homepages.inf.ed.ac.uk/libkin/papers/pods14.pdf`; `https://arxiv.org/pdf/2310.12694`.
- **Kontchakov, Pratt-Hartmann, Wolter, Zakharyaschev**, "Spatial logics with connectedness predicates," LMCS 6(3:7) 2010 — `https://lmcs.episciences.org/1229`, arXiv:1003.5399 (connectedness not derivable from base contact relations; the spatial-logic analog).

### Repos / live fetches

- `github.com/CainanD/RefAV` (`main`) — `refAV/atomic_functions.py` fetched to `/tmp` (32 defs, 0 free-space primitives), re-verified independently by I2, L1, L2, L3, L5.
- `github.com/robosense2025/track5`, `robosense2025.github.io/track5` — Track 5 = Cross-Platform 3D Object Detection; PV-RCNN + ST3D baseline; AP@0.50 R40.
- `huggingface.co/datasets/Pi3DET/data` — Pi3DET dataset underlying Track 5 (3D detection boxes).
- **Visionary / PRISM (L6):** `visionary.run` (company site); VentureSquare EN `venturesquare.net/en/1081959/` + KR `venturesquare.net/1081950/`; WOWTALE `en.wowtale.net/2026/05/13/234060/`; kspost `kspost.biz/en-us/articles/2525`; Platum `platum.kr/archives/286764` — PRISM = AD dataset-construction / scene-curation + labeling; **zero** occupancy / free-space terms across EN + KR.
