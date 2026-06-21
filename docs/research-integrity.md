# Research integrity — how spatial-probe refuses self-deception

The researcher's adversary is not nature. It is the researcher. We unconsciously bend the
analysis toward the answer we want; p-hacking is not malice, it is the default. So the rule of
this lab is: **design the work so you cannot fool yourself, and so that if you tried, the git
history would show it.** This is the no-band-aid rule applied to epistemics — make the wrong move
structurally hard, not a thing we promise to avoid.

## Seven principles

1. **Falsifiability (Popper).** Do not try to confirm. State, before running, "this observation
   means I am wrong." A hypothesis that no result could kill is not a hypothesis.
2. **Pre-registration (git + timestamp).** Fix the hypothesis, the analysis path, and the
   success/kill criteria *before looking at the data*, and commit them. Changing them after seeing
   results (HARKing) is post-hoc justification, not discovery — and the commit history makes the
   change visible. You cannot quietly move the goalposts.
3. **No garden of forking paths.** The same data under many cutoffs / subsets / metrics will throw
   one spurious "significant" result. Fix the path up front and report *all* of it. Never select
   the pre-registration using oracle output (the review caught S2 doing exactly this).
4. **explore / confirm split (held-out, blind).** Separate the data you formed the hypothesis on
   from the data you confirm it on, and look at the confirm set *once*. Seal held-out scene IDs so
   reuse also shows in history.
5. **Relative over absolute.** No movable absolute cutoff (0.90 F1). Claim the *relative gap* vs a
   pre-registered baseline. An absolute threshold is a knob you will turn toward yourself.
6. **Adversarial.** Try to kill your own result. Independent skeptics, each prompted to refute.
7. **Report everything.** Failed hypotheses and negative results are recorded, not dropped.
   cherry-picking is the lie of omission.

## Structural enforcement (not a checklist)

- **`preregistration.md` per experiment, committed before the data run.** Hypothesis, analysis,
  success + kill criteria, timestamp. Post-hoc edits are diffable.
- **`held-out.txt` sealed scene IDs.** Not opened during development; reuse is visible in history.
- **Every analysis path logged** (e.g. the 3-UnknownPolicy spread is run in full, not the
  favorable one picked).
- **Relative gap is the load-bearing claim**; absolute thresholds are internal-validity checks only.
- **An independent oracle must differ in modality AND algorithm AND data provenance** from the
  thing it grades, with a measured independence test. "Same sweeps, same EDT" is not independent.
- **A kill criterion must be reachable.** If every outcome is pre-declared publishable, the design
  is unfalsifiable — reject it.

## Caught in the act — 2026-06-21 program review

The six-stage adversarial review (35 agents, code+data grounded) found, in *our own plans*,
exactly the failures this document exists to prevent. Kept here as evidence the mechanism works:

- **Circular oracle (S2 occquery H3, S4 vis-calibration).** The "independent" raw-LiDAR oracle
  shares nuScenes sweeps and the EDT computation with the predicate it grades — the Stage-1 circular
  problem renamed, not solved. vis-calibration's unobserved-stratum GT is default-fill free, i.e.
  calibration against a near-constant label.
- **Affirming-the-consequent (S3 gt-distrust H2).** The synthetic-injection answer key was a
  monotone function of `d_true` and the predictor *was* `d_true`, yielding a guaranteed AP gap
  (reproduced 0.91 vs 0.60) that is informative about nothing in real GT.
- **Wrong load-bearing statistics (S3).** The motivational numbers ("~22% observed", "93%
  free/occupied") were false against the 10 scenes on disk (measured 15.6% observed) — the review
  recomputed them from `data/gts/*/labels.npz`.
- **Movable threshold inside a relative claim (S2).** A 0.90-F1 absolute gate contradicted the
  "relative-only" framing; the ≥20-F1 gap was near-definitional because the queries were chosen so
  the baseline must fail.
- **Unfalsifiable kill trees (S3, S5, S6).** Designs where every kill outcome was pre-declared
  publishable, so no result could falsify the thesis.
- **Repo self-contradiction (occquery 0/5).** `summary.md` calls it a category error; `h3-real-data-findings.md`
  calls it a real false-positive bug — unreconciled, and several plans cited it both ways.

The lesson holds verbatim: the failures were structural (predictor = answer key; oracle = the
thing it grades), not threshold-tuning. The fix is to rebuild the construct, never to nudge a number.
