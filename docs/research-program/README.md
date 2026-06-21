# spatial-probe — the six-paper research program

One thesis: **3D is queryable, updatable state, not the render.** Each paper takes one
falsifiable physical predicate, runs it as a test, and measures whether a representation stores
the signal needed to answer it — always with a fairness control so the instrument is not rigged.

The instrument (`src/probe/`) is the durable asset; each paper is an experiment on top
(`experiments/<name>/`). **Read [research-integrity.md](../research-integrity.md) first** — it is
the rule that keeps these papers honest. The per-stage plans and the frontier methodology that
grounds them are produced by the 2026-06-21 deep-research review (35 agents, code+data grounded).

## Program order (by external-anchor strength)

`occquery → dynfield → value-of-correction → asof → gt-distrust → vis-calibration`

Only **occquery** rides a live leaderboard (RefAV / HOTA-Temporal, a RIDE topic). The other five
DEFINE a new axis on public data: bar = public dataset + a measurable gap vs a public baseline +
a released benchmark, never a leaderboard win.

## The six papers

| # | paper | axis | one-line hypothesis | class | review verdict (2026-06-21) |
|---|---|---|---|---|---|
| 1 | occquery | state→query | occupancy predicates measure box-blind free-space that box-only query languages cannot express, and do so denotation-correctly | HYBRID | H1 sound; **H3 oracle circular** → revise |
| 2 | dynfield | dynamics | which stored motion field a planner needs is regime-dependent and identifiable (necessity ≠ "store everything") | DEFINE | revise (SH1 tautology; velocity provenance wrong) |
| 3 | value-of-correction | curate | value-ranked label fixes beat error-count ranking per budget (value ≠ error count) | DEFINE | revise (title claim unsourced; publishable tier GPU-gated) |
| 4 | asof | render→state | a state derived from a render preserves the planner-relevant measurement, and overlap (RayIoU) is not a sufficient proxy | HYBRID | revise (field metrics undefined; render substrate unconfirmed) |
| 5 | gt-distrust | visibility | occlusion-depth ranks which GT labels are untrustworthy, beyond a binary visibility mask | DEFINE | **REJECT-as-unsound** (stats wrong; oracle = affirming-the-consequent) |
| 6 | vis-calibration | uncertainty | model confidence is more over-confident where the sensor never observed | DEFINE | revise (unobserved-stratum GT circular) |

## Program coherence — COHERENT-WITH-FIXES

The six papers are MECE: they partition `render→state→query→trust→curate→dynamics` with no
load-bearing overlap, and all six inherit the "MEASURES, does not judge danger" fence (danger is
dynfield's, and even dynfield measures action-change, not safety). Required fixes:

1. **Numbering** — the canonical order above is authoritative (occquery=1 … vis-calibration=6). The
   review found a scrambled ordinal that disagreed with both the repo and each plan's own prose;
   an ordinal that means two things is the structural defect — pick one meaning.
2. **DAG invariant** — pin "every hard-upstream position < own position" so no stage is sequenced
   before the dependency it builds on (asof@4 needs occquery@1's executor; vis-calibration@6 needs
   gt-distrust@5's mask helper).
3. **Unify the injected-corruption oracle** — one released harness (owner: gt-distrust), imported by
   value-of-correction, not two divergent injectors + matched-control specs.
4. **Unify the released-checkpoint + RayIoU/SparseOcc harness** — shared by asof / gt-distrust /
   vis-calibration; one procurement + caching plan, each with a no-checkpoint fallback.
5. **Hoist the dense-GT 3-policy denotation-stability gate** — a shared precondition cited by all
   four GT-denominator stages, not just occquery and asof.
6. **Pin one mask-stratification helper** — gt-distrust's deliverable, imported unchanged by
   vis-calibration, so two divergent stratifiers cannot arise.

## Status

All six are PLANS. Only **occquery** has code (76 tests, synthetic smoke — not a scientific
result). The review defects above are tracked, not hidden. The headline correction for the active
stage: **occquery's H1 (expressivity) is the sound, oracle-free contribution; H3 (denotation
accuracy vs an "independent" oracle) is currently circular** and must be re-architected or demoted
before any denotation number is claimed as a result.
