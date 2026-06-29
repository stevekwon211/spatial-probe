---
name: rigor
description: The one method spine for any non-trivial work — research OR software engineering. Ground-truth-first, question the requirement, standard-over-homegrown, seal a falsifiable claim before acting, adversarially self-refute before claiming, report honestly. Use for any experiment, benchmark, measurement, feature, bugfix, refactor, or any claim of "X works / done / result / beats Y". Invoke `/rigor research <task>` or `/rigor build <task>`.
---

# rigor — one method, two modes

The spine is identical for research and software engineering. Run it top to bottom. Do not skip a step
because it "looks obvious" — the skipped step is the one that bites.

## 0. Mode
- `research` — an empirical claim (experiment, benchmark, measurement, "X beats Y").
- `build` — code that must work (feature, bugfix, refactor, migration).
Mixed → research first (establish the fact), then build (ship it).

## 1. Ground-truth first — never trust a proxy
Scan the REAL artifact before asserting anything about it: `ls`/`find`/`wc -l`/`git log` the code; read
the actual data, logs, and reproduction. Label every claim **CONFIRMED** (read it this session) /
**INFERENCE** / **UNVERIFIED**. A doc, summary, memory, or prior agent's claim is a hypothesis, not a fact.

## 2. Question the requirement
Attach a person to it. Can it be deleted, loosened, or dropped? Reuse / standard before building. The most
common error is improving a thing that should not exist.

## 3. Standard over homegrown — the _roc_auc lesson
Use a tested library for anything a library does: metrics (sklearn), stats, parsing, geometry, math. A
hand-rolled metric/stat is a bug waiting to deflate your result (a homegrown AUC that mishandled ties once
turned a 0.598 into a 0.259 and a wrong conclusion). If you must DIY, unit-test it against the standard on
tie/edge cases first.

## 4. Seal a falsifiable claim BEFORE acting
- research → `/pre-register`: hypothesis + success + **KILL** criteria + independence ledger, committed to
  git BEFORE the data. Changing it later (HARKing) then shows in the diff.
- build → write the spec / acceptance criteria + the **failing test** FIRST.
Say out loud: "this observation means I am wrong." A claim no result could kill is not a result.

## 5. Execute
Change ONE variable at a time so cause is clean. Suspect the apparatus before the hypothesis (effort, model,
scope, wiring, silent caps). Keep the diff minimal.

## 6. Adversarially self-refute BEFORE claiming — `/falsify`
The adversary is you. Spawn independent skeptics to REFUTE the result / that-it-works; kill if a majority
refute. Decide on a **confidence interval, not a point estimate**. Confirm against the PRIMARY source by
content — code + data + behavior — not an internal pass (compiled/ran/green is necessary, not sufficient).

## 7. Report honestly
Negatives are headlines, not footnotes. State what failed, what was skipped, what is unverified. Claim the
**relative gap vs a baseline**, not a movable absolute cutoff. Ship reproducible (code + seed + exact command).

## Run it
- Deterministic fan-out: `Workflow({name:'rigor', args:{mode:'research'|'build', task:'…', config:{…}}})`
  — scan → seal → execute → adversarial-verify → honest-report, run by `rigorist` subagents (which preload
  this spine).
- SWE build sub-flow: delegate to `/feature-factory` (story→spec→build→acceptance→validation, 3 checkpoints).
- Atoms this spine composes: `/pre-register` (step 4), `/falsify` (step 6), `/deep-research-plan` (planning).
- **Customize per case** via `args.config` or a project `.claude/skills/rigor` override (project beats
  personal). Keep this spine general; push case-specifics to the edges, never into the core.
