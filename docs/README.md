# spatial-probe docs — source-of-truth map

Read this first. It says which file is authoritative for what, so the same fact is never re-decided in
two places. When two docs disagree, the conflict rules below settle it.

## Authority

| Question | Source of truth | Status |
|---|---|---|
| The program: 6 papers, order, north-star thesis, per-stage hypothesis + success/kill | **`PLAN.md`** (repo root) | authoritative |
| Per-topic external anchor + the verification ledger (verified vs merely claimed) | **`docs/benchmark-anchors.md`** | authoritative; PLAN defers to it |
| Per-paper detailed plan + the 2026-06-21 adversarial review (defects + fixes) | **`docs/research-program/`** | derived from PLAN + review; where it contradicts PLAN, the review is newer |
| How to run an honest experiment (anti-self-deception) | **`docs/research-integrity.md`** | authoritative (method) |
| How frontier papers are built (the survey grounding the plans) | **`docs/research-program/frontier-methodology.md`** + `frontier-survey.md` | reference |
| The committed hypothesis + success/kill for a SPECIFIC run | **`experiments/<exp>/preregistration.md`** | authoritative for that run, and NEWEST |
| Design language for `web/` | **`docs/design-language.md`** | authoritative (design) |
| Real-data findings + the 5 false-positive modes | **`docs/h3-real-data-findings.md`** | log (historical record) |
| M2 data-wiring contract | **`docs/m2-adapter-contract.md`** | contract |
| H1 expressivity vs RefAV | **`docs/expressivity-vs-refav.md`** | evidence |

## Conflict rules (when two docs disagree)

1. **For a specific experiment's hypothesis/criteria, `preregistration.md` wins** — it is the committed,
   timestamped, newest version. PLAN describes the program; the preregistration describes the run.
   - occquery: `PLAN.md` (L117-200) is the original plan; the 2026-06-21 review
     (`research-program/occquery.md`) found H3's oracle circular; **`experiments/occquery_v0/preregistration.md`
     is the corrected, authoritative version** (H1 sole headline, H3 demoted to a consistency check).
     Trust the preregistration.
2. **For the program** (which papers, order, MECE), **`PLAN.md` + `benchmark-anchors.md` win**;
   `research-program/` is the reviewed elaboration (and records the defects, not hidden).
3. **A number** (test count, a statistic) is true only where the **code/data** says so. Docs are
   corrected to match the code, never the reverse (`research-integrity.md`, "verified three ways").

## NOT source of truth (presentation / scratch / regenerable)

- `web/` — build-in-public presentation, not the science.
- `experiments/<exp>/results/*` except a hand-written `summary.md` — regenerable output.
- `~/Desktop/*.md` — scratch drafts (Aletheon notes, an old research agenda), outside the repo. If any is
  still load-bearing, move it into `docs/` (or a private repo) and delete the desktop copy, so each fact
  has exactly one home.
