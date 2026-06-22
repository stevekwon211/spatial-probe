# spatial-probe

An instrument for **probing what a spatial representation actually stores** — and whether
that stored state is queryable and trustworthy enough for a machine to act on.

> Thesis: 3D's essence is queryable/updatable **state** (geometry, occupancy, dynamics,
> uncertainty, provenance), not the **render**. Method: run a *falsifiable physical
> predicate* as a test, paired with a fairness control, and measure whether a
> representation stores the signal needed to answer it.

`src/probe/` is the reusable instrument; each research question lives in
`experiments/<name>/`. See **[PLAN.md](PLAN.md)** for the full plan and the first
experiment's falsifiable protocol.

## Status

First experiment: **`experiments/occquery_v0`** — occupancy-native physical-predicate retrieval
(clearance / free-path / free-width) that object-box query languages cannot express. The core
instrument (DDA raycast, occupancy grid, C-space predicates, Occ3D-nuScenes adapter, query DSL,
retrieval, metrics) is implemented, tested (pure numpy/scipy, no torch), and **runs on real
Occ3D-nuScenes mini** (10 scenes / 404 frames).

**H1 — expressivity (the headline, oracle-free): holds.** Three non-identifiability witnesses
(`tests/test_expressivity.py`) show two scenes identical in every box+map observable but differing
in unboxed occupancy — a box-only language must return the same answer, the occupancy predicate
distinguishes them. RefAV's released 32-function set has 0 free-space primitives; 20 of 24
pre-registered queries are inexpressible in it. No oracle, reproducible by anyone.

**H3 — denotation correctness: DEMOTED to an internal-consistency check.** The only oracle buildable
on this hardware re-derives over the *same* Occ3D/LiDAR the predicate reads, so it measures
consistency, not external truth; it was left honestly unbuilt rather than faked. A real P/R/F1
number is blocked on a val-data download + a positive-containing scene set (the 10 mini scenes have
0 positives). **H2** (leaderboard) is not started — substrate mismatch (Occ3D-nuScenes vs RefAV's
Argoverse 2). See **[results/summary.md](experiments/occquery_v0/results/summary.md)** and the sealed
**[preregistration.md](experiments/occquery_v0/preregistration.md)**; success criteria in
**[docs/benchmark-anchors.md](docs/benchmark-anchors.md)**. Synthetic runs are smoke tests, never evidence.

## Layout

- `src/probe/` — core instrument (DDA raycast, occupancy grid, predicates, metrics, adapters)
- `experiments/occquery_v0/` — first falsifiable experiment
- `docs/` — benchmark anchors & success criteria (`docs/benchmark-anchors.md`)
- `tests/` — unit tests (TDD; core primitive first)
- `data/` — datasets (gitignored; never committed)
- `web/` — visualization / build-in-public site (Next.js + Tailwind, deployed on Vercel); reads experiment results

## Setup

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest                              # full suite, ~1.5s, no data needed
python experiments/occquery_v0/run.py         # synthetic smoke (NOT a result)
```

Datasets: nuScenes-mini + Occ3D-nuScenes labels into `data/` (free; nuScenes research
account + terms). v0 runs CPU-only on `mini`.

## License

Apache-2.0 © 2026 Doeon Kwon. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
