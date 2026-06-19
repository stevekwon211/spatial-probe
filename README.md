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

Scaffold + plan only — under review (M0). No experiment logic implemented yet.
First experiment: **`experiments/occquery_v0`** — occupancy-native physical-predicate
retrieval (clearance / free-path) that object-box query languages cannot express.

## Layout

- `src/probe/` — core instrument (DDA raycast, occupancy grid, predicates, metrics, adapters)
- `experiments/occquery_v0/` — first falsifiable experiment
- `tests/` — unit tests (TDD; core primitive first)
- `data/` — datasets (gitignored; never committed)
- `web/` — visualization / build-in-public site (Next.js + Tailwind, deployed on Vercel); reads experiment results

## Setup (once implementation starts)

```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # M1 target: green core-primitive tests
```

Datasets: nuScenes-mini + Occ3D-nuScenes labels into `data/` (free; nuScenes research
account + terms). v0 runs CPU-only on `mini`.

## License

Apache-2.0 © 2026 Doeon Kwon. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
