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

M1–M3 (the data-independent core) implemented and tested; M2 (Occ3D-nuScenes adapter) pending data.
First experiment: **`experiments/occquery_v0`** — occupancy-native physical-predicate retrieval
(clearance / free-path / free-width) that object-box query languages cannot express. The synthetic
runner is a **smoke test, not a scientific result**; external success criteria (which public
benchmarks, what counts as success) are in
**[docs/benchmark-anchors.md](docs/benchmark-anchors.md)**.

## Layout

- `src/probe/` — core instrument (DDA raycast, occupancy grid, predicates, metrics, adapters)
- `experiments/occquery_v0/` — first falsifiable experiment
- `docs/` — benchmark anchors & success criteria (`docs/benchmark-anchors.md`)
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
