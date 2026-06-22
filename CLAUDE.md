# CLAUDE.md — spatial-probe

Research instrument that probes what a spatial representation *stores* (occupancy /
geometry / dynamics) and whether that state is queryable + trustworthy enough to act on.
Two halves: a **Python 3.11 core** (`src/probe/`, numpy + scipy + pyyaml, pytest/TDD) that runs
falsifiable physical predicates over a voxel occupancy grid, and a **Next.js 15 / React 19
/ Tailwind 4 site** (`web/`) that reads experiment results for build-in-public.

North star, sequencing, and success/kill criteria live in **`PLAN.md`** and
**`docs/benchmark-anchors.md`** — read both before touching experiment logic or claiming
a result. The active experiment is `experiments/occquery_v0` (occupancy-native predicate
retrieval). Per-topic external anchors and the verification ledger are in
`docs/benchmark-anchors.md`; the M2 real-data wiring contract is `docs/m2-adapter-contract.md`.

## Build / test / run

Python core (from repo root):
```sh
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # numpy, scipy, pyyaml, pytest
python -m pytest                        # full suite (~1.5s), no data needed (count drifts; cite the command, not a frozen number)
python -m pytest --collect-only -q      # cheap structural check
pip install -e ".[data]"                # M2 only: adds nuscenes-devkit
python experiments/occquery_v0/run.py   # synthetic smoke run -> results/ (NOT a result)
```
Web (`web/`):
```sh
npm install
npm run dev        # next dev, http://localhost:3000
npm run build      # next build
npm run lint       # next lint
npx tsc --noEmit   # typecheck — there is NO `typecheck` npm script
```

## Module map (Python core)

- `src/probe/raycast.py` — the load-bearing primitive: 3D DDA line-of-sight (`traverse`,
  `line_of_sight`, `occlusion_depth`). Occupancy encoding constants live here: `OCCUPIED=1`,
  `FREE=0`, `UNKNOWN=-1`. Everything reduces to this.
- `src/probe/grid.py` — `OccupancyGrid` (world<->voxel, ground_height), `UnknownPolicy`
  (free / occupied / ignored).
- `src/probe/scene.py` — `Scene` / `Frame` / `EgoPose` / `TrackedBox` data model.
- `src/probe/predicates/` — falsifiable predicates: `clearance`, `freepath`, `reachable`,
  `objects`.
- `src/probe/query_spec.py` + `query_dsl.py` — schema-validated NL→predicate queries
  (`queries.yaml`); `retrieval.py` filters/retrieves matching scene sets (the scoring itself lives in the predicates, not here).
- `src/probe/adapters/occ3d.py` — Occ3D-nuScenes loader; must satisfy the
  `docs/m2-adapter-contract.md` output schema so `run.py` swaps `synthetic.SCENES` with no
  other change.

## Conventions & gotchas (repo-specific)

- **Python here uses snake_case modules + PEP8**, NOT the global kebab/camel rule — the AV /
  occupancy ecosystem is Python and this deviation is deliberate (PLAN §8).
- **`src/probe/` (core) gains code only on the *second* use.** Anything specific to one
  experiment stays in `experiments/<name>/`. Do not pre-build shared abstraction — this is an
  explicit invariant (PLAN §0, §2, risk table). Extracting too early is the named over-engineering trap.
- **Do NOT add Rust, numba, torch, or a `core-rs` crate now.** v0–v1 are pure Python + numpy;
  the escalation path (`numpy → numba → Rust`, PyO3 + WASM) is gated on profiling at ~M4 (PLAN §8).
- **Synthetic results are not scientific results.** `experiments/occquery_v0` runs on hand-built
  scenes (F1=1.00 by construction) = a smoke test. Real numbers need the M2 Occ3D adapter + the
  released oracle. Keep the three result classes distinct (unit test / synthetic smoke / externally
  validated — the last is NONE yet) and never present a synthetic number as evidence.
- **`data/` is gitignored — never commit datasets, `.npz` labels, or checkpoints.** Occ3D /
  nuScenes are gated behind a free nuScenes research account + terms.
- **Experiment outputs are gitignored except a hand-written `results/summary.md`** (`.gitignore`
  keeps that one file; raw dumps are ignored).
- **Self-chosen thresholds (0.90 F1, etc.) must be pre-registered and reported as a full curve**,
  and the load-bearing claim is the *relative* gap (≥20 F1 vs box-only, beat-random), not an
  absolute cutoff (`docs/benchmark-anchors.md` cross-cutting rules). Never let a result hinge on a
  movable number; never compare the same metric across different split/modality/supervision.
- Web reads `experiments/<name>/results/*.json` (intended to be copied into `web/public/data/` at build — that copy step is NOT yet wired and `web/public/data/` does not exist yet). No
  database — it serves precomputed static files. Vercel root directory = `web`.

## Env / secrets

No application secrets or `.env` in this repo. Only external requirement: a free **nuScenes
research account** + terms acceptance to download Occ3D-nuScenes into `data/` (gitignored).
Pin versions in `data/VERSIONS.txt` and `data/checksums.txt` per `docs/m2-adapter-contract.md`.

## Skills

- `next-best-practices`, `react-dev` — for `web/` (Next 15 RSC boundaries, React 19, typing).
- `frontend-design` / `web-design-guidelines` — only if reworking the site UI.
- No Python skill is installed — rely on the global rules for the Python core (TDD, immutable,
  small files/fns).

## Gotcha: the `.venv` here is Python 3.14, not 3.11

The committed setup says `python3.11`, `pyproject` is `requires-python = ">=3.11"`, but the
current `.venv` runs **Python 3.14.3**. Tests pass on it, but recreate the venv with `python3.11`
if you need to match the documented baseline. Verify with `.venv/bin/python --version`.
