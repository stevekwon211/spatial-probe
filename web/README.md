# spatial-probe / web

Visualization + build-in-public site for the spatial-probe research program
(Next.js + Tailwind, deployed on Vercel). Part of the monorepo; it reads experiment
outputs from `../experiments/*/results/*.json` as they land.

## Dev

```sh
cd web
npm install
npm run dev        # http://localhost:3000
```

## shadcn (optional — richer blocks)

```sh
npx shadcn@latest init
npx shadcn@latest add <block>     # e.g. a hero / feature / stats block
```

The current page is hand-rolled Tailwind (clean + minimal); drop shadcn blocks in as the
site grows.

## Deploy (Vercel, monorepo)

Import the GitHub repo in Vercel and set **Root Directory = `web`**. Next.js is
auto-detected. **No database** — the site serves precomputed static result files.

## Data contract

Python experiments write results to `experiments/<name>/results/*.json` (occupancy
slices, query results, metrics). The web app reads those (copied into `web/public/data/`
at build, or imported) to render scenes and, later, interactive predicate demos. Add a
DB only when there is genuinely dynamic / user-supplied data — there isn't yet.
