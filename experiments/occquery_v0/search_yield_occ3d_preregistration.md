# Pre-registration — corpus-scale physical-quantity SEARCH YIELD on Occ3D-nuScenes (+ honesty tags)

Committed BEFORE any corpus run of this experiment; a post-hoc change shows in git history.
Author: Doeon Kwon + Claude. Date: 2026-07-03.

## Motivation (the product question this measures)

Aletheon's pitch is *measurement-based search over spatial logs*: "find every frame where the free
corridor pinched below the car's width" — a physical-quantity filter, not embedding similarity.
H1 (expressivity) established that box-only tooling cannot even POSE the free-space queries
(100-pt coverage gap, sealed a47b500, cross-dataset). What H1 never measured: **when you actually
run the sealed query set over a full labeled corpus, does it MINE anything?** (yield), and **can a
search hit/non-hit be honestly tagged with what the sensor actually observed?** (honesty tags).
Both are new estimands; neither re-opens the closed denotation oracles (stereo / DAv2 / Occ3D
dense-GT consistency stay closed per docs + memory).

## Claims (two legs, each falsifiable)

**C1 (yield — "the corpus contains a minable long-tail at the sealed thresholds").**
Run the SEALED 24-query set (`queries.yaml`, sealed 2026-06-22, thresholds physically motivated,
never tuned on this data; 20 occupancy + 4 box-baseline controls — see Amendment 1) over every
Occ3D-nuScenes scene on disk, dense GT arm (`mask='none'`, `unknown_policy=FREE` — both sealed,
matching L1). A query's **scene yield** = retrieved scenes / headline scenes (scene matches iff
any frame satisfies the predicate — all 24 sealed queries are `scope: any`).
- **C1 HOLDS iff ≥ 10 of the 20 occupancy queries have headline scene-yield in (0%, 20%].**
- **C1 KILLED (empty) iff ≥ 16 of 20 occupancy queries yield exactly 0** — the search engine mines
  ~nothing from nuScenes at the sealed thresholds; headline becomes that negative.
- **C1 KILLED (saturated) iff ≥ 10 of 20 occupancy queries yield > 50%** — the events are not
  long-tail; the "long-tail mining" framing dies (measurements may still be useful, claim dies).
- Any other outcome: NO adjective claim; report the full yield table only.
The 20% band edge and the 10/16/10 counts are claim-boundary knobs sealed here, a priori; the FULL
per-query yield table + frame-level hit rates + measured-value distributions (P5/P50/P95 per
physical quantity) are reported regardless of verdict.

**C2 (honesty tags — "a tri-state observability tag is informative on single-frame search").**
On the OBSERVED arm (`mask='lidar'`: single-sweep visibility, unobserved voxels UNKNOWN, policy
FREE sealed), every (frame, query) decision for the **free_path + corridor families (10 queries)**
gets one tag:
- `CONFIRMED_HIT` — predicate fires. Under policy FREE a hit can only be caused by OBSERVED
  occupied voxels (UNKNOWN is read as free), so a hit carries direct observed evidence.
- `EXONERATED` — no hit AND band unknown-fraction ≤ ε: the decision region was sufficiently
  observed, so "no event" is a supported negative.
- `UNRESOLVED` — no hit AND band unknown-fraction > ε: absence of a hit is NOT exoneration
  (the region was mostly unobserved).
Band (region proxy, sealed) = the L1 in-path band at the query's own horizon h: volumetric voxel
fraction with value UNKNOWN over forward ∈ [0, length/2 + speed·h], |lateral| ≤ width/2 + 1.0 m,
z ∈ (ground, ground + ego.height]. ε = 0.05 sealed for the verdict; the ε-curve
{0.01, 0.05, 0.10, 0.20} is reported (knob-as-curve, repo convention).
- **C2 HOLDS iff, pooled over the 10 queries on the headline split, UNRESOLVED ≥ 10% of non-hit
  decisions AND EXONERATED ≥ 1% of non-hit decisions (at ε=0.05).** (Then a search UI without the
  tag materially over-claims exoneration, and exoneration is still sometimes possible.)
- **C2 KILLED (tag unnecessary) iff UNRESOLVED < 1% of non-hit decisions** — single-frame coverage
  suffices; the tag adds nothing on this corpus.
- **C2 KILLED (exoneration impossible) iff EXONERATED < 1% at EVERY ε tier** — single-frame
  occupancy can never honestly exonerate; the honest headline is "hit-only search; exoneration
  requires multi-frame fusion".
Clearance/centerline tagging is OUT OF SCOPE here (their decision region is abeam, not the in-path
band; a side-region observability proxy is a named follow-up, not smuggled in post hoc).

## Metric implementation

- Yield / hit-rate: plain counts (retrieved / total), scene-clustered bootstrap CI (1000 resamples,
  seed 0 — the L1 machinery pattern; pure numpy, no sklearn). Unit-tested against hand values
  BEFORE the run (`tests/test_search_yield.py`).
- Query evaluation: the sealed `probe.retrieval.namespace` bindings evaluated by
  `probe.query_dsl.safe_eval` — the exact sealed evaluator. A memoization layer caches each
  predicate call per (frame, args) so measurements are computed once and reused across queries;
  a unit test proves memoized evaluation ≡ direct `retrieval.frame_true` on synthetic scenes
  (semantics-preservation gate — if it fails, the run does not start).
- Measured-value distributions come from the same cached calls (the "measurement index"; dumped to
  a gitignored parquet as a product artifact, not a claim).
- Tag logic unit-tested on hand-built grids (hit from observed obstacle; non-hit fully observed;
  non-hit mostly unknown).

## Independence ledger (honest scope)

- The dense arm is Occ3D's aggregated auto-label — corpus DESCRIPTION / consistency class, NOT
  external world-truth. Yields describe what the labeled corpus contains; no denotation claim.
- Tag VALIDITY against dense GT is NOT claimable on Occ3D: mask_lidar marks ~100% of occupied
  voxels visible (L1 degeneracy, sealed 6475448 finding), so hit-recovery and exoneration-validity
  checks are identities by construction here — stated, not run. Tag mechanism validity rests on
  the sealed synthetic mechanism test (`synth_denotation.py`, 5844786): verdicts are
  occlusion-robust with false_block_rate 0.
- Box-baseline queries (4 tracking controls) run with `with_boxes=True` (nuScenes annotations,
  GPS/IMU-derived boxes — independent of occupancy); their yields are fairness context, no gap
  re-derivation. The 20 occupancy queries' box-only expressibility = 0 is RESTATED from the sealed
  flags (`refav_expressible`, verified against RefAV's released 32-function set), not recomputed.

## Data (sealed)

- Corpus: every scene in `data/annotations.json` `scene_infos` (850 expected; loader failures are
  skipped + reported, never silently dropped).
- Split: identical hygiene rule to L1 — first 20% by sorted scene id = dev; the rest = headline
  (~680 scenes). No parameter is tuned anywhere (all thresholds sealed 2026-06-22), the split is
  kept for comparability with L1 reporting.
- Ego extents: the adapter's fixed nuScenes ego (1.85 × 4.6 × 1.9 m) — sealed constant.
- Arms per scene: dense (`mask='none'`, with_boxes=True) for C1; observed (`mask='lidar'`,
  no boxes) for C2. `unknown_policy=FREE` everywhere (sealed, = L1).

## Run (post-seal, once)

```
python experiments/occquery_v0/search_yield_occ3d.py            # full sealed run
python experiments/occquery_v0/search_yield_occ3d.py --limit 5  # smoke only, never reported
```
Outputs: `results/search_yield_occ3d.json` + `results/search_yield_occ3d_summary.md` (+ per-scene
JSONL checkpoint + measurement parquet, gitignored). Seed 0. Commit hash recorded in the report.

## Reporting

Negatives are headlines. If C1 or C2 is killed, the summary leads with the kill. No number from
the smoke run is ever reported. The verdicts state C1/C2 against these sealed criteria verbatim.

## Amendment 1 (2026-07-03, BEFORE any corpus run — zero scenes had been processed)

The original seal said "20 queries / 16 occupancy", taken from the STALE header comment in
`queries.yaml` ("clearance/centerline 3 each") rather than the file's content. The actual sealed
set is **24 queries / 20 occupancy** (clearance 5, centerline 5, free_path 5, corridor 5, box
baseline 4; verified by `load_queries` — all 20 occupancy have `refav_expressible: false`; the 10
tag-family queries are unchanged). The runner's fail-loud count guard caught this at smoke start;
no corpus data was seen. Verdict counts are rescaled at the SAME ratios sealed above: C1 holds
≥ 50% of occupancy queries long-tail (8/16 → 10/20), killed-empty ≥ 80% zero (13/16 → 16/20),
killed-saturated ≥ 50% above 50% yield (8/16 → 10/20). C2 is untouched. The sealed queries.yaml
file itself is NOT edited (its stale comment stays as evidence).
