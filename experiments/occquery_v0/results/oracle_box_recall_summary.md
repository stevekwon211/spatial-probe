# Oracle-v2 — GT-box RECALL oracle: RECALL-SUPPORTED (same-modality consistency, 2026-06-25)

Pre-registration sealed BEFORE data: `oracle_box_recall_preregistration.md` (commit e55612d). Apparatus
fix (occupancy attr + ns-timestamp keying) committed before any estimand read: b37b7d2. Module pure numpy,
`--self-check` passes (box rasterizer hits the exact `_voxelize` voxel; ego/below-floor → 0 voxels).

Run (once, sealed): `--logs ALL --n-interior-min 5 --min-boxes 3 --range-bin-m 8 --null on-road-matched
--n-curve 1,3,5,10,20 --seed 0`. 27 logs, 4236 usable frames, 0 logs dropped (ann_ts==sweep_ts all logs).

## Verdict: RECALL-SUPPORTED (the pre-registered relative kill fired clean)
**true_miss 0.5065 CI[0.463, 0.550] vs region/range-matched on-road null 0.7334 CI[0.706, 0.761]**
→ `true_miss.hi 0.550 < null_miss.lo 0.706` → occupancy marks real (LiDAR-seen, ≥5 pts) boxes FREE
strictly LESS often than size/range/on-road-matched random locations → it recalls real structure beyond
its base occupied-density. All oracle-validity falsifiers cleared: N-curve monotone decreasing
(N1 0.624 → N3 0.560 → N5 0.506 → N10 0.414 → N20 0.309), null reachable (0.3 < 0.733 < 0.97),
C1 zero (no timestamp drift), self-check passed.

## Read this BEFORE quoting the number — the absolute 0.51 is a confound-inflated UPPER BOUND
- **RECALL-SUPPORTED is a RELATIVE claim** (occupancy beats matched-random), NOT "occupancy has good
  recall." Do not read 0.51 as "occupancy misses half of real objects."
- The 0.51 absolute miss is heavily inflated by **C2 floor-straddle**: 88.1% of box bottoms lie below
  `_ROAD_Z = 0.3 m`, and `num_interior_pts` counts returns over the FULL box (incl. the below-road slab),
  while the estimand only credits the above-road slab. So a box whose returns sit mostly on wheels/lower
  body is counted as a "miss" even though LiDAR saw it. This inflates `true_miss` well above occupancy's
  real internal-miss rate.
- **Why the verdict survives the confound:** the floor-straddle (and C4/C6) inflate BOTH arms — the null
  boxes are rasterized through the identical above-road-slab filter — so the inflation largely CANCELS in
  the relative gap. The 22-point gap is robust to the shared confound; the absolute level is not. This is
  exactly why the pre-reg claims the relative gap, never an absolute cutoff.

## The strata are the real diagnostic (where the pipeline loses structure)
- `pts==0` (sensor-blind, 45,968 boxes): 99.6% missed — correct, occupancy can't mark what LiDAR never
  returned (the gate's `pts==0` exclusion is doing its job).
- `1≤pts<5` (sparse, 65,304): 94.6% missed — monotone with the gate.
- per-box miss falls monotonically as the LiDAR-evidence threshold rises (N20 = 0.309) — even densely-seen
  objects are entirely-free ~31% of the time, most of it the floor-straddle channel.
- tall classes (BUS/LARGE_VEHICLE) miss 0.658 vs 0.464 for others — height-cap + range/occlusion.
- Headline takeaway: the `_ROAD_Z = 0.3 m` ground filter is a real completeness loss channel for
  low-profile and floor-straddling objects — a concrete, actionable diagnostic of the voxelization, which
  is the honest contribution of a same-modality consistency check.

## Net (two-sided, honest)
- **FALSE-POSITIVE side (traversal-v0.1): RELIABLE** — no hallucinated obstacles in the driven ribbon.
- **RECALL side, consistency-level (this oracle): SUPPORTED** — occupancy puts occupied-mass on real
  LiDAR-seen objects better than matched random, robustly (relative gap). But this is a SAME-MODALITY
  internal-consistency check (provenance + algorithm independence only, NOT external truth), and the
  absolute miss bound is high + confound-inflated.
- **RECALL side, externally independent: still open** — the cross-modal, label-free, covers-unlabeled
  recall (Oracle B, frozen Depth-Anything-V2 mono-depth) is the next step; it is the only design that
  clears modality+algorithm+provenance against AV2 and would answer "does occupancy miss real structure an
  independent SENSOR sees," which this oracle (sharing the LiDAR) structurally cannot.

## Apparatus note (process integrity)
The first confirmatory crashed at the data-load line (`OccupancyGrid` has no `.grid` attr — it is
`.occupancy`) BEFORE any miss number was computed, so no result was produced and the seal (e55612d) is
uncontaminated. Two wiring bugs fixed (apparatus, not estimand): the attr name, and keying occupancy by a
float-reconstructed ns timestamp (`int(round(fr.time*1e9))`; ns > 2^53 drifts through float64 — the same
class as the labeler bug) → now keyed by the exact sorted sweep ts (157/157 box timestamps match,
verified). Sealed estimand/null/kill unchanged; re-ran the same sealed command. Git: seal → apparatus fix
→ run.
