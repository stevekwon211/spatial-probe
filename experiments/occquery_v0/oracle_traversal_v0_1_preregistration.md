# Oracle-v0.1 — CLEAN held-out confirmatory of the ego-trajectory traversal oracle (PRE-REGISTRATION, seal before data, 2026-06-25)

Supersedes the confirmatory claim of `oracle_traversal_preregistration.md` (v0, commit 7474991) and
reconciles the two conflicting v0.1 drafts (`oracle_heldout_preregistration.md` and the earlier
`oracle_traversal_v0_1_preregistration.md`). v0's sealed run (`W=5`, full ribbon) and its exploratory
correction (`W=40`, `far=4`) are BOTH confounded; see `results/oracle_traversal_summary.md`. This v0.1
re-tests the v0 exploratory finding on HELD-OUT logs with mechanism-derived (NOT result-selected)
parameters. Nothing below was chosen after seeing a v0.1 number.

## Why (unchanged from v0)
The program's biggest exposure is the L5 "collapse" attack: *"a clearance predicate over an occupancy
grid is a half-day script; is it even RIGHT?"*. The durable rebuttal is denotation-CORRECTNESS via an
INDEPENDENT, label-free oracle. The ego's recorded future trajectory is ground truth for "free" — a
vehicle cannot drive through a real obstacle — so the ego's future swept footprint marks space that was
physically FREE at frame t, independent of the occupancy perception in both data source (recorded
`city_SE3_egovehicle` poses, not LiDAR returns) and algorithm (rigid-body sweep, not voxelization).

## What v0 got wrong — the TWO confounds this fixes
1. **Near-field trap (kills `W=5`).** On the stop-and-go FOLLOWING substrate the swept ribbon over a short
   window sits almost entirely inside the ego self-return-removal zone `x < _EGO_X1 = 3.9 m`
   (`src/probe/adapters/av2_sensor.py:41`), where occupancy is 0 BY CONSTRUCTION (the voxelizer drops
   in-ego-cuboid LiDAR returns, `av2_sensor.py:74`). So v0's `true_fp = 0.0000` means "occupancy is empty
   in the ego footprint we already cleared," NOT "occupancy keeps the driven road free." Fixed by a `far`
   restriction tied mechanically to the ego front bumper (§a).
2. **Window was result-selected (kills `W=40`).** `W=40` / `far=4` were chosen AFTER seeing that `W=5`
   failed — textbook HARKing. v0.1 derives `W` from grid geometry + a measured speed, sealed here (§b).

## Sealed parameters (DERIVED, computed in this doc before any v0.1 run)

### (a) `far` — the ego FRONT bumper, mechanism-derived (requirement a)
The ego self-return zone ends EXACTLY at `_EGO_X1 = 3.9 m` (`av2_sensor.py:41`; the AV2 ego cuboid front,
rear-axle origin). A real lead vehicle sits beyond the front bumper (`x > _EGO_X1`, stated in that same
adapter comment, `av2_sensor.py:40`), so removing the ego cuboid CANNOT drop a real obstacle — but it
DOES zero occupancy inside the zone. To count a swept cell as "real road the ego physically drove,"
require it past the bumper:

```
far = _EGO_X1 = 3.9 m
```

**This is the exact mechanism boundary — no added margin.** The earlier draft proposed
`far = _EGO_X1 + VOXEL_SIZE = 4.3 m` ("one-voxel margin"), but (i) that +0.4 m has no mechanism behind it
(it is a soft, result-flavored choice, not the construction boundary), and (ii) the SEALED input file
`oracle_heldout_logs.json` (sha256 `f58ae31c…`) declares `far_m = 3.9` and its load-bearing
`escape_frac = 1.00` guarantee for all 8 logs was COMPUTED AT far = 3.9. Sealing far = 4.3 against an
escape_frac measured at 3.9 is an internal contradiction (the "no near-field-trap frames survive" claim
would be unverified for the sealed value). v0.1 seals `far = 3.9` so the doc, the run command, and the
sha256-pinned input are byte-consistent. The grid quantization already enforces a `> far` strict
comparison at voxel resolution (`oracle_traversal.py:124`), so a cell at exactly 3.9 is excluded; no
extra margin is needed to clear the construction-zeroed region.

### (b) `W` — the MINIMUM window to escape the ego self-return zone, derived from first principles (requirement b)
The swept ego footprint at future forward displacement `D` (in frame-t coords) is a rectangle of length
`L = ego.length` centered at `x = D`; its FRONT edge reaches `x = D + L/2`. For ANY swept cell to clear
`far`, the ego CENTER must travel

```
D_needed = far - L/2 = 3.9 - (4.9 / 2) = 3.9 - 2.45 = 1.45 m
```

(`L = 4.9 m`, the AV2 ego length the sweep passes as `ego.length`; `av2_sensor.py:39` "~4.9 × 2.0 m".)
This is the precise form of the "ego center must travel ~1.7 m beyond its current front to reach
x > _EGO_X1(3.9 m)" that the v0 summary names qualitatively.

`W` = the number of 10 Hz frames to travel `D_needed` at the substrate's typical speed:

```
W = ceil( D_needed / (v_median * dt) ) ,   dt = 0.1 s
```

The FORMULA above is sealed. `v_median` is a MEASURED property of whatever substrate is sealed in §d; it
is read from the primary source (poses on disk), NOT inherited from prose.

**Primary-source check on the v0 summary's speed figure (requirement b: "state it").** The v0 summary
(line 13) states the ego "creeps only 0.91 m on average in 0.5 s → ~1.8 m/s" on the following danger
substrate. This is a PROXY (a sentence about the data, not the data). Verified against the poses on disk
(finite-difference per `av2_sensor.py:88-105`, over all 2819 danger frames of the 18 v0 logs):

```
following-substrate ego speed: median = 5.191 m/s   (mean 5.298; p25 1.284, p75 8.619, p90 10.980 m/s)
following-substrate W=5 displacement: mean = 2.65 m, median = 2.62 m   (NOT 0.91 m)
```

**The "~1.8 m/s / 0.91 m" figure does NOT reproduce.** It was the slow-frame SUBSET (the stop-and-go
tail, p25 ≈ 1.28 m/s), not the substrate median. Per "Proxy is a hypothesis" + "Research integrity — no
HARKing," v0.1 seals against the VERIFIED median, never the unverified prose number. Worked examples of
`W` (at far = 3.9, D_needed = 1.45 m) are recorded so the derivation is auditable either way:

```
W @ following median 5.191 m/s = ceil(1.45 / 0.5191) = ceil(2.79) = 3 frames (0.3 s)
W @ stale-proxy   1.800 m/s    = ceil(1.45 / 0.1800) = ceil(8.06) = 9 frames (0.9 s)   # for the record; NOT sealed
```

Either way the result-picked `W = 40` is ~4–13× too large; the derivation kills it.

### (c) Sealed input is FREE-DRIVING held-out — so the escape window is sealed at the round 10 Hz second
The sealed confirmatory runs on the FREE-DRIVING held-out set (§d): the cleanest fix the v0 summary
itself named (line 28 — "switch substrate to free-driving … the cleanest fix"), because a free-driving
ego has no close lead to vacate occupied space (confound 2 removed: `close_lead_frac_15m = 0.00` for all
8 logs, verified in the json) and always clears the self-return zone (confound 1 removed:
`escape_frac = 1.00` for all 8 logs, verified in the json — note this guarantee was computed at
far = 3.9, the value sealed in §a). At free-driving speeds the §b minimum window is only
`ceil(1.45 / (7.26*0.1)) = 2 frames` even for the SLOWEST held-out log. To keep the window a clean,
mechanism-anchored quantum rather than a per-log minimum, it is sealed at the round one-second window:

```
W = 10 frames (1.0 s @ 10 Hz)
```

This MATCHES the `W_frames = 10` already recorded in the sealed `oracle_heldout_logs.json`. `W = 10`
(1.0 s) is the same order as the following-substrate minimum and an order of magnitude below the
result-picked `W = 40`; the short window also admits the least cross-traffic motion. It is
mechanism-anchored (the round 10 Hz second on a substrate where the minimum is ≤ 2), NOT result-selected.

> If the run is instead pointed at a FOLLOWING held-out set, BOTH `far` and `W` MUST be re-stated for that
> set before the run: `far = 3.9` is substrate-independent (it is the construction boundary), but `W` MUST
> be recomputed from THAT set's own measured median by the §b formula and logged here BEFORE the run (the
> following median 5.191 m/s gives `W = ceil(1.45/0.5191) = 3`). The formula is sealed; the substrate's
> speed is a measured input.

### Per-frame escape gate (so `W` cannot be gamed by undersized displacement)
A frame whose footprint never reaches `far` contributes zero real-road cells and is DROPPED, not counted
with a degenerate denominator. The harness already enforces this: `sweep &= XX > args.far`
(`oracle_traversal.py:124`) then `if den < _MIN_SWEEP_CELLS: continue` (`oracle_traversal.py:126`,
`_MIN_SWEEP_CELLS = 5`). The denominator is therefore "real-road cells the ego provably drove past the
bumper," applied IDENTICALLY to the true and shuffled arms.

## (d) HELD-OUT logs — re-test the exploratory finding on fresh data (requirement d)
v0's `W=40` finding was exploratory on the 18 logs of `experiments/dynfield_v0/av2_danger_logs.json`.
v0.1 confirms on logs DISJOINT from those 18, sealed in `experiments/occquery_v0/oracle_heldout_logs.json`.

Sealed held-out set (verified at seal time against disk and the json):
- **K = 8 logs**, all FREE-DRIVING AV2-Sensor val, **NONE among the 18** (disjointness verified empty
  intersection).
- Selection used ONLY pre-existing motion/geometry criteria, never any oracle output: `escape_frac ≥ 0.99`
  (json shows 1.00 for all 8), `close_lead_frac_15m = 0.00` (all 8), speed-stratified across measured
  per-log displacement `[7.26, 14.95] m/s` for motion diversity. No `traversal_fp` was inspected during
  selection.
- Logs (UUIDs, `oracle_heldout_logs.json:logs`): `d5d6f11c…`, `c865c156…`, `c2d44a70…`, `bbd19ca1…`,
  `070bbf42…`, `a1589ae2…`, `c222c78d…`, `27be7d34…`. Total ~1257 frames; all 8 present on disk with
  `city_SE3_egovehicle.feather` (~2680 pose rows ≈ 197 Hz) + 156–159 LiDAR sweeps each (verified by
  reading every feather).
- 4 of the 8 (`d5d6f11c`, `27be7d34`, `c2d44a70`, `c222c78d`) carry SPARSE longitudinal-danger frames,
  but their whole-log `close_lead_frac_15m ≈ 0.00`, so they are legitimately free-driving AND disjoint
  from the 18 — both integrity asks satisfied.
- The selection confound-metrics (`escape_frac`, `close_lead_frac`) were validated against the downloaded
  ground-truth ego-frame `annotations.feather` after an apparatus bug was caught and fixed (poses are
  ~197 Hz not 10 Hz; mining-feather boxes are already ego-frame — a city-frame transform was wrong).
  Logged per "Order your suspects" (the apparatus was the first suspect and it WAS buggy).
  `annotations.feather` was read ONLY to validate the selection metric; it is NOT read by the FP estimand.

This is FRESH data (different logs) AND a cleaner substrate — both integrity asks satisfied at once.

## (e) The dynamic-lead confound — declared up front as a CONSERVATIVE inflator (requirement c)
Even on free-driving, a dynamic CROSS-traffic object can vacate a voxel within the lookahead: a cell
CORRECTLY occupied at frame t (an object was there) that the ego traverses at t+W is counted as an
apparent false positive even though occupancy was right. This INFLATES the apparent `true_fp` AGAINST
occupancy. Therefore any measured `true_fp` is an UPPER BOUND on occupancy's real hallucination rate; a
`true_fp` that still beats the null beats it CONSERVATIVELY. The free-driving substrate removes the
dominant form of this confound (no close lead: `close_lead_frac_15m = 0.00` for all 8 logs), and
`W = 10` (1.0 s) admits the least residual cross-traffic motion of any clean second-scale window. The
confound is minimized, not erased, and it works in our disfavor.

## Estimand (sealed; only `W`, `far`, and the input file change vs v0)
For each held-out frame t (`oracle_traversal.py:105-142`):
- Build the ego future swept volume over `W = 10` frames: transform `p(t+1 … t+10)` into frame-t ego
  coords and sweep `ego.length × ego.width` (`_rect_mask`, `oracle_traversal.py:73-78`), then restrict to
  real road: `sweep &= (x > far = 3.9)` (`oracle_traversal.py:124`).
- Drop the frame if `den = |sweep| < _MIN_SWEEP_CELLS (5)` (`oracle_traversal.py:126`).
- `traversal_fp(t) = |sweep ∧ occupied_bev| / den`, where `occupied_bev` is the BEV projection of OCCUPIED
  voxels capped at ego height (`grid.obstacle_centers(max_height_agl=ego.height)`, `oracle_traversal.py:129`).
- Report the per-log-clustered mean with a scene-clustered bootstrap 95% CI (`_boot_mean`,
  `harness_v2.py:55`, `n_boot=1000`; `defined=False` if < 4 usable frames → INDETERMINATE, not a pass).

## Pre-registered comparison (RELATIVE, not absolute)
Shuffled-occupancy null (`oracle_traversal.py:137-141`): relocate the SAME count of occupied cells at
random within the grid extent, recompute `traversal_fp` on the SAME swept ribbon. The load-bearing claim
is the RELATIVE gap `true_fp << shuffled_fp`, never an absolute cutoff (per `docs/benchmark-anchors.md`:
beat-random, not a movable number).

## Falsifiable kill (declared before data) (requirements e + f)
Decision rule (`oracle_traversal.py:151-154`), with both bootstrap CIs defined:
- **CONFIRMED** iff `true_fp.hi < shuffled_fp.lo` (true CI strictly below the shuffled-null CI).
- **KILL / FAIL** iff `true_fp.lo >= shuffled_fp.lo` (CIs touch or true is not below) — occupancy is no
  better than random at avoiding the driven path → its "blocked" denotation is untrustworthy → the
  collapse-attack rebuttal fails with it.
- **INDETERMINATE** otherwise (partial CI overlap, or `<4` usable frames) — reported as inconclusive,
  NEVER as a pass.

**This kill is REACHABLE.** The near-field-trap fix (`far = 3.9`) removes the by-construction zeros that
GUARANTEED a pass in v0's `W = 5`, so a genuine null is now possible: every counted cell is real road the
ego provably drove, and occupancy could put a hallucinated obstacle there. *"This observation means I am
wrong"*: if, on FRESH free-driving logs, with the construction-zeroed near field EXCLUDED and the
cross-traffic confound inflating against us, `true_fp` is NOT below the shuffled null, the oracle does not
demonstrate occupancy denotation-correctness on the driven path. A clean negative is the headline.

## Honest scope / ceiling (declared, one-sided) (requirement f)
- **ONE-SIDED — FALSE POSITIVES ONLY.** Measures occupancy hallucinations (OCCUPIED where the ego provably
  drove) in the driven ribbon. It does NOT and CANNOT measure RECALL — real obstacles occupancy MISSES —
  deferred to oracle-v1 (`oracle_stereo_recall_preregistration.md`). No safety/danger claim is made.
- The reported `true_fp` is an UPPER BOUND (the cross-traffic confound inflates it).
- The ego ribbon is a thin slice (only where the ego drove); it cannot certify occupancy OFF-path.
- Genuinely label-free: boxes/annotations are NOT read by the FP estimand. Inputs = the `av2_sensor`
  occupancy grid + recorded ego poses only.
- This is occquery H3-family denotation-correctness, oracle-free of boxes. Per repo `CLAUDE.md`, H3 is an
  internal-consistency demotion in the MAIN result; this oracle's INDEPENDENCE (recorded poses, not LiDAR;
  rigid-body sweep, not voxelization) is what lets it exceed mere consistency — the load-bearing claim.

## Exact run (after this doc is committed; run ONCE)
```sh
# Plumbing (registered here so git shows the held-out file was the input):
#   (1) add `--logs PATH` arg at oracle_traversal.py (default the v0 danger file);
#   (2) when the loaded JSON has top-level keys {_meta, logs}, read names from json["logs"]
#       (the held-out schema) instead of list(json)[:limit] (the flat {uuid:[ts]} v0 schema).
#   This is REQUIRED: list(danger) on the held-out file yields ['_meta','logs'] and crashes.
source .venv/bin/activate
python experiments/occquery_v0/oracle_traversal.py \
    --logs experiments/occquery_v0/oracle_heldout_logs.json \
    --window 10 --far 3.9 --limit 8
# writes experiments/occquery_v0/results/oracle_traversal.json (window_frames=10 recorded in the report)
```

## Seal checklist (fill / verify BEFORE the confirmatory run)
- [x] `far = _EGO_X1 = 3.9 m`, mechanism-derived (exact construction boundary, no added margin);
      MATCHES `oracle_heldout_logs.json:far_m = 3.9` (reconciled — the 4.3 contradiction is removed).
- [x] `W` formula sealed `W = ceil(D_needed / (v_median * dt))`, `D_needed = far - L/2 = 1.45 m`.
- [x] v0 summary's "1.8 m/s" verified against poses → does NOT reproduce (measured following median
      5.191 m/s); sealed `W` derived from a MEASURED speed, never the prose proxy.
- [x] Sealed input = free-driving held-out, `K = 8`, disjoint from the 18, all on disk; `W = 10`
      (= json `W_frames`; escape_frac = 1.00 all logs at far = 3.9), `far = 3.9` (= json `far_m`).
- [x] `oracle_heldout_logs.json` sha256 = `f58ae31c1a56ff29dc2ead4efc5a9e75d9180b21762b177ea3c0e44273e5e6f0`
      (verified byte-identical at seal time; re-verify before the run).
- [ ] `--logs` + `{_meta, logs}`-schema plumbing added to `oracle_traversal.py:37,87-88`; `git diff` shows
      the held-out file as input. **(REQUIRED — the run CANNOT execute until this is written.)**
- [ ] This doc committed (git + timestamp) BEFORE the run; confirmatory executed EXACTLY once.

## Next signals (named, not run)
oracle-v1 (`oracle_stereo_recall_preregistration.md`) adds the stereo-camera RECALL confirm — the
miss-side this oracle cannot see — with its OWN measured reliability against a human-labeled calibration
set. The contribution is the HONEST ACCOUNTING of how much a label-free verdict can be trusted.
