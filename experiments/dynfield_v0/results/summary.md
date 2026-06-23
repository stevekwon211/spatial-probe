# dynfield Tier-1 — results summary (action-sensitivity, Mac)

dynfield asks: which stored motion field changes a planner-surrogate's action, by regime? Framing is
ACTION-SENSITIVITY (did removing the field move vs leave the surrogate's action), NOT necessity —
necessity needs a closed-loop quality oracle (GPU Tier-2). Pre-registration: `../preregistration.md`.

## What holds

- **SH1 (necessity-witness → scope lemma, oracle-free): HOLDS.** Two frames identical in static
  occupancy but differing in stored motion force a static-only surrogate to act identically while a
  motion-aware one differs (`tests/test_dynfield_witness.py`, 4 tests). By construction, no oracle.
- **SH4 (leakage gate): PASSES.** On the held-out official nuScenes val split (150 scenes), the static
  field (lead distance) does NOT encode the ablated velocity — distance↔velocity correlation = 0.184
  (< the pre-registered 0.2). The static-only baseline is not motion-contaminated.
- **Per-object velocity is real** (adapter populates it from box_velocity differencing; 67/69 finite
  on scene-0061, plausible). The official val split is sealed in `held-out-val.txt`.

## What is NOT a result (the gates caught it — reported, not hidden)

The {velocity × agent-context} FLIP matrix is **NOT reportable** on the v1 surrogate:

- **surrogate-validity probe FAILED** (closing→brake 0.27, below the pre-registered 0.8). Partly a
  loose probe spec (the "closing" set included slow/far frames the surrogate correctly proceeds on),
  but it fails as pre-registered.
- **shuffled-velocity control FAILED**: the TRUE velocity flips the brake/proceed decision on only
  3.3% of frames — LESS than a shuffled velocity (8.5%). So on real, mostly-safe lead-following, the
  binary decision is dominated by distance and velocity rarely flips it; the flip signal is below the
  noise floor. The binary brake/proceed surrogate is too coarse to measure velocity action-sensitivity.

Per the pre-registered substrate-validity pivot, a surrogate that fails its probe + shuffled control
does NOT produce a reported matrix. This is the integrity machinery working (the dynfield analogue of
occquery's AUC-0.798 honest negative): no surrogate tuning to manufacture a result.

## The honest hint (a curve, not a claim)

The secondary continuous **decel-delta** (magnitude of velocity's effect on commanded deceleration) is
non-uniform across agent-context in the same run: vehicle_following **0.00** (velocity changes
nothing), vehicle_crossing **0.88**, vru 0.15, other 0.74. This is the hypothesized "redundant when
following / changing when crossing" shape — but the binary flip metric (the pre-registered primary)
is too coarse to certify it, and decel-delta was not shuffled-validated this run. It is a hint that a
richer surrogate could surface, not a result.

## Decision / pivot (pre-registered)

Ship SH1 + SH4 as the Mac-feasible Tier-1 deliverables. The velocity action-sensitivity MATRIX is
deferred to **v2 = a graded surrogate** (continuous required-deceleration so velocity's effect is
measurable as magnitude, not just a binary flip; primary metric = decel-delta with a shuffled-control
gate), pre-registered before its run — OR to **GPU Tier-2** (a real closed-loop planner, where the
word "necessary" is finally licensed). The occupancy-flow second field rides the same harness once a
non-degenerate surrogate exists.

Net: a clean oracle-free witness (SH1) + a passing leakage control (SH4) on real val data, and an
honestly-deferred matrix — the v1 binary surrogate is too coarse, said plainly rather than tuned.
