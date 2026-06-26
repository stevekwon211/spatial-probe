# H3b — occupancy-native vs box-only: expressivity dominance + FP-denotation (2026-06-26)

Pre-registration sealed before data: `h3b_expressivity_preregistration.md` (commit a47b500). Driver
`h3b_expressivity.py` (pure numpy, src/probe read-only). Run on real AV2 (not synthetic). Deterministic.

## Result
**Leg 1 — expressivity coverage (oracle-free, real AV2):** on the sealed 20-query free-space set, OCCUPANCY
expresses 100%, BOX-ONLY (RefAV) 0% (+100 pt gap); on the 4 box-baseline fairness controls box-only is
100% (occupancy n/a). Overall 83.3% vs 16.7% (+66.7 pt). All 20 occupancy queries evaluated without error
on a real AV2 scene. **H1 NOT falsified** (box-only expresses no free-space query — structural, verified vs
RefAV's released function set).

**Leg 2 — FP-side denotation vs the INDEPENDENT traversal oracle (8 free-driving logs, 1177 frames):**
occupancy `free_along_ego_path` agrees with the traversal-FREE truth (the ego's physically-driven ribbon)
**0.9936** pooled (0.5s 1.0 / 1.0s 1.0 / 2.0s 0.9975 / 4.0s 0.9771); `min_free_width_along_path`>0 = 1.0 at
every horizon; **voxel ribbon false-positive = 0.0000 CI[0,0]** (reproduces the RELIABLE `oracle_traversal`
on the identical substrate). Box-only free-space denotation = INAPPLICABLE (no primitive to grade).

## Verdict (honest, no inflation)
- **Expressivity dominance: YES** — occupancy expresses the free-space family box-only structurally cannot
  (+100 pt). The program's H1 headline, now a real-AV2 coverage number.
- **FP-side denotation correct: YES** — 0.99 agreement, voxel FP = 0. Occupancy ~never false-blocks the
  physically-driven path; box-only has no free-space denotation at all.
- **Both-sided >=20-F1 denotation gap: NOT CLAIMED** — the BLOCKED side (unboxed-obstacle recall truth)
  needs a non-circular cross-modal oracle, empirically CLOSED solo/CPU this session (stereo 0.259 / DAv2
  scale >9 m / free-driving vacuity); GPU-gated. Held, not silently passed (occquery.md:27's own rule).

## Honest caveat (reported straight)
4.0s-horizon agreement is 0.9771, not 1.0: 27/1177 frames denote BLOCKED where the ego drove — traced to
15 turning frames (the predicate's STRAIGHT constant-velocity extrusion overshoots on curves, a documented
`free_along_ego_path` model limitation) + 12 longest-reach grazes, NOT occupancy false-positives (voxel
ribbon FP is exactly 0). So the shortfall is the predicate's straight-line reach assumption, not the
occupancy representation. Net claim stands: occupancy EXPRESSES what box-only can't AND its free-side
denotation is independently validated; blocked-side honestly GPU-deferred.
