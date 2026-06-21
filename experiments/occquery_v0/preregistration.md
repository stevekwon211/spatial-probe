# OccQuery v1 — pre-registration (2026-06-21)

Committed BEFORE reading the val oracle. [research-integrity.md](../../docs/research-integrity.md)
applies: changing the hypothesis or the criteria after seeing results is HARKing, and the change is
visible in this file's git history. The git commit timestamp is the pre-registration timestamp.

## The correction this encodes (adversarial review, 2026-06-21)

The Stage-2 plan made H3 (denotation correctness vs an "independent" oracle) the headline. The review
showed that oracle is **circular on this hardware**:

- raw-LiDAR re-voxelization shares the SAME sensor returns as Occ3D (Occ3D is itself built from
  accumulated multi-sweep nuScenes LiDAR). For STATIC geometry — the entire scope — a disjoint sweep
  window re-measures the same walls from nearby poses; it de-correlates only DYNAMIC content, which is
  out of scope.
- the predicate computes clearance via `scipy.ndimage.distance_transform_edt`; the oracle computes it
  via exact EDT. **Same algorithm, same quantity** — agreement tests voxelization bookkeeping, not
  geometric correctness.
- no genuinely independent modality (camera-MVS / stereo / ruler-calibrated human annotation) is
  feasible on this Mac.

**Therefore: H1 (expressivity) is the sole headline. H3 is demoted to an internal-consistency check
until an independent oracle is earned.**

## H1 — headline (oracle-free)

Claim, tightened to exactly what the witness licenses (NOT overclaimed): the curated occupancy queries
are **non-identifiable under the RefAV box+map observation set**. Two scenes identical in EVERY RefAV
function input (tracked boxes AND HD-map AND velocities) but differing by unboxed occupancy force any
function of those inputs to return the same answer, while the occupancy predicate distinguishes them.

- This is NOT the claim "no composition of RefAV's 33 functions can approximate it" — a 2-function
  witness plus a grep cannot prove that. It is non-identifiability under the box+map observables.
- PASS: a paired non-identifiability witness for EACH of the 3 predicate families (lateral-clearance,
  free-path, corridor-narrows), all green in `tests/test_expressivity.py`, with each witness pair made
  identical in boxes AND map AND velocity (so a map- or velocity-reading function cannot distinguish it).
- Oracle dependence: NONE. This is the quasi-deductive contribution, the conclusion-lane = existence.

## H3 — internal-consistency check (DEMOTED)

Labeled "consistency between two reconstructions of the same LiDAR", NOT external denotation
correctness, until an independent oracle is built: a different computational family (exact
point-to-point / point-to-mesh nearest-neighbor over raw LiDAR points, no voxelization, no EDT) AND a
measured independence test (inject a ground-truth obstacle at a known metric offset; score both
pipelines against the physical truth; show they can disagree where they must).

- **No absolute F1 gate.** The 0.90 PASS threshold is removed; absolute F1 is a descriptive curve with
  CIs only (it contradicted the relative-only framing — a cutoff a reviewer can move is not a result).
- The ≥20-F1 relative gap over box-only is reported as a consistency comparison, not as proof of
  denotation correctness, until independence is earned.

## Pre-registered before reading val (sealed)

- N ≥ 20 occupancy queries; only 4 exist today (`queries.yaml`). The full list is committed before the
  val run.
- A positive-containing val scene set — the 10 mini scenes have NO positives, so recall is currently
  unmeasurable. Scene IDs sealed in `held-out.txt` before the run, opened once.
- exactly 3 UnknownPolicy rules {free, occupied, ignored}; bootstrap N=1000 for all CIs.
- Every threshold (0.5 m, 30 km/h, [75%,125%], the 0.4 m resolution floor) reported as a curve, never a
  cutoff.

## Kill / pivot (each reachable — no all-publishable escape hatch)

- **BOX-ONLY MATCHES**: if box+map composition recovers the geometry (gap < 20 F1, CIs overlap),
  occupancy adds no power → FALSIFIED; report the negative as the headline.
- **UNKNOWN-POLICY FLIP** on dense GT → scope-shrink to dense-LiDAR accumulated free-space, re-test.
- **ORACLE STILL CIRCULAR** (the current state): H3 stays an internal-consistency check; report only H1.
- **DANGER-RETRIEVAL CREEP**: stop — danger needs relative motion over time = dynfield, not OccQuery.

## What the web overlay proves (and does not)

The reachable-space overlay in `web/` is the VISUAL of H1: it renders the free-space the predicates
measure, so a viewer sees what a box-only language cannot. It is **visual-data agreement** (rendered ==
measured field), NOT measurement accuracy (that needs the independent oracle H3 does not yet have). The
web is build-in-public presentation, not part of the scientific claim.
