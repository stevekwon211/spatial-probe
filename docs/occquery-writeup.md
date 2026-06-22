# Can a box-only query language denote the space between boxes? Usually not.

_A build-in-public note on OccQuery. Draft — kept honest, not polished for reach._

AV scenario mining is box-native. RefAV, nuPlan, Waymo — you query over tracked-object cuboids and
HD-map polygons. "find me a pedestrian crossing in front of a turning vehicle." That works because a
pedestrian and a vehicle are boxes.

A whole class of safety geometry is not a box. The physical gap to an unboxed wall. The free width
left in a narrowing corridor. Whether the ego's swept footprint stays in reachable free space. This
lives in the space *between* boxes. I wanted to know one thing: can a box+map query language even
denote it? Usually the honest answer to "does the representation store this?" is: it can't. I tried
to prove that precisely.

## What I am claiming, and what I am not

I am claiming: two scenes that are identical in every box+map observable can differ only by unboxed
occupancy, so no function of those observables can tell them apart, while an occupancy predicate can.
That is non-identifiability under the box+map observation set.

I am not claiming that no composition of RefAV's functions could ever correlate with free-width on
some scenes. A two-function witness plus a grep cannot prove that. The scope fence goes before the
result, not after it.

## The witness is the proof

I build two scenes. Same tracked vehicle box (center, size, yaw, class, velocity). Same empty map.
The only difference is an unboxed obstacle — a wall the tracker never boxes. The box backend
(`distance_to_nearest_object`) returns 20.0 m for both, because its inputs are identical. The
occupancy predicate (`lateral_clearance`, `free_along_ego_path`, `min_free_width_along_path`) splits
them.

No oracle. No dataset. No model. The separation holds by construction and anyone can re-run it
(`tests/test_expressivity.py`). The numbers: RefAV's released function set has 32 functions and 0
free-space primitives. 8 of 8 witness tests pass. 20 of 24 pre-registered queries are inexpressible
in that set.

## Where I was wrong, on purpose, in public

My first real-data pass reported precision 0/5. Every retrieval looked like a false positive. The
easy move is to tune the predicate until the number goes up.

I opened scene-0061 in the 3D viewer instead. The "corridor false positive" was a lead car — the ego
follows it, decelerating 5.14 → 2.27 m/s over the window, staying 7–8 m ahead. The 0.80 m free-width
my predicate measured beside it was a *correct* measurement. I was grading a static geometric
measurement against a dynamic-danger answer key. That was my bug, not the predicate's.

So I narrowed the instrument's job. OccQuery measures static geometry. Whether a measured gap is
*dangerous* — a wall vs a lead car, a near-miss vs a parked car — needs relative motion over time,
which is a different experiment (dynfield). I refused the danger claim by construction rather than
fix it per-scene.

## The negative I will not hide

I wanted a denotation-correctness number: do the predicates compute the geometry *correctly* on real
data? That needs an oracle — an independent answer key.

The obvious oracle is circular. Re-deriving clearance from the same Occ3D voxels (built from the same
accumulated LiDAR the predicate reads) measures consistency, not truth. On static geometry, raw LiDAR
re-measures the same walls with the same sensor. So I demoted H3 to an internal-consistency check
before I looked at any val data, and I left the independent oracle unbuilt rather than dress up a
tautology as validation. In code, the `lateral_clearance` residual is labeled a tautology before it
is computed; it comes out identically 0 across 404 frames, exactly as a same-algorithm check should.

Then I tried a camera oracle — a different sensor (passive optical vs active LiDAR). I verified the
projection chain on real calibration (tracked boxes land on the actual objects in the photo). But a
pure-CV presence check failed its own reliability gate: ROC AUC 0.798, below the 0.80 I set before
running. And the forward agreement number (0.896) is vacuous — the mini set has zero centerline
blockages, so it is dominated by "both say free." I report it as a consistency cross-check, not
evidence. The honest ceiling on this hardware is a consistency probe, not external truth.

I also found, after the fact, that I had written "33 functions" in four files. RefAV has 32. I fixed
it. A wrong number is a wrong number even when it does not change the logic.

## The discipline is in the git history, not the prose

Pre-registration committed before data (timestamped). Kill criteria stated before running. Queries
sealed before the val run, which has not happened. No 0.90 F1 gate; the relative gap is a curve, not
a movable cutoff. Zero-positive truth returns NaN, never a vacuous 1.0. The residual-0 tautology
labeled as such in the source. You can check all of it in the diffs.

## Decision, and what is open

H1 is the result: an oracle-free expressivity separation, proven by construction. PURSUE.

H3's real number is one val-data download plus a sealed positive-containing scene set away — the 10
mini scenes have 0 positives, so recall is literally unmeasurable today. H2 (the RefAV HOTA-Temporal
leaderboard) is blocked on a substrate mismatch I chose not to paper over: OccQuery's substrate is
Occ3D-nuScenes, RefAV's is Argoverse 2.

Breadth was never the point. I could have spread thin across six planned experiments. The judgment —
pre-register, catch your own category error on real data, refuse a circular oracle, report the failed
AUC — is the thing worth showing.

## Limitations as boundaries, not softeners

- 0.4 m voxel resolution is a ceiling. A 0.07 m parked-beside gap cannot be told from contact; no
  camera method on this rig beats it either. Sub-voxel clearance is out of scope, not a future tweak.
- Danger is a boundary the instrument does not cross. It measures; dynfield judges.
- The external SOTA numbers (HOTA-Temporal 53.12 / 52.37) I cite from the leaderboard, I have not
  personally re-verified. Treat them as not locally reproducible.

---

_Instrument + experiment: pure numpy/scipy, no torch, runs on a laptop. Code, witnesses, and the
sealed pre-registration are in the repo; the synthetic runner is a smoke test, never evidence._
