# H3 denotation on real Occ3D-nuScenes mini — findings

What hand-labeling the occquery predicates on the 10 real Occ3D-nuScenes mini scenes (dense GT)
actually produced.

**Honest scope.** This is a **visual-agreement** pass: the ground truth is a human call on the SAME
dense-GT occupancy slice the predicate runs on, so high agreement is expected and does NOT prove
external correctness. A fully independent oracle (raw-LiDAR / image geometry, reconstructed apart
from the Occ3D field) is **v1** — that is where real denotation P/R/F1 and clearance MAE live (see
`benchmark-anchors.md` H3). Nothing here is a scientific result.

## The real payoff: real data hardened the predicates

Hand-labeling surfaced three predicate defects that were invisible on synthetic scenes — each found
by looking at a real scene where the retrieval was wrong, each fixed:

1. **`lateral_clearance` conflated a dead-ahead blockage with a lateral near-miss.** A lead vehicle
   2.2 m ahead read as 0 clearance, so 9/10 city scenes returned 0. Fix: only obstacles BESIDE the
   ego corridor (|lateral| beyond half-width) count; dead-ahead is `free_along_ego_path`'s job
   (commit `951861c`).
2. **`min_free_width` could go negative** on a centerline-straddling obstacle (`-0.00 m`). Floored at
   0 (commit `951861c`).
3. **`min_free_width` read a sub-voxel / float-residual gap (`2.8e-15 m`) as an open corridor**, so
   fully blocked scenes matched `corridor_narrows`. Gaps below one voxel now clamp to 0 = blocked
   (commit `ae95ab3`).

These three are the actual scientific value of the pass: the instrument is more correct than before,
and the failures are documented rather than hidden.

## Per-query result (dense GT, 10 mini scenes, VISUAL agreement only)

| query | retrieved | visual check |
|---|---|---|
| `corridor_narrows` | {scene-0061} | 0061 is a real ~0.8 m gap; the other 9 are wide lanes (4.8–14.4 m) or dead-ahead blockages (0103/1077, width ~0). Matches the visual call. |
| `tight_clearance_at_speed` | {scene-1077} | a 0.28 m side gap at 8.6 m/s (>30 km/h) — a plausible tight side pass. |
| `blocked_then_clears` | {scene-0103, 0655, 1077} | a temporal (blocked → clear within 3 frames) pattern over a horizon. Single-frame visual labeling is INSUFFICIENT to confirm the transition; deferred to v1 (frame-pair / trajectory review). |

## Deeper finding (2026-06-20): simple geometry confuses dead-ahead with side / corridor

Adding a measurement marker to viz (the exact spot the predicate took its reading) exposed that the
v0 retrievals are weaker than the first visual pass suggested:

- `corridor_narrows` {scene-0061}: the 0.80 m "gap" is the left/right edge of a single obstacle
  CLUMP ~9 m dead ahead, not two converging walls. `min_free_width` measures the lateral extent of a
  frontal object as if it were a corridor → effectively a **false positive**.
- `tight_clearance` {scene-1077}: the 0.28 m "side gap" is the side edge of a frontal clump ~22 m
  ahead, not a wall being passed.

**Root cause.** The v0 predicates are pure local geometry (distance / width along the straight
centerline). They have no notion of the ego PATH, of going AROUND an obstacle, or of free-space
connectivity, so they cannot separate "object dead ahead (drive around it)" from "wall beside me
(squeeze past)" from "corridor narrowing (cannot pass)". On synthetic scenes these were separated by
construction; on real city occupancy they collapse together.

**Honest consequence.** **H1 (expressivity vs RefAV) remains the strong, oracle-free result. H3
(denotation correctness) is weak under v0's simple geometry** and needs path-aware /
connected-free-region predicates (a reachable-free-space test, not a centerline scan) to be
trustworthy. That is real v1 research, not a tweak — recorded here rather than papered over.

## v0 retrieval audit (30-agent, 2026-06-20): precision 0/5

A 30-agent ultracode audit (one agent per query x scene, each running the predicate AND reading the
rendered viz image; viz axes independently verified -- known-position obstacles render correctly, and
Occ3D `semantics` is [X,Y,Z] = forward/left/up per the official convention + the road-surface
distribution) found: of 30 (query, scene) evaluations, v0 retrieved 5 scenes (corridor 1, tight 1,
blocked 3) and **all 5 are false positives -- 0 true positives, precision 0/5**. The 25 non-retrievals
are correct true-negatives, but several survive only by the speed gate or the floor-to-0 rule, not by
the metric discriminating. So the predicate has no demonstrated ability to surface a real positive,
and a 100% false-positive rate on everything it surfaced.

Five failure modes, all root-level (not per-scene):
1. **frontal-object-as-corridor** -- `min_free_width` pairs nearest-left vs nearest-right voxel at one
   station with no connectivity test; one frontal clump edge + a stray cell read as a two-sided narrows.
2. **frontal-edge-as-side-clearance** -- `lateral_clearance` takes the min over the whole 0-20 m forward
   window, so a wall far ahead (where the road bends) reads as a gap beside the ego.
3. **single-voxel-noise-as-blocked** -- no minimum cluster size; one isolated voxel (no neighbor within
   1.2 m) trips "blocked".
4. **single-frame-artifact-as-blocked** -- no temporal persistence; a 1-frame occupancy artifact
   satisfies blocked -> clears.
5. **ego-drives-past-static-point-as-clears** -- "clears" is satisfied by the ego driving past a STATIC
   voxel (closing speed == ego speed), mistaken for an obstacle pulling away.

Honest v0 claim: "correctly stays silent on scenes with no such situation, but every time it speaks it
is wrong." It is NOT a working retriever; do not quote any accuracy implying it works.

## What this is NOT

- Not a denotation P/R/F1: the GT shares the predicate's data source, so agreement is not external
  evidence. The honest headline result remains **H1 (expressivity vs RefAV)**, which needs no oracle.
- `tight_clearance` precision (0.5 m) is below reliable visual labeling on a 0.4 m voxel grid; its
  denotation MAE needs an independent metric oracle — v1.

## Next (v1) -- root-cause fixes (no band-aid: make the wrong reading unrepresentable)

1. **Connected-free-region / path-aware geometry** -- replace nearest-left/right voxel pairing with a
   free-space connectivity test: flood-fill the drivable free region from the ego footprint and measure
   the narrowest gate of the connected channel. A corridor exists only if both walls bound the SAME
   connected channel the ego is in -- structurally kills frontal-object-as-corridor and
   frontal-edge-as-side-clearance.
2. **Noise-robust occupancy** -- require a minimum cluster size / neighbor density before voxels count
   as an obstacle (a lone voxel with no neighbor must not block).
3. **Multi-frame persistence + true relative motion** -- blocked must persist >= 2 frames; clears must
   require the obstacle to actually move out of the path (track it; relative speed != ego speed), not
   merely fall out of the reach corridor because the ego drove forward past a static point.
4. **Restrict `lateral_clearance` to obstacles abeam the ego** (small |forward|), separating a side
   pass from a frontal blockage.
5. **A labeled set that CONTAINS positives** (these 10 mini scenes have none) so recall is measurable;
   add the five failure modes above as adversarial negatives in the regression suite.
6. Independent raw-LiDAR clearance/free-space oracle -> denotation MAE + P/R/F1, released with code +
   held-out scene IDs; expand from mini to nuScenes val.
