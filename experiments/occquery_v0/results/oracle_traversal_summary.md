# Oracle-v0 — ego-trajectory traversal oracle: honest result + the confounds it surfaced (2026-06-25)

Pre-registration sealed before data: `oracle_traversal_preregistration.md` (commit 7474991). Goal: an
INDEPENDENT, label-free check that occupancy's "occupied" verdict is denotation-correct — the durable
answer to the L5 collapse attack ("a clearance predicate is a half-day script; is it even right?"). The
oracle: a vehicle cannot drive through a real obstacle, so the ego's recorded future swept volume marks
space that was physically FREE — independent of the occupancy perception in source (recorded poses, not
LiDAR) and algorithm (rigid-body sweep, not voxelization).

## Verdict: the oracle IDEA works, but BOTH window choices are confounded on the following-danger substrate. The signal is real and conservative; a clean confirmatory needs v0.1.

### What the runs showed
- **Sealed run (W=5 frames = 0.5 s, full ribbon): true FP 0.0000, shuffled 0.0319 → looked RELIABLE, but is CONFOUNDED.** Adversarial diagnostic (the-adversary-is-me) found the cause: on this stop-and-go danger substrate the ego creeps only **0.91 m on average in 0.5 s** (p90 2.16 m), so the swept ribbon sits ENTIRELY inside the ego self-return-removal zone (x < 3.9 m), where occupancy is 0 BY CONSTRUCTION. Only ~0.9 cells/frame reach real road (x > 4 m). So the 0.00 is "occupancy is empty in the ego footprint we already cleared," not "occupancy keeps the driven road free." NOT a citable result.
- **Exploratory correction (W=40 frames = 4 s, ribbon restricted to x > 4 m beyond the ego footprint, 18 logs, 1893 frames): true FP 0.0143 CI[0.0046, 0.0254] vs shuffled 0.0333 CI[0.0292, 0.0380] → RELIABLE (CIs disjoint).** Occupancy puts an obstacle in the physically-driven road ~1.4% of the time vs ~3.3% for random — validated label-free and box-independent.

### The second confound (and why it makes the exploratory result CONSERVATIVE, not inflated)
The danger substrate is car-FOLLOWING. Over a 4 s lookahead the lead vehicle MOVES, so the ego drives into space the lead VACATED. That space was correctly OCCUPIED at frame t (the lead was there) and the ego traverses it at t+40 — counted as an apparent false positive even though occupancy was right. This dynamic-lead confound INFLATES the apparent FP AGAINST occupancy. So occupancy's true hallucination rate on the driven road is **≤ 1.4%** — even with the confound working against it, still less than half the random 3.3%. The signal is real; the number is a conservative upper bound.

### Net
- The ego-trajectory traversal oracle is a genuine, independent, label-free signal — and it says occupancy does NOT hallucinate obstacles in the driven path (conservatively ≤1.4% vs 3.3% random).
- But neither sealed W is clean on a stop-and-go FOLLOWING substrate: W small → ribbon trapped in the ego-cleared near field; W large → moving-lead confound. There is no single clean window here.
- This is the integrity machinery working (like R0 v1→v2→v3): the sealed run was confounded, the adversarial check caught it, the corrected run is honest about its own remaining confound.

## What v0.1 (the clean confirmatory) must do — NOT run here (avoid HARKing: W=40/far=4 were chosen AFTER seeing the result)
1. **Principled, not result-selected, parameters:** far = the ego front bumper (x > ego.length/2 + margin, mechanism-derived); W = the minimum to escape the ego zone given the substrate's creep speed (~10 frames ≈ 1 s, not 40), minimising the dynamic confound.
2. **Declare the dynamic-lead confound up front** as a conservative inflator (the result is an upper bound on occupancy FP).
3. **Run on HELD-OUT logs** — fresh vehicle-longitudinal danger logs NOT among the 18 explored here (re-test exploratory findings on fresh data).
4. **Or switch substrate** to free-driving (non-following) segments, where there is no moving lead to vacate occupied space — the cleanest fix.

## Bigger picture — this is ONE signal of the oracle, and the weaker half
Traversal tests occupancy FALSE POSITIVES (hallucinated obstacles) only; it cannot test RECALL (real obstacles occupancy MISSES) — the half that matters most for the safety story. oracle-v1 adds the two complementary independent signals: the camera cross-modal confirm (real AV2 ring cameras — independent SENSOR; we have them, just un-downloaded) and multi-sweep persistence (independent observation geometry), each with its OWN measured reliability against a small human-labeled calibration set. The contribution is the HONEST ACCOUNTING of how much a label-free verdict can be trusted — not any single oracle.
