# Oracle-v1.1 — stereo recall on FREE-DRIVING: INDETERMINATE-VACUOUS (2026-06-26)

Pre-registration sealed before data: `oracle_stereo_recall_freedriving_preregistration.md`. Downloaded
fresh stereo+calib for 4 free-driving held-out logs via s5cmd (no auth/GPU). One log (`c222c78d`) DROPPED
as an apparatus exclusion: its `egovehicle_SE3_sensor.feather` lacks the stereo_front_{left,right} rows
(ring cameras only) so the stereo geometry can't be built — verified by content (`load_stereo_calib` raises
KeyError; the other 3 load fine). Ran on the 3 usable logs (d5d6f11c, c2d44a70, 27be7d34).

## Verdict: INDETERMINATE-VACUOUS (the pre-registered vacuity outcome)
The sealed calibration sampler produced **0 positive patches** (30 neg only) → calibration AUC = NaN
(n_pos=0) → no recall miss-rate computable. Root cause, verified before the run: free-driving = a clean
ego path, so there are **almost no in-path obstacles to grade** (in-path LiDAR-seen boxes per log = 0, 22,
0; even c2d44a70's 22 are mostly lateral/outside the narrow stereo in-path band ⇒ 0 in-band positives).

## The finding (this IS the result, not a failure to hide)
External cross-modal recall via classical stereo is structurally stuck on the two AV2 substrates available
solo/CPU:
- **following** (the 3 danger logs): obstacles PRESENT but textureless/dark vehicle backs → stereo density
  fails (AUC 0.259).
- **free-driving** (these held-out logs): structure textured/lit BUT the clean path has **no in-path
  obstacles** → the recall test is vacuous (n_pos=0).
The fix the stereo pre-reg named (switch to free-driving) removes the very obstacles the recall test needs.
A substrate with BOTH many in-path obstacles AND good texture/lighting (e.g. dense daytime urban traffic)
would be needed — not cheaply curatable from what's on disk, and the metric-depth (DAv2) route is also
killed (scale). So: **solo/CPU external recall is closed on available AV2 substrates** — consistent across
all three attempts (stereo-following density, DAv2 scale, stereo-free-driving vacuity).

## Net (unchanged, now triangulated 3 ways)
Denotation-honesty ships FP-side EXTERNAL+RELIABLE (traversal) + recall-side CONSISTENCY-ONLY (box-recall).
External recall is honestly deferred to a GPU/curated-substrate path. Three independent pre-registered
attempts, three honest negatives — the conclusion is robust, not a single unlucky run.
