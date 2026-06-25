# Oracle-v3 — DAv2 cross-modal recall oracle: INVALID-SCALE (Gate-1 kill, 2026-06-26)

Pre-registration sealed BEFORE data: `oracle_depth_recall_preregistration.md` (commit 95ecfe4). Module
`oracle_depth_recall.py` (888 LOC, onnxruntime CPU, no torch). Self-check PASSED (ego round-trip 0.0000 m,
undistort↔redistort 0.0000 px, letterbox↔un-letterbox exact analytic inverse, DAv2 ONNX returns a metric
(518,518) map in (0,80]).

## Verdict: INVALID-SCALE — the pre-registered metric-scale falsifier (Gate 1) fired
On 16 known-good unoccluded annotation boxes (≥50 interior LiDAR pts, in band, 2–30 m) across the 3 logs:
**median |DAv2 depth − box range| = 9.838 m ≫ 0.5 m gate**, signed +9.838 m (DAv2 systematically FARTHER).
A near+far box scan shows DAv2/box ≈ **1.65× median** (clusters ~1.7–1.8 and ~0.9–0.98). → metric scale is
invalid on AV2 → STOP, no miss-rate (Gate 2 AUC and the band-local null were correctly never reached).

## Suspects ordered before the verdict (it is NOT an apparatus bug)
- Geometry exact (self-check ego round-trip 0.0000 m; the box projection lands on a 269-LiDAR-point vehicle).
- Comparison sign correct (near-face-vs-center would bias DAv2 *smaller*; observed is *larger* — opposite).
- Model output valid (range in (0,80], plausible road gradient).
→ This is a genuine metric-scale transfer failure of the frozen DAv2-metric net on AV2, not wiring. The
DAv2 weights are Virtual-KITTI (synthetic) — their absolute scale (VKITTI camera/scene geometry) does not
match AV2's. The RELATIVE depth (near<far ordering) is correct; only the absolute metric is off ~1.65×.

## What this means (the decisive A-vs-consolidate fork, resolved honestly)
The DAv2-absolute-metric recall oracle, AS SEALED, does NOT yield an external recall result on AV2. So:
- **FP-side denotation-honesty: external + RELIABLE** (traversal-v0.1) — stands.
- **Recall-side: still NO external oracle.** box-recall is same-modality consistency; stereo died at AUC
  0.259; DAv2-absolute dies at the scale gate. Recall remains consistency-only externally.
- This is a legitimate pre-registered kill (Gate 1 exists exactly to catch a scale-broken depth oracle
  before it reports a meaningless miss-rate). A negative is the headline.

## Named, re-pre-registerable fix (independence-preserving)
DAv2's relative depth is correct; only absolute scale fails. A scale correction from a source INDEPENDENT
of the LiDAR being graded restores the oracle without circularity:
- **Ground-plane rescale** (chosen next attempt): per frame, the camera sits at a known ego-frame height
  (calibration extrinsic, NOT LiDAR); DAv2 depth at road pixels + flat-ground ray geometry gives a
  per-frame scale; DAv2_corrected = DAv2 × scale. Uses calibration + image only → modality/algorithm/
  provenance independence preserved. Re-pre-registered as `oracle_depth_recall_v2` (sealed before its run).
- (Alternative, noted: rescale DAv2 by the SPARSE-but-metric stereo-pair matches — also LiDAR-independent.)
- Caveat carried forward: even with correct scale, Gate 2 (AUC ≥ 0.75) may still fail if DAv2-structure
  cannot separate human-labeled structure from empty road on dark vehicle backs — the same wall that sank
  classical stereo. That gate stays reachable.

This run is honest and uncontaminated: the seal (95ecfe4) preceded the run; Gate 1 short-circuited before
any miss-rate computation; result.json records INVALID-SCALE.
