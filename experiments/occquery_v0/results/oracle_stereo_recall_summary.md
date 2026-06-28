# Oracle-v1 — stereo-camera RECALL oracle: ORACLE-INSUFFICIENT (secondary kill, 2026-06-25)

> **CORRECTION (2026-06-28): the AUC below (0.259) is a `_roc_auc` tie-bug artifact.** The original scorer
> did not average tied ranks; the census score is an integer pixel-count with many ties, so the AUC was
> deflated. Re-graded with the fixed metric + a bootstrap CI: **AUC 0.598, 95% CI [0.517, 0.679]** — still
> ORACLE-INSUFFICIENT (the whole CI sits below the 0.75 gate, so census fails the gate *confidently*, not at
> "below chance"). The "below-chance / density-limited 0.259" reading is retracted. Bug fixed in
> `camera_oracle.py`; gate now reports `auc_ci95`. (IGEV sceneflow 0.662 / kitti 0.733 are also deflated
> lower bounds — settling those needs a GPU re-grade, currently deferred on RunPod provisioning.)

Pre-registration sealed BEFORE data: `oracle_stereo_recall_preregistration.md` (commit 6fdbf5c). Module
`oracle_stereo_recall.py` (commit 4e50a39), pure numpy, geometry self-checked (ego-point round-trip 0.0000m,
corner undistort↔redistort 0.0000px). 60-patch calibration set human-labeled by the repo owner (34 structure /
26 empty; 30/30 box-patches confirmed structure + 4 "random road" patches the human found to actually contain
structure — the pre-reg's annotation-gap catch working).

Run (once, sealed CLI): `--logs <3 stereo logs> --heldout-threshold-log 6aaf5b08 --z-min 2 --z-max 30
--n-stereo-min 8 --lr-consistency-px 1.0 --edge-discontinuity-m 1.5 --null band-local --shuffles 1000 --seed 0`
(downsample=2, the builder's declared default per the pre-reg's impl note).

## Verdict: ORACLE-INSUFFICIENT — the pre-registered SECONDARY KILL fired
**Calibration AUC = 0.259 < 0.75 → secondary kill → NO miss-rate reported.** This is a legitimate sealed
outcome (the pre-reg's "AUC < 0.75 → the stereo-presence signal cannot separate real structure from artifacts
→ the oracle is INSUFFICIENT, do NOT report a miss-rate, name the GPU-gated upgrade"). The recall question is
left UNANSWERED — the band-local-null miss-rate gap was never computed (correctly gated off).

## Diagnosis (does NOT change the sealed verdict; it directs the next step)
The two calibration numbers are in tension and together locate the failure:
- `auc = 0.259` (below 0.5 → human-labeled STRUCTURE patches got FEWER valid stereo disparities than empty-road
  patches).
- `operating_point_precision_at_n_stereo_min = 0.857` (of patches that DID fire ≥8 disparities, 86% are
  structure).
Reading: the matcher is **not random** (when it fires, it fires on structure — precision 0.86), but it is
**low-sensitivity** on this substrate — most structure patches never reach 8 disparities (smooth/dark lead-vehicle
backs, textureless following-danger scenes), so textured road outranks them and drags the AUC below 0.5. This is
the pre-reg's own correlated-failure caveat made real ("a textureless real obstacle is invisible to BOTH stereo
and LiDAR"), compounded by the ds=2 downsample removing matchable detail. So this is most likely a genuine
substrate/resolution insufficiency, NOT purely an apparatus bug — i.e. exactly the scenario the secondary kill
was pre-registered to catch.

## Net (two-sided, honest)
- **FALSE-POSITIVE side (traversal-v0.1): RELIABLE** — occupancy does not hallucinate obstacles in the driven
  ribbon (true FP 0.0000 vs shuffled 0.0357, held-out free-driving). See `oracle_traversal_summary.md`.
- **RECALL side (this oracle): INCONCLUSIVE** — the classical-stereo oracle is insufficient to grade occupancy
  recall on the following-danger substrate at ds=2. The miss-side of the safety story remains unmeasured.

## Next (ALL require RE-PRE-REGISTRATION — no quiet retry; that would be HARKing)
The miss-rate estimand data was never read (killed at the AUC gate), so the confirmatory is uncontaminated; but
any parameter/oracle change to chase a passing AUC must be re-pre-registered (the pre-reg states this, and the
verdict was reached on data we've now seen). Options, owner's call:
1. **ds=1 full-resolution retry** — cheapest; more matchable detail may lift structure-patch sensitivity.
   Re-pre-register with downsample sealed at 1 and the same kill criteria.
2. **GPU-gated learned monocular/stereo depth** — the pre-reg's named fallback; needs a torch+GPU dependency
   (crosses the repo no-torch rule), human sign-off, and a fresh independence accounting (a learned model is a
   less-independent, trained-on-similar-data oracle).
3. **Free-driving stereo** — download stereo for non-following logs (a download, not a method change) to test
   recall on a substrate where structure is better-textured and not backlit; lifts the following-only
   restriction too.

## Apparatus note (process integrity)
The first confirmatory run crashed (FileNotFoundError on stereo jpgs). Root cause was the HTML labeling helper,
not the oracle: it round-tripped the 18–19-digit nanosecond timestamps through JavaScript `JSON.parse`, where
float64 (> 2^53) silently rounded them (e.g. …927211 → …927230). The human labels (`id` strings + 0/1) were
intact; recovery merged them by `id` onto a freshly re-emitted (deterministic, seed 0) pristine manifest, and
the labeler now embeds timestamps as strings so the corruption is unrepresentable. The labels graded here are
exactly the owner's.
