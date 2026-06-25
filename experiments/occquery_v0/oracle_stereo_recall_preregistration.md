# Oracle-v1 — stereo-camera RECALL oracle for occupancy denotation-completeness (PRE-REGISTRATION, sealed before data 2026-06-25)

## Why
The traversal oracle (`oracle_traversal_preregistration.md` / `oracle_traversal_v0_1_preregistration.md`,
sealed 2026-06-25) is explicitly ONE-SIDED: it measures occupancy FALSE POSITIVES (hallucinated obstacles
in the ego's driven ribbon) and defers RECALL (real obstacles occupancy MISSES) to "oracle-v1 (camera
cross-modal)". This pre-reg IS oracle-v1. Recall is the safety-load-bearing half: a hallucinated obstacle
costs comfort (a phantom brake); a MISSED obstacle costs a collision. The traversal oracle structurally
cannot see a miss — the ego, by definition, did NOT drive through the missed obstacle. A DIFFERENT SENSOR
that looks where the ego did not drive is required. AV2 ships a forward STEREO pair; classical stereo gives
metric depth with NO learned model, satisfying the repo's no-torch / solo-buildable constraint.

## SUBSTRATE — declared up front (the load-bearing honesty fix)
**The only 3 AV2 logs with stereo downloaded ARE the stop-and-go FOLLOWING / danger substrate** — verified:
`201fe83b`, `2c652f9e`, `6aaf5b08` all appear in `experiments/dynfield_v0/av2_danger_logs.json` with ~157
danger frames each. There is NO free-driving stereo on disk. Consequence, stated plainly:
- This oracle's headline is **RESTRICTED to the following substrate** and labeled as such. It does NOT
  inherit the traversal oracle's free-driving cleanliness.
- The dominant following-substrate confound (a moving lead vehicle vacating space within the lookahead) is
  controlled HERE differently from the traversal oracle: the recall estimand compares occupancy to a
  *same-timestamp* stereo observation (no lookahead window), so lead-vehicle MOTION does not enter — the
  stereo and occupancy are read at the SAME frame t. The residual confound is stereo false positives
  (handled in Logical-cleanliness), not motion.
- **Pre-condition for the free-driving generalization (named, NOT a footnote):** to claim recall on a clean
  substrate, stereo for free-driving logs MUST be downloaded first (a download, not a method change — see
  RUN PLAN). Until then the claim is following-substrate-only.

## The oracle (independence by construction)
The AV2 stereo pair `stereo_front_left` / `stereo_front_right` is a PASSIVE-OPTICAL, TRIANGULATION-based
depth sensor. Occupancy is derived from ACTIVE-TOF LiDAR (`src/probe/adapters/av2_sensor.py:_voxelize`).
Independence per `docs/research-integrity.md:37` (modality AND algorithm AND provenance):
- **Modality:** passive optical triangulation vs active time-of-flight.
- **Algorithm:** block-matching disparity → `Z = f·B/d` vs voxelization of LiDAR points. No shared code.
- **Provenance:** stereo JPEGs vs LiDAR `.feather` sweeps.

NOT pristine-external (same vehicle, same timestamp, overlapping FOV) — so the honest claim is "much more
independent than the traversal oracle, not external ground truth", exactly the standard `camera_oracle.py`
already adopts. The correlated-failure mode (both degrade in rain/low-texture) is declared and bounded below.

## Prior-art differentiation (required, verified)
The DEFENSIBLE contribution is the COMPOSITION, not the components. v-disparity/Stixel free-space and
disparity-from-block-matching are well-trodden; this pre-reg does NOT claim them as novel. The un-taken slot
(verified against survey 2412.06869 §5.2.4, Ramanagopal 1707.00051, UnO 2406.08691) is: classical-stereo
geometric free-space used as a MODEL-AGNOSTIC, GROUND-TRUTH-FREE auditor of LiDAR-occupancy RECALL. Prior
camera↔LiDAR work is either camera↔camera FN (learned/2D) or LiDAR→camera-segmentation (opposite direction);
UnO measures occupancy recall but is LiDAR-self-supervised → circular. The claim is composition + honest
recall-without-GT, and the writeup must say exactly that.

## Estimand: does occupancy MISS structure the stereo independently sees?
The unit is a forward ego-frame voxel on the in-path band, read at the SAME frame t for both sensors. Per
frame: at a location where occupancy reports **FREE within the ego in-path band**, does the stereo depth map
place a **real surface inside the band** at that voxel? Such a voxel is a candidate occupancy MISS.

### Geometry (projection chain; `projection.py` math reused, AV2 calibration re-instantiated)
1. **Calibration (read from disk via pyarrow Arrow-IPC, no pandas, no learned model).** Per log, from
   `calibration/intrinsics.feather` + `calibration/egovehicle_SE3_sensor.feather`. VERIFIED VALUES
   (log `6aaf5b08`, representative; READ THE PER-LOG FEATHER for the others, fx varies ±~3 px, B ±~0.001 m):
   - `stereo_front_left`:  fx=fy=1686.79, cx=1025.56, cy=771.64, W=2048, H=1550, k1=−0.2742 k2=−0.0561 k3=0.1188.
   - `stereo_front_right`: fx=fy=1686.49, cx=1023.67, cy=770.97, W=2048, H=1550, k1=−0.2750 k2=−0.0530 k3=0.1166.
   - **BASELINE B (per log): 6aaf5b08 = 0.49905 m, 201fe83b = 0.49838 m, 2c652f9e = 0.49980 m**; separation
     is purely lateral in ego-y (Δx,Δz ≈ 1e-3..1e-4 m). Near-canonically rectified along ego-y → a
     horizontal-scanline disparity search is geometrically valid after a small rectifying alignment from the
     residual rotation between the two extrinsic quaternions. If the residual epipolar slope after alignment
     exceeds 1.0 px over the image width, the frame is dropped, logged `calib_reject`, NOT counted.
   - **Distortion is MANDATORY (the skip branch is DELETED).** Verified: corner `|k1|·(r/f)² ≈ 0.14`
     (~187 px radial displacement at the image corner) — an order of magnitude above any "negligible"
     threshold. Every pixel is undistorted with the AV2 3-coefficient radial model (`k1,k2,k3` only; the
     `Su`/pinhole-radial form — NOT a 5-coeff OpenCV model). Closed-form, pure numpy (no cv2). The sign/order
     of the radial polynomial is verified against `av2.geometry.camera` conventions before use, not assumed.
2. **Stereo depth.** For each matched pixel with disparity `d` (px, left−right): `Z = fx_left · B / d` m
   (`Z ≈ 841.6 / d` at fx≈1686.6, B≈0.499). Sub-pixel by parabolic interpolation of the 3 costs around the
   minimum. Keep only `Z ∈ [Z_min=2 m, Z_max=30 m]` (≈ `d ∈ [28, 421] px`; `Z_max` matches
   `camera_oracle._MAX_RANGE`). Back-project kept `(u,v,Z)` to the LEFT-CAMERA frame, then to EGO via
   `R_cam2ego @ p_cam + t_cam_in_ego` (the validated inverse of `projection.py:project_ego_points`).
3. **Stereo forward obstacle map.** Voxelize kept stereo 3D points into the SAME grid spec the occupancy
   adapter uses (`av2_sensor.py`: 200×200×16, 0.4 m, RANGE x/y∈[−40,40], z∈[−1,5.4]), applying the IDENTICAL
   ground / ego-self-return filters (`_ROAD_Z=0.3`, the ego cuboid `_EGO_X0..1`, `_EGO_HALF_W`) so a stereo
   "obstacle voxel" and a LiDAR "obstacle voxel" mean the same thing. A stereo-occupied voxel = ≥
   `n_stereo_min=8` back-projected points after filtering.
4. **In-path band.** `|y| ≤ ego_half_width` from `freepath.free_along_ego_path` (obstacles inflated by ego
   half-width ≈ 1.0 m ≈ `_EGO_HALF_W=1.05`), forward to `reach` capped at Z_max=30 m and the stereo FOV.
   horizon=0 (body band). Only voxels inside this band AND inside the stereo left-camera frustum (visible per
   `project_ego_points`) are in the denominator — we cannot fault occupancy for a miss the stereo could not
   have seen.

### The miss event (per voxel, in band ∩ stereo-FOV, same frame t)
- **occ_free** = occupancy reports this voxel FREE.
- **stereo_struct** = stereo back-projected ≥ n_stereo_min points into this voxel (real surface there).
- **MISS candidate** = occ_free ∧ stereo_struct.

### Estimator
`recall_miss_rate(frame) = (# MISS-candidate band∩FOV voxels) / (# stereo_struct band∩FOV voxels)`.
Report the **per-log-clustered mean** with a **log-clustered bootstrap 95% CI** (matches the traversal
oracle). Clustering is by AV2 log UUID — **only 3 stereo logs → 3 clusters → WIDE CI** (declared power
limit below).

## Pre-registered comparison (RELATIVE) — the BAND-LOCAL null (the reachable-kill fix)
A relocate-ANYWHERE null is structurally too weak: stereo_struct band∩FOV voxels are a thin target (tens of
voxels in a 200×200 grid), so random relocation of N occupied cells over the whole grid covers any target
voxel with prob ≈ N/40000 ≈ 0.005–0.10 → the shuffled miss-rate is ~0.90–0.995 BY CONSTRUCTION, and any
occupancy map with mass near the band beats it trivially. That is the S2-HARKing/unreachable-kill failure
`docs/research-integrity.md` warns about. **FIX — band-local null:** relocate the SAME count of occupied
voxels that fall **inside the in-path band ∩ FOV** to RANDOM voxels **inside the same band ∩ FOV** (the null
support = the band, not the grid), recompute `recall_miss_rate` against the SAME stereo structure map. 1000
shuffles → null distribution. The null now tests "does occupancy put its band-mass on the RIGHT band voxels
vs random band voxels" — a real recall test, not spatial coincidence.
- **Claim:** the GAP `(shuffled_miss − true_miss)` has a log-clustered bootstrap CI strictly above 0.
  Reported as the gap, never an absolute miss-rate cutoff.

## Falsifiable kill (declared before any data is seen)
**Primary kill:** if the bootstrap CI of the GAP `(shuffled − true)` against the BAND-LOCAL null INCLUDES 0,
then occupancy marks real in-path structure occupied no better than random WITHIN THE BAND → it MISSES
obstacles a passive second sensor sees → its "free path" denotation is unsafe → **FAIL**, reported as the
headline (per `docs/research-integrity.md`: a negative is a headline, not a footnote).
**Pre-stated "this observation means I am wrong":** `gap ≤ 0` on the oracle-clean subset refutes "occupancy
recalls in-path structure".
**Secondary kill (oracle insufficiency, not occupancy):** if the oracle's OWN reliability AUC (below) is
< 0.75 on the human-labeled calibration set, the stereo-presence signal cannot separate real structure from
artifacts → the oracle is INSUFFICIENT, we do NOT report a miss-rate, and we name the GPU-gated upgrade.

## Logical-cleanliness guard (the load-bearing confound)
`stereo_struct ∧ occ_free` has TWO explanations; only one is an occupancy miss:
1. **Occupancy MISS** (signal): a real obstacle is there; LiDAR/voxelization dropped it.
2. **Stereo FALSE POSITIVE** (confound): low-texture / repetitive / depth-bleeding edge gave a spurious
   disparity; nothing is there; occupancy is CORRECTLY free.
Guards, all sealed:
- **Confidence filter (n_stereo_min=8 + left-right consistency + uniqueness).** A voxel counts as
  `stereo_struct` ONLY if (a) ≥8 back-projected points support it, (b) pixels passed the **left↔right
  consistency check** (`|d_LR − d_RL| ≤ 1.0 px`), and (c) the cost peak-ratio (best/second-best) ≤ 0.85
  (uniqueness — kills repetitive-texture aliasing).
- **Edge-adjacency reject (the silhouette depth-bleed fix).** On the following substrate the dominant stereo
  FP is depth bleeding at the LEAD-VEHICLE silhouette edge projecting into the free band beside it. DROP any
  stereo voxel within 1 voxel of a large depth discontinuity (|ΔZ| > 1.5 m between neighbouring matched
  pixels). This is added precisely because the three filters above do NOT catch silhouette bleed.
- **Texture gate.** A pixel enters matching only where the local left-image gradient (the SAME
  `camera_oracle.py:patch_evidence` Sobel statistic) exceeds τ_tex, set at the 40th percentile of in-band
  gradient on a HELD-OUT log (one of the 3, NOT a result log). Low-texture regions — where BOTH stereo AND
  LiDAR fail together — are EXCLUDED from the denominator (honest scope restriction, not a contaminant).
- **Headline on the oracle-clean subset.** PRIMARY result = voxels passing all stereo-confidence filters +
  edge-reject + texture gate. The raw (unfiltered) miss-rate is reported too as the loose upper number, but
  the falsifiable claim rides on the clean subset.

## Correlated-failure caveat (declared; direction of bias stated)
Stereo and LiDAR share failure modes (heavy rain, glare, TEXTURELESS surfaces). Consequence:
- A textureless real obstacle (blank truck side, clean wall) is invisible to BOTH → stereo also misses it →
  NOT counted. So measured `true_miss_rate` is a **LOWER BOUND** on occupancy's true miss-rate; the both-fail
  cases are silently dropped, biasing occupancy to look BETTER. Stated explicitly; never claimed as the full
  rate.
- **Restriction over inflation:** restrict the confirmatory claim to TEXTURED, CLEAR frames (texture gate +
  a frame-level filter: drop frames whose in-band stereo valid-disparity fraction < 0.30, a rain/glare proxy;
  sealed threshold). Dropped frames logged with reason.

## The oracle's OWN reliability — small human-labeled calibration set
Before the stereo-presence signal is trusted to GRADE occupancy, measure how well it separates real
structure from no-structure (as `camera_oracle.py` calibrates its AUC).
- **Set:** 60 image patches across the 3 stereo logs, sampled by a script BEFORE any are looked at:
  30 POSITIVE = patches where a tracked AV2 annotation box (`annotations.feather`, ego-frame) projects into
  the left stereo image in the in-path band at ≤30 m; 30 NEGATIVE = random lower-image drivable-road patches
  in the band. Boxes are used ONLY to build calibration labels and ONLY to measure the oracle — NOT in the
  miss-rate estimand (label-free), so the result stays box-independent.
- **Human label pass:** I label each of the 60 by eye (the photo is the primary source) to catch annotation
  gaps; disagreements with the box label are kept as human truth.
- **Metric:** ROC AUC of `stereo_struct` (valid-disparity-count after all filters) via
  `camera_oracle._roc_auc` (Mann-Whitney). **Gate: AUC ≥ 0.75** to proceed; AUC < 0.75 → secondary kill.
  Also report the operating-point precision at n_stereo_min so the FP-leak rate into the miss-rate
  denominator is QUANTIFIED (the adversary's required addition).
- **Pre-registration cleanliness:** calibration AUC + gate evaluated BEFORE the shuffled-null comparison;
  calibration logs/frames disjoint from the threshold-tuning held-out log; τ_tex and clear-weather fraction
  fixed on the held-out log, never selected using the result logs' miss-rate (the S2 failure this repo caught).

## Honest scope / ceiling (declared)
- **Power:** only **3 of 28 local logs have stereo** (`201fe83b`, `2c652f9e`, `6aaf5b08`, ~319 stereo
  frame-pairs each ≈ ~957 pairs). 3 clusters → WIDE CI. A passing gap is suggestive, not definitive. Named
  next step: pull stereo for the remaining danger logs AND for free-driving logs (a download, AV2 ships
  stereo for the full Sensor split).
- **Following-substrate only** (see SUBSTRATE). Free-driving generalization gated on the download.
- **Forward-only:** the stereo pair is forward-facing; recall certified only in the forward in-path band.
- **Lower bound:** measured miss-rate UNDER-counts (correlated textureless failures dropped). A floor.
- **Not pristine truth:** same vehicle/timestamp/overlapping-FOV → "much more independent, not external truth."
- **Not a danger/safety claim:** occupancy denotation-COMPLETENESS (occquery H3-family, recall half),
  label-free in the estimand.

## GPU-gated fallback (named, NOT taken without sign-off) — the collaboration boundary
Classical stereo is FEASIBLE here: cv2 is ABSENT from `.venv` (verified `ModuleNotFoundError`), so the
**numpy census/SAD matcher is the PRIMARY path, not a fallback**; baseline ~0.499 m and full calibration are
present; ~957 pairs on disk; numpy 2.4.6 / scipy / Pillow 12.2.0 / pyarrow 24.0.0 present (pandas absent —
read feathers via `pyarrow.feather.read_table(...).to_pylist()`). IF classical stereo proves infeasible at
first run (calibration missing on a target log, baseline degenerate, or the AUC gate fails because untextured
danger scenes dominate), the ONLY known upgrade is a LEARNED monocular/stereo metric-depth network, which
needs torch + GPU and crosses the repo's no-torch rule. That is the explicit COLLABORATION BOUNDARY: named
here, NOT implemented, requiring (a) human sign-off for a GPU dependency and (b) re-pre-registration (a
learned depth model is a different, less-independent, trained-on-similar-data oracle whose independence
accounting must be redone).

## Pre-registered run command (sealed)
```
.venv/bin/python experiments/occquery_v0/oracle_stereo_recall.py \
  --logs 201fe83b-7dd7-38f4-9d26-7b4a668638a9 \
         2c652f9e-8db8-3572-aa49-fae1344a875b \
         6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
  --heldout-threshold-log 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
  --z-min 2.0 --z-max 30.0 --n-stereo-min 8 --lr-consistency-px 1.0 \
  --edge-discontinuity-m 1.5 --null band-local --shuffles 1000 --seed 0 \
  --out experiments/occquery_v0/results/oracle_stereo_recall.json
```
(The held-out threshold log's miss-rate is reported separately and NOT pooled into the confirmatory 2-log
gap, so thresholds are never tuned on the headline logs. NOTE: the matcher reads stereo images by
NEAREST-timestamp to each danger frame — camera frames are ~20 Hz and offset from the LiDAR/danger
timestamps; exact-timestamp matching does NOT work and is a known pitfall.)

## Independence ledger (sealed)
| axis        | occupancy (graded)            | stereo oracle (grader)                  | independent? |
|-------------|-------------------------------|------------------------------------------|--------------|
| modality    | active TOF LiDAR              | passive optical stereo triangulation     | YES          |
| algorithm   | point→voxel (`_voxelize`)     | block-match disparity → Z=fB/d → voxel    | YES          |
| provenance  | LiDAR `.feather` sweeps       | `stereo_front_{left,right}` JPEGs         | YES          |
| platform    | same AV2 vehicle/timestamp    | same AV2 vehicle/timestamp                | NO (declared)|
Verdict: differs on modality AND algorithm AND provenance (`research-integrity.md:37` bar); shares platform →
"much more independent, not external truth". Correlated textureless failure → texture gate + lower-bound
framing; silhouette depth-bleed → edge-adjacency reject.
