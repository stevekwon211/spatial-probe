# Oracle-v2 — LEARNED-stereo (IGEV) RECALL oracle, GPU pod (PRE-REGISTRATION, seal before data 2026-06-27)

Third (and decisive) attempt at the EXTERNAL recall half of occquery H3 — "does occupancy MISS real
obstacles?" — the half that the program has, so far, only been able to grade with a same-modality
(box-recall) consistency check. Two external attempts were pre-registered and KILLED honestly:
- **classical stereo** (`oracle_stereo_recall_preregistration.md`, git 6fdbf5c): AUC 0.259 on the 3
  following logs — failed on **DENSITY** (census/SAD cannot match the textureless, dark, backlit
  lead-vehicle backs). Scale was never the problem: classical stereo has EXACT metric scale from the
  0.5 m baseline geometry.
- **frozen DAv2-metric mono-depth** (`oracle_depth_recall_preregistration.md`): INVALID-SCALE (>9 m
  error, VKITTI-trained depth not metrically self-consistent on AV2, even ground-plane-rescaled) —
  failed on **SCALE**.

First-principles consequence: the two failures have *disjoint* causes (density vs scale). A method that
keeps stereo's exact geometric scale while fixing its density problem would close both at once. That
method is **a learned dense stereo matcher** (IGEV-Stereo): it predicts a dense disparity everywhere,
including on low-texture surfaces (learned priors / iterative geometry-encoding volume), and the depth
is still `Z = f·B/d` with the **exact 0.5 m baseline** — scale stays geometric, not learned. So this
run changes EXACTLY ONE thing in the depth front-end (census/SAD → IGEV-Stereo) and re-runs the
already-sealed oracle on the SAME substrate. Nothing below was chosen after seeing a v2 number.

## The single variable (declared in full, so it is not a hidden bundle)
The ONLY change vs the sealed classical run is **the depth front-end** = "the matcher". That front-end
necessarily carries three coupled, matcher-intrinsic pieces, all pre-registered here:
1. **Disparity algorithm:** census/SAD block-matching → **IGEV-Stereo**, official **Scene-Flow
   generalization checkpoint** (`sceneflow.pth`), used **zero-shot**. Scene Flow is *synthetic* and
   contains **no real driving** — the strongest independence-of-provenance choice (it has never seen
   AV2, nuScenes, or even KITTI). A driving-finetuned checkpoint would transfer better but weaken
   independence; it is therefore **NOT used here** and may only be a SEPARATELY pre-registered
   follow-up *if* the Scene-Flow checkpoint fails the AUC gate (declared, to forbid checkpoint-shopping).
2. **Rectification:** the classical path undistorted to normalized rays and matched in the raw
   geometry; a learned net requires horizontally-rectified epipolar pairs. So a standard stereo
   rectification (from the AV2 per-log stereo extrinsics/intrinsics) is added as the depth front-end's
   input step, and `Z = f_rect·B/d_rect`; the resulting 3-D points are returned to the ego frame by the
   inverse rectification rotation. Geometry only — no new estimand.
3. **Validity mask:** the census texture-gate (40th-pct) existed *only because* census is unreliable on
   low texture — it is a property of the census matcher, not of the estimand, and keeping it would
   suppress the very textureless pixels we now want to recover. It is therefore **dropped** and
   replaced by IGEV's own validity (left-right consistency at the SAME `tol = 1.0 px`, kept verbatim,
   run symmetrically L→R and R→L; plus IGEV's finite-disparity mask). This is part of "the matcher".

**EVERYTHING downstream is inherited byte-for-byte** from `oracle_stereo_recall.py` (sealed): the
estimand (occ_free ∧ stereo_struct in the in-path band∩FOV, same frame t), `z∈[2,30] m`, `|y|<3 m`,
`n_stereo_min = 8`, `edge-discontinuity 1.5 m`, voxelize-with-the-SAME-`av2_sensor._voxelize`-filters
(a stereo obstacle voxel ≡ a LiDAR obstacle voxel), the **band-local null** (1000 shuffles, seed 0),
the log-clustered bootstrap, the **AUC ≥ 0.75 calibration gate** on the **same 60 human-labeled
patches already labeled for the classical run** (same labels, new matcher — clean), the held-out
threshold log `6aaf5b08`, and the kill rule.

## Substrate (sealed) — identical to the classical run, on purpose
The SAME 3 following logs where classical stereo scored AUC 0.259:
`201fe83b…`, `2c652f9e…`, `6aaf5b08…` (held-out threshold = `6aaf5b08…`). Same logs ⇒ apples-to-apples;
the matcher is the only thing that moved. These logs HAVE in-path obstacles (lead vehicles) — the
exact textureless-back targets census missed — so vacuity (the free-driving failure mode) does not
apply here.

## Independence ledger
- **Modality:** passive optical stereo ≠ active LiDAR TOF.
- **Algorithm:** deep stereo matching (IGEV iterative geometry-encoding volume) ≠ `av2_sensor._voxelize`.
- **Provenance:** weights trained on **synthetic Scene Flow only** ≠ AV2 / any real driving; the graded
  occupancy is LiDAR-voxelized AV2. The two share only the platform (same vehicle/timestamp) — "much
  more independent than same-modality box-recall, not absolute external GT", same honest scope as the
  classical pre-reg.
- Scale is **geometric (0.5 m baseline)**, not model-output — so this is NOT the DAv2 scale-circularity.

## Calibration gate (self-reliability, evaluated FIRST, before the null comparison)
IGEV's `stereo_struct` signal (post-filter valid-disparity support) on the 60 human-labeled patches →
`_roc_auc` (VERBATIM) vs the human labels. **Gate: AUC ≥ 0.75 → proceed; < 0.75 → ORACLE-INSUFFICIENT**
(even SOTA zero-shot learned stereo cannot separate GT-surface from road on these following-distance
backs ⇒ passive stereo recall is matcher-independent dead on this substrate).

## Kill (reachable, declared before data)
- **ORACLE-INSUFFICIENT** iff IGEV AUC < 0.75 → external stereo recall CLOSED regardless of matcher
  quality; recall stays consistency-only externally, escalate only to a *trained AV-domain metric-depth*
  model (separate pre-reg) or curated substrate. This is a real, publishable negative.
- **FAIL** iff AUC ≥ 0.75 but the GAP `(band-local-shuffled − true)` bootstrap CI **includes 0** →
  occupancy's in-path recall is no better than band-local chance relocation, i.e. occupancy does NOT
  measurably miss more obstacles than a null that preserves band occupancy mass. Also a real finding.
- **RECALL-SUPPORTED** iff AUC ≥ 0.75 AND gap CI.lo **> 0** → the EXTERNAL recall half is finally
  achieved: occupancy demonstrably places FREE where an independent passive-stereo surface exists, more
  than chance. This is the moat's missing half opening — and would update the repo H3 framing (external
  recall would no longer be "honestly closed").

## "This observation means I am wrong"
If IGEV-Stereo (SOTA dense learned stereo) ALSO yields AUC < 0.75 separating GT-box surfaces from road
on these frames, then dark backlit lead-vehicle backs at following distance are unrecoverable by ANY
passive stereo, and the density hypothesis for the classical failure is *confirmed but unfixable by
matcher* — stereo recall is closed, full stop (consistent with escalating to active-sensor or trained
metric-depth). If the gap CI includes 0 despite a passing AUC, occupancy is honestly not missing
obstacles here and the recall-gap claim dies on this substrate.

## Architecture (no torch in the repo; pod is ephemeral)
The repo core/oracle stay **pure numpy** (CLAUDE.md: no torch in `.venv`). torch/IGEV run ONLY on the
external **RunPod GPU pod**, which consumes the stereo JPEGs + per-log calibration and emits, per
(log, frame, side), a **rectified disparity artifact** (`disp_<log>_<ts>_<side>.npz`, float32 +
validity) plus the rectification metadata. The artifacts are gitignored (data/checkpoints rule). The
**local numpy oracle** then grades them via a new `--disparity-source artifact <dir>` switch that
replaces the 4 `compute_disparity*` call sites with an artifact loader — so the confirmatory **re-grade
is reproducible locally with NO GPU**, and the IGEV step is a declared external preprocessing front-end.

## Run (after THIS doc is committed; once)
```sh
# on the GPU pod (torch + IGEV-Stereo sceneflow.pth), produces rectified disparity artifacts:
python experiments/occquery_v0/igev_disparity_pod.py \
  --logs 201fe83b… 2c652f9e… 6aaf5b08… \
  --checkpoint sceneflow.pth --out-dir results/igev_disp
# locally (numpy only), the SEALED confirmatory, matcher = artifact:
.venv/bin/python experiments/occquery_v0/oracle_stereo_recall.py \
  --logs 201fe83b… 2c652f9e… 6aaf5b08… --heldout-threshold-log 6aaf5b08… \
  --z-min 2.0 --z-max 30.0 --n-stereo-min 8 --lr-consistency-px 1.0 \
  --edge-discontinuity-m 1.5 --null band-local --shuffles 1000 --seed 0 \
  --disparity-source artifact results/igev_disp \
  --calib-json results/calib_patches/calib_patches.json \
  --out experiments/occquery_v0/results/oracle_stereo_recall_learned.json
```

## Honest scope
Cross-modal-ish (passive stereo vs active LiDAR), far more independent than box-recall, NOT absolute
external GT. Measured miss-rate is a LOWER BOUND (correlated stereo failures dropped with the validity
mask). The calibration AUC is necessary-not-sufficient (the 60-patch box-derived/human labels cannot
catch annotation gaps). A clean ORACLE-INSUFFICIENT or FAIL is the headline, reported as such — no
re-inflation of H3.
