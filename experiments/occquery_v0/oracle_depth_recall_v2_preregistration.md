# Oracle-v3.1 — DAv2 recall oracle with GROUND-PLANE rescale (PRE-REGISTRATION, seal before data 2026-06-26)

Re-pre-registration after `oracle_depth_recall_preregistration.md` (v3) returned **INVALID-SCALE** (Gate-1:
DAv2-VKITTI absolute metric over-estimates AV2 ranges ~1.65×, median 9.84 m). v3's RELATIVE depth is
correct (geometry exact, near<far ordering holds); ONLY the absolute scale fails. v3.1 adds a per-frame
scale correction from a source INDEPENDENT of the LiDAR being graded, then re-applies the SAME sealed
gates. Everything else is inherited from v3 verbatim. Nothing below was chosen after seeing a v3.1
miss-rate; the v3.1 scale method was chosen from v3's scale-direction finding (a known mono-metric
property), NOT from any recall number.

## The rescale (the only change vs v3) — independence-preserving
Per frame: the stereo_front_left camera sits at a known ego-frame height `h = |t_z|` from the calibration
extrinsic (`egovehicle_SE3_sensor`, NOT LiDAR). For road pixels in a geometric road prior (the lower-
center image wedge: v ∈ [0.75·H, 0.97·H], u ∈ [0.35·W, 0.65·W] of the undistorted image — a flat-ground
prior, no segmentation/LiDAR), each ray direction in ego frame `d_ego = R_cam2ego · K⁻¹·[u,v,1]` hits the
ground plane z=0 at camera-axis depth `Z_geo = h / (−d_ego,z) · (optical-axis component)`. The per-frame
scale `s = median( Z_geo / Z_DAv2 )` over those road pixels. Apply `Z_corrected = s · Z_DAv2` to ALL
pixels that frame. **Inputs: intrinsics K + extrinsic height + flat-ground + DAv2 image. NO LiDAR.** →
the modality+algorithm+provenance independence ledger from v3 is preserved (verified: LiDAR never enters
the scale).

## Gates (re-applied to the corrected depth; same thresholds as v3, sealed)
- **Gate 1 (re-check):** median |Z_corrected − box_range| on the same known-good unoccluded boxes must now
  be ≤ 0.5 m. If the ground-plane rescale does NOT bring it within 0.5 m (e.g. non-flat terrain on the
  following substrate makes per-frame scale too noisy) → INVALID-SCALE again → STOP, recall stays
  consistency-only. (Reachable kill: the rescale is allowed to fail.)
- **Gate 2 (self-reliability AUC):** unchanged from v3 — DAv2 `depth_struct` evidence on the 60 owner-
  labeled patches, AUC ≥ 0.75 via `_roc_auc`, else ORACLE-INSUFFICIENT. The corrected scale does not
  change WHICH pixels have structure, only their range — so this gate tests the same separation question
  (does DAv2 see structure where the human labeled it). If it fails here too, the "learned priors rescue
  textureless backs" hypothesis is wrong — same reachable kill as stereo (0.259).
- **Confirmatory + null + kill:** identical to v3 (band-local null, gap CI>0 → RECALL-SUPPORTED else FAIL,
  log-clustered bootstrap, held-out threshold log separate, clear-frame filter <0.30 dropped).

## Estimand, independence ledger, confounds, self-check, honest scope
ALL inherited from `oracle_depth_recall_preregistration.md` verbatim (the depth stage now emits
Z_corrected instead of raw Z; nothing else changes). Self-check adds: the ground-plane scale on a frame
returns a finite s ∈ (0.3, 3.0) and is computed without reading any LiDAR (assert the code path touches no
sweep/occupancy in scale estimation).

## Reachable outcomes (all legitimate, pre-registered)
- Gate-1 still fails → INVALID-SCALE (rescale insufficient; ground non-flat) → recall consistency-only.
- Gate-2 AUC < 0.75 → ORACLE-INSUFFICIENT (DAv2 can't separate structure on dark backs) → recall consistency-only.
- gap CI includes 0 → FAIL (occupancy no better than random within band) → headline negative.
- gap CI.lo > 0 → **RECALL-SUPPORTED**: the first EXTERNAL, independent, cross-modal recall result in the
  program — the recall half of denotation-honesty, solo/CPU, on data on disk.

## Sealed run (after commit; once)
```sh
.venv/bin/python experiments/occquery_v0/oracle_depth_recall.py --rescale ground-plane \
  --logs 201fe83b-7dd7-38f4-9d26-7b4a668638a9 2c652f9e-8db8-3572-aa49-fae1344a875b 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
  --heldout-threshold-log 6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c \
  --onnx data/models/dav2_metric_vkitti_vitl/depth_anything_v2_metric_vkitti_vitl.onnx \
  --camera stereo_front_left --z-min 2 --z-max 30 --n-depth-min 8 \
  --scale-gate-m 0.5 --auc-gate 0.75 --null band-local --shuffles 1000 --seed 0 \
  --calib-json experiments/occquery_v0/results/calib_patches/calib_patches.json \
  --out experiments/occquery_v0/results/oracle_depth_recall_v2.json
```
Order in code: ground-plane scale per frame → Gate 1 (corrected) → Gate 2 AUC → gap + band-local null.
