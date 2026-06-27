# Oracle-v2b — KITTI-finetuned IGEV RECALL oracle (PRE-REGISTRATION, seal before the kitti data 2026-06-27)

The declared follow-up from `oracle_stereo_recall_learned_preregistration.md` (git 1c8b357). That run
used the **Scene-Flow synthetic** IGEV checkpoint for MAXIMUM independence-of-provenance and got
**AUC 0.662 < 0.75** (ORACLE-INSUFFICIENT, `oracle_stereo_recall_learned_summary.md`, git 6d800c9):
the density hypothesis was directionally confirmed (census 0.259 → IGEV-sceneflow 0.662) but the
self-reliability gate did not clear. The sceneflow pre-reg explicitly permitted exactly ONE escalation,
SEPARATELY pre-registered to forbid checkpoint-shopping: a **driving-domain finetuned** checkpoint. This
is that pre-reg. Nothing below was chosen after seeing a kitti number.

## The single variable
ONLY the IGEV checkpoint changes: **`sceneflow.pth` → `kitti15.pth`** (the official gangweiX/IGEV-Stereo
KITTI-2015-finetuned weights, from the SAME pretrained-models folder already used). Pinned to exactly
`kitti15.pth` (NOT kitti12, NOT eth3d, NOT middlebury) — one checkpoint, declared, no further shopping.
EVERYTHING else is byte-for-byte the sceneflow run = byte-for-byte the classical sealed run: same 3
following logs (`201fe83b`, `2c652f9e`, `6aaf5b08`; held-out threshold `6aaf5b08`), same estimand
(occ_free ∧ stereo_struct, in-path band∩FOV, same frame t), same `z∈[2,30]`, `|y|<3`, `n_stereo_min=8`,
`edge 1.5 m`, `lr-consistency 1.0 px`, voxelize-with-the-SAME-filters, band-local null (1000 shuffles,
seed 0), the **AUC ≥ 0.75** gate on the SAME 60 human-labeled patches, the log-clustered bootstrap, and
the kill rule. Same rectify → IGEV → warp-back → census-drop-in artifact pipeline (`igev_disparity_pod.py`).

## Hypothesis (declared before the kitti run)
Real-driving finetuning supplies priors for exactly the failure mode (textureless, dark, backlit
vehicle backs at following distance) that synthetic Scene-Flow lacks. Predicted: AUC rises further from
0.662 and **clears 0.75**. This is a directional prediction with a hard, pre-registered bar — not a
movable cutoff.

## The independence trade-off (declared loudly — this is the cost of D2)
KITTI is REAL driving, like AV2. So `kitti15.pth` WEAKENS the independence-of-provenance that the
Scene-Flow (synthetic, never-real-driving) checkpoint maximized. It is **weakened, NOT broken**: the
oracle is still cross-MODALITY (passive optical stereo ≠ active LiDAR TOF), cross-ALGORITHM (deep stereo
matching ≠ `av2_sensor._voxelize`), cross-DATASET (KITTI ≠ AV2 — different vehicle, sensors, city,
country, capture rig), and the metric scale is still GEOMETRIC (0.5 m baseline), never model-output. So
a passing kitti result is a GENUINE external recall signal — materially stronger than same-modality
box-recall — but it must be reported with this explicit caveat: "external, cross-modal, cross-dataset;
the depth front-end's training domain (driving) overlaps AV2's domain, so the provenance independence is
weaker than the (failed) synthetic-checkpoint attempt." No silent upgrade of the independence claim.

## Kill (reachable, declared before the kitti data)
- **ORACLE-INSUFFICIENT** iff kitti AUC < 0.75 → even a SOTA *driving-finetuned* learned stereo cannot
  certify itself as a recall oracle on these following-distance backs ⇒ passive-stereo external recall is
  CLOSED on this substrate for good (no further checkpoint is pre-authorized; escalation would move to a
  different substrate, an active sensor, or trained AV-domain metric-depth, each a new pre-reg). A real,
  publishable terminal negative for this line.
- **FAIL** iff AUC ≥ 0.75 but the GAP `(band-local-shuffled − true)` bootstrap CI includes 0 → the gate
  is reliable but occupancy does not measurably miss more obstacles than the band-local null. Real finding.
- **RECALL-SUPPORTED** iff AUC ≥ 0.75 AND gap CI.lo > 0 → the EXTERNAL recall half is achieved (with the
  weakened-independence caveat above). Reported as the gap vs the 0.75 bar AND the Δ vs sceneflow (0.662)
  AND vs census (0.259), never as a bare absolute.

## "This observation means I am wrong"
If even kitti-finetuned IGEV gives AUC < 0.75 on the SAME 60 patches, the "driving priors will fix it"
hypothesis is falsified, and the trend (0.259 → 0.662 → still-sub-0.75) says passive stereo is simply
not a sufficient recall oracle on AV2 following backs — independent of matcher AND training domain.

## Run (after THIS doc is committed; once)
Identical to the sceneflow run, `--checkpoint .../kitti/kitti15.pth`:
```sh
# pod: python igev_disparity_pod.py --logs <3> --checkpoint .../kitti/kitti15.pth --out-dir results/igev_disp_kitti
# local/pod (numpy): oracle_stereo_recall.py --disparity-source artifact --disparity-artifact-dir results/igev_disp_kitti \
#   --calib-json results/calib_patches/calib_patches.json --out results/oracle_stereo_recall_kitti.json
```

## Honest scope
Same as the sceneflow pre-reg PLUS the weakened-independence caveat above. A clean ORACLE-INSUFFICIENT is
still a headline; a RECALL-SUPPORTED is reported only with the cross-dataset-but-same-domain caveat. No
H3 re-inflation beyond exactly what the gate + gap support.
