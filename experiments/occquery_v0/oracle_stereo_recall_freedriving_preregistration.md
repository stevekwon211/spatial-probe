# Oracle-v1.1 — classical-stereo RECALL on the FREE-DRIVING substrate (PRE-REGISTRATION, seal before data 2026-06-26)

Re-pre-registration of the stereo recall oracle (`oracle_stereo_recall_preregistration.md`, ORACLE-
INSUFFICIENT at AUC 0.259 on the stop-and-go FOLLOWING substrate). That pre-reg NAMED the fix: "switch
substrate to free-driving … where obstacles are textured/lit, not dark backlit lead-vehicle backs." This
runs the IDENTICAL stereo oracle (`oracle_stereo_recall.py`, unchanged) on downloaded free-driving stereo.
Classical stereo has EXACT metric scale (0.5 m baseline geometry) — it only failed on DENSITY (textureless
backs); free-driving structure (parked/oncoming/lateral vehicles, poles, pedestrians) is better textured/
lit, so the AUC may now clear. Nothing below was chosen after seeing a v1.1 number.

## Substrate (sealed) — the pre-registered fix
Free-driving AV2-Sensor val logs from the sealed held-out set (`oracle_heldout_logs.json`), stereo + calib
downloaded fresh via `s5cmd --no-sign-request` (no auth, no GPU). Logs: `d5d6f11c`, `c2d44a70`, `c222c78d`,
`27be7d34` (all escape_frac=1.00 free-driving, disjoint from the 18 AND from the 3 following logs where
stereo scored 0.259 — a genuine fresh-substrate re-test, not a re-fit).

**VERIFIED in-path obstacle density BEFORE seal (the key risk, stated honestly):** in-path (x∈2–30 m,
|y|<3 m) LiDAR-seen (≥5 pts) boxes per log = d5d6f11c:0, c2d44a70:22, c222c78d:8, 27be7d34:0. So 2 of 4
logs have ZERO in-path obstacles (free-driving means a clean path), and the other 2 are sparse. This makes
the VACUITY outcome the most likely one: free-driving fixes stereo's density problem (textured structure)
but largely removes the in-path obstacles the recall test needs. `c2d44a70` + `c222c78d` are the only real
headline candidates; the run is executed honestly under the vacuity guard. The pre-registered EXPECTATION
is therefore: a non-vacuous RECALL-SUPPORTED would be a strong positive, but INDETERMINATE-VACUOUS is the
more probable honest outcome and would itself be the finding (free-driving trades the density kill for an
obstacle-absence kill → external recall stays closed on available AV2 substrates). One log held out for
thresholds; the rest headline.

## What changes vs the sealed stereo pre-reg (ONLY two things)
1. **Substrate** = free-driving (above) instead of the 3 following logs.
2. **AUC calibration labels = BOX-DERIVED auto-labels, NOT human-eye.** The original used 60 owner-labeled
   patches; for fresh logs no human relabel is available in an autonomous run, so the 60 patches are
   auto-labeled by GROUND TRUTH: a patch sampled at a tracked-annotation-box projection (`pos_box`) → label
   1 (structure); a random lower-image drivable-road patch (`neg_road`) → label 0 (empty). This is a
   DEFENSIBLE automated self-reliability check (boxes are dataset GT of object presence) but it is WEAKER
   than the human pass — it cannot catch annotation gaps (an unlabeled object in a `neg_road` patch would
   be mislabeled empty). Declared, not hidden: the AUC here measures "does stereo `stereo_struct` separate
   GT-box locations from road locations," and a clean pass is necessary-not-sufficient vs the human gate.
EVERYTHING ELSE is inherited verbatim from `oracle_stereo_recall_preregistration.md`: the estimand
(occ_free ∧ stereo_struct in band∩FOV), undistortion, the census/SAD matcher + L/R consistency + edge
reject + texture gate, voxelize-with-same-filters, the **band-local null**, the **AUC ≥ 0.75 gate**, the
log-clustered bootstrap, and the kill rule.

## Gates + kill (sealed, same thresholds)
- **Gate (self-reliability):** box-derived AUC ≥ 0.75 → proceed; < 0.75 → ORACLE-INSUFFICIENT (stereo still
  too sparse even on textured free-driving structure → the density problem is substrate-independent).
- **Vacuity guard (reachable):** if a headline log has < `_MIN` usable frames with ≥1 `stereo_struct`
  band∩FOV voxel (free-driving may have few in-path obstacles) → that log is INDETERMINATE-VACUOUS, not a
  pass. If all headline logs are vacuous → INDETERMINATE-VACUOUS overall (a legitimate sealed outcome: the
  clean substrate removed the obstacles too).
- **Confirmatory:** band-local null, gap `(shuffled − true)` log-clustered bootstrap CI.
  **RECALL-SUPPORTED** iff gap CI.lo > 0; **FAIL** iff CI includes 0; **INDETERMINATE** otherwise.

## "This observation means I am wrong"
If stereo AUC < 0.75 AGAIN on textured free-driving structure, the density failure is NOT a
following-substrate artifact — classical stereo is just insufficient as a recall oracle here, full stop, and
external recall stays closed (consistent with the DAv2 scale kill). If the split is vacuous, free-driving
trades the density kill for an obstacle-absence kill — also honest, also closes the solo/CPU external-recall
route on available substrates.

## Reachable outcomes (all legitimate)
RECALL-SUPPORTED (external recall finally achieved — the moat half) / ORACLE-INSUFFICIENT (stereo density
substrate-independent) / INDETERMINATE-VACUOUS (free-driving has no in-path structure to grade) / FAIL
(occupancy no better than random within band).

## Sealed run (after this doc + the auto-labeled calib are committed; once)
```sh
# 1. emit patches on the free-driving logs, then auto-label by GT-box kind (pos_box=1, neg_road=0):
.venv/bin/python experiments/occquery_v0/oracle_stereo_recall.py --emit-calib-patches \
  --logs <the 4 free-driving logs> --calib-out-dir results/calib_patches_freedriving \
  --calib-json results/calib_patches_freedriving/calib_patches.json
#    (then a deterministic auto-label step fills label = 1 if kind==pos_box else 0)
# 2. the sealed confirmatory:
.venv/bin/python experiments/occquery_v0/oracle_stereo_recall.py \
  --logs d5d6f11c-... c2d44a70-... c222c78d-... 27be7d34-... \
  --heldout-threshold-log 27be7d34-... \
  --z-min 2 --z-max 30 --n-stereo-min 8 --lr-consistency-px 1.0 --edge-discontinuity-m 1.5 \
  --null band-local --shuffles 1000 --seed 0 \
  --calib-json results/calib_patches_freedriving/calib_patches.json \
  --out experiments/occquery_v0/results/oracle_stereo_recall_freedriving.json
```

## Honest scope
Same as the stereo pre-reg + the box-derived-AUC weakening above. Cross-modal-ish (passive optical stereo
vs active LiDAR) — much more independent than same-modality box-recall, not external ground truth. Measured
miss-rate is a LOWER BOUND (correlated textureless failures dropped). A clean negative/vacuous is the
headline.
