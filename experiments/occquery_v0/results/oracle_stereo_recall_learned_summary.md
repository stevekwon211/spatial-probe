# Oracle-v2 (learned-stereo / IGEV) recall result — ORACLE-INSUFFICIENT (honest negative)

Sealed pre-reg: `oracle_stereo_recall_learned_preregistration.md` (git 1c8b357). Ran 2026-06-27 on a
RunPod RTX-4090 pod (IGEV-Stereo, Scene-Flow zero-shot `sceneflow.pth`); disparity artifacts re-graded
by the pure-numpy oracle (`--disparity-source artifact`, deterministic, no GPU in the grade). Total
GPU cost ≈ **$1.01**.

## Verdict: ORACLE-INSUFFICIENT
`results/oracle_stereo_recall_learned.json`:
- **calibration AUC = 0.662** (n_pos 34, n_neg 26), operating-point precision @ n_stereo_min=8 = 0.743.
- Gate is **AUC ≥ 0.75 → < 0.75 fires the pre-registered secondary kill**; no miss-rate reported (the
  band-local-null miss-rate stage is gated behind a passing AUC, by design — same rule as the classical
  pre-reg).

## The headline number: 0.259 → 0.662 (density hypothesis directionally confirmed, gate NOT cleared)
The pre-reg's first-principles bet: classical stereo failed at AUC 0.259 on **DENSITY** (census/SAD
cannot match the textureless, dark, backlit lead-vehicle backs), not scale. Swapping ONLY the depth
front-end (census → IGEV, everything downstream byte-for-byte) moved the self-reliability AUC from
**0.259 → 0.662** on the SAME 3 following logs and the SAME 60 human-labeled patches. So the density
diagnosis was right — a SOTA learned matcher recovers far more real structure than block-matching — but
the gain stops short of the 0.75 bar that the pre-reg requires before any recall miss-rate is claimable.

This is a **pre-registered kill firing exactly as written** ("This observation means I am wrong": IGEV
AUC < 0.75 ⇒ dark backlit following-distance backs are not separable-enough from road by passive stereo
to serve as an external recall oracle here, regardless of matcher quality).

## Apparatus check (Order-your-suspects, before trusting the number)
Distrusted the wiring before the result: sampled IGEV artifacts are **98% finite** with a sensible
disparity range (median ~19.6 px on the 775×1024 undistorted-2×-downsampled grid) — NOT a NaN/
artifact-missing fallback. AUC 0.662 is distinct from both the census AUC (0.259) and chance (0.5),
so the IGEV disparities genuinely drove the grade (a silent census-fallback or missing-artifact bug
would have reproduced 0.259 or degenerated to ~0.5). The self-check passed first (median reproj error
0.434 m ≈ 1 voxel vs LiDAR), so the rectify→IGEV→warp-back→backproject geometry is sound. The number
is real.

## What this means for the program
External cross-modal RECALL is now triangulated CLOSED on the AV2 following substrate by **three**
pre-registered honest negatives with **disjoint** causes:
1. classical stereo — DENSITY (AUC 0.259),
2. frozen DAv2-metric mono-depth — SCALE (>9 m, VKITTI not metric on AV2),
3. **IGEV learned stereo — SELF-RELIABILITY GATE (AUC 0.662 < 0.75)**: density largely fixed, but not
   enough to certify the oracle on these backs.

The recall half of the honesty layer therefore stays **consistency-only externally** (box-recall,
relative-only). The FP half remains EXTERNAL + RELIABLE (traversal). No H3 re-inflation.

## What would actually clear the gate (resource map, honest)
- **Driving-domain finetuned stereo** (e.g. IGEV `kitti15.pth`): real-driving priors would likely lift
  AUC past 0.75 — but KITTI is real driving like AV2, so it WEAKENS the independence-of-provenance the
  Scene-Flow checkpoint was chosen for. The pre-reg explicitly allows this only as a SEPARATELY
  pre-registered follow-up now that the synthetic checkpoint failed the gate (forbids checkpoint-shopping).
- **Trained AV-domain metric-depth** (DAv2 finetuned on AV2/driving, GPU *training* not just inference).
- **Active-sensor cross-check** or a curated textured-AND-obstacle-dense substrate.
- A lower self-reliability bar would be HARKing — not on the table without a new sealed pre-reg.

## Reproducibility
Deterministic (seed 0). Re-run: `pod_setup.sh` (s5cmd the 3 val logs, clone IGEV, fetch sceneflow.pth)
→ `igev_disparity_pod.py` (emits `disp_<log>_<cam_ts>_<side>.npz`) → `oracle_stereo_recall.py
--disparity-source artifact`. The disparity artifacts (~6 GB) are gitignored; `rectify_meta_*.json`
(per-log rectification geometry) are kept for audit.
