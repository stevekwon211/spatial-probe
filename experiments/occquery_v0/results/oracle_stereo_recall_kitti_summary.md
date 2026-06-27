# Oracle-v2b (KITTI-finetuned IGEV) recall result — ORACLE-INSUFFICIENT (terminal negative)

Sealed pre-reg: `oracle_stereo_recall_kitti_preregistration.md` (git 91e9593). Ran 2026-06-27 on a
RunPod RTX-4090 (IGEV-Stereo `kitti15.pth`, driving-finetuned); same census-drop-in artifact pipeline,
re-graded by the pure-numpy oracle. Single variable vs the sceneflow run: the checkpoint only.

## Verdict: ORACLE-INSUFFICIENT
`results/oracle_stereo_recall_kitti.json`:
- **calibration AUC = 0.733** (n_pos 34, n_neg 26), operating-point precision @ n_stereo_min=8 = 0.692.
- Gate AUC ≥ 0.75 → 0.733 < 0.75 fires the pre-registered secondary kill. No miss-rate reported.

## The full monotonic trend (the real finding)
On the SAME 3 following logs and the SAME 60 human-labeled patches, the self-reliability AUC across
depth front-ends:

| depth front-end | AUC | gate (≥0.75) |
|---|---|---|
| classical census/SAD (CPU) | 0.259 | ✗ (below chance) |
| IGEV Scene-Flow, zero-shot (synthetic) | 0.662 | ✗ |
| **IGEV KITTI-finetuned (real driving)** | **0.733** | ✗ (0.017 short) |

Monotonic improvement with matcher quality AND domain match, **asymptoting just under the 0.75 bar**.
This is strong evidence the limit is the **sensing modality** (passive optical stereo on textureless,
dark, backlit lead-vehicle backs at following distance), not the matcher or its training domain — a
better/closer-domain network keeps helping but cannot cross the reliability threshold.

## Integrity: 0.733 is NOT a pass (no HARKing)
0.733 is tantalizingly close to 0.75, but the bar was pre-registered BEFORE any IGEV data and the kitti
pre-reg explicitly forbade lowering it ("a lower self-reliability bar would be HARKing"). So this is an
honest kill reported as the gap vs the fixed bar (−0.017), not a massaged pass. Lowering 0.75→0.73 to
claim a win would be exactly the self-deception the research-integrity rules exist to prevent.

## Independence note strengthens the closure
`kitti15.pth` was the WEAKENED-independence escalation (KITTI is real driving, like AV2). It STILL did
not certify. So the closure is robust: even the less-independent, domain-matched matcher cannot reach
the self-reliability bar. (Apparatus verified: AUC 0.733 non-degenerate, all 60 patches scored, distinct
from sceneflow 0.662 and census 0.259 — the kitti artifacts genuinely drove the grade.)

## Terminal status for this line
Passive-stereo external recall on the AV2 following substrate is now CLOSED by **four** pre-registered
points (census + three IGEV-class), three with disjoint root causes:
1. classical stereo — DENSITY (0.259),
2. DAv2 mono-depth — SCALE (>9 m),
3. IGEV sceneflow — gate-miss 0.662, IGEV kitti — gate-miss 0.733 (modality-limited, monotonic).

No further checkpoint is pre-authorized (would be shopping). Recall stays **consistency-only externally**;
the FP half stays EXTERNAL + RELIABLE (traversal). No H3 re-inflation.

## What would still open it (resource map, honest)
- **Trained AV-domain metric-depth** (e.g. DAv2 finetuned on AV2/driving — GPU *training*, a real
  project, not $1 inference). The only remaining depth source not yet tried; uncertain payoff.
- **Active-sensor cross-check** or a curated textured-and-obstacle-dense substrate.
- Each is a NEW sealed pre-reg. The monotonic 0.259→0.733 trend says passive optics is near its ceiling
  here, so the metric-depth-training path is the better bet if recall is pursued further.

## Cost / reproducibility
Two GPU runs (sceneflow + kitti) ≈ **$1.73** total. Deterministic (seed 0). Artifacts (~6 GB each)
gitignored; `rectify_meta_*.json` kept. Re-run: `run_kitti.sh`-style (s5cmd 3 val logs, IGEV
`kitti15.pth`) → `igev_disparity_pod.py` → `oracle_stereo_recall.py --disparity-source artifact`.
