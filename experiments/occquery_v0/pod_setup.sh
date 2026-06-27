#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
#
# ==============================================================================================
# RUNS ON THE RUNPOD GPU POD ONLY. Installs torch + IGEV-Stereo + cv2, fetches sceneflow.pth, then
# runs igev_disparity_pod.py to emit the census-drop-in disparity artifacts for the 3 sealed logs.
# The repo .venv core stays pure numpy/scipy/Pillow/pyarrow (CLAUDE.md) -- NOTHING here touches it.
# Idempotent: every step is guarded so re-running is safe. Cannot be exercised off-pod (no GPU/cv2/torch).
# ==============================================================================================
set -euo pipefail

# --- config (override via env) ---------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/spatial-probe}"          # this repo, cloned on the pod
IGEV_DIR="${IGEV_DIR:-$WORKSPACE/IGEV-Stereo}"            # gangweiX/IGEV-Stereo
DATA_ROOT="${DATA_ROOT:-$WORKSPACE/av2_sensor}"          # holds <log>/sensors + <log>/calibration
OUT_DIR="${OUT_DIR:-$REPO_DIR/experiments/occquery_v0/results/igev_disp}"
CKPT="${CKPT:-$IGEV_DIR/pretrained_models/sceneflow/sceneflow.pth}"

LOGS=(
  201fe83b-7dd7-38f4-9d26-7b4a668638a9
  2c652f9e-8db8-3572-aa49-fae1344a875b
  6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c
)

echo "== pod_setup: workspace=$WORKSPACE  data_root=$DATA_ROOT  igev=$IGEV_DIR =="

# --- 1) python deps (pod only: torch + cv2 + IGEV reqs; numpy/scipy/Pillow/pyarrow for the oracle) ---
# IGEV-Stereo is developed against torch>=1.7; any recent CUDA torch works for zero-shot inference.
python -m pip install --upgrade pip
python -m pip install torch torchvision                 # CUDA build per the pod image
python -m pip install opencv-python-headless            # cv2 (rectification) -- POD ONLY, never in core
python -m pip install numpy scipy pillow pyarrow tqdm gdown \
                      "opt_einsum" "timm==0.5.4" "scikit-image"   # IGEV-Stereo requirements

# --- 2) clone IGEV-Stereo (idempotent) -------------------------------------------------------
if [ ! -d "$IGEV_DIR/.git" ]; then
  git clone https://github.com/gangweiX/IGEV-Stereo.git "$IGEV_DIR"
else
  echo "  IGEV-Stereo already present at $IGEV_DIR (skip clone)"
fi

# --- 3) fetch the Scene-Flow zero-shot checkpoint sceneflow.pth (idempotent) ------------------
# IGEV-Stereo ships pretrained weights via the Google Drive folder linked in its README
# (https://github.com/gangweiX/IGEV-Stereo#pretrained-models). The exact file id is NOT hardcoded
# here to avoid silently fetching the wrong file: copy the `sceneflow.pth` share link from that README
# and set IGEV_SCENEFLOW_GDRIVE_ID, or drop the file at $CKPT manually. The pre-reg REQUIRES the
# Scene-Flow checkpoint (synthetic, never trained on real driving) -- do NOT substitute a KITTI/AV2
# finetuned checkpoint (that would weaken the independence-of-provenance the pre-reg relies on).
mkdir -p "$(dirname "$CKPT")"
if [ ! -f "$CKPT" ]; then
  if [ -n "${IGEV_SCENEFLOW_GDRIVE_ID:-}" ]; then
    gdown "https://drive.google.com/uc?id=${IGEV_SCENEFLOW_GDRIVE_ID}" -O "$CKPT"
  else
    echo "!! sceneflow.pth not found at $CKPT and IGEV_SCENEFLOW_GDRIVE_ID is unset."
    echo "!! Get the link from the IGEV-Stereo README pretrained-models section, then either:"
    echo "!!   IGEV_SCENEFLOW_GDRIVE_ID=<id> bash $0   (re-run), or place the file at $CKPT"
    exit 1
  fi
else
  echo "  checkpoint already present at $CKPT (skip download)"
fi

# --- 4) stereo + calibration for the 3 logs ---------------------------------------------------
# Expected layout (rsync'd from your machine OR pulled with s5cmd below):
#   $DATA_ROOT/<log>/sensors/cameras/stereo_front_left/*.jpg
#   $DATA_ROOT/<log>/sensors/cameras/stereo_front_right/*.jpg
#   $DATA_ROOT/<log>/sensors/lidar/*.feather              (for --self-check geometry sanity only)
#   $DATA_ROOT/<log>/calibration/{intrinsics,egovehicle_SE3_sensor}.feather
#
# AV2-Sensor val is a PUBLIC bucket (no credentials). Pull ONLY what we need (stereo + lidar + calib)
# with s5cmd (https://github.com/peak/s5cmd). Uncomment to fetch on the pod:
#
#   for LOG in 201fe83b-7dd7-38f4-9d26-7b4a668638a9 \
#              2c652f9e-8db8-3572-aa49-fae1344a875b \
#              6aaf5b08-9f84-3a2e-8a32-2e50e5e11a3c ; do
#     BASE="s3://argoverse/datasets/av2/sensor/val/$LOG"
#     s5cmd --no-sign-request cp "$BASE/sensors/cameras/stereo_front_left/*"  "$DATA_ROOT/$LOG/sensors/cameras/stereo_front_left/"
#     s5cmd --no-sign-request cp "$BASE/sensors/cameras/stereo_front_right/*" "$DATA_ROOT/$LOG/sensors/cameras/stereo_front_right/"
#     s5cmd --no-sign-request cp "$BASE/sensors/lidar/*"                      "$DATA_ROOT/$LOG/sensors/lidar/"
#     s5cmd --no-sign-request cp "$BASE/calibration/*"                        "$DATA_ROOT/$LOG/calibration/"
#   done
#
for LOG in "${LOGS[@]}"; do
  for sub in sensors/cameras/stereo_front_left sensors/cameras/stereo_front_right calibration; do
    if [ ! -d "$DATA_ROOT/$LOG/$sub" ]; then
      echo "!! missing $DATA_ROOT/$LOG/$sub -- rsync or s5cmd it (see commented block above)"; exit 1
    fi
  done
done
echo "  all 3 logs' stereo + calibration present under $DATA_ROOT"

# --- 5) geometry self-check on ONE frame, THEN the full pass ----------------------------------
cd "$REPO_DIR"
echo "== self-check (geometry sanity vs LiDAR, one frame) =="
python experiments/occquery_v0/igev_disparity_pod.py \
  --logs "${LOGS[@]}" --data-root "$DATA_ROOT" \
  --igev-repo "$IGEV_DIR" --checkpoint "$CKPT" --out-dir "$OUT_DIR" --self-check

echo "== full pass: emit disp_<log>_<cam_ts>_<side>.npz artifacts =="
python experiments/occquery_v0/igev_disparity_pod.py \
  --logs "${LOGS[@]}" --data-root "$DATA_ROOT" \
  --igev-repo "$IGEV_DIR" --checkpoint "$CKPT" --out-dir "$OUT_DIR"

echo "== DONE. rsync $OUT_DIR back, then re-grade LOCALLY (no GPU):"
echo "   .venv/bin/python experiments/occquery_v0/oracle_stereo_recall.py \\"
echo "     --disparity-source artifact <local>/igev_disp \\"
echo "     --calib-json results/calib_patches/calib_patches.json \\"
echo "     --out experiments/occquery_v0/results/oracle_stereo_recall_learned.json"
