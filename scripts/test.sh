#!/usr/bin/env bash
# Test the five seeds of one config and write full-resolution predictions.
#
#   scripts/test.sh semseg-ptv3-pheno4d-maize
#
# CKPT selects the checkpoint (model_last by default; the first batch of runs
# in the paper used model_best). Predictions are mapped back to every point of
# the original scan, so all models are scored on the same points.
set -euo pipefail

CFG=${1:?usage: test.sh <config-name-without-.py>}
CKPT=${CKPT:-model_last}
POINTCEPT=${POINTCEPT:-$HOME/Pointcept}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
GT=${GT:-$REPO/data/pheno4d}
OUT=${OUT:-$REPO/predictions}

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_SILENT=true

cd "$POINTCEPT"
for S in 0 1 2 3 4; do
  EXP="${CFG}-seed${S}"
  [ -f "exp/pheno4d/${EXP}/model/${CKPT}.pth" ] || { echo "skip ${EXP}: no ${CKPT}.pth"; continue; }
  echo "== ${EXP}"
  sh scripts/test.sh -g 1 -d pheno4d -c "$EXP" -n "$EXP" -w "$CKPT"
  python "$REPO/scripts/export_predictions.py" --gt_root "$GT" \
      --pred_root "exp/pheno4d/${EXP}/result" --out_root "$OUT/${EXP}"
done
