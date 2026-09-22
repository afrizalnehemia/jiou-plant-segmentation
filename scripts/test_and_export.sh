#!/usr/bin/env bash
# Run inference for each seed, then map the predictions back to full resolution
# so that every model is scored on exactly the same set of points.
#
# Note: train_seeds.sh generates one derived config per seed (<CFG>-seed<N>.py),
# so here the config AND the experiment name both use that derived name.
set -euo pipefail

# W&B off: Tensorboard already logs everything to exp/. Without these two lines
# Pointcept raises an interactive login prompt that hangs the process indefinitely
# -- fatal for a sequence left running unattended.
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_SILENT=true

CFG=${1:?usage: test_and_export.sh <config-name-without-.py>}
POINTCEPT=${POINTCEPT:-$HOME/G2/Pointcept}
# root of this repo; override with REPO=... if your layout differs
G2=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
GT=${GT:-$HOME/G2/data/pheno4d}

cd "$POINTCEPT"
for S in 0 1 2 3 4; do
  EXP="${CFG}-seed${S}"
  if [ ! -f "exp/pheno4d/${EXP}/model/model_best.pth" ]; then
    echo "== skipping ${EXP}: no model_best.pth yet"; continue
  fi
  echo "══════════════════════════════════════ ${EXP}"
  sh scripts/test.sh -g 1 -d pheno4d -c "$EXP" -n "$EXP" -w model_best
  python "$G2/03-RUN/export_predictions.py" \
      --gt_root   "$GT" \
      --pred_root "exp/pheno4d/${EXP}/result" \
      --out_root  "$G2/RESULTS/${EXP}"
done

echo; echo "berikutnya:"
echo "  python $G2/04-EVAL/score_runs.py --gt_root $GT \\"
echo "         --results_root $G2/RESULTS --out_csv $G2/RESULTS/skor.csv"
