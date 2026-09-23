#!/usr/bin/env bash
# Train one Pointcept config with five seeds (0 to 4).
#
# Pointcept's scripts/train.sh has no option for the seed, so for each seed a
# small config is written next to the parent config that only sets `_base_`
# and `seed`. Every run therefore keeps its own config on disk.
#
#   scripts/train.sh semseg-ptv3-pheno4d-maize [num_gpu]
set -euo pipefail

CFG=${1:?usage: train.sh <config-name-without-.py> [num_gpu]}
NGPU=${2:-1}
POINTCEPT=${POINTCEPT:-$HOME/Pointcept}
CFGDIR="$POINTCEPT/configs/pheno4d"

# Weights & Biases asks for a login and blocks the run unless it is disabled.
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_SILENT=true

[ -f "$CFGDIR/${CFG}.py" ] || { echo "no such config: $CFGDIR/${CFG}.py"; exit 1; }

cd "$POINTCEPT"
for S in 0 1 2 3 4; do
  EXP="${CFG}-seed${S}"
  if [ -f "exp/pheno4d/${EXP}/model/model_last.pth" ]; then
    echo "skip ${EXP} (already trained)"; continue
  fi
  cat > "$CFGDIR/${EXP}.py" <<CFGEOF
# Written by scripts/train.sh
_base_ = ["./${CFG}.py"]
seed = ${S}
CFGEOF
  echo "== ${EXP}"
  sh scripts/train.sh -g "$NGPU" -d pheno4d -c "$EXP" -n "$EXP"
done
