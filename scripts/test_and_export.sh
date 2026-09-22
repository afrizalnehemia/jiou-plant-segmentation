#!/usr/bin/env bash
# Inferensi untuk tiap seed, lalu petakan prediksi ke resolusi penuh supaya
# semua model dinilai pada himpunan titik yang persis sama.
#
# Catatan: train_seeds.sh menghasilkan satu config turunan per seed
# (<CFG>-seed<N>.py), jadi di sini config DAN nama eksperimen sama-sama memakai
# nama turunan itu.
set -euo pipefail

# W&B tidak dipakai: Tensorboard sudah mencatat semuanya ke exp/. Tanpa baris ini
# Pointcept memunculkan prompt login interaktif yang menggantung proses tanpa batas
# waktu -- fatal untuk rangkaian yang ditinggal jalan sendiri.
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_SILENT=true

CFG=${1:?usage: test_and_export.sh <nama-config-tanpa-.py>}
POINTCEPT=${POINTCEPT:-$HOME/G2/Pointcept}
# akar repo ini; timpa dengan REPO=... kalau strukturmu berbeda
G2=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
GT=${GT:-$HOME/G2/data/pheno4d}

cd "$POINTCEPT"
for S in 0 1 2 3 4; do
  EXP="${CFG}-seed${S}"
  if [ ! -f "exp/pheno4d/${EXP}/model/model_best.pth" ]; then
    echo "== lewati ${EXP}: belum ada model_best.pth"; continue
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
