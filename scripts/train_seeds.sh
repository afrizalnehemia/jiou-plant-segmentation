#!/usr/bin/env bash
# Latih satu konfigurasi model pada lima seed.
#
# Lima seed bukan pemanis. Seluruh argumen makalah bergantung pada kemampuan
# mengatakan apakah selisih antar model lebih besar daripada derau antar proses,
# dan satu kali pelatihan tidak bisa menjawab itu.
#
# Cara kerjanya: scripts/train.sh milik Pointcept tidak menerima opsi tambahan
# di baris perintah (getopts-nya hanya p d c n w g m r). Jadi alih-alih menambal
# skrip pihak ketiga, kita HASILKAN satu file config turunan per seed yang hanya
# berisi `_base_` ke config induk plus `seed = N`. Efek sampingnya bagus untuk
# reproduksibilitas: tiap run meninggalkan config-nya sendiri di disk.
#
#   ./train_seeds.sh semseg-ptv3-pheno4d-maize-noflash 1
set -euo pipefail

CFG=${1:?usage: train_seeds.sh <nama-config-tanpa-.py> [jumlah_gpu]}

# W&B tidak dipakai: Tensorboard sudah mencatat semuanya ke exp/. Tanpa baris ini
# Pointcept memunculkan prompt login interaktif yang menggantung proses tanpa batas
# waktu -- fatal untuk rangkaian yang ditinggal jalan sendiri.
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_SILENT=true

NGPU=${2:-1}
SEEDS=(0 1 2 3 4)
POINTCEPT=${POINTCEPT:-$HOME/G2/Pointcept}
CFGDIR="$POINTCEPT/configs/pheno4d"

[ -f "$CFGDIR/${CFG}.py" ] || { echo "config tidak ada: $CFGDIR/${CFG}.py"; exit 1; }

cd "$POINTCEPT"
for S in "${SEEDS[@]}"; do
  EXP="${CFG}-seed${S}"
  if [ -f "exp/pheno4d/${EXP}/model/model_best.pth" ]; then
    echo "== lewati ${EXP} (sudah ada model_best.pth)"; continue
  fi

  # config turunan: hanya menimpa seed
  cat > "$CFGDIR/${EXP}.py" <<EOF
# Dihasilkan otomatis oleh train_seeds.sh -- jangan diedit manual.
_base_ = ["./${CFG}.py"]
seed = ${S}
EOF

  echo "══════════════════════════════════════ ${EXP}"
  sh scripts/train.sh -g "$NGPU" -d pheno4d -c "$EXP" -n "$EXP"
done

echo; echo "selesai. hasil di ${POINTCEPT}/exp/pheno4d/"
echo "cek seed yang benar-benar terpakai:"
echo "  grep -h '^seed' ${POINTCEPT}/exp/pheno4d/${CFG}-seed*/config.py"
