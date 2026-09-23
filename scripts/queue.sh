#!/usr/bin/env bash
# Work through scripts/jobs.txt on one GPU.
#
#   scripts/queue.sh <gpu_id> [jobs_file]
#
# Start one copy per GPU, each in its own tmux window, after copying
# configs/revision/*.py to $POINTCEPT/configs/pheno4d/. Workers share the list:
# a job is claimed by creating a directory, which is atomic, so two GPUs never
# take the same job. Per job: train -> test with model_last -> copy the
# full-resolution predictions, the log and the config to $OUT/<job>/ -> DONE.
#
# Re-running is safe. Finished jobs (DONE present) are skipped. A job that
# FAILED keeps its claim so that no worker loops on it; to retry it:
#   rm -rf $OUT/_claims/<job> $POINTCEPT/exp/pheno4d/<job>
set -uo pipefail

GPU=${1:?usage: queue.sh <gpu_id> [jobs_file]}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
JOBS=${2:-$REPO/scripts/jobs.txt}
POINTCEPT=${POINTCEPT:-$HOME/Pointcept}
DATA=${DATA:-$POINTCEPT/data/pheno4d-fixed}
OUT=${OUT:-$REPO/predictions/revision}

export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=disabled WANDB_SILENT=true

mkdir -p "$OUT/_claims"
log() { echo "$(date '+%F %T') gpu$GPU $*" | tee -a "$OUT/queue.log"; }

cd "$POINTCEPT"
exec 3< <(grep -v '^\s*#' "$JOBS" | sed '/^\s*$/d')
while read -r -u 3 JOB; do
  [ -f "$OUT/$JOB/DONE" ] && continue
  mkdir "$OUT/_claims/$JOB" 2>/dev/null || continue
  [ -f "configs/pheno4d/$JOB.py" ] || { log "MISSING CONFIG $JOB"; continue; }
  C="$OUT/_claims/$JOB"
  log "start $JOB"
  T0=$(date +%s)
  if ! sh scripts/train.sh -g 1 -d pheno4d -c "$JOB" -n "$JOB" > "$C/train.out" 2>&1; then
    log "FAILED train $JOB (see $C/train.out)"; continue
  fi
  T1=$(date +%s)
  if ! sh scripts/test.sh -g 1 -d pheno4d -c "$JOB" -n "$JOB" -w model_last > "$C/test.out" 2>&1; then
    log "FAILED test $JOB (see $C/test.out)"; continue
  fi
  T2=$(date +%s)
  mkdir -p "$OUT/$JOB"
  if ! python "$REPO/scripts/export_predictions.py" --gt_root "$DATA" \
        --pred_root "exp/pheno4d/$JOB/result" --out_root "$OUT/$JOB/pred" > "$C/export.out" 2>&1; then
    log "FAILED export $JOB (see $C/export.out)"; continue
  fi
  cp "exp/pheno4d/$JOB/train.log" "configs/pheno4d/$JOB.py" "$OUT/$JOB/" 2>/dev/null
  echo "train_s=$((T1-T0)) test_s=$((T2-T1))" > "$OUT/$JOB/timing.txt"
  touch "$OUT/$JOB/DONE"
  log "done  $JOB  train $(( (T1-T0)/60 )) min, test $(( (T2-T1)/60 )) min"
done
log "queue finished"
