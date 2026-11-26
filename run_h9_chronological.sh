#!/usr/bin/env bash
# H9 A/B: clone-only temporal development adaptation. No original F2 retrain.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CHECKPOINT="${1:-${PMME_H9_CHECKPOINT:-$DEFAULT_CKPT}}"
EXPECTED_SHA="f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Existing original F2 checkpoint missing; no retraining permitted: $CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
FOUND_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$FOUND_SHA" != "$EXPECTED_SHA" ]]; then
  echo "[ERROR] Not the pinned original F2 checkpoint: $FOUND_SHA" >&2
  exit 2
fi

UPDATES="${PMME_H9_MAX_UPDATES:-1}"
SRC="${PMME_H9_SOURCE_POINTS:-256}"
TGT="${PMME_H9_TARGET_POINTS:-128}"
DAYS="${PMME_H9_VALIDATION_DAYS:-4}"
DEV_BS="${PMME_H9_DEV_BATCH_SIZE:-8}"
DEV_N="${PMME_H9_DEV_MAX_WINDOWS:-0}"
GPU="${PMME_H9_GPU:-0}"
case "$UPDATES" in 1|5|10|20) ;; *) echo "[ERROR] max updates must be 1/5/10/20" >&2; exit 2;; esac
case "$DAYS" in 2|3|4|5|6|7) ;; *) echo "[ERROR] development period must be 2..7 days" >&2; exit 2;; esac
if [[ "$SRC" == 0 && "$TGT" == 0 ]]; then
  GEOMETRY=full
else
  GEOMETRY=capped
fi
RUN_ROOT="${PMME_H9_RUN_ROOT:-$ROOT/experiments/h9-${GEOMETRY}-updates${UPDATES}-dev${DAYS}d}"
mkdir -p "$RUN_ROOT"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
BRANCH="$(git -C "$ROOT" branch --show-current)"
if [[ "$BRANCH" != "experiment/pmme-h9-chronological-holdout-adaptation-20261127" ]]; then
  echo "[ERROR] Wrong branch: $BRANCH" >&2
  exit 2
fi
CONFIG="$(printf 'branch=%s\ncommit=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nmax_updates=%s\nsource_points=%s\ntarget_points=%s\nvalidation_days=%s\nvalidation_batch_size=%s\nvalidation_max_windows=%s\ngpu=%s\nseed=1\noriginal_F2_updates=0\nstage2_runs=0\n' \
  "$BRANCH" "$COMMIT" "$CHECKPOINT" "$FOUND_SHA" \
  "$UPDATES" "$SRC" "$TGT" "$DAYS" "$DEV_BS" "$DEV_N" "$GPU")"
if [[ -f "$RUN_ROOT/run_config.txt" ]]; then
  if [[ "$(cat "$RUN_ROOT/run_config.txt")" != "$CONFIG" ]]; then
    echo "[ERROR] Run folder was created for different code or settings; choose a new directory." >&2
    exit 3
  fi
else
  printf '%s\n' "$CONFIG" >"$RUN_ROOT/run_config.txt"
fi
if [[ -f "$RUN_ROOT/COMPLETED" ]]; then
  echo "[SKIP] H9 run completed: $RUN_ROOT"
  exit 0
fi
rm -f "$RUN_ROOT/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[START] H9 chrono dev, independent model clones, full/capped=$GEOMETRY"
if (
  cd "$ROOT"
  python -u audit_h9_chronological_adaptation.py \
    --checkpoint "$CHECKPOINT" \
    --output "$RUN_ROOT/h9_results.json" \
    --max-updates "$UPDATES" \
    --source-points "$SRC" \
    --target-points "$TGT" \
    --validation-days "$DAYS" \
    --validation-batch-size "$DEV_BS" \
    --validation-max-windows "$DEV_N" \
    --gpu "$GPU"
) >"$RUN_ROOT/train.out" 2>&1; then
  if ! grep -Fq "H9_COMPLETED max_updates $UPDATES original_f2_updates 0 stage2_runs 0" "$RUN_ROOT/train.out"; then
    echo "[ERROR] H9 completion marker not found" >&2
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  if ! python "$ROOT/summarize_h9_chronological.py" \
      --json "$RUN_ROOT/h9_results.json" \
      --summary "$RUN_ROOT/h9_policy_development.csv" \
      --days "$RUN_ROOT/h9_daywise_development.csv" \
      --decision "$RUN_ROOT/h9_decision.json" \
      >"$RUN_ROOT/summary.txt" 2>"$RUN_ROOT/summary.err"; then
    echo "[ERROR] H9 summary failed" >&2
    cat "$RUN_ROOT/summary.err" >&2
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H9 development analysis; original F2 untouched: $RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H9 failed, see $RUN_ROOT/train.out" >&2
  tail -n 65 "$RUN_ROOT/train.out" >&2
  exit 1
fi
