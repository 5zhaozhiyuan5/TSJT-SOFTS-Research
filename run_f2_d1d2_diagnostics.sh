#!/usr/bin/env bash
# D1/D2 read-only diagnosis from an EXISTING original F2 Stage-I checkpoint.
# This DOES NOT retrain F2, use target test data, or write any checkpoint.
# Usage: bash run_f2_d1d2_diagnostics.sh /absolute/path/original-f2-stage1.pth
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || {
  echo "Usage: bash $0 /absolute/path/original-f2-stage1.pth" >&2
  exit 2
}
CKPT="$1"
[[ -f "$CKPT" ]] || {
  echo "[ERROR] Existing F2 checkpoint not found: $CKPT" >&2
  exit 2
}
WORK="${PMME_F2_DIAG_RUN_ROOT:-$ROOT/experiments/f2-d1d2-readonly-seed1}"
PROBES="${PMME_F2_DIAG_PROBES:-16}"
SEED="${PMME_F2_DIAG_SEED:-1}"
SOURCE_BUDGET="${PMME_F2_DIAG_SOURCE_BUDGET:-192}"
TARGET_BUDGET="${PMME_F2_DIAG_TARGET_BUDGET:-96}"
GPU="${PMME_F2_DIAG_GPU:-0}"
mkdir -p "$WORK"
[[ ! -f "$WORK/COMPLETED" ]] || {
  echo "[SKIP] Existing successful diagnostics $WORK"
  exit 0
}
printf 'branch=%s\ncommit=%s\ncheckpoint=%s\nseed=%s\nprobe_batches=%s\nsource_budget=%s\ntarget_budget=%s\ngpu=%s\n' \
  "$(git -C "$ROOT" branch --show-current)" \
  "$(git -C "$ROOT" rev-parse HEAD)" "$CKPT" "$SEED" "$PROBES" \
  "$SOURCE_BUDGET" "$TARGET_BUDGET" "$GPU" > "$WORK/run_config.txt"
rm -f "$WORK/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if (
  cd "$ROOT"
  python -u audit_f2_checkpoint.py \
    --checkpoint "$CKPT" \
    --probe-batches "$PROBES" \
    --seed "$SEED" \
    --source-budget "$SOURCE_BUDGET" \
    --target-budget "$TARGET_BUDGET" \
    --output "$WORK/f2_d1d2.jsonl"
) > "$WORK/train.out" 2>&1; then
  python "$ROOT/summarize_f2_d1d2.py" \
    --jsonl "$WORK/f2_d1d2.jsonl" \
    --csv "$WORK/f2_d1d2_summary.csv" > "$WORK/summary.txt"
  grep -q "F2_DIAGNOSTIC_FINISHED" "$WORK/train.out" || {
    echo "[ERROR] completion marker missing" >&2; touch "$WORK/FAILED"; exit 1;
  }
  touch "$WORK/COMPLETED"
  echo "[DONE] Read-only F2 D1/D2 audit: $WORK"
  cat "$WORK/summary.txt"
else
  touch "$WORK/FAILED"
  echo "[ERROR] See $WORK/train.out" >&2
  tail -n 50 "$WORK/train.out" >&2
  exit 1
fi
