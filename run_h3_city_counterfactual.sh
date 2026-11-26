#!/usr/bin/env bash
# H3: seven city F2/uniform/off one-step SHADOW policy tests; no training.
# A complete ORIGINAL F2 Stage-I checkpoint with AdamW optimizer is REQUIRED.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_CKPT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CKPT="${1:-${PMME_H3_CHECKPOINT:-$DEFAULT_CKPT}}"
if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] No original F2 checkpoint at $CKPT" >&2
  echo "Usage: bash run_h3_city_counterfactual.sh /absolute/path/original_F2_stageI.pth" >&2
  exit 2
fi
CKPT="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$CKPT")"
PROBES="${PMME_H3_PROBES:-8}"
SOURCE_POINTS="${PMME_H3_SOURCE_POINTS:-0}"
TARGET_POINTS="${PMME_H3_TARGET_POINTS:-0}"
GPU="${PMME_H3_GPU:-0}"
SEED="${PMME_H3_SEED:-1}"
if [[ "$SOURCE_POINTS" -gt 0 || "$TARGET_POINTS" -gt 0 ]]; then
  DEFAULT_RUN="$ROOT/experiments/h3-city-smoke"
else
  DEFAULT_RUN="$ROOT/experiments/h3-city-full"
fi
RUN_ROOT="${PMME_H3_RUN_ROOT:-$DEFAULT_RUN}"
mkdir -p "$RUN_ROOT"
CKPT_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
EXPECTED="$(printf 'branch=%s\ncommit=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nseed=%s\nprobe_batches=%s\nsource_points=%s\ntarget_points=%s\ngpu=%s\n' \
  "$(git -C "$ROOT" branch --show-current)" "$COMMIT" "$CKPT" "$CKPT_SHA" \
  "$SEED" "$PROBES" "$SOURCE_POINTS" "$TARGET_POINTS" "$GPU")"
if [[ -f "$RUN_ROOT/run_config.txt" ]]; then
  if [[ "$(cat "$RUN_ROOT/run_config.txt")" != "$EXPECTED" ]]; then
    echo "[ERROR] This output directory belongs to another checkpoint or settings" >&2
    exit 3
  fi
else
  printf '%s\n' "$EXPECTED" > "$RUN_ROOT/run_config.txt"
fi
if [[ -f "$RUN_ROOT/COMPLETED" ]]; then
  echo "[SKIP] Already completed: $RUN_ROOT"
  exit 0
fi
rm -f "$RUN_ROOT/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "[START] H3 read-only city audit, checkpoint=$CKPT, probes=$PROBES, points=$SOURCE_POINTS/$TARGET_POINTS"
if (
  cd "$ROOT"
  python -u audit_h3_city_checkpoint.py \
    --checkpoint "$CKPT" \
    --seed "$SEED" \
    --probe-batches "$PROBES" \
    --source-points "$SOURCE_POINTS" \
    --target-points "$TARGET_POINTS" \
    --output "$RUN_ROOT/h3_city_results.jsonl"
) > "$RUN_ROOT/train.out" 2>&1; then
  grep -q "H3_AUDIT_FINISHED" "$RUN_ROOT/train.out" || {
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] Missing completion marker" >&2
    exit 1
  }
  COUNT="$(wc -l < "$RUN_ROOT/h3_city_results.jsonl")"
  [[ "$COUNT" -eq "$PROBES" ]] || {
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] Unexpected number of H3 JSONL rows: $COUNT" >&2
    exit 1
  }
  python "$ROOT/summarize_h3_city_audit.py" \
    --jsonl "$RUN_ROOT/h3_city_results.jsonl" \
    --csv "$RUN_ROOT/h3_city_summary.csv" > "$RUN_ROOT/summary.txt"
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H3 city shadow audit: $RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H3 failed; inspect $RUN_ROOT/train.out" >&2
  tail -n 60 "$RUN_ROOT/train.out" >&2
  exit 1
fi
