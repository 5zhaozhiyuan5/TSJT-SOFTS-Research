#!/usr/bin/env bash
# H6 reuses the EXISTING original-F2 Stage-I checkpoint. Never retrains F2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CHECKPOINT="${1:-${PMME_H6_CHECKPOINT:-$DEFAULT_CKPT}}"
EXPECTED_SHA="f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Original F2 checkpoint absent: $CHECKPOINT" >&2
  echo "H6 refuses to retrain F2. Give an existing Stage-I ORIGINAL checkpoint." >&2
  exit 2
fi
CHECKPOINT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
FOUND_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$FOUND_SHA" != "$EXPECTED_SHA" ]]; then
  echo "[ERROR] Checkpoint SHA mismatch: $FOUND_SHA; required $EXPECTED_SHA" >&2
  echo "Refusing to run on PRC/H5 or an unverified checkpoint." >&2
  exit 2
fi
N="${PMME_H6_PROBES:-8}"
SRC="${PMME_H6_SOURCE_POINTS:-256}"
TGT="${PMME_H6_TARGET_POINTS:-128}"
GPU="${PMME_H6_GPU:-0}"
if [[ "$SRC" == 0 && "$TGT" == 0 ]]; then
  DEFAULT_DIR="$ROOT/experiments/h6-full-ot"
else
  DEFAULT_DIR="$ROOT/experiments/h6-capped-ot"
fi
RUN_ROOT="${PMME_H6_RUN_ROOT:-$DEFAULT_DIR}"
mkdir -p "$RUN_ROOT"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
CONFIG="$(printf 'branch=%s\ncommit=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nprobe_batches=%s\nsource_points=%s\ntarget_points=%s\nseed=1\ngpu=%s\not_scale=0.01\ntraining_updates=0\nstage2_runs=0\n' \
  "$(git -C "$ROOT" branch --show-current)" "$COMMIT" "$CHECKPOINT" \
  "$FOUND_SHA" "$N" "$SRC" "$TGT" "$GPU")"
if [[ -f "$RUN_ROOT/run_config.txt" ]]; then
  if [[ "$(cat "$RUN_ROOT/run_config.txt")" != "$CONFIG" ]]; then
    echo "[ERROR] Existing run folder has different commit/settings." >&2
    exit 3
  fi
else
  printf '%s\n' "$CONFIG" >"$RUN_ROOT/run_config.txt"
fi
if [[ -f "$RUN_ROOT/COMPLETED" ]]; then
  echo "[SKIP] H6 completed: $RUN_ROOT"
  exit 0
fi
rm -f "$RUN_ROOT/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[START] H6 checkpoint-only real OT gradient audit, probes=$N, source/target points=$SRC/$TGT"
if (
  cd "$ROOT"
  python -u audit_h6_ot_gradient.py \
    --checkpoint "$CHECKPOINT" \
    --probe-batches "$N" \
    --source-points "$SRC" \
    --target-points "$TGT" \
    --gpu "$GPU" \
    --output "$RUN_ROOT/h6_results.jsonl"
) >"$RUN_ROOT/train.out" 2>&1; then
  if ! grep -Fq "H6_AUDIT_FINISHED $N training_updates 0 stage2_runs 0" "$RUN_ROOT/train.out"; then
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] H6 audit missing completion marker" >&2
    exit 1
  fi
  COUNT="$(wc -l <"$RUN_ROOT/h6_results.jsonl")"
  if [[ "$COUNT" -ne "$N" ]]; then
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] Expected $N H6 results, got $COUNT" >&2
    exit 1
  fi
  if ! python "$ROOT/summarize_h6_ot_gradient.py" \
    --jsonl "$RUN_ROOT/h6_results.jsonl" \
    --summary "$RUN_ROOT/h6_summary.csv" \
    --gradients "$RUN_ROOT/h6_optimizer_gradient_traces.csv" \
    --city-gradients "$RUN_ROOT/h6_city_gradient_traces.csv" \
    >"$RUN_ROOT/summary.txt" 2>"$RUN_ROOT/summary.err"; then
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] H6 summary failed" >&2
    cat "$RUN_ROOT/summary.err" >&2
    exit 1
  fi
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H6 read-only completed, training_updates=0: $RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H6 failed; inspect $RUN_ROOT/train.out" >&2
  tail -n 60 "$RUN_ROOT/train.out" >&2
  exit 1
fi
