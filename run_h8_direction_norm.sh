#!/usr/bin/env bash
# H8 read-only direction/norm and AdamW-step-matched OT audit, no F2/Stage-II training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CHECKPOINT="${1:-${PMME_H8_CHECKPOINT:-$DEFAULT_CKPT}}"
EXPECTED_SHA="f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Original F2 checkpoint absent: $CHECKPOINT" >&2
  echo "H8 refuses to retrain F2. Give an existing Stage-I ORIGINAL checkpoint." >&2
  exit 2
fi
CHECKPOINT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
FOUND_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$FOUND_SHA" != "$EXPECTED_SHA" ]]; then
  echo "[ERROR] Checkpoint SHA mismatch: $FOUND_SHA; required $EXPECTED_SHA" >&2
  echo "Refusing to run on PRC/H5 or an unverified checkpoint." >&2
  exit 2
fi
N="${PMME_H8_PROBES:-8}"
SRC="${PMME_H8_SOURCE_POINTS:-256}"
TGT="${PMME_H8_TARGET_POINTS:-128}"
GPU="${PMME_H8_GPU:-0}"
if [[ "$SRC" == 0 && "$TGT" == 0 ]]; then
  DEFAULT_DIR="$ROOT/experiments/h8-full-ot"
else
  DEFAULT_DIR="$ROOT/experiments/h8-capped-ot"
fi
RUN_ROOT="${PMME_H8_RUN_ROOT:-$DEFAULT_DIR}"
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
  echo "[SKIP] H8 completed: $RUN_ROOT"
  exit 0
fi
rm -f "$RUN_ROOT/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[START] H8 OT direction-vs-magnitude & AdamW matched-step audit, existing checkpoint only, probes=$N, source/target points=$SRC/$TGT"
if (
  cd "$ROOT"
  python -u audit_h8_direction_norm.py \
    --checkpoint "$CHECKPOINT" \
    --probe-batches "$N" \
    --source-points "$SRC" \
    --target-points "$TGT" \
    --gpu "$GPU" \
    --output "$RUN_ROOT/h8_results.jsonl"
) >"$RUN_ROOT/train.out" 2>&1; then
  if ! grep -Fq "H8_AUDIT_FINISHED $N training_updates 0 stage2_runs 0" "$RUN_ROOT/train.out"; then
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] H8 audit missing completion marker" >&2
    exit 1
  fi
  COUNT="$(wc -l <"$RUN_ROOT/h8_results.jsonl")"
  if [[ "$COUNT" -ne "$N" ]]; then
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] Expected $N H8 results, got $COUNT" >&2
    exit 1
  fi
  H7_ARCHIVED="$ROOT/experiment_logs/experiment-pmme-h7-ot-gradient-projection-20261127/full_8/h7_results.jsonl"
  REPLAY_ARGS=()
  if [[ "$SRC" == 0 && "$TGT" == 0 && -f "$H7_ARCHIVED" ]]; then
    REPLAY_ARGS=(--h7-reference "$H7_ARCHIVED" --h7-replay-check "$RUN_ROOT/h8_vs_h7_reference_check.json")
  fi
  if ! python "$ROOT/summarize_h8_direction_norm.py" \
    --jsonl "$RUN_ROOT/h8_results.jsonl" \
    --summary "$RUN_ROOT/h8_policy_summary.csv" \
    --traces "$RUN_ROOT/h8_optimizer_direction_traces.csv" \
    --city-traces "$RUN_ROOT/h8_city_gradient_norms.csv" \
    --decision "$RUN_ROOT/h8_decision.json" \
    "${REPLAY_ARGS[@]}" \
    >"$RUN_ROOT/summary.txt" 2>"$RUN_ROOT/summary.err"; then
    touch "$RUN_ROOT/FAILED"
    echo "[ERROR] H8 summary failed" >&2
    cat "$RUN_ROOT/summary.err" >&2
    exit 1
  fi
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H8 eight-policy read-only completed, training_updates=0: $RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H8 failed; inspect $RUN_ROOT/train.out" >&2
  tail -n 60 "$RUN_ROOT/train.out" >&2
  exit 1
fi
