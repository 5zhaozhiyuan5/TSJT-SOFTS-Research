#!/usr/bin/env bash
# H5: EXISTING F2 Stage-I checkpoint only; no training; no Stage-II.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CHECKPOINT="${1:-${PMME_H5_CHECKPOINT:-$DEFAULT_CKPT}}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Existing ORIGINAL F2 full checkpoint absent: $CHECKPOINT" >&2
  echo "H5 deliberately refuses to start training F2. Pass an existing .pth path." >&2
  exit 2
fi
CHECKPOINT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
N="${PMME_H5_PROBES:-8}"
SRC="${PMME_H5_SOURCE_POINTS:-256}"
TGT="${PMME_H5_TARGET_POINTS:-128}"
SEED="${PMME_H5_SEED:-1}"
GPU="${PMME_H5_GPU:-0}"
if [[ "$SRC" == 0 && "$TGT" == 0 ]]; then
  DEFAULT_DIR="$ROOT/experiments/h5-full-ot"
else
  DEFAULT_DIR="$ROOT/experiments/h5-capped-ot"
fi
RUN_ROOT="${PMME_H5_RUN_ROOT:-$DEFAULT_DIR}"
mkdir -p "$RUN_ROOT"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
CKPT_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
CONFIG="$(printf 'branch=%s\ncommit=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nseed=%s\nprobes=%s\nsource_points=%s\ntarget_points=%s\nstrength=0.5\not_scale=0.01\ngpu=%s\ntraining_updates=0\nstage2_runs=0\n' \
  "$(git -C "$ROOT" branch --show-current)" "$COMMIT" "$CHECKPOINT" "$CKPT_SHA" \
  "$SEED" "$N" "$SRC" "$TGT" "$GPU")"
if [[ -f "$RUN_ROOT/run_config.txt" ]]; then
  [[ "$(cat "$RUN_ROOT/run_config.txt")" == "$CONFIG" ]] || {
    echo "[ERROR] Run folder already used by a different setting/commit." >&2
    exit 3
  }
else
  printf '%s\n' "$CONFIG" >"$RUN_ROOT/run_config.txt"
fi
if [[ -f "$RUN_ROOT/COMPLETED" ]]; then
  echo "[SKIP] Completed run: $RUN_ROOT"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
rm -f "$RUN_ROOT/FAILED"
echo "[START] H5 checkpoint-only 19-way shadow audit: $N batches, $SRC/$TGT OT points"
if (
 cd "$ROOT"
 python -u audit_h5_target_skill.py \
   --checkpoint "$CHECKPOINT" --output "$RUN_ROOT/h5_results.jsonl" \
   --probe-batches "$N" --source-points "$SRC" \
   --target-points "$TGT" --seed "$SEED" --gpu "$GPU"
) >"$RUN_ROOT/train.out" 2>&1; then
  if ! grep -Fq "H5_AUDIT_FINISHED $N training_updates 0" "$RUN_ROOT/train.out"; then
    touch "$RUN_ROOT/FAILED";echo "[ERROR] H5 missing end marker" >&2;exit 1
  fi
  COUNT="$(wc -l <"$RUN_ROOT/h5_results.jsonl")"
  if [[ "$COUNT" -ne "$N" ]]; then
    touch "$RUN_ROOT/FAILED";echo "[ERROR] H5 got $COUNT results, need $N" >&2;exit 1
  fi
  python "$ROOT/summarize_h5_target_skill.py" \
    --jsonl "$RUN_ROOT/h5_results.jsonl" \
    --csv "$RUN_ROOT/h5_summary.csv" \
    --contrasts "$RUN_ROOT/h5_pairwise_contrasts.csv" >"$RUN_ROOT/summary.txt"
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H5 no training. Results: $RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H5 failed. Inspect: $RUN_ROOT/train.out" >&2
  tail -n 40 "$RUN_ROOT/train.out" >&2
  exit 1
fi
