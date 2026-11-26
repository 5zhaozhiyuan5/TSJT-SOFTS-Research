#!/usr/bin/env bash
# H4: read-only counterfactual on existing original F2 checkpoint. NO TRAINING.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CHECKPOINT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CHECKPOINT="${1:-${PMME_H4_CHECKPOINT:-$DEFAULT_CHECKPOINT}}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Existing ORIGINAL F2 Stage-I checkpoint is missing: $CHECKPOINT" >&2
  echo "Usage: bash run_h4_error_profiles.sh /absolute/path/to/full_original_f2_stage1.pth" >&2
  echo "DO NOT retrain F2; locate the previous checkpoint instead." >&2
  exit 2
fi
CHECKPOINT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
N="${PMME_H4_PROBES:-4}"
SOURCE_POINTS="${PMME_H4_SOURCE_POINTS:-256}"
TARGET_POINTS="${PMME_H4_TARGET_POINTS:-128}"
MIN_GROUP="${PMME_H4_MIN_GROUP:-12}"
MIN_MASS="${PMME_H4_MIN_ALPHA_MASS:-0.02}"
GROUPS="${PMME_H4_GROUPS:-accurate,inaccurate,short_weak,mid_weak,long_weak,balanced}"
KEEP="${PMME_H4_KEEP_FRACTION:-0.5}"
SEED="${PMME_H4_SEED:-1}"
GPU="${PMME_H4_GPU:-0}"
if [[ "$SOURCE_POINTS" == 0 && "$TARGET_POINTS" == 0 ]]; then
  DEFAULT_DIR="$ROOT/experiments/h4-full-ot"
else
  DEFAULT_DIR="$ROOT/experiments/h4-capped-ot"
fi
RUN_ROOT="${PMME_H4_RUN_ROOT:-$DEFAULT_DIR}"
mkdir -p "$RUN_ROOT"
CKPT_SHA="$(sha256sum "$CHECKPOINT"|awk '{print $1}')"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
CONFIG="$(printf 'branch=%s\ncommit=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\nseed=%s\nprobe_batches=%s\nsource_points=%s\ntarget_points=%s\ngroups=%s\nkeep_fraction=%s\nmin_group=%s\nmin_mass=%s\ngpu=%s\ntraining_updates=0\n' \
  "$(git -C "$ROOT" branch --show-current)" "$COMMIT" "$CHECKPOINT" "$CKPT_SHA" \
  "$SEED" "$N" "$SOURCE_POINTS" "$TARGET_POINTS" "$GROUPS" "$KEEP" "$MIN_GROUP" "$MIN_MASS" "$GPU")"
if [[ -f "$RUN_ROOT/run_config.txt" ]]; then
  [[ "$(cat "$RUN_ROOT/run_config.txt")" == "$CONFIG" ]] || {
    echo "[ERROR] Run directory already used for different settings/commit" >&2
    exit 3
  }
else
  printf '%s\n' "$CONFIG" > "$RUN_ROOT/run_config.txt"
fi
if [[ -f "$RUN_ROOT/COMPLETED" ]]; then
  echo "[SKIP] H4 is already completed: $RUN_ROOT"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
rm -f "$RUN_ROOT/FAILED"
echo "[START] H4 support-only read-only error-profile audit: probes=$N source_points=$SOURCE_POINTS target_points=$TARGET_POINTS"
if (
  cd "$ROOT"
  python -u audit_h4_error_profiles.py \
    --checkpoint "$CHECKPOINT" \
    --output "$RUN_ROOT/h4_results.jsonl" \
    --probe-batches "$N" \
    --source-points "$SOURCE_POINTS" \
    --target-points "$TARGET_POINTS" \
    --groups "$GROUPS" \
    --keep-fraction "$KEEP" \
    --min-group-points "$MIN_GROUP" \
    --min-alpha-mass "$MIN_MASS" \
    --seed "$SEED" \
    --gpu "$GPU"
) >"$RUN_ROOT/train.out" 2>&1; then
  grep -Fq "H4_AUDIT_FINISHED $N training_updates 0" "$RUN_ROOT/train.out" || {
    touch "$RUN_ROOT/FAILED"; echo "[ERROR] Missing H4 completion marker" >&2;exit 1;
  }
  COUNT="$(wc -l < "$RUN_ROOT/h4_results.jsonl")"
  [[ "$COUNT" -eq "$N" ]] || {
    touch "$RUN_ROOT/FAILED";echo "[ERROR] Expected $N probe records; found $COUNT" >&2;exit 1;
  }
  python "$ROOT/summarize_h4_error_profiles.py" \
    --jsonl "$RUN_ROOT/h4_results.jsonl" \
    --csv "$RUN_ROOT/h4_summary.csv" >"$RUN_ROOT/summary.txt"
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H4, TRAINING_UPDATES=0, output=$RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H4 audit failed; inspect $RUN_ROOT/train.out" >&2
  tail -n 60 "$RUN_ROOT/train.out" >&2
  exit 1
fi
