#!/usr/bin/env bash
# H10 author-parity A and gradient provenance B. Never retrains original F2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="experiment/pmme-h10-update-provenance-20261127"
EXPECTED_SHA="f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf"
DEFAULT_CKPT="/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth"
CHECKPOINT="${PMME_H10_CHECKPOINT:-$DEFAULT_CKPT}"
PHASE="${PMME_H10_PHASE:-a}"
UPDATES="${PMME_H10_MAX_UPDATES:-1}"
SRC="${PMME_H10_SOURCE_POINTS:-0}"
TGT="${PMME_H10_TARGET_POINTS:-0}"
GPU="${PMME_H10_GPU:-0}"
MAX_DEV_WINDOWS="${PMME_H10_DEV_MAX_WINDOWS:-0}"
ATOL="0.00002"

case "$PHASE" in a|b|both) ;; *) echo "[ERROR] Phase a/b/both only" >&2;exit 2;;esac
case "$UPDATES" in 1|5) ;; *) echo "[ERROR] Updates 1 or 5 only" >&2;exit 2;;esac
if [[ "$PHASE" != "b" && ( "$SRC" != "0" || "$TGT" != "0" ) ]]; then
  echo "[ERROR] Official parity must be FULL OT: set source/target=0/0" >&2; exit 2
fi
if [[ "$PHASE" == "b" && ! -f "${PMME_H10_PARITY_REPORT:-}" ]]; then
  echo "[ERROR] H10-B requires passed H10-A JSON in PMME_H10_PARITY_REPORT" >&2;exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Original F2 checkpoint missing: $CHECKPOINT" >&2;exit 2
fi
CHECKPOINT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CHECKPOINT")"
SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$SHA" != "$EXPECTED_SHA" ]]; then
  echo "[ERROR] Pinned original F2 SHA256 mismatch: $SHA" >&2;exit 2
fi
if [[ "$(git -C "$ROOT" branch --show-current)" != "$BRANCH" ]]; then
  echo "[ERROR] Wrong checked-out branch; switch to $BRANCH" >&2;exit 2
fi
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
RUN_ROOT="${PMME_H10_RUN_ROOT:-$ROOT/experiments/h10-${PHASE}-s${SRC}-t${TGT}-u${UPDATES}}"
RUN_ROOT="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$RUN_ROOT")"
mkdir -p "$RUN_ROOT"
PAREQ="${PMME_H10_PARITY_REPORT:-}"
if [[ -n "$PAREQ" ]]; then
  PAREQ="$(python -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$PAREQ")"
fi
CONFIG="$(printf 'branch=%s\ncommit=%s\ncheckpoint=%s\ncheckpoint_sha=%s\nphase=%s\nparity_report=%s\nsource_points=%s\ntarget_points=%s\nmax_updates=%s\nvalidation_days=4\nvalidation_max_windows=%s\nseed=1\ngpu=%s\noriginal_checkpoint_updates=0\nstage2=0\n' \
  "$BRANCH" "$COMMIT" "$CHECKPOINT" "$SHA" "$PHASE" "$PAREQ" "$SRC" "$TGT" "$UPDATES" "$MAX_DEV_WINDOWS" "$GPU")"
if [[ -f "$RUN_ROOT/run_config.txt" ]];then
  if [[ "$(cat "$RUN_ROOT/run_config.txt")" != "$CONFIG" ]];then
    echo "[ERROR] Run config differs; use new PMME_H10_RUN_ROOT" >&2;exit 3
  fi
else
  printf '%s\n' "$CONFIG" >"$RUN_ROOT/run_config.txt"
fi
if [[ -f "$RUN_ROOT/COMPLETED" ]];then
  echo "[SKIP] H10 already completed: $RUN_ROOT";exit 0
fi
rm -f "$RUN_ROOT/FAILED"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

ARGS=(
  --phase "$PHASE"
  --checkpoint "$CHECKPOINT"
  --output "$RUN_ROOT/h10_results.json"
  --max-updates "$UPDATES"
  --source-points "$SRC"
  --target-points "$TGT"
  --validation-max-windows "$MAX_DEV_WINDOWS"
  --parity-atol "$ATOL"
  --gpu "$GPU"
)
if [[ "$PHASE" == "b" ]];then
  ARGS+=(--parity-report "$PAREQ")
fi
echo "[START] H10-$PHASE; official F2 projection=1.0 parity first; no original retrain"
if (
  cd "$ROOT"
  python -u audit_h10_update_provenance.py "${ARGS[@]}"
) >"$RUN_ROOT/train.out" 2>&1;then
  if ! grep -Fq "H10_COMPLETED phase $PHASE parity_passed True original_f2_updates 0 stage2_runs 0" "$RUN_ROOT/train.out";then
    echo "[ERROR] H10 completion marker missing" >&2
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  if ! python "$ROOT/summarize_h10_update_provenance.py" \
      --json "$RUN_ROOT/h10_results.json" \
      --summary "$RUN_ROOT/h10_policy_development.csv" \
      --daywise "$RUN_ROOT/h10_daywise_development.csv" \
      --decision "$RUN_ROOT/h10_decision.json" \
      >"$RUN_ROOT/summary.txt" 2>"$RUN_ROOT/summary.err";then
    echo "[ERROR] H10 report failed" >&2
    tail -n 30 "$RUN_ROOT/summary.err" >&2
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  touch "$RUN_ROOT/COMPLETED"
  echo "[DONE] H10-$PHASE: $RUN_ROOT"
  cat "$RUN_ROOT/summary.txt"
else
  touch "$RUN_ROOT/FAILED"
  echo "[ERROR] H10-$PHASE failed, see train.out. NO attribution conclusion." >&2
  tail -n 70 "$RUN_ROOT/train.out" >&2
  exit 1
fi
