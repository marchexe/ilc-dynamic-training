#!/usr/bin/env bash
# Stage 1 of the supervisor's 4-run LR-vs-mistag-score comparison matrix:
# launches run (1) (100 generations, 150k-event control, iutgpu01:0-3) and
# run (2) (100 generations, 1M-event control, iutgpu01:4-7) simultaneously
# via nohup, using the whole iutgpu01 node. See
# configs/experiments/anchor_copy_lr_recenter_100gen.yaml for the full
# matrix rationale.
#
# Run it yourself, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_matrix_stage1.sh > /tmp/lr_matrix_stage1_wrapper.log 2>&1 &
#
# Does not wait for either run to finish. Refuses to launch (without
# starting anything) if either target run directory already exists.
#
# After both finish, compare them before deciding on stage 2:
#   .venv/bin/python scripts/training/pbt/reporting/compare_runs.py \
#     runs/pbt/anchor_copy_lr_recenter_100gen_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_100gen_1mval_<TIMESTAMP> \
#     --label "100gen/150k" --label "100gen/1M" \
#     --plot runs/pbt/lr_matrix_stage1_comparison.png
# If run (2)'s Pearson r / 95% CI is materially tighter or stronger than
# run (1)'s, evaluation noise was a real contributor -- proceed to stage 2
# (scripts/launch/run_lr_correlation_matrix_stage2.sh), which spends more
# compute on the *training*-noise axis instead, now that it's the
# remaining open question. If the two are essentially the same, stage 2 is
# probably not worth running: evaluation noise wasn't the bottleneck, and
# the honest conclusion is that the LR effect is real but small (r^2 on
# the order of a few percent) rather than "unmeasured because too noisy".
#
# Timing: run (1) ~3h estimated (see its own config comment); run (2) is
# unmeasured -- see configs/presets/shared/proxy_control_1m_override.yaml's
# cost warning. Check both logs after the first several generations before
# assuming either finishes within any particular window.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
LAUNCH_DIR="$PROJECT_DIR/scripts/launch"
LOG_DIR="$PROJECT_DIR/runs/launch_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_150K="anchor_copy_lr_recenter_100gen_${TIMESTAMP}"
EXP_1M="anchor_copy_lr_recenter_100gen_1mval_${TIMESTAMP}"

RUN_DIR_150K="$PROJECT_DIR/runs/pbt/$EXP_150K"
RUN_DIR_1M="$PROJECT_DIR/runs/pbt/$EXP_1M"

for d in "$RUN_DIR_150K" "$RUN_DIR_1M"; do
  if [ -e "$d" ]; then
    echo "refusing to launch: run directory already exists: $d" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

LOG_150K="$LOG_DIR/${EXP_150K}.log"
LOG_1M="$LOG_DIR/${EXP_1M}.log"
PID_FILE_150K="$LOG_DIR/${EXP_150K}.pid"
PID_FILE_1M="$LOG_DIR/${EXP_1M}.pid"

cd "$PROJECT_DIR"

nohup bash "$LAUNCH_DIR/run_lr_correlation_100gen.sh" "$EXP_150K" >"$LOG_150K" 2>&1 &
PID_150K=$!
disown "$PID_150K"
echo "$PID_150K" > "$PID_FILE_150K"

nohup bash "$LAUNCH_DIR/run_lr_correlation_100gen_1mval.sh" "$EXP_1M" >"$LOG_1M" 2>&1 &
PID_1M=$!
disown "$PID_1M"
echo "$PID_1M" > "$PID_FILE_1M"

cat <<SUMMARY
Launched (1) 100gen/150k-val: name=$EXP_150K pid=$PID_150K
  log: $LOG_150K
  pid file: $PID_FILE_150K
  run dir (once created): $RUN_DIR_150K

Launched (2) 100gen/1M-val  : name=$EXP_1M pid=$PID_1M
  log: $LOG_1M
  pid file: $PID_FILE_1M
  run dir (once created): $RUN_DIR_1M

Both processes are detached (nohup + disown) and running in the background.
This script did not wait for either to finish. Estimated duration for (1):
~3h. Duration for (2) is unmeasured -- check its log after the first
several generations. Compare once both finish with compare_runs.py (see
this script's own header comment) before deciding on stage 2.
SUMMARY
