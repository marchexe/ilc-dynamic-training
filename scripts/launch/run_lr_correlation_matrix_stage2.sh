#!/usr/bin/env bash
# Stage 2 of the supervisor's 4-run LR-vs-mistag-score comparison matrix:
# launches run (3) (50 generations, 0.1 epoch/generation, 1M-event
# control, iutgpu01:0-3) and run (4) (25 generations, 0.2
# epoch/generation, 1M-event control, iutgpu01:4-7) simultaneously via
# nohup. See configs/experiments/anchor_copy_lr_recenter_100gen.yaml for
# the full matrix rationale.
#
# Gate: only run this after stage1 (run_lr_correlation_matrix_stage1.sh)
# has finished AND compare_runs.py shows a real difference between run (1)
# (100gen/150k) and run (2) (100gen/1M) -- if evaluation noise wasn't the
# bottleneck there, spending more compute on the training-noise axis these
# two runs target is unlikely to be informative either. This script does
# NOT check that for you -- it is your judgment call after reading the
# stage1 comparison, same as the supervisor's original conditional
# ("(3)/(4) only if (2) shows a real difference vs (1)").
#
# Run it yourself, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_matrix_stage2.sh > /tmp/lr_matrix_stage2_wrapper.log 2>&1 &
#
# Does not wait for either run to finish. Refuses to launch (without
# starting anything) if either target run directory already exists.
#
# After both finish, add them to the same comparison:
#   .venv/bin/python scripts/training/pbt/reporting/compare_runs.py \
#     runs/pbt/anchor_copy_lr_recenter_100gen_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_100gen_1mval_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_50gen_1mval_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_25gen_1mval_<TIMESTAMP> \
#     --label "100gen/150k" --label "100gen/1M" --label "50gen/0.1ep/1M" --label "25gen/0.2ep/1M" \
#     --plot runs/pbt/lr_matrix_full_comparison.png
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
LAUNCH_DIR="$PROJECT_DIR/scripts/launch"
LOG_DIR="$PROJECT_DIR/runs/launch_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_50GEN="anchor_copy_lr_recenter_50gen_1mval_${TIMESTAMP}"
EXP_25GEN="anchor_copy_lr_recenter_25gen_1mval_${TIMESTAMP}"

RUN_DIR_50GEN="$PROJECT_DIR/runs/pbt/$EXP_50GEN"
RUN_DIR_25GEN="$PROJECT_DIR/runs/pbt/$EXP_25GEN"

for d in "$RUN_DIR_50GEN" "$RUN_DIR_25GEN"; do
  if [ -e "$d" ]; then
    echo "refusing to launch: run directory already exists: $d" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

LOG_50GEN="$LOG_DIR/${EXP_50GEN}.log"
LOG_25GEN="$LOG_DIR/${EXP_25GEN}.log"
PID_FILE_50GEN="$LOG_DIR/${EXP_50GEN}.pid"
PID_FILE_25GEN="$LOG_DIR/${EXP_25GEN}.pid"

cd "$PROJECT_DIR"

nohup bash "$LAUNCH_DIR/run_lr_correlation_50gen_1mval.sh" "$EXP_50GEN" >"$LOG_50GEN" 2>&1 &
PID_50GEN=$!
disown "$PID_50GEN"
echo "$PID_50GEN" > "$PID_FILE_50GEN"

nohup bash "$LAUNCH_DIR/run_lr_correlation_25gen_1mval.sh" "$EXP_25GEN" >"$LOG_25GEN" 2>&1 &
PID_25GEN=$!
disown "$PID_25GEN"
echo "$PID_25GEN" > "$PID_FILE_25GEN"

cat <<SUMMARY
Launched (3) 50gen/0.1ep/1M-val: name=$EXP_50GEN pid=$PID_50GEN
  log: $LOG_50GEN
  pid file: $PID_FILE_50GEN
  run dir (once created): $RUN_DIR_50GEN

Launched (4) 25gen/0.2ep/1M-val: name=$EXP_25GEN pid=$PID_25GEN
  log: $LOG_25GEN
  pid file: $PID_FILE_25GEN
  run dir (once created): $RUN_DIR_25GEN

Both processes are detached (nohup + disown) and running in the background.
This script did not wait for either to finish. Both durations are
unmeasured (1M-val control tier) -- check logs after the first several
generations.
SUMMARY
