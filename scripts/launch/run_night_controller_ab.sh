#!/usr/bin/env bash
# Launch both legs of the overnight controller-off/controller-active A/B
# experiment simultaneously via nohup, on the same node (iutgpu01:0-3 for
# controller-off, iutgpu01:4-7 for controller-active). Not executed
# automatically -- run it yourself, e.g.:
#   nohup bash scripts/launch/run_night_controller_ab.sh > /tmp/ab_wrapper_night.log 2>&1 &
# (the wrapper itself backgrounds both children immediately and returns, so
# wrapping the wrapper in nohup is only needed if you want the wrapper's own
# summary output captured after your shell disconnects).
#
# Does not wait for either run to finish. Refuses to launch (without
# starting anything) if either target run directory already exists, so
# re-running this script never silently clobbers an in-progress or
# completed run.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
LAUNCH_DIR="$PROJECT_DIR/scripts/launch"
LOG_DIR="$PROJECT_DIR/runs/launch_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_OFF="pretrained_pbt_4gpu_night_controller_off_${TIMESTAMP}"
EXP_ACTIVE="pretrained_pbt_4gpu_night_controller_active_${TIMESTAMP}"

RUN_DIR_OFF="$PROJECT_DIR/runs/pbt/$EXP_OFF"
RUN_DIR_ACTIVE="$PROJECT_DIR/runs/pbt/$EXP_ACTIVE"

for d in "$RUN_DIR_OFF" "$RUN_DIR_ACTIVE"; do
  if [ -e "$d" ]; then
    echo "refusing to launch: run directory already exists: $d" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"

LOG_OFF="$LOG_DIR/${EXP_OFF}.log"
LOG_ACTIVE="$LOG_DIR/${EXP_ACTIVE}.log"
PID_FILE_OFF="$LOG_DIR/${EXP_OFF}.pid"
PID_FILE_ACTIVE="$LOG_DIR/${EXP_ACTIVE}.pid"

cd "$PROJECT_DIR"

nohup bash "$LAUNCH_DIR/run_night_controller_off.sh" "$EXP_OFF" >"$LOG_OFF" 2>&1 &
PID_OFF=$!
disown "$PID_OFF"
echo "$PID_OFF" > "$PID_FILE_OFF"

nohup bash "$LAUNCH_DIR/run_night_controller_active.sh" "$EXP_ACTIVE" >"$LOG_ACTIVE" 2>&1 &
PID_ACTIVE=$!
disown "$PID_ACTIVE"
echo "$PID_ACTIVE" > "$PID_FILE_ACTIVE"

cat <<SUMMARY
Launched controller-off  : name=$EXP_OFF pid=$PID_OFF
  log: $LOG_OFF
  pid file: $PID_FILE_OFF
  run dir (once created): $RUN_DIR_OFF

Launched controller-active: name=$EXP_ACTIVE pid=$PID_ACTIVE
  log: $LOG_ACTIVE
  pid file: $PID_FILE_ACTIVE
  run dir (once created): $RUN_DIR_ACTIVE

Both processes are detached (nohup + disown) and running in the background.
This script did not wait for either to finish. Estimated duration: ~10h
(see pretrained_pbt_4gpu_night_controller_off.yaml for the timing derivation
and how to rescale generations for a different target length).
SUMMARY
