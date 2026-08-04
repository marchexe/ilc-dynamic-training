#!/usr/bin/env bash
# Launch the controller-off (dynamic_controller.mode: disabled) leg of the
# long A/B experiment on iutgpu01:0-3. Not executed automatically -- run it
# yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_long_controller_off.sh > /tmp/controller_off.log 2>&1 &
# or invoke it through scripts/launch/run_long_controller_ab.sh, which
# launches this and the controller-active sibling together with a shared
# timestamp and its own log/pid bookkeeping.
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The AB wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/pretrained_pbt_4gpu_long_controller_off.yaml"
SLOTS="iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3"
EXPERIMENT_NAME="${1:-pretrained_pbt_4gpu_long_controller_off}"

cd "$PROJECT_DIR"

RUN_DIR="$PROJECT_DIR/runs/pbt/$EXPERIMENT_NAME"
if [ -e "$RUN_DIR" ]; then
  echo "refusing to launch: run directory already exists: $RUN_DIR" >&2
  exit 1
fi

exec "$PROJECT_DIR/.venv/bin/python" scripts/training/run_pbt.py \
  --config "$CONFIG" \
  --slots "$SLOTS" \
  --experiment-name "$EXPERIMENT_NAME"
