#!/usr/bin/env bash
# Launch the controller-active (dynamic_controller.mode: active) leg of the
# long A/B experiment on iutgpu01:4-7. Not executed automatically -- run it
# yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_long_controller_active.sh > /tmp/controller_active.log 2>&1 &
# or invoke it through scripts/launch/run_long_controller_ab.sh, which
# launches this and the controller-off sibling together with a shared
# timestamp and its own log/pid bookkeeping.
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The AB wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/pretrained_pbt_4gpu_long_controller_active.yaml"
SLOTS="iutgpu01:4,iutgpu01:5,iutgpu01:6,iutgpu01:7"
EXPERIMENT_NAME="${1:-pretrained_pbt_4gpu_long_controller_active}"

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
