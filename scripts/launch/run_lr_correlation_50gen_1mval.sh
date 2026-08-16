#!/usr/bin/env bash
# Launch run (3) of the supervisor's 4-run LR-vs-mistag-score comparison
# matrix (see configs/experiments/anchor_copy_lr_recenter_100gen.yaml's own
# comment for the full matrix layout and rationale) on iutgpu01:0-3. Only
# worth launching if stage1 (runs (1) vs (2)) shows a real difference --
# see scripts/launch/run_lr_correlation_matrix_stage2.sh's own comment.
# Not executed automatically -- run it yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_50gen_1mval.sh > /tmp/lr_50gen_1mval.log 2>&1 &
# or invoke it through scripts/launch/run_lr_correlation_matrix_stage2.sh.
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The stage2 wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/anchor_copy_lr_recenter_50gen_1mval.yaml"
SLOTS="iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_50gen_1mval}"

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
