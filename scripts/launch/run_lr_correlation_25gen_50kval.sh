#!/usr/bin/env bash
# Launch run (4) of the supervisor's 4-run LR-vs-mistag-score comparison
# matrix, revised after stage1 (see
# configs/experiments/anchor_copy_lr_recenter_50gen_50kval.yaml's own
# comment for why this now uses the cheap 150k-row control tier instead of
# the originally-planned 1M-row tier) on iutgpu02:0-3 (iutgpu01 is fully
# occupied by another user's job as of 2026-08-16; iutgpu02 only has 4
# free GPUs, so this and the 50gen sibling run sequentially, not in
# parallel -- see run_lr_correlation_matrix_stage2_50kval.sh). Not executed
# automatically -- run it yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_25gen_50kval.sh > /tmp/lr_25gen_50kval.log 2>&1 &
# or invoke it through scripts/launch/run_lr_correlation_matrix_stage2.sh.
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The stage2 wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/anchor_copy_lr_recenter_25gen_50kval.yaml"
SLOTS="iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_25gen_50kval}"

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
