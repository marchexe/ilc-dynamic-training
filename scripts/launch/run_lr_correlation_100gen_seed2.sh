#!/usr/bin/env bash
# Launch seed replicate 1 of 3 of run (1) -- see
# configs/experiments/anchor_copy_lr_recenter_100gen_seed2.yaml's own
# comment for why (measuring run-to-run variance before trusting any
# single run's correlation number, including (1)/(2) themselves) -- on
# iutgpu02:0-3. Not executed automatically -- run it yourself, typically
# via nohup, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_100gen_seed2.sh > /tmp/lr_100gen_seed2.log 2>&1 &
# or invoke it through scripts/launch/run_lr_correlation_seed_replicates.sh,
# which runs all 3 replicates back to back on the same 4 GPUs.
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/anchor_copy_lr_recenter_100gen_seed2.yaml"
SLOTS="iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_100gen_seed2}"

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
