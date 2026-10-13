#!/usr/bin/env bash
# Launch seed replicate 2 of 3 -- see run_lr_correlation_100gen_seed2.sh
# and the config's own comment for the full rationale. On 4 GPUs of one node.
# Not executed automatically -- run it yourself, typically via nohup, or
# through scripts/launch/run_lr_correlation_seed_replicates.sh.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="configs/experiments/anchor_copy_lr_recenter_100gen_seed3.yaml"
SLOTS="${ILC_SLOTS:?set ILC_SLOTS to 4 comma-separated host:gpu slots of idle GPUs}"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_100gen_seed3}"

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
