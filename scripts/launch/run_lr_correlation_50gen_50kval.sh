#!/usr/bin/env bash
# Launch run (3) of the supervisor's 4-run LR-vs-mistag-score comparison
# matrix, revised after stage1 (see
# configs/experiments/anchor_copy_lr_recenter_50gen_50kval.yaml's own
# comment for why this now uses the cheap 150k-row control tier instead of
# the originally-planned 1M-row tier) on 4 GPUs of one node (the larger node was
# unavailable at the time; the other node only had 4
# free GPUs, so this and the 25gen sibling run sequentially, not in
# parallel -- see run_lr_correlation_matrix_stage2_50kval.sh). Not executed
# automatically -- run it yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_50gen_50kval.sh > /tmp/lr_50gen_50kval.log 2>&1 &
# or invoke it through scripts/launch/run_lr_correlation_matrix_stage2.sh.
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The stage2 wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="configs/experiments/anchor_copy_lr_recenter_50gen_50kval.yaml"
SLOTS="${ILC_SLOTS:?set ILC_SLOTS to 4 comma-separated host:gpu slots of idle GPUs}"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_50gen_50kval}"

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
