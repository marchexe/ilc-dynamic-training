#!/usr/bin/env bash
# Launch run (2) of the supervisor's 4-run LR-vs-mistag-score comparison
# matrix (see configs/experiments/anchor_copy_lr_recenter_100gen.yaml's own
# comment for the full matrix layout and rationale) on iutgpu01:4-7. Not
# executed automatically -- run it yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_100gen_1mval.sh > /tmp/lr_100gen_1mval.log 2>&1 &
# or invoke it through scripts/launch/run_lr_correlation_matrix_stage1.sh,
# which launches this and the 150k-val sibling together on separate GPU
# groups with a shared timestamp and its own log/pid bookkeeping.
#
# Cost is unmeasured -- see configs/presets/shared/proxy_control_1m_override.yaml's
# warning. Watch the first few generations' actual wall-clock (the run's
# events.jsonl / pbt log) before assuming this fits the same window as (1).
#
# Optional argument: an experiment name override (defaults to the config's
# own name). The stage1 wrapper passes a timestamped name here.
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/anchor_copy_lr_recenter_100gen_1mval.yaml"
SLOTS="iutgpu01:4,iutgpu01:5,iutgpu01:6,iutgpu01:7"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_100gen_1mval}"

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
