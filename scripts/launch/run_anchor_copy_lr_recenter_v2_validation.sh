#!/usr/bin/env bash
# Shape/correctness validation of the rewritten anchor_copy_lr_recenter
# strategy (widened symmetric spread, recenter_momentum_fraction,
# plateau_escape_after_generations/plateau_escape_widen_factor) -- see
# configs/experiments/anchor_copy_lr_recenter_v2_validation.yaml's own
# comment for what this run is and is not for.
#
# On iutgpu02:0-3 (4 GPUs, A100-PCIe-40GB) plus one GPU borrowed from
# iutgpu05 (iutgpu05:1, A100-80GB-PCIe, the only free one there as of
# 2026-08-20) -- true 1-GPU-per-member parallelism for all 5 members,
# multi-host --slots (host:gpu pairs, same project .venv path on both
# hosts, per run_pbt.py's own --slots help text). Avoids both the
# round-robin slowdown of packing 5 members onto 4 slots and switching to
# the weaker/heterogeneous iutgpu06 (RTX A6000/4000 Ada) node.
#
# Run it yourself, typically via nohup, e.g.:
#   nohup bash scripts/launch/run_anchor_copy_lr_recenter_v2_validation.sh > /tmp/anchor_v2_validation.log 2>&1 &
#
# Optional argument: an experiment name override (defaults to the config's
# own name).
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
CONFIG="configs/experiments/anchor_copy_lr_recenter_v2_validation.yaml"
SLOTS="iutgpu05:1,iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3"
EXPERIMENT_NAME="${1:-anchor_copy_lr_recenter_v2_validation}"

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
