#!/usr/bin/env bash
# Prepared only; not executed. Recheck iutgpu01 GPU 0-4 availability first.
set -euo pipefail
cd /data/suehara/part/march
mkdir -p logs/experiments
nohup .venv/bin/python scripts/training/run_pbt.py \
  --config configs/experiments/foundation_fixed_lr_20epochs.yaml \
  --slots iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3,iutgpu01:4 \
  > logs/experiments/foundation_fixed_lr_20epochs_20260916.log 2>&1 &
printf '%s\n' "$!" > logs/experiments/foundation_fixed_lr_20epochs_20260916.pid
