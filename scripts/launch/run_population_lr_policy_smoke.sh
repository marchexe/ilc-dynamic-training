#!/usr/bin/env bash
# Launch a small, real end-to-end verification run of the population_lr_policy
# PBT strategy (scripts/training/pbt/planning/population_lr_policy.py) on
# 4 GPUs of one node. Not a unit test: this trains for real, evaluates the real
# control (5k rows/class) and monitor (50k rows/class) proxy-validation
# tiers, and exercises real forward-decision / accept / rollback checkpoint
# copies against configs/experiments/population_lr_policy_smoke.yaml (7
# generations, monitor tier every other generation -- small enough to finish
# in minutes on 4 GPUs). Not executed automatically -- run it yourself,
# typically via nohup, e.g.:
#   nohup bash scripts/launch/run_population_lr_policy_smoke.sh > /tmp/population_lr_policy_smoke.log 2>&1 &
#
# Optional argument: an experiment name override (defaults to the config's
# own name).
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="configs/experiments/population_lr_policy_smoke.yaml"
SLOTS="${ILC_SLOTS:?set ILC_SLOTS to 4 comma-separated host:gpu slots of idle GPUs}"
EXPERIMENT_NAME="${1:-population_lr_policy_smoke}"

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
