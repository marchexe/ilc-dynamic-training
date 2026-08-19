#!/usr/bin/env bash
# Sequential launch of 3 seed replicates of run (1)
# (anchor_copy_lr_recenter_100gen_20260816_105629) -- same condition,
# only shared.seed differs, see
# configs/experiments/anchor_copy_lr_recenter_100gen_seed2.yaml's own
# comment for the full rationale: measure run-to-run variance in the
# LR-vs-mistag-score correlation before trusting any single run's number,
# including (1)/(2) themselves. Runs on iutgpu02:0-3, one after another
# (only 4 GPUs free there -- iutgpu01 is occupied by another user's job as
# of 2026-08-16). Prioritized ahead of the originally-planned (3)/(4)
# ("longer generation" axis) -- see this project's conversation history:
# replicating an unreplicated single-run finding is more valuable right
# now than testing a new axis on top of unreplicated ones.
#
# Run it yourself, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_seed_replicates.sh > /tmp/lr_seed_replicates_wrapper.log 2>&1 &
#
# Blocks until all three finish (same reasoning as
# run_lr_correlation_matrix_stage2_50kval.sh: each leg script's `exec`
# only replaces the per-run subprocess this wrapper spawns) -- wrap the
# whole thing in nohup if you want it to survive a disconnected shell.
# Estimated ~2h44m each (run (1)'s measured time for the identical
# condition) -- roughly 8h for all three, but check the first replicate's
# actual time before assuming the other two match.
#
# Refuses to launch (without starting anything) if any target run
# directory already exists.
#
# After all three finish, compare all four seeds together (run (1) plus
# these three):
#   .venv/bin/python scripts/training/pbt/reporting/compare_runs.py \
#     runs/pbt/anchor_copy_lr_recenter_100gen_20260816_105629 \
#     runs/pbt/anchor_copy_lr_recenter_100gen_seed2_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_100gen_seed3_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_100gen_seed4_<TIMESTAMP> \
#     --label seed1 --label seed2 --label seed3 --label seed4 \
#     --plot runs/pbt/lr_matrix_seed_replicates.png
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
LAUNCH_DIR="$PROJECT_DIR/scripts/launch"
LOG_DIR="$PROJECT_DIR/runs/launch_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_SEED2="anchor_copy_lr_recenter_100gen_seed2_${TIMESTAMP}"
EXP_SEED3="anchor_copy_lr_recenter_100gen_seed3_${TIMESTAMP}"
EXP_SEED4="anchor_copy_lr_recenter_100gen_seed4_${TIMESTAMP}"

RUN_DIR_SEED2="$PROJECT_DIR/runs/pbt/$EXP_SEED2"
RUN_DIR_SEED3="$PROJECT_DIR/runs/pbt/$EXP_SEED3"
RUN_DIR_SEED4="$PROJECT_DIR/runs/pbt/$EXP_SEED4"

for d in "$RUN_DIR_SEED2" "$RUN_DIR_SEED3" "$RUN_DIR_SEED4"; do
  if [ -e "$d" ]; then
    echo "refusing to launch: run directory already exists: $d" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

echo "$(date -Iseconds) launching seed replicate 1/3 on iutgpu02:0-3: $EXP_SEED2"
bash "$LAUNCH_DIR/run_lr_correlation_100gen_seed2.sh" "$EXP_SEED2" >"$LOG_DIR/${EXP_SEED2}.log" 2>&1
echo "$(date -Iseconds) replicate 1/3 finished: $(.venv/bin/python -c "import json; print(json.load(open('$RUN_DIR_SEED2/manifest.json'))['status'])")"

echo "$(date -Iseconds) launching seed replicate 2/3 on iutgpu02:0-3: $EXP_SEED3"
bash "$LAUNCH_DIR/run_lr_correlation_100gen_seed3.sh" "$EXP_SEED3" >"$LOG_DIR/${EXP_SEED3}.log" 2>&1
echo "$(date -Iseconds) replicate 2/3 finished: $(.venv/bin/python -c "import json; print(json.load(open('$RUN_DIR_SEED3/manifest.json'))['status'])")"

echo "$(date -Iseconds) launching seed replicate 3/3 on iutgpu02:0-3: $EXP_SEED4"
bash "$LAUNCH_DIR/run_lr_correlation_100gen_seed4.sh" "$EXP_SEED4" >"$LOG_DIR/${EXP_SEED4}.log" 2>&1
echo "$(date -Iseconds) replicate 3/3 finished: $(.venv/bin/python -c "import json; print(json.load(open('$RUN_DIR_SEED4/manifest.json'))['status'])")"

cat <<SUMMARY
All 3 seed replicates finished.
  seed2: $RUN_DIR_SEED2
  seed3: $RUN_DIR_SEED3
  seed4: $RUN_DIR_SEED4
Compare all four seeds (including run (1) itself) with
scripts/training/pbt/reporting/compare_runs.py -- see this script's own
header comment for the exact command.
SUMMARY
