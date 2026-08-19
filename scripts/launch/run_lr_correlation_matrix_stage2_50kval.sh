#!/usr/bin/env bash
# Sequential (not parallel) launch of the revised runs (3) and (4) -- see
# configs/experiments/anchor_copy_lr_recenter_50gen_50kval.yaml's own
# comment for why these use the cheap 150k-row control tier instead of
# the originally-planned 1M-row tier. iutgpu01 is fully occupied by
# another user's job as of 2026-08-16 (was free for the stage1 runs, is
# not now); iutgpu02 has 4 free GPUs, enough for one 4-member population
# at a time, not both simultaneously like stage1 used iutgpu01's 8 -- so
# this runs (3) then (4) back to back on iutgpu02:0-3, not in parallel on
# separate GPU groups.
#
# Run it yourself, e.g.:
#   nohup bash scripts/launch/run_lr_correlation_matrix_stage2_50kval.sh > /tmp/lr_matrix_stage2_50kval_wrapper.log 2>&1 &
#
# Blocks until both finish (each leg script's `exec` only replaces the
# per-run subprocess this wrapper spawns, not the wrapper itself, so the
# second launch genuinely waits for the first's process to exit) -- wrap
# the whole thing in nohup, not just note it, if you want it to survive a
# disconnected shell.
#
# Refuses to launch (without starting anything) if either target run
# directory already exists.
#
# After both finish, add them to the same comparison as stage1:
#   .venv/bin/python scripts/training/pbt/reporting/compare_runs.py \
#     runs/pbt/anchor_copy_lr_recenter_100gen_20260816_105629 \
#     runs/pbt/anchor_copy_lr_recenter_100gen_1mval_20260816_105629 \
#     runs/pbt/anchor_copy_lr_recenter_50gen_50kval_<TIMESTAMP> \
#     runs/pbt/anchor_copy_lr_recenter_25gen_50kval_<TIMESTAMP> \
#     --label "100gen/150k" --label "100gen/1M" --label "50gen/0.1ep/50k" --label "25gen/0.2ep/50k" \
#     --plot runs/pbt/lr_matrix_full_comparison.png
set -euo pipefail

PROJECT_DIR="/data/suehara/part/march"
LAUNCH_DIR="$PROJECT_DIR/scripts/launch"
LOG_DIR="$PROJECT_DIR/runs/launch_logs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_50GEN="anchor_copy_lr_recenter_50gen_50kval_${TIMESTAMP}"
EXP_25GEN="anchor_copy_lr_recenter_25gen_50kval_${TIMESTAMP}"

RUN_DIR_50GEN="$PROJECT_DIR/runs/pbt/$EXP_50GEN"
RUN_DIR_25GEN="$PROJECT_DIR/runs/pbt/$EXP_25GEN"

for d in "$RUN_DIR_50GEN" "$RUN_DIR_25GEN"; do
  if [ -e "$d" ]; then
    echo "refusing to launch: run directory already exists: $d" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

echo "$(date -Iseconds) launching (3) 50gen/0.1ep/50k-val on iutgpu02:0-3: $EXP_50GEN"
bash "$LAUNCH_DIR/run_lr_correlation_50gen_50kval.sh" "$EXP_50GEN" >"$LOG_DIR/${EXP_50GEN}.log" 2>&1
echo "$(date -Iseconds) (3) finished: $(.venv/bin/python -c "import json; print(json.load(open('$RUN_DIR_50GEN/manifest.json'))['status'])")"

echo "$(date -Iseconds) launching (4) 25gen/0.2ep/50k-val on iutgpu02:0-3: $EXP_25GEN"
bash "$LAUNCH_DIR/run_lr_correlation_25gen_50kval.sh" "$EXP_25GEN" >"$LOG_DIR/${EXP_25GEN}.log" 2>&1
echo "$(date -Iseconds) (4) finished: $(.venv/bin/python -c "import json; print(json.load(open('$RUN_DIR_25GEN/manifest.json'))['status'])")"

cat <<SUMMARY
Both sequential runs finished.
  (3) 50gen/0.1ep/50k-val: $RUN_DIR_50GEN
  (4) 25gen/0.2ep/50k-val: $RUN_DIR_25GEN
Compare with scripts/training/pbt/reporting/compare_runs.py (see this
script's own header comment for the exact command).
SUMMARY
