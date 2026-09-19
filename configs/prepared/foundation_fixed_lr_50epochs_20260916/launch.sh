#!/usr/bin/env bash
# User-run only. Production launch or same-config resume; never reduced --smoke.
set -euo pipefail
cd /data/suehara/part/march
[[ $(hostname -s) == iutgpu01 ]] || { echo 'Run directly on iutgpu01' >&2; exit 1; }
run_name=foundation_fixed_lr_50epochs_20260916
run_dir="runs/pbt/$run_name"
log_dir="logs/experiments/$run_name"
resume_args=()
if [[ $# == 1 && $1 == --resume ]]; then
  [[ -f "$run_dir/manifest.json" ]] || { echo 'No production manifest to resume' >&2; exit 1; }
  resume_args=(--resume)
elif [[ $# != 0 ]]; then
  echo 'Usage: launch.sh [--resume]' >&2; exit 1
elif [[ -e "$run_dir" ]]; then
  echo 'Production output already exists; inspect it and use --resume only after prior processes exit' >&2; exit 1
fi
# Reuse the shared availability check and pin the same physical GPU order.
CUDA_VISIBLE_DEVICES="$(.venv/bin/python - <<'PY'
import contextlib,sys
sys.path.insert(0,'scripts')
from launch.experiment import available_gpus
with contextlib.redirect_stdout(sys.stderr):
    devices=available_gpus(range(5))
print(','.join(devices))
PY
)"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
mkdir -p "$log_dir"
# flock prevents a second launcher/resume while this production runner is alive.
nohup flock --nonblock --no-fork "$log_dir/runner.lock" \
  .venv/bin/python scripts/training/run_pbt.py \
  --config configs/experiments/foundation_fixed_lr_50epochs.yaml \
  --gpus 0,1,2,3,4 "${resume_args[@]}" \
  </dev/null >>"$log_dir/main.log" 2>&1 &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$log_dir/launcher.pid"
echo "Submitted runner PID $runner_pid; check $log_dir/main.log and the manifest for startup/completion."
