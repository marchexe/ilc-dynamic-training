#!/usr/bin/env bash
# Freeze and launch a holdout addendum by itself once every training run of its specification has completed.
#
#   setsid nohup bash scripts/experiments/auto_holdout.sh configs/prepared/NAME > /dev/null 2>&1 < /dev/null &
#
# Waits (checking every 10 min) until each run in NAME/spec.json has manifest status completed, then
# runs holdout_addendum.py --freeze, commits and pushes NAME/protocol.json, and launches the holdout, retrying
# every 10 min while no node has enough idle GPUs (exit 69). It stops without launching if any run ends in
# another state, if the freeze fails, or if the launch fails otherwise. The holdout never selects anything:
# the freeze records each run's validation-selected checkpoint. Progress: logs/experiments/auto-holdout-NAME.log.
set -u
cd "$(dirname "$0")/../.."
prepared=${1%/}
name=$(basename "$prepared")
py=.venv/bin/python
log=logs/experiments/auto-holdout-$name.log
mkdir -p "$(dirname "$log")"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }

runs_state() {
  $py - "$prepared/spec.json" <<'EOF'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
states = []
for run in spec['runs']:
    path = Path('runs/pbt') / run['run'] / 'manifest.json'
    states.append(json.loads(path.read_text()).get('status') if path.is_file() else 'missing')
if all(s == 'completed' for s in states):
    print('completed')
elif any(s in ('failed', 'interrupted') for s in states):
    print('failed ' + ' '.join(states))
else:
    print('waiting ' + ' '.join(states))
EOF
}

say "start (pid $$): $prepared"
while true; do
  s=$(runs_state)
  case "$s" in
    completed) say "all runs completed"; break ;;
    failed*) say "a run did not complete ($s); nothing is frozen or launched"; exit 1 ;;
  esac
  sleep 600
done

if [ ! -f "$prepared/protocol.json" ]; then
  $py scripts/experiments/holdout_addendum.py --prepared "$prepared" --freeze >> "$log" 2>&1 \
    || { say "freeze failed; stop"; exit 1; }
  for attempt in 1 2 3; do
    if git add "$prepared/protocol.json" "$prepared"/protocol.sha256 2>/dev/null \
       && git commit -q -m "Freeze the $name holdout selection" -- "$prepared" >> "$log" 2>&1; then
      break
    fi
    sleep 30
  done
  git push -q >> "$log" 2>&1 || say "push failed; the freeze commit is local only"
  say "frozen and committed: $(git log --oneline -1)"
fi

while true; do
  $py scripts/experiments/holdout_addendum.py --prepared "$prepared" >> "$log" 2>&1
  code=$?
  [ $code -eq 0 ] && { say "holdout launched"; break; }
  [ $code -ne 69 ] && { say "launch failed with exit $code; stop"; exit 1; }
  say "no GPUs for the holdout yet; retrying in 10 min"; sleep 600
done
