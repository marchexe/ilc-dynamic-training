#!/usr/bin/env bash
# Run population configurations one after another, so that at most one run holds GPUs at a time.
#
#   setsid nohup bash scripts/experiments/run_queue.sh NAME CONFIG [CONFIG ...] > /dev/null 2>&1 < /dev/null &
#
# Each CONFIG starts with train800k.py once the previous run has completed. While no node has an idle GPU per
# member (exit 69), the start is retried every 10 minutes. Any other failure, or a run that ends without
# completing, stops the queue; nothing after it starts. Progress: logs/experiments/queue-NAME.log.
set -u
cd "$(dirname "$0")/../.."
name=$1; shift
py=.venv/bin/python
log=logs/experiments/queue-$name.log
mkdir -p "$(dirname "$log")"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }

state() {
  $py scripts/experiments/train800k.py status --config "$1" 2>/dev/null | $py -c '
import sys, json
d = json.load(sys.stdin)
print("dead" if d.get("incomplete_without_live_processes") else d.get("run_status"))'
}

say "queue start (pid $$): $*"
for config in "$@"; do
  while true; do
    $py scripts/experiments/train800k.py start --config "$config" >> "$log" 2>&1
    code=$?
    [ $code -eq 0 ] && { say "started $config"; break; }
    [ $code -ne 69 ] && { say "start of $config failed with exit $code; queue stops"; exit 1; }
    say "no placement for $config yet; retrying in 10 min"; sleep 600
  done
  sleep 300
  while true; do
    s=$(state "$config")
    case "$s" in
      completed) say "$config completed"; break ;;
      failed|interrupted|dead) say "$config ended as $s; queue stops"; exit 1 ;;
    esac
    sleep 300
  done
  sleep 60
done
say "queue done"
