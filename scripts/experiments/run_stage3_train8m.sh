#!/usr/bin/env bash
# Stage 3 one-command launcher: verifies revision, frozen protocol, arm parity,
# data and disk, then submits the whole campaign (sealed-test preparation,
# full-epoch gate, both arms, analysis, sealed test, report) to detached tmux.
# Rerunning the same command resumes; it never restarts a run from epoch 17.
#   --dry-run          read-only preflight and GPU discovery; nothing launched
#   --status           gate, per-arm progress, GPU-hours, tmux liveness
#   --parity           arm parity (read-only)
#   --interim-report   descriptive snapshot at the last epoch both arms completed
#   --retry-gate       retry a gate that failed its criteria (after review)
set -euo pipefail
STAGE3_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$STAGE3_PROJECT_DIR"
exec "$STAGE3_PROJECT_DIR/.venv/bin/python" scripts/experiments/stage3_campaign.py "$@"
