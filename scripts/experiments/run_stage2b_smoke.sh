#!/usr/bin/env bash
# Stage 2B one-command launcher: verifies the revision and frozen inputs, then
# submits dataset preparation (only if needed) + the bounded train8m GPU smoke,
# verification and the readiness report to detached remote tmux.
#   --dry-run       read-only preflight and GPU discovery; nothing is converted or trained
#   --status        show dataset-conversion vs GPU-smoke progress
#   --prepare-only  convert/verify train8m without the GPU smoke
set -euo pipefail
STAGE2B_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$STAGE2B_PROJECT_DIR"
exec "$STAGE2B_PROJECT_DIR/.venv/bin/python" scripts/experiments/stage2b_campaign.py "$@"
