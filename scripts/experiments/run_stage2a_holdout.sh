#!/usr/bin/env bash
# A normal invocation submits the entire frozen campaign to remote tmux.
# --dry-run performs read-only preflight/GPU queries and never starts inference.
set -euo pipefail
STAGE2A_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$STAGE2A_PROJECT_DIR"
exec "$STAGE2A_PROJECT_DIR/.venv/bin/python" scripts/experiments/stage2a_campaign.py "$@"
