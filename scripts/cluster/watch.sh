#!/usr/bin/env bash
# Live GPU cluster view. Wraps check.sh with watch so the terminal is updated
# in-place instead of printing a new report every time.
#
# Usage:
#   scripts/cluster/watch.sh
#   scripts/cluster/watch.sh 2
#   ML_PFA_CLUSTER_NODES="node-a node-b" scripts/cluster/watch.sh 1

set -euo pipefail

INTERVAL="${1:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="$SCRIPT_DIR/check.sh"

if ! command -v watch >/dev/null 2>&1; then
    echo "error: 'watch' command is not available"
    echo "fallback: run this manually:"
    echo "  while true; do clear; bash $CHECK_SCRIPT; sleep $INTERVAL; done"
    exit 1
fi

# Nodes: ML_PFA_CLUSTER_NODES, else the local site configuration when present.
export ML_PFA_CLUSTER_NODES="${ML_PFA_CLUSTER_NODES:-$(python3 "$SCRIPT_DIR/../experiments/site_config.py" nodes 2>/dev/null || true)}"
export ML_PFA_ALLOW_PASSWORD="${ML_PFA_ALLOW_PASSWORD:-0}"
export ML_PFA_DEBUG="${ML_PFA_DEBUG:-0}"
export ML_PFA_SSH_CONFIG="${ML_PFA_SSH_CONFIG:-}"

watch -n "$INTERVAL" -c "bash '$CHECK_SCRIPT'"
