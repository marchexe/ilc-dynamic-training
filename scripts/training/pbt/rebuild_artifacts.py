#!/usr/bin/env python3
"""Rebuild canonical PBT artifacts for an existing run without training."""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from training.pbt.reporting import write_canonical_outputs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild PBT report artifacts from an existing manifest.json.")
    parser.add_argument("run", type=Path, help="PBT run directory or manifest.json")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow rebuilding non-completed runs")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.run / "manifest.json" if args.run.is_dir() else args.run
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" and not args.allow_incomplete:
        raise SystemExit(f"run is not completed: {manifest.get('status')}; pass --allow-incomplete to rebuild anyway")
    artifacts = write_canonical_outputs(run_dir, manifest)
    print(artifacts["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
