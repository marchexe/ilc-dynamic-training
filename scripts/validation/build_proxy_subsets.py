#!/usr/bin/env python3
"""CLI for creating deterministic proxy-validation parquet subsets."""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validation.proxy_subsets import DEFAULT_PROXY_NAME, build_proxy_subsets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/20250218_ilc_nnqq_sgvnew_parquet")
    parser.add_argument("--output-root", default="datasets/proxy_validation")
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--name", default=DEFAULT_PROXY_NAME)
    parser.add_argument("--data-extension", default="parquet")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--control-rows-per-class", type=int, default=5000)
    parser.add_argument("--monitor-rows-per-class", type=int, default=10000)
    parser.add_argument("--compression", default="lz4")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_output = args.manifest_output or f"datasets/manifests/{args.name}.json"
    manifest = build_proxy_subsets(
        dataset=args.dataset,
        output_root=args.output_root,
        manifest_output=manifest_output,
        name=args.name,
        data_extension=args.data_extension,
        seed=args.seed,
        control_rows_per_class=args.control_rows_per_class,
        monitor_rows_per_class=args.monitor_rows_per_class,
        compression=args.compression,
        force=args.force,
    )
    print(json.dumps({
        "manifest": manifest_output,
        "control_dataset": manifest["levels"]["control"]["dataset"],
        "monitor_dataset": manifest["levels"]["monitor"]["dataset"],
        "full_dataset": manifest["levels"]["full"]["dataset"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
