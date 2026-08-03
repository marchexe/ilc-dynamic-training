#!/usr/bin/env python3
"""CLI for creating disjoint tail proxy-validation parquet files."""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validation.tail_proxy_subsets import build_tail_proxy_subsets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet")
    parser.add_argument("--output-dataset", default=None)
    parser.add_argument("--manifest-output", default="datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json")
    parser.add_argument("--name", default="20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1")
    parser.add_argument("--source-suffix", default="val1000k")
    parser.add_argument("--control-suffix", default="val5k_tail")
    parser.add_argument("--monitor-suffix", default="val50k_tail")
    parser.add_argument("--full-holdout-suffix", default="val_holdout")
    parser.add_argument("--control-rows-per-class", type=int, default=5000)
    parser.add_argument("--monitor-rows-per-class", type=int, default=50000)
    parser.add_argument("--no-full-holdout", action="store_true", help="Skip building the independent full_holdout tier")
    parser.add_argument("--compression", default="lz4")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_tail_proxy_subsets(
        dataset=args.dataset,
        output_dataset=args.output_dataset,
        manifest_output=args.manifest_output,
        name=args.name,
        source_suffix=args.source_suffix,
        control_suffix=args.control_suffix,
        monitor_suffix=args.monitor_suffix,
        full_holdout_suffix=args.full_holdout_suffix,
        control_rows_per_class=args.control_rows_per_class,
        monitor_rows_per_class=args.monitor_rows_per_class,
        build_full_holdout=not args.no_full_holdout,
        compression=args.compression,
        force=args.force,
    )
    output = {
        "manifest": args.manifest_output,
        "control_suffix": manifest["levels"]["control"]["suffix"],
        "control_rows": manifest["levels"]["control"]["rows_total"],
        "monitor_suffix": manifest["levels"]["monitor"]["suffix"],
        "monitor_rows": manifest["levels"]["monitor"]["rows_total"],
    }
    if "full_holdout" in manifest["levels"]:
        output["full_holdout_suffix"] = manifest["levels"]["full_holdout"]["suffix"]
        output["full_holdout_rows"] = manifest["levels"]["full_holdout"]["rows_total"]
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
