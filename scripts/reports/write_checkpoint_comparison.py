#!/usr/bin/env python3
"""Write a compact fixed-WP comparison table for completed runs/evaluations."""

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.write_metrics_summary import showcase_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compare fixed-WP metrics from metrics_summary.json files.")
    parser.add_argument("summaries", nargs="+", type=Path, help="metrics_summary.json files or run directories")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/fixed_wp_checkpoint_comparison.csv"))
    return parser.parse_args()


def resolve_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / "metrics_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def row_from_summary(path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    global_best = summary.get("global_best") or {}
    details = global_best.get("details") or {}
    core = details.get("core_metrics") or {}
    showcase = details.get("showcase_metrics") or {}
    logic = summary.get("training_logic") or {}
    return {
        "experiment": summary.get("experiment", path.parent.name),
        "selection_metric": (summary.get("metric") or {}).get("name"),
        "best_generation": global_best.get("generation"),
        "best_member": global_best.get("member"),
        "lr": global_best.get("lr"),
        "avg_fixed_wp_mistag_percent": showcase.get("average_mistag_percent"),
        "validation_accuracy": core.get("accuracy"),
        "validation_auc": core.get("auc"),
        "validation_loss": core.get("loss"),
        "training_start_mistag_percent": (logic.get("start") or {}).get("avg_mistag_percent"),
        "training_best_mistag_percent": (logic.get("best") or {}).get("avg_mistag_percent"),
        "training_final_mistag_percent": (logic.get("final") or {}).get("avg_mistag_percent"),
        "delta_start_to_best_percent": logic.get("delta_start_to_best_percent"),
        "delta_best_to_final_percent": logic.get("delta_best_to_final_percent"),
        "exact_stat_uncertainty_available": logic.get("exact_stat_uncertainty_available"),
        "summary_path": str(path),
    }


def main():
    args = parse_args()
    rows = [row_from_summary(resolve_summary(path)) for path in args.summaries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
