#!/usr/bin/env python3
"""Compare internal PBT scores with fixed-WP physics metrics across manifests."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.write_metrics_summary import showcase_metrics, worker_lr


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze metric ranking agreement in PBT manifests.")
    parser.add_argument("manifests", nargs="+", type=Path, help="manifest.json files or run directories")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval/metric_alignment"))
    return parser.parse_args()


def resolve_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def completed_generations(manifest):
    return [generation for generation in manifest.get("generations", []) if generation.get("status") == "completed"]


def finite(value):
    return value is not None and math.isfinite(float(value))


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def worker_rows(manifest, manifest_path):
    experiment = manifest.get("experiment", manifest_path.parent.name)
    rows = []
    for generation in completed_generations(manifest):
        for member, worker in sorted((generation.get("workers") or {}).items()):
            metrics = worker.get("metrics") or {}
            old_score = metrics.get("validation_bkg_rejection_score")
            mistag = showcase_metrics(metrics).get("average_mistag_percent")
            loss = metrics.get("validation_loss")
            auc = metrics.get("validation_auc")
            accuracy = metrics.get("validation_accuracy")
            rows.append({
                "experiment": experiment,
                "generation": generation.get("index"),
                "member": member,
                "lr": worker_lr(worker),
                "old_log_rejection_score": old_score,
                "fixed_wp_mistag_percent": mistag,
                "physics_quality_negative_mistag": None if mistag is None else -mistag,
                "validation_loss": loss,
                "validation_auc": auc,
                "validation_accuracy": accuracy,
            })
    return rows


def best_member(rows, metric, mode):
    candidates = [row for row in rows if finite(row.get(metric))]
    if not candidates:
        return None
    key = lambda row: float(row[metric])
    return (min if mode == "min" else max)(candidates, key=key)


def analyze_manifest(manifest, manifest_path):
    rows = worker_rows(manifest, manifest_path)
    paired = [row for row in rows if finite(row.get("old_log_rejection_score")) and finite(row.get("physics_quality_negative_mistag"))]
    old = [float(row["old_log_rejection_score"]) for row in paired]
    physics = [float(row["physics_quality_negative_mistag"]) for row in paired]
    loss_paired = [row for row in rows if finite(row.get("validation_loss")) and finite(row.get("physics_quality_negative_mistag"))]
    loss = [float(row["validation_loss"]) for row in loss_paired]
    loss_phys = [float(row["physics_quality_negative_mistag"]) for row in loss_paired]

    agreements = []
    for generation in completed_generations(manifest):
        gen_rows = [row for row in rows if row["generation"] == generation.get("index")]
        old_best = best_member(gen_rows, "old_log_rejection_score", "max")
        physics_best = best_member(gen_rows, "fixed_wp_mistag_percent", "min")
        if old_best and physics_best:
            agreements.append({
                "generation": generation.get("index"),
                "old_score_best_member": old_best["member"],
                "physics_best_member": physics_best["member"],
                "agreement": old_best["member"] == physics_best["member"],
                "old_score_best_value": old_best["old_log_rejection_score"],
                "physics_best_mistag_percent": physics_best["fixed_wp_mistag_percent"],
            })

    return {
        "experiment": manifest.get("experiment", manifest_path.parent.name),
        "manifest": str(manifest_path),
        "n_workers": len(rows),
        "n_paired_old_vs_fixed_wp": len(paired),
        "pearson_old_score_vs_negative_mistag": pearson(old, physics),
        "spearman_old_score_vs_negative_mistag": spearman(old, physics),
        "pearson_loss_vs_negative_mistag": pearson(loss, loss_phys),
        "spearman_loss_vs_negative_mistag": spearman(loss, loss_phys),
        "selection_agreement_fraction": None if not agreements else sum(row["agreement"] for row in agreements) / len(agreements),
        "selection_agreement_by_generation": agreements,
        "rows": rows,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for manifest_arg in args.manifests:
        manifest_path = resolve_manifest(manifest_arg)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = analyze_manifest(manifest, manifest_path)
        summaries.append({key: value for key, value in result.items() if key != "rows"})
        all_rows.extend(result["rows"])

    json_path = args.output_dir / "metric_alignment_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "metric_alignment_rows.csv"
    fieldnames = [
        "experiment", "generation", "member", "lr", "old_log_rejection_score",
        "fixed_wp_mistag_percent", "physics_quality_negative_mistag",
        "validation_loss", "validation_auc", "validation_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
