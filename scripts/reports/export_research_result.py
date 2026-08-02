#!/usr/bin/env python3
"""Export lightweight, Git-trackable research summaries from PBT manifests."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export compact research result JSON/CSV from manifests.")
    parser.add_argument("manifests", nargs="+", type=Path, help="manifest.json files or run directories")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--csv-output", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def completed_generations(manifest):
    return [
        generation
        for generation in manifest.get("generations", [])
        if generation.get("status") == "completed"
    ]


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def worker_metric(generation, member, metric):
    value = ((generation.get("workers") or {}).get(member, {}).get("metrics") or {}).get(metric)
    return _finite(value)


def best_generation_row(manifest, generation):
    config = manifest.get("config") or {}
    pbt = config.get("pbt") or {}
    metric = pbt.get("metric", "validation_working_point_mistag_percent")
    mode = pbt.get("mode", "min")
    candidates = []
    for member, worker in (generation.get("workers") or {}).items():
        value = _finite((worker.get("metrics") or {}).get(metric))
        if value is not None:
            candidates.append({"member": member, "metric_value": value})
    if not candidates:
        return None
    key = lambda row: row["metric_value"]
    return (max if mode == "max" else min)(candidates, key=key)


def checkpoint_baseline_metric(manifest):
    config = manifest.get("config") or {}
    pbt = config.get("pbt") or {}
    value = _finite(pbt.get("baseline_metric_value"))
    if value is not None:
        return value
    best = manifest.get("best") or {}
    if best.get("generation") == -1:
        return _finite(best.get("metric_value"))
    return None


def improvement_from_baseline(mode, baseline, value):
    baseline = _finite(baseline)
    value = _finite(value)
    if baseline is None or value is None:
        return None, None
    absolute = value - baseline if mode == "max" else baseline - value
    relative_percent = None if baseline == 0 else 100.0 * absolute / abs(baseline)
    return absolute, relative_percent


def summarize_manifest(manifest, manifest_path):
    config = manifest.get("config") or {}
    pbt = config.get("pbt") or {}
    shared = config.get("shared") or {}
    metric = pbt.get("metric", "validation_working_point_mistag_percent")
    mode = pbt.get("mode", "min")
    generations = completed_generations(manifest)
    start = best_generation_row(manifest, generations[0]) if generations else None
    final = best_generation_row(manifest, generations[-1]) if generations else None
    global_best = manifest.get("best") or {}
    best_value = _finite(global_best.get("metric_value"))
    baseline = checkpoint_baseline_metric(manifest)
    improvement_abs, improvement_rel = improvement_from_baseline(mode, baseline, best_value)

    return {
        "manifest": str(manifest_path),
        "experiment": manifest.get("experiment", manifest_path.parent.name),
        "status": manifest.get("status"),
        "metric": metric,
        "mode": mode,
        "seed": shared.get("seed"),
        "backend": pbt.get("backend"),
        "strategy": pbt.get("strategy"),
        "confidence_aware_selection": pbt.get("confidence_aware_selection"),
        "selection_uncertainty_sigma": pbt.get("selection_uncertainty_sigma"),
        "anchored_weight_source": pbt.get("anchored_weight_source"),
        "dynamic_controller_mode": (pbt.get("dynamic_controller") or {}).get("mode"),
        "samples_per_epoch": shared.get("samples_per_epoch"),
        "samples_per_epoch_val": shared.get("samples_per_epoch_val"),
        "generations_completed": len(generations),
        "checkpoint_baseline_metric": baseline,
        "start_generation_metric": None if start is None else start["metric_value"],
        "start_generation_member": None if start is None else start["member"],
        "best_metric": best_value,
        "best_member": global_best.get("member"),
        "best_generation": global_best.get("generation"),
        "final_generation_metric": None if final is None else final["metric_value"],
        "final_generation_member": None if final is None else final["member"],
        "improvement_vs_checkpoint_abs": improvement_abs,
        "improvement_vs_checkpoint_percent": improvement_rel,
    }


def _stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"count": 0, "mean": None, "std": None}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"count": len(values), "mean": mean, "std": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"count": len(values), "mean": mean, "std": math.sqrt(variance)}


def build_export(manifest_paths):
    runs = []
    for raw_path in manifest_paths:
        manifest, manifest_path = load_manifest(raw_path)
        runs.append(summarize_manifest(manifest, manifest_path))
    return {
        "schema_version": 1,
        "runs": runs,
        "aggregate": {
            "run_count": len(runs),
            "best_metric": _stats(row["best_metric"] for row in runs),
            "improvement_vs_checkpoint_abs": _stats(row["improvement_vs_checkpoint_abs"] for row in runs),
            "improvement_vs_checkpoint_percent": _stats(row["improvement_vs_checkpoint_percent"] for row in runs),
            "seeds": sorted({row["seed"] for row in runs if row["seed"] is not None}),
        },
    }


def write_csv(path, runs):
    if not runs:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    return path


def write_export(manifest_paths, output, csv_output=None):
    output = Path(output)
    payload = build_export(manifest_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if csv_output:
        write_csv(csv_output, payload["runs"])
    return output


def main():
    args = parse_args()
    print(write_export(args.manifests, args.output, args.csv_output))


if __name__ == "__main__":
    main()
