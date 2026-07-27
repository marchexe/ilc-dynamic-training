#!/usr/bin/env python3
"""Write a compact JSON metrics summary from a PBT manifest."""

import argparse
import json
from pathlib import Path

from reports.plot_mistag_tables import (
    TAG_BACKGROUNDS,
    completed_generations,
    load_manifest,
    mistag_percent,
    worker_for_row,
)


DEFAULT_TAG_TABLES = {
    "c": (0.5, 0.8),
    "b": (0.8, 0.9),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Write metrics_summary.json from a PBT manifest.")
    parser.add_argument("manifest", type=Path, help="PBT manifest.json or run directory")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def generation_events(manifest, generation):
    samples_per_epoch = int(manifest.get("config", {}).get("shared", {}).get("samples_per_epoch", 0))
    epoch = int(generation.get("epoch", generation.get("index", 0)))
    return (epoch + 1) * samples_per_epoch


def metric_value(worker, metric):
    value = (worker.get("metrics") or {}).get(metric)
    return float(value) if value is not None else None


def best_member_for_generation(manifest, generation):
    metric = manifest.get("config", {}).get("pbt", {}).get("metric", "validation_bkg_rejection_score")
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    candidates = []
    for name, worker in generation.get("workers", {}).items():
        value = metric_value(worker, metric)
        if value is not None:
            candidates.append((name, value))
    if not candidates:
        return None, None
    return (max if mode == "max" else min)(candidates, key=lambda item: item[1])


def build_mistag_table(manifest, tag, efficiencies, member="global_best"):
    worker, generation, member_name = worker_for_row(manifest, member)
    metrics = worker.get("metrics") or {}
    backgrounds = TAG_BACKGROUNDS[tag]
    rows = []
    for eff in efficiencies:
        mistags = {}
        for background in backgrounds:
            mistags[f"{background}_bkg_percent"] = mistag_percent(metrics, tag, eff, background)
        rows.append(
            {
                "fixed_efficiency": eff,
                "mistag_percent": mistags,
            }
        )
    return {
        "tag": tag,
        "source": {
            "member_mode": member,
            "generation": generation["index"],
            "member": member_name,
        },
        "backgrounds": list(backgrounds),
        "rows": rows,
    }


def build_summary(manifest, manifest_path):
    config = manifest.get("config", {})
    shared = config.get("shared", {})
    pbt = config.get("pbt", {})
    metric = pbt.get("metric", "validation_bkg_rejection_score")
    best = manifest.get("best") or None

    generations = []
    for generation in completed_generations(manifest):
        member_name, score = best_member_for_generation(manifest, generation)
        worker = generation.get("workers", {}).get(member_name, {}) if member_name else {}
        lr = worker.get("lr") or (manifest.get("members", {}).get(member_name, {}) or {}).get("lr")
        generations.append(
            {
                "generation": generation["index"],
                "epoch": generation.get("epoch"),
                "training_events": generation_events(manifest, generation),
                "best_member": member_name,
                "best_metric_value": score,
                "best_lr": float(lr) if lr is not None else None,
            }
        )

    mistag_tables = []
    if best:
        for tag, efficiencies in DEFAULT_TAG_TABLES.items():
            try:
                mistag_tables.append(build_mistag_table(manifest, tag, efficiencies))
            except Exception as error:
                mistag_tables.append(
                    {
                        "tag": tag,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    global_best = None
    if best:
        global_best = {
            "generation": best.get("generation"),
            "member": best.get("member"),
            "epoch": best.get("epoch"),
            "lr": best.get("lr"),
            "metric": best.get("metric", metric),
            "metric_value": best.get("metric_value"),
            "state_path": best.get("state_path"),
        }

    return {
        "schema_version": 1,
        "experiment": manifest.get("experiment", manifest_path.parent.name),
        "status": manifest.get("status"),
        "metric": {
            "name": metric,
            "mode": pbt.get("mode", "max"),
        },
        "strategy": pbt.get("strategy"),
        "inputs": {
            "dataset": shared.get("dataset"),
            "checkpoint": shared.get("checkpoint"),
            "data_config": shared.get("data_config"),
            "network_config": shared.get("network_config"),
        },
        "global_best": global_best,
        "generations": generations,
        "mistag_percentages": mistag_tables,
    }


def write_summary(manifest_path, output=None):
    manifest, resolved_manifest_path = load_manifest(manifest_path)
    output = Path(output) if output is not None else resolved_manifest_path.with_name("metrics_summary.json")
    summary = build_summary(manifest, resolved_manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main():
    args = parse_args()
    print(write_summary(args.manifest, args.output))


if __name__ == "__main__":
    main()
