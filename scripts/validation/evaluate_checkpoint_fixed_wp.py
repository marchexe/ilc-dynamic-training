#!/usr/bin/env python3
"""Evaluate one checkpoint on SGV parquet validation data and write fixed-WP reports."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reports.plot_background_efficiency_curves import plot_manifest as plot_background_efficiency
from reports.plot_mistag_tables import collect_tables, write_csv
from reports.plot_physics_performance import plot_manifest as plot_physics_performance
from reports.write_metrics_summary import write_summary
from training.runtime import data_paths, git_metadata, normalize_data_extension, read_metrics, utc_now, weaver_executable


DEFAULT_DATASET = Path("datasets/20250218_ilc_nnqq_sgvnew_parquet")
DEFAULT_DATA_CONFIG = Path("/data/suehara/part/data/ilc_nnqq_sgvnew_3cat_cut.217feb3dc9ed1ee6978db1c04604f81b.auto.yaml")
DEFAULT_NETWORK_CONFIG = Path("networks/pretrained_sgv_particle_transformer.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and generate fixed-WP reports.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--name", required=True, help="Output run name under --output-root.")
    parser.add_argument("--output-root", type=Path, default=Path("runs/eval"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data-extension", default="parquet")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--network-config", type=Path, default=DEFAULT_NETWORK_CONFIG)
    parser.add_argument("--samples-per-epoch-val", type=int, default=150000)
    parser.add_argument("--validation-suffix", default="val50k")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fetch-step", default="0.01")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--amp-dtype", default="fp16")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def build_test_command(args, log_path):
    dataset = project_path(args.dataset)
    data_extension = normalize_data_extension(args.data_extension)
    val_paths = [
        value.split(":", 1)[1]
        for value in data_paths(
            dataset,
            data_extension,
            validation_suffix=args.validation_suffix,
        )["val"]
    ]
    return [
        str(weaver_executable()),
        "--run-mode",
        "test",
        "--data-test",
        *val_paths,
        "--data-config",
        str(project_path(args.data_config)),
        "--network-config",
        str(project_path(args.network_config)),
        "--model-prefix",
        str(project_path(args.checkpoint)),
        "--log-file",
        str(log_path),
        "--batch-size",
        str(args.batch_size),
        "--use-amp",
        "--amp-dtype",
        str(args.amp_dtype),
        "--num-workers",
        str(args.num_workers),
        "--fetch-step",
        str(args.fetch_step),
        "--gpus",
        str(args.gpu),
        "--predict-gpus",
        str(args.gpu),
    ]


def build_manifest(args, metrics, log_path, command):
    now = utc_now()
    checkpoint = project_path(args.checkpoint)
    run_dir = project_path(args.output_root) / args.name
    return {
        "schema_version": 1,
        "experiment": args.name,
        "status": "completed",
        "created_at": now,
        "updated_at": now,
        "finished_at": now,
        "git": git_metadata(),
        "config": {
            "shared": {
                "dataset": str(project_path(args.dataset)),
                "data_extension": normalize_data_extension(args.data_extension),
                "validation_suffix": args.validation_suffix,
                "checkpoint": str(checkpoint),
                "data_config": str(project_path(args.data_config)),
                "network_config": str(project_path(args.network_config)),
                "training_controller": None,
                "seed": 12345,
                "generations": 1,
                "weaver_epochs_per_generation": 1,
                "samples_per_epoch": 0,
                "samples_per_epoch_val": int(args.samples_per_epoch_val),
                "batch_size": int(args.batch_size),
                "optimizer": "none",
                "lr_scheduler": "none",
                "num_workers": int(args.num_workers),
                "fetch_step": args.fetch_step,
                "use_amp": True,
                "amp_dtype": args.amp_dtype,
                "no_remake_weights": True,
            },
            "pbt": {
                "strategy": "checkpoint_eval",
                "metric": "validation_working_point_mistag_percent",
                "mode": "min",
            },
        },
        "members": {"checkpoint": {"name": "checkpoint", "lr": None, "parent": None}},
        "best": {
            "generation": 0,
            "epoch": 0,
            "member": "checkpoint",
            "lr": None,
            "metric": "validation_working_point_mistag_percent",
            "metric_value": metrics.get("validation_working_point_mistag_percent"),
            "metrics": metrics,
            "state_path": str(checkpoint),
            "updated_at": now,
        },
        "generations": [
            {
                "index": 0,
                "epoch": 0,
                "status": "completed",
                "started_at": now,
                "finished_at": now,
                "workers": {
                    "checkpoint": {
                        "status": "completed",
                        "lr": None,
                        "returncode": 0,
                        "command": command,
                        "log_path": str(log_path),
                        "state_path": str(checkpoint),
                        "metrics": metrics,
                    }
                },
            }
        ],
    }


def write_reports(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plot_path = plot_physics_performance(manifest_path)
    manifest["physics_performance_plot"] = str(plot_path)
    plot_path = plot_background_efficiency(manifest_path)
    manifest["background_efficiency_curves_plot"] = str(plot_path)
    for tag, efficiencies in {"c": (0.5, 0.8), "b": (0.8, 0.9)}.items():
        tables = collect_tables([(manifest.get("experiment", manifest_path.parent.name), manifest_path)], tag, efficiencies, "best_physics")
        csv_path = manifest_path.parent / "plots" / "report" / f"{tag}tag_mistag_tables.csv"
        write_csv(csv_path, tables, tag)
        manifest[f"{tag}tag_mistag_table_csv"] = str(csv_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return write_summary(manifest_path)


def main():
    args = parse_args()
    run_dir = project_path(args.output_root) / args.name
    log_path = run_dir / "checkpoint_eval.log"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"manifest already exists, use --force to overwrite: {manifest_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_test_command(args, log_path)
    result = subprocess.run(command, cwd=Path.cwd(), text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    metrics = read_metrics(log_path)
    if not metrics or not metrics.get("validation_bkg_rejection_at_eff"):
        raise RuntimeError(f"failed to parse fixed-WP metrics from {log_path}")
    manifest = build_manifest(args, metrics, log_path, command)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = write_reports(manifest_path)
    print(manifest_path)
    print(summary_path)


if __name__ == "__main__":
    main()
