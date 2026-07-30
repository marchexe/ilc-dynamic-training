#!/usr/bin/env python3
"""Run epoch-level Population Based Training on independent Weaver workers."""

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import yaml

from training.pbt.backend import backend_from_config, format_duration, log_event
from training.pbt.config import contract_fingerprint, load_config, validate_inputs
from training.pbt.models.manifest import PBTManifest
from training.runtime import (
    PROJECT_DIR,
    atomic_json,
    git_metadata,
    sha256,
    utc_now,
)
from training.pbt.checkpointing import bootstrap_initial_checkpoint, epoch_for_generation
from training.pbt.metrics import update_generation_health, update_global_best
from training.pbt.planning import (
    add_baseline_guard_rollbacks,
    add_global_best_rollbacks,
    plan_for_strategy,
    strategy_uses_population_rollbacks,
)
from training.pbt.transitions import apply_exploit


DEFAULT_CONFIG = PROJECT_DIR / "configs/experiments/pbt_smoke.yaml"
MANIFEST_NAME = "manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-name")
    parser.add_argument("--gpus", help="Comma-separated GPU slots, e.g. 0,2")
    parser.add_argument(
        "--slots",
        help=(
            "Comma-separated host:gpu slots for multi-node runs. "
            "Every host must see the project .venv at the same path, "
            "e.g. iutgpu01:6,iutgpu05:4"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two members, two generations and small train/validation budgets",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def initial_manifest(config, fingerprint):
    shared = config["shared"]
    checkpoint = Path(shared["checkpoint"])
    initial_state = Path(shared["initial_state"]) if shared.get("initial_state") else None
    initial_optimizer = Path(shared["initial_optimizer"]) if shared.get("initial_optimizer") else None
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "fingerprint": fingerprint,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "next_generation": 0,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint),
            "resolved_path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        },
        "initial_resume": None if initial_state is None else {
            "epoch": int(shared["initial_epoch"]),
            "state_path": str(initial_state),
            "state_resolved_path": str(initial_state.resolve()),
            "state_sha256": sha256(initial_state),
            "optimizer_path": str(initial_optimizer),
            "optimizer_resolved_path": str(initial_optimizer.resolve()),
            "optimizer_sha256": sha256(initial_optimizer),
            "optimizer_mode": shared.get("initial_optimizer_mode", "raw"),
            "optimizer_damping": shared.get("initial_optimizer_damping", 0.1),
        },
        "git": git_metadata(),
        "members": {
            member["name"]: {
                "name": member["name"],
                "lr": float(member["start_lr"]),
                "parent": None,
            }
            for member in config["population"]
        },
        "best": None,
        "generations": [],
    }
    return PBTManifest.parse_payload(manifest).to_runtime_dict()


def run(args):
    config = load_config(args)
    validate_inputs(config)
    fingerprint = contract_fingerprint(config)
    experiment_dir = Path(config["output_root"]) / config["experiment_name"]
    manifest_path = experiment_dir / MANIFEST_NAME
    pbt_log_path = manifest_path.with_name("pbt.log")

    backend = backend_from_config(config)

    if getattr(backend, "handles_run", False):
        return backend.run(config, experiment_dir, dry_run=args.dry_run)

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False).rstrip())
        for index, member_config in enumerate(config["population"]):
            member = {"name": member_config["name"], "lr": float(member_config["start_lr"])}
            command, _, _ = backend.command_for(
                config,
                member,
                config["slots"][index % len(config["slots"])],
                experiment_dir / member["name"],
                0,
            )
            print(f"[{member['name']}] {shlex.join(command)}")
        return 0

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        manifest = PBTManifest.parse_payload(json.loads(manifest_path.read_text())).to_runtime_dict()
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resolved PBT contract differs from the saved manifest")
        if manifest["status"] == "completed":
            print(f"already completed: {experiment_dir}")
            return 0
        manifest["status"] = "running"
        manifest.pop("failure", None)
        manifest["updated_at"] = utc_now()
    else:
        if experiment_dir.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_dir}")
        experiment_dir.mkdir(parents=True)
        manifest = initial_manifest(config, fingerprint)
    run_started_monotonic = time.monotonic()

    for name in manifest["members"]:
        member_dir = experiment_dir / name
        member_dir.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            bootstrap_initial_checkpoint(config, member_dir)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    atomic_json(manifest_path, manifest)
    log_event(
        pbt_log_path,
        f"run started experiment={config['experiment_name']} "
        f"backend={backend.name} slots={','.join(backend.slot_label(slot) for slot in config['slots'])} "
        f"members={len(manifest['members'])} "
        f"batch_size={config['shared']['batch_size']} metric={config['pbt']['metric']}",
    )

    try:
        for generation in range(
            int(manifest["next_generation"]), int(config["shared"]["generations"])
        ):
            existing = next(
                (item for item in manifest["generations"] if item["index"] == generation),
                None,
            )
            if existing is None:
                existing = {
                    "index": generation,
                    "epoch": epoch_for_generation(config, generation),
                    "seed": int(config["shared"]["seed"]) + generation,
                    "status": "training",
                    "workers": {
                        name: {"status": "pending"} for name in manifest["members"]
                    },
                    "ranking": None,
                    "exploit": None,
                    "started_at": utc_now(),
                }
                manifest["generations"].append(existing)
                atomic_json(manifest_path, manifest)
            existing["status"] = "training"
            pending = [
                name
                for name, record in existing["workers"].items()
                if record["status"] != "completed"
            ]
            backend.run_generation(
                config,
                experiment_dir,
                manifest,
                existing,
                pending,
                manifest_path,
            )

            if existing["exploit"] is None:
                ranking, plan = plan_for_strategy(
                    config, existing, manifest["members"], manifest
                )
                existing["ranking"] = ranking
                improved = update_global_best(
                    experiment_dir, manifest, existing, manifest_path
                )
                health = update_generation_health(config, manifest, existing)
                early_stop_after = int(config["pbt"].get("early_stop_degraded_generations", 0))
                early_stop_triggered = (
                    early_stop_after > 0
                    and health["consecutive_degraded_generations"] >= early_stop_after
                )
                existing["early_stop_triggered"] = early_stop_triggered
                will_exploit = (
                    generation != int(config["shared"]["generations"]) - 1
                    and not early_stop_triggered
                )
                if will_exploit:
                    plan = add_baseline_guard_rollbacks(
                        config, manifest, existing, manifest["members"], plan
                    )
                    if strategy_uses_population_rollbacks(config):
                        plan = add_global_best_rollbacks(
                            config, manifest, existing, manifest["members"], plan
                        )
                else:
                    existing.pop("baseline_guard", None)
                existing["exploit"] = plan if will_exploit else []
                existing["status"] = "exploiting"
                best = manifest.get("best") or {}
                log_event(
                    pbt_log_path,
                    "health generation=%d current_best=%s %.6g global_best=%s %.6g "
                    "delta=%s degraded=%s consecutive=%d improved=%s lrs=%s"
                    % (
                        existing["index"],
                        health["current_best_member"],
                        health["current_best_metric"],
                        best.get("member"),
                        best.get("metric_value", float("nan")),
                        (
                            "n/a"
                            if health["relative_to_global_best"] is None
                            else f"{health['relative_to_global_best']:+.3%}"
                        ),
                        health["status"],
                        health["consecutive_degraded_generations"],
                        improved,
                        ",".join(
                            f"{name}:{lr:.3g}"
                            for name, lr in sorted(health["member_lrs"].items())
                        ),
                    ),
                )
                guard = existing.get("baseline_guard")
                if guard:
                    log_event(
                        pbt_log_path,
                        "baseline guard generation=%d action=%s rollbacks=%s"
                        % (
                            existing["index"],
                            guard.get("action"),
                            ",".join(
                                "%s:%.3g->%.3g metric=%.6g baseline=%.6g"
                                % (
                                    event["recipient"],
                                    event["recipient_lr"],
                                    event["new_lr"],
                                    event["metric_value"],
                                    event["baseline_metric"],
                                )
                                for event in guard.get("events", [])
                            ),
                        ),
                    )
                atomic_json(manifest_path, manifest)

            apply_exploit(experiment_dir, manifest, existing, manifest_path)
            existing["status"] = "completed"
            existing["finished_at"] = utc_now()
            manifest["next_generation"] = generation + 1
            manifest["updated_at"] = utc_now()
            if existing.get("early_stop_triggered"):
                manifest["early_stop"] = {
                    "generation": generation,
                    "reason": "consecutive_degraded_generations",
                    "consecutive_degraded_generations": existing["health"]["consecutive_degraded_generations"],
                    "threshold": int(config["pbt"].get("early_stop_degraded_generations", 0)),
                }
                atomic_json(manifest_path, manifest)
                log_event(
                    pbt_log_path,
                    "early stop generation=%d reason=consecutive_degraded_generations consecutive=%d threshold=%d"
                    % (
                        generation,
                        existing["health"]["consecutive_degraded_generations"],
                        int(config["pbt"].get("early_stop_degraded_generations", 0)),
                    ),
                )
                break
            atomic_json(manifest_path, manifest)

        manifest["status"] = "completed"
        manifest.pop("failure", None)
        manifest["finished_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        try:
            from reports.write_metrics_summary import write_summary

            summary_path = write_summary(manifest_path)
            manifest["metrics_summary"] = str(summary_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"metrics summary: {summary_path}")
        except Exception as summary_error:
            log_event(pbt_log_path, f"warning: failed to write metrics summary: {summary_error}")
        try:
            from reports.plot_pbt_summary import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["training_diagnostics_plot"] = str(plot_path)
            manifest.pop("pbt_objective_diagnostics_plot", None)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"training diagnostics plot: {plot_path}")
        except Exception as plot_error:
            log_event(pbt_log_path, f"warning: failed to create training diagnostics plot: {plot_error}")
        try:
            from reports.plot_physics_performance import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["physics_performance_plot"] = str(plot_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"physics performance plot: {plot_path}")
        except Exception as plot_error:
            log_event(pbt_log_path, f"warning: failed to create physics performance plot: {plot_error}")
        try:
            from reports.plot_background_rejection_curves import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["background_rejection_curves_plot"] = str(plot_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"background rejection curves plot: {plot_path}")
        except Exception as plot_error:
            log_event(
                pbt_log_path,
                f"warning: failed to create background rejection curves plot: {plot_error}",
            )
        try:
            from reports.plot_background_efficiency_curves import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["background_efficiency_curves_plot"] = str(plot_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"background efficiency curves plot: {plot_path}")
        except Exception as plot_error:
            log_event(
                pbt_log_path,
                f"warning: failed to create background efficiency curves plot: {plot_error}",
            )
        try:
            from reports.plot_fixed_b_efficiency import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["btag_background_efficiency_plot"] = str(plot_path)
            manifest.pop("working_point_mistag_history_plot", None)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"b-tag background efficiency plot: {plot_path}")
        except Exception as plot_error:
            log_event(
                pbt_log_path,
                f"warning: failed to create b-tag background efficiency plot: {plot_error}",
            )
        try:
            from reports.plot_selection_timeline import plot_manifest

            plot_path = plot_manifest(manifest_path)
            manifest["selection_timeline_plot"] = str(plot_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"selection timeline plot: {plot_path}")
        except Exception as plot_error:
            log_event(
                pbt_log_path,
                f"warning: failed to create selection timeline plot: {plot_error}",
            )
        try:
            from reports.plot_mistag_tables import collect_tables, write_csv

            table_specs = {"c": (0.5, 0.8), "b": (0.8, 0.9)}
            for tag, efficiencies in table_specs.items():
                tables = collect_tables(
                    [(manifest.get("experiment", experiment_dir.name), manifest_path)],
                    tag=tag,
                    efficiencies=efficiencies,
                    member="best_physics",
                    manifests={manifest_path: manifest},
                )
                csv_path = experiment_dir / "plots" / "report" / f"{tag}tag_mistag_tables.csv"
                write_csv(csv_path, tables, tag)
                manifest.pop(f"{tag}tag_mistag_table_plot", None)
                manifest[f"{tag}tag_mistag_table_csv"] = str(csv_path)
                log_event(pbt_log_path, f"{tag}-tag mistag CSV: {csv_path}")
            for key in (
                "btag_rejection_evolution_plot",
                "ctag_rejection_evolution_plot",
                "btag_mistag_evolution_plot",
                "ctag_mistag_evolution_plot",
                "working_point_mistag_history_plot",
                "global_best_all_pair_rejection_curves_plot",
                "pbt_lr_response_plot",
            ):
                manifest.pop(key, None)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
        except Exception as plot_error:
            log_event(
                pbt_log_path,
                f"warning: failed to create fixed-efficiency mistag CSV tables: {plot_error}",
            )
        try:
            from reports.write_metrics_summary import write_summary

            summary_path = write_summary(manifest_path)
            manifest["metrics_summary"] = str(summary_path)
            manifest["updated_at"] = utc_now()
            atomic_json(manifest_path, manifest)
            log_event(pbt_log_path, f"metrics summary updated with plots: {summary_path}")
        except Exception as summary_error:
            log_event(pbt_log_path, f"warning: failed to refresh metrics summary: {summary_error}")
        log_event(
            pbt_log_path,
            f"run completed experiment={config['experiment_name']} "
            f"elapsed={format_duration(time.monotonic() - run_started_monotonic)} "
            f"directory={experiment_dir} "
            f"recommended_checkpoint={(manifest.get('best') or {}).get('state_path')}",
        )
        return 0
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        log_event(
            pbt_log_path,
            f"run {manifest['status']} experiment={config['experiment_name']} "
            f"elapsed={format_duration(time.monotonic() - run_started_monotonic)} "
            f"error={type(error).__name__}: {error}",
        )
        raise


def main():
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
