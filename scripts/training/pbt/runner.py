#!/usr/bin/env python3
"""Run epoch-level Population Based Training on independent Weaver workers."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from training.pbt.artifacts import (
    ensure_run_layout,
    record_controller_lr_change,
    record_initial_evaluation,
    record_skipped_exploit,
    record_tiered_evaluation_round,
    run_contract,
    write_canonical_outputs,
    write_resolved_config,
)
from training.pbt.backend import backend_from_config, finite_metric_ok, format_duration, log_event, run_tiered_evaluation
from training.pbt.config import contract_fingerprint, load_config, validate_inputs
from training.pbt.models.manifest import PBTManifest
from training.runtime import (
    PROJECT_DIR,
    atomic_json,
    git_metadata,
    read_metrics,
    sha256,
    utc_now,
)
from training.pbt.checkpointing import (
    bootstrap_initial_checkpoint,
    checkpoint_paths,
    epoch_for_generation,
    seed_initial_global_best,
)
from training.pbt.controller import apply_actions_to_plan, apply_controller_actions_to_members, run_generation_controller
from training.pbt.metrics import update_generation_health, update_global_best
from training.pbt.planning import (
    add_baseline_guard_rollbacks,
    add_global_best_rollbacks,
    in_burn_in,
    plan_for_strategy,
    raw_metric_ranking,
    should_apply_exploit,
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


def initial_evaluation_enabled(config):
    controller = config["pbt"].get("dynamic_controller") or {}
    return bool(controller.get("evaluate_initial_checkpoint"))


def promote_initial_evaluation_baseline(config, manifest, metric_name, metrics):
    metric_value = float(metrics[metric_name])
    for pbt in (config["pbt"], manifest["config"]["pbt"]):
        pbt.setdefault("configured_baseline_metric_value", pbt.get("baseline_metric_value"))
        pbt["runtime_baseline_metric_value"] = metric_value
        pbt["baseline_metric_value"] = metric_value

    best = manifest.get("best") or {}
    if best.get("generation") != -1 or best.get("member") != "initial_resume":
        return
    best["metric_value"] = metric_value
    best["metrics"] = metrics
    best["updated_at"] = utc_now()
    best["source"] = "initial_evaluation"
    metadata_path = best.get("metadata_path")
    if metadata_path:
        atomic_json(Path(metadata_path), best)


def run_initial_evaluation(config, backend, experiment_dir, manifest, manifest_path, pbt_log_path):
    if not initial_evaluation_enabled(config):
        return False
    existing = manifest.get("initial_evaluation") or {}
    if existing.get("status") == "completed":
        return False

    slot = config["slots"][0]
    command, log_path = backend.initial_evaluation_command_for(config, slot, experiment_dir)
    console_path = log_path.with_suffix(".console.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "running",
        "checkpoint": config["shared"].get("initial_state") or config["shared"]["checkpoint"],
        "metric": config["pbt"]["metric"],
        "slot": backend.slot_label(slot),
        "command": command,
        "log": str(log_path),
        "console_log": str(console_path),
        "started_at": utc_now(),
        "finished_at": None,
        "returncode": None,
        "metrics": None,
    }
    manifest["initial_evaluation"] = record
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    log_event(pbt_log_path, f"started initial_evaluation slot={backend.slot_label(slot)}")

    with console_path.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            check=False,
        )

    metrics = read_metrics(log_path)
    metric_name = config["pbt"]["metric"]
    metric_ok = finite_metric_ok(metrics, metric_name)
    status = "completed" if result.returncode == 0 and metric_ok else "failed"
    record.update(
        status=status,
        returncode=result.returncode,
        metrics=metrics,
        finished_at=utc_now(),
    )
    if metric_ok:
        promote_initial_evaluation_baseline(config, manifest, metric_name, metrics)
        manifest.setdefault("run", {}).setdefault("baseline_evaluation", {})["measured_metric_value"] = metrics[metric_name]
        manifest.setdefault("run", {}).setdefault("baseline_evaluation", {})["measured_source"] = "initial_evaluation"
        manifest.setdefault("baseline_evaluation", {})["measured_metric_value"] = metrics[metric_name]
        manifest.setdefault("baseline_evaluation", {})["measured_source"] = "initial_evaluation"
    record_initial_evaluation(experiment_dir, config, record)
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    log_event(
        pbt_log_path,
        "finished initial_evaluation returncode=%d metric=%s"
        % (
            result.returncode,
            "n/a" if not metric_ok else "%.6g" % metrics[metric_name],
        ),
    )
    if metrics is not None and metrics.get("validation_shutdown_warning"):
        log_event(pbt_log_path, "WARNING: validation_shutdown_warning generation=-1 worker=initial_evaluation")
    if status == "failed":
        raise RuntimeError("initial checkpoint evaluation failed")

    run_initial_evaluation_other_tiers(config, experiment_dir, manifest, manifest_path, pbt_log_path, record)
    return True


def run_initial_evaluation_other_tiers(config, experiment_dir, manifest, manifest_path, pbt_log_path, baseline_record):
    """Optionally evaluate the pretrained baseline on the monitor/full tiers
    too (`tiered_validation.evaluate_initial_checkpoint_all_tiers`), so
    baseline-vs-selected comparisons can be made on every tier, not just
    control. Read-only diagnostics -- failures here never abort the run.
    """
    tiered = config["pbt"].get("tiered_validation") or {}
    if not tiered.get("evaluate_initial_checkpoint_all_tiers"):
        return
    proxy = config["shared"].get("proxy_validation") or {}
    checkpoint = config["shared"].get("initial_state") or config["shared"]["checkpoint"]
    for tier in ("monitor", "full"):
        dataset = proxy.get(f"{tier}_dataset")
        suffix = proxy.get(f"{tier}_suffix")
        if not dataset or not suffix:
            continue
        results = run_tiered_evaluation(
            config, experiment_dir, -1, tier, dataset, suffix, {"initial_resume": checkpoint}, pbt_log_path
        )
        record_tiered_evaluation_round(
            experiment_dir, manifest, {"index": -1}, tier, dataset, suffix, results,
            config["pbt"]["metric"], config["pbt"]["mode"],
        )
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)


def run_scheduled_tiered_evaluations(config, experiment_dir, manifest, generation_record, manifest_path, pbt_log_path):
    """Run monitor/full proxy-tier evaluation for the whole population when
    due, on their own independent cadence from control-tier evaluation and
    from PBT exploit. Evaluates every population member's just-trained
    (pre-exploit) checkpoint, so control/monitor/full all describe the exact
    same model state for that generation -- required for paired
    control<->monitor<->full comparison. Read-only: never feeds back into
    ranking/exploit/controller (see planning.py/controller.py, neither of
    which reads manifest["tiered_evaluations"]).
    """
    tiered = config["pbt"].get("tiered_validation") or {}
    proxy = config["shared"].get("proxy_validation") or {}
    generation_index = generation_record["index"]
    due_tiers = []
    monitor_interval = tiered.get("monitor_interval_generations")
    if monitor_interval and (generation_index + 1) % int(monitor_interval) == 0:
        due_tiers.append("monitor")
    full_interval = tiered.get("full_interval_generations")
    if full_interval and (generation_index + 1) % int(full_interval) == 0:
        due_tiers.append("full")
        # full_holdout is evaluated alongside full: they're a matched pair
        # (headline full-validation number + the independent, non-overlapping
        # slice used for control<->full correlation/ranking diagnostics).
        if proxy.get("full_holdout_dataset") and proxy.get("full_holdout_suffix"):
            due_tiers.append("full_holdout")
    if not due_tiers:
        return

    metric_name = config["pbt"]["metric"]
    mode = config["pbt"]["mode"]

    # Snapshot this generation's already-collected control-tier results too,
    # so they can be paired against monitor/full at the exact same checkpoint
    # without needing to cross-reference metrics.csv separately.
    control_results = {
        name: {"status": worker.get("status"), "metrics": worker.get("metrics")}
        for name, worker in generation_record.get("workers", {}).items()
    }
    record_tiered_evaluation_round(
        experiment_dir, manifest, generation_record, "control",
        config["shared"].get("validation_dataset"), config["shared"].get("validation_suffix"),
        control_results, metric_name, mode,
    )

    epoch = generation_record["epoch"]
    for tier in due_tiers:
        dataset = proxy.get(f"{tier}_dataset")
        suffix = proxy.get(f"{tier}_suffix")
        if not dataset or not suffix:
            log_event(
                pbt_log_path,
                f"tiered_evaluation tier={tier} generation={generation_index} skipped: "
                f"no proxy_validation.{tier}_dataset/{tier}_suffix configured",
            )
            continue
        member_checkpoints = {}
        for name in manifest["members"]:
            state_path, _ = checkpoint_paths(experiment_dir / name, epoch)
            if state_path.is_file():
                member_checkpoints[name] = state_path
        if not member_checkpoints:
            log_event(pbt_log_path, f"tiered_evaluation tier={tier} generation={generation_index} skipped: no member checkpoints found")
            continue
        results = run_tiered_evaluation(
            config, experiment_dir, generation_index, tier, dataset, suffix, member_checkpoints, pbt_log_path
        )
        record_tiered_evaluation_round(experiment_dir, manifest, generation_record, tier, dataset, suffix, results, metric_name, mode)

    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)


def initial_manifest(config, fingerprint, command=None, backend_name=None):
    shared = config["shared"]
    checkpoint = Path(shared["checkpoint"])
    initial_state = Path(shared["initial_state"]) if shared.get("initial_state") else None
    initial_optimizer = Path(shared["initial_optimizer"]) if shared.get("initial_optimizer") else None
    contract = run_contract(config, command, backend_name)
    manifest = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "fingerprint": fingerprint,
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "command": list(command or []),
        "method": config["pbt"].get("strategy", "exploit_mutate"),
        "run": contract,
        "datasets": contract["datasets"],
        "metric_definition": contract["metric"],
        "baseline_evaluation": contract["baseline_evaluation"],
        "next_generation": 0,
        "config": config,
        "checkpoint": {
            "path": str(checkpoint),
            "resolved_path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        },
        "optimizer_checkpoint": None if initial_optimizer is None else {
            "path": str(initial_optimizer),
            "resolved_path": str(initial_optimizer.resolve()),
            "sha256": sha256(initial_optimizer),
            "mode": shared.get("initial_optimizer_mode", "raw"),
            "damping": shared.get("initial_optimizer_damping", 0.1),
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

    backend = backend_from_config(config)
    pbt_log_path = experiment_dir / "logs" / "pbt.log"
    launch_command = [sys.executable, *sys.argv]

    if getattr(backend, "handles_run", False):
        return backend.run(config, experiment_dir, dry_run=args.dry_run)

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False).rstrip())
        if initial_evaluation_enabled(config):
            command, _ = backend.initial_evaluation_command_for(
                config,
                config["slots"][0],
                experiment_dir,
            )
            print(f"[initial_evaluation] {shlex.join(command)}")
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
        manifest.setdefault("command", launch_command)
        manifest.setdefault("method", config["pbt"].get("strategy", "exploit_mutate"))
        manifest.setdefault("run", run_contract(config, launch_command, backend.name))
        manifest.setdefault("datasets", manifest["run"]["datasets"])
        manifest.setdefault("metric_definition", manifest["run"]["metric"])
        manifest.setdefault("baseline_evaluation", manifest["run"].get("baseline_evaluation"))
        manifest["updated_at"] = utc_now()
        ensure_run_layout(experiment_dir)
        write_resolved_config(experiment_dir, config)
    else:
        if experiment_dir.exists():
            raise FileExistsError(f"Experiment already exists: {experiment_dir}")
        experiment_dir.mkdir(parents=True)
        ensure_run_layout(experiment_dir)
        write_resolved_config(experiment_dir, config)
        manifest = initial_manifest(config, fingerprint, launch_command, backend.name)
    run_started_monotonic = time.monotonic()

    for name in manifest["members"]:
        member_dir = experiment_dir / name
        member_dir.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            bootstrap_initial_checkpoint(config, member_dir)
    seed_initial_global_best(config, experiment_dir, manifest)
    run_initial_evaluation(config, backend, experiment_dir, manifest, manifest_path, pbt_log_path)
    write_resolved_config(experiment_dir, config)
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
            run_scheduled_tiered_evaluations(
                config, experiment_dir, manifest, existing, manifest_path, pbt_log_path
            )

            if existing["exploit"] is None:
                ranking, plan = plan_for_strategy(
                    config, existing, manifest["members"], manifest
                )
                existing["ranking"] = ranking
                existing["raw_ranking"] = raw_metric_ranking(config, existing, manifest["members"])
                for skipped_event in existing.get("skipped_exploits") or []:
                    record_skipped_exploit(experiment_dir, existing, skipped_event)
                improved = update_global_best(
                    experiment_dir, manifest, existing, manifest_path
                )
                health = update_generation_health(config, manifest, existing)
                controller_record = run_generation_controller(config, manifest, existing, experiment_dir)
                if controller_record:
                    log_event(
                        pbt_log_path,
                        "controller generation=%d actions=%s"
                        % (
                            existing["index"],
                            ",".join(
                                "%s:%s/%s"
                                % (
                                    name,
                                    action["state_label"],
                                    action["action"],
                                )
                                for name, action in sorted(
                                    existing.get("controller_actions", {}).items()
                                )
                            ),
                        ),
                    )
                early_stop_after = int(config["pbt"].get("early_stop_degraded_generations", 0))
                early_stop_triggered = (
                    early_stop_after > 0
                    and health["consecutive_degraded_generations"] >= early_stop_after
                )
                existing["early_stop_triggered"] = early_stop_triggered
                is_final_generation = generation == int(config["shared"]["generations"]) - 1
                existing["burn_in"] = in_burn_in(config, generation)
                will_exploit = should_apply_exploit(config, generation, is_final_generation, early_stop_triggered)
                if existing["burn_in"]:
                    log_event(
                        pbt_log_path,
                        "burn_in generation=%d (%d/%d) -- observing only, no exploit/controller LR action applied"
                        % (generation, generation + 1, int(config["pbt"].get("burn_in_generations", 0) or 0)),
                    )

                # Fine controller LR nudges: applied to every member's own LR
                # every non-burn-in generation, independent of whether PBT
                # exploit fires this generation -- this is what makes the
                # controller genuinely more frequent than exploit once
                # exploit_interval_generations > 1, not just a config field.
                # Exclude this generation's exploit recipients: their LR is
                # about to come from the donor copy instead.
                if not existing["burn_in"] and not early_stop_triggered:
                    exploit_recipients = {event["recipient"] for event in plan} if will_exploit else set()
                    controller_changes = apply_controller_actions_to_members(
                        config, manifest, existing, exclude_members=exploit_recipients
                    )
                    for member_name, change in controller_changes.items():
                        record_controller_lr_change(experiment_dir, existing, member_name, change)
                    if controller_changes:
                        log_event(
                            pbt_log_path,
                            "controller_lr_change generation=%d members=%s"
                            % (existing["index"], ",".join(sorted(controller_changes))),
                        )

                if will_exploit:
                    plan = apply_actions_to_plan(config, existing, plan)
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
        artifacts = write_canonical_outputs(experiment_dir, manifest)
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
        log_event(pbt_log_path, f"canonical artifacts: {artifacts['report']}")
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
