#!/usr/bin/env python3
"""Minimal proxy audit: evaluate several distinct checkpoints on both
control_proxy_50k (val50k_tail) and full_validation (val_holdout), and
report how well the fixed 50k proxy reproduces full-validation metrics and
checkpoint ranking.

Reuses existing inference/eval machinery instead of reinventing it:
`run_tiered_evaluation` (training.pbt.execution.backend) already does
exactly what this needs -- dispatch one checkpoint per free GPU slot
(local or SSH-remote via `wrap_remote_command`), run Weaver in
`--run-mode test`, parse metrics with `read_metrics`, and never let one
checkpoint's failure abort the others. Every checkpoint here is passed to
it as if it were a population "member" for one synthetic generation.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import yaml

from training.pbt.execution.backend import finite_metric_ok, format_duration, log_event, run_tiered_evaluation
from training.runtime import git_metadata, project_path, sha256, utc_now
from research.proxy_sanity_check import run_sanity_check
from research.proxy_statistics import full_summary

CHECKPOINT_EPOCH_RE = re.compile(r"net_epoch-(\d+)_state\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/nightly_proxy_audit.yaml"))
    parser.add_argument("--run-id", default=None, help="Defaults to a UTC timestamp")
    parser.add_argument("--output-root", type=Path, default=None, help="Overrides the config's output_root")
    return parser.parse_args()


def load_audit_config(path):
    payload = yaml.safe_load(project_path(path).read_text())
    return payload


def build_weaver_config(audit_config):
    """Shapes audit_config into the {"shared": ..., "slots": ...} dict
    make_tiered_evaluation_command/_test_mode_command expect -- the exact
    fields the live PBT tiered-evaluation command builder reads."""
    return {
        "shared": {
            "data_config": str(project_path(audit_config["data_config"])),
            "network_config": str(project_path(audit_config["network_config"])),
            "data_extension": audit_config.get("data_extension", "parquet"),
            "batch_size": int(audit_config.get("batch_size", 1024)),
            "num_workers": int(audit_config.get("num_workers", 1)),
            "fetch_step": audit_config.get("fetch_step", "0.01"),
            "use_amp": True,
            "amp_dtype": audit_config.get("amp_dtype", "fp16"),
        },
        "slots": audit_config["slots"],
    }


def dedupe_checkpoints(checkpoints):
    """Hash every checkpoint; distinct entries keep their id, duplicates
    are recorded (never silently dropped) and excluded from evaluation --
    evaluating the same weights twice under two ids would double-count a
    single data point in the correlation/ranking analysis."""
    seen_by_hash = {}
    distinct = []
    duplicates = []
    for entry in checkpoints:
        path = project_path(entry["path"])
        if not path.is_file():
            duplicates.append({**entry, "sha256": None, "duplicate_of": None, "reason": "checkpoint_file_missing"})
            continue
        digest = sha256(path)
        if digest in seen_by_hash:
            duplicates.append({**entry, "sha256": digest, "duplicate_of": seen_by_hash[digest], "reason": "identical_sha256"})
            continue
        seen_by_hash[digest] = entry["id"]
        distinct.append({**entry, "sha256": digest, "path": str(path)})
    return distinct, duplicates


def provenance_for(entry):
    """Best-effort checkpoint/training provenance: epoch, generation, lr,
    and the metric value PBT itself recorded for this checkpoint, when
    available. Never fabricated -- fields stay None (recorded, not
    silently omitted) when the source data doesn't have them."""
    provenance = entry.get("provenance")
    result = {"epoch": None, "generation": None, "lr": None, "pbt_recorded_metric_value": None, "member": entry.get("provenance", {}).get("member") if provenance else None}
    checkpoint_path = Path(entry["path"])
    epoch_match = CHECKPOINT_EPOCH_RE.search(checkpoint_path.name)
    if epoch_match:
        result["epoch"] = int(epoch_match.group(1))

    if provenance is None:
        return result

    if provenance["type"] == "global_best_metadata":
        metadata_path = project_path(provenance["path"])
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            result["epoch"] = metadata.get("epoch", result["epoch"])
            result["generation"] = metadata.get("generation")
            result["lr"] = metadata.get("lr")
            result["pbt_recorded_metric_value"] = metadata.get("metric_value")
            result["member"] = metadata.get("member")
        return result

    if provenance["type"] == "manifest_member":
        manifest_path = project_path(provenance["path"])
        member = provenance["member"]
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            member_record = (manifest.get("members") or {}).get(member) or {}
            result["lr"] = member_record.get("lr")
            for generation_record in manifest.get("generations", []):
                if generation_record.get("epoch") != result["epoch"]:
                    continue
                worker = (generation_record.get("workers") or {}).get(member)
                if worker and worker.get("status") == "completed":
                    result["generation"] = generation_record.get("index")
                    result["pbt_recorded_metric_value"] = (worker.get("metrics") or {}).get("validation_working_point_mistag_percent")
                    break
        return result

    return result


def build_checkpoint_metrics_rows(distinct_checkpoints, provenance_by_id, control_results, full_results, metric_schema_version):
    rows = []
    for entry in distinct_checkpoints:
        checkpoint_id = entry["id"]
        provenance = provenance_by_id[checkpoint_id]
        control = control_results.get(checkpoint_id, {})
        full = full_results.get(checkpoint_id, {})
        control_metrics = control.get("metrics") or {}
        full_metrics = full.get("metrics") or {}
        control_metric_name = "validation_working_point_mistag_percent"
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": entry["path"],
                "sha256": entry["sha256"],
                "source_run": entry.get("source_run"),
                "member": provenance.get("member"),
                "epoch": provenance.get("epoch"),
                "generation": provenance.get("generation"),
                "lr": provenance.get("lr"),
                "pbt_recorded_metric_value": provenance.get("pbt_recorded_metric_value"),
                "control_proxy_50k_status": control.get("status"),
                "control_proxy_50k_metric_finite": finite_metric_ok(control_metrics, control_metric_name),
                "control_proxy_50k_working_point_mistag_percent": control_metrics.get(control_metric_name),
                "control_proxy_50k_ctag_reference_mistag_percent": control_metrics.get("validation_ctag_reference_mistag_percent"),
                "control_proxy_50k_loss": control_metrics.get("validation_loss"),
                "control_proxy_50k_auc": control_metrics.get("validation_auc"),
                "full_validation_status": full.get("status"),
                "full_validation_metric_finite": finite_metric_ok(full_metrics, control_metric_name),
                "full_validation_working_point_mistag_percent": full_metrics.get(control_metric_name),
                "full_validation_ctag_reference_mistag_percent": full_metrics.get("validation_ctag_reference_mistag_percent"),
                "full_validation_loss": full_metrics.get("validation_loss"),
                "full_validation_auc": full_metrics.get("validation_auc"),
                "control_proxy_50k_log": control.get("log"),
                "full_validation_log": full.get("log"),
                "metric_schema_version": metric_schema_version,
            }
        )
        for name, value in control_metrics.items():
            if name.startswith("validation_") and ("mistag_eff" in name) and isinstance(value, (int, float)):
                rows[-1][f"control_proxy_50k_{name}"] = value
        for name, value in full_metrics.items():
            if name.startswith("validation_") and ("mistag_eff" in name) and isinstance(value, (int, float)):
                rows[-1][f"full_validation_{name}"] = value
    return rows


def write_csv_rows(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    import csv

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_outputs(
    experiment_dir,
    args,
    audit_config,
    distinct_checkpoints,
    duplicates,
    provenance_by_id,
    control_results,
    full_results,
    metric_schema_version,
    metric_name,
    metric_mode,
    runtime_seconds,
    run_status,
    run_id,
):
    """Writes checkpoint_metrics.csv, summary.json, and proxy_vs_full.csv
    from whatever results exist so far. Called after control_proxy_50k
    completes (full_results={}, run_status="partial_control_only") AND
    again after full_validation completes (run_status="completed") -- so a
    process killed between the two tiers still leaves a real, readable
    partial result on disk instead of only per-job logs. Every call
    overwrites the previous one (atomic replace via write_csv_rows), so
    there is only ever one, always-current checkpoint_metrics.csv/
    summary.json per run, never stale partial files left behind after a
    successful completion.
    """
    rows = build_checkpoint_metrics_rows(distinct_checkpoints, provenance_by_id, control_results, full_results, metric_schema_version)
    write_csv_rows(experiment_dir / "checkpoint_metrics.csv", rows)

    statistics = full_summary(control_results, full_results, metric_name, metric_mode)

    manifest_path = project_path(audit_config.get("proxy_manifest", "datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json"))
    sanity = run_sanity_check(manifest_path) if manifest_path.is_file() else {"status": "unavailable", "reason": f"manifest not found: {manifest_path}"}

    summary = {
        "run_id": run_id,
        "run_status": run_status,
        "updated_at": utc_now(),
        "git": git_metadata(),
        "config_path": str(project_path(args.config)),
        "audit_config": audit_config,
        "checkpoints_requested": len(audit_config["checkpoints"]),
        "checkpoints_distinct": len(distinct_checkpoints),
        "checkpoints_duplicate": duplicates,
        "runtime_seconds": runtime_seconds,
        "proxy_vs_full": statistics,
        "proxy_sanity_check": sanity,
    }
    tmp_summary_path = (experiment_dir / "summary.json").with_suffix(".json.tmp")
    tmp_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp_summary_path.replace(experiment_dir / "summary.json")

    proxy_vs_full_rows = []
    for entry in distinct_checkpoints:
        checkpoint_id = entry["id"]
        control_metrics = (control_results.get(checkpoint_id) or {}).get("metrics") or {}
        full_metrics = (full_results.get(checkpoint_id) or {}).get("metrics") or {}
        proxy_vs_full_rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "control_proxy_50k_metric": control_metrics.get(metric_name),
                "full_validation_metric": full_metrics.get(metric_name),
                "difference_full_minus_control": (
                    (full_metrics[metric_name] - control_metrics[metric_name])
                    if isinstance(control_metrics.get(metric_name), (int, float)) and isinstance(full_metrics.get(metric_name), (int, float))
                    else None
                ),
            }
        )
    write_csv_rows(experiment_dir / "proxy_vs_full.csv", proxy_vs_full_rows)
    return summary


def main():
    args = parse_args()
    audit_config = load_audit_config(args.config)
    run_id = args.run_id or utc_now().replace(":", "").replace("-", "").split(".")[0].replace("T", "_")
    output_root = project_path(args.output_root or audit_config.get("output_root", "results/research"))
    experiment_dir = output_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
    audit_log_path = experiment_dir / "logs" / "audit.log"

    started_monotonic = time.monotonic()
    log_event(audit_log_path, f"proxy audit started run_id={run_id} config={args.config}")

    # Written immediately, before any evaluation runs: neither depends on
    # results, and having them on disk from the start means a run killed
    # before a single checkpoint finishes still records what was launched.
    (experiment_dir / "resolved_config.yaml").write_text(yaml.safe_dump(audit_config, sort_keys=False), encoding="utf-8")
    (experiment_dir / "environment.json").write_text(json.dumps(environment_info(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    weaver_config = build_weaver_config(audit_config)
    distinct_checkpoints, duplicates = dedupe_checkpoints(audit_config["checkpoints"])
    duplicates_note = (
        "; excluded: " + ", ".join(f"{dup['id']}->{dup.get('duplicate_of')} ({dup.get('reason')})" for dup in duplicates)
        if duplicates
        else ""
    )
    log_event(
        audit_log_path,
        f"checkpoints: {len(distinct_checkpoints)} distinct, {len(duplicates)} duplicate(s){duplicates_note}",
    )

    provenance_by_id = {entry["id"]: provenance_for(entry) for entry in distinct_checkpoints}
    member_checkpoints = {entry["id"]: Path(entry["path"]) for entry in distinct_checkpoints}

    dataset = str(project_path(audit_config["dataset"]))
    control_suffix = audit_config["control_proxy_50k"]["suffix"]
    full_suffix = audit_config["full_validation"]["suffix"]
    metric_schema_version = audit_config.get("metric_schema_version", 1)
    metric_name = audit_config.get("metric_name", "validation_working_point_mistag_percent")
    metric_mode = audit_config.get("metric_mode", "min")

    log_event(audit_log_path, f"evaluating {len(member_checkpoints)} checkpoints on control_proxy_50k ({control_suffix})")
    control_started = time.monotonic()
    control_results = run_tiered_evaluation(
        weaver_config, experiment_dir, 0, "control_proxy_50k", dataset, control_suffix, member_checkpoints, audit_log_path
    )
    control_elapsed = time.monotonic() - control_started
    log_event(audit_log_path, f"control_proxy_50k evaluation finished in {format_duration(control_elapsed)}")

    # Incremental write #1: control_proxy_50k is done, full_validation
    # hasn't started yet. If the process dies during full_validation (the
    # much longer tier), this is what survives -- a complete, correct
    # checkpoint_metrics.csv/summary.json with full_validation columns
    # explicitly empty/status-less, not fabricated.
    write_outputs(
        experiment_dir, args, audit_config, distinct_checkpoints, duplicates, provenance_by_id,
        control_results, {}, metric_schema_version, metric_name, metric_mode,
        {"control_proxy_50k": control_elapsed, "full_validation": None, "total": time.monotonic() - started_monotonic},
        "partial_control_only", run_id,
    )

    log_event(audit_log_path, f"evaluating {len(member_checkpoints)} checkpoints on full_validation ({full_suffix})")
    full_started = time.monotonic()
    full_results = run_tiered_evaluation(
        weaver_config, experiment_dir, 0, "full_validation", dataset, full_suffix, member_checkpoints, audit_log_path
    )
    full_elapsed = time.monotonic() - full_started
    log_event(audit_log_path, f"full_validation evaluation finished in {format_duration(full_elapsed)}")

    total_elapsed = time.monotonic() - started_monotonic
    # Incremental write #2 (final): both tiers done. Overwrites the
    # partial write above with the complete result -- exactly one
    # checkpoint_metrics.csv/summary.json exists at the end, not two.
    write_outputs(
        experiment_dir, args, audit_config, distinct_checkpoints, duplicates, provenance_by_id,
        control_results, full_results, metric_schema_version, metric_name, metric_mode,
        {"control_proxy_50k": control_elapsed, "full_validation": full_elapsed, "total": total_elapsed},
        "completed", run_id,
    )

    log_event(audit_log_path, f"proxy audit finished run_id={run_id} elapsed={format_duration(total_elapsed)} distinct_checkpoints={len(distinct_checkpoints)}")
    print(str(experiment_dir))
    return experiment_dir


def environment_info():
    import platform
    import subprocess

    info = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git": git_metadata(),
    }
    try:
        info["torch_version"] = __import__("torch").__version__
    except Exception as error:
        info["torch_version"] = f"unavailable: {error}"
    try:
        info["gpu_info"] = subprocess.run(
            ["ssh", "iutgpu01", "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except Exception as error:
        info["gpu_info"] = f"unavailable: {error}"
    return info


if __name__ == "__main__":
    main()
