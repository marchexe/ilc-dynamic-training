#!/usr/bin/env python3
"""Qualify fixed proxy-validation sets against canonical checkpoint metrics."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from training.pbt.execution.backend import run_tiered_evaluation
from training.runtime import atomic_json, git_metadata, project_path, sha256, utc_now
from validation.proxy_qualification import (
    DEFAULT_METRIC,
    WORKING_POINTS,
    build_checkpoint_panel,
    paired_event_bootstrap,
    plot_summary,
    summarize_candidate,
    write_results_csv,
)


DEFAULT_RUNS = (
    "runs/pbt/foundation_fixed_lr_50epochs_20260916",
    "runs/pbt/fixed_lr_continuation_20260917_lr_14e-6",
    "runs/pbt/fixed_lr_continuation_20260917_lr_8_5e-6",
    "runs/pbt/windowed_pbt_v2",
    "runs/pbt/windowed_pbt_v2_100epochs",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/eval/proxy_qualification_v1"))
    parser.add_argument("--proxy-sets", type=Path, default=None)
    parser.add_argument("--source-run", action="append", default=[])
    parser.add_argument("--target-panel-size", type=int, default=42)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--data-config", type=Path, default=Path("checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/data_config.auto.yaml"))
    parser.add_argument("--network-config", type=Path, default=Path("networks/pretrained_sgv_particle_transformer.py"))
    parser.add_argument("--host", default="iutgpu01")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fetch-step", default="0.01")
    parser.add_argument("--bootstrap-replicates", type=int, default=50)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def slots(args):
    result = []
    for gpu in (value.strip() for value in args.gpus.split(",")):
        if not gpu:
            continue
        result.append({"host": args.host, "gpu": gpu, "label": f"{args.host}:{gpu}"} if args.host else gpu)
    if not result:
        raise ValueError("at least one GPU slot is required")
    return result


def cache_valid(record, checkpoint, candidate):
    return (
        record.get("status") == "completed"
        and record.get("checkpoint_sha256") == checkpoint["sha256"]
        and record.get("proxy_manifest_sha256") == candidate["proxy_manifest_sha256"]
        and (record.get("metrics") or {}).get(DEFAULT_METRIC) is not None
        and project_path(record.get("prediction_path", "missing")).is_file()
    )


def checkpoint_shard(checkpoints, num_shards, shard_index):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return list(checkpoints)[shard_index::num_shards]


def _read_cache(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def run_candidate(
    args,
    config,
    run_dir,
    candidate,
    checkpoints,
    candidate_index,
    *,
    cache_path=None,
    read_only_cache_paths=(),
    evaluation_dir=None,
    qualification_log=None,
):
    cache_path = cache_path or run_dir / "cache" / "results" / f"{candidate['id']}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _read_cache(cache_path)
    reusable = {}
    for path in read_only_cache_paths:
        reusable.update(_read_cache(path))
    reusable.update(cache)
    pending = [
        checkpoint
        for checkpoint in checkpoints
        if not cache_valid(reusable.get(checkpoint["id"], {}), checkpoint, candidate)
    ]
    slot_count = len(config["slots"])
    for start in range(0, len(pending), slot_count):
        batch = pending[start : start + slot_count]
        member_checkpoints = {
            checkpoint["id"]: project_path(checkpoint["canonical_source_path"])
            for checkpoint in batch
        }
        results = run_tiered_evaluation(
            config,
            evaluation_dir or run_dir / "cache" / "evaluations",
            candidate_index,
            candidate["id"],
            project_path(candidate["dataset"]),
            candidate["validation_suffix"],
            member_checkpoints,
            qualification_log or run_dir / "qualification.log",
        )
        by_id = {checkpoint["id"]: checkpoint for checkpoint in batch}
        for checkpoint_id, result in results.items():
            result["checkpoint_sha256"] = by_id[checkpoint_id]["sha256"]
            result["proxy_manifest_sha256"] = candidate["proxy_manifest_sha256"]
            cache[checkpoint_id] = result
        atomic_json(cache_path, cache)
    reusable.update(cache)
    return reusable


def merge_shard_caches(run_dir, candidates, checkpoints, num_shards):
    """Validate isolated shard outputs, then atomically merge each candidate."""
    merged = {}
    canonical_root = run_dir / "cache" / "results"
    for candidate in candidates:
        candidate_cache = _read_cache(canonical_root / f"{candidate['id']}.json")
        for shard_index in range(num_shards):
            shard_root = run_dir / "cache" / "shards" / f"shard-{shard_index:03d}-of-{num_shards:03d}"
            shard_manifest = _read_cache(shard_root / "manifest.json")
            if shard_manifest.get("status") != "completed":
                raise RuntimeError(f"shard did not complete: {shard_root}")
            expected_ids = {
                checkpoint["id"] for checkpoint in checkpoint_shard(checkpoints, num_shards, shard_index)
            }
            shard_cache = _read_cache(shard_root / "results" / f"{candidate['id']}.json")
            unexpected = set(shard_cache) - expected_ids
            if unexpected:
                raise RuntimeError(f"unexpected checkpoints in {shard_root}: {sorted(unexpected)}")
            for checkpoint_id in sorted(shard_cache):
                incoming = shard_cache[checkpoint_id]
                existing = candidate_cache.get(checkpoint_id)
                if existing is not None and existing != incoming:
                    raise RuntimeError(f"conflicting cached result for {candidate['id']}:{checkpoint_id}")
                candidate_cache[checkpoint_id] = incoming
        missing = [
            checkpoint["id"]
            for checkpoint in checkpoints
            if not cache_valid(candidate_cache.get(checkpoint["id"], {}), checkpoint, candidate)
        ]
        if missing:
            raise RuntimeError(f"candidate {candidate['id']} is missing valid results: {missing}")
        ordered = {checkpoint["id"]: candidate_cache[checkpoint["id"]] for checkpoint in checkpoints}
        atomic_json(canonical_root / f"{candidate['id']}.json", ordered)
        merged[candidate["id"]] = ordered
    return merged


def result_rows(candidate, checkpoints, cache):
    rows = []
    for checkpoint in checkpoints:
        result = cache.get(checkpoint["id"], {})
        metrics = result.get("metrics") or {}
        proxy_metric = metrics.get(DEFAULT_METRIC)
        reference_metric = checkpoint.get("reference_metric")
        if proxy_metric is None or reference_metric is None:
            continue
        row = {
            "candidate_id": candidate["id"],
            "proxy_type": candidate["proxy_type"],
            "proxy_size": candidate["size"],
            "checkpoint_id": checkpoint["id"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_path": checkpoint["canonical_source_path"],
            "run": checkpoint.get("run"),
            "member": checkpoint.get("member"),
            "epoch": checkpoint.get("epoch"),
            "generation": checkpoint.get("generation"),
            "reference_metric": float(reference_metric),
            "proxy_metric": float(proxy_metric),
            "proxy_metric_uncertainty": metrics.get(f"{DEFAULT_METRIC}_uncertainty"),
            "runtime_seconds": result.get("elapsed_seconds"),
            "reference_runtime_seconds": checkpoint.get("reference_runtime_seconds"),
            "prediction_path": result.get("prediction_path"),
            "raw_or_corrected": "raw",
            "correction_status": candidate.get("distribution_correction"),
        }
        for key in WORKING_POINTS:
            row[f"reference_{key}"] = (checkpoint.get("reference_metrics") or {}).get(key)
            row[f"proxy_{key}"] = metrics.get(key)
            row[f"proxy_{key}_uncertainty"] = metrics.get(f"{key}_uncertainty")
        rows.append(row)
    return rows


def best_by(summaries, getter, maximize=True):
    values = [(name, getter(summary)) for name, summary in summaries.items()]
    values = [(name, value) for name, value in values if value is not None and math.isfinite(value)]
    if not values:
        return None
    return (max if maximize else min)(values, key=lambda item: item[1])[0]


def qualification_decision(summaries):
    qualified = []
    for name, summary in summaries.items():
        spearman = summary.get("spearman")
        pairwise = (summary.get("pairwise_ordering") or {}).get("fraction")
        temporal = (summary.get("temporal_direction") or {}).get("fraction")
        bootstrap = summary.get("bootstrap") or {}
        spearman_low = (((bootstrap.get("statistics") or {}).get("spearman") or {}).get("ci95") or [None])[0]
        if (
            spearman is not None and spearman >= 0.8
            and pairwise is not None and pairwise >= 0.8
            and temporal is not None and temporal >= 0.7
            and (spearman_low is None or spearman_low >= 0.5)
        ):
            qualified.append(name)
    return {
        "reliable_enough_for_shadow_adaptive_lr": bool(qualified),
        "qualified_candidates": qualified,
        "criteria": {
            "spearman": ">= 0.8",
            "pairwise_ordering": ">= 0.8",
            "temporal_direction": ">= 0.7",
            "bootstrap_spearman_ci95_lower": ">= 0.5 when available",
        },
        "scope": "shadow testing only; hard/mixed raw metrics are not absolute-calibration substitutes",
    }


def main():
    args = parse_args()
    if args.finalize_only and args.shard_index is not None:
        raise SystemExit("--finalize-only and --shard-index are mutually exclusive")
    if not args.finalize_only and args.num_shards > 1 and args.shard_index is None:
        raise SystemExit("--shard-index is required when --num-shards is greater than one")
    if args.shard_index is not None and args.num_shards == 1:
        raise SystemExit("--shard-index requires --num-shards greater than one")
    run_dir = project_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    proxy_sets_path = project_path(args.proxy_sets) if args.proxy_sets else run_dir / "proxy_sets.json"
    if not proxy_sets_path.is_file():
        raise SystemExit(f"proxy set manifest missing; run build_proxy_sets.py first: {proxy_sets_path}")
    proxy_sets = json.loads(proxy_sets_path.read_text())
    panel_path = run_dir / "checkpoint_panel.json"
    source_runs = args.source_run or list(DEFAULT_RUNS)
    if args.rebuild_panel or not panel_path.is_file():
        panel = build_checkpoint_panel(
            runs=source_runs,
            output=panel_path,
            target_size=args.target_panel_size,
            metric_name=args.metric,
        )
    else:
        panel = json.loads(panel_path.read_text())
    checkpoints = panel["checkpoints"]
    source_snapshot = {
        "run_manifests": {str(project_path(run) / "manifest.json"): sha256(project_path(run) / "manifest.json") for run in source_runs},
        "checkpoints": {checkpoint["canonical_source_path"]: checkpoint["sha256"] for checkpoint in checkpoints},
    }
    command_record = {
        "build": "PYTHONPATH=scripts .venv/bin/python3 scripts/validation/build_proxy_sets.py",
        "qualify": "PYTHONPATH=scripts .venv/bin/python3 scripts/validation/qualify_proxy.py",
    }
    shard_root = None
    if args.shard_index is not None:
        shard_root = run_dir / "cache" / "shards" / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    manifest_path = (shard_root / "manifest.json") if shard_root else (run_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.monotonic()
    manifest = {
        "schema_version": 1,
        "experiment": run_dir.name,
        "status": "prepared" if args.prepare_only else "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "git": git_metadata(),
        "observational_only": True,
        "canonical_metric": args.metric,
        "canonical_reference_source": "existing per-generation validation metrics; no reference rerun",
        "checkpoint_panel": str(panel_path),
        "proxy_sets": str(proxy_sets_path),
        "source_snapshot_before": source_snapshot,
        "commands": command_record,
        "execution": {
            "mode": "finalize" if args.finalize_only else ("shard" if shard_root else "unsharded"),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "slots": slots(args) if not args.finalize_only else [],
        },
    }
    atomic_json(manifest_path, manifest)
    if args.prepare_only:
        print(panel_path)
        return

    config = {
        "shared": {
            "data_config": str(project_path(args.data_config)),
            "network_config": str(project_path(args.network_config)),
            "data_extension": "parquet",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "fetch_step": args.fetch_step,
            "use_amp": True,
            "amp_dtype": "fp16",
            "deterministic": True,
            "data_audit": True,
            "save_predictions": True,
        },
        "slots": slots(args),
    }
    candidates = proxy_sets["candidates"]
    if shard_root:
        shard_checkpoints = checkpoint_shard(checkpoints, args.num_shards, args.shard_index)
        for index, candidate in enumerate(candidates):
            cache = run_candidate(
                args,
                config,
                run_dir,
                candidate,
                shard_checkpoints,
                index,
                cache_path=shard_root / "results" / f"{candidate['id']}.json",
                read_only_cache_paths=(run_dir / "cache" / "results" / f"{candidate['id']}.json",),
                evaluation_dir=shard_root / "evaluations",
                qualification_log=shard_root / "qualification.log",
            )
            invalid = [
                checkpoint["id"]
                for checkpoint in shard_checkpoints
                if not cache_valid(cache.get(checkpoint["id"], {}), checkpoint, candidate)
            ]
            if invalid:
                raise RuntimeError(f"shard has invalid results for {candidate['id']}: {invalid}")
        source_snapshot_after = {
            "run_manifests": {path: sha256(project_path(path)) for path in source_snapshot["run_manifests"]},
            "checkpoints": {path: sha256(project_path(path)) for path in source_snapshot["checkpoints"]},
        }
        if source_snapshot_after != source_snapshot:
            raise RuntimeError("source run/checkpoint mutation detected during qualification shard")
        manifest.update(
            status="completed",
            updated_at=utc_now(),
            finished_at=utc_now(),
            wall_clock_seconds=time.monotonic() - started_monotonic,
            source_snapshot_after=source_snapshot_after,
        )
        atomic_json(manifest_path, manifest)
        print(manifest_path)
        return

    if args.finalize_only:
        caches = merge_shard_caches(run_dir, candidates, checkpoints, args.num_shards)
    else:
        caches = {
            candidate["id"]: run_candidate(args, config, run_dir, candidate, checkpoints, index)
            for index, candidate in enumerate(candidates)
        }

    rows = []
    summaries = {}
    for index, candidate in enumerate(candidates):
        cache = caches[candidate["id"]]
        candidate_rows = result_rows(candidate, checkpoints, cache)
        rows.extend(candidate_rows)
        summary = summarize_candidate(candidate_rows, metric_name=args.metric)
        summary["bootstrap"] = paired_event_bootstrap(
            candidate_rows,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index,
        )
        summaries[candidate["id"]] = summary
        write_results_csv(run_dir / "results.csv", rows)
        atomic_json(run_dir / "summary.partial.json", {"candidates": summaries})

    best = {
        "pearson": best_by(summaries, lambda item: item.get("pearson")),
        "spearman": best_by(summaries, lambda item: item.get("spearman")),
        "pairwise_ordering": best_by(summaries, lambda item: (item.get("pairwise_ordering") or {}).get("fraction")),
        "temporal_direction": best_by(summaries, lambda item: (item.get("temporal_direction") or {}).get("fraction")),
        "lowest_regret": best_by(summaries, lambda item: (item.get("best_checkpoint") or {}).get("regret"), maximize=False),
        "fastest": best_by(summaries, lambda item: (item.get("runtime_seconds") or {}).get("mean"), maximize=False),
    }
    summary_payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "metric": args.metric,
        "checkpoint_count": len(checkpoints),
        "candidate_count": len(proxy_sets["candidates"]),
        "finalization_wall_clock_seconds": time.monotonic() - started_monotonic,
        "candidates": summaries,
        "best_candidate_by_statistic": best,
        "decision": qualification_decision(summaries),
        "uncertainty": {
            "working_points": "Jeffreys Beta(1/2,1/2) posterior SD; non-zero for zero observed mistakes",
            "ranking": "paired class-stratified event bootstrap with deterministic seeds",
        },
    }
    atomic_json(run_dir / "summary.json", summary_payload)
    plot_summary(run_dir, rows, summaries)

    source_snapshot_after = {
        "run_manifests": {path: sha256(project_path(path)) for path in source_snapshot["run_manifests"]},
        "checkpoints": {path: sha256(project_path(path)) for path in source_snapshot["checkpoints"]},
    }
    if source_snapshot_after != source_snapshot:
        raise RuntimeError("source run/checkpoint mutation detected during qualification")
    manifest.update(
        status="completed",
        updated_at=utc_now(),
        finished_at=utc_now(),
        wall_clock_seconds=time.monotonic() - started_monotonic,
        source_snapshot_after=source_snapshot_after,
    )
    atomic_json(manifest_path, manifest)
    print(run_dir / "summary.json")


if __name__ == "__main__":
    main()
