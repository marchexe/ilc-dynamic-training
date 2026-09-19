#!/usr/bin/env python3
"""Core helpers for fixed proxy-validation qualification.

This module is deliberately observational: it only reads canonical run
artifacts and writes new data below the caller-selected proxy/evaluation
directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.runtime import PROJECT_DIR, atomic_json, project_path, sha256, utc_now


FLAVORS = ("bb", "cc", "dd")
CLASS_INDEX = {"bb": 0, "cc": 1, "dd": 2}
DEFAULT_METRIC = "validation_total_reference_mistag_geomean_percent"
WORKING_POINTS = (
    "validation_bc_mistag_eff_0.80_percent",
    "validation_bd_mistag_eff_0.80_percent",
    "validation_bc_mistag_eff_0.90_percent",
    "validation_bd_mistag_eff_0.90_percent",
    "validation_cb_mistag_eff_0.50_percent",
    "validation_cd_mistag_eff_0.50_percent",
    "validation_cb_mistag_eff_0.80_percent",
    "validation_cd_mistag_eff_0.80_percent",
)
EPOCH_RE = re.compile(r"net_epoch-(\d+)_state\.pt$")


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def stable_seed(seed, *parts):
    payload = ":".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validation_files(dataset, suffix="val50k_tail"):
    dataset = project_path(dataset)
    files = {}
    for flavor in FLAVORS:
        matches = sorted(dataset.glob(f"*_{flavor}_{suffix}.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one *_{flavor}_{suffix}.parquet in {dataset}, found {len(matches)}"
            )
        files[flavor] = matches[0]
    return files


def _prediction_arrays(prediction_path, source_rows):
    table = pq.read_table(project_path(prediction_path), columns=["scores", "_label_"])
    scores = np.asarray(table.column("scores").to_pylist(), dtype=np.float64)
    labels = np.asarray(table.column("_label_").to_pylist(), dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] < 3:
        raise ValueError(f"anchor predictions need >=3 score columns: {prediction_path}")
    result = {}
    for flavor in FLAVORS:
        selected = scores[labels == CLASS_INDEX[flavor]]
        if len(selected) != source_rows[flavor]:
            raise ValueError(
                f"anchor predictions for {flavor}: expected {source_rows[flavor]} rows, got {len(selected)}"
            )
        result[flavor] = selected
    return result


def predictive_entropy(scores):
    probabilities = np.clip(np.asarray(scores, dtype=np.float64), 1.0e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return -np.sum(probabilities * np.log(probabilities), axis=1)


def select_indices(total_rows, rows, proxy_type, *, seed, flavor, hardness, mixed_hard_fraction):
    if rows > total_rows:
        raise ValueError(f"requested {rows} rows for {flavor}, only {total_rows} available")
    representative_order = list(range(total_rows))
    random.Random(stable_seed(seed, flavor)).shuffle(representative_order)
    hard_order = np.lexsort((np.arange(total_rows), -np.asarray(hardness))).tolist()
    if proxy_type == "representative":
        selected = representative_order[:rows]
        inclusion = {"design": "simple_random_without_replacement", "probability": rows / total_rows}
    elif proxy_type == "hard":
        selected = hard_order[:rows]
        inclusion = {
            "design": "deterministic_top_entropy",
            "probability": "1 for selected events, 0 otherwise",
        }
    elif proxy_type == "mixed":
        hard_rows = int(round(rows * mixed_hard_fraction))
        representative_rows = rows - hard_rows
        hard_selected = hard_order[:hard_rows]
        hard_set = set(hard_selected)
        representative_selected = [index for index in representative_order if index not in hard_set][
            :representative_rows
        ]
        selected = representative_selected + hard_selected
        inclusion = {
            "design": "deterministic_hard_plus_random_remainder_without_replacement",
            "hard_probability": 1.0,
            "non_hard_probability": representative_rows / (total_rows - hard_rows),
            "hard_rows": hard_rows,
            "representative_rows": representative_rows,
        }
    else:
        raise ValueError(f"unknown proxy type: {proxy_type}")
    return sorted(selected), inclusion


def _event_id_values(table, indices):
    for name in ("event_id", "event_nbh"):
        if name in table.column_names:
            values = table.column(name).take(pa.array(indices, type=pa.int64())).to_pylist()
            if all(not isinstance(value, (list, dict)) for value in values):
                return name, values
    return None, None


def build_proxy_sets(
    *,
    dataset,
    source_suffix,
    output_root,
    manifest_output,
    anchor_checkpoint,
    anchor_predictions,
    sizes=(15000, 30000, 60000),
    proxy_types=("representative", "hard", "mixed"),
    seed=20260920,
    mixed_hard_fraction=0.30,
    compression="lz4",
    force=False,
):
    """Build frozen class-balanced proxy parquets and membership manifests."""
    dataset = project_path(dataset)
    output_root = project_path(output_root)
    manifest_output = project_path(manifest_output)
    anchor_checkpoint = project_path(anchor_checkpoint)
    anchor_predictions = project_path(anchor_predictions)
    files = validation_files(dataset, source_suffix)
    source_rows = {flavor: pq.ParquetFile(path).metadata.num_rows for flavor, path in files.items()}
    anchor_scores = _prediction_arrays(anchor_predictions, source_rows)
    hardness = {flavor: predictive_entropy(scores) for flavor, scores in anchor_scores.items()}
    anchor_sha = sha256(anchor_checkpoint)

    if manifest_output.exists() and not force:
        raise FileExistsError(f"proxy manifest exists; membership is frozen (use --force explicitly): {manifest_output}")

    candidates = []
    for size in map(int, sizes):
        if size % len(FLAVORS):
            raise ValueError(f"proxy total size must be divisible by {len(FLAVORS)}: {size}")
        rows_per_class = size // len(FLAVORS)
        for proxy_type in proxy_types:
            size_label = f"{size // 1000}k" if size >= 1000 and size % 1000 == 0 else str(size)
            candidate_id = f"{proxy_type}_{size_label}"
            candidate_dir = output_root / candidate_id
            membership_path = candidate_dir / "membership.json"
            if membership_path.exists() and not force:
                raise FileExistsError(f"proxy candidate already frozen: {membership_path}")
            membership = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "selection_method": proxy_type,
                "difficulty": "anchor predictive entropy" if proxy_type in {"hard", "mixed"} else None,
                "seed": int(seed),
                "source_validation_dataset": display_path(dataset),
                "source_suffix": source_suffix,
                "anchor_checkpoint": display_path(anchor_checkpoint),
                "anchor_checkpoint_sha256": anchor_sha,
                "anchor_predictions": display_path(anchor_predictions),
                "anchor_predictions_sha256": sha256(anchor_predictions),
                "mixed_hard_fraction": mixed_hard_fraction if proxy_type == "mixed" else None,
                "rows_total": size,
                "class_counts": {},
                "events_by_class": {},
                "files": {},
                "raw_metrics_only": proxy_type in {"hard", "mixed"},
                "distribution_correction": (
                    "not applied: the existing evaluator has no event-weighted fixed-working-point path; "
                    "raw hard-enriched metrics are suitable for ranking/trend tests, not absolute calibration"
                    if proxy_type in {"hard", "mixed"}
                    else "not needed: equal-probability class-stratified sample"
                ),
            }
            for flavor, source_path in files.items():
                source_table = pq.read_table(source_path)
                indices, inclusion = select_indices(
                    source_table.num_rows,
                    rows_per_class,
                    proxy_type,
                    seed=seed,
                    flavor=flavor,
                    hardness=hardness[flavor],
                    mixed_hard_fraction=mixed_hard_fraction,
                )
                subset = source_table.take(pa.array(indices, type=pa.int64()))
                output_path = candidate_dir / source_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(subset, output_path, compression=compression)
                event_id_column, event_ids = _event_id_values(source_table, indices)
                membership["class_counts"][flavor] = len(indices)
                membership["events_by_class"][flavor] = {
                    "source_indices": indices,
                    "event_id_column": event_id_column,
                    "event_ids": event_ids,
                    "inclusion_probabilities": inclusion,
                }
                membership["files"][flavor] = {
                    "path": display_path(output_path),
                    "sha256": sha256(output_path),
                    "source": display_path(source_path),
                    "source_sha256": sha256(source_path),
                    "rows": len(indices),
                }
            atomic_json(membership_path, membership)
            candidates.append(
                {
                    "id": candidate_id,
                    "proxy_type": proxy_type,
                    "size": size,
                    "class_counts": membership["class_counts"],
                    "dataset": display_path(candidate_dir),
                    "validation_suffix": source_suffix,
                    "membership_manifest": display_path(membership_path),
                    "proxy_manifest_sha256": sha256(membership_path),
                    "selection_method": membership["selection_method"],
                    "seed": int(seed),
                    "anchor_checkpoint": display_path(anchor_checkpoint),
                    "anchor_checkpoint_sha256": anchor_sha,
                    "source_validation_dataset": display_path(dataset),
                    "distribution_correction": membership["distribution_correction"],
                }
            )

    manifest = {
        "schema_version": 1,
        "name": manifest_output.parent.name,
        "created_at": utc_now(),
        "source_validation_dataset": display_path(dataset),
        "source_suffix": source_suffix,
        "seed": int(seed),
        "anchor_checkpoint": display_path(anchor_checkpoint),
        "anchor_checkpoint_sha256": anchor_sha,
        "anchor_predictions": display_path(anchor_predictions),
        "anchor_predictions_sha256": sha256(anchor_predictions),
        "candidates": candidates,
    }
    atomic_json(manifest_output, manifest)
    return manifest


def _worker_candidates(run_path, metric_name):
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = []
    for generation in manifest.get("generations", []):
        epoch = generation.get("epoch")
        for member, worker in (generation.get("workers") or {}).items():
            metrics = worker.get("metrics") or {}
            reference = metrics.get(metric_name)
            checkpoint = run_path / member / f"net_epoch-{epoch}_state.pt"
            if worker.get("status") != "completed" or not checkpoint.is_file():
                continue
            try:
                reference = float(reference)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(reference):
                continue
            runtime_seconds = None
            try:
                runtime_seconds = (
                    datetime.fromisoformat(worker["finished_at"]) - datetime.fromisoformat(worker["started_at"])
                ).total_seconds()
            except (KeyError, TypeError, ValueError):
                pass
            candidates.append(
                {
                    "canonical_source_path": str(checkpoint.resolve()),
                    "run": display_path(run_path),
                    "member": member,
                    "lineage": (manifest.get("members") or {}).get(member),
                    "generation": generation.get("index"),
                    "epoch": epoch,
                    "full_epoch": epoch,
                    "reference_metric_name": metric_name,
                    "reference_metric": reference,
                    "reference_runtime_seconds": runtime_seconds,
                    "reference_metrics": {key: metrics.get(key) for key in (metric_name, *WORKING_POINTS)},
                    "reference_dataset": (manifest.get("config") or {}).get("shared", {}).get("dataset"),
                    "reference_suffix": (manifest.get("config") or {}).get("shared", {}).get("validation_suffix"),
                    "provenance": {"kind": "generation_worker", "manifest": display_path(manifest_path)},
                }
            )
    return manifest, candidates


def _evenly_spaced(items, count):
    if len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, count)
    return [items[int(round(position))] for position in positions]


def build_checkpoint_panel(*, runs, output, target_size=42, metric_name=DEFAULT_METRIC):
    """Build a trajectory-diverse panel and deduplicate actual state bytes."""
    output = project_path(output)
    run_groups = []
    all_candidates = []
    for run in runs:
        run_path = project_path(run)
        manifest, candidates = _worker_candidates(run_path, metric_name)
        by_member = defaultdict(list)
        for candidate in candidates:
            by_member[candidate["member"]].append(candidate)
        ordered = []
        for member in sorted(by_member):
            ordered.extend(sorted(by_member[member], key=lambda item: item["epoch"]))

        specials = []
        for role, record in (("protected", manifest.get("protected_best")), ("global_best", manifest.get("best"))):
            if not record:
                continue
            source = ((record.get("checkpoint") or {}).get("state") or {}).get("path")
            source = source or record.get("source_state_path") or record.get("state_path")
            if not source or not Path(source).is_file():
                continue
            metrics = record.get("metrics") or {}
            specials.append(
                {
                    "canonical_source_path": str(Path(source).resolve()),
                    "run": display_path(run_path),
                    "member": record.get("member"),
                    "lineage": (manifest.get("members") or {}).get(record.get("member")),
                    "generation": record.get("generation"),
                    "epoch": record.get("epoch"),
                    "full_epoch": record.get("epoch"),
                    "reference_metric_name": metric_name,
                    "reference_metric": metrics.get(metric_name, record.get("metric_value")),
                    "reference_runtime_seconds": None,
                    "reference_metrics": {key: metrics.get(key) for key in (metric_name, *WORKING_POINTS)},
                    "reference_dataset": (manifest.get("config") or {}).get("shared", {}).get("dataset"),
                    "reference_suffix": (manifest.get("config") or {}).get("shared", {}).get("validation_suffix"),
                    "provenance": {"kind": role, "manifest": display_path(run_path / "manifest.json")},
                }
            )
        run_groups.append((specials, ordered))
        all_candidates.extend(specials)

    remaining = max(0, int(target_size) - len(all_candidates))
    per_run = max(1, math.ceil(remaining / max(len(run_groups), 1)))
    for _specials, ordered in run_groups:
        # Sorting by (epoch, metric) keeps the sample temporal while retaining
        # good/medium/bad members at shared epochs.
        all_candidates.extend(_evenly_spaced(sorted(ordered, key=lambda item: (item["epoch"], item["reference_metric"])), per_run))

    distinct = []
    duplicate_records = []
    seen = {}
    for candidate in all_candidates:
        path = Path(candidate["canonical_source_path"])
        digest = sha256(path)
        if digest in seen:
            duplicate_records.append(
                {
                    "path": display_path(path),
                    "sha256": digest,
                    "duplicate_of": seen[digest],
                    "provenance": candidate["provenance"],
                }
            )
            continue
        candidate = dict(candidate)
        candidate["canonical_source_path"] = display_path(path)
        candidate["sha256"] = digest
        candidate["id"] = f"checkpoint_{len(distinct):03d}_{digest[:10]}"
        seen[digest] = candidate["id"]
        distinct.append(candidate)
        if len(distinct) >= target_size:
            break

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "metric_name": metric_name,
        "metric_mode": "min",
        "target_size": int(target_size),
        "unique_model_states": len(distinct),
        "source_runs": [display_path(project_path(run)) for run in runs],
        "checkpoints": distinct,
        "duplicates": duplicate_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload


def deduplicate_checkpoint_entries(entries):
    """Small reusable helper used by tests and callers with custom panels."""
    distinct, duplicates, seen = [], [], {}
    for entry in entries:
        item = dict(entry)
        digest = item.get("sha256") or sha256(project_path(item["canonical_source_path"]))
        item["sha256"] = digest
        if digest in seen:
            duplicates.append({**item, "duplicate_of": seen[digest]})
        else:
            seen[digest] = item.get("id") or item["canonical_source_path"]
            distinct.append(item)
    return distinct, duplicates


def _correlations(xs, ys):
    if len(xs) < 3:
        return {"n": len(xs), "pearson": None, "spearman": None}
    from scipy import stats

    pearson = stats.pearsonr(xs, ys).statistic
    spearman = stats.spearmanr(xs, ys).statistic
    return {"n": len(xs), "pearson": float(pearson), "spearman": float(spearman)}


def pairwise_ordering(reference, proxy):
    agreeing = total = 0
    for left in range(len(reference)):
        for right in range(left + 1, len(reference)):
            rd = reference[left] - reference[right]
            pd = proxy[left] - proxy[right]
            if rd == 0 or pd == 0:
                continue
            total += 1
            agreeing += (rd > 0) == (pd > 0)
    return {"agreeing": int(agreeing), "pairs": total, "fraction": agreeing / total if total else None}


def temporal_direction(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("member") is not None and row.get("epoch") is not None:
            grouped[(row.get("run"), row.get("member"))].append(row)
    agreeing = total = 0
    for items in grouped.values():
        items.sort(key=lambda row: row["epoch"])
        for previous, current in zip(items, items[1:]):
            reference_delta = current["reference_metric"] - previous["reference_metric"]
            proxy_delta = current["proxy_metric"] - previous["proxy_metric"]
            if reference_delta == 0 or proxy_delta == 0:
                continue
            total += 1
            agreeing += (reference_delta > 0) == (proxy_delta > 0)
    return {"agreeing": int(agreeing), "comparisons": total, "fraction": agreeing / total if total else None}


def summarize_candidate(rows, *, metric_name=DEFAULT_METRIC):
    paired = [row for row in rows if math.isfinite(row["reference_metric"]) and math.isfinite(row["proxy_metric"])]
    reference = [row["reference_metric"] for row in paired]
    proxy = [row["proxy_metric"] for row in paired]
    result = _correlations(reference, proxy)
    result["pairwise_ordering"] = pairwise_ordering(reference, proxy)
    result["temporal_direction"] = temporal_direction(paired)
    if paired:
        reference_best = min(paired, key=lambda row: row["reference_metric"])
        proxy_best = min(paired, key=lambda row: row["proxy_metric"])
        result["best_checkpoint"] = {
            "agrees": reference_best["checkpoint_id"] == proxy_best["checkpoint_id"],
            "reference_best": reference_best["checkpoint_id"],
            "proxy_best": proxy_best["checkpoint_id"],
            "regret": proxy_best["reference_metric"] - reference_best["reference_metric"],
        }
        runtimes = [row["runtime_seconds"] for row in paired if row.get("runtime_seconds") is not None]
        result["runtime_seconds"] = {
            "total": sum(runtimes),
            "mean": sum(runtimes) / len(runtimes) if runtimes else None,
        }
        reference_runtimes = [row.get("reference_runtime_seconds") for row in paired]
        reference_runtimes = [value for value in reference_runtimes if value]
        result["speedup_vs_reference"] = (
            (sum(reference_runtimes) / len(reference_runtimes)) / result["runtime_seconds"]["mean"]
            if reference_runtimes and result["runtime_seconds"]["mean"]
            else None
        )
    wp = {}
    for key in WORKING_POINTS:
        pairs = [
            (row.get(f"reference_{key}"), row.get(f"proxy_{key}"))
            for row in paired
            if row.get(f"reference_{key}") is not None and row.get(f"proxy_{key}") is not None
        ]
        if pairs:
            values = _correlations([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            values["mean_absolute_error_pp"] = float(np.mean([abs(a - b) for a, b in pairs]))
            wp[key] = values
    result["working_points"] = wp
    result["metric_name"] = metric_name
    return result


def _prediction_file(path):
    table = pq.read_table(path, columns=["scores", "_label_"])
    return (
        np.asarray(table.column("_label_").to_pylist(), dtype=np.int64),
        np.asarray(table.column("scores").to_pylist(), dtype=np.float64),
    )


def paired_event_bootstrap(rows, *, replicates=50, seed=20260920):
    """Paired, class-stratified bootstrap over one proxy's frozen events."""
    if not rows or replicates <= 0:
        return {"status": "disabled", "replicates": 0}
    predictions = {}
    labels = None
    for row in rows:
        path = row.get("prediction_path")
        if not path or not project_path(path).is_file():
            return {"status": "unavailable", "reason": "event prediction cache missing"}
        current_labels, scores = _prediction_file(project_path(path))
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            return {"status": "unavailable", "reason": "prediction event order differs across checkpoints"}
        predictions[row["checkpoint_id"]] = scores

    from training.runtime import _working_point_metrics
    from weaver.utils.nn.metrics import bkg_rejection_at_eff

    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == class_index) for class_index in range(3)]
    statistics = defaultdict(list)
    selected_best = defaultdict(int)
    for _ in range(int(replicates)):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in by_class])
        bootstrap_rows = []
        for row in rows:
            curves = bkg_rejection_at_eff(labels[sampled], predictions[row["checkpoint_id"]][sampled])
            curve_record = {
                "efficiencies": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "pairs": curves,
            }
            metric = _working_point_metrics(curve_record).get(DEFAULT_METRIC)
            if metric is None or not math.isfinite(metric):
                continue
            bootstrap_rows.append({**row, "proxy_metric": metric})
        if len(bootstrap_rows) != len(rows):
            continue
        summary = summarize_candidate(bootstrap_rows)
        for name, value in (
            ("pearson", summary.get("pearson")),
            ("spearman", summary.get("spearman")),
            ("pairwise_ordering", summary.get("pairwise_ordering", {}).get("fraction")),
            ("temporal_direction", summary.get("temporal_direction", {}).get("fraction")),
            ("regret", summary.get("best_checkpoint", {}).get("regret")),
        ):
            if value is not None and math.isfinite(value):
                statistics[name].append(value)
        best = summary.get("best_checkpoint", {}).get("proxy_best")
        if best:
            selected_best[best] += 1

    def interval(values):
        return {
            "median": float(np.median(values)),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "n": len(values),
        } if values else None

    return {
        "status": "ok",
        "method": "paired class-stratified event bootstrap; identical resample indices for every checkpoint",
        "seed": int(seed),
        "replicates_requested": int(replicates),
        "replicates_completed": max((len(values) for values in statistics.values()), default=0),
        "statistics": {name: interval(values) for name, values in statistics.items()},
        "proxy_best_selection_fraction": {
            checkpoint: count / max(sum(selected_best.values()), 1)
            for checkpoint, count in sorted(selected_best.items())
        },
        "calibration_note": "For hard/mixed sets this quantifies conditional ranking stability, not natural-distribution calibration.",
    }


def write_results_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _representative_limits(rows):
    values = [
        float(row[key])
        for row in rows
        if row.get("proxy_type") == "representative"
        for key in ("reference_metric", "proxy_metric")
    ]
    if not values:
        return None
    lower = min(values)
    upper = max(values)
    padding = max((upper - lower) * 0.05, 1.0e-6)
    return lower - padding, upper + padding


def plot_proxy_vs_reference(run_dir, rows, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = Path(run_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by_candidate = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)

    candidates = ("representative_15k", "representative_30k", "representative_60k")
    limits = _representative_limits(rows)
    if limits is None or any(candidate not in by_candidate for candidate in candidates):
        raise ValueError("all representative proxy rows are required for the calibration plot")

    lower, upper = limits
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), constrained_layout=True)
    for ax, candidate in zip(axes, candidates):
        items = by_candidate[candidate]
        summary = summaries[candidate]
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="0.45", linewidth=1.1, zorder=0)
        ax.scatter(
            [row["reference_metric"] for row in items],
            [row["proxy_metric"] for row in items],
            s=24,
            color="tab:blue",
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(candidate)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.04,
            0.96,
            f"Pearson $r$ = {summary['pearson']:.3f}\nSpearman $\\rho$ = {summary['spearman']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    fig.supxlabel("Reference mistag [%]")
    fig.supylabel("Proxy mistag [%]")
    fig.suptitle("Representative proxy calibration vs 150k reference", fontsize=14)
    fig.savefig(plots / "proxy_vs_reference.png", dpi=150)
    plt.close(fig)


def plot_all_proxies_vs_reference(run_dir, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = Path(run_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by_candidate = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)

    representative_limits = _representative_limits(rows)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (candidate, items) in zip(axes.flat, sorted(by_candidate.items())):
        ax.scatter([row["reference_metric"] for row in items], [row["proxy_metric"] for row in items], s=14)
        ax.set_title(candidate)
        ax.set_xlabel("reference")
        ax.set_ylabel("proxy (raw)")
        if items[0].get("proxy_type") == "representative" and representative_limits is not None:
            lower, upper = representative_limits
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="0.4", linewidth=1, zorder=0)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat[len(by_candidate):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots / "proxy_vs_reference_all_proxies.png", dpi=150)
    plt.close(fig)


def plot_summary(run_dir, rows, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = Path(run_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_proxy_vs_reference(run_dir, rows, summaries)
    plot_all_proxies_vs_reference(run_dir, rows)

    names = sorted(summaries)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [summaries[name].get("spearman") or 0 for name in names], 0.4, label="Spearman")
    ax.bar(x + 0.2, [(summaries[name].get("pairwise_ordering") or {}).get("fraction") or 0 for name in names], 0.4, label="pairwise")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "ranking_agreement.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, [(summaries[name].get("temporal_direction") or {}).get("fraction") or 0 for name in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("direction agreement")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots / "temporal_direction_agreement.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [(summary.get("runtime_seconds") or {}).get("mean") for summary in summaries.values()],
        [summary.get("spearman") for summary in summaries.values()],
    )
    for name, summary in summaries.items():
        ax.annotate(name, ((summary.get("runtime_seconds") or {}).get("mean"), summary.get("spearman")), fontsize=7)
    ax.set_xlabel("mean runtime / checkpoint (s)")
    ax.set_ylabel("Spearman")
    fig.tight_layout()
    fig.savefig(plots / "runtime_vs_proxy_quality.png", dpi=150)
    plt.close(fig)

    wp_names = list(WORKING_POINTS)
    matrix = [[(summaries[name].get("working_points", {}).get(wp) or {}).get("spearman") or 0 for wp in wp_names] for name in names]
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(len(wp_names)), [wp.replace("validation_", "").replace("_percent", "") for wp in wp_names], rotation=45, ha="right")
    fig.colorbar(image, ax=ax, label="Spearman")
    fig.tight_layout()
    fig.savefig(plots / "working_point_agreement.png", dpi=150)
    plt.close(fig)
