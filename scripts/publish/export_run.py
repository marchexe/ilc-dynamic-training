#!/usr/bin/env python3
"""Export a small, reproducible publication bundle from recorded run evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRIC = "validation_total_reference_mistag_geomean_percent"
MOVED_PREFIXES = {
    "runs/prepared": "configs/prepared",
    "runs/launch_logs": "logs/experiments",
    "runs/launchers": "logs/experiments/launchers",
    "runs/dev": "runs/archive/diagnostics",
    "runs/showcase": "runs/archive/showcases",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_recorded_path(value: str | Path, *, source_run: Path) -> Path:
    """Resolve live and pre-cleanup recorded paths without editing the evidence."""
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((PROJECT_ROOT / raw, source_run / raw))
    else:
        try:
            candidates.append(PROJECT_ROOT / raw.relative_to(PROJECT_ROOT))
        except ValueError:
            pass
    text = str(raw)
    root_text = str(PROJECT_ROOT)
    relative_text = text[len(root_text) + 1 :] if text.startswith(root_text + os.sep) else text
    for old, new in MOVED_PREFIXES.items():
        if relative_text == old or relative_text.startswith(old + os.sep):
            candidates.append(PROJECT_ROOT / (new + relative_text[len(old) :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def compact_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or record if isinstance(record, dict) else {}
    keep = {
        key: value
        for key, value in metrics.items()
        if key in {METRIC, "validation_accuracy", "validation_auc", "validation_loss"}
        or (key.startswith("validation_") and "mistag" in key and "uncertainty" not in key)
    }
    return keep


def compact_selection(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    keys = ("member", "generation", "epoch", "lr", "metric", "metric_value")
    result = {key: record.get(key) for key in keys if record.get(key) is not None}
    if "member" not in result and record.get("trial") is not None:
        result["member"] = record["trial"]
    if "lr" not in result and record.get("LR") is not None:
        result["lr"] = record["LR"]
    if "metric" not in result and record.get("optimization_metric_name") is not None:
        result["metric"] = record["optimization_metric_name"]
    if "metric_value" not in result and record.get("optimization_metric_value") is not None:
        result["metric_value"] = record["optimization_metric_value"]
    result["metrics"] = compact_metrics(record)
    return result


def copy_csvs(run_dirs: list[Path], destination: Path) -> bool:
    sources = [(run, run / "metrics.csv") for run in run_dirs if (run / "metrics.csv").is_file()]
    if not sources:
        return False
    if len(sources) == 1:
        shutil.copy2(sources[0][1], destination)
        return True
    rows: list[dict[str, str]] = []
    fields: list[str] = ["source_run"]
    for run, path in sources:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
            for row in reader:
                rows.append({"source_run": display_path(run), **row})
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_lr_history(metrics_path: Path, destination: Path) -> bool:
    if not metrics_path.is_file():
        return False
    wanted = ("source_run", "generation", "training_chunk", "trial", "LR",
              "optimization_metric_name", "optimization_metric_value", "total_mistag_score")
    with metrics_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = [field for field in wanted if field in (reader.fieldnames or [])]
        if "LR" not in fields:
            return False
        rows = [{field: row.get(field, "") for field in fields} for row in reader]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_exploit_history(manifest: dict[str, Any], destination: Path) -> bool:
    rows: list[dict[str, Any]] = []
    for generation in manifest.get("generations", []):
        for event in generation.get("exploit", []) or []:
            row = {"generation": generation.get("index"), "epoch": generation.get("epoch")}
            if isinstance(event, dict):
                row.update({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                            for key, value in event.items()})
            else:
                row["event"] = str(event)
            rows.append(row)
    if not rows:
        return False
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return True


def copy_plots(source_run: Path, destination: Path) -> list[str]:
    plot_root = source_run / "plots"
    if not plot_root.is_dir():
        return []
    copied = []
    for source in sorted(plot_root.rglob("*.png")):
        relative = source.relative_to(plot_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(relative.as_posix())
    return copied


def model_record(source: Path, target: Path, *, role: str, reason: str,
                 epoch: Any, metric: Any, existing: dict[str, Path]) -> dict[str, Any]:
    digest = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if digest in existing:
        os.link(existing[digest], target)
    else:
        shutil.copy2(source, target)
        existing[digest] = target
    return {
        "file": target.name,
        "role": role,
        "selection_reason": reason,
        "source_path": str(source),
        "epoch": epoch,
        "metric_name": METRIC,
        "metric_value": metric,
        "sha256": digest,
        "size_bytes": target.stat().st_size,
    }


def export_models(source_run: Path, manifest: dict[str, Any], destination: Path,
                  *, resume_bundle: str | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    method = manifest.get("method")
    if method == "cadenced_pbt_v1":
        if resume_bundle:
            raise ValueError("cadenced_pbt_v1 exports its global-best resume assets automatically")
        models: list[dict[str, Any]] = []
        existing: dict[str, Path] = {}
        best = manifest.get("best") or {}
        final = (manifest.get("generations") or [{}])[-1]
        final_member = (final.get("ranking") or [None])[0]
        if final_member is None:
            raise ValueError("cadenced_pbt_v1 manifest has no final-ranked member")
        final_worker = (final.get("workers") or {}).get(final_member, {})
        selections = [
            ("global_best_state", best.get("state_path"), "state",
             "best checkpoint according to the primary full-reference metric"),
            ("global_best_optimizer", best.get("optimizer_path"), "optimizer",
             "optimizer paired with the primary-metric global best"),
            ("global_best_scaler", best.get("scaler_path") or source_run / "checkpoints/global_best_scaler.pt", "scaler",
             "AMP scaler paired with the primary-metric global best"),
            ("final_best_state", source_run / str(final_member) / f"net_epoch-{final.get('epoch')}_state.pt",
             "state", "top-ranked population member at the final horizon"),
        ]
        for filename, recorded, kind, reason in selections:
            if not recorded:
                raise ValueError(f"cadenced_pbt_v1 is missing the required {filename} path")
            source = resolve_recorded_path(recorded, source_run=source_run)
            if not source.is_file():
                raise FileNotFoundError(source)
            metric = (best.get("metric_value") if filename.startswith("global_best")
                      else (final_worker.get("metrics") or {}).get(METRIC))
            epoch = best.get("epoch") if filename.startswith("global_best") else final.get("epoch")
            record = model_record(source, destination / f"{filename}.pt", role=filename,
                                  reason=reason, epoch=epoch, metric=metric, existing=existing)
            record["kind"] = kind
            record["member"] = best.get("member") if filename.startswith("global_best") else final_member
            models.append(record)
        bundle = {
            "selection": "global_best",
            "epoch": best.get("epoch"),
            "member": best.get("member"),
            "assets": [model for model in models if model["file"].startswith("global_best_")],
        }
        return models, bundle
    if method != "windowed_pbt_v2":
        return [], None
    models: list[dict[str, Any]] = []
    existing: dict[str, Path] = {}
    best = manifest.get("best", {})
    protected = manifest.get("protected_best", {})
    final = (manifest.get("generations") or [{}])[-1]
    final_member = (final.get("ranking") or [None])[0]
    final_worker = (final.get("workers") or {}).get(final_member, {})
    selections = [
        ("best_single", best.get("state_path"), best.get("epoch"), best.get("metric_value"),
         "lowest recorded individual metric"),
        ("protected_best", protected.get("state_path"), protected.get("epoch"),
         protected.get("single_epoch_metric"), "best protected window selection"),
        ("final_best", source_run / str(final_member) / f"net_epoch-{final.get('epoch')}_state.pt",
         final.get("epoch"), (final_worker.get("metrics") or {}).get(METRIC),
         "top-ranked member at the final horizon"),
    ]
    for role, recorded, epoch, metric, reason in selections:
        if not recorded or final_member is None and role == "final_best":
            continue
        source = resolve_recorded_path(recorded, source_run=source_run)
        if source.is_file():
            models.append(model_record(source, destination / f"{role}_state.pt", role=role,
                                       reason=reason, epoch=epoch, metric=metric, existing=existing))
    bundle = None
    if resume_bundle:
        if resume_bundle != "protected_best":
            raise ValueError("Only protected_best is supported for --resume-bundle")
        assets = {kind: protected.get(f"{kind}_path") for kind in ("state", "optimizer", "scaler")}
        exported = []
        for kind, recorded in assets.items():
            if not recorded:
                continue
            source = resolve_recorded_path(recorded, source_run=source_run)
            target = destination / f"protected_best_{kind}.pt"
            if kind == "state" and target.exists():
                digest = sha256(target)
            elif source.is_file():
                digest = sha256(source)
                if digest in existing:
                    os.link(existing[digest], target)
                else:
                    shutil.copy2(source, target)
                    existing[digest] = target
            else:
                continue
            exported.append({"kind": kind, "file": target.name, "source_path": str(source),
                             "sha256": digest, "size_bytes": target.stat().st_size})
        if exported:
            bundle = {"selection": "protected_best", "epoch": protected.get("epoch"), "assets": exported}
    return models, bundle


def write_readme(path: Path, published: dict[str, Any], provenance: dict[str, Any]) -> None:
    if published.get("strategy") == "cadenced_pbt_v1":
        best = published["best"]
        dynamics = published["pbt_dynamics"]
        selection = published.get("checkpoint_selection") or {}
        lines = [
            f"# {published['title']}", "",
            "A completed pretrained population-based training run with one-epoch generations, "
            "copy and learning-rate mutation opportunities after every eligible epoch, a two-epoch "
            "warm-up, and five population members.", "",
            f"- Status: `{published['status']}`",
            f"- Method: `{published['strategy']}`",
            f"- Training seed: `{published['seed']}`",
            f"- Canonical server path: `{published['source_run']}`",
            "- Generation length: `1 epoch`",
            "- Copy opportunity: `every epoch after warm-up`",
            "- LR mutation opportunity: `every epoch after warm-up`",
            "- Warm-up: `2 epochs`",
            "- Population: `5`",
            "- Initialization: `pretrained epoch 17`", "",
            "## Main results", "",
            f"- Final-10 current-best mean: `{published['final10_current_best_mean']}`",
            f"- Global best: `{best['metric_value']}`",
            f"- Global-best completed / Weaver epoch: `{published['global_best_completed_epoch']}` / `{best['epoch']}`",
            f"- Global-best member / LR: `{best['member']}` / `{best['lr']}`", "",
            "## PBT dynamics", "",
            f"- Exploit events: `{dynamics['exploit_count']}`",
            f"- LR mutations: `{dynamics['mutation_count']}`",
            f"- Copy-only events: `{dynamics['copy_only_count']}`",
            f"- Mutation factors: `×0.8` on `{dynamics['mutation_factors']['0.8']}` events; "
            f"`×1.2` on `{dynamics['mutation_factors']['1.2']}` events",
            f"- Initial-lineage collapse: completed epoch `{dynamics['lineage_collapse_completed_epoch']}`",
            "- Five unique live LRs remained present at every completed epoch.", "",
            "After completed epoch 5, every live member descended from the initial `lr_14e-6` ancestry. "
            "The plots describe recorded temporal relationships; they do not claim that an individual "
            "LR mutation caused a later performance change.", "",
            "## Checkpoint selection", "",
            "The global-best checkpoint is selected by the primary full-reference optimization metric. "
            "The final-best checkpoint is the top-ranked member at completed epoch 50; the two meanings "
            "are kept separate.", "",
        ]
        if selection and not selection.get("agrees_with_best_physics", True):
            lines.extend((
                "The supplementary physics report's best-physics checkpoint is not the primary PBT "
                "global-best checkpoint. The figures are retained as recorded and do not redefine the "
                "primary selection.", "",
            ))
        lines.extend((
            "## Contents", "",
            "`metrics.csv`, `lr_history.csv`, `decision_history.csv`, and `exploit_history.csv` are "
            "generated or copied from recorded evidence. `plots/` contains standalone presentation "
            "figures and verified physics outputs. `models/` contains explicitly selected release assets.", "",
            "## Provenance", "",
            f"Source manifest SHA256: `{provenance['manifest_sha256']}`", "",
        ))
        path.write_text("\n".join(lines))
        return
    best = published.get("best") or {}
    lines = [f"# {published['title']}", "", published["purpose"], "",
             f"- Status: `{published.get('status', 'unknown')}`",
             f"- Method: `{published.get('strategy', 'evaluation')}`",
             f"- Canonical server path: `{published['source_run']}`"]
    if best:
        lines.extend((f"- Best `{best.get('metric')}`: `{best.get('metric_value')}`",
                      f"- Best member / generation: `{best.get('member')}` / `{best.get('generation')}`"))
    lines.extend(("", "## Contents", "",
                  "The JSON and CSV files are generated from recorded evidence. `plots/` contains selected run figures; "
                  "`models/` contains only explicitly selected release assets.", "", "## Provenance", "",
                  f"Source manifest SHA256: `{provenance.get('manifest_sha256', 'not available')}`", ""))
    path.write_text("\n".join(lines))


def rst_page(published: dict[str, Any], provenance: dict[str, Any], plot_files: list[str]) -> str:
    if published.get("strategy") == "cadenced_pbt_v1":
        title = published["title"]
        best = published["best"]
        final = published["final_best"]
        dynamics = published["pbt_dynamics"]
        lines = [
            title, "=" * len(title), "",
            "Completed pretrained PBT with one-epoch generations, copy and deterministic LR-mutation "
            "opportunities after every eligible epoch, a two-epoch warm-up, and five population members.", "",
            "Primary results", "---------------", "",
            ".. list-table::", "   :header-rows: 1", "   :widths: 45 55", "",
            "   * - Quantity", "     - Recorded value",
            f"   * - Final-10 current-best full-reference mean", f"     - ``{published['final10_current_best_mean']}``",
            f"   * - Global-best full-reference score", f"     - ``{best['metric_value']}``",
            f"   * - Global-best completed / Weaver epoch", f"     - ``{published['global_best_completed_epoch']}`` / ``{best['epoch']}``",
            f"   * - Global-best member / LR", f"     - ``{best['member']}`` / ``{best['lr']}``",
            f"   * - Final-best member / score", f"     - ``{final['member']}`` / ``{final['metric_value']}``", "",
            "Method and dynamics", "-------------------", "",
            "* Generation length: one epoch.",
            "* Copy and LR mutation opportunity: every epoch after the two-epoch warm-up.",
            "* Initialization: pretrained epoch 17; training seed ``12345``.",
            f"* Exploit / mutation / copy-only counts: ``{dynamics['exploit_count']}`` / "
            f"``{dynamics['mutation_count']}`` / ``{dynamics['copy_only_count']}``.",
            f"* Initial-lineage collapse: completed epoch ``{dynamics['lineage_collapse_completed_epoch']}``.",
            "* Five unique live learning rates remained present at every completed epoch.", "",
            "The figures describe recorded temporal relationships and do not claim that a particular "
            "learning-rate mutation caused a later performance improvement.", "",
            "Performance progression", "-----------------------", "",
            f".. image:: ../../published/experiments/{published['slug']}/plots/01_performance_progression.png",
            f"   :alt: {title} — performance progression", "",
            "Population learning rates", "-------------------------", "",
            f".. image:: ../../published/experiments/{published['slug']}/plots/02_population_lr_trajectories.png",
            f"   :alt: {title} — population learning-rate trajectories", "",
            "Post-boundary LR changes take effect in the following training epoch.", "",
            "Global-best LR lineage", "----------------------", "",
            f".. image:: ../../published/experiments/{published['slug']}/plots/03_global_best_lr_lineage.png",
            f"   :alt: {title} — global-best learning-rate lineage", "",
            "Physics performance", "-------------------", "",
            f".. image:: ../../published/experiments/{published['slug']}/plots/physics_performance.png",
            f"   :alt: {title} — physics performance", "",
            f".. image:: ../../published/experiments/{published['slug']}/plots/background_efficiency_curves.png",
            f"   :alt: {title} — background efficiency curves", "",
            "The physics figures are the verified recorded outputs. Their best-physics checkpoint is not "
            "the primary full-reference PBT global-best checkpoint; the primary selection is unchanged.", "",
            "Metrics and history", "-------------------", "",
            f"* :download:`Recorded metrics <../../published/experiments/{published['slug']}/metrics.csv>`",
            f"* :download:`Learning-rate history <../../published/experiments/{published['slug']}/lr_history.csv>`",
            f"* :download:`Boundary decisions <../../published/experiments/{published['slug']}/decision_history.csv>`",
            f"* :download:`Applied exploit history <../../published/experiments/{published['slug']}/exploit_history.csv>`", "",
            "Models", "------", "",
        ]
        for model in provenance["models"]:
            lines.append(f"* ``{model['file']}`` — {model['selection_reason']}, SHA256 ``{model['sha256']}``")
        lines.extend((
            "", "Model files are local release assets and are intentionally excluded from the Pages build.", "",
            "Provenance and reproducibility", "------------------------------", "",
            f"* Canonical server path: ``{published['source_run']}``",
            f"* Source commit: ``{published['source_commit']}``",
            f"* Source manifest SHA256: ``{provenance['manifest_sha256']}``",
            f"* :download:`Resolved configuration <../../published/experiments/{published['slug']}/config.yaml>`",
            f"* :download:`Publication provenance <../../published/experiments/{published['slug']}/provenance.json>`",
        ))
        return "\n".join(lines) + "\n"
    title = published["title"]
    best = published.get("best") or {}
    lines = [title, "=" * len(title), "", published["purpose"], "", "Result summary", "--------------", "",
             f"* Status: ``{published.get('status', 'unknown')}``",
             f"* Method: ``{published.get('strategy', 'evaluation')}``",
             f"* Seed: ``{published.get('seed', 'not recorded')}``",
             f"* Horizon: ``{published.get('horizon', 'not recorded')}``"]
    if best:
        lines.append(f"* Best recorded metric: ``{best.get('metric_value')}`` ({best.get('metric')})")
    baseline = published.get("baseline") or {}
    if baseline:
        lines.append(f"* Recorded baseline metric: ``{baseline.get('metric_value')}``")
    if published.get("best_improvement_vs_baseline") is not None:
        lines.append(f"* Relative improvement against the recorded baseline: ``{100 * published['best_improvement_vs_baseline']:.4f}%``")
    if plot_files:
        lines.extend(("", "Key plots", "---------", ""))
        preferred = [p for p in plot_files if Path(p).name in {
            "01_performance_progression.png", "02_learning_rate_evolution.png"}]
        for plot in (preferred or plot_files)[:3]:
            lines.extend((f".. image:: ../../published/experiments/{published['slug']}/plots/{plot}",
                          f"   :alt: {title} — {Path(plot).stem.replace('_', ' ')}", ""))
    lines.extend(("", "Metrics and history", "-------------------", "",
                  f"* :download:`Recorded metrics <../../published/experiments/{published['slug']}/metrics.csv>`" if published.get("has_metrics") else "* No run-level metrics CSV was recorded.",
                  f"* :download:`Learning-rate history <../../published/experiments/{published['slug']}/lr_history.csv>`" if published.get("has_lr_history") else "* No learning-rate history was recorded.",
                  f"* :download:`Exploit history <../../published/experiments/{published['slug']}/exploit_history.csv>`" if published.get("has_exploit_history") else "* This run has no exploit history.",
                  "", "Models", "------", ""))
    models = provenance.get("models", [])
    if models:
        for model in models:
            lines.append(f"* ``{model['file']}`` — epoch {model.get('epoch')}, {model['selection_reason']}, SHA256 ``{model['sha256']}``")
        lines.append("\nModel files are prepared as release assets and are intentionally excluded from the Pages build.")
    else:
        lines.append("No model checkpoint is included in this publication bundle.")
    lines.extend(("", "Provenance and reproducibility", "------------------------------", "",
                  f"* Canonical server path: ``{published['source_run']}``",
                  f"* Source commit: ``{published.get('source_commit', 'not recorded')}``",
                  f"* Source manifest SHA256: ``{provenance.get('manifest_sha256', 'not available')}``"))
    if published.get("has_config"):
        lines.append(f"* :download:`Resolved configuration <../../published/experiments/{published['slug']}/config.yaml>`")
    lines.append(f"* :download:`Publication provenance <../../published/experiments/{published['slug']}/provenance.json>`")
    return "\n".join(lines) + "\n"


def export(args: argparse.Namespace) -> Path:
    source_run = Path(args.run).resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    related = [Path(path).resolve() for path in args.related_run]
    run_dirs = [source_run, *related]
    for run in run_dirs:
        if not run.is_dir():
            raise FileNotFoundError(run)
    output_root = Path(args.output_root).resolve()
    docs_dir = Path(args.docs_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = source_run / "manifest.json"
    summary_path = source_run / "summary.json"
    config_source = next((source_run / name for name in ("resolved_config.yaml", "config.yaml")
                          if (source_run / name).is_file()), None)
    manifest = load_json(manifest_path)
    source_summary = load_json(summary_path)
    evidence_paths = [manifest_path, summary_path, source_run / "metrics.csv", source_run / "events.jsonl"]
    if config_source:
        evidence_paths.append(config_source)
    evidence_paths.extend(sorted((source_run / "plots").rglob("*.png"))
                          if (source_run / "plots").is_dir() else [])
    before = {path: sha256(path) for path in evidence_paths if path.is_file()}
    method = manifest.get("method", source_summary.get("method", "checkpoint_evaluation"))
    with tempfile.TemporaryDirectory(prefix=f".{args.slug}-", dir=output_root) as temporary:
        target = Path(temporary)
        plots_dir = target / "plots"
        models_dir = target / "models"
        plots_dir.mkdir()
        if config_source:
            shutil.copy2(config_source, target / "config.yaml")
        has_metrics = copy_csvs(run_dirs, target / "metrics.csv")
        plots = copy_plots(source_run, plots_dir)
        cadenced = None
        has_decisions = False
        if method == "cadenced_pbt_v1":
            try:
                from scripts.publish.cadenced_pbt_v1 import build as build_cadenced
            except ModuleNotFoundError:  # Direct execution from scripts/publish.
                from cadenced_pbt_v1 import build as build_cadenced
            cadenced = build_cadenced(source_run, target, manifest)
            has_lr = True
            has_decisions = True
            has_exploit = True
            plots = sorted(set([*plots, *cadenced["plots"]]))
        else:
            has_lr = write_lr_history(target / "metrics.csv", target / "lr_history.csv")
            has_exploit = write_exploit_history(manifest, target / "exploit_history.csv")
        models, resume = export_models(source_run, manifest, models_dir, resume_bundle=args.resume_bundle)
        if not models and models_dir.exists():
            models_dir.rmdir()
        run_meta = manifest.get("run", {})
        final = (manifest.get("generations") or [{}])[-1]
        horizon = len(manifest.get("generations", [])) or len(source_summary.get("results", [])) or None
        recorded_best = source_summary.get("best") or manifest.get("best")
        if not recorded_best and source_summary.get("results"):
            result = min(source_summary["results"], key=lambda item: item.get(METRIC, float("inf")))
            recorded_best = {"epoch": result.get("epoch"), "metric": METRIC,
                             "metric_value": result.get(METRIC), "metrics": result}
        published = {
            "schema_version": 1,
            "slug": args.slug,
            "title": args.title,
            "purpose": args.purpose or "Curated publication of recorded experiment evidence.",
            "source_run": display_path(source_run),
            "related_runs": [display_path(path) for path in related],
            "status": manifest.get("status", source_summary.get("status", "completed")),
            "seed": run_meta.get("seed"),
            "horizon": horizon,
            "strategy": method,
            "method": method,
            "metric": source_summary.get("metric", {"name": METRIC, "mode": "min"}),
            "best": compact_selection(recorded_best),
            "final_best": compact_selection(source_summary.get("final_best")),
            "baseline": compact_selection(source_summary.get("baseline")),
            "best_improvement_vs_baseline": source_summary.get("best_improvement_vs_baseline"),
            "working_point_metrics": compact_metrics(source_summary.get("best", {})),
            "source_commit": (manifest.get("git") or run_meta.get("git") or {}).get("commit"),
            "final_generation": final.get("index"),
            "has_config": config_source is not None,
            "has_metrics": has_metrics,
            "has_lr_history": has_lr,
            "has_decision_history": has_decisions,
            "has_exploit_history": has_exploit,
            "plots": plots,
        }
        if cadenced:
            schedule = run_meta.get("schedule") or {}
            final_best = published.get("final_best") or {}
            published.update({
                "initialization_type": "pretrained",
                "generation_length_epochs": ((schedule.get("training_interval") or {}).get(
                    "weaver_epochs_per_generation", 1)),
                "adaptation_interval_epochs": ((schedule.get("exploit_interval") or {}).get("epochs", 1)),
                "population_size": len(manifest.get("members") or {}),
                "final10_current_best_mean": cadenced["final10_current_best_mean"],
                "global_best_completed_epoch": int((published.get("best") or {}).get("generation", -1)) + 1,
                "global_best": published.get("best"),
                "pbt_dynamics": {
                    "exploit_count": cadenced["exploit_count"],
                    "mutation_count": cadenced["mutation_count"],
                    "copy_only_count": cadenced["copy_only_count"],
                    "mutation_factors": cadenced["mutation_factors"],
                    "lineage_collapse_completed_epoch": cadenced["lineage_collapse_completed_epoch"],
                    "unique_lr_count_min": min(cadenced["unique_lr_counts"]),
                    "unique_lr_count_max": max(cadenced["unique_lr_counts"]),
                },
                "scientific_endpoint": source_summary.get("scientific_endpoint"),
                "checkpoint_selection": source_summary.get("checkpoint_selection"),
            })
            if final_best and final_best.get("epoch") is None:
                final_best["epoch"] = final.get("epoch")
        provenance = {
            "schema_version": 1,
            "source_run": published["source_run"],
            "related_runs": published["related_runs"],
            "manifest_sha256": before.get(manifest_path),
            "summary_sha256": before.get(summary_path),
            "config_sha256": sha256(config_source) if config_source else None,
            "events_sha256": before.get(source_run / "events.jsonl"),
            "metrics_sha256": before.get(source_run / "metrics.csv"),
            "source_commit": published["source_commit"],
            "source_git": manifest.get("git") or run_meta.get("git"),
            "dataset": manifest.get("datasets") or run_meta.get("datasets"),
            "dataset_fingerprints": ((manifest.get("datasets") or run_meta.get("datasets") or {}).get(
                "fingerprints")),
            "starting_checkpoint": run_meta.get("checkpoint") or manifest.get("checkpoint"),
            "starting_optimizer": run_meta.get("optimizer_checkpoint") or manifest.get("optimizer_checkpoint"),
            "fingerprint": manifest.get("fingerprint"),
            "config_fingerprint": manifest.get("fingerprint"),
            "models": models,
            "resume_bundle": resume,
            "path_compatibility": MOVED_PREFIXES,
        }
        if cadenced:
            exporter_path = Path(__file__).resolve()
            helper_path = exporter_path.with_name("cadenced_pbt_v1.py")
            try:
                publisher_commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
                    capture_output=True, text=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                publisher_commit = None
            provenance.update({
                "source_run_absolute": str(source_run),
                "publication": {
                    "schema_version": 2,
                    "repository_commit": publisher_commit,
                    "scripts": [
                        {"path": display_path(exporter_path), "sha256": sha256(exporter_path)},
                        {"path": display_path(helper_path), "sha256": sha256(helper_path)},
                    ],
                },
                "checkpoint_selection": source_summary.get("checkpoint_selection"),
            })
        (target / "summary.json").write_text(json.dumps(published, indent=2, sort_keys=True) + "\n")
        (target / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        write_readme(target / "README.md", published, provenance)
        final_target = output_root / args.slug
        if final_target.exists():
            shutil.rmtree(final_target)
        shutil.move(str(target), final_target)
    after = {path: sha256(path) for path in before}
    if before != after:
        raise RuntimeError("Source evidence changed during export")
    for model in models:
        source = resolve_recorded_path(model["source_path"], source_run=source_run)
        if sha256(source) != model["sha256"]:
            raise RuntimeError(f"Source model changed during export: {source}")
    docs_page = docs_dir / f"{args.slug}.rst"
    docs_page.write_text(rst_page(published, provenance, plots))
    if args.release_tag and models:
        model_glob = f"published/experiments/{args.slug}/models/*"
        title = f"{args.title} models"
        note = ("Global-best resume assets and final-best state; hashes are in provenance.json."
                if method == "cadenced_pbt_v1"
                else "Selected checkpoints and one protected resume bundle; hashes are in provenance.json.")
        create = f"gh release create {shlex.quote(args.release_tag)} --title {shlex.quote(title)} --notes {shlex.quote(note)}"
        upload = f"gh release upload {shlex.quote(args.release_tag)} {model_glob} --clobber"
        (final_target / "release-command.txt").write_text(create + "\n" + upload + "\n")
        if args.upload_release_assets:
            if shutil.which("gh") is None:
                raise RuntimeError("gh is not installed; export completed but release upload was not run")
            exists = subprocess.run(["gh", "release", "view", args.release_tag], cwd=PROJECT_ROOT,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not exists:
                subprocess.run(["gh", "release", "create", args.release_tag, "--title", title,
                                "--notes", note], cwd=PROJECT_ROOT, check=True)
            files = [str(path) for path in sorted((final_target / "models").iterdir())]
            subprocess.run(["gh", "release", "upload", args.release_tag, *files, "--clobber"],
                           cwd=PROJECT_ROOT, check=True)
    return final_target


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("run")
    result.add_argument("--slug", required=True, type=lambda value: value if re.fullmatch(r"[a-z0-9][a-z0-9-]*", value) else (_ for _ in ()).throw(argparse.ArgumentTypeError("invalid slug")))
    result.add_argument("--title", required=True)
    result.add_argument("--purpose")
    result.add_argument("--related-run", action="append", default=[])
    result.add_argument("--output-root", default=PROJECT_ROOT / "published/experiments")
    result.add_argument("--docs-dir", default=PROJECT_ROOT / "docs/experiments")
    result.add_argument("--resume-bundle", choices=("protected_best",))
    result.add_argument("--release-tag")
    result.add_argument("--upload-release-assets", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.upload_release_assets and not args.release_tag:
        raise SystemExit("--upload-release-assets requires --release-tag")
    print(export(args))


if __name__ == "__main__":
    main()
