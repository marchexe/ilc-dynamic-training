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
    if manifest.get("method") != "windowed_pbt_v2":
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
            "01_performance_progression.png", "02_learning_rate_evolution.png",
            "physics_performance.png", "mistag_score_evolution.png", "learning_rate_lineage.png"}]
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
    before = {path: sha256(path) for path in (manifest_path, summary_path) if path.is_file()}
    with tempfile.TemporaryDirectory(prefix=f".{args.slug}-", dir=output_root) as temporary:
        target = Path(temporary)
        plots_dir = target / "plots"
        models_dir = target / "models"
        plots_dir.mkdir()
        if config_source:
            shutil.copy2(config_source, target / "config.yaml")
        has_metrics = copy_csvs(run_dirs, target / "metrics.csv")
        has_lr = write_lr_history(target / "metrics.csv", target / "lr_history.csv")
        has_exploit = write_exploit_history(manifest, target / "exploit_history.csv")
        plots = copy_plots(source_run, plots_dir)
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
            "strategy": manifest.get("method", source_summary.get("method", "checkpoint_evaluation")),
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
            "has_exploit_history": has_exploit,
            "plots": plots,
        }
        provenance = {
            "schema_version": 1,
            "source_run": published["source_run"],
            "related_runs": published["related_runs"],
            "manifest_sha256": before.get(manifest_path),
            "summary_sha256": before.get(summary_path),
            "config_sha256": sha256(config_source) if config_source else None,
            "source_commit": published["source_commit"],
            "dataset": manifest.get("datasets") or run_meta.get("datasets"),
            "starting_checkpoint": run_meta.get("checkpoint") or manifest.get("checkpoint"),
            "fingerprint": manifest.get("fingerprint"),
            "models": models,
            "resume_bundle": resume,
            "path_compatibility": MOVED_PREFIXES,
        }
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
    docs_page = docs_dir / f"{args.slug}.rst"
    docs_page.write_text(rst_page(published, provenance, plots))
    if args.release_tag and models:
        model_glob = f"published/experiments/{args.slug}/models/*"
        title = f"{args.title} models"
        note = "Selected checkpoints and one protected resume bundle; hashes are in provenance.json."
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
