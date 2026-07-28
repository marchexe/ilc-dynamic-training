#!/usr/bin/env python3
"""Write a compact JSON metrics summary from a PBT manifest."""

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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
FIXED_EFFICIENCY_POINTS = {
    "b": (0.5, 0.8, 0.9),
    "c": (0.5, 0.8, 0.9),
}
SHOWCASE_WORKING_POINTS = (
    ("b", 0.80, "c"),
    ("b", 0.80, "d"),
    ("b", 0.90, "c"),
    ("b", 0.90, "d"),
    ("c", 0.50, "b"),
    ("c", 0.50, "d"),
    ("c", 0.80, "b"),
    ("c", 0.80, "d"),
)
SHOWCASE_METRIC_DEFINITION = {
    "display_name": "Average mistag at fixed working points",
    "formula": "arithmetic mean of mistag percentages at b-tag efficiencies 0.80/0.90 and c-tag efficiencies 0.50/0.80",
    "mode": "min",
    "unit": "percent",
    "note": "Lower is better. This is intended for presentation plots; the PBT ranking score is kept separately for reproducibility.",
}
PAIR_SCORE_METRICS = {
    "bc": "validation_bkg_rejection_bc_score",
    "bd": "validation_bkg_rejection_bd_score",
    "cb": "validation_bkg_rejection_cb_score",
    "cd": "validation_bkg_rejection_cd_score",
}
CORE_METRICS = (
    "validation_accuracy",
    "validation_auc",
    "validation_loss",
    "validation_bkg_rejection_score",
    "validation_b_tag_rejection_score",
    "validation_c_tag_rejection_score",
    "validation_working_point_mistag_percent",
    "validation_ctag_reference_mistag_percent",
)


METRIC_DEFINITIONS = {
    "validation_bkg_rejection_score": {
        "display_name": "PBT objective: mean ln(BGrej), all pairs",
        "formula": "mean(log(BGrej_pair(eff))) over pairs {bc, bd, cb, cd} and signal efficiencies {0.2, ..., 1.0}",
        "pair_definition": "xy means x-tag signal efficiency with y-flavour background rejection",
        "bgrej_definition": "BGrej = 1 / background_efficiency at a fixed signal/tag efficiency",
        "note": "This is an internal scalar objective for ranking PBT workers, not a standard publication metric. Fixed-efficiency mistag percentages are easier to read physically.",
    },
    "validation_b_tag_rejection_score": {
        "display_name": "PBT objective: mean ln(BGrej), b-tag pairs",
        "formula": "mean(log(BGrej_pair(eff))) over pairs {bc, bd} and signal efficiencies {0.2, ..., 1.0}",
        "pair_definition": "bc means b-tag efficiency with c-background rejection; bd means b-tag efficiency with d-background rejection",
        "bgrej_definition": "BGrej = 1 / background_efficiency at a fixed signal/tag efficiency",
        "note": "Better aligned with b-tag optimization than the all-pair objective.",
    },
    "validation_c_tag_rejection_score": {
        "display_name": "PBT objective: mean ln(BGrej), c-tag pairs",
        "formula": "mean(log(BGrej_pair(eff))) over pairs {cb, cd} and signal efficiencies {0.2, ..., 1.0}",
        "pair_definition": "cb means c-tag efficiency with b-background rejection; cd means c-tag efficiency with d-background rejection",
        "bgrej_definition": "BGrej = 1 / background_efficiency at a fixed signal/tag efficiency",
        "note": "Better aligned with c-tag optimization than the all-pair objective.",
    },
    "validation_working_point_mistag_percent": {
        "display_name": "Selection metric: average mistag at reference working points",
        "formula": "arithmetic mean of mistag percentages for b-tag 0.80/0.90 and c-tag 0.50/0.80 working points",
        "pair_definition": "xy means x-tag signal efficiency with y-flavour background rejection",
        "bgrej_definition": "mistag [%] = 100 / BGrej at a fixed signal/tag efficiency",
        "note": "Lower is better. This is the recommended fine-tuning selection metric for the current study.",
    },
    "validation_ctag_reference_mistag_percent": {
        "display_name": "C-tag reference metric: average mistag at 50/80% efficiency",
        "formula": "arithmetic mean of b-background and d-background mistag percentages at c-tag efficiencies 0.50 and 0.80",
        "pair_definition": "cb/cd mean c-tag signal efficiency with b/d-flavour background rejection",
        "bgrej_definition": "mistag [%] = 100 / BGrej at a fixed c-tag efficiency",
        "note": "Lower is better. This matches the c-tag 50% and 80% reference tables.",
    },
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


def worker_lr(worker, fallback=None):
    if worker.get("lr") is not None:
        return float(worker["lr"])
    command = worker.get("command") or []
    if "--start-lr" in command:
        index = command.index("--start-lr")
        if index + 1 < len(command):
            return float(command[index + 1])
    return float(fallback) if fallback is not None else None


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



def rejection_value(metrics, tag, eff, background):
    lookup = metrics.get("validation_bkg_rejection_at_eff_lookup") or {}
    row = lookup.get(f"{tag}_tag_eff_{eff:.2f}") or {}
    value = row.get(f"{background}_bkg_rejection")
    if value is not None:
        return float(value)

    curves = metrics.get("validation_bkg_rejection_at_eff") or {}
    efficiencies = curves.get("efficiencies") or []
    pairs = curves.get("pairs") or {}
    if eff not in efficiencies:
        return None
    index = efficiencies.index(eff)
    pair = f"{tag}{background}"
    values = pairs.get(pair) or []
    return float(values[index]) if index < len(values) else None


def fixed_efficiency_rows(metrics, tag, efficiencies):
    rows = []
    for eff in efficiencies:
        backgrounds = {}
        for background in TAG_BACKGROUNDS[tag]:
            rejection = rejection_value(metrics, tag, eff, background)
            backgrounds[background] = {
                "background_rejection": rejection,
                "mistag_percent": None if rejection is None or rejection <= 0 else 100.0 / rejection,
            }
        rows.append({"tag_efficiency": eff, "backgrounds": backgrounds})
    return rows


def showcase_metrics(metrics):
    points = []
    for tag, eff, background in SHOWCASE_WORKING_POINTS:
        rejection = rejection_value(metrics, tag, eff, background)
        mistag = None if rejection is None or rejection <= 0 else 100.0 / rejection
        points.append(
            {
                "tag": tag,
                "tag_efficiency": eff,
                "background": background,
                "background_rejection": rejection,
                "mistag_percent": mistag,
            }
        )
    values = [point["mistag_percent"] for point in points if point["mistag_percent"] is not None]
    return {
        "average_mistag_percent": sum(values) / len(values) if values else None,
        "working_points": points,
    }


def pair_objective_components(metrics):
    return {
        pair: metric_value({"metrics": metrics}, metric_name)
        for pair, metric_name in PAIR_SCORE_METRICS.items()
    }


def compact_metrics(metrics):
    return {name.replace("validation_", ""): metrics.get(name) for name in CORE_METRICS if name in metrics}


def worker_summary(name, worker, metric, fallback_lr=None):
    metrics = worker.get("metrics") or {}
    return {
        "member": name,
        "status": worker.get("status"),
        "lr": worker_lr(worker, fallback_lr),
        "objective_value": metric_value(worker, metric),
        "core_metrics": compact_metrics(metrics),
        "pair_objective_components": pair_objective_components(metrics),
        "showcase_metrics": showcase_metrics(metrics),
        "fixed_efficiency_metrics": {
            tag: fixed_efficiency_rows(metrics, tag, efficiencies)
            for tag, efficiencies in FIXED_EFFICIENCY_POINTS.items()
        },
    }


def final_completed_generation(manifest):
    generations = completed_generations(manifest)
    return generations[-1] if generations else None


def best_physics_for_generation(generation):
    candidates = []
    for member_name, worker in generation.get("workers", {}).items():
        metrics = worker.get("metrics") or {}
        score = showcase_metrics(metrics)["average_mistag_percent"]
        if score is not None:
            candidates.append((score, member_name, worker))
    if not candidates:
        return None
    score, member_name, worker = min(candidates, key=lambda item: item[0])
    return {
        "generation": generation.get("index"),
        "member": member_name,
        "avg_mistag_percent": score,
        "lr": worker_lr(worker),
        "exact_stat_uncertainty_available": bool((worker.get("metrics") or {}).get("validation_bkg_rejection_at_eff_counts")),
    }


def build_training_logic_summary(manifest):
    rows = [row for generation in completed_generations(manifest) for row in [best_physics_for_generation(generation)] if row]
    if not rows:
        return None
    start = rows[0]
    best = min(rows, key=lambda row: row["avg_mistag_percent"])
    final = rows[-1]
    return {
        "metric": SHOWCASE_METRIC_DEFINITION,
        "start": start,
        "best": best,
        "final": final,
        "delta_start_to_best_percent": best["avg_mistag_percent"] - start["avg_mistag_percent"],
        "delta_best_to_final_percent": final["avg_mistag_percent"] - best["avg_mistag_percent"],
        "exact_stat_uncertainty_available": all(row["exact_stat_uncertainty_available"] for row in (start, best, final)),
        "uncertainty_note": "Exact binomial/bootstrap uncertainty needs background pass/total counters; current manifests store only rejection values."
        if not all(row["exact_stat_uncertainty_available"] for row in (start, best, final))
        else "Exact uncertainty counters are available in the selected checkpoints.",
    }


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
        lr = worker_lr(worker, (manifest.get("members", {}).get(member_name, {}) or {}).get("lr"))
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
        best_generation = next(
            (generation for generation in completed_generations(manifest) if generation.get("index") == best.get("generation")),
            None,
        )
        best_worker = (best_generation.get("workers", {}).get(best.get("member"), {}) if best_generation else {})
        global_best = {
            "generation": best.get("generation"),
            "member": best.get("member"),
            "epoch": best.get("epoch"),
            "lr": best.get("lr"),
            "metric": best.get("metric", metric),
            "metric_value": best.get("metric_value"),
            "state_path": best.get("state_path"),
            "details": worker_summary(
                best.get("member"),
                best_worker,
                metric,
                (manifest.get("members", {}).get(best.get("member"), {}) or {}).get("lr"),
            ) if best_worker else None,
        }

    final_generation_record = final_completed_generation(manifest)
    final_generation = None
    if final_generation_record:
        final_generation = {
            "generation": final_generation_record.get("index"),
            "epoch": final_generation_record.get("epoch"),
            "workers": [
                worker_summary(
                    name,
                    worker,
                    metric,
                    (manifest.get("members", {}).get(name, {}) or {}).get("lr"),
                )
                for name, worker in sorted(final_generation_record.get("workers", {}).items())
            ],
        }

    plot_layout = {
        "report_plots": {
            "physics_performance_plot": "report/physics_performance.png",
            "training_diagnostics_plot": "report/training_diagnostics.png",
            "btag_mistag_table_csv": "report/btag_mistag_tables.csv",
            "ctag_mistag_table_csv": "report/ctag_mistag_tables.csv",
        },
        "diagnostic_plots": {
            "background_rejection_curves_plot": "diagnostics/background_rejection_curves.png",
            "background_efficiency_curves_plot": "diagnostics/background_efficiency_curves.png",
            "btag_background_efficiency_plot": "diagnostics/btag_background_efficiency_vs_training_size.png",
            "selection_timeline_plot": "diagnostics/selection_timeline.png",
        },
    }
    grouped_plots = {group: {} for group in plot_layout}
    for group, filenames in plot_layout.items():
        for key, filename in filenames.items():
            value = manifest.get(key)
            if value:
                candidate = Path(value)
                relative_candidate = manifest_path.parent / candidate
                if candidate.exists() or (not candidate.is_absolute() and relative_candidate.exists()):
                    grouped_plots[group][key] = value
                    continue
            path = manifest_path.parent / "plots" / filename
            if not path.exists():
                path = manifest_path.parent / "plots" / Path(filename).name
            if path.exists():
                grouped_plots[group][key] = str(path)
    plots = {
        key: value
        for group_plots in grouped_plots.values()
        for key, value in group_plots.items()
    }

    return {
        "schema_version": 1,
        "experiment": manifest.get("experiment", manifest_path.parent.name),
        "status": manifest.get("status"),
        "metric": {
            "name": metric,
            "mode": pbt.get("mode", "max"),
            "definition": METRIC_DEFINITIONS.get(metric),
        },
        "showcase_metric": SHOWCASE_METRIC_DEFINITION,
        "strategy": pbt.get("strategy"),
        "inputs": {
            "dataset": shared.get("dataset"),
            "checkpoint": shared.get("checkpoint"),
            "data_config": shared.get("data_config"),
            "network_config": shared.get("network_config"),
        },
        "plots": plots,
        "report_plots": grouped_plots["report_plots"],
        "diagnostic_plots": grouped_plots["diagnostic_plots"],
        "global_best": global_best,
        "training_logic": build_training_logic_summary(manifest),
        "generations": generations,
        "final_generation": final_generation,
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
