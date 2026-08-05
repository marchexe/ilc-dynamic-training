#!/usr/bin/env python3
"""Render NIGHTLY_RESULT.md and diagnostic plots from a proxy-audit run
directory's summary.json / checkpoint_metrics.csv.

Must run safely on partial results: a run interrupted mid-way still leaves
a usable summary.json (written once, at the end of run_proxy_audit.py) and
a checkpoint_metrics.csv with whatever rows completed before interruption
-- this script only ever reads what's actually on disk and reports
"unavailable"/"incomplete" for anything missing, never fabricates it.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

VERDICTS = (
    "50k proxy supported for limited control use",
    "50k proxy promising but evidence insufficient",
    "50k proxy ranking unreliable",
    "validation data integrity failed",
    "experiment incomplete",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def load_json(path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def load_csv_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(value):
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if result == result else None  # excludes NaN
    except (TypeError, ValueError):
        return None


def determine_verdict(summary, rows):
    """Maps the audit's measured statistics onto the task's fixed verdict
    vocabulary. Deliberately conservative: ties or ambiguous evidence fall
    to a weaker verdict rather than the strongest one that could plausibly
    be argued -- see the task's scientific rules (never claim proxy
    validity from one correlation coefficient, never oversell a weak
    result)."""
    checkpoints_distinct = summary.get("checkpoints_distinct", 0)
    control_completed = sum(1 for row in rows if row.get("control_proxy_50k_status") == "completed")
    full_completed = sum(1 for row in rows if row.get("full_validation_status") == "completed")

    if checkpoints_distinct == 0 or control_completed == 0 or full_completed == 0:
        return "experiment incomplete", "No checkpoint completed evaluation on both tiers."

    proxy_vs_full = summary.get("proxy_vs_full") or {}
    sanity = summary.get("proxy_sanity_check") or {}

    for tier_key, tier_label in (
        ("control_proxy_50k_class_balance", "control_proxy_50k"),
        ("full_validation_class_balance", "full_validation"),
    ):
        fractions = ((sanity.get(tier_key) or {}).get("fraction_by_flavor") or {}).values()
        if fractions and max(fractions) > 0.6:
            return "validation data integrity failed", f"{tier_label} truth-class proportions are severely imbalanced (max class fraction > 60%)."
    if (sanity.get("train_overlap") or {}).get("train800k_referenced_anywhere_in_manifest"):
        return "validation data integrity failed", "train800k is referenced in the proxy-validation manifest -- possible train/validation overlap."

    if control_completed < 6 or full_completed < 6:
        return "experiment incomplete", f"Only {control_completed} control_proxy_50k / {full_completed} full_validation evaluations completed (fewer than the planned 12)."

    if proxy_vs_full.get("insufficient_evidence"):
        return "50k proxy promising but evidence insufficient", "Fewer than 3 checkpoints had paired finite metrics on both tiers -- correlation is not meaningful."

    spearman = (proxy_vs_full.get("pearson_spearman") or {}).get("spearman_rho")
    pairwise = (proxy_vs_full.get("pairwise_direction_agreement") or {}).get("agreement_fraction")
    best_agrees = (proxy_vs_full.get("best_checkpoint_agreement") or {}).get("agrees")

    if spearman is None or pairwise is None:
        return "50k proxy promising but evidence insufficient", "Correlation could not be computed (scipy unavailable or too few paired points)."

    if spearman >= 0.8 and pairwise >= 0.8 and best_agrees:
        return "50k proxy supported for limited control use", f"Spearman rho={spearman:.3f}, pairwise agreement={pairwise:.1%}, best-checkpoint agreement=True."
    if spearman >= 0.5 or pairwise >= 0.6:
        return "50k proxy promising but evidence insufficient", f"Spearman rho={spearman:.3f}, pairwise agreement={pairwise:.1%} -- positive but below the threshold for a supported-use verdict."
    return "50k proxy ranking unreliable", f"Spearman rho={spearman:.3f}, pairwise agreement={pairwise:.1%} -- too weak to trust the proxy for ranking."


def build_plots(run_dir, summary, rows):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"status": "unavailable", "reason": f"matplotlib import failed: {error}"}

    paired = [
        (row["checkpoint_id"], _float(row.get("control_proxy_50k_working_point_mistag_percent")), _float(row.get("full_validation_working_point_mistag_percent")))
        for row in rows
    ]
    paired = [item for item in paired if item[1] is not None and item[2] is not None]

    written = {}

    if paired:
        fig, ax = plt.subplots(figsize=(6, 6))
        xs = [item[1] for item in paired]
        ys = [item[2] for item in paired]
        ax.scatter(xs, ys, color="#4c78a8")
        lo, hi = min(xs + ys), max(xs + ys)
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#999999", linewidth=1, label="y = x")
        ax.set_xlabel("control_proxy_50k working-point mistag %")
        ax.set_ylabel("full_validation working-point mistag %")
        ax.set_title("Proxy vs. full-validation metric, per checkpoint")
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "proxy_vs_full_scatter.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written["proxy_vs_full_scatter"] = str(path)

    if len(paired) >= 2:
        control_rank = {name: rank for rank, (name, _, _) in enumerate(sorted(paired, key=lambda item: item[1]))}
        full_rank = {name: rank for rank, (name, _, _) in enumerate(sorted(paired, key=lambda item: item[2]))}
        fig, ax = plt.subplots(figsize=(6, 6))
        xs = [control_rank[name] for name, _, _ in paired]
        ys = [full_rank[name] for name, _, _ in paired]
        ax.scatter(xs, ys, color="#59a14f")
        n = len(paired)
        ax.plot([0, n - 1], [0, n - 1], linestyle="--", color="#999999", linewidth=1, label="perfect agreement")
        ax.set_xlabel("control_proxy_50k rank (0 = best)")
        ax.set_ylabel("full_validation rank (0 = best)")
        ax.set_title("Checkpoint ranking: proxy vs. full validation")
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "ranking_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written["ranking_comparison"] = str(path)

    working_points = [
        "validation_bc_mistag_eff_0.80_percent", "validation_bd_mistag_eff_0.80_percent",
        "validation_bc_mistag_eff_0.90_percent", "validation_bd_mistag_eff_0.90_percent",
        "validation_cb_mistag_eff_0.50_percent", "validation_cd_mistag_eff_0.50_percent",
        "validation_cb_mistag_eff_0.80_percent", "validation_cd_mistag_eff_0.80_percent",
    ]
    diffs_by_wp = {}
    for wp in working_points:
        control_key, full_key = f"control_proxy_50k_{wp}", f"full_validation_{wp}"
        diffs = [
            _float(row.get(full_key)) - _float(row.get(control_key))
            for row in rows
            if _float(row.get(control_key)) is not None and _float(row.get(full_key)) is not None
        ]
        if diffs:
            diffs_by_wp[wp.replace("validation_", "").replace("_percent", "")] = diffs
    if diffs_by_wp:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(diffs_by_wp.keys())
        ax.boxplot([diffs_by_wp[label] for label in labels], tick_labels=labels)
        ax.axhline(0.0, color="#999999", linewidth=1, linestyle="--")
        ax.set_ylabel("full_validation - control_proxy_50k (percentage points)")
        ax.set_title("Per-working-point proxy/full difference across checkpoints")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        path = plots_dir / "per_working_point_differences.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written["per_working_point_differences"] = str(path)

    runtime = summary.get("runtime_seconds") or {}
    if runtime.get("control_proxy_50k") is not None and runtime.get("full_validation") is not None:
        fig, ax = plt.subplots(figsize=(4, 5))
        labels = ["control_proxy_50k", "full_validation"]
        values = [runtime["control_proxy_50k"], runtime["full_validation"]]
        ax.bar(labels, values, color=["#4c78a8", "#e15759"])
        ax.set_ylabel("wall-clock seconds (all checkpoints, parallelized)")
        ax.set_title("Evaluation runtime by tier")
        fig.tight_layout()
        path = plots_dir / "runtime_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written["runtime_comparison"] = str(path)

    return {"status": "ok", "written": written}


def format_seconds(seconds):
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def build_checkpoint_table(rows):
    lines = [
        "| checkpoint | proxy aggregate metric | full aggregate metric | proxy rank | full rank | rank agreement | worst working-point disagreement |",
        "|---|---|---|---|---|---|---|",
    ]
    paired = []
    for row in rows:
        control = _float(row.get("control_proxy_50k_working_point_mistag_percent"))
        full = _float(row.get("full_validation_working_point_mistag_percent"))
        paired.append((row["checkpoint_id"], control, full, row))
    ranked_by_control = sorted([item for item in paired if item[1] is not None], key=lambda item: item[1])
    ranked_by_full = sorted([item for item in paired if item[2] is not None], key=lambda item: item[2])
    control_rank = {name: index + 1 for index, (name, _, _, _) in enumerate(ranked_by_control)}
    full_rank = {name: index + 1 for index, (name, _, _, _) in enumerate(ranked_by_full)}

    working_points = [
        "validation_bc_mistag_eff_0.80_percent", "validation_bd_mistag_eff_0.80_percent",
        "validation_bc_mistag_eff_0.90_percent", "validation_bd_mistag_eff_0.90_percent",
        "validation_cb_mistag_eff_0.50_percent", "validation_cd_mistag_eff_0.50_percent",
        "validation_cb_mistag_eff_0.80_percent", "validation_cd_mistag_eff_0.80_percent",
    ]
    for name, control, full, row in paired:
        cr = control_rank.get(name)
        fr = full_rank.get(name)
        agree = "yes" if (cr is not None and fr is not None and cr == fr) else ("n/a" if cr is None or fr is None else "no")
        worst_wp, worst_diff = None, 0.0
        for wp in working_points:
            control_wp = _float(row.get(f"control_proxy_50k_{wp}"))
            full_wp = _float(row.get(f"full_validation_{wp}"))
            if control_wp is None or full_wp is None:
                continue
            diff = abs(full_wp - control_wp)
            if diff >= worst_diff:
                worst_diff = diff
                worst_wp = wp.replace("validation_", "").replace("_percent", "")
        worst_str = f"{worst_wp} ({worst_diff:.3f} pp)" if worst_wp else "n/a"
        control_str = f"{control:.4f}" if control is not None else "FAILED"
        full_str = f"{full:.4f}" if full is not None else "FAILED"
        lines.append(f"| {name} | {control_str} | {full_str} | {cr or 'n/a'} | {fr or 'n/a'} | {agree} | {worst_str} |")
    return "\n".join(lines)


def build_summary_table(summary):
    proxy_vs_full = summary.get("proxy_vs_full") or {}
    correlation = proxy_vs_full.get("pearson_spearman") or {}
    kendall = proxy_vs_full.get("kendall") or {}
    pairwise = proxy_vs_full.get("pairwise_direction_agreement") or {}
    best_agreement = proxy_vs_full.get("best_checkpoint_agreement") or {}
    runtime = summary.get("runtime_seconds") or {}
    n = summary.get("checkpoints_distinct", 0)
    spearman = correlation.get("spearman_rho")
    tau = kendall.get("tau")
    agreement_fraction = pairwise.get("agreement_fraction")
    lines = [
        "| number of checkpoints | Spearman | Kendall | pairwise direction agreement | best-checkpoint agreement | 50k runtime | full-validation runtime |",
        "|---|---|---|---|---|---|---|",
        "| {n} | {spearman} | {kendall} | {pairwise} | {best} | {control_rt} | {full_rt} |".format(
            n=n,
            spearman=f"{spearman:.3f}" if spearman is not None else "unavailable",
            kendall=f"{tau:.3f}" if tau is not None else "unavailable",
            pairwise=f"{agreement_fraction:.1%}" if agreement_fraction is not None else "unavailable",
            best="agrees" if best_agreement.get("agrees") else ("disagrees" if best_agreement.get("agrees") is False else "unavailable"),
            control_rt=format_seconds(runtime.get("control_proxy_50k")),
            full_rt=format_seconds(runtime.get("full_validation")),
        ),
    ]
    return "\n".join(lines)


def build_nightly_result_md(run_dir, summary, rows):
    verdict, verdict_reason = determine_verdict(summary, rows)
    proxy_vs_full = summary.get("proxy_vs_full") or {}
    sanity = summary.get("proxy_sanity_check") or {}
    duplicates = summary.get("checkpoints_duplicate") or []
    runtime = summary.get("runtime_seconds") or {}
    git = summary.get("git") or {}

    pairwise = proxy_vs_full.get("pairwise_direction_agreement") or {}
    disagreements = pairwise.get("disagreements") or []
    disagreement_lines = "\n".join(
        f"- {d['checkpoint_a']} vs {d['checkpoint_b']}: control_proxy_50k prefers `{d.get('control_prefers')}`, full_validation prefers `{d.get('full_holdout_prefers')}`"
        for d in disagreements
    ) or "None -- control_proxy_50k and full_validation agreed on every non-tied checkpoint pair."

    duplicate_lines = "\n".join(
        f"- `{d['id']}` is byte-identical to `{d.get('duplicate_of')}` ({d.get('reason')}) -- excluded from evaluation"
        for d in duplicates
    ) or "None."

    class_balance_lines = "\n".join(
        f"- {label}: " + ", ".join(f"{flavor}={fraction:.1%}" for flavor, fraction in (sanity.get(key) or {}).get("fraction_by_flavor", {}).items())
        for label, key in (("control_proxy_50k", "control_proxy_50k_class_balance"), ("full_validation", "full_validation_class_balance"))
    )

    checkpoint_table = build_checkpoint_table(rows)
    summary_table = build_summary_table(summary)

    reproduction_command = (
        f"PYTHONPATH=scripts .venv/bin/python3 scripts/research/run_proxy_audit.py "
        f"--config {summary.get('config_path', 'configs/research/nightly_proxy_audit.yaml')} --run-id {summary.get('run_id', 'REPLACE_ME')}"
    )
    n_checkpoints = summary.get("checkpoints_distinct", 0)
    if n_checkpoints < 6:
        checkpoint_count_note = f"{n_checkpoints} checkpoints is below the task's suggested 6-12 range -- correlation coefficients above are indicative at best."
    elif n_checkpoints <= 12:
        checkpoint_count_note = f"{n_checkpoints} checkpoints is within the task's suggested 6-12 range, but still a small sample for correlation analysis; treat coefficients as indicative, not as a large-sample statistical guarantee."
    else:
        checkpoint_count_note = f"{n_checkpoints} checkpoints exceeds the task's suggested 6-12 range."

    content = f"""# Nightly Result: 50k Control Proxy vs. Full Validation

## 1. Executive verdict

**{verdict}**

{verdict_reason}

## 2. What changed in the code

- Added `configs/presets/shared/proxy_control_50k_override.yaml`: repoints the PBT controller's decision tier from the 5,000-events/class `val5k_tail` control proxy to the existing 50,000-events/class `val50k_tail` proxy (`control_proxy_50k` in this report), scoped to a new opt-in experiment config only -- the shared base preset and the 11 other experiment configs that compose it are unmodified.
- Added `configs/experiments/nightly_proxy_control50k_smoke.yaml`: a smoke-tested live-PBT config confirming the switched tier launches cleanly, produces finite metrics, and never schedules the legacy overlapping `val1000k` tier.
- Fixed a real (if previously unreachable in production) gap in `scripts/training/pbt/controller/decision.py::classify_observation`: an `Inf` metric value could trigger a genuine (non-NaN-safe) comparison and select an active LR-decrease action; added an explicit finiteness guard.
- Added the standalone audit itself: `scripts/research/run_proxy_audit.py`, `proxy_statistics.py`, `proxy_sanity_check.py`, and this report generator -- all reusing the existing `run_tiered_evaluation`, `read_metrics`, `finite_metric_ok`, and `reporting/statistics.py` correlation/ranking code rather than duplicating it.
- No changes to `runner.py`, `planning/`, or PBT algorithm code. No new controllers, no population-size changes.

Full detail: `docs/nightly_plan.md`.

## 3. Experimental setup

- **control_proxy_50k**: `val50k_tail`, 50,000 events/class (150,000 total), fixed tail-window slice, evaluated with Weaver `--run-mode test` (full file, no subsampling).
- **full_validation**: `val_holdout`, ~941,708 events/class average (2,825,125 total), disjoint from control_proxy_50k by construction (built to exclude both the 5k control and 50k monitor tail windows).
- Metric: `{summary.get('audit_config', {}).get('metric_name', 'validation_working_point_mistag_percent')}` (mode: `{summary.get('audit_config', {}).get('metric_mode', 'min')}`), plus all individual b-tag/c-tag working points, aggregate ranking score, and validation loss -- same metric implementation (`scripts/training/runtime.py::read_metrics`) for both tiers.
- Dataset: `{summary.get('audit_config', {}).get('dataset')}` / data config `{summary.get('audit_config', {}).get('data_config')}`.
- Git commit: `{git.get('commit', 'unknown')}` (dirty: `{git.get('dirty', 'unknown')}`).

## 4. Checkpoints evaluated

{summary.get('checkpoints_distinct', 0)} distinct checkpoints (by SHA256), out of {summary.get('checkpoints_requested', 0)} requested. Duplicates excluded:

{duplicate_lines}

## 5. 50k proxy versus full-validation results

{checkpoint_table}

## 6. Ranking disagreements

{disagreement_lines}

## 7. Proxy sanity-check findings

Truth-class (flavor) proportions:

{class_balance_lines}

Positional-bias hypothesis: {(sanity.get('positional_bias') or {}).get('hypothesis', 'unavailable')}

Train/proxy overlap: {(sanity.get('train_overlap') or {}).get('note', 'unavailable')}

Kinematic variables: {(sanity.get('kinematic_variables') or {}).get('status', 'unavailable')} -- {(sanity.get('kinematic_variables') or {}).get('reason', '')}

## 8. Runtime and computational cost

{summary_table}

Total audit wall-clock time: {format_seconds(runtime.get('total'))}.

## 9. Known limitations

- No event-level predictions are stored anywhere in this codebase, so bootstrap confidence intervals on the correlation/agreement statistics are **unavailable** -- reported as such, not fabricated.
- control_proxy_50k and full_validation are deterministic, disjoint, contiguous row windows of the same source file, not independent random samples -- see the sanity check's positional-bias hypothesis (§7), which this audit can neither confirm nor rule out without per-event kinematic data.
- {checkpoint_count_note}
- This audit evaluates the *existing* population of checkpoints; it does not itself run new PBT training, and does not claim anything about PBT/controller correctness beyond what the checkpoints' own recorded metrics show.

## 10. Claims that must not be made

- This audit does **not** show that the dynamic controller or PBT improved the model -- only that certain checkpoints' recorded metrics track between the two validation tiers in a certain way.
- A single correlation coefficient above must not be quoted alone as "the proxy is valid" -- always report it together with pairwise agreement, best-checkpoint agreement, and the disagreement list.
- The switched 50k control tier in `nightly_proxy_control50k_smoke.yaml` was smoke-tested for a few generations only; this is not evidence that a full-length PBT run with this tier behaves identically to the existing 5k-tier runs.

## 11. One concrete next experiment

Run a real, full-length PBT experiment (not a smoke test) using `configs/presets/shared/proxy_control_50k_override.yaml` end-to-end, with `full_validation` (`val_holdout`) evaluated on the existing `full_interval_generations` cadence throughout, to see whether the proxy-vs-full agreement measured here on independently-trained checkpoints also holds *within* a single run's trajectory (checkpoint-to-checkpoint, not just across unrelated runs).

## 12. Exact reproduction command

```
{reproduction_command}
PYTHONPATH=scripts .venv/bin/python3 scripts/reports/build_proxy_audit_report.py {run_dir}
```
"""
    return content


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = load_json(run_dir / "summary.json", default={})
    rows = load_csv_rows(run_dir / "checkpoint_metrics.csv")

    plots_status = build_plots(run_dir, summary, rows)

    content = build_nightly_result_md(run_dir, summary, rows)
    (run_dir / "NIGHTLY_RESULT.md").write_text(content, encoding="utf-8")

    print(f"NIGHTLY_RESULT.md written to {run_dir / 'NIGHTLY_RESULT.md'}")
    print(f"plots: {plots_status}")


if __name__ == "__main__":
    main()
