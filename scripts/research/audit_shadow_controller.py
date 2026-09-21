#!/usr/bin/env python3
"""Read-only audit of a completed fixed-LR shadow-controller run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validation.controller_replay import ReplayPolicy, replay_series  # noqa: E402


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def member_order(manifest):
    return sorted(manifest["members"], key=lambda name: manifest["members"][name]["lr"])


def signal_code(signal):
    return {"DOWN": -1, "FLAT": 0, "UP": 1}[signal]


def verify_checkpoints(run, members):
    expected = []
    missing = []
    corrupt = []
    for member in members:
        for epoch in range(18, 114):
            for kind in ("state", "optimizer", "scaler"):
                path = run / member / f"net_epoch-{epoch}_{kind}.pt"
                expected.append(path)
                if not path.is_file():
                    missing.append(str(path))
    for path in expected:
        if not path.is_file():
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                failed_member = archive.testzip()
            if failed_member:
                corrupt.append({"path": str(path), "member": failed_member})
        except Exception as exc:  # pragma: no cover - audit path
            corrupt.append({"path": str(path), "error": repr(exc)})
    return {
        "expected_generation_components": len(expected),
        "present_generation_components": len(expected) - len(missing),
        "missing": missing,
        "corrupt": corrupt,
        "crc_checked_bytes": sum(path.stat().st_size for path in expected if path.is_file()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit recorded proxy/reference signals and shadow-controller decisions."
    )
    parser.add_argument("run", type=Path, help="Completed shadow-controller run directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for CSV, JSON, and dashboard outputs")
    parser.add_argument("--verify-crc", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.output_dir.resolve()

    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    metrics = read_csv(run / "metrics.csv")
    tiers = read_csv(run / "tiered_metrics.csv")
    members = member_order(manifest)
    threshold = 0.004579530054761879

    decision_rows = []
    for generation in manifest["generations"]:
        for member in members:
            action = generation["controller_actions"][member]
            observation = generation["controller_observations"][member]
            decision_rows.append(
                {
                    "generation": generation["index"],
                    "member": member,
                    "lr": observation["lr"],
                    "metric": observation["metric_value"],
                    "metric_delta": observation.get("metric_delta"),
                    "metric_ema": observation["metric_ema"],
                    "metric_ema_delta": observation.get("metric_ema_delta"),
                    "state": action["state_label"],
                    "direction_streak": observation["direction_streak"],
                    "action": action["action"],
                    "reason": action["reason"],
                    "action_ready": action["action_ready"],
                    "safety_check": action["safety_check"],
                    "proposed_lr": action["proposed_lr"],
                    "bounded_lr": action["bounded_lr"],
                    "applied": action["applied"],
                }
            )

    per_member = {}
    for member in members:
        rows = [row for row in decision_rows if row["member"] == member]
        active = [row["action"] for row in rows if row["action"] != "keep"]
        per_member[member] = {
            "UP": sum(row["action"] == "lr_mul_1_05" for row in rows),
            "KEEP": sum(row["action"] == "keep" for row in rows),
            "DOWN": sum(row["action"] == "lr_mul_0_95" for row in rows),
            "cooldown_suppressed": sum(not row["action_ready"] for row in rows),
            "patience_suppressed": sum("patience" in row["reason"] for row in rows),
            "active_action_reversals": sum(left != right for left, right in zip(active, active[1:])),
        }

    paired = defaultdict(dict)
    for row in tiers:
        generation = int(row["generation"])
        if generation >= 0:
            paired[(row["member"], generation)][row["tier"]] = float(row["metric_value"])
    initial_reference = float(
        next(row["metric_value"] for row in tiers if int(row["generation"]) == -1)
    )
    initial_proxy = float(summary["baseline"]["metric_value"])

    reference_rows = []
    raw_change_pairs = []
    policy = ReplayPolicy(ema_beta=0.5, patience=2, cooldown_observations=2)
    for member in members:
        series = [{"generation": -1, "proxy": initial_proxy, "reference": initial_reference}]
        series.extend(
            {
                "generation": generation,
                "proxy": values["control"],
                "reference": values["monitor"],
            }
            for (name, generation), values in sorted(paired.items())
            if name == member
        )
        proxy_replay = replay_series(series, "proxy", threshold, policy)
        reference_replay = replay_series(series, "reference", threshold, policy)
        actual_by_generation = {
            row["generation"]: row for row in decision_rows if row["member"] == member
        }
        for index, (point, proxy_item, reference_item) in enumerate(
            zip(series, proxy_replay, reference_replay)
        ):
            if point["generation"] < 0:
                continue
            previous = series[index - 1]
            proxy_raw_delta = point["proxy"] - previous["proxy"]
            reference_raw_delta = point["reference"] - previous["reference"]
            raw_change_pairs.append((proxy_raw_delta, reference_raw_delta))
            actual = actual_by_generation[point["generation"]]
            reference_rows.append(
                {
                    "member": member,
                    "generation": point["generation"],
                    "proxy_metric": point["proxy"],
                    "reference_metric": point["reference"],
                    "proxy_raw_delta": proxy_raw_delta,
                    "reference_raw_delta": reference_raw_delta,
                    "raw_sign_disagrees": proxy_raw_delta * reference_raw_delta < 0,
                    "proxy_ema_delta": proxy_item["ema_delta"],
                    "reference_ema_delta": reference_item["ema_delta"],
                    "proxy_signal": proxy_item["signal"],
                    "reference_signal": reference_item["signal"],
                    "direction_disagrees": proxy_item["signal"] != reference_item["signal"],
                    "proxy_paired_action": proxy_item["action"],
                    "reference_paired_action": reference_item["action"],
                    "paired_action_disagrees": proxy_item["action"] != reference_item["action"],
                    "actual_shadow_state": actual["state"],
                    "actual_shadow_action": actual["action"],
                }
            )

    raw_sign_disagreements = [row for row in reference_rows if row["raw_sign_disagrees"]]
    raw_meaningful = [
        row
        for row in raw_sign_disagreements
        if abs(row["proxy_raw_delta"]) >= threshold
        or abs(row["reference_raw_delta"]) >= threshold
    ]
    proxy_changes = [pair[0] for pair in raw_change_pairs]
    reference_changes = [pair[1] for pair in raw_change_pairs]
    residuals = [left - right for left, right in raw_change_pairs]
    residual_median = statistics.median(residuals)
    residual_mad = statistics.median(abs(value - residual_median) for value in residuals)

    expected_lrs = {member: float(manifest["members"][member]["lr"]) for member in members}
    lr_mismatches = []
    for generation in manifest["generations"]:
        for member in members:
            worker = generation["workers"][member]
            observed = [
                float(worker["lr"]),
                float(generation["controller_observations"][member]["lr"]),
                float(generation["controller_observations"][member]["optimizer_lr_mean"]),
            ]
            if any(not math.isclose(value, expected_lrs[member], rel_tol=0.0, abs_tol=1e-15) for value in observed):
                lr_mismatches.append({"generation": generation["index"], "member": member, "values": observed})

    checkpoint_audit = verify_checkpoints(run, members) if args.verify_crc else {
        "expected_generation_components": 8 * 96 * 3,
        "present_generation_components": sum(
            (run / member / f"net_epoch-{epoch}_{kind}.pt").is_file()
            for member in members
            for epoch in range(18, 114)
            for kind in ("state", "optimizer", "scaler")
        ),
        "missing": [],
        "corrupt": "not_checked_without_--verify-crc",
    }

    result = {
        "verdict": "C",
        "integrity": {
            "manifest_status": manifest["status"],
            "generation_count": len(manifest["generations"]),
            "generation_indices_complete": [g["index"] for g in manifest["generations"]] == list(range(96)),
            "all_workers_completed": all(
                len(g["workers"]) == 8 and all(w["status"] == "completed" for w in g["workers"].values())
                for g in manifest["generations"]
            ),
            "proxy_metric_rows": len(metrics),
            "reference_monitor_rows": sum(row["tier"] == "monitor" for row in tiers),
            "reference_rounds": sorted({int(row["generation"]) for row in tiers if row["tier"] == "monitor" and int(row["generation"]) >= 0}),
            "control_tier_rows": sum(row["tier"] == "control" for row in tiers),
            "initial_proxy_evaluations": summary["event_counts"]["evaluation"] - len(metrics),
            "lr_mismatches": lr_mismatches,
            "applied_action_count": sum(row["applied"] for row in decision_rows),
            "exploit_event_count": len(summary["exploit_history"]),
            "checkpoint_audit": checkpoint_audit,
            "terminal_applied_count_field_missing": manifest["generations"][-1]["dynamic_controller"].get("applied_action_count") is None,
        },
        "controller": {
            "counts": {
                "UP": sum(row["action"] == "lr_mul_1_05" for row in decision_rows),
                "KEEP": sum(row["action"] == "keep" for row in decision_rows),
                "DOWN": sum(row["action"] == "lr_mul_0_95" for row in decision_rows),
            },
            "per_member": per_member,
            "active_timeline": [
                {key: row[key] for key in ("generation", "member", "state", "action", "metric_ema_delta", "proposed_lr", "bounded_lr", "safety_check")}
                for row in decision_rows
                if row["action"] != "keep"
            ],
            "raw_threshold_crossings": sum(
                row["metric_delta"] is not None and abs(row["metric_delta"]) > threshold
                for row in decision_rows
            ),
            "ema_threshold_crossings": sum(
                row["metric_ema_delta"] is not None and abs(row["metric_ema_delta"]) > threshold
                for row in decision_rows
            ),
            "patience_suppressed": sum("patience" in row["reason"] for row in decision_rows),
            "cooldown_suppressed": sum(not row["action_ready"] for row in decision_rows),
            "active_action_reversals": sum(item["active_action_reversals"] for item in per_member.values()),
            "permanently_inactive_members": [member for member, item in per_member.items() if item["UP"] + item["DOWN"] == 0],
            "clamped_active_proposals": sum(row["action"] != "keep" and row["safety_check"] == "clamped" for row in decision_rows),
        },
        "proxy_reference": {
            "paired_round_member_rows": len(reference_rows),
            "paired_raw_changes": len(raw_change_pairs),
            "raw_sign_disagreements": len(raw_sign_disagreements),
            "raw_sign_disagreements_both_subthreshold": sum(
                abs(row["proxy_raw_delta"]) < threshold and abs(row["reference_raw_delta"]) < threshold
                for row in raw_sign_disagreements
            ),
            "raw_sign_disagreements_either_meaningful": len(raw_meaningful),
            "thresholded_ema_direction_disagreements": sum(row["direction_disagrees"] for row in reference_rows),
            "opposite_active_direction_disagreements": sum(
                {row["proxy_signal"], row["reference_signal"]} == {"UP", "DOWN"}
                for row in reference_rows
            ),
            "paired_policy_action_disagreements": sum(row["paired_action_disagrees"] for row in reference_rows),
            "paired_proxy_action_counts": dict(Counter(row["proxy_paired_action"] for row in reference_rows)),
            "paired_reference_action_counts": dict(Counter(row["reference_paired_action"] for row in reference_rows)),
            "change_pearson": float(np.corrcoef(proxy_changes, reference_changes)[0, 1]),
            "change_residual_mae": statistics.mean(abs(value) for value in residuals),
            "change_residual_robust_sigma": 1.4826 * residual_mad,
            "meaningful_raw_disagreement_cases": raw_meaningful,
            "paired_action_disagreement_cases": [row for row in reference_rows if row["paired_action_disagrees"]],
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "controller_decisions.csv", decision_rows)
    write_csv(out / "reference_comparison.csv", reference_rows)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(members)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    for color, member in zip(colors, members):
        dense = [row for row in decision_rows if row["member"] == member]
        refs = [row for row in reference_rows if row["member"] == member]
        ax.plot([row["generation"] for row in dense], [row["metric"] for row in dense], color=color, lw=1, alpha=0.75)
        ax.plot([row["generation"] for row in refs], [row["reference_metric"] for row in refs], color=color, ls="--", marker=".", ms=3, alpha=0.75)
        for row in dense:
            if row["action"] != "keep":
                ax.scatter(row["generation"], row["metric"], color=color, marker="^", s=80, edgecolor="black", zorder=5)
    ax.set(title="Proxy (solid) / 150k reference (dashed); triangles = proposals", xlabel="Generation", ylabel="Mistag score (%)")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    action_matrix = np.zeros((len(members), 96))
    for row in decision_rows:
        action_matrix[members.index(row["member"]), row["generation"]] = {"lr_mul_0_95": -1, "keep": 0, "lr_mul_1_05": 1}[row["action"]]
    ax.imshow(action_matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#d62728", "#eeeeee", "#1f77b4"]), vmin=-1, vmax=1)
    ax.set(title="Actual shadow proposal timeline (red DOWN / gray KEEP / blue UP)", xlabel="Generation", yticks=range(len(members)), yticklabels=members)

    ax = axes[1, 0]
    ref_generations = sorted({row["generation"] for row in reference_rows})
    agreement = np.zeros((len(members), len(ref_generations)))
    for row in reference_rows:
        left, right = row["proxy_signal"], row["reference_signal"]
        value = 0 if left == right else 2 if {left, right} == {"UP", "DOWN"} else 1
        agreement[members.index(row["member"]), ref_generations.index(row["generation"])] = value
    ax.imshow(agreement, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#2ca02c", "#ffbf00", "#d62728"]), vmin=0, vmax=2)
    ax.set(title="Paired-cadence direction: green agree / amber active-vs-flat / red opposite", xlabel="Reference generation", yticks=range(len(members)), yticklabels=members, xticks=range(len(ref_generations)), xticklabels=ref_generations)
    ax.tick_params(axis="x", labelrotation=45)

    ax = axes[1, 1]
    generations = range(96)
    cumulative = {}
    for label, action in (("UP", "lr_mul_1_05"), ("DOWN", "lr_mul_0_95"), ("KEEP", "keep")):
        counts = []
        total = 0
        for generation in generations:
            total += sum(row["generation"] == generation and row["action"] == action for row in decision_rows)
            counts.append(total)
        cumulative[label] = counts
    ax.plot(list(generations), cumulative["UP"], label="UP", color="#1f77b4", lw=2)
    ax.plot(list(generations), cumulative["DOWN"], label="DOWN", color="#d62728", lw=2)
    ax.set(title="Cumulative shadow proposals", xlabel="Generation", ylabel="Active proposal count", ylim=(-0.1, 2.4))
    keep_ax = ax.twinx()
    keep_ax.plot(list(generations), cumulative["KEEP"], label="KEEP", color="#777777", ls="--", alpha=0.7)
    keep_ax.set_ylabel("KEEP count", color="#777777")
    handles, labels = ax.get_legend_handles_labels()
    keep_handles, keep_labels = keep_ax.get_legend_handles_labels()
    ax.legend(handles + keep_handles, labels + keep_labels, loc="center right")
    ax.grid(alpha=0.2)

    fig.savefig(out / "controller_dashboard.png", dpi=170)
    plt.close(fig)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
