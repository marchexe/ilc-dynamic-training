#!/usr/bin/env python3
"""Build deterministic publication tables and figures for cadenced PBT v1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC = "validation_total_reference_mistag_geomean_percent"
MEMBER_ORDER = (
    "lr_3e-6",
    "lr_5_75e-6",
    "lr_8_5e-6",
    "lr_11_25e-6",
    "lr_14e-6",
)
COLORS = {
    "lr_3e-6": "#0072B2",
    "lr_5_75e-6": "#E69F00",
    "lr_8_5e-6": "#009E73",
    "lr_11_25e-6": "#D55E00",
    "lr_14e-6": "#CC79A7",
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _members(manifest: dict[str, Any]) -> list[str]:
    recorded = set((manifest.get("members") or {}).keys())
    ordered = [member for member in MEMBER_ORDER if member in recorded]
    ordered.extend(sorted(recorded.difference(ordered)))
    return ordered


def _load_metrics(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["generation"]), row["trial"])
        if key in result:
            raise ValueError(f"Duplicate metrics row for generation/member {key}")
        result[key] = row
    return result


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decision_type(decision: dict[str, Any], event: dict[str, Any] | None) -> str:
    if decision.get("reason") == "warmup":
        return "warmup"
    if decision.get("reason") == "terminal_generation":
        return "terminal_suppression"
    if decision.get("reason") == "within_margin":
        return "within_margin_no_op"
    if event and not event.get("mutation_applied"):
        return "copy_only"
    if event:
        return "copy_plus_mutation"
    return "no_op"


def _rejections(event: dict[str, Any] | None) -> str:
    if not event:
        return "[]"
    return _json(event.get("rejected_mutations") or [])


def evidence(source_run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load and cross-check the three independent run ledgers."""
    if manifest.get("method") != "cadenced_pbt_v1":
        raise ValueError("cadenced publication requires method=cadenced_pbt_v1")
    if manifest.get("status") != "completed":
        raise ValueError("cadenced publication requires a completed run")
    generations = manifest.get("generations") or []
    members = _members(manifest)
    metrics = _load_metrics(source_run / "metrics.csv")
    events = _load_events(source_run / "events.jsonl")
    decisions_by_generation = {
        int(event["generation"]): event
        for event in events
        if event.get("event_type") == "cadenced_pbt_decision"
    }
    exploits_by_generation: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("event_type") == "exploit" and event.get("applied"):
            exploits_by_generation.setdefault(int(event["generation"]), []).append(event)
    expected_metric_keys = {(int(generation["index"]), member)
                            for generation in generations for member in members}
    if set(metrics) != expected_metric_keys:
        missing = sorted(expected_metric_keys.difference(metrics))
        extra = sorted(set(metrics).difference(expected_metric_keys))
        raise ValueError(f"metrics coverage mismatch: missing={missing[:3]} extra={extra[:3]}")
    if set(decisions_by_generation) != {int(generation["index"]) for generation in generations}:
        raise ValueError("event log does not contain exactly one decision per generation")

    lr_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    exploit_rows: list[dict[str, Any]] = []
    trajectories: dict[str, list[float]] = {member: [] for member in members}
    scores: dict[str, list[float]] = {member: [] for member in members}
    manifest_events: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for generation in generations:
        index = int(generation["index"])
        completed_epoch = index + 1
        weaver_epoch = int(generation["epoch"])
        decision = generation.get("cadenced_pbt_v1") or {}
        logged_decision = decisions_by_generation[index]
        applied = generation.get("exploit") or []
        if len(applied) > 1:
            raise ValueError(f"generation {index} has more than one exploit")
        event = applied[0] if applied else None
        logged_exploits = exploits_by_generation.get(index, [])
        if len(logged_exploits) != len(applied):
            raise ValueError(f"manifest/event exploit mismatch at generation {index}")
        for key in ("completed_epoch", "reason", "donor", "recipient", "terminal"):
            if logged_decision.get(key) != decision.get(key):
                raise ValueError(f"decision mismatch at generation {index}: {key}")
        if event:
            logged = logged_exploits[0]
            for manifest_key, log_key in (("donor", "donor"), ("recipient", "recipient"),
                                          ("new_lr", "new_lr"), ("mutation_factor", "mutation")):
                if event.get(manifest_key) != logged.get(log_key):
                    raise ValueError(f"exploit mismatch at generation {index}: {manifest_key}")
            manifest_events.append((generation, event))

        decision_kind = _decision_type(decision, event)
        rejection_text = _rejections(event)
        decision_rows.append({
            "generation": index,
            "completed_epoch": completed_epoch,
            "weaver_epoch": weaver_epoch,
            "decision_type": decision_kind,
            "eligible_boundary": str(bool(decision.get("copy_opportunity"))).lower(),
            "warmup_active_during_training": str(bool(decision.get("warmup_active_during_training"))).lower(),
            "terminal": str(bool(decision.get("terminal"))).lower(),
            "copy_opportunity": str(bool(decision.get("copy_opportunity"))).lower(),
            "lr_mutation_opportunity": str(bool(decision.get("lr_mutation_opportunity"))).lower(),
            "copy_applied": str(bool(event)).lower(),
            "mutation_applied": str(bool(event and event.get("mutation_applied"))).lower(),
            "copy_only": str(bool(event and not event.get("mutation_applied"))).lower(),
            "reason": decision.get("reason", ""),
            "donor": decision.get("donor", ""),
            "recipient": decision.get("recipient", ""),
            "donor_lr": decision.get("donor_lr", ""),
            "recipient_lr_before": decision.get("old_lr", ""),
            "recipient_lr_after": decision.get("new_lr", ""),
            "mutation_factor": decision.get("mutation_factor", ""),
            "mutation_reason": decision.get("mutation_reason", ""),
            "metric_gap": decision.get("metric_gap", ""),
            "decision_margin": decision.get("decision_margin", ""),
            "rejected_candidates": rejection_text,
        })

        for member in members:
            metric_row = metrics[(index, member)]
            lr_training = float(metric_row["LR"])
            score = float(metric_row["optimization_metric_value"])
            worker_score = float(generation["workers"][member]["metrics"][METRIC])
            if score != worker_score:
                raise ValueError(f"metric mismatch at generation {index}, member {member}")
            trajectories[member].append(lr_training)
            scores[member].append(score)
            copied = bool(event and event.get("recipient") == member)
            lr_after = float(event["new_lr"]) if copied else lr_training
            lr_rows.append({
                "completed_epoch": completed_epoch,
                "weaver_epoch": weaver_epoch,
                "member": member,
                "lr_training": repr(lr_training),
                "lr_after_adaptation": repr(lr_after),
                "copied": str(copied).lower(),
                "donor": event.get("donor", "") if copied else "",
                "recipient": event.get("recipient", "") if copied else "",
                "mutation_factor": event.get("mutation_factor", "") if copied else "",
                "mutation_type": decision_kind if copied else "none",
                "copy_only": str(bool(copied and not event.get("mutation_applied"))).lower(),
                "rejected_candidates": rejection_text if copied else "[]",
                "decision_reason": decision.get("reason", ""),
                "full_reference_score": repr(score),
            })

        if event:
            pre_copy = event.get("pre_copy") or {}
            post_copy = event.get("post_copy") or {}
            donor_checkpoint = event.get("donor_checkpoint") or {}
            exploit_rows.append({
                "generation": index,
                "completed_epoch": completed_epoch,
                "weaver_epoch": weaver_epoch,
                "event_id": event.get("event_id", ""),
                "donor": event.get("donor", ""),
                "recipient": event.get("recipient", ""),
                "donor_lr": event.get("donor_lr", ""),
                "recipient_lr_before": event.get("recipient_lr", ""),
                "recipient_lr_after": event.get("new_lr", ""),
                "mutation_factor": event.get("mutation_factor", ""),
                "mutation_type": decision_kind,
                "copy_only": str(not bool(event.get("mutation_applied"))).lower(),
                "mutation_reason": event.get("mutation_reason", ""),
                "metric_gap": event.get("metric_gap", ""),
                "decision_margin": event.get("decision_margin", ""),
                "rejected_candidates": rejection_text,
                "pre_copy_state_sha256": (pre_copy.get("state") or {}).get("sha256", ""),
                "donor_state_sha256": (donor_checkpoint.get("state") or {}).get("sha256", ""),
                "post_copy_state_sha256": (post_copy.get("state") or {}).get("sha256", ""),
                "post_copy_optimizer_sha256": (post_copy.get("optimizer") or {}).get("sha256", ""),
                "post_copy_scaler_sha256": (post_copy.get("scaler") or {}).get("sha256", ""),
            })

    current_best = [min(scores[member][i] for member in members) for i in range(len(generations))]
    best_so_far: list[float] = []
    for value in current_best:
        best_so_far.append(min(value, best_so_far[-1]) if best_so_far else value)
    ancestors = {member: member for member in members}
    collapse_epoch = None
    for generation, event in manifest_events:
        ancestors[event["recipient"]] = ancestors[event["donor"]]
        if collapse_epoch is None and len(set(ancestors.values())) == 1:
            collapse_epoch = int(generation["index"]) + 1
    return {
        "members": members,
        "generations": generations,
        "lr_rows": lr_rows,
        "decision_rows": decision_rows,
        "exploit_rows": exploit_rows,
        "manifest_events": manifest_events,
        "trajectories": trajectories,
        "scores": scores,
        "current_best": current_best,
        "best_so_far": best_so_far,
        "collapse_epoch": collapse_epoch,
    }


def global_best_lineage(manifest: dict[str, Any]) -> tuple[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    best = manifest["best"]
    target = best["member"]
    generation_limit = int(best["generation"])
    chain: list[tuple[dict[str, Any], dict[str, Any]]] = []
    generations = manifest.get("generations") or []
    while True:
        found = None
        for generation in reversed(generations[:generation_limit]):
            for event in generation.get("exploit") or []:
                if event.get("recipient") == target:
                    found = (generation, event)
                    break
            if found:
                break
        if not found:
            break
        chain.append(found)
        generation, event = found
        target = event["donor"]
        generation_limit = int(generation["index"])
    chain.reverse()
    return target, chain


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, metadata={"Software": "matplotlib"})


def plot_performance(data: dict[str, Any], manifest: dict[str, Any], path: Path) -> None:
    plt = _plot_setup()
    epochs = list(range(1, len(data["generations"]) + 1))
    population_min = data["current_best"]
    population_max = [max(data["scores"][member][i] for member in data["members"])
                      for i in range(len(epochs))]
    medians = []
    for i in range(len(epochs)):
        values = sorted(data["scores"][member][i] for member in data["members"])
        medians.append(values[len(values) // 2])
    fig, ax = plt.subplots(figsize=(13.2, 7.5))
    ax.axvspan(40.5, 50.5, color="#DCEAF2", alpha=.65, label="Final-10 window")
    ax.fill_between(epochs, population_min, population_max, color="#9ECAE1", alpha=.28,
                    label="Population min–max")
    ax.plot(epochs, medians, color="#7B8790", lw=1.5, label="Population median")
    ax.plot(epochs, population_min, color="#0072B2", lw=2.5, label="Current best")
    ax.plot(epochs, data["best_so_far"], color="#26343C", lw=2.0, ls=(0, (5, 3)),
            label="Best so far")
    best = manifest["best"]
    best_epoch = int(best["generation"]) + 1
    ax.scatter(best_epoch, best["metric_value"], marker="*", s=220, color="#D55E00",
               edgecolors="white", linewidths=1.0, zorder=8, label="Global best")
    ax.annotate(f"Global best: E{best_epoch}\n{best['metric_value']:.6f}%",
                (best_epoch, best["metric_value"]), xytext=(12, 18), textcoords="offset points",
                fontsize=10, color="#26343C",
                bbox=dict(boxstyle="round,pad=.35", fc="white", ec="#D6DFE5"))
    ax.set_xlim(.5, 50.5)
    ax.set_xlabel("Completed epoch")
    ax.set_ylabel("Full-reference mistag metric (%) — lower is better")
    ax.grid(axis="y", color="#DDE3E7", lw=.8)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Cadenced PBT v1 performance progression", x=.09, ha="left",
                 fontsize=22, fontweight="bold", color="#172A36")
    fig.text(.09, .91, "Current population performance and the pre-registered final-10 window",
             color="#53616B", fontsize=12)
    fig.tight_layout(rect=(.07, .07, .98, .88))
    _save(fig, path)
    plt.close(fig)


def plot_population_lrs(data: dict[str, Any], path: Path) -> None:
    from matplotlib.lines import Line2D
    plt = _plot_setup()
    epochs = list(range(1, len(data["generations"]) + 1))
    fig, ax = plt.subplots(figsize=(13.2, 7.5))
    ax.axvspan(.5, 2.5, color="#ECEFF1", alpha=.85, label="Warm-up (E1–2)")
    for member in data["members"]:
        values = [lr * 1e6 for lr in data["trajectories"][member]]
        ax.step(epochs, values, where="post", lw=2.0, color=COLORS.get(member), label=member)
    for generation, event in data["manifest_events"]:
        boundary = int(generation["index"]) + 1
        x = boundary + .5
        y = float(event["new_lr"]) * 1e6
        if not event.get("mutation_applied"):
            marker, color = "s", "#26343C"
        elif float(event["mutation_factor"]) < 1:
            marker, color = "v", "#0072B2"
        else:
            marker, color = "^", "#D55E00"
        ax.scatter(x, y, marker=marker, s=48, facecolor=color, edgecolor="white",
                   linewidth=.7, zorder=8)
    ax.axvline(50, color="#7B8790", lw=1.2, ls=(0, (3, 3)))
    ax.text(49.7, 3.2, "terminal\nno action", ha="right", va="bottom",
            fontsize=8.5, color="#53616B")
    ax.set_xlim(.5, 50.5)
    ax.set_ylim(0, 32)
    ax.set_xlabel("Completed epoch")
    ax.set_ylabel("Learning rate (×10⁻⁶)")
    ax.grid(axis="y", color="#DDE3E7", lw=.8)
    top = ax.secondary_xaxis("top", functions=(lambda x: x + 17, lambda x: x - 17))
    top.set_xlabel("Absolute Weaver epoch")
    event_handles = [
        Line2D([], [], marker="v", ls="none", color="#0072B2", markeredgecolor="white",
               label="Copy + LR ×0.8"),
        Line2D([], [], marker="^", ls="none", color="#D55E00", markeredgecolor="white",
               label="Copy + LR ×1.2"),
        Line2D([], [], marker="s", ls="none", color="#26343C", markeredgecolor="white",
               label="Copy only"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([*handles, *event_handles], [*labels, *[h.get_label() for h in event_handles]],
              frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.13))
    fig.suptitle("Cadenced PBT v1 population learning rates", x=.09, ha="left",
                 y=.975, fontsize=22, fontweight="bold", color="#172A36")
    fig.text(.09, .91, "Recorded LR used by each population member during every training epoch",
             color="#53616B", fontsize=12)
    fig.text(.09, .025, "Post-boundary LR changes take effect in the following training epoch.",
             color="#53616B", fontsize=10)
    fig.subplots_adjust(left=.09, right=.97, bottom=.25, top=.78)
    _save(fig, path)
    plt.close(fig)


def _short_member(member: str) -> str:
    return member.removeprefix("lr_").removesuffix("e-6").replace("_", ".")


def plot_global_best_lineage(data: dict[str, Any], manifest: dict[str, Any], path: Path) -> None:
    plt = _plot_setup()
    ancestor, chain = global_best_lineage(manifest)
    best = manifest["best"]
    best_epoch = int(best["generation"]) + 1
    active_member = ancestor
    segment_start = 1
    labelled_members: set[str] = set()
    fig, ax = plt.subplots(figsize=(13.2, 7.5))
    offsets = {2: (5, 30), 3: (18, -54), 5: (8, 25), 13: (8, 22),
               23: (8, -52), 39: (-78, 28), 40: (8, -58)}
    for generation, event in chain:
        boundary = int(generation["index"]) + 1
        xs = list(range(segment_start, boundary + 1))
        ys = [data["trajectories"][active_member][epoch - 1] * 1e6 for epoch in xs]
        label = active_member if active_member not in labelled_members else None
        labelled_members.add(active_member)
        ax.plot(xs, ys, lw=3.0, color=COLORS.get(active_member), label=label)
        donor_y = data["trajectories"][active_member][boundary - 1] * 1e6
        new_y = float(event["new_lr"]) * 1e6
        ax.plot([boundary, boundary + 1], [donor_y, new_y], lw=1.8,
                color=COLORS.get(event["recipient"]), zorder=5)
        marker = "s" if not event.get("mutation_applied") else "D"
        ax.scatter(boundary + 1, new_y, marker=marker, s=62,
                   color=COLORS.get(event["recipient"]), edgecolor="white", zorder=7)
        factor = "copy only" if not event.get("mutation_applied") else f"×{event['mutation_factor']:g}"
        label = (f"E{boundary}: {_short_member(event['donor'])}→{_short_member(event['recipient'])}  {factor}\n"
                 f"next LR {new_y:g}×10⁻⁶")
        ax.annotate(label, (boundary + 1, new_y), xytext=offsets.get(boundary, (8, 20)),
                    textcoords="offset points", fontsize=8.5, color="#44525B",
                    bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#D6DFE5", alpha=.94),
                    arrowprops=dict(arrowstyle="-", color="#9AA6AE", lw=.8))
        active_member = event["recipient"]
        segment_start = boundary + 1
    xs = list(range(segment_start, best_epoch + 1))
    ys = [data["trajectories"][active_member][epoch - 1] * 1e6 for epoch in xs]
    label = active_member if active_member not in labelled_members else None
    ax.plot(xs, ys, lw=3.0, color=COLORS.get(active_member), label=label)
    ax.scatter(best_epoch, float(best["lr"]) * 1e6, marker="*", s=230, color="#172A36",
               edgecolor="white", zorder=9)
    ax.annotate(f"Global best at E{best_epoch}\n{best['member']} · LR {float(best['lr']) * 1e6:g}×10⁻⁶",
                (best_epoch, float(best["lr"]) * 1e6), xytext=(-175, 32), textcoords="offset points",
                fontsize=10, color="#172A36",
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#9AA6AE"),
                arrowprops=dict(arrowstyle="->", color="#53616B"))
    ax.set_xlim(.5, best_epoch + 1.5)
    ax.set_ylim(10, 32)
    ax.set_xlabel("Completed epoch")
    ax.set_ylabel("Lineage learning rate (×10⁻⁶)")
    ax.grid(axis="y", color="#DDE3E7", lw=.8)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), frameon=False, ncol=5,
               loc="upper center", bbox_to_anchor=(.5, .84))
    fig.suptitle("Learning-rate lineage of the global-best checkpoint", x=.09, ha="left",
                 fontsize=22, fontweight="bold", color="#172A36")
    fig.text(.09, .91, "Recorded donor→recipient transitions along the successful ancestry",
             color="#53616B", fontsize=12)
    fig.text(.09, .025,
             "Copy events transfer model/optimizer/scaler state; LR changes are shown separately and are not causal claims.",
             color="#53616B", fontsize=9.5)
    fig.tight_layout(rect=(.07, .07, .98, .77))
    _save(fig, path)
    plt.close(fig)


def build(source_run: Path, destination: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Generate cadence-specific histories and standalone plots in ``destination``."""
    data = evidence(source_run, manifest)
    lr_fields = [
        "completed_epoch", "weaver_epoch", "member", "lr_training", "lr_after_adaptation",
        "copied", "donor", "recipient", "mutation_factor", "mutation_type", "copy_only",
        "rejected_candidates", "decision_reason", "full_reference_score",
    ]
    decision_fields = list(data["decision_rows"][0])
    exploit_fields = list(data["exploit_rows"][0])
    _write_csv(destination / "lr_history.csv", lr_fields, data["lr_rows"])
    _write_csv(destination / "decision_history.csv", decision_fields, data["decision_rows"])
    _write_csv(destination / "exploit_history.csv", exploit_fields, data["exploit_rows"])
    plots = destination / "plots"
    plot_performance(data, manifest, plots / "01_performance_progression.png")
    plot_population_lrs(data, plots / "02_population_lr_trajectories.png")
    plot_global_best_lineage(data, manifest, plots / "03_global_best_lr_lineage.png")
    factors = [event.get("mutation_factor") for _, event in data["manifest_events"]
               if event.get("mutation_applied")]
    return {
        "lr_history_rows": len(data["lr_rows"]),
        "decision_count": len(data["decision_rows"]),
        "exploit_count": len(data["exploit_rows"]),
        "mutation_count": len(factors),
        "copy_only_count": sum(1 for _, event in data["manifest_events"] if not event.get("mutation_applied")),
        "mutation_factors": {
            "0.8": sum(1 for factor in factors if factor == .8),
            "1.2": sum(1 for factor in factors if factor == 1.2),
        },
        "lineage_collapse_completed_epoch": data["collapse_epoch"],
        "unique_lr_counts": [len({data["trajectories"][member][i] for member in data["members"]})
                             for i in range(len(data["generations"]))],
        "final10_current_best_mean": sum(data["current_best"][-10:]) / 10,
        "plots": [
            "01_performance_progression.png",
            "02_population_lr_trajectories.png",
            "03_global_best_lr_lineage.png",
        ],
    }
