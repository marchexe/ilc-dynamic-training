#!/usr/bin/env python3
"""Replay a conservative LR controller over proxy-qualification trajectories.

This command is observational: it reads existing qualification CSV rows and
writes controller proposals.  It never evaluates a checkpoint, starts
training, or changes an optimizer state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class ReplayPolicy:
    ema_beta: float = 0.5
    threshold_multiplier: float = 1.0
    patience: int = 2
    cooldown_observations: int = 2
    up_factor: float = 1.05
    down_factor: float = 0.95


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("runs/eval/proxy_qualification_v1/results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/eval/controller_replay_representative_60k"),
    )
    parser.add_argument("--candidate", default="representative_60k")
    return parser.parse_args()


def load_trajectories(path, candidate):
    trajectories = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["candidate_id"] != candidate:
                continue
            item = dict(row)
            item["epoch"] = int(row["epoch"])
            item["proxy_metric"] = float(row["proxy_metric"])
            item["reference_metric"] = float(row["reference_metric"])
            trajectories[(row["run"], row["member"])].append(item)
    if not trajectories:
        raise ValueError(f"No rows found for candidate {candidate!r} in {path}")
    for rows in trajectories.values():
        rows.sort(key=lambda row: (row["epoch"], row["checkpoint_sha256"]))
    return dict(sorted(trajectories.items()))


def adjacent_pairs(trajectories):
    for (run, member), rows in trajectories.items():
        for previous, current in zip(rows, rows[1:]):
            yield run, member, previous, current


def estimate_change_noise(trajectories):
    """Robust scale of paired proxy-minus-reference adjacent changes."""
    residuals = []
    for _, _, previous, current in adjacent_pairs(trajectories):
        proxy_delta = current["proxy_metric"] - previous["proxy_metric"]
        reference_delta = current["reference_metric"] - previous["reference_metric"]
        residuals.append(proxy_delta - reference_delta)
    if not residuals:
        raise ValueError("At least one adjacent trajectory pair is required")
    center = statistics.median(residuals)
    mad = statistics.median(abs(value - center) for value in residuals)
    return {
        "adjacent_pair_count": len(residuals),
        "residual_median": center,
        "residual_mad": mad,
        "robust_sigma": MAD_TO_SIGMA * mad,
    }


def signal_for_delta(delta, threshold):
    if delta is None or abs(delta) < threshold:
        return "FLAT"
    return "DOWN" if delta > 0.0 else "UP"


def replay_series(rows, metric_name, threshold, policy):
    """Replay one metric series; lower values are better.

    UP means a proposed +5% LR nudge after sustained improvement. DOWN means
    a proposed -5% nudge after sustained degradation. Every proposal is
    observational only.
    """
    output = []
    ema = None
    previous_ema = None
    streak_signal = None
    streak_length = 0
    cooldown_remaining = 0
    for row in rows:
        value = float(row[metric_name])
        ema = value if ema is None else policy.ema_beta * ema + (1.0 - policy.ema_beta) * value
        delta = None if previous_ema is None else ema - previous_ema
        signal = signal_for_delta(delta, threshold)

        if cooldown_remaining > 0:
            action = "KEEP"
            reason = "cooldown"
            cooldown_remaining -= 1
            streak_signal = None
            streak_length = 0
        else:
            if signal == "FLAT":
                streak_signal = None
                streak_length = 0
            elif signal == streak_signal:
                streak_length += 1
            else:
                streak_signal = signal
                streak_length = 1

            if signal != "FLAT" and streak_length >= policy.patience:
                action = signal
                reason = "sustained_improvement" if action == "UP" else "sustained_degradation"
                cooldown_remaining = policy.cooldown_observations
                streak_signal = None
                streak_length = 0
            else:
                action = "KEEP"
                reason = "below_threshold" if signal == "FLAT" else "patience"

        output.append(
            {
                "value": value,
                "ema": ema,
                "ema_delta": delta,
                "signal": signal,
                "action": action,
                "reason": reason,
                "cooldown_remaining": cooldown_remaining,
            }
        )
        previous_ema = ema
    return output


def count_reversals(actions):
    active = [action for action in actions if action != "KEEP"]
    return sum(left != right for left, right in zip(active, active[1:]))


def replay(trajectories, base_threshold, policy):
    threshold = base_threshold * policy.threshold_multiplier
    rows_out = []
    action_counts = Counter()
    reference_action_counts = Counter()
    reversals = 0
    reference_reversals = 0
    action_disagreements = 0
    active_action_disagreements = 0

    for (run, member), rows in trajectories.items():
        proxy = replay_series(rows, "proxy_metric", threshold, policy)
        reference = replay_series(rows, "reference_metric", threshold, policy)
        reversals += count_reversals([item["action"] for item in proxy])
        reference_reversals += count_reversals([item["action"] for item in reference])
        for row, proxy_item, reference_item in zip(rows, proxy, reference):
            action_counts[proxy_item["action"]] += 1
            reference_action_counts[reference_item["action"]] += 1
            disagrees = proxy_item["action"] != reference_item["action"]
            action_disagreements += int(disagrees)
            active_action_disagreements += int(
                disagrees
                and (proxy_item["action"] != "KEEP" or reference_item["action"] != "KEEP")
            )
            rows_out.append(
                {
                    "run": run,
                    "member": member,
                    "epoch": row["epoch"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "proxy_metric": row["proxy_metric"],
                    "reference_metric": row["reference_metric"],
                    "proxy_ema": proxy_item["ema"],
                    "proxy_ema_delta": proxy_item["ema_delta"],
                    "proxy_signal": proxy_item["signal"],
                    "proposed_action": proxy_item["action"],
                    "proposal_reason": proxy_item["reason"],
                    "reference_ema": reference_item["ema"],
                    "reference_ema_delta": reference_item["ema_delta"],
                    "reference_signal": reference_item["signal"],
                    "reference_action": reference_item["action"],
                    "action_agrees_with_reference": not disagrees,
                }
            )
    return {
        "threshold": threshold,
        "rows": rows_out,
        "action_counts": {name: action_counts[name] for name in ("UP", "KEEP", "DOWN")},
        "reference_action_counts": {
            name: reference_action_counts[name] for name in ("UP", "KEEP", "DOWN")
        },
        "action_reversals": reversals,
        "reference_action_reversals": reference_reversals,
        "action_disagreements": action_disagreements,
        "active_action_disagreements": active_action_disagreements,
    }


def direction_disagreements(trajectories, threshold):
    output = []
    for run, member, previous, current in adjacent_pairs(trajectories):
        proxy_delta = current["proxy_metric"] - previous["proxy_metric"]
        reference_delta = current["reference_metric"] - previous["reference_metric"]
        if proxy_delta * reference_delta >= 0.0:
            continue
        output.append(
            {
                "run": run,
                "member": member,
                "from_epoch": previous["epoch"],
                "to_epoch": current["epoch"],
                "proxy_delta": proxy_delta,
                "reference_delta": reference_delta,
                "proxy_change_below_threshold": abs(proxy_delta) < threshold,
            }
        )
    return output


def sensitivity_grid(trajectories, base_threshold):
    output = []
    for beta, multiplier, patience, cooldown in product(
        (0.3, 0.5, 0.7), (0.75, 1.0, 1.25), (2, 3), (1, 2, 3)
    ):
        policy = ReplayPolicy(
            ema_beta=beta,
            threshold_multiplier=multiplier,
            patience=patience,
            cooldown_observations=cooldown,
        )
        result = replay(trajectories, base_threshold, policy)
        output.append(
            {
                **asdict(policy),
                "threshold": result["threshold"],
                **{name.lower(): count for name, count in result["action_counts"].items()},
                "action_reversals": result["action_reversals"],
                "active_action_disagreements": result["active_action_disagreements"],
            }
        )
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Replay output contains a non-finite float")
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    trajectories = load_trajectories(args.results, args.candidate)
    noise = estimate_change_noise(trajectories)
    policy = ReplayPolicy()
    replay_result = replay(trajectories, noise["robust_sigma"], policy)
    disagreements = direction_disagreements(trajectories, replay_result["threshold"])
    sensitivity = sensitivity_grid(trajectories, noise["robust_sigma"])

    output_dir = args.output_dir
    actions_path = output_dir / "actions.csv"
    sensitivity_path = output_dir / "sensitivity.csv"
    summary_path = output_dir / "summary.json"
    write_csv(actions_path, replay_result.pop("rows"))
    write_csv(sensitivity_path, sensitivity)

    active_counts = [row["up"] + row["down"] for row in sensitivity]
    mismatch_variants = sum(row["active_action_disagreements"] > 0 for row in sensitivity)
    summary = finite_json(
        {
            "schema_version": 1,
            "observational_only": True,
            "candidate": args.candidate,
            "input_results": str(args.results),
            "input_results_sha256": sha256(args.results),
            "trajectory_count": len(trajectories),
            "observation_count": sum(len(rows) for rows in trajectories.values()),
            "noise_estimate": noise,
            "policy": asdict(policy),
            "base_replay": replay_result,
            "raw_adjacent_direction_disagreements": disagreements,
            "raw_direction_disagreement_counts": {
                "total": len(disagreements),
                "below_base_threshold": sum(
                    item["proxy_change_below_threshold"] for item in disagreements
                ),
                "at_or_above_base_threshold": sum(
                    not item["proxy_change_below_threshold"] for item in disagreements
                ),
            },
            "sensitivity": {
                "configuration_count": len(sensitivity),
                "active_action_count_min": min(active_counts),
                "active_action_count_max": max(active_counts),
                "up_count_min": min(row["up"] for row in sensitivity),
                "up_count_max": max(row["up"] for row in sensitivity),
                "keep_count_min": min(row["keep"] for row in sensitivity),
                "keep_count_max": max(row["keep"] for row in sensitivity),
                "down_count_min": min(row["down"] for row in sensitivity),
                "down_count_max": max(row["down"] for row in sensitivity),
                "maximum_action_reversals": max(row["action_reversals"] for row in sensitivity),
                "maximum_active_action_disagreements": max(
                    row["active_action_disagreements"] for row in sensitivity
                ),
                "configurations_with_active_action_disagreement": mismatch_variants,
            },
            "interpretation": {
                "stable_enough_for_shadow": (
                    replay_result["action_reversals"] == 0
                    and replay_result["active_action_disagreements"] == 0
                    and max(row["action_reversals"] for row in sensitivity) == 0
                ),
                "not_evidence_for_live_adaptive_lr": True,
                "limitations": [
                    "historical observations are sparse and irregularly spaced",
                    "the base replay produced no DOWN proposal, so degradation handling needs shadow coverage",
                ],
            },
            "artifacts": {
                "actions": str(actions_path),
                "sensitivity": str(sensitivity_path),
                "summary": str(summary_path),
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
