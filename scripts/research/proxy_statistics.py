#!/usr/bin/env python3
"""Proxy-vs-full-validation statistics for the nightly checkpoint audit.

Reuses scripts/training/pbt/reporting/statistics.py wherever the shape
fits -- Pearson/Spearman correlation (tier_correlation), ranking agreement
(ranking_agreement), and best-checkpoint-by-tier are unmodified from
there, fed a synthetic manifest shaped the same way a live PBT run's
manifest.json is (one "generation" round per tier, each audited checkpoint
as a "member"). corroboration_status is NOT reused: it compares a
generation-history baseline ("-1"/"initial_resume") against a
control-selected checkpoint across live-run generations, a shape this
audit's N-independent-checkpoints comparison doesn't have -- forcing it in
would mean fabricating a baseline round that doesn't exist. What that
module doesn't already provide -- Kendall tau and explicit
pairwise-direction agreement -- is implemented here.

No event-level predictions exist anywhere in this codebase (Weaver's
`--predict-output` is never wired into any command builder here), so
bootstrap confidence intervals are not computable from what this audit can
gather. Every function below returns an explicit "unavailable" reason in
that case rather than fabricating an uncertainty estimate.
"""

import math

from training.pbt.reporting.statistics import (
    best_checkpoint_by_tier,
    ranking_agreement,
    tier_correlation,
)

CONTROL_TIER = "control"
FULL_TIER = "full_holdout"


def _is_finite(value):
    return value is not None and math.isfinite(float(value))


def build_round(tier, dataset, suffix, member_results, metric_name, mode):
    """One tiered_evaluations round, shaped like
    reporting/events.py::record_tiered_evaluation_round's output (ranking
    sorted by mode, missing/non-finite metrics excluded) -- but built
    in-memory only, with no run-directory side effects, since the audit
    has its own output layout and isn't a live PBT run."""
    ranking = sorted(
        (
            name
            for name, record in member_results.items()
            if _is_finite((record.get("metrics") or {}).get(metric_name))
        ),
        key=lambda name: float(member_results[name]["metrics"][metric_name]),
        reverse=(mode == "max"),
    )
    return {
        "schema_version": 1,
        "generation": 0,
        "tier": tier,
        "dataset": dataset,
        "suffix": suffix,
        "metric_name": metric_name,
        "mode": mode,
        "members": member_results,
        "ranking": ranking,
    }


def build_synthetic_manifest(control_results, full_results, metric_name, mode):
    """control_results/full_results: {checkpoint_id: {"status": ..., "metrics": {...}}}
    for control_proxy_50k and full_validation respectively. Produces a
    manifest-shaped dict that reporting/statistics.py's tier_correlation,
    ranking_agreement, best_checkpoint_by_tier, and corroboration_status
    all accept unmodified -- every audited checkpoint becomes a "member"
    observed once, at synthetic generation 0, under tiers "control" and
    "full_holdout" (the same tier names the live PBT tiered-validation
    system uses, so no separate vocabulary is introduced)."""
    control_round = build_round(CONTROL_TIER, None, None, control_results, metric_name, mode)
    full_round = build_round(FULL_TIER, None, None, full_results, metric_name, mode)
    return {
        "config": {"pbt": {"mode": mode}},
        "tiered_evaluations": [control_round, full_round],
        "best": {"generation": 0, "member": control_round["ranking"][0] if control_round["ranking"] else None},
    }


def kendall_tau(manifest, tier_a=CONTROL_TIER, tier_b=FULL_TIER):
    """Kendall's tau-b between tier_a and tier_b's metric values at every
    checkpoint evaluated on both. Mirrors tier_correlation's n<3 and
    scipy-unavailable guards -- never fabricate a coefficient from too few
    paired points."""
    pairs = _paired_values(manifest, tier_a, tier_b)
    n = len(pairs)
    if n < 3:
        return {"n": n, "tau": None, "p_value": None, "reason": "insufficient_paired_observations"}
    try:
        from scipy import stats
    except ImportError:
        return {"n": n, "tau": None, "p_value": None, "reason": "scipy_unavailable"}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    tau, p_value = stats.kendalltau(xs, ys)
    return {"n": n, "tau": None if tau is None else float(tau), "p_value": None if p_value is None else float(p_value), "reason": None}


def _paired_values(manifest, tier_a, tier_b):
    by_tier = {round_record["tier"]: round_record for round_record in manifest.get("tiered_evaluations", [])}
    if tier_a not in by_tier or tier_b not in by_tier:
        return []
    round_a, round_b = by_tier[tier_a], by_tier[tier_b]
    metric_name = round_a["metric_name"]
    pairs = []
    for name, record_a in round_a["members"].items():
        record_b = round_b["members"].get(name)
        if record_b is None:
            continue
        value_a = (record_a.get("metrics") or {}).get(metric_name)
        value_b = (record_b.get("metrics") or {}).get(metric_name)
        if not _is_finite(value_a) or not _is_finite(value_b):
            continue
        pairs.append((float(value_a), float(value_b), name))
    return pairs


def pairwise_direction_agreement(manifest, tier_a=CONTROL_TIER, tier_b=FULL_TIER):
    """For every pair of checkpoints, do tier_a and tier_b agree on which
    one is better (per the manifest's configured min/max mode)? Reports the
    fraction of agreeing pairs plus the explicit list of disagreements
    (checkpoint pairs where the tiers pick opposite winners) -- this is
    what section 6 of the task calls "pairwise direction agreement",
    distinct from the rank-correlation coefficients above."""
    pairs = _paired_values(manifest, tier_a, tier_b)
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "min")
    n = len(pairs)
    if n < 2:
        return {
            "n_checkpoints": n,
            "n_pairs": 0,
            "agreeing_pairs": 0,
            "agreement_fraction": None,
            "disagreements": [],
            "reason": "insufficient_paired_observations",
        }
    better = (lambda x, y: x > y) if mode == "max" else (lambda x, y: x < y)
    total_pairs = 0
    agreeing = 0
    disagreements = []
    for i in range(len(pairs)):
        value_a_i, value_b_i, name_i = pairs[i]
        for j in range(i + 1, len(pairs)):
            value_a_j, value_b_j, name_j = pairs[j]
            if value_a_i == value_a_j or value_b_i == value_b_j:
                # A genuine tie on either tier has no well-defined "better"
                # direction to compare -- exclude rather than force a
                # spurious agree/disagree call.
                continue
            total_pairs += 1
            tier_a_prefers_i = better(value_a_i, value_a_j)
            tier_b_prefers_i = better(value_b_i, value_b_j)
            if tier_a_prefers_i == tier_b_prefers_i:
                agreeing += 1
            else:
                disagreements.append(
                    {
                        "checkpoint_a": name_i,
                        "checkpoint_b": name_j,
                        f"{tier_a}_prefers": name_i if tier_a_prefers_i else name_j,
                        f"{tier_b}_prefers": name_i if tier_b_prefers_i else name_j,
                    }
                )
    return {
        "n_checkpoints": n,
        "n_pairs": total_pairs,
        "agreeing_pairs": agreeing,
        "agreement_fraction": (agreeing / total_pairs) if total_pairs else None,
        "disagreements": disagreements,
        "reason": None if total_pairs else "no_non_tied_pairs",
    }


def best_checkpoint_agreement(manifest):
    """Does control_proxy_50k's best checkpoint match full_validation's
    best checkpoint? Reuses best_checkpoint_by_tier unmodified."""
    best_by_tier = best_checkpoint_by_tier(manifest)
    control_best = best_by_tier.get(CONTROL_TIER)
    full_best = best_by_tier.get(FULL_TIER)
    if control_best is None or full_best is None:
        return {"agrees": None, "control_best": control_best, "full_best": full_best, "reason": "missing_tier_data"}
    return {
        "agrees": control_best["member"] == full_best["member"],
        "control_best": control_best,
        "full_best": full_best,
        "reason": None,
    }


def full_summary(control_results, full_results, metric_name, mode):
    """Everything section 6 of the task asks for, in one call: Spearman +
    Pearson (tier_correlation, reused), Kendall (new), ranking agreement
    (reused), pairwise direction agreement (new), best-checkpoint agreement
    (reused via best_checkpoint_by_tier), and corroboration status
    (reused)."""
    manifest = build_synthetic_manifest(control_results, full_results, metric_name, mode)
    correlation = tier_correlation(manifest, CONTROL_TIER, FULL_TIER)
    kendall = kendall_tau(manifest, CONTROL_TIER, FULL_TIER)
    agreement_rounds = ranking_agreement(manifest, CONTROL_TIER, FULL_TIER)
    pairwise = pairwise_direction_agreement(manifest, CONTROL_TIER, FULL_TIER)
    best_agreement = best_checkpoint_agreement(manifest)
    n_evaluated = len(_paired_values(manifest, CONTROL_TIER, FULL_TIER))
    insufficient_evidence = n_evaluated < 3
    return {
        "n_checkpoints_paired": n_evaluated,
        "insufficient_evidence": insufficient_evidence,
        "pearson_spearman": correlation,
        "kendall": kendall,
        "ranking_agreement": agreement_rounds[0] if agreement_rounds else None,
        "pairwise_direction_agreement": pairwise,
        "best_checkpoint_agreement": best_agreement,
        "bootstrap_confidence_intervals": "unavailable: no event-level predictions are stored by any code path in this repository (weaver --predict-output is never passed); only aggregate per-checkpoint metrics exist, so paired bootstrap cannot be computed",
    }
