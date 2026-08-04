#!/usr/bin/env python3
"""Cross-tier (control/monitor/full) correlation, ranking-agreement, and
proxy-overfitting diagnostics."""

import math

from training.pbt.reporting.metrics_rows import _better

def _paired_tier_values(manifest, tier_a, tier_b):
    """(x, y, member, generation) tuples for every (member, generation) that
    has a valid metric under both tier_a and tier_b -- the paired
    observations proxy-vs-monitor/full correlation requires.
    """
    by_key = {}
    for round_record in manifest.get("tiered_evaluations", []):
        tier = round_record.get("tier")
        if tier not in (tier_a, tier_b):
            continue
        metric_name = round_record.get("metric_name")
        generation = round_record.get("generation")
        for member, record in (round_record.get("members") or {}).items():
            value = (record.get("metrics") or {}).get(metric_name)
            if value is None or not math.isfinite(float(value)):
                continue
            by_key.setdefault((generation, member), {})[tier] = float(value)
    pairs = []
    for (generation, member), values in sorted(by_key.items(), key=lambda item: (item[0][0] or -999, item[0][1] or "")):
        if tier_a in values and tier_b in values:
            pairs.append((values[tier_a], values[tier_b], member, generation))
    return pairs


def tier_correlation(manifest, tier_a, tier_b):
    """Pearson r and Spearman rho between tier_a and tier_b's metric values
    at every (member, generation) evaluated on both. Returns None (with a
    reason) rather than a number when there are too few paired points for a
    correlation to mean anything -- never fabricate significance from n<3.
    """
    pairs = _paired_tier_values(manifest, tier_a, tier_b)
    n = len(pairs)
    if n < 3:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "insufficient_paired_observations"}
    try:
        from scipy import stats
    except ImportError:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "scipy_unavailable"}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    pearson_r, pearson_p = stats.pearsonr(xs, ys)
    spearman_rho, spearman_p = stats.spearmanr(xs, ys)
    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "reason": None,
    }


def _paired_round_rankings(manifest, tier_a, tier_b):
    """(generation, ranking_a, ranking_b, shared_members) for every
    generation where both tiers recorded a round -- the full-ordering data
    ranking-agreement/top-k-agreement is computed from.
    """
    rounds_by_generation = {}
    for round_record in manifest.get("tiered_evaluations", []):
        rounds_by_generation.setdefault(round_record.get("generation"), {})[round_record.get("tier")] = round_record
    paired = []
    for generation, by_tier in sorted(rounds_by_generation.items(), key=lambda item: item[0] if item[0] is not None else -999):
        if tier_a not in by_tier or tier_b not in by_tier:
            continue
        ranking_a = by_tier[tier_a].get("ranking") or []
        ranking_b = by_tier[tier_b].get("ranking") or []
        shared = [name for name in ranking_a if name in ranking_b]
        if len(shared) < 2:
            continue
        paired.append((generation, ranking_a, ranking_b, shared))
    return paired


def ranking_agreement(manifest, tier_a, tier_b):
    """Per paired-generation ranking agreement between tier_a and tier_b:
    Spearman rho of rank positions (restricted to members ranked by both),
    plus top-1 and top-min(3,n) overlap -- "does the proxy pick the same
    winner/leaders as monitor/full", not just "are the raw values correlated".
    """
    paired = _paired_round_rankings(manifest, tier_a, tier_b)
    results = []
    for generation, ranking_a, ranking_b, shared in paired:
        rank_a = {name: index for index, name in enumerate(ranking_a) if name in shared}
        rank_b = {name: index for index, name in enumerate(ranking_b) if name in shared}
        ordered = sorted(shared, key=lambda name: rank_a[name])
        ranks_a = [rank_a[name] for name in ordered]
        ranks_b = [rank_b[name] for name in ordered]
        spearman_rho = None
        if len(ordered) >= 3:
            try:
                from scipy import stats

                spearman_rho, _ = stats.spearmanr(ranks_a, ranks_b)
                spearman_rho = float(spearman_rho)
            except ImportError:
                spearman_rho = None
        top1_a = ranking_a[0] if ranking_a else None
        top1_b = ranking_b[0] if ranking_b else None
        k = min(3, len(shared))
        top_k_a = set(ranking_a[:k])
        top_k_b = set(ranking_b[:k])
        results.append(
            {
                "generation": generation,
                "members_compared": len(shared),
                "spearman_rho": spearman_rho,
                "top1_agrees": bool(top1_a is not None and top1_a == top1_b),
                "top_k": k,
                "top_k_overlap_fraction": len(top_k_a & top_k_b) / k if k else None,
            }
        )
    return results


def best_checkpoint_by_tier(manifest):
    """The (generation, member, metric_value) with the best metric under
    each tier, across every recorded round of that tier -- so we can ask
    "does control's favorite checkpoint match monitor/full's".
    """
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    best_by_tier = {}
    for round_record in manifest.get("tiered_evaluations", []):
        tier = round_record.get("tier")
        metric_name = round_record.get("metric_name")
        for member, record in (round_record.get("members") or {}).items():
            value = (record.get("metrics") or {}).get(metric_name)
            if value is None or not math.isfinite(float(value)):
                continue
            value = float(value)
            candidate = {"generation": round_record.get("generation"), "member": member, "metric_value": value}
            current = best_by_tier.get(tier)
            if current is None or _better(mode, value, current["metric_value"]):
                best_by_tier[tier] = candidate
    return best_by_tier


def proxy_selected_checkpoint_other_tiers(manifest):
    """What do monitor/full say about the checkpoint control-based PBT
    actually selected as global best? None entries mean that checkpoint was
    never evaluated on that tier.
    """
    best = manifest.get("best") or {}
    generation = best.get("generation")
    member = best.get("member")
    out = {}
    for round_record in manifest.get("tiered_evaluations", []):
        if round_record.get("generation") != generation or round_record.get("tier") == "control":
            continue
        record = (round_record.get("members") or {}).get(member)
        if record is None:
            continue
        metric_name = round_record.get("metric_name")
        out[round_record.get("tier")] = {
            "metric_value": (record.get("metrics") or {}).get(metric_name),
            "status": record.get("status"),
        }
    return {"generation": generation, "member": member, "tiers": out}


def proxy_overfitting_cases(manifest, tier_b="monitor"):
    """(member, generation_from, generation_to) triples where control
    improved but tier_b (monitor by default) did not, for the same member
    across the same pair of paired generations -- the explicit "proxy got
    better, physics didn't" signal this whole diagnostic exists to catch.
    A control-only improvement must never be reported as a confirmed
    physics result; this is the concrete evidence for when that would be wrong.
    """
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    paired_generations = sorted(
        {round_record.get("generation") for round_record in manifest.get("tiered_evaluations", []) if round_record.get("tier") == tier_b}
    )
    by_generation_tier = {}
    for round_record in manifest.get("tiered_evaluations", []):
        by_generation_tier.setdefault(round_record.get("generation"), {})[round_record.get("tier")] = round_record

    def metric_for(generation, tier, member):
        round_record = by_generation_tier.get(generation, {}).get(tier)
        if not round_record:
            return None
        metric_name = round_record.get("metric_name")
        value = (round_record.get("members", {}).get(member, {}).get("metrics") or {}).get(metric_name)
        return None if value is None else float(value)

    cases = []
    for previous_generation, current_generation in zip(paired_generations, paired_generations[1:]):
        control_prev = by_generation_tier.get(previous_generation, {}).get("control")
        control_curr = by_generation_tier.get(current_generation, {}).get("control")
        if not control_prev or not control_curr:
            continue
        members = set(control_prev.get("members", {})) & set(control_curr.get("members", {}))
        for member in sorted(members):
            control_before = metric_for(previous_generation, "control", member)
            control_after = metric_for(current_generation, "control", member)
            other_before = metric_for(previous_generation, tier_b, member)
            other_after = metric_for(current_generation, tier_b, member)
            if None in (control_before, control_after, other_before, other_after):
                continue
            control_improved = _better(mode, control_after, control_before)
            other_improved = _better(mode, other_after, other_before)
            if control_improved and not other_improved:
                cases.append(
                    {
                        "member": member,
                        "generation_from": previous_generation,
                        "generation_to": current_generation,
                        "control_before": control_before,
                        "control_after": control_after,
                        f"{tier_b}_before": other_before,
                        f"{tier_b}_after": other_after,
                    }
                )
    return cases


def _tier_metric_for_member_generation(manifest, tier, generation, member):
    for round_record in manifest.get("tiered_evaluations", []):
        if round_record.get("tier") != tier or round_record.get("generation") != generation:
            continue
        metric_name = round_record.get("metric_name")
        record = (round_record.get("members") or {}).get(member)
        if record is None:
            return None
        value = (record.get("metrics") or {}).get(metric_name)
        return None if value is None else float(value)
    return None


def corroboration_status(manifest):
    """provisional / monitor-corroborated / full-corroborated for the
    control-selected global-best checkpoint: same-direction agreement with
    monitor/full measured at that SAME (member, generation) checkpoint.

    Deliberately not a significance test and not an error-bar-overlap rule
    (see write_report's effect-size/uncertainty lines, reported alongside
    this label, not folded into it) -- this label only says which tiers of
    evidence exist and agree on direction; the human reader judges magnitude.
    """
    mode = manifest.get("config", {}).get("pbt", {}).get("mode", "max")
    best = manifest.get("best") or {}
    generation, member = best.get("generation"), best.get("member")
    status = "provisional"
    details = {}
    for tier in ("monitor", "full"):
        baseline_tier = _tier_metric_for_member_generation(manifest, tier, -1, "initial_resume")
        selected_tier = _tier_metric_for_member_generation(manifest, tier, generation, member)
        if baseline_tier is None or selected_tier is None:
            details[tier] = {"available": False}
            continue
        improved = _better(mode, selected_tier, baseline_tier)
        details[tier] = {"available": True, "baseline": baseline_tier, "selected": selected_tier, "improved": improved}
        if improved:
            status = f"{tier}-corroborated"
    return status, details
