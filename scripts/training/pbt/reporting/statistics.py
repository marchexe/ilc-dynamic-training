#!/usr/bin/env python3
"""Cross-tier (control/monitor/full) correlation, ranking-agreement,
proxy-overfitting diagnostics, and the LR-vs-mistag-score population
correlation."""

import math
import warnings
from statistics import median as _median

from training.pbt.reporting.constants import TOTAL_SCORE_COLUMN
from training.pbt.reporting.metrics_rows import _better

def _pearson_spearman(xs, ys):
    """(n, pearson_r, pearson_p, spearman_rho, spearman_p, reason) shared by
    every paired-observation correlation in this module -- None (with a
    reason) rather than a number when there are too few points, scipy isn't
    installed, or one side is constant (scipy returns nan without raising
    in that case -- e.g. a fixture/short run where every observation has
    the identical value; caught here so a NaN never leaks into a report as
    a fabricated-looking coefficient)."""
    n = len(xs)
    if n < 3:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "insufficient_paired_observations"}
    try:
        from scipy import stats
    except ImportError:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "scipy_unavailable"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson_r, pearson_p = stats.pearsonr(xs, ys)
        spearman_rho, spearman_p = stats.spearmanr(xs, ys)
    if math.isnan(pearson_r) or math.isnan(spearman_rho):
        return {"n": n, "pearson_r": None, "spearman_rho": None, "reason": "zero_variance"}
    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "reason": None,
    }


def _percentile(sorted_values, q):
    """Linear-interpolation percentile (matches numpy's default method) of
    already-sorted values -- a single value for n=1 rather than raising,
    since a population of 1 still has a well-defined (degenerate) band."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    index = q / 100 * (n - 1)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def generation_score_band(member_rows, metric_column=TOTAL_SCORE_COLUMN):
    """Per-generation (median, q25, q75, n) across every population member
    with a valid `metric_column` value that generation -- one point per
    generation, sorted. IQR (25th-75th percentile) rather than min/max: a
    population this small (typically single digits of members) makes
    min/max mostly noise from whichever single member happened to be best/
    worst that generation, not a stable notion of "typical spread". The
    single shared definition of "typical for this generation": the report
    figure's training-dynamics panel bands the population around this
    median, and lr_mistag_correlation below detrends by it, so both views
    of a run agree on what "typical" means by construction rather than by
    two independently-written formulas happening to match.
    """
    scores_by_generation = {}
    for row in member_rows:
        generation = row.get("generation")
        value = row.get(metric_column)
        if generation is None or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            scores_by_generation.setdefault(generation, []).append(value)
    result = []
    for generation, values in sorted(scores_by_generation.items()):
        ordered = sorted(values)
        result.append({
            "generation": generation,
            "median": _median(values),
            "q25": _percentile(ordered, 25),
            "q75": _percentile(ordered, 75),
            "n": len(values),
        })
    return result


def lr_mistag_correlation(member_rows, metric_column=TOTAL_SCORE_COLUMN):
    """Pearson r (on log10(LR) -- LR is configured/explored on a log scale
    and spans multiple decades, so any real effect is expected to be
    multiplicative, not additive) and Spearman rho (rank-based, invariant to
    that transform either way) between LR and a *detrended* `metric_column`:
    each row's value minus its own generation's median, from
    generation_score_band above (median across every row with a valid
    value that generation, not just the LR-valid ones).

    Pooling the raw metric across generations would conflate any real LR
    effect with ordinary training progress -- the score improves over
    generations regardless of LR, so generation itself is a confound (e.g.
    on this repo's own anchor_copy_lr_recenter 8h showcase run, generation
    alone correlates with total_mistag_score at Pearson r=-0.37, comparable
    in size to the confounded raw LR correlation, which is not a
    coincidence: most of that raw correlation is the training-progress
    trend, not an LR effect). Detrending isolates "did this LR do
    better/worse than typical for its own generation," not "did later
    (=more trained) generations do better" -- one population-wide point per
    (member, generation) observation with both values present, not one per
    generation or per winner only.

    Returns the usual {"n", "pearson_r", "pearson_p", "spearman_rho",
    "spearman_p", "reason"} from _pearson_spearman, plus "pairs":
    [(log10_lr, residual), ...] -- the actual detrended points, exposed so
    a caller that wants to plot them (e.g. the report figure) never
    recomputes the detrending independently.
    """
    generation_median = {row["generation"]: row["median"] for row in generation_score_band(member_rows, metric_column)}

    pairs = []
    for row in member_rows:
        generation = row.get("generation")
        lr = row.get("LR")
        value = row.get(metric_column)
        if generation not in generation_median or lr is None or value is None:
            continue
        try:
            lr = float(lr)
            value = float(value)
        except (TypeError, ValueError):
            continue
        if lr > 0 and math.isfinite(value):
            pairs.append((math.log10(lr), value - generation_median[generation]))

    result = _pearson_spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    result["pairs"] = pairs
    return result


def ols_trend_line(pairs):
    """(slope, intercept) of the ordinary-least-squares line through
    (x, y) `pairs`, or None if there are fewer than 2 points or every x is
    identical (a vertical "fit" is undefined) -- never a fabricated flat
    or NaN line. Plain closed-form OLS, no scipy/numpy dependency, since
    this is a visual aid for the LR-vs-residual scatter, not a reported
    statistic in its own right."""
    n = len(pairs)
    if n < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    return slope, mean_y - slope * mean_x


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
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return _pearson_spearman(xs, ys)


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
