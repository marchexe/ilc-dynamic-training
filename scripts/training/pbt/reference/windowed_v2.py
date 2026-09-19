"""Frozen windowed_pbt_v2 decision semantics (reference revision a4f7507).

Keep scoring, tie handling, exploration, loss counters and terminal behavior
unchanged. New adaptive policies must use a different strategy version.
"""
import itertools
import math
from statistics import mean

from training.runtime import WORKING_POINT_DEFINITION

REFERENCE_REVISION = "a4f750793911508812b5b09296afe5306306c930"

STRATEGY = "windowed_pbt_v2"
WORKING_POINTS = tuple(f"validation_{pair}_mistag_eff_{eff:.2f}_percent" for pair, eff in WORKING_POINT_DEFINITION)


def exceeds_margin(gap, margin):
    return gap > margin and not math.isclose(gap, margin, rel_tol=1e-12, abs_tol=1e-15)


def _mutations(config, members, donor, recipients):
    """Maximize distinct children, then balanced directions and log-LR separation.

    Exhaustive search is at most nine assignments (two children, two factors plus
    no-op). Clamped duplicates are skipped instead of silently collapsing diversity.
    """
    p = config['pbt']
    donor_lr = members[donor]['lr']
    best_key, best = None, []
    for factors in itertools.product([None, *p['mutation_factors']], repeat=len(recipients)):
        lrs = {n: m['lr'] for n, m in members.items()}
        events = []
        for name, factor in zip(recipients, factors):
            if factor is None:
                continue
            lr = min(p['max_lr'], max(p['min_lr'], donor_lr * factor))
            lrs[name] = lr
            events.append(dict(source='population', strategy=STRATEGY, donor=donor, recipient=name,
                               donor_lr=donor_lr, recipient_lr=members[name]['lr'], new_lr=lr,
                               mutation_factor=factor, effective_factor=lr / donor_lr,
                               reason='persistent_window_loss', applied=False))
        distances = [abs(math.log(lrs[e['recipient']] / lr)) for e in events
                     for n, lr in lrs.items() if n != e['recipient']]
        if any(d < 1e-12 for d in distances):
            continue
        key = (len(events), len({e['mutation_factor'] for e in events}), min(distances, default=0),
               sum(distances), tuple(lrs[n] for n in recipients))
        if best_key is None or key > best_key:
            best_key, best = key, events
    return best


def windowed_pbt_v2_plan(config, generation, members, manifest=None):
    """Attach a complete decision record at boundaries; never mutate live members."""
    p = config['pbt']; options = p['windowed_pbt_v2']
    index = generation['index']; width = options['window_epochs']; metric = p['metric']
    raw = sorted(members, key=lambda n: (generation['workers'][n]['metrics'][metric], n))
    if (index + 1) % width:
        return raw, []
    history = {g['index']: g for g in (manifest or {}).get('generations', []) if g['index'] < index}
    history[index] = generation
    indices = list(range(index + 1 - width, index + 1))
    if any(i not in history for i in indices):
        raise ValueError('Window evidence is incomplete')
    previous = history.get(index - width, {}).get('windowed_pbt_v2', {})
    if index >= width and not previous:
        raise ValueError('Previous window decision is missing')
    evidence = {}
    keys = [metric, 'validation_loss', *WORKING_POINTS]
    for name in members:
        workers = [history[i]['workers'][name] for i in indices]
        if any(w.get('status') != 'completed' or w.get('returncode') != 0 for w in workers):
            raise ValueError('Every window epoch must complete before any decision')
        if any(w['lr'] != members[name]['lr'] for w in workers):
            raise ValueError('LR changed inside a window')
        values = {key: [w['metrics'][key] for w in workers] for key in keys}
        if any(not math.isfinite(v) for vs in values.values() for v in vs):
            raise ValueError('Non-finite window evidence; stop without exploitation')
        ys = values[metric]; center = (width - 1) / 2
        slope = sum((i - center) * y for i, y in enumerate(ys)) / sum((i - center)**2 for i in range(width))
        evidence[name] = dict(lr=members[name]['lr'], window_score=mean(ys[-options['score_epochs']:]),
                              mean_window=mean(ys), slope=slope, final_metric=ys[-1],
                              metrics={k: dict(mean_window=mean(v), mean_score=mean(v[-options['score_epochs']:]),
                                               final=v[-1]) for k, v in values.items()})
    ranking = sorted(evidence, key=lambda n: (evidence[n]['window_score'], n))
    donor = ranking[0]; best_score = evidence[donor]['window_score']; margin = options['decision_margin']
    group, group_score = 0, best_score
    for rank, name in enumerate(ranking, 1):
        item = evidence[name]; gap = item['window_score'] - best_score
        if exceeds_margin(item['window_score'] - group_score, margin):
            group += 1; group_score = item['window_score']
        losing = exceeds_margin(gap, margin)
        previous_count = previous.get('members', {}).get(name, {}).get('next_loss_count', 0)
        count = previous_count + 1 if losing else 0
        item.update(rank=rank, tie_group=group, donor_gap=gap, consecutive_losses=count, next_loss_count=count,
                    eligible=count >= options['losing_windows'], copied=False,
                    reason='persistent_loss' if count >= options['losing_windows'] else 'first_loss' if losing else 'within_margin')
    eligible = [n for n in reversed(ranking) if evidence[n]['eligible']][:options['max_recipients']]
    for name in ranking:
        if evidence[name]['eligible'] and name not in eligible:
            evidence[name]['reason'] = 'recipient_limit'
    terminal = index + 1 == config['shared']['generations']
    plan = [] if terminal else _mutations(config, members, donor, eligible)
    selected = {e['recipient'] for e in plan}
    for name in eligible:
        evidence[name]['reason'] = 'terminal_boundary' if terminal else 'selected' if name in selected else 'no_distinct_mutation'
    for name in selected:
        evidence[name]['next_loss_count'] = 0
    generation['windowed_pbt_v2'] = dict(window_number=(index + 1) // width, generation_indices=indices,
        epochs=[history[i]['epoch'] for i in indices], full_epochs=[i + 1 for i in indices],
        decision_margin=margin, score_epochs=options['score_epochs'], members=evidence,
        donor=donor, terminal=terminal, protected_best_update=None)
    return ranking, plan
