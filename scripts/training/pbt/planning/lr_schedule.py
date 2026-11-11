"""Per-member learning-rate schedules for fixed_lr_grid runs.

Every generation is one full epoch, Weaver runs with --lr-scheduler none and loads the member's LR from
--start-lr (--override-load-lr). A schedule therefore sets the member's LR in the manifest when a generation
is first started; the worker command, the worker record and the checkpoint all follow from it.

kinds (G = shared.generations, g = generation index from 0, LR = start_lr * factor(g)):

* flat_decay: Weaver's flat+decay (MultiStepLR) over G epochs. The last n = max(1, int(G * decay_fraction))
  epochs each multiply the LR by final_factor ** (1 / n), so the last epoch trains at start_lr * final_factor.
* cosine: from start_lr at g = 0 to start_lr * final_factor at g = G - 1, half a cosine period.

windowed_pbt_v2 with final_decay: the last n generations make no decisions (planning/windowed_pbt_v2.py), and
from generation G - n on each member's LR is its LR at that point times the flat_decay factor, so a PBT run ends
like standard ParT training. The base LR is kept in the manifest member as final_decay_base_lr.
"""
import math


def schedule_factor(schedule, generation, generations):
    kind = schedule['kind']
    if not 0 <= generation < generations:
        raise ValueError(f'generation {generation} outside the {generations}-generation schedule')
    final = float(schedule.get('final_factor', 0.01))
    if kind == 'flat_decay':
        decay = max(1, int(generations * float(schedule.get('decay_fraction', 0.3))))
        first = generations - decay
        return (final ** (1.0 / decay)) ** max(0, generation - first + 1)
    if kind == 'cosine':
        if generations == 1:
            return 1.0
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * generation / (generations - 1)))
    raise ValueError(f'Unknown lr_schedule kind: {kind}')


def scheduled_lr(member, generation, generations):
    return float(member['start_lr']) * schedule_factor(member['lr_schedule'], generation, generations)


def final_decay(config):
    pbt = config.get('pbt') or {}
    return (pbt.get('windowed_pbt_v2') or {}).get('final_decay') if pbt.get('strategy') == 'windowed_pbt_v2' else None


def final_decay_start(config):
    """First generation of the final decay (G - n), or None without final_decay."""
    decay = final_decay(config)
    if not decay:
        return None
    generations = int(config['shared']['generations'])
    return generations - max(1, int(generations * float(decay.get('decay_fraction', 0.3))))


def apply_lr_schedule(config, manifest, generation):
    """Set the LR of every scheduled member for a generation that is being started; return the changes."""
    changes = {}
    start = final_decay_start(config)
    if start is not None and generation >= start:
        decay = final_decay(config)
        schedule = dict(kind='flat_decay', decay_fraction=decay.get('decay_fraction', 0.3),
                        final_factor=decay.get('final_factor', 0.01))
        factor = schedule_factor(schedule, generation, int(config['shared']['generations']))
        for name, live in manifest['members'].items():
            base = live.setdefault('final_decay_base_lr', live['lr'])
            lr = base * factor
            if live['lr'] != lr:
                changes[name] = dict(old_lr=live['lr'], new_lr=lr)
                live['lr'] = lr
        return changes
    for member in config['population']:
        if not member.get('lr_schedule'):
            continue
        lr = scheduled_lr(member, generation, int(config['shared']['generations']))
        live = manifest['members'][member['name']]
        if live['lr'] != lr:
            changes[member['name']] = dict(old_lr=live['lr'], new_lr=lr)
            live['lr'] = lr
    return changes
