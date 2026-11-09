"""Per-member learning-rate schedules for fixed_lr_grid runs.

Every generation is one full epoch, Weaver runs with --lr-scheduler none and loads the member's LR from
--start-lr (--override-load-lr). A schedule therefore sets the member's LR in the manifest when a generation
is first started; the worker command, the worker record and the checkpoint all follow from it.

kinds (G = shared.generations, g = generation index from 0, LR = start_lr * factor(g)):

* flat_decay: Weaver's flat+decay (MultiStepLR) over G epochs. The last n = max(1, int(G * decay_fraction))
  epochs each multiply the LR by final_factor ** (1 / n), so the last epoch trains at start_lr * final_factor.
* cosine: from start_lr at g = 0 to start_lr * final_factor at g = G - 1, half a cosine period.
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


def apply_lr_schedule(config, manifest, generation):
    """Set the LR of every scheduled member for a generation that is being started; return the changes."""
    changes = {}
    for member in config['population']:
        if not member.get('lr_schedule'):
            continue
        lr = scheduled_lr(member, generation, int(config['shared']['generations']))
        live = manifest['members'][member['name']]
        if live['lr'] != lr:
            changes[member['name']] = dict(old_lr=live['lr'], new_lr=lr)
            live['lr'] = lr
    return changes
