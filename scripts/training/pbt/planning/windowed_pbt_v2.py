"""Compatibility entry point and unchanged transitions for windowed PBT v2."""
import math
from pathlib import Path

from training.pbt.reference.windowed_v2 import (
    STRATEGY, WORKING_POINTS, exceeds_margin, _mutations, windowed_pbt_v2_plan as reference_plan,
)
from training.checkpoints import bundle_paths, bundle_identity, check_bundle, copy_bundle as _copy_bundle
from training.pbt.state.optimizer_state import atomic_set_optimizer_lr
from training.runtime import atomic_json, utc_now


def windowed_pbt_v2_plan(config, generation, members, manifest=None):
    """The frozen reference decisions, plus fallback copies when `fallback_factors` is set.

    The reference skips a selected loser when both mutation factors give an LR that another member already
    has. With factors 0.5 and 2 the LRs fall on one power-of-two ladder and this repeats window after window,
    so the loser neither copies nor mutates. A fallback copy gives it the donor's weights and the fallback
    factor that keeps its LR farthest from the other LRs. It stays skipped only if every fallback LR is in use.
    """
    ranking, plan = reference_plan(config, generation, members, manifest)
    factors = (config['pbt'].get(STRATEGY) or {}).get('fallback_factors')
    decision = generation.get(STRATEGY)
    if not factors or not decision or decision['terminal']:
        return ranking, plan
    return ranking, plan + _fallback_mutations(config, members, decision, ranking, plan, factors)


def _fallback_mutations(config, members, decision, ranking, plan, factors):
    p = config['pbt']; donor = decision['donor']; donor_lr = members[donor]['lr']
    lrs = {n: m['lr'] for n, m in members.items()}
    lrs.update({e['recipient']: e['new_lr'] for e in plan})
    events = []
    for name in reversed(ranking):
        item = decision['members'][name]
        if item['reason'] != 'no_distinct_mutation':
            continue
        best = None
        for factor in factors:
            lr = min(p['max_lr'], max(p['min_lr'], donor_lr * factor))
            gap = min(abs(math.log(lr / other)) for n, other in lrs.items() if n != name)
            if gap >= 1e-12 and (best is None or gap > best[0]):
                best = (gap, factor, lr)
        if best is None:
            continue
        _, factor, lr = best
        lrs[name] = lr
        events.append(dict(source='population', strategy=STRATEGY, donor=donor, recipient=name,
                           donor_lr=donor_lr, recipient_lr=members[name]['lr'], new_lr=lr,
                           mutation_factor=factor, effective_factor=lr / donor_lr,
                           reason='persistent_window_loss', fallback=True, applied=False))
        item.update(reason='selected_fallback', next_loss_count=0)
    return events


def prepare_boundary(run, manifest, generation):
    """Archive evidence before persisting the plan and before touching any live state."""
    decision = generation.get(STRATEGY)
    if decision is None:
        return
    run = Path(run); epoch = generation['epoch']; donor = decision['donor']
    donor_paths = bundle_paths(run / donor, epoch)
    identity = bundle_identity(donor_paths)
    decision['donor_checkpoint'] = identity
    item = decision['members'][donor]
    protected = manifest.get('protected_best')
    if protected is None or item['window_score'] < protected['window_score']:
        paths = bundle_paths(run / 'checkpoints' / 'protected' / f"window-{decision['window_number']:03d}", epoch)
        archived = _copy_bundle(donor_paths, paths)
        protected = dict(member=donor, epoch=epoch, generation=generation['index'], lr=item['lr'],
                         window_number=decision['window_number'], window_score=item['window_score'],
                         single_epoch_metric=item['final_metric'], window_metrics=item['metrics'],
                         metrics=generation['workers'][donor]['metrics'], checkpoint=archived,
                         state_path=archived['state']['path'], optimizer_path=archived['optimizer']['path'],
                         scaler_path=archived['scaler']['path'])
        decision['protected_best_update'] = protected
    decision['protected_best_score'] = protected['window_score']
    for event in generation['exploit']:
        name = event['recipient']
        source = bundle_paths(run / name, epoch)
        snapshot = bundle_paths(run / 'checkpoints' / f"window-{decision['window_number']:03d}" / name, epoch)
        pre_copy = bundle_identity(source)
        archived = _copy_bundle(source, snapshot)
        event.update(window_number=decision['window_number'], pre_copy=pre_copy,
                     pre_copy_archive=archived, donor_checkpoint=identity)
    manifest['protected_best'] = protected


def apply_window_exploits(run, manifest, generation, manifest_path):
    """Reapply from an unchanged donor after interruption; persist each finished copy."""
    for event in generation['exploit']:
        if event['applied']:
            continue
        check_bundle(event['donor_checkpoint'])
        check_bundle(event['pre_copy_archive'])
        source = {k: Path(v['path']) for k, v in event['donor_checkpoint'].items()}
        dest = bundle_paths(Path(run) / event['recipient'], generation['epoch'])
        copied = _copy_bundle(source, dest)
        atomic_set_optimizer_lr(dest['optimizer'], event['new_lr'])
        after = bundle_identity(dest)
        check_bundle(event['donor_checkpoint'])
        event.update(copied_checkpoint=copied, post_copy=after, applied=True)
        member = manifest['members'][event['recipient']]
        member.update(lr=event['new_lr'], parent=event['donor'], parent_source=STRATEGY,
                      last_exploit_generation=generation['index'])
        generation[STRATEGY]['members'][event['recipient']]['copied'] = True
        manifest['updated_at'] = utc_now()
        atomic_json(manifest_path, manifest)
