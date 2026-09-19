"""Compatibility entry point and unchanged transitions for windowed PBT v2."""
from pathlib import Path

from training.pbt.reference.windowed_v2 import (
    STRATEGY, WORKING_POINTS, exceeds_margin, _mutations, windowed_pbt_v2_plan,
)
from training.checkpoints import bundle_paths, bundle_identity, check_bundle, copy_bundle as _copy_bundle
from training.pbt.state.optimizer_state import atomic_set_optimizer_lr
from training.runtime import atomic_json, utc_now


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
