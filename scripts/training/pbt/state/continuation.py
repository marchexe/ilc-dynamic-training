"""Continue a finalized windowed population in a new run, keeping its history.

The source and its ancestors remain read-only dependencies. Historical records
retain their original paths and terminal horizons; only live checkpoint bundles
are copied. A pinned manifest and state digest bind launch to reviewed evidence.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from training.pbt.config import contract_fingerprint
from training.pbt.planning.windowed_pbt_v2 import STRATEGY, bundle_identity, bundle_paths, check_bundle, _copy_bundle
from training.pbt.state.optimizer_state import atomic_copy, load_optimizer_state
from training.runtime import atomic_json, sha256


def source_manifest(config):
    spec = config['continuation']
    run = Path(spec['source_run']).resolve()
    path = run / 'manifest.json'
    if sha256(path) != spec['source_manifest_sha256']:
        raise ValueError('Continuation source manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'completed':
        raise ValueError('Continuation requires a completed source')
    return run, manifest


def training_contract(config):
    """Undo the recorded baseline measurement; permit only a longer horizon."""
    config = copy.deepcopy(config)
    config.pop('continuation', None)
    config['shared']['generations'] = 0
    pbt = config['pbt']
    if 'runtime_baseline_metric_value' in pbt:
        original = pbt.pop('configured_baseline_metric_value', None)
        if original is None:
            pbt.pop('baseline_metric_value', None)
        else:
            pbt['baseline_metric_value'] = original
        pbt.pop('runtime_baseline_metric_value')
    return contract_fingerprint(config)


def source_state(run, manifest):
    """Read and hash every live component, plus both retained best archives."""
    epoch = manifest['generations'][-1]['epoch']
    members = {name: bundle_identity(bundle_paths(Path(run) / name, epoch)) for name in manifest['members']}
    protected = manifest['protected_best']['checkpoint']
    check_bundle(protected)
    best = manifest['best']
    paths = {part: Path(best[part + '_path']) for part in ('state', 'optimizer')}
    paths['scaler'] = paths['optimizer'].with_name(paths['optimizer'].name.replace('_optimizer.pt', '_scaler.pt'))
    archives = dict(protected=protected, best=bundle_identity(paths))
    payload = dict(members=members, **archives)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload, digest


def plan_continuation(config):
    """Read-only launch plan. Does not copy files or run training/evaluation."""
    import torch
    run, source = source_manifest(config)
    old = source['config']
    end = old['shared']['generations']
    width = config['pbt']['windowed_pbt_v2']['window_epochs']
    destination = (Path(config['output_root']) / config['experiment_name']).resolve()
    if destination == run or run in destination.parents or destination in run.parents:
        raise ValueError('Continuation output must be separate from its source')
    if (old['pbt']['strategy'] != STRATEGY or training_contract(config) != training_contract(old)
            or config['shared']['generations'] <= end or end % width):
        raise ValueError('Continuation must preserve the training contract and extend complete windows')
    history = source['generations']
    history_sources(run, source)
    if (source['next_generation'] != end or len(history) != end
            or [g['index'] for g in history] != list(range(end))
            or any(g['status'] != 'completed' for g in history)
            or not history[-1][STRATEGY]['terminal'] or history[-1]['exploit']):
        raise ValueError('Source must have a finalized terminal boundary without pending actions')
    if set(source['members']) != {m['name'] for m in config['population']}:
        raise ValueError('Continuation population identities differ')
    state, digest = source_state(run, source)
    if digest != config['continuation']['source_state_sha256']:
        raise ValueError('Continuation source checkpoint state changed')
    final = source['final_evaluations']['control']
    for name, bundle in state['members'].items():
        worker = history[-1]['workers'][name]
        if (worker['status'] != 'completed' or worker['returncode'] != 0
                or source['members'][name]['lr'] != worker['lr']
                or final[name]['status'] != 'completed'
                or final[name]['checkpoint_sha256'] != bundle['state']['sha256']):
            raise ValueError(f'Unverified terminal member: {name}')
        optimizer = load_optimizer_state(bundle['optimizer']['path'])
        if not optimizer.get('state') or not optimizer.get('param_groups') or any(
                not math.isclose(group['lr'], worker['lr'], rel_tol=1e-12, abs_tol=0)
                for group in optimizer['param_groups']):
            raise ValueError(f'Optimizer state/LR does not match terminal member: {name}')
        model = torch.load(bundle['state']['path'], map_location='cpu', weights_only=False)
        scaler = torch.load(bundle['scaler']['path'], map_location='cpu', weights_only=False)
        if not model or not scaler.get('scale', 0) > 0 or '_growth_tracker' not in scaler:
            raise ValueError(f'Missing model or FP16 scaler state: {name}')
    if final['selected_best']['checkpoint_sha256'] != state['best']['state']['sha256']:
        raise ValueError('Selected-best checkpoint differs from final evaluation')
    return source, dict(source_run=str(run), source_manifest_sha256=config['continuation']['source_manifest_sha256'],
                        source_state_sha256=digest, start_generation=end,
                        live_members=copy.deepcopy(source['members']), **state, initialized=False)


def inherit_history(manifest, source, evidence):
    """Keep all algorithmic state verbatim; use the new run's reporting contract."""
    for key in ('members', 'generations', 'best', 'protected_best', 'initial_evaluation',
                'initial_resume', 'baseline_evaluation', 'datasets', 'metric_definition'):
        manifest[key] = copy.deepcopy(source[key])
    manifest['next_generation'] = evidence['start_generation']
    manifest['continuation'] = copy.deepcopy(evidence)
    return manifest


def bootstrap_continuation(run, manifest, manifest_path):
    """Restartable initial copy; never recopy after training has begun."""
    evidence = manifest['continuation']
    if evidence['initialized']:
        return
    if manifest['next_generation'] != evidence['start_generation']:
        raise ValueError('Cannot bootstrap after continuation training has started')
    source_manifest(manifest['config'])
    epoch = manifest['generations'][-1]['epoch']
    for name, bundle in evidence['members'].items():
        check_bundle(bundle)
        _copy_bundle({k: Path(v['path']) for k, v in bundle.items()}, bundle_paths(Path(run) / name, epoch))
    # Preserve the event history used by canonical plots/tables without appending
    # to the parent's log. This copy is safe to retry until initialized is saved.
    events = Path(evidence['source_run']) / 'events.jsonl'
    if events.is_file():
        atomic_copy(events, Path(run) / events.name)
    evidence['initialized'] = True
    atomic_json(manifest_path, manifest)


def history_sources(run, manifest):
    """Owner/config for each epoch: replay old terminal decisions as recorded.

    Recursive ownership also supports extending a continuation again. The source
    digest and exact history-prefix comparison reject changed parent evidence.
    """
    evidence = manifest.get('continuation')
    if not evidence:
        return [(Path(run).resolve(), manifest['config'])] * len(manifest['generations'])
    parent_run, parent = source_manifest(manifest['config'])
    start = parent['next_generation']
    if (evidence['source_run'] != str(parent_run) or evidence['start_generation'] != start
            or parent['generations'] != manifest['generations'][:start]
            or evidence['live_members'] != parent['members']
            or training_contract(parent['config']) != training_contract(manifest['config'])):
        raise ValueError('Continuation history or inherited population changed')
    owners = history_sources(parent_run, parent)
    if len(owners) != start:
        raise ValueError('Incomplete parent history')
    return owners + [(Path(run).resolve(), manifest['config'])] * (len(manifest['generations']) - start)
