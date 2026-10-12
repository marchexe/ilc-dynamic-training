"""Read-only verification of one bounded train8m smoke run.

Every check is evidence-based: commands, Weaver logs, data-audit records,
serialized checkpoints and the supervisor's process/GPU observations. Passing
requires the actual train8m files (fingerprint, rows, SHA256) in every
training audit; a train800k or synthetic substitution fails.
"""
import copy
import math
from pathlib import Path

from experiments.stage2b.common import PRIMARY, load_json
from training.checkpoints import bundle_paths, selected_best_bundle
from training.pbt.state.optimizer_state import load_optimizer_state
from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key
from validation.results import finite_metric_ok, require_checkpoint_result

WORKING_POINTS = [mistag_percent_key(pair, eff) for pair, eff in WORKING_POINT_DEFINITION]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _tensors(value):
    import torch
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _all_finite(state):
    import torch
    return all(torch.isfinite(t).all().item() for t in _tensors(state) if t.is_floating_point())


def _states_differ(left, right):
    import torch
    return left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left)


def _radam_steps(optimizer):
    return {key: int(state['step'].item() if hasattr(state['step'], 'item') else state['step'])
            for key, state in optimizer['state'].items() if 'step' in state}


def step_deltas(optimizer, previous):
    """Per-parameter RAdam step increments (parameters without gradients keep theirs)."""
    after, before = _radam_steps(optimizer), _radam_steps(previous)
    if not after or set(before) - set(after):
        return None
    return sorted({after[key] - before.get(key, 0) for key in after})


def _lr(optimizer):
    values = {float(g['lr'].item() if hasattr(g['lr'], 'item') else g['lr']) for g in optimizer['param_groups']}
    return values.pop() if len(values) == 1 else None


def verify_run(run, *, dataset_manifest, validation_audit, policy, observations):
    import torch
    run = Path(run)
    manifest = load_json(run / 'manifest.json')
    config, smoke, budgets = manifest['config'], policy['smoke'], policy['budgets']
    shared = config['shared']
    steps, batch = smoke['steps_per_generation'], smoke['batch_size']
    results = []

    def check(name, condition, detail=None):
        results.append(dict(check=name, passed=bool(condition), detail=detail))
        return bool(condition)

    train_dir = Path(dataset_manifest['files'][0]['resolved_path']).parent
    expected_train = sorted((f['resolved_path'], f['rows'], f['sha256']) for f in dataset_manifest['files'])
    check('run completed after resume', manifest.get('status') == 'completed'
          and manifest.get('next_generation') == smoke['generations'] == shared['generations'])
    check('configuration uses the prepared train8m input', Path(shared['dataset']) == train_dir
          and shared['train_suffix'] == 'train8m' and shared['samples_per_epoch'] == steps * batch
          and shared['batch_size'] == batch, dict(dataset=shared['dataset'], suffix=shared['train_suffix']))
    check('configuration selects on the isolated split', shared['validation_suffix'] == smoke['validation_suffix'])
    check('bounded population', len(manifest['members']) == smoke['members'])
    initial = manifest.get('initial_evaluation') or {}
    check('initial checkpoint evaluated', initial.get('status') == 'completed'
          and finite_metric_ok(initial.get('metrics'), PRIMARY))
    initial_epoch = int(shared['initial_epoch'])
    peak_cuda = 0.0
    for generation in manifest['generations']:
        epoch = generation['epoch']
        for name, worker in generation['workers'].items():
            label = f'generation {generation["index"]} {name}'
            metrics = worker.get('metrics') or {}
            command = worker.get('command') or []
            ok = check(f'{label}: worker completed', worker.get('status') == 'completed' and worker.get('returncode') == 0)
            if not ok:
                continue
            train_globs = command[command.index('--data-train') + 1:command.index('--data-val')]
            check(f'{label}: command reads only train8m', len(train_globs) == 3 and all(
                g.split(':', 1)[1].startswith(str(train_dir) + '/') and g.endswith('_train8m.parquet') for g in train_globs)
                and not any('train800k' in str(part) for part in command), train_globs)
            check(f'{label}: bounded and resumed from the previous epoch',
                  command[command.index('--samples-per-epoch') + 1] == str(steps * batch)
                  and command[command.index('--load-epoch') + 1] == str(epoch - 1)
                  and command[command.index('--model-prefix') + 1] == str(run / name / 'net'))
            check(f'{label}: finite primary and all eight working points',
                  all(finite_metric_ok(metrics, key) for key in [PRIMARY, *WORKING_POINTS]))
            check(f'{label}: finite losses and gradients', all(finite(metrics.get(k)) for k in
                  ('train_loss', 'validation_loss', 'train_max_grad_norm')))
            cuda = metrics.get('train_max_cuda_memory_mb')
            peak_cuda = max(peak_cuda, cuda or 0.0)
            check(f'{label}: bounded CUDA memory', finite(cuda) and cuda <= budgets['max_cuda_memory_mib'], cuda)
            audit = metrics.get('train_data_audit') or {}
            consumed = audit.get('consumed') or {}
            dataset = audit.get('dataset') or {}
            skipped = metrics.get('train_amp_skipped_optimizer_steps', 0)
            check(f'{label}: training audit is the real train8m dataset',
                  dataset.get('fingerprint') == dataset_manifest['weaver_audit']['fingerprint']
                  and dataset.get('total_rows') == dataset_manifest['rows_total']
                  and sorted((f['path'], f['rows'], f['sha256']) for f in dataset.get('files', [])) == expected_train,
                  dataset.get('fingerprint'))
            check(f'{label}: optimizer stepped on unique rows', audit.get('epoch') == epoch
                  and audit.get('batches') == steps and consumed.get('count') == steps * batch
                  and consumed.get('unique_ids') == consumed.get('count') and consumed.get('repeated_ids') == 0
                  and audit.get('optimizer_steps') == steps - skipped > 0 and finite(audit.get('loss'))
                  and skipped <= budgets['max_amp_skipped_fraction'] * steps,
                  dict(batches=audit.get('batches'), optimizer_steps=audit.get('optimizer_steps'), amp_skipped=skipped))
            validation = metrics.get('validation_data_audit') or {}
            check(f'{label}: complete validation pass', validation.get('exhausted') is True
                  and validation.get('optimizer_steps') == 0
                  and (validation.get('dataset') or {}).get('fingerprint') == validation_audit['fingerprint']
                  and (validation.get('consumed') or {}).get('count') == smoke['validation_rows'])
            log = Path(worker['log']).read_text(errors='replace') if Path(worker.get('log', '')).is_file() else ''
            scaler = run / name / f'net_epoch-{epoch - 1}_scaler.pt'
            amp_line = (f'Restored AMP scaler from {scaler}' if epoch - 1 > initial_epoch
                        else 'Initial checkpoint has no AMP scaler')
            check(f'{label}: new process restored model, Ranger state, LR and AMP scaler',
                  f'Resume training from epoch {epoch - 1}' in log and amp_line in log
                  and finite(metrics.get('train_loaded_optimizer_lr'))
                  and math.isclose(metrics['train_loaded_optimizer_lr'], worker['lr'], rel_tol=1e-5), amp_line)
            paths = bundle_paths(run / name, epoch)
            previous = bundle_paths(run / name, epoch - 1)
            if not check(f'{label}: model/optimizer/scaler serialized', all(p.is_file() for p in paths.values())):
                continue
            state = torch.load(paths['state'], map_location='cpu', weights_only=True)
            before = torch.load(previous['state'], map_location='cpu', weights_only=True)
            check(f'{label}: finite updated weights', _all_finite(state) and _states_differ(state, before))
            optimizer, prior = load_optimizer_state(paths['optimizer']), load_optimizer_state(previous['optimizer'])
            delta = step_deltas(optimizer, prior)
            check(f'{label}: Ranger state advanced by the applied steps',
                  delta is not None and max(delta) == audit.get('optimizer_steps') and set(delta) <= {0, max(delta)}
                  and _all_finite(optimizer['state']) and math.isclose(_lr(optimizer) or 0, worker['lr'], rel_tol=1e-6),
                  dict(delta=delta, optimizer_steps=audit.get('optimizer_steps')))
            scale = torch.load(paths['scaler'], map_location='cpu', weights_only=True).get('scale')
            scale = float(scale.item() if hasattr(scale, 'item') else scale) if scale is not None else None
            check(f'{label}: finite AMP scaler state', finite(scale) and scale > 0, scale)
    final_epoch = manifest['generations'][-1]['epoch'] if manifest['generations'] else None
    if final_epoch is not None:
        finals = [torch.load(bundle_paths(run / n, final_epoch)['state'], map_location='cpu', weights_only=True)
                  for n in manifest['members'] if bundle_paths(run / n, final_epoch)['state'].is_file()]
        check('population members remain isolated', len(finals) == len(manifest['members'])
              and all(_states_differ(a, b) for i, a in enumerate(finals) for b in finals[i + 1:]))
    try:
        selected_best_bundle(manifest['best'])
        check('selected-best bundle integrity (Stage 1)', True, manifest['best'].get('member'))
    except (OSError, ValueError, KeyError, TypeError) as error:
        check('selected-best bundle integrity (Stage 1)', False, str(error))
    control = (manifest.get('final_evaluations') or {}).get('control') or {}
    for name in [*manifest['members'], 'selected_best']:
        path = (Path(manifest['best']['state_path']) if name == 'selected_best'
                else bundle_paths(run / name, final_epoch)['state'])
        try:
            # Bound to the hash recorded when the runner evaluated it, never a fresh one.
            record = copy.deepcopy(control[name])
            expected = record['checkpoint_sha256']
            if name == 'selected_best' and expected != manifest['best']['checkpoint']['state']['sha256']:
                raise ValueError('selected-best evaluation names a different bundle')
            require_checkpoint_result(record, path, expected, name, PRIMARY)
            check(f'final evaluation {name}', True)
        except (KeyError, ValueError, RuntimeError, OSError) as error:
            check(f'final evaluation {name}', False, str(error))
    interruption = observations.get('interruption') or {}
    check('interrupted mid-training before serialization', interruption.get('triggered') is True
          and interruption.get('generation') == smoke['interrupt_generation']
          and interruption.get('runner_exit_code') == 130 and interruption.get('manifest_status') == 'interrupted'
          and interruption.get('terminated_workers') and not interruption.get('checkpoints_present_at_interrupt'),
          interruption)
    check('resume completed in a new runner process', observations.get('resume_exit_code') == 0)
    check('clean termination (no surviving smoke processes)', observations.get('leftover_processes') == [])
    check('no foreign GPU process interference', not observations.get('foreign_gpu_events'))
    check('bounded GPU memory (supervisor sampling)',
          (observations.get('peak_gpu_memory_mib') or 0) <= budgets['max_cuda_memory_mib'],
          observations.get('peak_gpu_memory_mib'))
    disk = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    check('bounded run disk use', disk <= budgets['max_run_disk_bytes'], disk)
    return dict(passed=all(r['passed'] for r in results), checks=results,
                measured=dict(peak_worker_cuda_memory_mb=peak_cuda, run_disk_bytes=disk))
