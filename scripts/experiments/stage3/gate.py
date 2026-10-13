"""Full-epoch gate: one real, uncapped train8m epoch must pass before any arm starts.

Checks reuse scripts/validation/verify_fixed_lr.verify (full traversal, raw
optimizer continuation, final-evaluation parity) and the Stage 2B checkpoint
helpers, then add the preregistered gate criteria and a compute/disk budget
projection from the measured epoch. A failing gate blocks the campaign.
"""
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from experiments.stage2b.verify import _all_finite, _lr, finite, step_deltas
from experiments.stage3.common import PRIMARY, load_json
from training.checkpoints import bundle_paths
from training.pbt.state.optimizer_state import load_optimizer_state
from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key

OPERATIONAL, CRITERIA = 'failed_operational', 'failed_criteria'


def seconds(start, end):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def disk_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def project(member_epoch_seconds, protocol, policy, *, concurrent):
    """Campaign GPU-hours/wall from the measured gate epoch (an estimate, labelled as such)."""
    factor = policy['budgets']['projection_safety_factor']
    epochs = protocol['design']['horizon_epochs']
    members = len(protocol['design']['population'])
    member_epochs = 2 * members * epochs
    gpu_hours = member_epochs * member_epoch_seconds * factor / 3600
    arm_wall = epochs * member_epoch_seconds * factor / 3600
    test_rows = protocol['design']['data']['sealed_test']['rows']
    # Stage 2A measured ~2,100-2,500 rows/s per inference GPU; 12 sealed + 2 holdout evaluations.
    evaluation_gpu_hours = (12 * test_rows + 2 * 2_825_125) / 2100 / 3600
    return dict(member_epoch_seconds=member_epoch_seconds, safety_factor=factor, member_epochs=member_epochs,
                training_gpu_hours=gpu_hours, evaluation_gpu_hours=evaluation_gpu_hours,
                total_gpu_hours=gpu_hours + evaluation_gpu_hours,
                wall_hours=arm_wall * (1 if concurrent else 2) + evaluation_gpu_hours / 5,
                schedule='concurrent arms' if concurrent else 'sequential arms', basis='estimate from the measured gate epoch')


def verify_gate(run, protocol, policy, observations):
    """Return (verdict, checks, measurements); verdict is passed/failed_criteria."""
    from validation import verify_fixed_lr as verifier
    run = Path(run)
    criteria = protocol['gate']
    checks = []

    def check(name, condition, detail=None):
        checks.append(dict(check=name, passed=bool(condition), detail=detail))

    evidence = verifier.verify(run)
    check('verify_fixed_lr: full traversal, raw continuation, final-evaluation parity', evidence['passed'],
          evidence['failures'][:20])
    manifest = load_json(run / 'manifest.json')
    generation = manifest['generations'][0] if manifest.get('generations') else {}
    worker = (generation.get('workers') or {}).get(criteria['member'], {})
    metrics = worker.get('metrics') or {}
    audit = metrics.get('train_data_audit') or {}
    traversal = audit.get('traversal') or []
    scanned = sum(t.get('scanned', {}).get('count', 0) for t in traversal)
    accepted = (audit.get('consumed') or {}).get('count', 0)
    command = worker.get('command') or []
    check('no sample caps in the training command', '--samples-per-epoch' not in command
          and '--steps-per-epoch' not in command and manifest['config']['shared'].get('samples_per_epoch') is None)
    check('full traversal of every train8m row', audit.get('exhausted') is True and scanned == criteria['scanned_rows'],
          dict(scanned=scanned, expected=criteria['scanned_rows']))
    check('accepted rows match the reweighting expectation',
          abs(accepted - criteria['expected_accepted_rows']) <= criteria['accepted_rows_tolerance'],
          dict(accepted=accepted, expected=round(criteria['expected_accepted_rows']), tolerance=criteria['accepted_rows_tolerance']))
    batches, steps = audit.get('batches', 0), audit.get('optimizer_steps', 0)
    skipped = metrics.get('train_amp_skipped_optimizer_steps', 0) or 0
    check('one optimizer step per full batch except AMP skips', batches == math.ceil(accepted / 1024) and steps == batches - skipped > 0,
          dict(batches=batches, optimizer_steps=steps, amp_skipped=skipped))
    check('AMP skipped-step fraction', batches and skipped / batches <= criteria['max_amp_skipped_fraction'], skipped)
    check('finite loss, gradient norm, primary metric and eight working points',
          all(finite(metrics.get(k)) for k in ('train_loss', 'validation_loss', 'train_max_grad_norm', PRIMARY,
                                               *[mistag_percent_key(p, e) for p, e in WORKING_POINT_DEFINITION])))
    validation = metrics.get('validation_data_audit') or {}
    check('complete validation pass', validation.get('exhausted') is True
          and (validation.get('consumed') or {}).get('count') == criteria['validation_rows'])
    paths = bundle_paths(run / criteria['member'], generation.get('epoch', 0))
    previous = bundle_paths(run / criteria['member'], generation.get('epoch', 0) - 1)
    if all(p.is_file() for p in paths.values()):
        state = torch.load(paths['state'], map_location='cpu', weights_only=True)
        optimizer, prior = load_optimizer_state(paths['optimizer']), load_optimizer_state(previous['optimizer'])
        delta = step_deltas(optimizer, prior)
        scaler = torch.load(paths['scaler'], map_location='cpu', weights_only=True)
        check('model/optimizer/scaler bundle finite and consistent', _all_finite(state) and _all_finite(optimizer['state'])
              and delta is not None and max(delta) == steps and set(delta) <= {0, steps}
              and math.isclose(_lr(optimizer) or 0, worker.get('lr', -1), rel_tol=1e-6)
              and finite(float(scaler.get('scale', float('nan')))) and float(scaler['scale']) > 0, dict(step_delta=delta))
    else:
        check('model/optimizer/scaler bundle finite and consistent', False, 'missing bundle component')
    reference = criteria['cuda_max_allocated_mb']
    cuda = metrics.get('train_max_cuda_memory_mb')
    check('CUDA memory consistent with Stage 2B', finite(cuda)
          and abs(cuda - reference['reference']) <= reference['relative_tolerance'] * reference['reference'], cuda)
    check('GPU memory within budget (nvidia-smi)', (observations.get('peak_gpu_memory_mib') or 0) <= policy['budgets']['max_cuda_memory_mib'],
          observations.get('peak_gpu_memory_mib'))
    check('host RAM within budget', 0 < (observations.get('peak_rss_bytes') or 0) <= criteria['max_tree_rss_bytes'],
          observations.get('peak_rss_bytes'))
    timing = observations.get('timing') or {}
    rate = timing.get('train_rate')
    check('training throughput plausible (Stage 2B/production range)', finite(rate)
          and criteria['training_rows_per_second'][0] <= rate <= criteria['training_rows_per_second'][1], rate)
    check('startup within limit', finite(timing.get('startup_seconds')) and timing['startup_seconds'] <= criteria['max_startup_seconds'],
          timing.get('startup_seconds'))
    disk = disk_bytes(run)
    check('run disk growth within limit', disk <= criteria['max_run_disk_bytes'], disk)
    check('no foreign GPU process and clean termination', not observations.get('foreign_gpu_events')
          and observations.get('leftover_processes') == [] and observations.get('exit_code') == 0)
    elapsed = seconds(worker['started_at'], worker['finished_at']) if worker.get('started_at') and worker.get('finished_at') else None
    measurements = dict(member_epoch_seconds=elapsed, accepted_rows=accepted, scanned_rows=scanned, batches=batches,
                        optimizer_steps=steps, amp_skipped=skipped, cuda_max_allocated_mb=cuda,
                        peak_gpu_memory_mib=observations.get('peak_gpu_memory_mib'),
                        peak_rss_bytes=observations.get('peak_rss_bytes'), run_disk_bytes=disk, timing=timing,
                        primary=metrics.get(PRIMARY), train_loss=metrics.get('train_loss'),
                        initial_primary=((manifest.get('initial_evaluation') or {}).get('metrics') or {}).get(PRIMARY),
                        reference_state_sha256=_sha(paths['state']) if paths['state'].is_file() else None)
    if elapsed:
        budgets = policy['budgets']
        for concurrent in (True, False):
            projection = project(elapsed, protocol, policy, concurrent=concurrent)
            measurements['projection_' + ('concurrent' if concurrent else 'sequential')] = projection
        projection = measurements['projection_sequential']
        check('projected campaign GPU-hours within budget', projection['total_gpu_hours'] <= budgets['max_campaign_gpu_hours'],
              round(projection['total_gpu_hours'], 1))
        check('projected campaign wall time within budget (sequential worst case)',
              projection['wall_hours'] <= budgets['max_campaign_wall_hours'], round(projection['wall_hours'], 1))
    else:
        check('measured member-epoch wall time', False, 'worker timestamps missing')
    verdict = 'passed' if all(c['passed'] for c in checks) else CRITERIA
    return verdict, checks, measurements


def _sha(path):
    from experiments.stage3.common import sha256
    return sha256(path)


def gate_record(output):
    """The latest gate verdict, or None."""
    try:
        return load_json(Path(output) / 'gate' / 'verdict.json')
    except (OSError, ValueError, json.JSONDecodeError):
        return None
