"""Full-epoch gate: one real, uncapped train8m epoch must pass before any arm starts.

Checks reuse scripts/validation/verify_fixed_lr.verify (full traversal, raw
optimizer continuation, final-evaluation parity) and the Stage 2B checkpoint
helpers, then add the preregistered gate criteria and a compute/disk budget
projection from the measured epoch. A failing gate blocks the campaign.
"""
import json
import math
import re
from datetime import datetime
from pathlib import Path

import torch

from experiments.stage2b.verify import _all_finite, _lr, finite, step_deltas
from experiments.stage3.common import PRIMARY, load_json
from training.checkpoints import bundle_paths
from training.pbt.state.optimizer_state import load_optimizer_state
from training.runtime import WORKING_POINT_DEFINITION, mistag_percent_key

OPERATIONAL, CRITERIA = 'failed_operational', 'failed_criteria'
GRAD_NORM_LINE = re.compile(r'Max Grad Norm: (\S+)')
AMP_SKIPPED_LINE = re.compile(r'AMP skipped optimizer steps: (\S+)')
DECIMAL = re.compile(r'\d+(\.\d*)?([eE][+-]?\d+)?')


def seconds(start, end):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def disk_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def nonfinite_metrics(metrics, keys):
    """Missing or non-finite metrics with their observed values (as text, so the verdict stays valid JSON)."""
    return {key: 'missing' if key not in metrics else repr(metrics[key]) for key in keys if not finite(metrics.get(key))}


def gradient_norm_check(worker, metrics, audit, *, accounting, amp_fraction):
    """Return (passed, detail) for the worker's logged epoch maximum gradient norm.

    Weaver's 'Max Grad Norm' also covers AMP-skipped steps, whose unscaled gradients are non-finite by
    construction, and the shared log parser drops the literal 'inf'. The value is therefore read from the
    worker log. A finite norm passes; '+inf' passes only when the log, the manifest and the data-audit step
    accounting agree on at least one AMP-skipped step and the step-accounting and AMP-fraction checks pass.
    'nan', '-inf', malformed, missing or duplicated values and inconsistent counters fail.
    """
    recorded = metrics.get('train_max_grad_norm')
    detail = dict(logged=None, amp_skipped_log=None, amp_skipped_manifest=metrics.get('train_amp_skipped_optimizer_steps'),
                  amp_skipped_audit=None, manifest_value=None if recorded is None else repr(recorded))

    def fail(reason):
        return False, dict(detail, reason=reason)

    log = Path(worker['log']) if worker.get('log') else None
    if log is None or not log.is_file():
        return fail('worker log missing')
    text = log.read_text(errors='replace')
    norms, skips = GRAD_NORM_LINE.findall(text), AMP_SKIPPED_LINE.findall(text)
    if len(norms) != 1 or len(skips) != 1:
        return fail(f'expected one gradient-norm and one AMP-skip log line, found {len(norms)} and {len(skips)}')
    detail.update(logged=norms[0], amp_skipped_log=skips[0])
    manifest_skipped = metrics.get('train_amp_skipped_optimizer_steps')
    if not skips[0].isdigit() or type(manifest_skipped) is not int or int(skips[0]) != manifest_skipped:
        return fail('AMP skip counter missing, malformed or inconsistent between log and manifest')
    if type(audit.get('batches')) is not int or type(audit.get('optimizer_steps')) is not int:
        return fail('data-audit step counters missing')
    detail['amp_skipped_audit'] = audit['batches'] - audit['optimizer_steps']
    if detail['amp_skipped_audit'] != manifest_skipped:
        return fail('AMP skip counter inconsistent with the data-audit step accounting')
    if DECIMAL.fullmatch(norms[0]):
        value = float(norms[0])
        if not math.isfinite(value):
            return fail('gradient norm overflows')
        if not finite(recorded) or recorded != value:
            return fail('manifest gradient norm disagrees with the log')
        return True, dict(detail, reason='finite')
    if norms[0] != 'inf':
        return fail('gradient norm is NaN, negative infinity or malformed')
    if recorded is not None:
        return fail('manifest records a gradient norm although the log reports inf')
    if manifest_skipped < 1:
        return fail('infinite gradient norm without AMP-skipped steps')
    if not (accounting and amp_fraction):
        return fail('infinite gradient norm with failing step accounting or AMP fraction')
    return True, dict(detail, reason='inf only from AMP-skipped steps')


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
    accounting = batches == math.ceil(accepted / 1024) and steps == batches - skipped > 0
    amp_fraction = bool(batches) and skipped / batches <= criteria['max_amp_skipped_fraction']
    check('one optimizer step per full batch except AMP skips', accounting,
          dict(batches=batches, optimizer_steps=steps, amp_skipped=skipped))
    check('AMP skipped-step fraction', amp_fraction, skipped)
    invalid = nonfinite_metrics(metrics, ('train_loss', 'validation_loss', PRIMARY,
                                          *[mistag_percent_key(p, e) for p, e in WORKING_POINT_DEFINITION]))
    check('finite loss, primary metric and eight working points', not invalid, invalid or None)
    check('maximum gradient norm finite, or +inf only with confirmed AMP-skipped steps',
          *gradient_norm_check(worker, metrics, audit, accounting=accounting, amp_fraction=amp_fraction))
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
