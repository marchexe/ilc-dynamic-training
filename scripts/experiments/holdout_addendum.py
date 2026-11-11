#!/usr/bin/env python3
"""Evaluate further validation-selected checkpoints on the frozen Stage 2A holdout, paired with its results.

    .venv/bin/python scripts/experiments/holdout_addendum.py --prepared DIR \
        {--freeze | --dry-run | (launch) | --retry | --status | --stop}

``DIR/spec.json`` names, before any of these runs is evaluated, the training runs, the checkpoints re-evaluated
as references, and the B-minus-A pairs. A reference is a Stage 2A checkpoint id, or
``{"id", "checkpoint_id", "protocol", "results"}`` for a checkpoint another campaign evaluated on the same
holdout under the same evaluation contract (a Stage 3 descriptive holdout or an earlier addendum). ``--freeze`` (after the runs complete) records each run's
validation-selected checkpoint (``manifest['best']``, the Stage 2A selection rule) in ``DIR/protocol.json``;
the holdout never selects anything. A launch runs inference through the Stage 2A campaign (same evaluation
contract, leases, full-pass audit, row-ID coverage) in a detached tmux session on one node.

Differences from the frozen Stage 2A launcher, which stays unchanged:
* Parallelism is the number of idle GPUs found on the chosen node, at most the number of checkpoints, so no
  evaluation starts without a GPU to take (in the Stage 3 sealed test, ten evaluations shared fewer idle GPUs and
  five gave up after the 120 s resource wait).
* The tmux server starts without the user-manager bus, outside systemd-oomd management (as in Stage 3).
* Each reference must reproduce its Stage 2A holdout metric; the report states whether it did, bit for bit.
* The supervisor ignores SIGINT: Ctrl-C typed into an attached tmux pane reaches it, and must not stop the
  campaign. SIGTERM and SIGHUP (``--stop``, which kills the tmux session) still stop it, and every signal it
  receives is logged and recorded in ``status.json``. The attach hint is read-only (``attach -r``).
* A launch after a failed or interrupted one needs ``--retry``. It first copies the earlier launch records
  (``status.json``, ``launch*.json``, each ``result.json``) to ``failed_launches/NNN``; the evaluation logs of
  every attempt stay in ``evaluations/<id>/attempt-NNN``. Completed evaluations are not repeated.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage2a.common import (PRIMARY, file_record, lock_file, sha256, utc_now,  # noqa: E402
                                        verify_file, write_json)
from experiments.stage2a.scheduling import choose_cohort, discover, ssh_args  # noqa: E402
from training.runtime import PROJECT_DIR, WORKING_POINT_DEFINITION, mistag_percent_key  # noqa: E402

STAGE2A = PROJECT_DIR / 'configs/prepared/stage2a_holdout_v1'
STAGE2A_RESULTS = PROJECT_DIR / 'runs/eval/stage2a_holdout_v1/evaluations'
TMUX = ['tmux', '-L', 'ilc-campaign']
OOMD_MANAGED_SCOPE = '/user@'
MAX_PARALLELISM = 9  # Stage 2A cohort limit
GPU_BUDGET = 5  # the supervisor's limit on GPUs in use at one time


def load_json(path):
    return json.loads(Path(path).read_text())


def output_dir(spec):
    return PROJECT_DIR / 'runs/eval' / spec['campaign']


def reference_id(reference):
    return reference if isinstance(reference, str) else reference['id']


def validate_spec(spec):
    ids = [r['id'] for r in spec['runs']] + [reference_id(r) for r in spec['references']]
    if spec.get('schema_version') != 1 or not spec['runs'] or len(set(ids)) != len(ids):
        raise ValueError('spec needs schema_version 1, at least one run and unique checkpoint ids')
    if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(ids) for pair in spec['pairs']):
        raise ValueError('every pair names two different checkpoints of this spec')
    return spec


def selected_checkpoint(ident, run, evaluation):
    """The run's validation global best, exactly as the Stage 2A inventory selects it."""
    path = PROJECT_DIR / 'runs/pbt' / run / 'manifest.json'
    manifest = load_json(path)
    config = manifest['config']
    shared, best = config['shared'], manifest['best']
    if manifest['status'] != 'completed' or config['pbt']['metric'] != PRIMARY or config['pbt']['mode'] != 'min':
        raise ValueError(f'{run}: not a completed run selected on the primary validation metric')
    for key in ('data_config', 'network_config'):
        if sha256(Path(shared[key])) != sha256(Path(evaluation[key])):
            raise ValueError(f'{run}: {key} differs from the holdout evaluation contract')
    return dict(id=ident, run=run, selection='validation_primary_global_best', member=best['member'],
                generation=best['generation'], weaver_epoch=best['epoch'], training_seed=shared['seed'],
                validation_metric=best['metric_value'], checkpoint=file_record(best['state_path']),
                provenance=[file_record(path)], run_fingerprint=manifest.get('fingerprint'))


def averaged_checkpoint(ident, averaged, evaluation):
    """A weight average chosen on the selection set by scripts/experiments/weight_averaging.py."""
    path = PROJECT_DIR / averaged['selection']
    record = next(r for r in load_json(path)['runs'] if r['id'] == averaged['id'])
    if record['selection'] == 'plain':
        raise ValueError(f'{ident}: the selection set chose the plain checkpoint; its holdout result already stands')
    plain = selected_checkpoint(ident, record['run'], evaluation)
    verify_file(record['selection_checkpoint'])
    return dict(plain, selection=f'selection_set_weight_average:{record["selection"]}',
                validation_metric=record['selection_metric'], checkpoint=record['selection_checkpoint'],
                plain_checkpoint=plain['checkpoint'], provenance=plain['provenance'] + [file_record(path)])


def reference_checkpoint(ident, stage2a):
    """A Stage 2A checkpoint and the holdout metric Stage 2A measured for it."""
    record = next(c for c in stage2a['checkpoints'] if c['id'] == ident)
    result = load_json(STAGE2A_RESULTS / ident / 'result.json')
    if result.get('status') != 'completed':
        raise ValueError(f'{ident}: Stage 2A evaluation is not completed')
    return dict(id=ident, selection='stage2a_reference', checkpoint=record['checkpoint'],
                stage2a_holdout_metric=result['metrics'][PRIMARY],
                stage2a_result=file_record(STAGE2A_RESULTS / ident / 'result.json'))


def external_reference(reference, stage2a):
    """A checkpoint another campaign evaluated on the Stage 2A holdout, with the metric it measured there."""
    if isinstance(reference, str):
        return reference_checkpoint(reference, stage2a)
    source = load_json(PROJECT_DIR / reference['protocol'])
    if source['evaluation'] != stage2a['evaluation'] or any(
            source['dataset'][k] != stage2a['dataset'][k] for k in ('rows', 'audit_fingerprint', 'holdout_files')):
        raise ValueError(f'{reference["id"]}: {reference["protocol"]} is not the Stage 2A holdout evaluation')
    record = next(c for c in source['checkpoints'] if c['id'] == reference['checkpoint_id'])
    path = PROJECT_DIR / reference['results'] / 'evaluations' / reference['checkpoint_id'] / 'result.json'
    result = load_json(path)
    if result.get('status') != 'completed' or result['checkpoint'] != record['checkpoint']['path']:
        raise ValueError(f'{reference["id"]}: no completed evaluation of this checkpoint in {reference["results"]}')
    return dict(id=reference['id'], selection='stage2a_reference', source=reference['checkpoint_id'],
                checkpoint=record['checkpoint'], stage2a_holdout_metric=result['metrics'][PRIMARY],
                source_protocol=file_record(PROJECT_DIR / reference['protocol']), stage2a_result=file_record(path))


def strict_load(checkpoints, evaluation):
    from experiments.stage2a.protocol import cpu_model
    model = cpu_model(evaluation['data_config'], evaluation['network_config'])
    for checkpoint in checkpoints:
        state = torch.load(checkpoint['checkpoint']['path'], map_location='cpu', weights_only=False)
        if any(not torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
            raise ValueError(f'{checkpoint["id"]}: non-finite model weights')
        model.load_state_dict(state, strict=True)


def freeze(prepared, *, check_models=True):
    prepared = Path(prepared)
    if (prepared / 'protocol.json').exists():
        raise FileExistsError('Protocol already frozen; it is never replaced')
    spec = validate_spec(load_json(prepared / 'spec.json'))
    stage2a = load_json(STAGE2A / 'protocol.json')
    if sha256(STAGE2A / 'protocol.json') != (STAGE2A / 'protocol.sha256').read_text().strip():
        raise ValueError('Stage 2A protocol hash changed')
    evaluation = stage2a['evaluation']
    checkpoints = ([averaged_checkpoint(r['id'], r['averaged'], evaluation) if 'averaged' in r
                    else selected_checkpoint(r['id'], r['run'], evaluation) for r in spec['runs']]
                   + [external_reference(reference, stage2a) for reference in spec['references']])
    if check_models:
        strict_load(checkpoints, evaluation)
    protocol = dict(schema_version=1, campaign=spec['campaign'], purpose=spec['purpose'], frozen_at=utc_now(),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_DIR, text=True).strip(),
                    spec=file_record(prepared / 'spec.json'), stage2a_protocol=file_record(STAGE2A / 'protocol.json'),
                    primary_metric=PRIMARY, evaluation=evaluation,
                    dataset={k: stage2a['dataset'][k] for k in ('rows', 'audit_fingerprint', 'holdout_files')},
                    allocation=dict(stage2a['allocation'], max_parallelism=MAX_PARALLELISM),
                    checkpoints=checkpoints, pairs=spec['pairs'], no_holdout_reselection=True,
                    analysis=stage2a['analysis'])
    write_json(prepared / 'protocol.json', protocol)
    (prepared / 'protocol.sha256').write_text(sha256(prepared / 'protocol.json') + '\n')
    return protocol


def load_protocol(prepared):
    from experiments.stage2a.dataset import verify_evaluation_files
    prepared = Path(prepared)
    digest = sha256(prepared / 'protocol.json')
    if digest != (prepared / 'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol hash changed')
    protocol = load_json(prepared / 'protocol.json')
    verify_file(protocol['spec'])
    verify_file(protocol['stage2a_protocol'])
    verify_evaluation_files(protocol)
    for checkpoint in protocol['checkpoints']:
        verify_file(checkpoint['checkpoint'])
    return protocol, digest


def report(protocol, protocol_sha, results, output):
    """Paired comparisons and reference reproduction; holdout metrics never feed back into any choice."""
    from experiments.stage3 import testset
    pairs = {f'{right}_minus_{left}': (left, right) for left, right in protocol['pairs']}
    metrics, comparisons = testset.paired(results, protocol, pairs, protocol['analysis'])
    references = {c['id']: dict(stage2a=c['stage2a_holdout_metric'], now=metrics[c['id']][PRIMARY],
                                identical=metrics[c['id']][PRIMARY] == c['stage2a_holdout_metric'])
                  for c in protocol['checkpoints'] if c['selection'] == 'stage2a_reference'}
    analysis = dict(campaign=protocol['campaign'], created_at=utc_now(), metrics=metrics, comparisons=comparisons,
                    references_reproduced=references, gpus={k: (r.get('gpu') or {}).get('name') for k, r in results.items()})
    output = Path(output)
    write_json(output / 'analysis.json', analysis)
    lines = [f'# {protocol["campaign"]}', '', protocol['purpose'], '',
             '| checkpoint | selection | holdout primary (%) |', '|---|---|---|']
    lines += [f'| {c["id"]} | {c["selection"]} | {metrics[c["id"]][PRIMARY]:.6f} |' for c in protocol['checkpoints']]
    lines += ['', '| comparison (B minus A) | delta (pp) | change (%) |', '|---|---|---|']
    lines += [f'| {name} | {v["delta_pp"]:+.6f} | {-v["reduction_percent"]:+.2f} |' for name, v in comparisons.items()]
    lines += ['', 'Background jets passing each working point for only one side (A only / B only):', '']
    for name, value in comparisons.items():
        points = ', '.join(f'{p}{round(100 * e)} {w["left_only"]}/{w["right_only"]}' for (p, e), w in
                           ((pe, value['working_points'][mistag_percent_key(*pe)]) for pe in WORKING_POINT_DEFINITION))
        lines.append(f'* {name}: {points}')
    lines += ['', 'No confidence interval: jets of one physical event cannot be linked (as in Stage 2A).']
    lines += ['', 'References re-evaluated (their earlier metric on this holdout, and now):', '']
    lines += [f'* {k}: earlier {v["stage2a"]:.12f}, now {v["now"]:.12f}, '
              f'{"identical" if v["identical"] else "DIFFERENT"}' for k, v in references.items()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    artifacts = [file_record(output / name) for name in ('analysis.json', 'report.md')]
    write_json(output / 'report_integrity.json', dict(protocol_sha256=protocol_sha, artifacts=artifacts))
    return dict(analysis=str(output / 'analysis.json'), references_reproduced=all(v['identical'] for v in references.values()))


def plan(protocol, nodes):
    """One node; one evaluation per idle GPU there, at most one per checkpoint."""
    cohort = choose_cohort(nodes, protocol['allocation'], min(len(protocol['checkpoints']), MAX_PARALLELISM, GPU_BUDGET))
    return dict(cohort, max_parallelism=len(cohort['gpus']))


def tmux_alive(host, session):
    result = subprocess.run(ssh_args(host, shlex.join([*TMUX, 'has-session', '-t', '=' + session])),
                            capture_output=True, text=True, timeout=15)
    if result.returncode not in (0, 1):
        raise RuntimeError(f'Cannot establish session status on {host}: {result.stderr.strip()}')
    return result.returncode == 0


def previous_launch(output, protocol_sha):
    """'completed', 'running', 'failed' (a launch whose supervisor is gone without a report) or None."""
    if completed(output, protocol_sha):
        return 'completed'
    if not (output / 'launch.json').is_file():
        return None
    previous = load_json(output / 'launch.json')
    return 'running' if tmux_alive(previous['host'], previous['session']) else 'failed'


def archive_launch(output):
    """Copy the records a relaunch overwrites; attempt directories are never overwritten."""
    root = output / 'failed_launches'
    target = root / f'{1 + len(list(root.glob("[0-9][0-9][0-9]"))):03d}'
    target.mkdir(parents=True)
    for name in ('status.json', 'launch.json', 'launch_cohort.json'):
        if (output / name).is_file():
            shutil.copy2(output / name, target / name)
    for result in sorted((output / 'evaluations').glob('*/result.json')):
        shutil.copy2(result, target / f'{result.parent.name}.result.json')
    return target


def launch_tmux(host, session, prepared, output, cohort_path):
    script = shlex.join([str(PROJECT_DIR / '.venv/bin/python'), str(Path(__file__).resolve()), '--on-node',
                         '--prepared', str(prepared), '--cohort', str(cohort_path)])
    inner = ('cd ' + shlex.quote(str(PROJECT_DIR)) + ' && exec ' + script + ' >> '
             + shlex.quote(str(output / 'supervisor.log')) + ' 2>&1')
    detached = ['env', '-u', 'DBUS_SESSION_BUS_ADDRESS', '-u', 'XDG_RUNTIME_DIR', *TMUX]
    subprocess.run(ssh_args(host, shlex.join([*detached, 'new-session', '-d', '-s', session, 'bash', '-lc', inner])),
                   check=True, capture_output=True, text=True, timeout=20)
    pane = shlex.join([*TMUX, 'list-panes', '-t', '=' + session, '-F', '#{pane_pid}'])
    cgroup = subprocess.run(ssh_args(host, f'cat /proc/$({pane} | head -n 1)/cgroup'), check=True,
                            capture_output=True, text=True, timeout=20).stdout.strip()
    if OOMD_MANAGED_SCOPE in cgroup:
        subprocess.run(ssh_args(host, shlex.join([*TMUX, 'kill-session', '-t', '=' + session])),
                       capture_output=True, text=True, timeout=20)
        raise RuntimeError(f'Supervisor landed in an oomd-managed scope ({cgroup}); session stopped')
    return cgroup


def completed(output, protocol_sha):
    try:
        status = load_json(Path(output) / 'status.json')
        integrity = load_json(Path(output) / 'report_integrity.json')
        for artifact in integrity['artifacts']:
            verify_file(artifact)
        return status['status'] == 'completed' and status['protocol_sha256'] == protocol_sha
    except (OSError, KeyError, ValueError, TypeError):
        return False


def start(prepared, *, dry_run=False, retry=False, discover_nodes=discover):
    protocol, digest = load_protocol(prepared)
    output = output_dir(protocol)
    output.mkdir(parents=True, exist_ok=True)
    with lock_file(output / '.launch.lock'):
        archived = None
        if not dry_run:
            state = previous_launch(output, digest)
            if state == 'completed':
                return dict(status='already_completed', output=str(output))
            if state == 'running':
                return dict(status='already_running', **load_json(output / 'launch.json'))
            if state == 'failed' and not retry:
                raise RuntimeError(f'The previous launch ended without a report (see {output / "status.json"}); '
                                   'relaunch with --retry')
            if state is None and retry:
                raise RuntimeError('Nothing to retry: this campaign has not been launched')
        cohort = plan(protocol, discover_nodes(protocol['allocation']))
        prepared_plan = dict(status='ready', checked_at=utc_now(), protocol_sha256=digest, cohort=cohort,
                             checkpoints=[c['id'] for c in protocol['checkpoints']], inference_launched=False)
        if dry_run:
            return prepared_plan
        if retry:
            archived = archive_launch(output)
        session = f'holdout-{digest[:12]}'
        write_json(output / 'launch_cohort.json', cohort)
        cgroup = launch_tmux(cohort['host'], session, Path(prepared).resolve(), output, output / 'launch_cohort.json')
        launch = dict(host=cohort['host'], session=session, protocol_sha256=digest, started_at=utc_now(),
                      max_parallelism=cohort['max_parallelism'], cgroup=cgroup,
                      attach=f'ssh -t {cohort["host"]} {shlex.join(TMUX)} attach -r -t {session}')
        if archived:
            launch['previous_launch_records'] = str(archived)
        write_json(output / 'launch.json', launch)
        return dict(status='submitted_to_persistent_tmux', **launch)


def last_batch(console):
    try:
        counts = re.findall(rb'(\d+)it \[', console.read_bytes()[-20000:])
    except OSError:
        return None
    return int(counts[-1]) if counts else None


def progress(prepared):
    """Where each evaluation stands, without attaching to the tmux session."""
    protocol, digest = load_protocol(prepared)
    output = output_dir(protocol)
    checkpoints = {}
    for checkpoint in protocol['checkpoints']:
        directory = output / 'evaluations' / checkpoint['id']
        record = load_json(directory / 'result.json') if (directory / 'result.json').is_file() else {}
        attempts = sorted(directory.glob('attempt-*'))
        checkpoints[checkpoint['id']] = dict(
            status=record.get('status'), attempt=record.get('attempt'), started_at=record.get('started_at'),
            batches=last_batch(attempts[-1] / 'console.log') if attempts else None, error=record.get('error'))
    status = load_json(output / 'status.json') if (output / 'status.json').is_file() else {}
    return dict(launch=previous_launch(output, digest), campaign=status.get('status'), signals=status.get('signals'),
                expected_batches=math.ceil(protocol['dataset']['rows'] / protocol['evaluation']['batch_size']),
                timeout_seconds=protocol['allocation']['evaluation_timeout_seconds'], checkpoints=checkpoints)


def stop_launch(prepared):
    """Kill the tmux session: the supervisor gets SIGHUP, stops its inference processes and records why."""
    protocol, digest = load_protocol(prepared)
    output = output_dir(protocol)
    if previous_launch(output, digest) != 'running':
        raise RuntimeError('No running launch of this campaign')
    launch = load_json(output / 'launch.json')
    subprocess.run(ssh_args(launch['host'], shlex.join([*TMUX, 'kill-session', '-t', '=' + launch['session']])),
                   check=True, capture_output=True, text=True, timeout=20)
    return dict(status='stop_requested', host=launch['host'], session=launch['session'])


def supervise_signals(stop, received):
    """SIGTERM/SIGHUP stop the campaign; SIGINT (Ctrl-C in an attached pane) is logged and ignored."""
    def handle(number, _frame):
        name = signal.Signals(number).name
        received.append(dict(signal=name, at=utc_now(), stopped=number != signal.SIGINT))
        if number == signal.SIGINT:
            print(f'{utc_now()} {name} ignored; stop with holdout_addendum.py --stop', file=sys.stderr, flush=True)
            return
        print(f'{utc_now()} {name} received; stopping the campaign', file=sys.stderr, flush=True)
        stop.set()
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, handle)


def on_node(prepared, cohort_path):
    from experiments.stage2a.execution import run_campaign
    protocol, digest = load_protocol(prepared)
    cohort = load_json(cohort_path)
    if socket.gethostname().split('.')[0] != cohort['host']:
        raise ValueError('Supervisor started on a different node than planned')
    stop, received = threading.Event(), []
    supervise_signals(stop, received)
    output = output_dir(protocol)

    def finalize(results):
        if load_protocol(prepared)[1] != digest:
            raise ValueError('Frozen protocol changed during inference; report blocked')
        return report(protocol, digest, results, output)
    campaign = run_campaign(protocol, digest, output, cohort, stop=stop, finalize=finalize)
    if received:
        status = load_json(output / 'status.json')
        write_json(output / 'status.json', dict(status, signals=received))
    return 0 if campaign['status'] == 'completed' else 130 if campaign['status'] == 'interrupted' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--freeze', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--on-node', action='store_true')
    modes.add_argument('--retry', action='store_true', help='relaunch after a failed or interrupted launch')
    modes.add_argument('--status', action='store_true', help='progress of each evaluation')
    modes.add_argument('--stop', action='store_true', help='stop a running launch')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cohort', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.freeze:
            protocol = freeze(args.prepared)
            print(json.dumps(dict(protocol=str(args.prepared / 'protocol.json'),
                                  checkpoints=[c['id'] for c in protocol['checkpoints']]), indent=2))
            return 0
        if args.on_node:
            return on_node(args.prepared, args.cohort)
        if args.status or args.stop:
            print(json.dumps(progress(args.prepared) if args.status else stop_launch(args.prepared), indent=2))
            return 0
        print(json.dumps(start(args.prepared, dry_run=args.dry_run, retry=args.retry), indent=2))
        return 0
    except BlockingIOError:
        print('error: another launch of this campaign holds the lock', file=sys.stderr)
        return 75
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f'{utc_now()} error: {error}', file=sys.stderr)
        return 69 if isinstance(error, RuntimeError) else 2


if __name__ == '__main__':
    raise SystemExit(main())
