#!/usr/bin/env python3
"""Evaluate further validation-selected checkpoints on the frozen Stage 2A holdout, paired with its results.

    .venv/bin/python scripts/experiments/holdout_addendum.py --prepared DIR {--freeze | --dry-run | (launch)}

``DIR/spec.json`` names, before any of these runs is evaluated, the training runs, the Stage 2A checkpoints
re-evaluated as references, and the B-minus-A pairs. ``--freeze`` (after the runs complete) records each run's
validation-selected checkpoint (``manifest['best']``, the Stage 2A selection rule) in ``DIR/protocol.json``;
the holdout never selects anything. A launch runs inference through the Stage 2A campaign (same evaluation
contract, leases, full-pass audit, row-ID coverage) in a detached tmux session on one node.

Differences from the frozen Stage 2A launcher, which stays unchanged:
* Parallelism is the number of idle GPUs found on the chosen node, at most the number of checkpoints, so no
  evaluation starts without a GPU to take (in the Stage 3 sealed test, ten evaluations shared fewer idle GPUs and
  five gave up after the 120 s resource wait).
* The tmux server starts without the user-manager bus, outside systemd-oomd management (as in Stage 3).
* Each reference must reproduce its Stage 2A holdout metric; the report states whether it did, bit for bit.
"""
import argparse
import json
from pathlib import Path
import shlex
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


def load_json(path):
    return json.loads(Path(path).read_text())


def output_dir(spec):
    return PROJECT_DIR / 'runs/eval' / spec['campaign']


def validate_spec(spec):
    ids = [r['id'] for r in spec['runs']] + list(spec['references'])
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


def reference_checkpoint(ident, stage2a):
    """A Stage 2A checkpoint and the holdout metric Stage 2A measured for it."""
    record = next(c for c in stage2a['checkpoints'] if c['id'] == ident)
    result = load_json(STAGE2A_RESULTS / ident / 'result.json')
    if result.get('status') != 'completed':
        raise ValueError(f'{ident}: Stage 2A evaluation is not completed')
    return dict(id=ident, selection='stage2a_reference', checkpoint=record['checkpoint'],
                stage2a_holdout_metric=result['metrics'][PRIMARY],
                stage2a_result=file_record(STAGE2A_RESULTS / ident / 'result.json'))


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
    checkpoints = ([selected_checkpoint(r['id'], r['run'], evaluation) for r in spec['runs']]
                   + [reference_checkpoint(ident, stage2a) for ident in spec['references']])
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
    lines += ['', 'Stage 2A references re-evaluated:', '']
    lines += [f'* {k}: Stage 2A {v["stage2a"]:.12f}, now {v["now"]:.12f}, '
              f'{"identical" if v["identical"] else "DIFFERENT"}' for k, v in references.items()]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    artifacts = [file_record(output / name) for name in ('analysis.json', 'report.md')]
    write_json(output / 'report_integrity.json', dict(protocol_sha256=protocol_sha, artifacts=artifacts))
    return dict(analysis=str(output / 'analysis.json'), references_reproduced=all(v['identical'] for v in references.values()))


def plan(protocol, nodes):
    """One node; one evaluation per idle GPU there, at most one per checkpoint."""
    cohort = choose_cohort(nodes, protocol['allocation'], min(len(protocol['checkpoints']), MAX_PARALLELISM))
    return dict(cohort, max_parallelism=len(cohort['gpus']))


def tmux_alive(host, session):
    result = subprocess.run(ssh_args(host, shlex.join([*TMUX, 'has-session', '-t', '=' + session])),
                            capture_output=True, text=True, timeout=15)
    if result.returncode not in (0, 1):
        raise RuntimeError(f'Cannot establish session status on {host}: {result.stderr.strip()}')
    return result.returncode == 0


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


def start(prepared, *, dry_run=False, discover_nodes=discover):
    protocol, digest = load_protocol(prepared)
    output = output_dir(protocol)
    output.mkdir(parents=True, exist_ok=True)
    with lock_file(output / '.launch.lock'):
        if not dry_run:
            if completed(output, digest):
                return dict(status='already_completed', output=str(output))
            if (output / 'launch.json').is_file():
                previous = load_json(output / 'launch.json')
                if tmux_alive(previous['host'], previous['session']):
                    return dict(status='already_running', **previous)
        cohort = plan(protocol, discover_nodes(protocol['allocation']))
        prepared_plan = dict(status='ready', checked_at=utc_now(), protocol_sha256=digest, cohort=cohort,
                             checkpoints=[c['id'] for c in protocol['checkpoints']], inference_launched=False)
        if dry_run:
            return prepared_plan
        session = f'holdout-{digest[:12]}'
        write_json(output / 'launch_cohort.json', cohort)
        cgroup = launch_tmux(cohort['host'], session, Path(prepared).resolve(), output, output / 'launch_cohort.json')
        launch = dict(host=cohort['host'], session=session, protocol_sha256=digest, started_at=utc_now(),
                      max_parallelism=cohort['max_parallelism'], cgroup=cgroup,
                      attach=f'ssh -t {cohort["host"]} {shlex.join(TMUX)} attach -t {session}')
        write_json(output / 'launch.json', launch)
        return dict(status='submitted_to_persistent_tmux', **launch)


def on_node(prepared, cohort_path):
    from experiments.stage2a.execution import run_campaign
    protocol, digest = load_protocol(prepared)
    cohort = load_json(cohort_path)
    if socket.gethostname().split('.')[0] != cohort['host']:
        raise ValueError('Supervisor started on a different node than planned')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    output = output_dir(protocol)

    def finalize(results):
        if load_protocol(prepared)[1] != digest:
            raise ValueError('Frozen protocol changed during inference; report blocked')
        return report(protocol, digest, results, output)
    campaign = run_campaign(protocol, digest, output, cohort, stop=stop, finalize=finalize)
    return 0 if campaign['status'] == 'completed' else 130 if campaign['status'] == 'interrupted' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--freeze', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--on-node', action='store_true')
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
        print(json.dumps(start(args.prepared, dry_run=args.dry_run), indent=2))
        return 0
    except BlockingIOError:
        print('error: another launch of this campaign holds the lock', file=sys.stderr)
        return 75
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f'{utc_now()} error: {error}', file=sys.stderr)
        return 69 if isinstance(error, RuntimeError) else 2


if __name__ == '__main__':
    raise SystemExit(main())
