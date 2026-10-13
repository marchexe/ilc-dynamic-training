#!/usr/bin/env python3
"""Stage 2B: prepare/verify train8m and run the bounded GPU readiness smoke.

Default: read-only preflight on the submitting host, then submit the whole
campaign (dataset phase, then GPU smoke, verification and report) to detached
tmux on one authorized GPU node. Exit codes: 0 submitted/completed, 1 smoke or
verification failure, 2 invalid configuration/inputs, 69 unavailable resources
(GPU or disk), 75 another launch/campaign holds the lock, 130 interrupted.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.stage2a_campaign import tmux_alive  # noqa: E402
from experiments.stage2a.scheduling import discover, free_gpu, ssh_args  # noqa: E402
from experiments.stage2b.common import (EXIT_FAILED, EXIT_INTERRUPTED, EXIT_INVALID, EXIT_LOCKED, EXIT_OK,  # noqa: E402
                                        EXIT_RESOURCES, OUTPUT, PREPARED, PROJECT_DIR, SPEC, POLICY, disk_free,
                                        load_json, load_policy, lock_file, utc_now, verify_file, write_json)
from experiments.stage2b import dataset as data  # noqa: E402

TRACKED = ('scripts/experiments/stage2b_campaign.py', 'scripts/experiments/run_stage2b_smoke.sh',
           'scripts/experiments/stage2b', 'configs/prepared/stage2b_train8m_v1/dataset_spec.json',
           'configs/prepared/stage2b_train8m_v1/smoke_policy.json', 'configs/experiments/stage2b_train8m_smoke.yaml',
           'configs/drafts/stage3_train8m_v1')


def git(*args):
    return subprocess.run(['git', *args], cwd=PROJECT_DIR, capture_output=True, text=True)


def repository_revision():
    """Tracked tree must be clean and contain the Stage 2B workflow; untracked outputs are allowed."""
    head = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if head.returncode or dirty.returncode:
        raise ValueError('Cannot establish the repository revision')
    if dirty.stdout.strip():
        raise ValueError('Tracked files have uncommitted changes; commit them before launching:\n' + dirty.stdout)
    for path in TRACKED:
        if git('ls-files', '--error-unmatch', path).returncode:
            raise ValueError(f'Stage 2B file is not committed: {path}')
    return dict(commit=head.stdout.strip(), branch=git('branch', '--show-current').stdout.strip(), dirty=False)


def choose_host(nodes, allocation):
    """Authorized node with enough idle GPUs of one model; preferred node, then A100, then memory."""
    groups = {}
    for node in nodes:
        if node['host'] not in allocation['authorized_nodes']:
            continue
        for gpu in node['gpus']:
            if free_gpu(gpu, allocation):
                groups.setdefault((node['host'], gpu['name'], gpu['total_mib']), []).append(gpu)
    groups = {k: v for k, v in groups.items() if len(v) >= allocation['gpus_required']}
    if not groups:
        raise RuntimeError(f'No authorized node has {allocation["gpus_required"]} idle GPUs with '
                           f'>= {allocation["min_gpu_memory_mib"]} MiB; nothing submitted')
    (host, name, memory), gpus = max(groups.items(), key=lambda item: (
        item[0][0] == allocation['preferred_node'], 'A100' in item[0][1], item[0][2], len(item[1]), item[0][0]))
    return dict(host=host, name=name, total_mib=memory, max_parallelism=allocation['gpus_required'],
                gpus=sorted(gpus, key=lambda g: g['index'])[:allocation['gpus_required']])


def completed(output):
    """True only when the published report verifies and declares readiness."""
    try:
        status = load_json(Path(output) / 'status.json')
        integrity = load_json(Path(output) / 'report_integrity.json')
        for artifact in integrity['artifacts']:
            verify_file(artifact)
        return status.get('status') == 'completed' and load_json(Path(output) / 'readiness.json')['ready'] is True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def preflight(output, spec, policy):
    """Read-only checks on the submitting host (sources, disk, revision)."""
    revision = repository_revision()
    data.check_validation_suffix(spec, policy['smoke']['validation_suffix'])
    pre = data.preflight(spec)
    free = disk_free(output)
    if free < policy['budgets']['min_free_bytes_after_dataset']:
        raise OSError(28, f'Less than {policy["budgets"]["min_free_bytes_after_dataset"]} bytes free for smoke outputs')
    return dict(git=revision, dataset=pre['status'], free_bytes=pre['free_bytes'],
                required_free_bytes=pre['required_free_bytes'], sources_verified=len(pre['sources']))


def cluster_snapshot():
    # check.sh is an independent operational record, not allocation authority.
    from experiments.site_config import cluster
    env = dict(os.environ, ML_PFA_CLUSTER_NODES=' '.join(cluster()['authorized_nodes']))
    result = subprocess.run(['bash', str(PROJECT_DIR / 'scripts/cluster/check.sh')], capture_output=True, text=True,
                            timeout=120, env=env)
    return result.stdout + result.stderr


def launch_tmux(host, session, output, cohort_path, *, mode):
    probe = 'command -v tmux >/dev/null && test -x ' + shlex.quote(str(PROJECT_DIR / '.venv/bin/python'))
    subprocess.run(ssh_args(host, probe), check=True, capture_output=True, text=True, timeout=20)
    script = shlex.join([str(PROJECT_DIR / '.venv/bin/python'), str(Path(__file__).resolve()), '--on-node',
                         '--output', str(output), '--cohort', str(cohort_path), *(['--prepare-only'] if mode == 'prepare' else [])])
    inner = 'cd ' + shlex.quote(str(PROJECT_DIR)) + ' && exec ' + script + ' >> ' + shlex.quote(str(output / 'supervisor.log')) + ' 2>&1'
    subprocess.run(ssh_args(host, shlex.join(['tmux', 'new-session', '-d', '-s', session, 'bash', '-lc', inner])),
                   check=True, capture_output=True, text=True, timeout=20)


def start(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec, spec_sha = data.load_spec(args.spec)
    policy, policy_sha = load_policy(args.policy)
    with lock_file(output / '.launch.lock'):
        if not args.dry_run and completed(output):
            print(json.dumps(dict(status='already_completed_and_verified', report=str(output / 'report.md')), indent=2))
            return EXIT_OK
        launch = output / 'launch.json'
        if not args.dry_run and launch.exists():
            previous = load_json(launch)
            if tmux_alive(previous['host'], previous['session']):
                print(json.dumps(dict(status='already_running', **previous), indent=2))
                return EXIT_OK
        checks = preflight(output, spec, policy)
        allocation = policy['allocation']
        nodes = discover(allocation)
        if args.prepare_only:
            reachable = [n['host'] for n in nodes if not n['error']]
            if not reachable:
                raise RuntimeError('No authorized node is reachable for dataset preparation')
            host = allocation['preferred_node'] if allocation['preferred_node'] in reachable else reachable[0]
            cohort = dict(host=host, name='prepare_only', total_mib=0, max_parallelism=0, gpus=[])
        else:
            try:
                cohort = choose_host(nodes, allocation)
            except RuntimeError as error:
                write_json(output / 'resource_failure.json', dict(checked_at=utc_now(), error=str(error), nodes=nodes))
                raise
        plan = dict(status='ready', checked_at=utc_now(), spec_sha256=spec_sha, policy_sha256=policy_sha,
                    preflight=checks, cohort=cohort, gpu_observation=nodes, gpus_reserved=False, training_launched=False,
                    conversion_launched=False, mode='prepare' if args.prepare_only else 'full')
        if args.dry_run:
            write_json(output / 'dry_run.json', plan)
            print(json.dumps({k: v for k, v in plan.items() if k != 'gpu_observation'}, indent=2))
            return EXIT_OK
        (output / 'cluster_snapshot.txt').write_text(cluster_snapshot())
        write_json(output / 'runtime_preflight.json', plan)
        cohort_path = output / 'launch_cohort.json'
        write_json(cohort_path, dict(cohort, git=checks['git'], spec_sha256=spec_sha, policy_sha256=policy_sha))
        session = 'stage2b-train8m-' + spec_sha[:8]
        try:
            launch_tmux(cohort['host'], session, output, cohort_path, mode=plan['mode'])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            detail = getattr(error, 'stderr', '') or str(error)
            write_json(output / 'last_failure.json', dict(time=utc_now(), error=f'Remote submission failed: {detail}'))
            raise RuntimeError(f'Remote tmux submission to {cohort["host"]} failed: {detail}') from error
        record = dict(host=cohort['host'], session=session, mode=plan['mode'], started_at=utc_now(),
                      git=checks['git'], spec_sha256=spec_sha, policy_sha256=policy_sha, output=str(output))
        write_json(launch, record)
        print(json.dumps(dict(status='submitted_to_persistent_tmux', **record,
                              watch=f'bash scripts/experiments/run_stage2b_smoke.sh --status',
                              attach=f'ssh -t {cohort["host"]} tmux attach -t {session}'), indent=2))
        return EXIT_OK


def on_node(args):
    from experiments.stage2a.common import digest_json
    from experiments.stage2b.smoke import SmokeFailure, Status, draft_dry_runs, run_dataset_phase, run_smoke
    from experiments.stage2b import parity, report
    cohort = load_json(args.cohort)
    if socket.gethostname().split('.')[0] != cohort['host']:
        raise ValueError('Supervisor started on a different node than the launch cohort')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    output = args.output.resolve()
    with lock_file(output / '.campaign.lock') as campaign_lock:
        spec, spec_sha = data.load_spec(args.spec)
        policy, policy_sha = load_policy(args.policy)
        revision = repository_revision()
        if (revision['commit'] != cohort['git']['commit'] or spec_sha != cohort['spec_sha256']
                or policy_sha != cohort['policy_sha256']):
            raise ValueError('Repository revision or frozen inputs changed after submission')
        status = Status(output, dict(git=revision, spec_sha256=spec_sha, policy_sha256=policy_sha, host=cohort['host'],
                                     mode='prepare' if args.prepare_only else 'full', started_at=utc_now()))
        try:
            manifest = run_dataset_phase(spec, policy, status, stop)
            if args.prepare_only:
                status.finish('dataset_ready')
                return EXIT_OK
            from weaver.utils.data_audit import parquet_manifest
            validation_dir = (PROJECT_DIR / spec['validation']['dataset']).resolve()
            suffix = policy['smoke']['validation_suffix']
            validation_audit = parquet_manifest(sorted(str(p) for f in data.FLAVORS
                                                       for p in validation_dir.glob(f'*_{f}_{suffix}.parquet')))
            smoke = run_smoke(spec, policy, manifest, cohort, status, stop, validation_audit=validation_audit,
                              inherit_fds=(campaign_lock.fileno(),))
            status.phase('stage3_dry_run', 'GPU')
            dry_runs = draft_dry_runs(policy)
            checked = parity.check(spec)
            write_json(output / 'parity.json', checked)
            status.phase('report', 'GPU')
            provenance = load_json(PREPARED / 'provenance.json') if (PREPARED / 'provenance.json').is_file() else None
            summary = report.build(output, status.value, smoke=smoke, parity=checked, dry_runs=dry_runs,
                                   provenance=dict(sha256=digest_json(provenance)) if provenance else None)
            status.finish('completed' if summary['ready'] else 'failed', ready=summary['ready'],
                          report=str(output / 'report.md'))
            status.log('GPU', f'readiness={summary["ready"]}; report {output / "report.md"}')
            return EXIT_OK if summary['ready'] else EXIT_FAILED
        except InterruptedError as error:
            status.finish('interrupted', error=str(error))
            return EXIT_INTERRUPTED
        except BaseException as error:
            kind = type(error).__name__
            status.finish('failed', error=f'{kind}: {error}')
            status.log('FAIL', f'{kind}: {error}')
            try:
                # A failed campaign still publishes a NOT READY report with the cause.
                report.build(output, status.value)
            except Exception as report_error:
                status.log('FAIL', f'failure report could not be written: {report_error}')
            if isinstance(error, (SmokeFailure, data.DatasetError, ValueError, RuntimeError, OSError)):
                return EXIT_RESOURCES if isinstance(error, OSError) else EXIT_FAILED
            raise


def show_status(output):
    try:
        status = load_json(Path(output) / 'status.json')
    except (OSError, ValueError):
        print(json.dumps(dict(status='not_started', output=str(output)), indent=2))
        return EXIT_OK
    summary = {k: status.get(k) for k in ('status', 'phase', 'host', 'mode', 'updated_at', 'error', 'report')}
    dataset = status.get('dataset') or {}
    summary['dataset'] = {k: dataset.get(k) for k in ('state', 'rows_done', 'rows_total', 'manifest')}
    summary['dataset_progress'] = {f: {k: p.get(k) for k in ('phase', 'rows_done', 'rows_total', 'eta_seconds')}
                                   for f, p in (dataset.get('progress') or {}).items()}
    summary['gpu_smoke'] = status.get('smoke')
    launch = Path(output) / 'launch.json'
    if launch.exists():
        record = load_json(launch)
        summary['tmux'] = dict(host=record['host'], session=record['session'],
                               alive=tmux_alive(record['host'], record['session']))
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def convert_flavor(args):
    stop = threading.Event()
    def handler(*_):
        stop.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)
    spec, _ = data.load_spec(args.spec)
    record = data.convert_flavor(spec, args.convert_flavor, stop=stop)
    print(json.dumps({k: record[k] for k in ('flavor', 'action', 'rows', 'sha256', 'path')}))
    return EXIT_OK


def provenance(args):
    from experiments.stage2b.provenance import markdown, verify_root_provenance
    spec, _ = data.load_spec(args.spec)
    evidence = verify_root_provenance(spec)
    write_json(PREPARED / 'provenance.json', evidence)
    (PREPARED / 'PROVENANCE.md').write_text(markdown(evidence, spec))
    print(json.dumps(evidence['totals'], indent=2))
    return EXIT_OK


def check_parity(args):
    from experiments.stage2b import parity
    spec, _ = data.load_spec(args.spec)
    result = parity.check(spec)
    write_json(args.output / 'parity.json', result)
    print(json.dumps(dict(passed=result['passed'], failed=[c for c in result['checks'] if not c['passed']],
                          budget=result['budget'], cadence=result['cadence']), indent=2))
    return EXIT_OK if result['passed'] else EXIT_FAILED


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for flag in ('--dry-run', '--status', '--on-node', '--provenance', '--parity'):
        modes.add_argument(flag, action='store_true')
    modes.add_argument('--convert-flavor', choices=data.FLAVORS)
    parser.add_argument('--prepare-only', action='store_true', help='Convert/verify train8m only; no GPU smoke')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--cohort', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.status:
            return show_status(args.output)
        if args.convert_flavor:
            return convert_flavor(args)
        if args.provenance:
            return provenance(args)
        if args.parity:
            return check_parity(args)
        if args.on_node:
            if not args.cohort:
                parser.error('--on-node requires the internal launch cohort')
            code = on_node(args)
            write_json(args.output / 'supervisor_exit.json', dict(time=utc_now(), exit_code=code))
            return code
        return start(args)
    except BlockingIOError:
        print(f'{utc_now()} Stage2B: another launch or campaign holds the lock', file=sys.stderr)
        return EXIT_LOCKED
    except Exception as error:
        print(f'{utc_now()} Stage2B failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        try:
            write_json(args.output / 'last_failure.json', dict(time=utc_now(), error=f'{type(error).__name__}: {error}',
                                                               training_launched=False if args.dry_run else None))
        except OSError:
            pass
        if args.on_node:
            try:
                write_json(args.output / 'supervisor_exit.json', dict(time=utc_now(), exit_code=EXIT_FAILED, error=str(error)))
            except OSError:
                pass
        if isinstance(error, OSError) and error.errno == 28 or isinstance(error, RuntimeError):
            return EXIT_RESOURCES
        return EXIT_INVALID


if __name__ == '__main__':
    raise SystemExit(main())
