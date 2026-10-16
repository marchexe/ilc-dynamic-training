#!/usr/bin/env python3
"""Stage 3: matched fixed-LR vs frozen windowed_pbt_v2 campaign on train8m.

Default: read-only preflight on the submitting host (clean revision, frozen
protocol identity, arm parity, data and disk), GPU discovery, then submission
of one detached tmux supervisor that runs: sealed-test preparation, the
full-epoch gate, both arms (resuming existing runs), analysis, the sealed test
and the final report. Exit codes: 0 submitted/completed, 1 gate/arm/validity
failure, 2 invalid inputs or protocol drift, 69 unavailable resources,
75 locked, 130 interrupted.
"""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.stage2a.scheduling import discover, ssh_args  # noqa: E402
from experiments.stage2a_campaign import tmux_alive as default_socket_alive  # noqa: E402
from experiments.stage2b import dataset as data  # noqa: E402
from experiments.stage2b_campaign import choose_host, cluster_snapshot, git  # noqa: E402
from experiments.stage3.common import (ARMS, ARM_CONFIGS, EXIT_FAILED, EXIT_INTERRUPTED, EXIT_INVALID,  # noqa: E402
                                       EXIT_LOCKED, EXIT_OK, EXIT_RESOURCES, OUTPUT, PROJECT_DIR, PROTOCOL, disk_free,
                                       load_json, load_policy, lock_file, resolve, utc_now, verify_file, write_json)

TRACKED = ('scripts/experiments/stage3_campaign.py', 'scripts/experiments/run_stage3_train8m.sh', 'scripts/experiments/stage3',
           'configs/prepared/stage3_train8m_v1', 'configs/experiments/stage3_train8m_fixed_lr.yaml',
           'configs/experiments/stage3_train8m_windowed_pbt_v2.yaml', 'configs/experiments/stage3_train8m_gate.yaml',
           'datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_train8m_parquet_v1.json',
           'scripts/experiments/site_config.py', 'configs/site.example.json')

# tmux >= 3.5 moves every new session into a tmux-spawn scope under user@<uid>.service, where systemd-oomd
# kills on memory pressure; that killed the campaign on 2026-10-14. A server started without the user-manager
# bus keeps the campaign in the SSH session scope, which oomd does not monitor. A dedicated socket ensures the
# server is started this way rather than reusing an existing one.
TMUX_SOCKET = 'ilc-campaign'
TMUX = ['tmux', '-L', TMUX_SOCKET]
OOMD_MANAGED_SCOPE = '/user@'


def repository_revision():
    head, dirty = git('rev-parse', 'HEAD'), git('status', '--porcelain', '--untracked-files=no')
    if head.returncode or dirty.returncode:
        raise ValueError('Cannot establish the repository revision')
    if dirty.stdout.strip():
        raise ValueError('Tracked files have uncommitted changes; commit them before launching:\n' + dirty.stdout)
    for path in TRACKED:
        if git('ls-files', '--error-unmatch', path).returncode:
            raise ValueError(f'Stage 3 file is not committed: {path}')
    return dict(commit=head.stdout.strip(), branch=git('branch', '--show-current').stdout.strip(), dirty=False)


def parity_checks():
    """Stage 2B parity of the two arms plus the Stage 3 horizon/test-seal contract."""
    from experiments.stage2b import parity
    spec, _ = data.load_spec()
    result = parity.check(spec, configs=ARM_CONFIGS)
    configs = {arm: resolve(path) for arm, path in ARM_CONFIGS.items()}
    test_suffix = load_json(PROJECT_DIR / 'configs/prepared/stage3_train8m_v1/test3000k_spec.json')['suffix']
    extra = [dict(check='both arms configured for the fixed 50-epoch horizon',
                  passed=all(c['shared']['generations'] == 50 for c in configs.values())),
             dict(check='sealed test split is not used by any arm', passed=all(
                 test_suffix not in (c['shared']['train_suffix'], c['shared']['validation_suffix']) for c in configs.values())),
             dict(check='no continuation source (native horizon)', passed=not any(c.get('continuation') for c in configs.values()))]
    result['checks'] += extra
    result['passed'] = result['passed'] and all(c['passed'] for c in extra)
    return result


def preflight(output):
    from experiments.stage3 import protocol as frozen
    revision = repository_revision()
    protocol, protocol_sha = frozen.load()
    changed = frozen.drift(protocol)
    if changed:
        raise ValueError(f'Frozen protocol identity changed: {changed}')
    policy, _ = load_policy()
    parity = parity_checks()
    if not parity['passed']:
        raise ValueError('Arm parity failed: ' + '; '.join(c['check'] for c in parity['checks'] if not c['passed']))
    train_spec, _ = data.load_spec()
    data.verify_dataset(train_spec, rehash=False)
    free = disk_free(output)
    if free < policy['budgets']['min_free_disk_bytes']:
        raise OSError(28, f'Insufficient disk: {free} bytes free, {policy["budgets"]["min_free_disk_bytes"]} required')
    return dict(git=revision, protocol_sha256=protocol_sha, parity=dict(passed=True, checks=len(parity['checks'])),
                free_bytes=free, dataset='train8m quick-verified (full re-hash on node)'), protocol, protocol_sha, policy


def completed(output):
    try:
        status = load_json(Path(output) / 'status.json')
        for artifact in load_json(Path(output) / 'report' / 'report_integrity.json')['artifacts']:
            verify_file(artifact)
        return status.get('status') == 'completed'
    except (OSError, ValueError, KeyError, TypeError):
        return False


def tmux_alive(host, session):
    """Alive on the campaign socket, or on the default socket used by launches before the oomd fix."""
    result = subprocess.run(ssh_args(host, shlex.join([*TMUX, 'has-session', '-t', '=' + session])),
                            capture_output=True, text=True, timeout=15)
    if result.returncode not in (0, 1):
        raise RuntimeError(f'Cannot establish persistent-session status on {host}: {result.stderr.strip()}')
    return result.returncode == 0 or default_socket_alive(host, session)


def launch_tmux(host, session, output, cohort_path, *, retry_gate=False):
    """Start the supervisor in a detached tmux session outside oomd management; return its cgroup."""
    probe = 'command -v tmux >/dev/null && test -x ' + shlex.quote(str(PROJECT_DIR / '.venv/bin/python'))
    subprocess.run(ssh_args(host, probe), check=True, capture_output=True, text=True, timeout=20)
    script = shlex.join([str(PROJECT_DIR / '.venv/bin/python'), str(Path(__file__).resolve()), '--on-node', '--output',
                         str(output), '--cohort', str(cohort_path), *(['--retry-gate'] if retry_gate else [])])
    inner = 'cd ' + shlex.quote(str(PROJECT_DIR)) + ' && exec ' + script + ' >> ' + shlex.quote(str(output / 'supervisor.log')) + ' 2>&1'
    detached = ['env', '-u', 'DBUS_SESSION_BUS_ADDRESS', '-u', 'XDG_RUNTIME_DIR', *TMUX]
    subprocess.run(ssh_args(host, shlex.join([*detached, 'new-session', '-d', '-s', session, 'bash', '-lc', inner])),
                   check=True, capture_output=True, text=True, timeout=20)
    pane = shlex.join([*TMUX, 'list-panes', '-t', '=' + session, '-F', '#{pane_pid}'])
    cgroup = subprocess.run(ssh_args(host, f'cat /proc/$({pane} | head -n 1)/cgroup'), check=True, capture_output=True,
                            text=True, timeout=20).stdout.strip()
    if OOMD_MANAGED_SCOPE in cgroup:
        subprocess.run(ssh_args(host, shlex.join([*TMUX, 'kill-session', '-t', '=' + session])), capture_output=True,
                       text=True, timeout=20)
        raise RuntimeError(f'Supervisor landed in an oomd-managed scope ({cgroup}); session stopped, nothing trained')
    return cgroup


def start(args):
    from experiments.stage3 import gate as gate_module
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with lock_file(output / '.launch.lock'):
        if not args.dry_run and completed(output):
            print(json.dumps(dict(status='already_completed_and_verified', report=str(output / 'report' / 'report.md')), indent=2))
            return EXIT_OK
        launch = output / 'launch.json'
        if not args.dry_run and launch.exists():
            previous = load_json(launch)
            if tmux_alive(previous['host'], previous['session']):
                print(json.dumps(dict(status='already_running', **previous), indent=2))
                return EXIT_OK
        checks, protocol, protocol_sha, policy = preflight(output)
        verdict = gate_module.gate_record(output)
        if (verdict and verdict.get('protocol_sha256') == protocol_sha and verdict['verdict'] == gate_module.CRITERIA
                and not args.retry_gate):
            print(json.dumps(dict(status='blocked_by_failed_gate', verdict=str(output / 'gate' / 'verdict.json'),
                                  failed=[c['check'] for c in verdict['checks'] if not c['passed']]), indent=2))
            return EXIT_FAILED
        allocation = policy['allocation']
        nodes = discover(allocation)
        try:
            cohort = choose_host(nodes, dict(allocation, gpus_required=allocation['gpus_per_arm']))
        except RuntimeError as error:
            write_json(output / 'resource_failure.json', dict(checked_at=utc_now(), error=str(error), nodes=nodes))
            raise
        idle = sum(1 for n in nodes if n['host'] == cohort['host'] for g in n['gpus']
                   if g['name'] == cohort['name'] and not g['pids'] and g['used_mib'] < allocation['max_used_memory_mib']
                   and g['utilization'] < allocation['max_util_percent'])
        plan = dict(status='ready', checked_at=utc_now(), preflight=checks, cohort=cohort, idle_campaign_gpus_observed=idle,
                    expected_schedule='concurrent arms (10 GPUs)' if idle >= 10 else 'arms start as five-GPU sets free up',
                    gate_state=(verdict or {}).get('verdict', 'pending'), gpus_reserved=False, training_launched=False)
        if args.dry_run:
            write_json(output / 'dry_run.json', dict(plan, gpu_observation=nodes))
            print(json.dumps(plan, indent=2))
            return EXIT_OK
        (output / 'cluster_snapshot.txt').write_text(cluster_snapshot())
        write_json(output / 'runtime_preflight.json', dict(plan, gpu_observation=nodes))
        cohort_path = output / 'launch_cohort.json'
        write_json(cohort_path, dict(cohort, git=checks['git'], protocol_sha256=protocol_sha))
        session = 'stage3-train8m-' + protocol_sha[:8]
        try:
            cgroup = launch_tmux(cohort['host'], session, output, cohort_path, retry_gate=args.retry_gate)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as error:
            detail = getattr(error, 'stderr', '') or str(error)
            write_json(output / 'last_failure.json', dict(time=utc_now(), error=f'Remote submission failed: {detail}'))
            raise RuntimeError(f'Remote tmux submission to {cohort["host"]} failed: {detail}') from error
        record = dict(host=cohort['host'], session=session, started_at=utc_now(), git=checks['git'],
                      protocol_sha256=protocol_sha, output=str(output), tmux_socket=TMUX_SOCKET, cgroup=cgroup)
        write_json(launch, record)
        print(json.dumps(dict(status='submitted_to_persistent_tmux', **record,
                              watch='bash scripts/experiments/run_stage3_train8m.sh --status',
                              attach=f'ssh -t {cohort["host"]} tmux -L {TMUX_SOCKET} attach -t {session}'), indent=2))
        return EXIT_OK


def on_node(args):
    from experiments.stage2b.smoke import Status
    from experiments.stage3 import campaign, protocol as frozen
    cohort = load_json(args.cohort)
    if socket.gethostname().split('.')[0] != cohort['host']:
        raise ValueError('Supervisor started on a different node than the launch cohort')
    stop = threading.Event()
    campaign.install_stop(stop)
    output = args.output.resolve()
    with lock_file(output / '.campaign.lock') as campaign_lock:
        protocol, protocol_sha = frozen.load()
        revision = repository_revision()
        changed = frozen.drift(protocol)
        if revision['commit'] != cohort['git']['commit'] or protocol_sha != cohort['protocol_sha256'] or changed:
            raise ValueError(f'Revision or frozen protocol changed after submission: {changed}')
        policy, _ = load_policy()
        status = Status(output, dict(git=revision, protocol_sha256=protocol_sha, host=cohort['host'], started_at=utc_now()))
        fds = (campaign_lock.fileno(),)
        try:
            campaign.prepare_data(protocol, policy, status, stop)
            gate = campaign.run_gate(protocol, protocol_sha, policy, cohort, output, status, stop, retry=args.retry_gate,
                                     inherit_fds=fds)
            outcome = campaign.run_arms(protocol, policy, cohort, output, status, stop, gate, inherit_fds=fds)
            if any(state != 'completed' for state in outcome.values()):
                status.finish('failed', arms_outcome=outcome, error=f'Arms did not complete: {outcome}')
                return EXIT_RESOURCES if 'waiting_timeout' in outcome.values() else EXIT_FAILED
            report = campaign.finalize(protocol, protocol_sha, policy, cohort, output, status, stop, gate)
            ok = report['status'] == 'COMPLETE'
            status.finish('completed' if ok else 'failed', report=str(output / 'report' / 'report.md'),
                          report_status=report['status'])
            return EXIT_OK if ok else EXIT_FAILED
        except InterruptedError as error:
            status.finish('interrupted', error=str(error))
            return EXIT_INTERRUPTED
        except campaign.GateBlocked as error:
            status.finish('gate_blocked', error=str(error))
            return EXIT_FAILED
        except Exception as error:
            status.finish('failed', error=f'{type(error).__name__}: {error}')
            status.log('FAIL', f'{type(error).__name__}: {error}')
            return EXIT_RESOURCES if isinstance(error, OSError) else EXIT_FAILED


def show_status(output):
    try:
        status = load_json(Path(output) / 'status.json')
    except (OSError, ValueError):
        print(json.dumps(dict(status='not_started', output=str(output)), indent=2))
        return EXIT_OK
    summary = {k: status.get(k) for k in ('status', 'phase', 'host', 'updated_at', 'error', 'report', 'report_status',
                                          'gpu_hours_consumed', 'arms')}
    verdict = Path(output) / 'gate' / 'verdict.json'
    if verdict.is_file():
        record = load_json(verdict)
        summary['gate'] = dict(verdict=record['verdict'], attempt=record['attempt'],
                               member_epoch_seconds=record['measurements'].get('member_epoch_seconds'))
    launch = Path(output) / 'launch.json'
    if launch.exists():
        record = load_json(launch)
        summary['tmux'] = dict(host=record['host'], session=record['session'], alive=tmux_alive(record['host'], record['session']))
    print(json.dumps(summary, indent=2, default=str))
    return EXIT_OK


def interim_report(output):
    from experiments.stage3 import analysis, campaign, gate as gate_module, protocol as frozen
    protocol, _ = frozen.load()
    runs = {arm: str(campaign.arm_run(protocol, arm)) for arm in ARMS}
    end = min(len(analysis.completed(load_json(Path(r) / 'manifest.json'))) for r in runs.values())
    target = Path(output) / 'interim' / f'epoch-{end:03d}'
    result = analysis.build(target, protocol, runs, interim=True, gate=gate_module.gate_record(output))
    print(json.dumps(dict(status=result['status'], completed_generations=end, report=str(target / 'report.md')), indent=2))
    return EXIT_OK


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for flag in ('--dry-run', '--status', '--on-node', '--parity', '--freeze', '--interim-report'):
        modes.add_argument(flag, action='store_true')
    parser.add_argument('--retry-gate', action='store_true', help='Retry a gate that failed its criteria (after review)')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.status:
            return show_status(args.output)
        if args.parity:
            result = parity_checks()
            print(json.dumps(dict(passed=result['passed'], failed=[c for c in result['checks'] if not c['passed']],
                                  budget=result['budget']), indent=2))
            return EXIT_OK if result['passed'] else EXIT_FAILED
        if args.freeze:
            from experiments.stage3 import protocol as frozen
            if PROTOCOL.exists():
                raise ValueError('Protocol is already frozen; a new protocol needs a new version directory')
            protocol = frozen.freeze()
            print(json.dumps(dict(protocol=str(PROTOCOL), horizon=protocol['horizon']['decision']), indent=2))
            return EXIT_OK
        if args.interim_report:
            return interim_report(args.output)
        if args.on_node:
            if not args.cohort:
                parser.error('--on-node requires the internal launch cohort')
            code = on_node(args)
            write_json(args.output / 'supervisor_exit.json', dict(time=utc_now(), exit_code=code))
            return code
        return start(args)
    except BlockingIOError:
        print(f'{utc_now()} Stage3: another launch or campaign holds the lock', file=sys.stderr)
        return EXIT_LOCKED
    except Exception as error:
        print(f'{utc_now()} Stage3 failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        try:
            write_json(args.output / 'last_failure.json', dict(time=utc_now(), error=f'{type(error).__name__}: {error}'))
            if args.on_node:
                write_json(args.output / 'supervisor_exit.json', dict(time=utc_now(), exit_code=EXIT_INVALID, error=str(error)))
        except OSError:
            pass
        if isinstance(error, OSError) and error.errno == 28 or isinstance(error, RuntimeError):
            return EXIT_RESOURCES
        return EXIT_INVALID


if __name__ == '__main__':
    raise SystemExit(main())
