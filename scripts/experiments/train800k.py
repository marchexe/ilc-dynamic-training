#!/usr/bin/env python3
"""Place a train800k experiment on idle GPUs of one authorized node and start it there.

    .venv/bin/python scripts/experiments/train800k.py {start,resume,status,stop} --config C [--dry-run]

Runs from the login node. ``start`` probes the site's cluster nodes, picks one node with one idle GPU per
member, all of one model (the preferred node first, as in the Stage 2A/3 campaigns), and starts the run
there with ``scripts/launch/experiment.py``, which re-checks the GPUs immediately before launching. Partial
GPU sets are never taken. The runner must not land in a scope that systemd-oomd manages (the 2026-10-14
Stage 3 failure); if it does, it is stopped before its first epoch completes. ``resume``, ``status`` and
``stop`` run ``experiment.py`` on the recorded node. Nothing here waits for, retries or takes over GPUs.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import site_config  # noqa: E402
from experiments.stage2a.common import write_json  # noqa: E402
from experiments.stage2a.scheduling import discover, free_gpu, ssh_args  # noqa: E402
from launch.experiment import configuration, locations  # noqa: E402
from training.runtime import PROJECT_DIR, utc_now  # noqa: E402

# Same idleness rule as the inference campaigns; 40 GB covers the measured ~25 GB training peak.
POLICY = dict(max_used_memory_mib=500, max_util_percent=5, min_gpu_memory_mib=40000)
OOMD_MANAGED_SCOPE = '/user@'
PLACEMENTS = 'placements.json'  # one record per start/resume, newest last


def remote(host, arguments, *, check=True):
    return subprocess.run(ssh_args(host, arguments), capture_output=True, text=True, timeout=120, check=check)


def experiment_command(action, config_path, gpus=None):
    command = [str(PROJECT_DIR / '.venv/bin/python'), str(PROJECT_DIR / 'scripts/launch/experiment.py'), action,
               '--config', str(Path(config_path).resolve())]
    if gpus is not None:
        command += ['--gpus', ','.join(str(gpu['index']) for gpu in gpus)]
    # Without the user-manager bus nothing the runner starts can be moved under user@<uid>.service.
    return 'cd ' + shlex.quote(str(PROJECT_DIR)) + ' && ' + shlex.join(
        ['env', '-u', 'DBUS_SESSION_BUS_ADDRESS', '-u', 'XDG_RUNTIME_DIR', *command])


def placement(config, nodes, *, host=None):
    """One node with one idle GPU per member, all of one model; None if no node has enough.

    Among nodes that have enough, the order of the Stage 2A/3 campaigns: preferred node, A100, memory.
    """
    policy = dict(POLICY, **site_config.cluster())
    needed = len(config['population'])
    groups = {}
    for node in nodes:
        if node['host'] in policy['authorized_nodes'] and host in (None, node['host']):
            for gpu in node['gpus']:
                if free_gpu(gpu, policy):
                    groups.setdefault((node['host'], gpu['name'], gpu['total_mib']), []).append(gpu)
    eligible = [(key, gpus) for key, gpus in groups.items() if len(gpus) >= needed]
    if not eligible:
        return None
    (chosen, name, memory), gpus = max(eligible, key=lambda item: (
        item[0][0] == policy['preferred_node'], 'A100' in item[0][1], item[0][2], item[0][0]))
    return dict(host=chosen, name=name, total_mib=memory, gpus=sorted(gpus, key=lambda g: g['index'])[:needed])


def placements(config):
    path = locations(config)[1] / PLACEMENTS
    return json.loads(path.read_text()) if path.is_file() else []


def recorded_host(config):
    history = placements(config)
    if not history:
        raise FileNotFoundError(f'No placement recorded for {config["experiment_name"]}; it was not started here')
    return history[-1]['host']


def launch(action, config, config_path, *, dry_run=False, discover_nodes=discover, run=remote):
    logs = locations(config)[1]
    host = None
    if action == 'resume':
        host = recorded_host(config)
    elif logs.exists():
        raise FileExistsError(f'Launcher output already exists: {logs}; use resume or status')
    nodes = discover_nodes(dict(site_config.cluster()))
    cohort = placement(config, nodes, host=host)
    plan = dict(action=action, experiment=config['experiment_name'], checked_at=utc_now(),
                members=[m['name'] for m in config['population']], cohort=cohort,
                observation=[dict(host=n['host'], error=n['error'],
                                  idle=sum(1 for g in n['gpus'] if not g['pids'] and g['used_mib'] < POLICY['max_used_memory_mib']))
                             for n in nodes])
    if cohort is None:
        where = f'on {host}' if host else 'on any authorized node'
        raise RuntimeError(f'No {len(config["population"])} idle GPUs of one model {where}; nothing launched\n'
                           + json.dumps(plan['observation'], indent=2))
    if dry_run:
        return plan
    started = run(cohort['host'], experiment_command(action, config_path, cohort['gpus']), check=False)
    if started.returncode != 0:
        raise RuntimeError(f'experiment.py {action} on {cohort["host"]} failed: '
                           f'{(started.stderr or started.stdout).strip()}')
    # Read on the node: the login node's NFS client may not see the new directory for a while.
    launcher = json.loads(run(cohort['host'], 'cat ' + shlex.quote(str(logs / 'launcher.json'))).stdout)
    cgroup = run(cohort['host'], f'cat /proc/{int(launcher["pid"])}/cgroup').stdout.strip()
    record = dict(plan, host=cohort['host'], gpus=cohort['gpus'], gpu_model=cohort['name'], runner_pid=launcher['pid'],
                  runner_cgroup=cgroup, output=started.stdout.strip())
    os.listdir(logs.parent)  # revalidates the cached lookup of the directory the node just created
    write_json(logs / PLACEMENTS, [*placements(config), record])
    if OOMD_MANAGED_SCOPE in cgroup:
        run(cohort['host'], experiment_command('stop', config_path), check=False)
        raise RuntimeError(f'Runner landed in an oomd-managed scope ({cgroup}); stopped it')
    return record


def on_recorded_host(action, config, config_path, *, run=remote):
    result = run(recorded_host(config), experiment_command(action, config_path), check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['start', 'resume', 'status', 'stop'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='probe and plan only; start or resume nothing')
    args = parser.parse_args(argv)
    try:
        config = configuration(args.config)
        if args.action in ('status', 'stop'):
            return on_recorded_host(args.action, config, args.config)
        print(json.dumps(launch(args.action, config, args.config, dry_run=args.dry_run), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 69 if isinstance(error, RuntimeError) else 2


if __name__ == '__main__':
    raise SystemExit(main())
