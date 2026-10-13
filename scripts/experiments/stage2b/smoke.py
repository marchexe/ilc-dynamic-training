"""On-node Stage 2B supervisor: dataset phase first, then the bounded GPU smoke.

Phases are recorded in status.json and prefixed in campaign.log as [DATASET]
(CPU conversion/verification) or [GPU] (smoke training), so conversion progress
is never confused with training progress. The smoke runs the production
runner twice in new processes: first until a deliberate SIGINT inside the
configured generation's training phase, then with --resume.
"""
import csv
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from experiments.stage2a.execution import acquire_gpu
from experiments.stage2b.common import (PROJECT_DIR, SMOKE_CONFIG, DRAFT_CONFIGS, load_json, log, own_processes_with,
                                        utc_now, write_json)
from experiments.stage2b import dataset as data

RUNNER = PROJECT_DIR / 'scripts/training/run_pbt.py'


class SmokeFailure(RuntimeError):
    """A prerequisite or a smoke check failed; recorded, never retried silently."""


class Status:
    """Single writer of status.json; every update is atomic."""

    def __init__(self, output, base):
        self.path = Path(output) / 'status.json'
        self.log_path = Path(output) / 'campaign.log'
        self.value = dict(base, status='running', phase=None, phases=[], updated_at=utc_now())
        self.save()

    def save(self):
        self.value['updated_at'] = utc_now()
        write_json(self.path, self.value)

    def phase(self, name, kind):
        if self.value['phases'] and 'finished_at' not in self.value['phases'][-1]:
            self.value['phases'][-1]['finished_at'] = utc_now()
        self.value['phase'] = name
        self.value['phases'].append(dict(phase=name, kind=kind, started_at=utc_now()))
        self.save()
        self.log(kind, f'phase {name}')

    def log(self, kind, message):
        log(self.log_path, f'[{kind}] {message}')

    def update(self, **values):
        self.value.update(values)
        self.save()

    def finish(self, status, **values):
        if self.value['phases'] and 'finished_at' not in self.value['phases'][-1]:
            self.value['phases'][-1]['finished_at'] = utc_now()
        self.value.update(values, status=status, finished_at=utc_now())
        self.save()


# ---------------------------------------------------------------- dataset phase

def conversion_command(flavor, spec_path=None):
    command = [sys.executable, str(PROJECT_DIR / 'scripts/experiments/stage2b_campaign.py'), '--convert-flavor', flavor]
    return command + (['--spec', str(spec_path)] if spec_path else [])


def run_dataset_phase(spec, policy, status, stop, *, launch=subprocess.Popen, poll_seconds=10, log_seconds=60,
                      verify=data.verify_dataset, isolation=None, publish=data.publish_manifest, spec_path=None,
                      prefix='dataset', kind='DATASET'):
    """Prepare (if needed) and fully verify train8m; returns the realized manifest."""
    from experiments.stage2b.provenance import verify_realized_isolation
    isolation = isolation or verify_realized_isolation
    status.phase(f'{prefix}_preflight', kind)
    pre = data.preflight(spec)
    status.update(dataset=dict(state=pre['status']['state'], free_bytes=pre['free_bytes'],
                               required_free_bytes=pre['required_free_bytes']))
    pending = [f for f in data.FLAVORS if f not in pre['status']['completed']]
    if pending:
        status.phase(f'{prefix}_conversion', kind)
        status.log(kind, f'converting {",".join(pending)} (expected {spec["expected_rows_total"]:,} rows in total)')
        workers = int(policy['budgets']['conversion_workers'])
        queue, running, logged = list(pending), {}, None
        try:
            while queue or running:
                while queue and len(running) < workers:
                    flavor = queue.pop(0)
                    stream = (Path(status.path.parent) / 'logs' / f'convert-{flavor}.log')
                    stream.parent.mkdir(parents=True, exist_ok=True)
                    handle = stream.open('a')
                    running[flavor] = (launch(conversion_command(flavor, spec_path), cwd=PROJECT_DIR, stdout=handle,
                                              stderr=subprocess.STDOUT, start_new_session=True,
                                              preexec_fn=lambda: os.nice(int(policy['budgets']['conversion_nice']))), handle)
                for flavor, (process, handle) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    running.pop(flavor)
                    if code != 0:
                        raise SmokeFailure(f'train8m {flavor} conversion failed (exit {code}); see logs/convert-{flavor}.log')
                    status.log(kind, f'{flavor} converted and verified')
                progress = {}
                for flavor in data.FLAVORS:
                    try:
                        progress[flavor] = load_json(data.state_dir(spec) / 'progress' / f'{flavor}.json')
                    except (OSError, ValueError):
                        progress[flavor] = dict(phase='queued' if flavor in queue else 'done' if flavor not in pending else 'starting')
                done = sum(p.get('rows_done', 0) for p in progress.values())
                status.update(dataset=dict(status.value['dataset'], state='converting', progress=progress,
                                           rows_done=done, rows_total=spec['expected_rows_total']))
                if logged is None or time.monotonic() - logged >= log_seconds or not (queue or running):
                    logged = time.monotonic()
                    status.log(kind, f'conversion {100 * done / spec["expected_rows_total"]:.1f}% '
                               + ' '.join(f'{f}:{p.get("phase")}:{p.get("rows_done", 0):,}' for f, p in progress.items()))
                if (queue or running) and stop.wait(poll_seconds):
                    raise InterruptedError('Supervisor stopped during conversion')
        finally:
            for process, handle in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                handle.close()
    status.phase(f'{prefix}_verification', kind)
    manifest = verify(spec, rehash=True)
    manifest_path = publish(spec, manifest)
    status.phase(f'{prefix}_isolation' if prefix != 'dataset' else 'split_isolation', kind)
    evidence = isolation(spec)
    write_json(Path(status.path.parent) / f'{prefix}_isolation.json', evidence)
    status.update(dataset=dict(state='complete_verified', manifest=str(manifest_path), rows_total=manifest['rows_total'],
                               size_bytes_total=manifest['size_bytes_total'],
                               fingerprint=manifest['weaver_audit']['fingerprint']))
    status.log(kind, f'train8m ready: {manifest["rows_total"]:,} rows, {manifest["size_bytes_total"]:,} bytes')
    return manifest


# ---------------------------------------------------------------- process/GPU observation

def process_table(proc=Path('/proc')):
    table = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            rss = next((int(line.split()[1]) * 1024 for line in (entry / 'status').read_text().splitlines()
                        if line.startswith('VmRSS:')), 0)
        except (OSError, IndexError, ValueError):
            continue
        table[int(entry.name)] = dict(ppid=int(fields[1]), rss=rss)
    return table


def descendants(root, table):
    found, frontier = {root}, [root]
    while frontier:
        parent = frontier.pop()
        for pid, item in table.items():
            if item['ppid'] == parent and pid not in found:
                found.add(pid)
                frontier.append(pid)
    return found


def gpu_apps():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    apps = []
    for row in csv.reader(result.stdout.splitlines()):
        if row and any(x.strip() for x in row):
            pid, gpu, used = (x.strip() for x in row)
            apps.append(dict(pid=int(pid), uuid=gpu, used_mib=int(used) if used.isdigit() else 0))
    return apps


class Observer:
    """Samples the runner's process tree, its GPU memory and foreign GPU users."""

    def __init__(self, uuids, policy, *, apps=gpu_apps, table=process_table):
        self.uuids, self.policy, self.apps, self.table = set(uuids), policy, apps, table
        self.peak_gpu_memory_mib, self.peak_rss_bytes = 0, 0
        self.foreign_streak, self.foreign_events = 0, []

    def sample(self, root):
        """`root` is one runner PID or several (concurrent arms share one budget)."""
        table = self.table()
        tree = set()
        for pid in (root if isinstance(root, (list, tuple, set)) else [root]):
            tree |= descendants(pid, table) if pid in table else {pid}
        apps = [a for a in self.apps() if a['uuid'] in self.uuids]
        per_gpu = {}
        for app in apps:
            if app['pid'] in tree:
                per_gpu[app['uuid']] = per_gpu.get(app['uuid'], 0) + app['used_mib']
        self.peak_gpu_memory_mib = max([self.peak_gpu_memory_mib, *per_gpu.values()])
        self.peak_rss_bytes = max(self.peak_rss_bytes, sum(table[p]['rss'] for p in tree if p in table))
        foreign = [a for a in apps if a['pid'] not in tree]
        self.foreign_streak = self.foreign_streak + 1 if foreign else 0
        if self.foreign_streak >= self.policy['allocation']['foreign_process_grace_samples']:
            self.foreign_events.append(dict(time=utc_now(), processes=foreign))
            raise SmokeFailure(f'Foreign GPU process on a leased smoke GPU: {foreign}; stopping only this smoke')
        if self.peak_gpu_memory_mib > self.policy['budgets']['max_cuda_memory_mib']:
            raise SmokeFailure(f'GPU memory budget exceeded: {self.peak_gpu_memory_mib} MiB')
        return dict(processes=len(tree), gpu_memory_mib=per_gpu)


# ---------------------------------------------------------------- runner control

def runner_command(name, gpus, *, resume=False, config=SMOKE_CONFIG):
    command = [sys.executable, str(RUNNER), '--config', str(config), '--gpus', ','.join(str(i) for i in range(gpus)),
               '--experiment-name', name]
    return command + (['--resume'] if resume else [])


def read_manifest(run):
    try:
        return load_json(Path(run) / 'manifest.json')
    except (OSError, ValueError):
        return None


def stop_runner(process, timeout, token, *, own=own_processes_with):
    """Graceful SIGINT (the runner terminates its own workers), then force.

    Returns the runner exit code and any token-carrying stragglers, recorded
    before they are killed so clean termination is evidence, not assumption.
    """
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    stragglers = own(token)
    for pid in stragglers:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return process.returncode, stragglers


def interruption_ready(run, generation, now, state):
    """True once every worker of `generation` is running and has entered training."""
    manifest = read_manifest(run) or {}
    record = next((g for g in manifest.get('generations', []) if g['index'] == generation), None)
    if (manifest.get('status') != 'running' or not record
            or any(w.get('status') != 'running' for w in record['workers'].values())):
        return False
    epoch = state['epoch'] = record['epoch']
    for worker in record['workers'].values():
        path = Path(worker.get('log', ''))
        if not path.is_file() or f'Epoch #{epoch} training' not in path.read_text(errors='replace'):
            return False
    state.setdefault('training_seen_at', now)
    return now - state['training_seen_at'] >= state['delay']


def supervise(command, env, observer, status, *, run, deadline, stop, interrupt=None, popen=subprocess.Popen,
              clock=time.monotonic, poll_seconds=5, token, kind, inherit_fds=(), own=own_processes_with):
    """Run one runner process under budget/foreign-process/stop monitoring."""
    handle = (Path(status.path.parent) / 'logs' / f'{kind}.console.log')
    handle.parent.mkdir(parents=True, exist_ok=True)
    with handle.open('a') as stream:
        stream.write(f'{utc_now()} {shlex.join(command)}\n')
        stream.flush()
        # Inherited leases keep GPUs/campaign locked for as long as the runner lives.
        process = popen(command, cwd=PROJECT_DIR, env=env, stdout=stream, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=tuple(inherit_fds))
        state = dict(delay=(interrupt or {}).get('delay', 0))
        try:
            while process.poll() is None:
                observer.sample(process.pid)
                if stop.is_set():
                    raise InterruptedError('Supervisor stopped')
                if clock() > deadline:
                    raise SmokeFailure('Smoke wall-clock/GPU-hour budget exhausted')
                manifest = read_manifest(run) or {}
                status.update(smoke=dict(status.value.get('smoke', {}), runner=kind, generation=manifest.get('next_generation'),
                                         manifest_status=manifest.get('status')))
                if interrupt and interruption_ready(run, interrupt['generation'], clock(), state):
                    epoch = state['epoch']
                    present = sorted(str(p) for p in Path(run).glob(f'*/net_epoch-{epoch}_*.pt'))
                    status.log('GPU', f'interrupting generation {interrupt["generation"]} during training (SIGINT)')
                    code, stragglers = stop_runner(process, interrupt['timeout'], token, own=own)
                    after = read_manifest(run) or {}
                    record = next((g for g in after.get('generations', []) if g['index'] == interrupt['generation']), {})
                    return dict(exit_code=code, interruption=dict(
                        triggered=True, generation=interrupt['generation'], epoch=epoch, runner_exit_code=code,
                        manifest_status=after.get('status'), checkpoints_present_at_interrupt=present,
                        terminated_workers=sorted(n for n, w in record.get('workers', {}).items()
                                                  if w.get('status') == 'terminated'),
                        stragglers=stragglers, at=utc_now()))
                time.sleep(poll_seconds)
        except BaseException:
            stop_runner(process, 120, token, own=own)
            raise
    return dict(exit_code=process.returncode, interruption=dict(triggered=False) if interrupt else None)


def next_attempt(output, prefix):
    directory = Path(output) / 'attempts'
    directory.mkdir(parents=True, exist_ok=True)
    number = 1 + len(list(directory.glob('attempt-*')))
    path = directory / f'attempt-{number:03d}'
    path.mkdir()
    return number, path, f'{prefix}_a{number:03d}'


def smoke_run_root(config=SMOKE_CONFIG):
    """Where the production runner writes the smoke run (its resolved output_root)."""
    import argparse
    from training.pbt.config import load_config
    return Path(load_config(argparse.Namespace(config=Path(config), experiment_name=None, gpus=None, slots=None,
                                               smoke=False))['output_root'])


def run_smoke(spec, policy, dataset_manifest, cohort, status, stop, *, validation_audit, inherit_fds=(),
              run_root=None, acquire=acquire_gpu, observer_factory=Observer, popen=subprocess.Popen,
              clock=time.monotonic, poll_seconds=5, own=own_processes_with, verify=None):
    from experiments.stage2b.verify import verify_run
    verify = verify or verify_run
    smoke, allocation, budgets = policy['smoke'], policy['allocation'], policy['budgets']
    output = Path(status.path.parent)
    number, attempt_dir, name = next_attempt(output, smoke['experiment_prefix'])
    run = Path(run_root or smoke_run_root()) / name
    status.update(smoke=dict(attempt=number, experiment=name, run=str(run)))
    status.phase('gpu_acquisition', 'GPU')
    leases, gpus = [], []
    try:
        for _ in range(allocation['gpus_required']):
            gpu, lease = acquire(output, cohort, allocation, stop)
            leases.append(lease)
            gpus.append(gpu)
        write_json(attempt_dir / 'gpus.json', gpus)
        status.log('GPU', 'leased ' + ', '.join(f'{g["host"]}:{g["index"]} {g["name"]}' for g in gpus))
        token = uuid.uuid4().hex
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in gpus), PYTHONUNBUFFERED='1',
                   STAGE2B_SMOKE_TOKEN=token, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        wall = min(budgets['max_wall_seconds'], budgets['max_gpu_hours'] * 3600 / len(gpus))
        deadline = clock() + wall
        observer = observer_factory([g['uuid'] for g in gpus], policy)
        status.phase('smoke_training', 'GPU')
        fds = (*inherit_fds, *(lease.fileno() for lease in leases))
        common = dict(stop=stop, popen=popen, clock=clock, poll_seconds=poll_seconds, token=token,
                      inherit_fds=fds, own=own, run=run, deadline=deadline)
        first = supervise(runner_command(name, len(gpus)), env, observer, status, kind=f'{name}.phaseA',
                          interrupt=dict(generation=smoke['interrupt_generation'], delay=smoke['interrupt_delay_seconds'],
                                         timeout=budgets['interrupt_exit_timeout_seconds']), **common)
        interruption = first['interruption']
        observations = dict(interruption=interruption, first_exit_code=first['exit_code'])
        write_json(attempt_dir / 'interruption.json', observations)
        if not interruption.get('triggered'):
            raise SmokeFailure(f'Interruption point was missed (runner exit {first["exit_code"]})')
        if (interruption['runner_exit_code'] != 130 or interruption['manifest_status'] != 'interrupted'
                or interruption['stragglers'] or interruption['checkpoints_present_at_interrupt']):
            raise SmokeFailure(f'Interruption was not clean: {interruption}')
        status.phase('smoke_resume', 'GPU')
        second = supervise(runner_command(name, len(gpus), resume=True), env, observer, status,
                           kind=f'{name}.phaseB', **common)
        observations.update(resume_exit_code=second['exit_code'], leftover_processes=own(token),
                            peak_gpu_memory_mib=observer.peak_gpu_memory_mib, peak_rss_bytes=observer.peak_rss_bytes,
                            foreign_gpu_events=observer.foreign_events, gpus=gpus, wall_seconds_budget=wall)
        write_json(attempt_dir / 'observations.json', observations)
        if second['exit_code'] != 0:
            raise SmokeFailure(f'Resumed runner failed (exit {second["exit_code"]}); see {run}/logs/pbt.log')
    finally:
        for lease in leases:
            lease.close()
    status.phase('verification', 'GPU')
    verification = verify(run, dataset_manifest=dataset_manifest, validation_audit=validation_audit,
                              policy=policy, observations=observations)
    write_json(attempt_dir / 'verification.json', verification)
    return dict(attempt=number, run=str(run), observations=observations, verification=verification)


def draft_dry_runs(policy, *, run=subprocess.run, configs=DRAFT_CONFIGS):
    """Real --dry-run of the inactive Stage 3 drafts against the prepared data."""
    results = {}
    for key in policy['smoke']['draft_dry_runs']:
        command = [sys.executable, str(RUNNER), '--config', str(configs[key]), '--dry-run']
        result = run(command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=600)
        weaver = [line for line in result.stdout.splitlines() if line.startswith('[') and 'weaver' in line]
        train = [line for line in weaver if '--data-train' in line]
        results[key] = dict(command=command, returncode=result.returncode, weaver_commands=len(weaver),
                            uses_train8m=bool(train) and all('_train8m.parquet' in line and 'train800k' not in line
                                                             for line in train),
                            stderr=result.stderr[-2000:])
    return results
