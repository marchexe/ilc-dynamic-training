"""On-node Stage 3 supervisor (runs inside the detached tmux session).

Phase order: PREP (data verification, sealed-test preparation) -> GATE (one
real full epoch; blocks everything on failure) -> ARMS (fixed LR and windowed
PBT through the production runner, each on five leased GPUs of one model) ->
REPORT (validation analysis, selection freeze, sealed test, final report).
Runs never restart from epoch 17 once a run directory exists: an existing,
unfinished run is resumed with `--resume` after its last committed checkpoint
bundle re-verifies against the ledger.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from experiments.stage2a.execution import acquire_gpu
from experiments.stage2b import dataset as data
from experiments.stage2b.smoke import Observer, run_dataset_phase, stop_runner, supervise
from experiments.stage3.common import (ARMS, ARM_CONFIGS, GATE_CONFIG, PROJECT_DIR, TEST_SPEC, load_json,
                                       own_processes_with, project_file, resolve, sha256, utc_now, write_json)
from training.checkpoints import bundle_paths
from training.pbt.config import contract_fingerprint

RUNNER = PROJECT_DIR / 'scripts/training/run_pbt.py'


class CampaignFailure(RuntimeError):
    """A prerequisite, gate or arm failed; recorded and never silently bypassed."""


class GateBlocked(CampaignFailure):
    pass


# ----------------------------------------------------------------- ledger

def ledger_path(output, arm):
    return Path(output) / 'ledger' / f'{arm}.json'


def bundle_hashes(run, name, epoch):
    return {part: sha256(path) for part, path in bundle_paths(Path(run) / name, epoch).items() if path.is_file()}


def completed_generations(manifest):
    rows = sorted((g for g in manifest.get('generations', []) if g.get('status') == 'completed'), key=lambda g: g['index'])
    indices = [g['index'] for g in rows]
    if indices != list(range(len(indices))) or len({g['index'] for g in manifest.get('generations', [])}) != len(manifest.get('generations', [])):
        raise CampaignFailure('Manifest has duplicate or non-contiguous completed generations')
    return rows


def update_ledger(output, arm, run, manifest):
    """Record committed member bundles of every newly completed generation."""
    path = ledger_path(output, arm)
    ledger = load_json(path) if path.is_file() else dict(arm=arm, run=str(run), generations={})
    changed = False
    for generation in completed_generations(manifest):
        key = str(generation['index'])
        if key not in ledger['generations']:
            ledger['generations'][key] = dict(epoch=generation['epoch'], recorded_at=utc_now(),
                                              members={n: bundle_hashes(run, n, generation['epoch']) for n in generation['workers']})
            changed = True
    if changed:
        write_json(path, ledger)
    return ledger


def verify_ledger(output, arm, run, manifest, *, full=False):
    """Fail if a committed checkpoint bundle changed (resume source or, with full, all)."""
    path = ledger_path(output, arm)
    if not path.is_file():
        return dict(verified=0)
    ledger = load_json(path)
    rows = completed_generations(manifest)
    recorded = sorted(int(k) for k in ledger['generations'])
    if recorded and recorded[-1] >= len(rows):
        raise CampaignFailure(f'{arm}: ledger records generations the manifest does not (manifest rolled back?)')
    targets = recorded if full else recorded[-1:]
    for index in targets:
        entry = ledger['generations'][str(index)]
        for name, expected in entry['members'].items():
            if bundle_hashes(run, name, entry['epoch']) != expected:
                raise CampaignFailure(f'{arm}: checkpoint corruption detected for {name} epoch {entry["epoch"]}')
    return dict(verified=len(targets))


# ----------------------------------------------------------------- GPUs

def acquire_group(output, cohort, allocation, stop, count, *, acquire=acquire_gpu, wait_seconds=5):
    """All-or-nothing lease of `count` idle GPUs of the campaign model; never hoards a partial set."""
    quick = dict(allocation, max_resource_wait_seconds=wait_seconds)
    gpus, leases = [], []
    try:
        for _ in range(count):
            gpu, lease = acquire(output, cohort, quick, stop)
            gpus.append(gpu)
            leases.append(lease)
    except RuntimeError:
        for lease in leases:
            lease.close()
        return None
    return gpus, leases


def idle_campaign_gpus(cohort, allocation):
    """Idle GPUs of the campaign model on this node right now (observation, not a lease)."""
    from experiments.stage2a.scheduling import free_gpu, probe_local
    return sum(1 for gpu in probe_local() if gpu['name'] == cohort.get('name') and gpu['total_mib'] == cohort.get('total_mib')
               and free_gpu(gpu, allocation))


def node_available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


# ----------------------------------------------------------------- arm state

def arm_run(protocol, arm):
    frozen = protocol['identity']['configs'][arm]
    return project_file(frozen['output_root']) / frozen['experiment']


def arm_state(protocol, arm):
    """absent | completed | resumable (never a silent fresh start into existing evidence)."""
    run = arm_run(protocol, arm)
    if not run.exists():
        return 'absent', None
    path = run / 'manifest.json'
    if not path.is_file():
        raise CampaignFailure(f'{arm}: run directory exists without a manifest; refusing to restart from epoch 17')
    manifest = load_json(path)
    if manifest.get('fingerprint') != contract_fingerprint(resolve(ARM_CONFIGS[arm])):
        raise CampaignFailure(f'{arm}: existing run belongs to a different configuration; resume refused')
    return ('completed' if manifest.get('status') == 'completed' else 'resumable'), manifest


def runner_command(config, gpus, *, resume=False, name=None):
    command = [sys.executable, str(RUNNER), '--config', str(config), '--gpus', ','.join(str(i) for i in range(gpus))]
    return command + (['--experiment-name', name] if name else []) + (['--resume'] if resume else [])


def start_arm(arm, protocol, gpus, leases, output, status, *, inherit_fds=(), popen=subprocess.Popen, resume=False):
    token = uuid.uuid4().hex
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in gpus), PYTHONUNBUFFERED='1',
               STAGE2B_SMOKE_TOKEN=token, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = runner_command(ARM_CONFIGS[arm], len(gpus), resume=resume)
    log_path = Path(output) / 'logs' / f'{arm}.console.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open('a')
    stream.write(f'{utc_now()} {" ".join(command)}\n')
    stream.flush()
    process = popen(command, cwd=PROJECT_DIR, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(*inherit_fds, *(lease.fileno() for lease in leases)))
    record = dict(arm=arm, pid=process.pid, gpus=[dict(index=g['index'], uuid=g['uuid'], name=g['name']) for g in gpus],
                  resume=resume, started_at=utc_now(), command=command, token=token)
    history = load_json(Path(output) / 'allocations.json') if (Path(output) / 'allocations.json').is_file() else []
    write_json(Path(output) / 'allocations.json', [*history, record])
    status.log('ARM', f'{arm}: {"resumed" if resume else "started"} on GPUs ' + ','.join(str(g['index']) for g in gpus))
    return dict(process=process, stream=stream, leases=leases, token=token, gpus=gpus, record=record)


def consumed_gpu_hours(protocol, output):
    """Worker GPU-hours already spent by both arms plus gate attempts (all launches)."""
    from datetime import datetime
    total = 0.0
    gate = protocol['identity']['gate']
    runs = [arm_run(protocol, arm) for arm in ARMS] + sorted(project_file(gate['output_root']).glob(gate['experiment'] + '_a*'))
    for run in runs:
        path = Path(run) / 'manifest.json'
        if not path.is_file():
            continue
        for generation in load_json(path).get('generations', []):
            for worker in generation.get('workers', {}).values():
                if worker.get('started_at'):
                    end = worker.get('finished_at') or utc_now()
                    total += (datetime.fromisoformat(end) - datetime.fromisoformat(worker['started_at'])).total_seconds()
    return total / 3600


# ----------------------------------------------------------------- phases

def prepare_data(protocol, policy, status, stop, **dataset_phase):
    """Re-hash train8m against its realized manifest; prepare/verify the sealed test split."""
    from experiments.stage3.testset import verify_converted_matches_root
    train_spec, _ = data.load_spec()
    status.phase('train8m_verification', 'PREP')
    manifest = data.verify_dataset(train_spec, rehash=True)
    frozen = load_json(PROJECT_DIR / protocol['identity']['train_manifest']['path'])
    if ([(f['path'], f['sha256'], f['rows']) for f in manifest['files']]
            != [(f['path'], f['sha256'], f['rows']) for f in frozen['files']]
            or manifest['weaver_audit']['fingerprint'] != protocol['design']['data']['train']['weaver_fingerprint']):
        raise CampaignFailure('train8m files disagree with the frozen realized manifest')
    test_spec = data.validate_spec(load_json(TEST_SPEC))
    test_manifest = run_dataset_phase(test_spec, policy, status, stop, spec_path=TEST_SPEC, prefix='test3000k', kind='PREP',
                                      isolation=verify_converted_matches_root, **dataset_phase)
    return manifest, test_manifest


def run_gate(protocol, protocol_sha, policy, cohort, output, status, stop, *, retry=False, inherit_fds=(),
             acquire=acquire_gpu, observer_factory=Observer, popen=subprocess.Popen, poll_seconds=10, verify=None):
    """Return a passed verdict (reused when already passed for this protocol) or raise."""
    from experiments.stage2b.report import log_timings
    from experiments.stage3 import gate as gate_module
    verify = verify or gate_module.verify_gate
    directory = Path(output) / 'gate'
    previous = gate_module.gate_record(output)
    if previous and previous.get('protocol_sha256') == protocol_sha:
        if previous['verdict'] == 'passed':
            status.log('GATE', f'reusing passed gate attempt {previous["attempt"]}')
            return previous
        if previous['verdict'] == gate_module.CRITERIA and not retry:
            raise GateBlocked('The full-epoch gate failed its criteria; campaign blocked. Inspect gate/verdict.json; '
                              'rerun with --retry-gate only after the cause is understood.')
    status.phase('gate', 'GATE')
    directory.mkdir(parents=True, exist_ok=True)
    number = 1 + len(list(directory.glob('attempt-*.json')))
    name = f'{protocol["identity"]["gate"]["experiment"]}_a{number:03d}'
    run = project_file(protocol['identity']['gate']['output_root']) / name
    lease_pair, observations = None, {}
    try:
        lease_pair = acquire_group(output, cohort, policy['allocation'], stop, 1, acquire=acquire,
                                   wait_seconds=policy['allocation']['max_resource_wait_seconds'])
        if not lease_pair:
            raise CampaignFailure('No idle campaign GPU for the gate')
        gpus, leases = lease_pair
        token = uuid.uuid4().hex
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0]['uuid'], PYTHONUNBUFFERED='1', STAGE2B_SMOKE_TOKEN=token,
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        observer = observer_factory([gpus[0]['uuid']], dict(policy, budgets=dict(policy['budgets'])))
        result = supervise(runner_command(GATE_CONFIG, 1, name=name), env, observer, status, run=run,
                           deadline=time.monotonic() + policy['budgets']['gate_max_wall_hours'] * 3600, stop=stop,
                           popen=popen, poll_seconds=poll_seconds, token=token, kind=f'gate_a{number:03d}',
                           inherit_fds=(*inherit_fds, *(lease.fileno() for lease in leases)))
        observations = dict(exit_code=result['exit_code'], leftover_processes=own_processes_with(token),
                            peak_gpu_memory_mib=observer.peak_gpu_memory_mib, peak_rss_bytes=observer.peak_rss_bytes,
                            foreign_gpu_events=observer.foreign_events, gpu=gpus[0])
        manifest = load_json(run / 'manifest.json') if (run / 'manifest.json').is_file() else {}
        worker = next(iter((manifest.get('generations') or [{}])[0].get('workers', {}).values()), {})
        if worker.get('log') and Path(worker['log']).is_file():
            observations['timing'] = log_timings(worker['log'], manifest['generations'][0]['epoch'])
        if result['exit_code'] != 0:
            raise CampaignFailure(f'Gate runner exited {result["exit_code"]}; see {run}/logs/pbt.log')
        verdict, checks, measurements = verify(run, protocol, policy, observations)
        category = verdict
    except InterruptedError:
        raise
    except Exception as error:
        verdict, category, checks, measurements = gate_module.OPERATIONAL, gate_module.OPERATIONAL, [], dict(error=str(error))
    finally:
        if lease_pair:
            for lease in lease_pair[1]:
                lease.close()
    record = dict(protocol_sha256=protocol_sha, attempt=number, run=str(run), verdict=verdict, category=category,
                  checks=checks, measurements=measurements, observations=observations, gpu_model=cohort.get('name'),
                  finished_at=utc_now())
    write_json(directory / f'attempt-{number:03d}.json', record)
    write_json(directory / 'verdict.json', record)
    status.log('GATE', f'gate attempt {number}: {verdict}')
    if verdict != 'passed':
        reasons = '; '.join(c['check'] for c in checks if not c['passed']) if checks else str(measurements.get('error'))
        raise GateBlocked(f'Full-epoch gate {verdict}: {reasons}')
    return record


def check_memory(policy, gate_record, workers):
    per_worker = max(policy['budgets']['worker_host_ram_bytes'], gate_record['measurements'].get('peak_rss_bytes') or 0)
    needed = int(workers * per_worker * policy['budgets']['node_ram_margin_factor'])
    available = node_available_ram()
    return dict(needed=needed, available=available, ok=available >= needed)


def run_arms(protocol, policy, cohort, output, status, stop, gate_record, *, inherit_fds=(), acquire=acquire_gpu,
             observer_factory=Observer, popen=subprocess.Popen, clock=time.monotonic, poll_seconds=None,
             idle=idle_campaign_gpus):
    """Run/resume both arms; each starts as soon as five campaign GPUs are free. Returns per-arm outcome."""
    allocation, budgets = policy['allocation'], policy['budgets']
    poll_seconds = budgets['monitor_poll_seconds'] if poll_seconds is None else poll_seconds
    status.phase('arms', 'ARM')
    outcome, running, waiting_since = {}, {}, {}
    for arm in ARMS:
        state, manifest = arm_state(protocol, arm)
        if state != 'absent':
            # A changed committed bundle stops everything before any process starts.
            verify_ledger(output, arm, arm_run(protocol, arm), manifest)
            update_ledger(output, arm, arm_run(protocol, arm), manifest)
        if state == 'completed':
            outcome[arm] = 'completed'
    observer = observer_factory([], policy)
    started, next_attempt = clock(), clock()
    try:
        while True:
            if clock() >= next_attempt:
                for arm in [a for a in ARMS if a not in outcome and a not in running]:
                    in_use = sum(len(r['gpus']) for r in running.values())
                    if in_use + allocation['gpus_per_arm'] > allocation['max_concurrent_gpus']:
                        break
                    memory = check_memory(policy, gate_record, in_use + allocation['gpus_per_arm'])
                    # Lease only when a full five-GPU set is idle: never hold a partial set.
                    enough = memory['ok'] and idle(cohort, allocation) >= allocation['gpus_per_arm']
                    group = acquire_group(output, cohort, allocation, stop, allocation['gpus_per_arm'],
                                          acquire=acquire) if enough else None
                    if not group:
                        waiting_since.setdefault(arm, clock())
                        status.log('ARM', f'{arm}: waiting for {allocation["gpus_per_arm"]} idle campaign GPUs'
                                   + ('' if memory['ok'] else f' and host RAM {memory}'))
                        if clock() - waiting_since[arm] > allocation['max_arm_wait_hours'] * 3600:
                            outcome[arm] = 'waiting_timeout'
                        continue
                    waiting_since.pop(arm, None)
                    state, _ = arm_state(protocol, arm)
                    running[arm] = start_arm(arm, protocol, *group, output, status, inherit_fds=inherit_fds,
                                             popen=popen, resume=state == 'resumable')
                    observer.uuids |= {g['uuid'] for g in group[0]}
                next_attempt = clock() + allocation['arm_wait_poll_seconds']
            if not running and all(a in outcome for a in ARMS):
                return outcome
            if stop.is_set():
                raise InterruptedError('Supervisor stopped')
            if running:
                observer.sample([r['process'].pid for r in running.values()])
            spent = consumed_gpu_hours(protocol, output)
            if spent > budgets['max_campaign_gpu_hours'] or clock() - started > budgets['max_campaign_wall_hours'] * 3600:
                raise CampaignFailure(f'Campaign budget exhausted ({spent:.1f} GPU-hours); arms stopped and resumable')
            for arm, record in list(running.items()):
                run = arm_run(protocol, arm)
                manifest = load_json(run / 'manifest.json') if (run / 'manifest.json').is_file() else {}
                if manifest:
                    update_ledger(output, arm, run, manifest)
                code = record['process'].poll()
                if code is None:
                    continue
                record['stream'].close()
                for lease in record['leases']:
                    lease.close()
                observer.uuids -= {g['uuid'] for g in record['gpus']}
                running.pop(arm)
                leftovers = own_processes_with(record['token'])
                outcome[arm] = ('completed' if code == 0 and manifest.get('status') == 'completed' and not leftovers
                                else 'failed')
                status.log('ARM', f'{arm}: runner exit {code}, manifest {manifest.get("status")}, outcome {outcome[arm]}'
                           + (f', leftover processes {leftovers}' if leftovers else ''))
                next_attempt = clock()  # freed GPUs may let the other arm start now
            status.update(arms={a: _progress(protocol, a, outcome.get(a) or ('running' if a in running else 'waiting'))
                                for a in ARMS}, gpu_hours_consumed=round(spent, 2),
                          peak_gpu_memory_mib=observer.peak_gpu_memory_mib, peak_rss_bytes=observer.peak_rss_bytes)
            stop.wait(poll_seconds)
    except BaseException:
        for arm, record in running.items():
            code, stragglers = stop_runner(record['process'], budgets['runner_stop_timeout_seconds'], record['token'])
            record['stream'].close()
            for lease in record['leases']:
                lease.close()
            status.log('ARM', f'{arm}: stopped (exit {code}); stragglers {stragglers}; resumable with the same command')
        raise


def _progress(protocol, arm, state):
    run = arm_run(protocol, arm)
    try:
        manifest = load_json(run / 'manifest.json')
    except (OSError, ValueError):
        return dict(state=state, completed_generations=0)
    rows = [g for g in manifest.get('generations', []) if g.get('status') == 'completed']
    current = next((g for g in manifest.get('generations', []) if g.get('status') != 'completed'), None)
    return dict(state=state, manifest_status=manifest.get('status'), completed_generations=len(rows),
                horizon=((manifest.get('config') or {}).get('shared') or {}).get('generations'),
                running_generation=current['index'] if current else None,
                workers={n: w.get('status') for n, w in (current or {}).get('workers', {}).items()})


def finalize(protocol, protocol_sha, policy, cohort, output, status, stop, gate_record, *, evaluate=None):
    """Validation analysis -> selection freeze -> sealed test (+ descriptive holdout) -> report."""
    from experiments.stage2a.execution import run_campaign
    from experiments.stage3 import analysis, testset
    evaluate = evaluate or run_campaign
    runs = {arm: str(arm_run(protocol, arm)) for arm in ARMS}
    for arm in ARMS:
        verify_ledger(output, arm, runs[arm], load_json(Path(runs[arm]) / 'manifest.json'), full=True)
    status.phase('validation_analysis', 'REPORT')
    validation = analysis.build(Path(output) / 'validation_analysis', protocol, runs, gate=gate_record)
    if validation['status'] != 'COMPLETE':
        raise CampaignFailure(f'Validation analysis is {validation["status"]}; the sealed test stays closed')
    status.phase('selection_freeze', 'REPORT')
    manifests = {arm: dict(load_json(Path(runs[arm]) / 'manifest.json'), run_dir=runs[arm]) for arm in ARMS}
    checkpoints = testset.selection_freeze(manifests)
    freeze_path = Path(output) / 'sealed_test' / 'selection_freeze.json'
    freeze = dict(protocol_sha256=protocol_sha, validation_analysis_sha256=sha256(Path(output) / 'validation_analysis' / 'analysis.json'),
                  checkpoints=checkpoints)
    if freeze_path.is_file():
        previous = load_json(freeze_path)
        if previous['checkpoints'] != checkpoints or previous['protocol_sha256'] != protocol_sha:
            raise CampaignFailure('Checkpoint selection changed after it was frozen')
    else:
        write_json(freeze_path, dict(freeze, frozen_at=utc_now()))
    arm_shared = resolve(ARM_CONFIGS['fixed_lr'])['shared']
    test_spec = data.validate_spec(load_json(TEST_SPEC))
    tiers = dict(sealed_test=(testset.evaluation_contract(data.output_dir(test_spec), test_spec['suffix'], arm_shared=arm_shared,
                                                          seed=arm_shared['seed'], name='sealed_test3000k'), checkpoints),
                 holdout_descriptive=(testset.load_holdout_contract(), [c for c in checkpoints if c['role'] == 'validation_global_best']))
    results = {}
    for tier, (contract, items) in tiers.items():
        status.phase(tier, 'REPORT')
        evaluation_protocol, evaluation_sha = testset.sealed_protocol(contract, items, policy['allocation'], policy['budgets'])
        write_json(Path(output) / tier / 'evaluation_protocol.json', evaluation_protocol)
        eval_cohort = dict(cohort, max_parallelism=min(policy['budgets']['test_eval_parallelism'], len(items)), gpus=[])
        campaign = evaluate(evaluation_protocol, evaluation_sha, Path(output) / tier, eval_cohort, stop=stop)
        if campaign['status'] not in ('evaluated', 'completed'):
            raise CampaignFailure(f'{tier} evaluation {campaign["status"]}; rerun the same command to retry')
        results[tier] = (evaluation_protocol, campaign['checkpoints'])
    protocol_t, records = results['sealed_test']
    metrics, comparisons = testset.paired(records, protocol_t, testset.preregistered_pairs(checkpoints))
    finals = {arm: [metrics[c['id']]['validation_total_reference_mistag_geomean_percent'] for c in checkpoints
                    if c['arm'] == arm and c['role'] == 'final_member'] for arm in ARMS}
    a, b = (sum(v) / len(v) for v in (finals['fixed_lr'], finals['windowed_pbt_v2']))
    test = dict(metrics=metrics, comparisons=comparisons, checkpoints=checkpoints,
                population_mean=dict(fixed_lr=a, windowed_pbt_v2=b, delta_pp=b - a, reduction_percent=-100 * (b - a) / a))
    protocol_h, records_h = results['holdout_descriptive']
    holdout_metrics, holdout_pairs = testset.paired(records_h, protocol_h, dict(selected_best=(
        'fixed_lr__selected_best', 'windowed_pbt_v2__selected_best')))
    status.phase('final_report', 'REPORT')
    operations = dict(allocations=load_json(Path(output) / 'allocations.json') if (Path(output) / 'allocations.json').is_file() else [],
                      gpu_hours_consumed=consumed_gpu_hours(protocol, output),
                      ledger={arm: verify_ledger(output, arm, runs[arm], manifests[arm], full=True) for arm in ARMS})
    return analysis.build(Path(output) / 'report', protocol, runs, gate=gate_record, test=test,
                          holdout=dict(metrics=holdout_metrics, comparisons=holdout_pairs), operations=operations)


def install_stop(stop):
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())


def summary_json(value):
    return json.dumps(value, indent=2, default=str)
