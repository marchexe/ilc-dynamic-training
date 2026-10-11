"""Inference-only execution and restart validation on one persistent node."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import awkward as ak
import numpy as np

from experiments.stage2a.common import PRIMARY, environment, lock_file, log, sha256, utc_now, verify_file, write_json
from experiments.stage2a.scheduling import free_gpu, probe_local
from training.pbt.execution.weaver_command import make_tiered_evaluation_command
from training.runtime import WORKING_POINT_DEFINITION, git_metadata, mistag_percent_key, read_metrics
from validation.results import finite_metric_ok, require_checkpoint_result


def inference_command(protocol, checkpoint, log_path):
    config = dict(shared=protocol['evaluation'])
    shared = protocol['evaluation']
    command,_ = make_tiered_evaluation_command(config,'0',Path(checkpoint['checkpoint']['path']),
        Path(shared['dataset']),shared['validation_suffix'],Path(log_path))
    command = [sys.executable,*command,'--seed',str(shared['seed']),'--save-prediction-row-ids']
    if command[command.index('--run-mode')+1] != 'test' or '--data-train' in command or '--load-epoch' in command:
        raise ValueError('Non-inference command forbidden')
    return command


def prediction_arrays(record, protocol):
    predictions = ak.from_parquet(record['prediction_path'])
    if any(key not in predictions.fields for key in ('scores','_label_','__weaver_row_id__')):
        raise ValueError('Prediction artifact is missing scores, class labels or actual row IDs')
    scores = np.asarray(ak.to_numpy(predictions['scores']),dtype='<f4')
    labels = np.asarray(ak.to_numpy(predictions['_label_']))
    n = protocol['dataset']['rows']
    if scores.shape != (n,3) or labels.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError('Incomplete/invalid three-class predictions')
    if (set(np.unique(labels)) != {0,1,2} or np.any(scores<0) or np.any(scores>1)
            or not np.allclose(scores.sum(axis=1),1,atol=2e-5)):
        raise ValueError('Invalid three-class score/label contract')
    audit = record['metrics']['validation_data_audit']
    raw_ids = np.asarray(ak.to_numpy(predictions['__weaver_row_id__']))
    if raw_ids.dtype.kind not in 'iu':
        raise ValueError('Prediction row IDs must be integers')
    ids = raw_ids.astype('<i8')
    canonical = np.arange(n,dtype='<i8')
    if ids.shape != (n,) or not np.array_equal(np.sort(ids),canonical):
        raise ValueError('Prediction row IDs are missing, repeated or outside the frozen dataset')
    sequence = hashlib.sha256(ids.tobytes()).hexdigest()
    id_set = hashlib.sha256(canonical.tobytes()).hexdigest()
    if (audit['dataset']['fingerprint'] != protocol['dataset']['audit_fingerprint']
            or audit['dataset']['total_rows'] != n or audit['consumed']['count'] != n
            or audit['consumed']['sequence_sha256'] != sequence
            or audit['consumed']['id_set_sha256'] != id_set
            or audit['optimizer_steps'] != 0):
        raise ValueError('Predictions are not a complete, ordered, optimizer-free holdout pass')
    expected_labels = np.concatenate([np.full(r['rows'],i,dtype=np.int64)
                                      for i,r in enumerate(protocol['dataset']['holdout_files'])])
    if not np.array_equal(labels,expected_labels[ids]):
        raise ValueError('Prediction class order disagrees with frozen sorted bb/cc/dd files')
    return ids, labels, scores


def validate_result(record, checkpoint, protocol, protocol_sha):
    if record.get('protocol_sha256') != protocol_sha or record.get('checkpoint_id') != checkpoint['id']:
        raise ValueError('Result belongs to a different frozen campaign/checkpoint')
    if not record.get('log') or not record.get('prediction_path'):
        raise ValueError('Missing required inference artifacts')
    require_checkpoint_result(record,Path(checkpoint['checkpoint']['path']),checkpoint['checkpoint']['sha256'],
                              checkpoint['id'],PRIMARY)
    for pair,eff in WORKING_POINT_DEFINITION:
        if not finite_metric_ok(record['metrics'],mistag_percent_key(pair,eff)):
            raise ValueError(f'Missing/non-finite working point {pair}/{eff}')
    prediction_arrays(record,protocol)
    for artifact in record.get('artifacts',[]):
        verify_file(artifact)
    return record


def verified_completed(path, checkpoint, protocol, protocol_sha):
    try:
        record = json.loads(Path(path).read_text())
        if record.get('status') != 'completed' or not record.get('artifacts'):
            return None
        return validate_result(record,checkpoint,protocol,protocol_sha)
    except (OSError,ValueError,KeyError,RuntimeError,TypeError,IndexError):
        return None


def terminate_own(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=10)


def acquire_gpu(output, cohort, policy, stop, *, probe=probe_local):
    deadline = time.monotonic()+policy['max_resource_wait_seconds']
    while not stop.is_set():
        available = {g['uuid']:g for g in probe()}
        # The observation is not a reservation. Use newly idle GPUs of the
        # same homogeneous node/model, even if the initially listed GPU is busy.
        candidates = (sorted((g for g in available.values() if g['host']==cohort['host']
                             and g['name']==cohort['name'] and g['total_mib']==cohort['total_mib']),key=lambda g:g['index'])
                      if cohort.get('name') and cohort['name']!='analysis_only' else cohort['gpus'])
        for candidate in candidates:
            current = available.get(candidate['uuid'])
            if not current or not free_gpu(current,policy):
                continue
            try:
                lease = lock_file(Path(output)/'.gpu-locks'/f'{cohort["host"]}_{current["uuid"]}.lock')
            except BlockingIOError:
                continue
            # Re-check after acquiring the cooperative lease, immediately before launch.
            latest = {g['uuid']:g for g in probe()}.get(current['uuid'])
            if latest and free_gpu(latest,policy):
                return latest,lease
            lease.close()
        if time.monotonic() >= deadline:
            raise RuntimeError('Authorized cohort became busy; bounded resource wait exhausted')
        stop.wait(min(5,max(0,deadline-time.monotonic())))
    raise RuntimeError('Campaign interrupted before GPU acquisition')


def evaluate(checkpoint, protocol, protocol_sha, output, cohort, stop, *, probe=probe_local):
    with lock_file(Path(output)/'evaluations'/checkpoint['id']/'checkpoint.lock') as checkpoint_lock:
        return _evaluate_locked(checkpoint,protocol,protocol_sha,output,cohort,stop,checkpoint_lock,probe=probe)


def _evaluate_locked(checkpoint, protocol, protocol_sha, output, cohort, stop, checkpoint_lock, *, probe=probe_local):
    directory = Path(output)/'evaluations'/checkpoint['id']
    directory.mkdir(parents=True,exist_ok=True)
    result_path = directory/'result.json'
    existing = verified_completed(result_path,checkpoint,protocol,protocol_sha)
    if existing:
        return existing
    # Preserve evidence from earlier failed/interrupted attempts.
    attempt = 1+len(list(directory.glob('attempt-*')))
    attempt_dir = directory/f'attempt-{attempt:03d}'
    attempt_dir.mkdir()
    log_path = attempt_dir/'evaluation.log'
    record = dict(checkpoint_id=checkpoint['id'],protocol_sha256=protocol_sha,status='running',
                  checkpoint=checkpoint['checkpoint']['path'],log=str(log_path),
                  prediction_path=str(log_path.with_suffix('.predictions.parquet')),started_at=utc_now(),attempt=attempt)
    record.update(environment=environment(),execution_git=git_metadata())
    write_json(result_path,record)
    lease = None; process = None
    try:
        verify_file(checkpoint['checkpoint'])
        gpu,lease = acquire_gpu(output,cohort,protocol['allocation'],stop,probe=probe)
        record['gpu'] = gpu
        command = inference_command(protocol,checkpoint,log_path)
        record['command'] = command
        write_json(result_path,record)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        started = time.monotonic()
        with (attempt_dir/'console.log').open('w') as stream:
            process = subprocess.Popen(command,cwd=Path(protocol['evaluation']['network_config']).parents[1],
                stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,
                pass_fds=(lease.fileno(),checkpoint_lock.fileno()))
            deadline = started+protocol['allocation']['evaluation_timeout_seconds']
            while process.poll() is None:
                if stop.wait(5) or time.monotonic()>deadline:
                    raise RuntimeError('Inference interrupted or timed out')
                current = {g['uuid']:g for g in probe()}.get(gpu['uuid'])
                if process.poll() is None and (not current or any(pid != process.pid for pid in current['pids'])):
                    raise RuntimeError('Foreign GPU process detected; stopping only this campaign worker')
            record.update(returncode=process.returncode,elapsed_seconds=time.monotonic()-started,
                          metrics=read_metrics(log_path),status='completed' if process.returncode==0 else 'failed')
        validate_result(record,checkpoint,protocol,protocol_sha)
        record['artifacts'] = [dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha256(p))
                               for p in (log_path,Path(record['prediction_path']),attempt_dir/'console.log')]
    except Exception as error:
        if process is not None:
            terminate_own(process)
        # Failed records deliberately omit possibly non-finite parsed metrics.
        record.update(status='failed',error=str(error),metrics=None)
    finally:
        if lease:
            lease.close()
        record['finished_at'] = utc_now()
        write_json(result_path,record)
    return record


def run_campaign(protocol, protocol_sha, output, cohort, *, evaluator=evaluate, stop=None, finalize=None):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    stop = stop or threading.Event()
    with lock_file(output/'.campaign.lock'):
        campaign = dict(protocol_sha256=protocol_sha,status='running',started_at=utc_now(),cohort=cohort,checkpoints={})
        write_json(output/'status.json',campaign)
        log(output/'campaign.log','campaign start; no holdout-based checkpoint selection')
        with ThreadPoolExecutor(max_workers=cohort['max_parallelism']) as pool:
            futures = {pool.submit(evaluator,c,protocol,protocol_sha,output,cohort,stop):c for c in protocol['checkpoints']}
            for future in as_completed(futures):
                checkpoint = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = dict(checkpoint_id=checkpoint['id'],status='failed',error=str(error))
                campaign['checkpoints'][checkpoint['id']] = record
                log(output/'campaign.log',f'{checkpoint["id"]}: {record["status"]}')
                write_json(output/'status.json',campaign)
        complete = all(r.get('status')=='completed' for r in campaign['checkpoints'].values()) and len(campaign['checkpoints'])==len(protocol['checkpoints'])
        campaign['status'] = 'evaluated' if complete else 'interrupted' if stop.is_set() else 'failed'
        if complete and finalize:
            try:
                campaign['report'] = finalize(campaign['checkpoints'])
                campaign['status'] = 'completed'
            except Exception as error:
                campaign.update(status='failed',analysis_error=str(error))
        campaign['finished_at'] = utc_now()
        write_json(output/'status.json',campaign)
        return campaign
