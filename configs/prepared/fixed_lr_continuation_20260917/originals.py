"""Prepared read-only reevaluation; run only on explicit user launch."""
import argparse, concurrent.futures, csv, fcntl, json, math, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from launch.experiment import available_gpus, configuration
from training.pbt.execution.weaver_command import make_tiered_evaluation_command
from training.runtime import atomic_json, read_metrics, sha256
PREP = Path(__file__).resolve().parent
OUT = ROOT/'runs/eval/original_pretrained_0_19_20260917'
OLD = ROOT/'runs/pbt/foundation_fixed_lr_50epochs_20260916'
SOURCES = json.loads((PREP/'original_checkpoint_hashes.json').read_text())
BASE = json.loads((OLD/'manifest.json').read_text())['initial_evaluation']['metrics']
METRIC = 'validation_total_reference_mistag_geomean_percent'
KEYS = ['validation_loss', METRIC] + sorted(k for k in BASE if '_mistag_eff_' in k)


def command(epoch):
    config = configuration(PREP/'lr_14e-6.yaml')
    shared = config['shared']
    folder = OUT/f'epoch-{epoch:02d}'
    cmd, log = make_tiered_evaluation_command(config, '0', Path(SOURCES[str(epoch)]['path']),
                   shared['validation_dataset'],shared['validation_suffix'],folder/'evaluation.log')
    # Retain event scores/labels at negligible cost for future post-hoc analysis.
    return cmd + ['--predict-output',str(folder/'predictions.parquet')], log


def valid(record):
    metrics = record.get('metrics') or {}
    audit = metrics.get('validation_data_audit', {})
    reference = BASE['validation_data_audit']
    epoch = str(record.get('epoch'))
    return (epoch in SOURCES and record.get('status') == 'completed' and record.get('returncode') == 0
            and record.get('checkpoint_sha256') == SOURCES[epoch]['sha256']
            and audit.get('dataset') == reference['dataset'] and audit.get('consumed') == reference['consumed']
            and audit.get('exhausted') and bool(audit.get('prediction_sha256')) and bool(audit.get('traversal'))
            and all(t.get('wraps') == 0 for t in audit.get('traversal', []))
            and all(isinstance(metrics.get(k),(int,float)) and math.isfinite(metrics[k]) for k in KEYS))


def summarize():
    rows = []
    for epoch in range(20):
        record = json.loads((OUT/f'epoch-{epoch:02d}/result.json').read_text())
        if not valid(record):
            raise RuntimeError(f'Incomplete/inconsistent epoch {epoch}')
        rows.append(dict(epoch=epoch, checkpoint_sha256=record['checkpoint_sha256'], **{k:record['metrics'][k] for k in KEYS}))
    best = min(rows,key=lambda r:r[METRIC]); selected = rows[17]
    # Competition rank preserves exact ties; retain all tied best epochs.
    rank = 1 + sum(r[METRIC] < selected[METRIC] for r in rows)
    delta = selected[METRIC]-best[METRIC]
    epoch17 = json.loads((OUT/'epoch-17/result.json').read_text())['metrics']
    if epoch17['validation_data_audit']['prediction_sha256'] != BASE['validation_data_audit']['prediction_sha256']:
        raise RuntimeError('Epoch-17 baseline prediction parity failed')
    if any(epoch17[k] != BASE[k] for k in KEYS):
        raise RuntimeError('Epoch-17 baseline metric parity failed')
    report=dict(passed=True, results=rows, best_epoch=best['epoch'], best_mistag_percent=best[METRIC],
                tied_best_epochs=[r['epoch'] for r in rows if r[METRIC]==best[METRIC]], epoch17_rank=rank,
                epoch17_mistag_percent=selected[METRIC], absolute_gap_percentage_points=delta,
                epoch17_relative_excess_vs_best_percent=100*delta/best[METRIC],
                best_relative_improvement_vs_epoch17_percent=100*delta/selected[METRIC],
                interpretation='Any improvement changes pretrained checkpoint selection retrospectively. '
                'The two continuations still share epoch-17 ancestry and remain a controlled comparison; '
                'this reevaluation alone cannot establish a better eventual training trajectory.')
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'summary.json',report)
    with (OUT/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))
    return report


def run(gpus):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuids=available_gpus(gpus)
        def worker(slot):
            for epoch in range(slot,20,len(uuids)):
                folder=OUT/f'epoch-{epoch:02d}';folder.mkdir(exist_ok=True)
                result_path=folder/'result.json'
                if result_path.exists() and valid(json.loads(result_path.read_text())):
                    continue
                source=SOURCES[str(epoch)]
                if sha256(Path(source['path'])) != source['sha256']:
                    raise RuntimeError(f'Original checkpoint changed: {epoch}')
                cmd,log=command(epoch)
                record=dict(epoch=epoch, checkpoint=source['path'],checkpoint_sha256=source['sha256'],
                            command=cmd,status='running',gpu_uuid=uuids[slot])
                atomic_json(result_path,record)
                started=time.monotonic()
                with (folder/'console.log').open('w') as stream:
                    result=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[slot]),
                                          stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                record.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,
                              elapsed_seconds=time.monotonic()-started,metrics=read_metrics(log))
                if not valid(record):record['status']='failed'
                atomic_json(result_path,record)
                if not valid(record):raise RuntimeError(f'Original checkpoint {epoch} evaluation failed')
                print(f'epoch {epoch}: completed in {record["elapsed_seconds"]:.1f}s',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(uuids)) as pool:
            list(pool.map(worker,range(len(uuids))))
        summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','summarize','commands'])
    parser.add_argument('--gpus',default='0,2')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='run':run([int(v) for v in args.gpus.split(',')])
    elif args.action=='summarize':summarize()
    else:print(json.dumps({str(e):command(e)[0] for e in range(20)},indent=2))
