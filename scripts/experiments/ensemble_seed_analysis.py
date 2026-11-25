#!/usr/bin/env python3
"""The two questions of configs/prepared/ensemble_seed22345_v1/spec.json, from finished checkpoints.

    .venv/bin/python scripts/experiments/ensemble_seed_analysis.py --gpus 6,7

(A) Does the gain of a two-model ensemble come from the different start LRs or only from training two models
separately? Selection set (sel1m_tail) only. (B) Does the ensemble of the two seed-22345 standard ParT models beat
the better of them on the holdout set? Each seed-22345 model is evaluated once on the holdout, with the settings
of the frozen holdout protocol of holdout_sel1m_train8m_lr3e-3_v1, after both checkpoints are fixed.

Predictions are written with Weaver row IDs; every metric recomputed from predictions must equal the metric of
the Weaver log. Results go to runs/eval/ensemble_seed22345_v1/ (result.json). The rules are those of the spec;
this script does not choose anything that the spec leaves open.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.holdout_block_bootstrap import blocks  # noqa: E402
from experiments.stage2a.analysis import paired_bootstrap, score_metrics  # noqa: E402
from experiments.stage2a.common import PRIMARY, utc_now  # noqa: E402
from training.runtime import PROJECT_DIR, read_metrics, weaver_executable  # noqa: E402

SPEC = PROJECT_DIR / 'configs/prepared/ensemble_seed22345_v1/spec.json'
PROTOCOL = PROJECT_DIR / 'configs/prepared/holdout_sel1m_train8m_lr3e-3_v1/protocol.json'
OUT = PROJECT_DIR / 'runs/eval/ensemble_seed22345_v1'
RUNS = PROJECT_DIR / 'runs/pbt'
SEL1M = PROJECT_DIR / 'datasets/20250711_ilc_nnqq_sgv_10m_3cat_sel1m_parquet'
FIXED = {  # the seed-12345 models of the spec (as in configs/prepared/ensemble_8m_dev_v1)
    's1_3e-3': ('sel1m_train8m_standard_part_3e-3', 'flat_decay_3000e-6', 62),
    's1_1e-3': ('sel1m_train8m_standard_part', 'flat_decay_1000e-6', 67),
}
SEED_RUN = 'sel1m_train8m_standard_part_seed22345'
SEED_MEMBERS = {'s2_3e-3': 'flat_decay_3000e-6', 's2_1e-3': 'flat_decay_1000e-6'}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def best_epoch(run, member):
    """The epoch with the member's best sel1m_tail value (spec: models)."""
    manifest = json.loads((RUNS / run / 'manifest.json').read_text())
    values = [(g['workers'][member]['metrics'][PRIMARY], g['epoch']) for g in manifest['generations']
              if (g['workers'].get(member) or {}).get('metrics')]
    if len(values) < manifest['config']['shared']['generations']:
        raise SystemExit(f'{run}/{member}: {len(values)} of {manifest["config"]["shared"]["generations"]} epochs')
    return min(values)[1], min(values)[0]


def models():
    out = {k: dict(run=r, member=m, epoch=e) for k, (r, m, e) in FIXED.items()}
    for key, member in SEED_MEMBERS.items():
        epoch, value = best_epoch(SEED_RUN, member)
        out[key] = dict(run=SEED_RUN, member=member, epoch=epoch, training_time_sel1m=value)
    for m in out.values():
        m['checkpoint'] = str(RUNS / m['run'] / m['member'] / f'net_epoch-{m["epoch"]}_state.pt')
        m['sha256'] = sha256(m['checkpoint'])
    return out


def command(checkpoint, files, directory, gpu, protocol):
    e = protocol['evaluation']
    cmd = [str(weaver_executable()), '--run-mode', 'test', '--data-test', *files, '--data-config', e['data_config'],
           '--network-config', e['network_config'], '--model-prefix', checkpoint,
           '--log-file', str(directory / 'evaluation.log'), '--batch-size', str(e['batch_size']),
           '--num-workers', str(e['num_workers']), '--fetch-step', str(e['fetch_step']), '--gpus', gpu,
           '--predict-gpus', gpu, '--predict-output', str(directory / 'predictions.parquet'),
           '--prefetch-factor', str(e['prefetch_factor']), '--seed', str(e['seed']), '--data-audit',
           '--save-prediction-row-ids']
    if e['deterministic']:
        cmd.append('--deterministic')
    if e['use_amp']:
        cmd += ['--use-amp', '--amp-dtype', e['amp_dtype']]
    return cmd


def evaluate_all(jobs, gpus, protocol):
    """jobs: [(directory, checkpoint, files)]; skips directories with a finished evaluation."""
    pending, lock, errors = [j for j in jobs if not (j[0] / 'done.json').exists()], threading.Lock(), []

    def worker(gpu):
        while True:
            with lock:
                if not pending:
                    return
                directory, checkpoint, files = pending.pop(0)
            directory.mkdir(parents=True, exist_ok=True)
            cmd = command(checkpoint, files, directory, gpu, protocol)
            with open(directory / 'console.log', 'w') as console:
                code = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=console, stderr=subprocess.STDOUT).returncode
            metrics = read_metrics(directory / 'evaluation.log')
            if code or not metrics or metrics.get(PRIMARY) is None:
                errors.append(f'{directory}: exit {code}')
                continue
            (directory / 'done.json').write_text(json.dumps(dict(command=cmd, metric=metrics[PRIMARY],
                                                                 finished_at=utc_now()), indent=1))
            print(f'{directory.relative_to(OUT)} {metrics[PRIMARY]:.5f}', flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise SystemExit('evaluation failed: ' + '; '.join(errors))


def load(directory):
    table = pq.read_table(directory / 'predictions.parquet')
    ids = table['__weaver_row_id__'].to_numpy()
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(ids))):
        raise ValueError(f'{directory}: row IDs are not 0..N-1')
    labels = table['_label_'].to_numpy()[order]
    scores = np.stack(table['scores'].to_numpy(zero_copy_only=False))[order]  # float32, as Weaver scores it
    recorded = json.loads((directory / 'done.json').read_text())['metric']
    if not np.isclose(score_metrics(labels, scores)[PRIMARY], recorded, rtol=1e-9):
        raise ValueError(f'{directory}: recomputed metric differs from the Weaver log')
    return labels, scores


def gain(labels, a, b):
    """Ensemble against the better of the two models, in percent (negative is better)."""
    va, vb = score_metrics(labels, a)[PRIMARY], score_metrics(labels, b)[PRIMARY]
    ens = score_metrics(labels, (a + b) / 2)[PRIMARY]
    return dict(left=va, right=vb, ensemble=ens, gain_percent=100 * (ens / min(va, vb) - 1),
                disagree_percent=100 * float(np.mean(a.argmax(1) != b.argmax(1))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--skip-holdout', action='store_true', help='answer question A only')
    args = parser.parse_args(argv)
    gpus = [g for g in args.gpus.split(',') if g]
    if not 1 <= len(gpus) <= 3:
        raise SystemExit('use one to three GPUs')
    spec, protocol = json.loads(SPEC.read_text()), json.loads(PROTOCOL.read_text())
    chosen = models()

    sel_files = [str(p) for flavor in ('bb', 'cc', 'dd') for p in sorted(SEL1M.glob(f'*_{flavor}_sel1m_tail.parquet'))]
    evaluate_all([(OUT / 'sel1m' / k, m['checkpoint'], sel_files) for k, m in chosen.items()], gpus, protocol)
    sel = {k: load(OUT / 'sel1m' / k) for k in chosen}
    labels = sel['s1_3e-3'][0]
    if any(not np.array_equal(v[0], labels) for v in sel.values()):
        raise ValueError('the sel1m_tail predictions are not on the same rows')
    for key in SEED_MEMBERS:  # the evaluation must reproduce the training-time value
        if not np.isclose(score_metrics(labels, sel[key][1])[PRIMARY], chosen[key]['training_time_sel1m'], rtol=1e-6):
            raise ValueError(f'{key}: sel1m_tail value differs from the training-time value')

    qa = spec['question_A']
    same = [dict(pair=p, **gain(labels, sel[p[0]][1], sel[p[1]][1])) for p in qa['same_lr_pairs']]
    diff = [dict(pair=p, **gain(labels, sel[p[0]][1], sel[p[1]][1])) for p in qa['different_lr_pairs']]
    mean_same = float(np.mean([r['gain_percent'] for r in same]))
    mean_diff = float(np.mean([r['gain_percent'] for r in diff]))
    delta = mean_diff - mean_same  # negative: different LRs give the larger gain
    verdict = ('start LR adds to the gain' if delta < -1 else
               'different LRs hurt' if delta > 1 else 'separate training alone explains the gain')
    result = dict(schema_version=1, spec=str(SPEC.relative_to(PROJECT_DIR)), spec_sha256=sha256(SPEC),
                  finished_at=utc_now(), models=chosen,
                  question_A=dict(same_lr=same, different_lr=diff, mean_gain_same_lr_percent=mean_same,
                                  mean_gain_different_lr_percent=mean_diff,
                                  difference_percentage_points=delta, verdict=verdict))

    if not args.skip_holdout:
        holdout = protocol['dataset']['holdout_files']
        for f in holdout:
            if sha256(f['path']) != f['sha256']:
                raise ValueError(f'{f["path"]}: holdout file changed')
        files = [f['path'] for f in holdout]
        evaluate_all([(OUT / 'holdout' / k, chosen[k]['checkpoint'], files) for k in SEED_MEMBERS], gpus, protocol)
        h = {k: load(OUT / 'holdout' / k) for k in SEED_MEMBERS}
        hl = h['s2_3e-3'][0]
        if not np.array_equal(hl, h['s2_1e-3'][0]):
            raise ValueError('the holdout predictions are not on the same rows')
        left = min(SEED_MEMBERS, key=lambda k: chosen[k]['training_time_sel1m'])  # better on sel1m_tail
        a, b = h[left][1], (h['s2_3e-3'][1] + h['s2_1e-3'][1]) / 2
        iv = spec['question_B']['interval']
        boot = paired_bootstrap(hl, a, b, physical_event_ids=blocks(hl, iv['block_rows']),
                                replicates=iv['replicates'], seed=iv['seed'])
        base, ens = score_metrics(hl, a)[PRIMARY], score_metrics(hl, b)[PRIMARY]
        low, high = boot['intervals'][PRIMARY]
        interval = [100 * low / base, 100 * high / base]
        result['question_B'] = dict(
            left=left, holdout={k: score_metrics(hl, v[1])[PRIMARY] for k, v in h.items()}, ensemble=ens,
            change_percent=100 * (ens / base - 1), interval_percent=interval,
            repeats=bool(ens < base and interval[1] < 0))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}, indent=1))


if __name__ == '__main__':
    main()
