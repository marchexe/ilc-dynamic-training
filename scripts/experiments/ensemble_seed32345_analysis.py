#!/usr/bin/env python3
"""The questions of configs/prepared/ensemble_seed32345_v1/spec.json (the third seed and the three-seed mean).

    .venv/bin/python scripts/experiments/ensemble_seed32345_analysis.py --gpus 4,5 [--workers 9]

Uses the evaluation, loading and pair code of ensemble_seed_analysis.py. Seed 12345 holdout predictions are those
of configs/prepared/ensemble_8m_dev_v1, seed 22345 those of ensemble_seed22345_v1 (which must be complete); the
two seed-32345 models are evaluated here, on sel1m_tail and once on the holdout set. The pooled interval resamples
one set of holdout blocks per replicate for all three seeds (as --pooled of holdout_block_bootstrap.py).
Results go to runs/eval/ensemble_seed32345_v1/result.json.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import ensemble_seed_analysis as base  # noqa: E402
from experiments.holdout_block_bootstrap import blocks  # noqa: E402
from experiments.stage2a.analysis import paired_bootstrap, score_metrics  # noqa: E402
from experiments.stage2a.common import PRIMARY, utc_now  # noqa: E402
from training.runtime import PROJECT_DIR  # noqa: E402

SPEC = PROJECT_DIR / 'configs/prepared/ensemble_seed32345_v1/spec.json'
DEV_SPEC = PROJECT_DIR / 'configs/prepared/ensemble_8m_dev_v1/spec.json'
PREV = PROJECT_DIR / 'runs/eval/ensemble_seed22345_v1'
OUT = PROJECT_DIR / 'runs/eval/ensemble_seed32345_v1'
SEED_RUN = 'sel1m_train8m_standard_part_seed32345'
SEED_MEMBERS = {'s3_3e-3': 'flat_decay_3000e-6', 's3_1e-3': 'flat_decay_1000e-6'}


def load_campaign_predictions(directory):
    """(labels, scores) of a holdout campaign evaluation (attempt-*/evaluation.predictions.parquet)."""
    import pyarrow.parquet as pq
    d = PROJECT_DIR / directory
    result = json.loads((d / 'result.json').read_text())
    table = pq.read_table(sorted(d.glob('attempt-*/evaluation.predictions.parquet'))[-1])
    ids = table['__weaver_row_id__'].to_numpy()
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(ids))):
        raise ValueError(f'{directory}: row IDs are not 0..N-1')
    labels = table['_label_'].to_numpy()[order]
    scores = np.stack(table['scores'].to_numpy(zero_copy_only=False))[order]
    if not np.isclose(score_metrics(labels, scores)[PRIMARY], result['metrics'][PRIMARY], rtol=1e-10):
        raise ValueError(f'{directory}: recomputed metric differs')
    return labels, scores


def _pooled_chunk(job):
    pairs_path, size, replicates, seed = job
    data = np.load(pairs_path)
    labels, block = data['labels'], blocks(data['labels'], size)
    n_blocks = block[-1] + 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
        rows = np.repeat(np.arange(len(labels)), counts[block])
        changes = []
        for k in range(3):
            a, b = data[f'left{k}'][rows], data[f'ens{k}'][rows]
            changes.append(100 * (score_metrics(labels[rows], b)[PRIMARY] / score_metrics(labels[rows], a)[PRIMARY] - 1))
        out.append(changes)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--workers', type=int, default=9)
    args = parser.parse_args(argv)
    gpus = [g for g in args.gpus.split(',') if g]
    spec, protocol = json.loads(SPEC.read_text()), json.loads(base.PROTOCOL.read_text())
    previous = json.loads((PREV / 'result.json').read_text())

    chosen = dict(previous['models'])
    for key, member in SEED_MEMBERS.items():
        epoch, value = base.best_epoch(SEED_RUN, member)
        checkpoint = base.RUNS / SEED_RUN / member / f'net_epoch-{epoch}_state.pt'
        chosen[key] = dict(run=SEED_RUN, member=member, epoch=epoch, training_time_sel1m=value,
                           checkpoint=str(checkpoint), sha256=base.sha256(checkpoint))

    sel_files = [str(p) for flavor in ('bb', 'cc', 'dd')
                 for p in sorted(base.SEL1M.glob(f'*_{flavor}_sel1m_tail.parquet'))]
    base.OUT = OUT
    base.evaluate_all([(OUT / 'sel1m' / k, chosen[k]['checkpoint'], sel_files) for k in SEED_MEMBERS], gpus, protocol)
    sel = {k: base.load(PREV / 'sel1m' / k) for k in ('s1_3e-3', 's1_1e-3', 's2_3e-3', 's2_1e-3')}
    sel.update({k: base.load(OUT / 'sel1m' / k) for k in SEED_MEMBERS})
    labels = sel['s1_3e-3'][0]
    if any(not np.array_equal(v[0], labels) for v in sel.values()):
        raise ValueError('the sel1m_tail predictions are not on the same rows')
    for key in SEED_MEMBERS:
        if not np.isclose(score_metrics(labels, sel[key][1])[PRIMARY], chosen[key]['training_time_sel1m'], rtol=1e-6):
            raise ValueError(f'{key}: sel1m_tail value differs from the training-time value')

    qa = spec['question_A']
    same = [dict(pair=p, **base.gain(labels, sel[p[0]][1], sel[p[1]][1])) for p in qa['same_lr_pairs']]
    diff = [dict(pair=p, **base.gain(labels, sel[p[0]][1], sel[p[1]][1])) for p in qa['different_lr_pairs']]
    mean_same = float(np.mean([r['gain_percent'] for r in same]))
    mean_diff = float(np.mean([r['gain_percent'] for r in diff]))
    delta = mean_diff - mean_same
    verdict = ('start LR adds to the gain' if delta < -1 else
               'different LRs hurt' if delta > 1 else 'separate training alone explains the gain')

    holdout = protocol['dataset']['holdout_files']
    for f in holdout:
        if base.sha256(f['path']) != f['sha256']:
            raise ValueError(f'{f["path"]}: holdout file changed')
    base.evaluate_all([(OUT / 'holdout' / k, chosen[k]['checkpoint'], [f['path'] for f in holdout])
                       for k in SEED_MEMBERS], gpus, protocol)
    dev = json.loads(DEV_SPEC.read_text())['ensemble']['members']
    s1 = [load_campaign_predictions(m['predictions']) for m in dev]  # [3e-3, 1e-3]
    h = {'s1_3e-3': s1[0], 's1_1e-3': s1[1],
         's2_3e-3': base.load(PREV / 'holdout' / 's2_3e-3'), 's2_1e-3': base.load(PREV / 'holdout' / 's2_1e-3'),
         's3_3e-3': base.load(OUT / 'holdout' / 's3_3e-3'), 's3_1e-3': base.load(OUT / 'holdout' / 's3_1e-3')}
    hl = h['s1_3e-3'][0]
    if any(not np.array_equal(v[0], hl) for v in h.values()):
        raise ValueError('the holdout predictions are not on the same rows')

    per_seed, arrays = {}, {'labels': hl}
    for k, tag in enumerate(('s1', 's2', 's3')):
        left = min((f'{tag}_3e-3', f'{tag}_1e-3'), key=lambda m: chosen[m]['training_time_sel1m'])
        a = h[left][1]
        b = ((h[f'{tag}_3e-3'][1] + h[f'{tag}_1e-3'][1]) / 2).astype(a.dtype)
        base_value, ens = score_metrics(hl, a)[PRIMARY], score_metrics(hl, b)[PRIMARY]
        per_seed[tag] = dict(left=left, left_holdout=base_value, ensemble=ens, change_percent=100 * (ens / base_value - 1))
        arrays[f'left{k}'], arrays[f'ens{k}'] = a, b
    iv = spec['question_B']['interval']
    boot = paired_bootstrap(hl, arrays['left2'], arrays['ens2'], physical_event_ids=blocks(hl, iv['block_rows']),
                            replicates=iv['replicates'], seed=iv['seed'])
    low, high = boot['intervals'][PRIMARY]
    per_seed['s3']['interval_percent'] = [100 * low / per_seed['s3']['left_holdout'],
                                          100 * high / per_seed['s3']['left_holdout']]

    OUT.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT / 'pooled_inputs.npz'
    np.savez(pairs_path, **arrays)
    share = [iv['replicates'] // args.workers + (i < iv['replicates'] % args.workers) for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        draws = [d for chunk in pool.map(_pooled_chunk, [(str(pairs_path), iv['block_rows'], n, iv['seed'] + i)
                                                         for i, n in enumerate(share)]) for d in chunk]
    pairs_path.unlink()
    means = np.mean(draws, axis=1)
    mean_change = float(np.mean([per_seed[t]['change_percent'] for t in ('s1', 's2', 's3')]))
    pooled = [float(x) for x in np.quantile(means, [0.025, 0.975])]
    s3 = per_seed['s3']
    result = dict(schema_version=1, spec=str(SPEC.relative_to(PROJECT_DIR)), spec_sha256=base.sha256(SPEC),
                  finished_at=utc_now(), models=chosen,
                  question_A=dict(same_lr=same, different_lr=diff, mean_gain_same_lr_percent=mean_same,
                                  mean_gain_different_lr_percent=mean_diff, difference_percentage_points=delta,
                                  verdict=verdict, seed22345_verdict=previous['question_A']['verdict']),
                  question_B=dict(per_seed=per_seed,
                                  seed32345_repeats=bool(s3['change_percent'] < 0 and s3['interval_percent'][1] < 0),
                                  mean_change_percent=mean_change, pooled_interval_percent=pooled,
                                  pooled_replicates=len(draws),
                                  three_seed_claim_holds=bool(mean_change < 0 and pooled[1] < 0)))
    (OUT / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('models',)}, indent=1))


if __name__ == '__main__':
    main()
