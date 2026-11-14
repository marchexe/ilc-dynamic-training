#!/usr/bin/env python3
"""Confidence intervals for paired holdout comparisons, resampling blocks of adjacent jets.

    .venv/bin/python scripts/experiments/holdout_block_bootstrap.py [--block 100] [--replicates 500] [--workers 9]

The holdout files have no event identifier, so the earlier holdout reports give no interval. The jets of one
event are adjacent rows of a file, however: the invariant mass of two adjacent jets peaks at 115-130 GeV (the
Higgs boson of the nunu qq sample), and the mass of two random jets does not. Some events lose a jet to the
selection, so adjacent pairs cannot be matched one to one. A block of B adjacent rows of one file holds whole
events except at its two ends, so resampling blocks keeps the jets of an event together (block bootstrap).
--sensitivity repeats one comparison for several block sizes: the interval must not grow once B covers an event.
--pooled gives the average change of PBT against standard ParT over the seeds; every replicate draws one set of
blocks and uses it for all seeds, because the seeds share the holdout jets.

The interval is for these fixed checkpoints and this holdout sample. It does not include the variation between
training seeds; that is what the seeds themselves show. Results go to runs/eval/holdout_block_bootstrap_v1/.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage2a.analysis import paired_bootstrap, score_metrics  # noqa: E402
from experiments.stage2a.common import PRIMARY, utc_now, write_json  # noqa: E402
from training.runtime import PROJECT_DIR  # noqa: E402

EVAL = PROJECT_DIR / 'runs/eval'
OUTPUT = EVAL / 'holdout_block_bootstrap_v1'
SEEDS = [('s12345', 'holdout_sel1m_v1'), ('s22345', 'holdout_sel1m_v1'), ('s32345', 'holdout_sel1m_seed32345_v1')]
PAIRS = [('standard', 'pbt'), ('fixed', 'pbt'), ('standard', 'fixed')]


def predictions(campaign, ident):
    """(labels, scores) ordered by Weaver row ID; checks the result's metric and that the IDs are 0..N-1."""
    result = json.loads((EVAL / campaign / 'evaluations' / ident / 'result.json').read_text())
    attempts = sorted((EVAL / campaign / 'evaluations' / ident).glob('attempt-*/evaluation.predictions.parquet'))
    table = pq.read_table(attempts[-1])
    ids = table['__weaver_row_id__'].to_numpy()
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(ids))):
        raise ValueError(f'{ident}: row IDs are not 0..N-1')
    labels = table['_label_'].to_numpy()[order]
    scores = np.stack(table['scores'].to_numpy(zero_copy_only=False))[order]
    metric = score_metrics(labels, scores)[PRIMARY]
    if not np.isclose(metric, result['metrics'][PRIMARY], rtol=1e-10, atol=1e-12):
        raise ValueError(f'{ident}: recomputed metric {metric} differs from {result["metrics"][PRIMARY]}')
    return labels, scores


def blocks(labels, size):
    """Block ID per row: the rows of each file (one label per file, files in row-ID order) cut into `size` rows."""
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    ids = np.empty(len(labels), dtype=np.int64)
    offset = 0
    for begin, end in zip(starts, np.r_[starts[1:], len(labels)]):
        ids[begin:end] = offset + np.arange(end - begin) // size
        offset = ids[end - 1] + 1
    return ids


def compare(job):
    seed, campaign, left, right, size, replicates = job
    labels, a = predictions(campaign, f'{left}_sel1m_{seed}')
    labels_b, b = predictions(campaign, f'{right}_sel1m_{seed}')
    if not np.array_equal(labels, labels_b):
        raise ValueError('the two checkpoints were not evaluated on the same rows')
    base, other = score_metrics(labels, a)[PRIMARY], score_metrics(labels, b)[PRIMARY]
    result = paired_bootstrap(labels, a, b, physical_event_ids=blocks(labels, size), replicates=replicates)
    low, high = result['intervals'][PRIMARY] if result['status'] == 'available' else (None, None)
    return dict(seed=seed, campaign=campaign, left=left, right=right, block=size, left_primary=base,
                right_primary=other, change_percent=100 * (other / base - 1),
                interval_percent=None if low is None else [100 * low / base, 100 * high / base], bootstrap=result)


def _pooled_chunk(job):
    pairs, size, replicates, seed = job
    loaded = [(predictions(campaign, f'{left}_sel1m_{s}'), predictions(campaign, f'{right}_sel1m_{s}'))
              for s, campaign, left, right in pairs]
    labels = loaded[0][0][0]
    if any(not np.array_equal(a[0], labels) or not np.array_equal(b[0], labels) for a, b in loaded):
        raise ValueError('the seeds were not evaluated on the same holdout rows')
    block = blocks(labels, size)
    n_blocks = block[-1] + 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
        rows = np.repeat(np.arange(len(labels)), counts[block])
        changes = [100 * (score_metrics(labels[rows], b[1][rows])[PRIMARY] / score_metrics(labels[rows], a[1][rows])[PRIMARY] - 1)
                   for a, b in loaded]
        out.append(changes)
    return out


def pooled(left, right, size, replicates, workers):
    """Average change of right against left over the seeds, with the same resampled blocks for every seed."""
    pairs = [(s, campaign, left, right) for s, campaign in SEEDS]
    point = []
    for s, campaign, _, _ in pairs:
        labels, a = predictions(campaign, f'{left}_sel1m_{s}')
        _, b = predictions(campaign, f'{right}_sel1m_{s}')
        point.append(100 * (score_metrics(labels, b)[PRIMARY] / score_metrics(labels, a)[PRIMARY] - 1))
    share = [replicates // workers + (i < replicates % workers) for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        draws = [d for chunk in pool.map(_pooled_chunk, [(pairs, size, n, 20261115 + i) for i, n in enumerate(share)])
                 for d in chunk]
    means = np.mean(draws, axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(left=left, right=right, block=size, replicates=len(draws), seeds=[s for s, _ in SEEDS],
                change_percent_per_seed=point, mean_change_percent=float(np.mean(point)),
                mean_interval_percent=[float(low), float(high)],
                fraction_of_replicates_with_mean_below_zero=float(np.mean(means < 0)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--block', type=int, default=100)
    parser.add_argument('--replicates', type=int, default=500)
    parser.add_argument('--workers', type=int, default=9)
    parser.add_argument('--sensitivity', type=int, nargs='*', help='block sizes for standard vs PBT of seed 32345')
    parser.add_argument('--pooled', action='store_true', help='average over the seeds of standard ParT against PBT')
    args = parser.parse_args(argv)
    if args.pooled:
        result = pooled('standard', 'pbt', args.block, args.replicates, args.workers)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / f'pooled_block{args.block}.json', dict(schema_version=1, finished_at=utc_now(), **result))
        low, high = result['mean_interval_percent']
        print(f"pbt vs standard, mean of {len(result['seeds'])} seeds: {result['mean_change_percent']:+.2f}% "
              f"[{low:+.2f}%, {high:+.2f}%]")
        return 0
    if args.sensitivity:
        seed, campaign = SEEDS[-1]
        jobs = [(seed, campaign, 'standard', 'pbt', size, args.replicates) for size in args.sensitivity]
        name = 'sensitivity.json'
    else:
        jobs = [(seed, campaign, left, right, args.block, args.replicates) for seed, campaign in SEEDS
                for left, right in PAIRS]
        name = f'block{args.block}.json'
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(compare, jobs))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / name, dict(schema_version=1, finished_at=utc_now(), replicates=args.replicates,
                                   unit='block of adjacent rows of one holdout file', comparisons=results))
    for r in results:
        interval = r['interval_percent']
        print(f"{r['seed']} {r['right']} vs {r['left']} block {r['block']}: {r['change_percent']:+.2f}% "
              + ('(no interval)' if interval is None else f'[{interval[0]:+.2f}%, {interval[1]:+.2f}%]'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
