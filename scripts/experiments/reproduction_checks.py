#!/usr/bin/env python3
"""Bit-level reproduction checks on completed runs (CPU only; reads checkpoints, writes one JSON record).

    .venv/bin/python scripts/experiments/reproduction_checks.py full-state
    .venv/bin/python scripts/experiments/reproduction_checks.py anchor [--run R --control C --member M]

full-state: the full Lookahead contract on GPU at production scale. One train800k epoch trained as five
restarted generations (full_state_gpu_fifths) must equal the same epoch trained uninterrupted
(full_state_gpu_one_epoch): identical restart bundle (model, RAdam and Lookahead state, AMP scaler), slices
that partition the epoch, and the same validation metric. Also measures the cost of the restarts, which the
0.2-epoch scenario pays every generation. Exit 1 if the contract does not hold.

anchor: one member of a new run repeats the same member of an earlier run (same start, seed, data and Weaver
command) on whatever hardware it ran. By default lr_14e-6 of the extended fixed-LR grid against the original
50-epoch control; other runs name their own run, control and member. Reports, per epoch, whether the model weights are
identical and how far the validation metric moved. Descriptive: a difference measures the run-to-run variation
between the two hardware/software environments that the high-grid comparison inherits.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage2a.common import PRIMARY, utc_now, write_json  # noqa: E402
from experiments.stage3.checkpoint_contents import bundle_differences, differences, load, restart_bundle  # noqa: E402
from training.runtime import PROJECT_DIR  # noqa: E402

RUNS = PROJECT_DIR / 'runs/pbt'
OUTPUT = PROJECT_DIR / 'runs/eval/reproduction_checks'
MEMBER = 'lr_14e-6'
CONTROL = 'foundation_fixed_lr_50epochs_20260916'


def completed(run):
    manifest = json.loads((RUNS / run / 'manifest.json').read_text())
    if manifest.get('status') != 'completed':
        raise ValueError(f'{run} is not completed')
    return manifest


def seconds(worker):
    return (datetime.fromisoformat(worker['finished_at']) - datetime.fromisoformat(worker['started_at'])).total_seconds()


def full_state(one_run='full_state_gpu_one_epoch', sliced_run='full_state_gpu_fifths'):
    one, sliced = completed(one_run), completed(sliced_run)
    reference = one['generations'][0]['workers'][MEMBER]
    slices = [g['workers'][MEMBER] for g in sliced['generations']]
    batches = int(sliced['config']['shared']['batches_per_generation'])
    total = reference['metrics']['train_data_audit']
    audits = [s['metrics']['train_data_audit'] for s in slices]
    partition = dict(
        batches=[a['batches'] for a in audits], reference_batches=total['batches'],
        rows=sum(a['consumed']['count'] for a in audits), reference_rows=total['consumed']['count'],
        exhausted=[a['exhausted'] for a in audits])
    partition['ok'] = (partition['batches'] == [batches] * (len(audits) - 1) + [total['batches'] - batches * (len(audits) - 1)]
                       and partition['rows'] == partition['reference_rows']
                       and partition['exhausted'] == [False] * (len(audits) - 1) + [True])
    one_epoch, sliced_epoch = one['generations'][0]['epoch'], sliced['generations'][-1]['epoch']
    found = bundle_differences(restart_bundle(RUNS / one_run / MEMBER, one_epoch),
                               restart_bundle(RUNS / sliced_run / MEMBER, sliced_epoch))
    metric = dict(uninterrupted=reference['metrics'][PRIMARY], restarted=slices[-1]['metrics'][PRIMARY])
    metric['identical'] = metric['uninterrupted'] == metric['restarted']
    cost = dict(uninterrupted_seconds=seconds(reference), restarted_seconds=[seconds(s) for s in slices])
    cost['restarted_over_uninterrupted'] = sum(cost['restarted_seconds']) / cost['uninterrupted_seconds']
    # Generation 0 of both runs starts from the historical epoch-17 checkpoint, so its model weights should also
    # equal the original control's epoch 18 if the hardware/software environment reproduces it.
    control = differences(load(restart_bundle(RUNS / CONTROL / MEMBER, 18)['state']),
                          load(restart_bundle(RUNS / one_run / MEMBER, one_epoch)['state']))
    result = dict(check='full_state_gpu_equivalence', checked_at=utc_now(), runs=[one_run, sliced_run],
                  bundles=dict(uninterrupted_epoch=one_epoch, restarted_epoch=sliced_epoch, differences=found[:20]),
                  partition=partition, validation_metric=metric, cost=cost,
                  original_control_epoch18_model_identical=not control)
    result['equivalent'] = not found and partition['ok'] and metric['identical']
    return result


ANCHOR_RUN = 'fixed_lr_high_grid_50epochs'


def anchor(run=ANCHOR_RUN, control=CONTROL, member=MEMBER):
    new, old = completed(run), completed(control)
    epochs = []
    for left, right in zip(old['generations'], new['generations']):
        if left['epoch'] != right['epoch']:
            raise ValueError('Generations of the two runs do not align by epoch')
        same = not differences(load(restart_bundle(RUNS / control / member, left['epoch'])['state']),
                               load(restart_bundle(RUNS / run / member, right['epoch'])['state']))
        a, b = left['workers'][member]['metrics'][PRIMARY], right['workers'][member]['metrics'][PRIMARY]
        epochs.append(dict(epoch=left['epoch'], model_identical=same, control_metric=a, repeat_metric=b, delta=b - a))
    differing = [e['epoch'] for e in epochs if not e['model_identical']]
    deltas = [abs(e['delta']) for e in epochs]
    # The first anchor record kept the plain name; later anchors are named after their run.
    name = 'anchor_member_reproduction' + ('' if run == ANCHOR_RUN else f'_{run}')
    return dict(check=name, checked_at=utc_now(), runs=[control, run], member=member,
                identical_epochs=len(epochs) - len(differing), epochs_compared=len(epochs),
                first_differing_epoch=differing[0] if differing else None,
                max_abs_metric_delta=max(deltas), mean_abs_metric_delta=sum(deltas) / len(deltas), epochs=epochs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check', choices=['full-state', 'anchor'])
    parser.add_argument('--run', default=ANCHOR_RUN, help='anchor: the new run')
    parser.add_argument('--control', default=CONTROL, help='anchor: the earlier run it repeats')
    parser.add_argument('--member', default=MEMBER, help='anchor: the member both runs share')
    args = parser.parse_args(argv)
    try:
        result = full_state() if args.check == 'full-state' else anchor(args.run, args.control, args.member)
    except (OSError, ValueError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    path = OUTPUT / f'{result["check"]}.json'
    write_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'epochs'}, indent=2))
    print(f'Written: {path}')
    return 1 if result.get('equivalent') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
