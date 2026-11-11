#!/usr/bin/env python3
"""Weight averages of trained checkpoints, scored on the selection set (sel1m_tail).

    .venv/bin/python scripts/experiments/weight_averaging.py --prepared configs/prepared/weight_averaging_sel1m_v1 \
        --gpus 0,1,2,3,4 [--only ID ...]

The selected checkpoint of a run is one epoch of one member, and the selection-set metric of a member moves by
0.002-0.003 from epoch to epoch at a constant LR, as much as the differences between the arms. A uniform average
of nearby checkpoints of the same lineage (stochastic weight averaging) or of the related members of a population
(a "model soup") is one model of the same size and cost. DIR/spec.json fixes the recipes and the selection rule
before anything is scored. Results go to runs/eval/<name>/: one directory per run and candidate with the averaged
state, its sources and the Weaver test log, and selection.json with every score and each run's selection.

Training runs with --freeze-batch-norm, so the BatchNorm running statistics are the same in every checkpoint and
an average of them is exact.
"""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage2a.common import PRIMARY, file_record, sha256, utc_now, write_json  # noqa: E402
from training.runtime import PROJECT_DIR, data_paths, git_metadata, read_metrics, weaver_executable  # noqa: E402


def load_json(path):
    return json.loads(Path(path).read_text())


def run_manifest(run):
    return load_json(PROJECT_DIR / 'runs/pbt' / run / 'manifest.json')


def state_path(run_dir, member, epoch):
    return Path(run_dir) / member / f'net_epoch-{epoch}_state.pt'


def copies_by_epoch(manifest):
    """{epoch: {recipient: donor}} for the applied copies; a copy at epoch t replaces the recipient's epoch-t file."""
    copies = {}
    for generation in manifest['generations']:
        for event in generation.get('exploit') or []:
            if event.get('applied'):
                copies.setdefault(generation['epoch'], {})[event['recipient']] = event['donor']
    return copies


def lineage(manifest, member, epoch, run_dir):
    """{t: checkpoint path} of the ancestry of (member, epoch), for every trained epoch t <= epoch.

    At an epoch where the member received a copy its file already holds the donor's weights, so the ancestry
    switches to the donor for the earlier epochs."""
    copies = copies_by_epoch(manifest)
    first = min(g['epoch'] for g in manifest['generations'])
    paths, current = {}, member
    for t in range(epoch, first - 1, -1):
        paths[t] = state_path(run_dir, current, t)
        current = copies.get(t, {}).get(current, current)
    return paths


def window_paths(manifest, run_dir, recipe):
    """Checkpoints of a lineage window around the selected checkpoint (manifest best)."""
    best = manifest['best']
    member, epoch = best['member'], best['epoch']
    epochs = [g['epoch'] for g in manifest['generations']]
    first, last = min(epochs), max(epochs)
    ancestry = lineage(manifest, member, epoch, run_dir)
    paths = [ancestry[t] for t in range(max(first, epoch - recipe['before']), epoch + 1)]
    later = copies_by_epoch(manifest)
    for t in range(epoch + 1, min(last, epoch + recipe['after']) + 1):
        if member in later.get(t, {}):
            break  # the member's later files belong to another lineage from here on
        paths.append(state_path(run_dir, member, t))
    return paths


def soup_order(manifest, run_dir):
    """The members' epoch-e checkpoints, best training-time selection metric first, without copy recipients."""
    best = manifest['best']
    epoch = best['epoch']
    generation = next(g for g in manifest['generations'] if g['epoch'] == epoch)
    recipients = copies_by_epoch(manifest).get(epoch, {})
    ranked = sorted((w['metrics'][PRIMARY], name) for name, w in generation['workers'].items()
                    if name not in recipients and w.get('metrics'))
    names = [name for _, name in ranked]
    if names[0] != best['member']:
        raise ValueError(f'the best member at epoch {epoch} is {names[0]}, not the selected {best["member"]}')
    return [state_path(run_dir, name, epoch) for name in names]


def average_states(paths):
    """Uniform average of state dicts; integer tensors (BatchNorm counters) must agree and are kept."""
    states = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    out = {}
    for key, first in states[0].items():
        values = [s[key] for s in states]
        if first.is_floating_point():
            out[key] = (torch.stack([v.double() for v in values]).mean(0)).to(first.dtype)
        else:
            if any(not torch.equal(v, first) for v in values[1:]):
                raise ValueError(f'{key}: integer tensors differ between checkpoints')
            out[key] = first.clone()
    if any(set(s) != set(out) for s in states):
        raise ValueError('checkpoints have different keys')
    return out


class Evaluator:
    """Weaver test mode on the selection set; GPUs are handed out one evaluation at a time."""

    def __init__(self, manifest, gpus):
        shared = manifest['config']['shared']
        self.shared = shared
        self.gpus = queue.Queue()
        for gpu in gpus:
            self.gpus.put(gpu)

    def command(self, checkpoint, log_path, gpu):
        shared = self.shared
        val = [v.split(':', 1)[1] for v in data_paths(shared['validation_dataset'], shared['data_extension'],
                                                         validation_suffix=shared['validation_suffix'])['val']]
        return [str(weaver_executable()), '--run-mode', 'test', '--data-test', *val,
                '--data-config', shared['data_config'], '--network-config', shared['network_config'],
                '--model-prefix', str(checkpoint), '--log-file', str(log_path),
                '--batch-size', str(shared['batch_size']), '--use-amp', '--amp-dtype', shared['amp_dtype'],
                '--num-workers', str(shared['num_workers']), '--fetch-step', str(shared['fetch_step']),
                '--gpus', str(gpu), '--predict-gpus', str(gpu)]

    def __call__(self, checkpoint, directory):
        result_path = directory / 'score.json'
        if result_path.exists():
            score = load_json(result_path)
            if score['checkpoint']['sha256'] == sha256(checkpoint):
                return score['metric']
        gpu = self.gpus.get()
        try:
            log_path = directory / 'test.log'
            command = self.command(checkpoint, log_path, gpu)
            with open(directory / 'console.log', 'w') as console:
                code = subprocess.run(command, cwd=PROJECT_DIR, stdout=console, stderr=subprocess.STDOUT).returncode
        finally:
            self.gpus.put(gpu)
        metrics = read_metrics(log_path)
        if code != 0 or not metrics or metrics.get(PRIMARY) is None:
            raise RuntimeError(f'evaluation failed ({code}): {log_path}')
        write_json(result_path, dict(checkpoint=file_record(checkpoint), metric=metrics[PRIMARY], metrics=metrics,
                                     command=command, finished_at=utc_now()))
        return metrics[PRIMARY]


def materialize(paths, directory):
    """Write the average of paths to directory/state.pt (a single path is used as it is)."""
    directory.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        checkpoint = Path(paths[0])
    else:
        checkpoint = directory / 'state.pt'
        if not checkpoint.exists():
            torch.save(average_states(paths), directory / 'state.pt.tmp')
            (directory / 'state.pt.tmp').rename(checkpoint)
    write_json(directory / 'sources.json', dict(sources=[file_record(p) for p in paths]))
    return checkpoint


def score_run(entry, recipes, output, evaluate):
    manifest = run_manifest(entry['run'])
    run_dir = PROJECT_DIR / 'runs/pbt' / entry['run']
    best = manifest['best']
    if sha256(state_path(run_dir, best['member'], best['epoch'])) != best['source_checkpoint']['state']['sha256']:
        raise ValueError(f'{entry["run"]}: the selected checkpoint file changed')
    base = output / entry['id']
    candidates = []

    def score(name, paths, **extra):
        checkpoint = materialize(paths, base / name)
        metric = evaluate(checkpoint, base / name)
        candidates.append(dict(recipe=name, metric=metric, checkpoint=file_record(checkpoint),
                               n_sources=len(paths), epochs=[int(Path(p).name.split('-')[1].split('_')[0]) for p in paths],
                               members=[Path(p).parent.name for p in paths], **extra))
        return metric

    for recipe in recipes:
        if recipe['kind'] == 'plain':
            score(recipe['name'], [state_path(run_dir, best['member'], best['epoch'])])
        elif recipe['kind'] == 'lineage_window':
            score(recipe['name'], window_paths(manifest, run_dir, recipe))
        elif recipe['kind'] == 'greedy_population_soup':
            order = soup_order(manifest, run_dir)
            kept = [order[0]]
            current = next(c['metric'] for c in candidates if c['recipe'] == 'plain')
            trials = []
            for index, path in enumerate(order[1:], start=1):
                metric = score(f'{recipe["name"]}_trial{index}', kept + [path])
                accepted = metric < current
                trials.append(dict(added=Path(path).parent.name, metric=metric, accepted=accepted))
                if accepted:
                    kept, current = kept + [path], metric
            candidates = [c for c in candidates if not c['recipe'].startswith(recipe['name'] + '_trial')]
            candidates.append(dict(recipe=recipe['name'], metric=current, members=[Path(p).parent.name for p in kept],
                                   n_sources=len(kept), trials=trials,
                                   checkpoint=file_record(materialize(kept, base / recipe['name']))))
        else:
            raise ValueError(f'unknown recipe kind {recipe["kind"]}')
    order = [r['name'] for r in recipes]
    chosen = min(candidates, key=lambda c: (c['metric'], order.index(c['recipe'])))
    return dict(id=entry['id'], run=entry['run'], selected_member=best['member'], selected_epoch=best['epoch'],
                training_time_metric=best['metric_value'], candidates=candidates, selection=chosen['recipe'],
                selection_metric=chosen['metric'], selection_checkpoint=chosen['checkpoint'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--gpus', required=True, help='comma-separated GPU ids, at most five')
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args(argv)
    gpus = [g for g in args.gpus.split(',') if g]
    if not 1 <= len(gpus) <= 5:
        raise SystemExit('use one to five GPUs')
    spec = load_json(args.prepared / 'spec.json')
    output = PROJECT_DIR / 'runs/eval' / spec['name']
    output.mkdir(parents=True, exist_ok=True)
    entries = [e for e in spec['runs'] if not args.only or e['id'] in args.only]
    evaluate = Evaluator(run_manifest(entries[0]['run']), gpus)
    for entry in entries[1:]:
        shared = run_manifest(entry['run'])['config']['shared']
        if any(shared[k] != evaluate.shared[k] for k in ('validation_dataset', 'validation_suffix', 'data_config',
                                                         'network_config', 'batch_size', 'amp_dtype', 'fetch_step')):
            raise ValueError(f'{entry["run"]}: evaluation settings differ between runs')
    results, errors = {}, {}

    def work(entry):
        try:
            results[entry['id']] = score_run(entry, spec['recipes'], output, evaluate)
        except Exception as error:  # noqa: BLE001 - reported per run
            errors[entry['id']] = repr(error)

    threads = [threading.Thread(target=work, args=(e,)) for e in entries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_json(output / 'selection.json', dict(schema_version=1, name=spec['name'], finished_at=utc_now(),
                                               spec=file_record(args.prepared / 'spec.json'), git=git_metadata(),
                                               runs=[results[e['id']] for e in entries if e['id'] in results],
                                               errors=errors))
    for entry in entries:
        r = results.get(entry['id'])
        print(entry['id'], errors.get(entry['id']) if r is None else
              f"plain {next(c['metric'] for c in r['candidates'] if c['recipe'] == 'plain'):.5f} "
              f"-> {r['selection']} {r['selection_metric']:.5f}")
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
