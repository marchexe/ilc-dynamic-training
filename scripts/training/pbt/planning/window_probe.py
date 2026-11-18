"""Cool-down probes for windowed_pbt_v2: judge each member's LR after a short LR decrease, not at its raw value.

On train8m the windowed PBT lowered every LR within ten epochs and finished 10.8% behind standard ParT training
(docs/results.rst, pbt-vs-standard-part-8m). In its decision log, the member whose LR had just been lowered
won every window from epoch 22 to 33, although in standard ParT the 1e-3 member is ahead of 3e-4 and 1e-4 at
every epoch. A two-epoch window rewards the quick gain that a lower LR gives at once and cannot see the larger
gain that a high LR gives once it decays (short-horizon bias).

With `windowed_pbt_v2.probe`, at every boundary that can make a decision, each member's checkpoint is copied
and trained for `samples` training jets at `lr_factor` times its LR, then validated on the full selection set.
The decision then ranks the members on these probe values instead of their last window epoch (score_epochs
must be 1); everything else (margin, loss counters, mutations, fallback copies) is the frozen reference
decision. Probe weights never return to the population and are never selected; their compute is recorded.
"""
import copy
import shutil
from pathlib import Path

from training.checkpoints import bundle_paths, copy_bundle
from training.runtime import atomic_json, utc_now
from training.pbt.planning.lr_schedule import final_decay_start

STRATEGY = 'windowed_pbt_v2'


def probe_options(config):
    pbt = config.get('pbt') or {}
    if pbt.get('strategy') != STRATEGY:
        return None
    return (pbt.get(STRATEGY) or {}).get('probe')


def probe_due(config, index):
    """True when generation `index` ends a window whose decision can copy members."""
    options = probe_options(config)
    if not options:
        return False
    width = int(config['pbt'][STRATEGY]['window_epochs'])
    generations = int(config['shared']['generations'])
    if (index + 1) % width or index + 1 >= generations:
        return False
    start = final_decay_start(config)
    return start is None or index < start - 1


def run_window_probes(config, backend, experiment_dir, manifest, generation, manifest_path):
    """Train and validate one probe per member; persist the result in generation['probes']."""
    if not probe_due(config, generation['index']) or (generation.get('probes') or {}).get('status') == 'completed':
        return generation.get('probes')
    options = probe_options(config)
    experiment_dir = Path(experiment_dir)
    epoch = generation['epoch']
    root = experiment_dir / 'probes' / f"generation-{generation['index']:03d}"
    if root.exists():
        shutil.rmtree(root)  # an interrupted probe restarts from the unchanged member checkpoints
    names = sorted(manifest['members'])
    members = {}
    for name in names:
        lr = float(manifest['members'][name]['lr'])
        source = bundle_paths(experiment_dir / name, epoch)
        copied = copy_bundle(source, bundle_paths(root / name, epoch))
        members[name] = dict(name=name, lr=lr * float(options['lr_factor']), base_lr=lr, checkpoint=copied)
    probe_config = copy.deepcopy(config)
    probe_config['shared']['samples_per_epoch'] = int(options['samples'])
    record = dict(index=generation['index'] + 1, epoch=epoch + 1, seed=None,
                  workers={n: dict(status='pending') for n in names})
    probe_manifest = dict(members={n: dict(name=n, lr=m['lr']) for n, m in members.items()}, generations=[record])
    generation['probes'] = dict(status='running', started_at=utc_now(), lr_factor=float(options['lr_factor']),
                                samples=int(options['samples']), from_epoch=epoch, directory=str(root),
                                members=members)
    atomic_json(manifest_path, manifest)
    backend.run_generation(probe_config, root, probe_manifest, record, names, root / 'probe_manifest.json')
    metric = config['pbt']['metric']
    for name in names:
        worker = record['workers'][name]
        if worker.get('status') != 'completed':
            raise RuntimeError(f'probe of {name} did not complete')
        members[name].update(metrics=worker['metrics'], value=worker['metrics'][metric], log=worker.get('log'),
                             command=worker.get('command'))
    generation['probes'].update(status='completed', finished_at=utc_now())
    atomic_json(manifest_path, manifest)
    return generation['probes']


def probe_scored_generation(config, generation):
    """A copy of the generation whose last-epoch metrics are the probe metrics, for the reference decision."""
    probes = generation.get('probes')
    if not probe_due(config, generation['index']):
        return generation
    if not probes or probes.get('status') != 'completed':
        raise ValueError('a probe-scored decision needs completed probes')
    scored = copy.deepcopy(generation)
    for name, item in probes['members'].items():
        scored['workers'][name]['metrics'] = item['metrics']
    return scored
