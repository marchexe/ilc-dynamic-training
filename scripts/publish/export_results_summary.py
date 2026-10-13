#!/usr/bin/env python3
"""Export the public results summary: validation endpoints, independent-holdout results and figures.

Read-only with respect to every source: it reads completed run manifests (runs/pbt/*) and the
completed independent-holdout evaluation (runs/eval/stage2a_holdout_v1), then writes repository-relative
tables, a JSON summary with source hashes, and figures to published/results/. Raw predictions, logs and
machine-specific paths are never copied.

    python scripts/publish/export_results_summary.py [--source-root <checkout with runs/>]
"""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from training.pbt.reporting.endpoints import current_best_trajectory, final_window_mean  # noqa: E402

PRIMARY = 'validation_total_reference_mistag_geomean_percent'
WORKING_POINTS = ['validation_bc_mistag_eff_0.80_percent', 'validation_bd_mistag_eff_0.80_percent',
                  'validation_bc_mistag_eff_0.90_percent', 'validation_bd_mistag_eff_0.90_percent',
                  'validation_cb_mistag_eff_0.50_percent', 'validation_cd_mistag_eff_0.50_percent',
                  'validation_cb_mistag_eff_0.80_percent', 'validation_cd_mistag_eff_0.80_percent']
WP_LABELS = ['b vs c @ 80% b-eff.', 'b vs d @ 80% b-eff.', 'b vs c @ 90% b-eff.', 'b vs d @ 90% b-eff.',
             'c vs b @ 50% c-eff.', 'c vs d @ 50% c-eff.', 'c vs b @ 80% c-eff.', 'c vs d @ 80% c-eff.']
PRETRAINED_EPOCH = 17  # Weaver epoch of the starting checkpoint; training epoch = Weaver epoch - 17

# Validation-endpoint experiments: public id -> (label, policy, run segments in order).
RUNS = {
    'fixed-lr-50': ('Fixed-LR population, 50 epochs', 'fixed_lr_grid', ['foundation_fixed_lr_50epochs_20260916']),
    'windowed-pbt-50': ('Windowed PBT, 50 epochs', 'windowed_pbt_v2', ['windowed_pbt_v2']),
    'windowed-pbt-100': ('Windowed PBT, 100 epochs', 'windowed_pbt_v2', ['windowed_pbt_v2_100epochs']),
    'fixed-lr-1.4e-5-100': ('Fixed LR 1.4e-5, 100 epochs (single member)', 'fixed_lr_grid',
                            ['foundation_fixed_lr_50epochs_20260916:lr_14e-6', 'fixed_lr_continuation_20260917_lr_14e-6']),
    'fixed-lr-8.5e-6-100': ('Fixed LR 8.5e-6, 100 epochs (single member)', 'fixed_lr_grid',
                            ['foundation_fixed_lr_50epochs_20260916:lr_8_5e-6', 'fixed_lr_continuation_20260917_lr_8_5e-6']),
    'every-epoch-pbt-s12345': ('Every-epoch PBT, seed 12345', 'cadenced_pbt_v1 (interval 1)', ['cadenced_pbt_v1_50epochs']),
    'every-5-epochs-pbt-s12345': ('Every-5-epochs PBT, seed 12345', 'cadenced_pbt_v1 (interval 5)',
                                  ['cadenced_pbt_v1_cadence5_50epochs']),
    'every-epoch-pbt-s22345': ('Every-epoch PBT, seed 22345', 'cadenced_pbt_v1 (interval 1)',
                               ['cadenced_pbt_v1_seed22345_50epochs']),
    'every-5-epochs-pbt-s22345': ('Every-5-epochs PBT, seed 22345', 'cadenced_pbt_v1 (interval 5)',
                                  ['cadenced_pbt_v1_cadence5_seed22345_50epochs']),
}
# Independent-holdout checkpoint id -> (public id of its experiment, public label).
CHECKPOINTS = {
    'pretrained17': (None, 'Pretrained start (epoch 17)'),
    'fixed50_s12345': ('fixed-lr-50', 'Fixed-LR population, 50 ep.'),
    'windowed50_s12345': ('windowed-pbt-50', 'Windowed PBT, 50 ep.'),
    'windowed100_s12345': ('windowed-pbt-100', 'Windowed PBT, 100 ep.'),
    'fixed14_100_s12345': ('fixed-lr-1.4e-5-100', 'Fixed LR 1.4e-5, 100 ep.'),
    'cadence1_s12345': ('every-epoch-pbt-s12345', 'Every-epoch PBT, seed 12345'),
    'cadence5_s12345': ('every-5-epochs-pbt-s12345', 'Every-5-epochs PBT, seed 12345'),
    'cadence1_s22345': ('every-epoch-pbt-s22345', 'Every-epoch PBT, seed 22345'),
    'cadence5_s22345': ('every-5-epochs-pbt-s22345', 'Every-5-epochs PBT, seed 22345'),
}
# Matched comparisons: (public label, treatment checkpoint, comparator checkpoint, caveat).
COMPARISONS = [
    ('Windowed PBT vs fixed-LR population, 50 ep.', 'windowed50_s12345', 'fixed50_s12345', 'matched population and seed'),
    ('Windowed PBT vs fixed LR 1.4e-5, 100 ep.', 'windowed100_s12345', 'fixed14_100_s12345',
     'asymmetric: single fixed-LR member chosen after epoch 50'),
    ('Every-epoch vs every-5-epochs PBT, seed 12345', 'cadence1_s12345', 'cadence5_s12345', 'matched; differs only in interval'),
    ('Every-epoch vs every-5-epochs PBT, seed 22345', 'cadence1_s22345', 'cadence5_s22345', 'matched; differs only in interval'),
    ('Every-epoch vs windowed PBT, 50 ep.', 'cadence1_s12345', 'windowed50_s12345',
     'policies differ in several rules, not cadence only'),
]
HOLDOUT = 'runs/eval/stage2a_holdout_v1'
COLORS = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', surface='#fcfcfb', text='#0b0b0b', muted='#52514e',
              grid='#e4e3df')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def seconds(worker):
    return (datetime.fromisoformat(worker['finished_at']) - datetime.fromisoformat(worker['started_at'])).total_seconds()


def load_segment(source, segment, sources):
    name, _, member = segment.partition(':')
    path = source / 'runs/pbt' / name / 'manifest.json'
    sources.append(dict(artifact=f'runs/pbt/{name}/manifest.json', sha256=sha256(path)))
    manifest = json.loads(path.read_text())
    generations = [g for g in manifest['generations'] if g['status'] == 'completed']
    members = [member] if member else sorted({n for g in generations for n in g['workers']})
    return manifest, generations, members


def endpoints(source, sources):
    rows, trajectories = [], {}
    for public, (label, policy, segments) in RUNS.items():
        values, epochs, durations, copies, best, seed = {}, [], [], 0, None, None
        initial = None
        for segment in segments:
            manifest, generations, members = load_segment(source, segment, sources)
            if initial is None:
                initial = manifest['initial_evaluation']['metrics'][PRIMARY]
                seed = generations[0]['seed']
            for generation in generations:
                epochs.append(generation['epoch'] - PRETRAINED_EPOCH)
                for name in members:
                    worker = generation['workers'][name]
                    values.setdefault(name, []).append(worker['metrics'][PRIMARY])
                    durations.append(seconds(worker))
                    if best is None or worker['metrics'][PRIMARY] < best[0]:
                        best = (worker['metrics'][PRIMARY], name, generation['epoch'] - PRETRAINED_EPOCH, worker['lr'])
                copies += sum(1 for e in generation.get('exploit') or [] if e.get('applied'))
        if epochs != list(range(1, len(epochs) + 1)):
            raise ValueError(f'{public}: training epochs are not contiguous')
        trajectory = current_best_trajectory(values)
        trajectories[public] = trajectory
        rows.append(dict(
            experiment=public, label=label, policy=policy, training_seed=seed, horizon_epochs=len(epochs),
            members=len(values), initial_primary=initial, final10_current_best=final_window_mean(trajectory),
            final10_window=f'{len(epochs) - 9}-{len(epochs)}', global_best=best[0], global_best_member=best[1],
            global_best_training_epoch=best[2], global_best_lr=best[3], applied_copies=copies,
            member_final10={name: final_window_mean(v) for name, v in sorted(values.items())},
            member_epoch_seconds=dict(min=min(durations), median=median(durations), max=max(durations)),
            gpu_hours=sum(durations) / 3600))
    return rows, trajectories


def paired_counts(record):
    """Background jets passing the tag (mistagged) at one working point; the recorded left side is the comparator."""
    return dict(background_jets=record['n_paired_background_rows'], mistagged_by_both=record['both_passed'],
                mistagged_by_comparator_only=record['left_only'], mistagged_by_treatment_only=record['right_only'],
                mistagged_by_neither=record['neither_passed'], delta_pp=record['delta_pp'])


def holdout(source, sources):
    root = source / HOLDOUT
    integrity = json.loads((root / 'report_integrity.json').read_text())
    recorded = {Path(a['path']).name: a['sha256'] for a in integrity['artifacts']}
    for name in ('analysis.json', 'comparison.csv'):
        if sha256(root / name) != recorded[name]:
            raise ValueError(f'{name} differs from its recorded integrity hash')
        sources.append(dict(artifact=f'{HOLDOUT}/{name}', sha256=recorded[name]))
    analysis = json.loads((root / 'analysis.json').read_text())
    checkpoints = []
    for row in analysis['rows']:
        public, label = CHECKPOINTS[row['checkpoint_id']]
        checkpoints.append(dict(
            checkpoint=row['checkpoint_id'], experiment=public, label=label, selection=row['selection'],
            training_seed=row['training_seed'], horizon_epochs=row['horizon'],
            training_epoch=row['weaver_epoch'] - PRETRAINED_EPOCH, checkpoint_sha256=row['checkpoint_sha256'],
            selected_validation_primary=row['original_validation_primary'],
            final10_current_best_validation=row['final10_current_best'],
            holdout_primary=row['independent_holdout_primary'],
            holdout_working_points={k: row['working_points'][k] for k in WORKING_POINTS}))
    by_id = {c['checkpoint']: c for c in checkpoints}
    matched = {(c['left'], c['right']): c for c in analysis['matched_comparisons']}
    comparisons = []
    for label, treatment, comparator, caveat in COMPARISONS:
        record = matched[(comparator, treatment)]
        t, c = by_id[treatment], by_id[comparator]
        delta = t['holdout_primary'] - c['holdout_primary']
        if not math.isclose(delta, record['delta_pp'], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f'{label}: holdout difference does not reproduce the recorded analysis')
        validation = t['final10_current_best_validation'] - c['final10_current_best_validation']
        comparisons.append(dict(
            comparison=label, treatment=treatment, comparator=comparator, caveat=caveat,
            validation_final10_delta_pp=validation,
            validation_final10_relative_percent=100 * validation / c['final10_current_best_validation'],
            holdout_selected_delta_pp=delta, holdout_selected_relative_percent=100 * delta / c['holdout_primary'],
            holdout_rows=record['n_paired_rows'], event_level_interval='unavailable (no physical event identifier)',
            paired_background_counts={k: paired_counts(record['working_points'][k]) for k in WORKING_POINTS}))
    return dict(rows=2_825_125, rows_by_class=dict(b=943_645, c=941_995, d=939_485),
                protocol_sha256=analysis['protocol_sha256'], no_holdout_reselection=analysis['no_holdout_reselection'],
                spearman_validation_vs_holdout=analysis['spearman_rank_correlation'],
                validation_order=[by_id[c]['label'] for c in analysis['validation_order']],
                holdout_order=[by_id[c]['label'] for c in analysis['diagnostic_holdout_order']],
                checkpoints=checkpoints, comparisons=comparisons)


def write_csv(path, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def style(plt):
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': COLORS['muted'], 'axes.labelcolor': COLORS['text'],
                         'xtick.color': COLORS['muted'], 'ytick.color': COLORS['muted'], 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': COLORS['surface'],
                         'axes.facecolor': COLORS['surface'], 'savefig.facecolor': COLORS['surface'],
                         'axes.grid': True, 'grid.color': COLORS['grid'], 'grid.linewidth': 0.6,
                         'legend.frameon': False, 'lines.linewidth': 2.0})


def figures(output, trajectories, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw=dict(width_ratios=[1, 2]))
    panels = [(axes[0], '50 epochs', [('windowed-pbt-50', 'Windowed PBT (5 members)', COLORS['blue']),
                                      ('fixed-lr-50', 'Fixed-LR population (5 members)', COLORS['orange'])]),
              (axes[1], '100 epochs', [('windowed-pbt-100', 'Windowed PBT (5 members)', COLORS['blue']),
                                       ('fixed-lr-1.4e-5-100', 'Fixed LR 1.4e-5 (1 member)', COLORS['orange'])])]
    for ax, title, series in panels:
        horizon = len(trajectories[series[0][0]])
        ax.axvspan(horizon - 9.5, horizon + 0.5, color=COLORS['grid'], alpha=0.6, lw=0)
        for public, label, color in series:
            values = trajectories[public]
            ax.plot(range(1, len(values) + 1), values, color=color, label=label)
        ax.set_title(title, loc='left', color=COLORS['text'], fontsize=11)
        ax.set_xlabel('Training epoch')
        ax.set_xlim(0.5, horizon + 0.5)
        ax.set_ylim(0.315, 0.42)
        ax.legend(loc='upper right')
    axes[0].set_ylabel('Current-best validation mistag (%)')
    fig.tight_layout()
    files.append(output / 'figures' / 'current_best_trajectories.png')
    fig.savefig(files[-1], dpi=200)
    plt.close(fig)

    trained = [c for c in results['checkpoints'] if c['experiment']]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    positions = range(len(trained))[::-1]
    for y, c in zip(positions, trained):
        ax.plot([c['holdout_primary'], c['selected_validation_primary']], [y, y], color=COLORS['grid'], lw=2, zorder=1)
    ax.scatter([c['selected_validation_primary'] for c in trained], list(positions), s=55, color=COLORS['orange'],
               edgecolor=COLORS['surface'], linewidth=2, zorder=3, label='Selection validation (selected checkpoint)')
    ax.scatter([c['holdout_primary'] for c in trained], list(positions), s=55, color=COLORS['blue'],
               edgecolor=COLORS['surface'], linewidth=2, zorder=3, label='Independent holdout (same checkpoint)')
    ax.set_yticks(list(positions), [c['label'] for c in trained])
    ax.set_xlabel('Composite mistag (%), lower is better')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2)
    fig.tight_layout()
    files.append(output / 'figures' / 'holdout_selected_checkpoints.png')
    fig.savefig(files[-1], dpi=200)
    plt.close(fig)

    comparisons = results['comparisons']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    positions = list(range(len(comparisons)))[::-1]
    for ax, key, title in ((axes[0], 'validation_final10_delta_pp', 'Validation: final-10 current-best (population)'),
                           (axes[1], 'holdout_selected_delta_pp', 'Independent holdout: selected checkpoint')):
        ax.axvline(0, color=COLORS['muted'], lw=1)
        values = [c[key] for c in comparisons]
        ax.scatter(values, positions, s=55, color=COLORS['blue'], edgecolor=COLORS['surface'], linewidth=2, zorder=3)
        ax.set_title(title, loc='left', color=COLORS['text'], fontsize=10)
        ax.set_xlabel('Treatment - comparator (pp)')
        ax.set_xlim(-0.0105, 0.0035)
        ax.set_xticks([-0.008, -0.004, 0.0, 0.002])
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(positions, [c['comparison'] for c in comparisons])
    fig.tight_layout()
    files.append(output / 'figures' / 'matched_differences.png')
    fig.savefig(files[-1], dpi=200)
    plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source-root', type=Path, default=PROJECT_DIR,
                        help='checkout containing the completed runs/ directory (read only)')
    parser.add_argument('--output', type=Path, default=PROJECT_DIR / 'published/results')
    args = parser.parse_args(argv)
    source, output = args.source_root.resolve(), args.output
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    sources = []
    rows, trajectories = endpoints(source, sources)
    results = holdout(source, sources)
    write_csv(output / 'validation_endpoints.csv', [
        {k: v for k, v in r.items() if k not in ('member_final10', 'member_epoch_seconds')} for r in rows])
    write_csv(output / 'holdout_checkpoints.csv', [
        dict({k: v for k, v in c.items() if k != 'holdout_working_points'},
             **{f'holdout_{k[len("validation_"):]}': v for k, v in c['holdout_working_points'].items()})
        for c in results['checkpoints']])
    write_csv(output / 'holdout_comparisons.csv', [
        {k: v for k, v in c.items() if k != 'paired_background_counts'} for c in results['comparisons']])
    write_csv(output / 'holdout_paired_counts.csv', [
        dict(comparison=c['comparison'], working_point=k[len('validation_'):], **counts)
        for c in results['comparisons'] for k, counts in c['paired_background_counts'].items()])
    write_csv(output / 'current_best_trajectories.csv', [
        dict(experiment=public, training_epoch=i + 1, current_best_validation=value)
        for public, values in trajectories.items() for i, value in enumerate(values)])
    plots = figures(output, trajectories, results)
    summary = dict(
        schema_version=1, metric=PRIMARY, lower_is_better=True, working_points=dict(zip(WORKING_POINTS, WP_LABELS)),
        training_epoch_definition='Weaver epoch minus 17 (the pretrained starting checkpoint)',
        validation_endpoints=rows, independent_holdout=results,
        sources=sorted({s['artifact']: s for s in sources}.values(), key=lambda s: s['artifact']),
        outputs={str(p.relative_to(output)): sha256(p) for p in sorted(output.glob('*.csv')) + plots})
    text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    if str(source) in text or str(PROJECT_DIR) in text:
        raise ValueError('machine-specific path in the public summary')
    (output / 'summary.json').write_text(text)
    print(json.dumps(dict(output=str(output.relative_to(PROJECT_DIR)) if output.is_relative_to(PROJECT_DIR) else str(output),
                          experiments=len(rows), checkpoints=len(results['checkpoints']), figures=len(plots)), indent=2))


if __name__ == '__main__':
    main()
