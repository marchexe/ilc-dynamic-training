#!/usr/bin/env python3
"""Export the completed large-dataset comparison to published/results/large_dataset/.

Reads only the campaign's final report (analysis.json), checks it against report_integrity.json, and refuses
unless the preregistered analysis is COMPLETE. Writes curated tables, a JSON summary with source hashes, and
figures. Predictions, logs, host names and GPU identifiers are not copied.

    python scripts/publish/export_large_dataset_results.py [--source-root <checkout with runs/>]
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from publish.export_results_summary import (COLORS, PRIMARY, WORKING_POINTS, paired_counts, sha256,  # noqa: E402
                                            style, write_csv)

REPORT = 'runs/eval/stage3_train8m_v1/report'
ARMS = {'fixed_lr': 'Fixed learning rate', 'windowed_pbt_v2': 'Windowed PBT'}
COMPARISONS = {'selected_best': 'Selected checkpoint (validation global best)',
               'final_current_best_member': 'Best final member', 'final_member_': 'Final member '}


def comparison_label(name):
    return COMPARISONS.get(name) or COMPARISONS['final_member_'] + name[len('final_member_'):]


def load_report(report):
    integrity = json.loads((report / 'report_integrity.json').read_text())
    recorded = {Path(a['path']).name: a['sha256'] for a in integrity['artifacts']}
    if sha256(report / 'analysis.json') != recorded.get('analysis.json'):
        raise ValueError('analysis.json differs from its recorded integrity hash')
    data = json.loads((report / 'analysis.json').read_text())
    if data['status'] != 'COMPLETE' or data['interim']:
        raise ValueError(f'The preregistered analysis is {data["status"]}; only a COMPLETE report is published')
    return data, recorded['analysis.json']


def summarize(data):
    arms = {}
    for arm, label in ARMS.items():
        s = data['summaries'][arm]
        arms[arm] = dict(label=label, final10_current_best=s['final10_current_best'], final10_epochs=s['final10_epochs'],
                         global_best=s['global_best'], final_population_mean=s['final_population_mean'],
                         final_members=s['final_members'], copies=len(s['exploit_events']),
                         optimizer_steps=s['optimizer_steps'], accepted_rows=s['accepted_rows'], amp_skipped=s['amp_skipped'],
                         worker_gpu_hours=s['worker_gpu_hours'], wall_hours=s['wall_hours'])
    test = data['sealed_test']
    checkpoints = [dict(id=c['id'], arm=c['arm'], role=c['role'], member=c.get('member'),
                        final_current_best=bool(c.get('final_current_best')), primary=test['metrics'][c['id']][PRIMARY],
                        working_points={k: test['metrics'][c['id']][k] for k in WORKING_POINTS})
                   for c in test['checkpoints']]
    comparisons = {name: dict(label=comparison_label(name), left=c['left'], right=c['right'], delta_pp=c['delta_pp'],
                              reduction_percent=c['reduction_percent'],
                              paired_background_counts={k: paired_counts(c['working_points'][k]) for k in WORKING_POINTS})
                   for name, c in test['comparisons'].items()}
    holdout = data['holdout']
    return dict(status=data['status'], determinism_confirmed=data['determinism_confirmed'],
                completed_epochs=data['completed_generations'], arms=arms, validation_comparison=data['comparison'],
                sealed_test=dict(checkpoints=checkpoints, comparisons=comparisons, population_mean=test['population_mean']),
                holdout_descriptive=dict(metrics={k: v[PRIMARY] for k, v in holdout['metrics'].items()},
                                         comparisons={k: dict(delta_pp=v['delta_pp'], reduction_percent=v['reduction_percent'])
                                                      for k, v in holdout['comparisons'].items()}),
                gpu_hours_consumed=(data.get('operations') or {}).get('gpu_hours_consumed'))


def figures(output, data, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style(plt)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (arm, label), color in zip(ARMS.items(), (COLORS['orange'], COLORS['blue'])):
        values = data['summaries'][arm]['current_best']
        ax.plot(range(1, len(values) + 1), values, color=color, label=label)
    horizon = len(data['summaries']['fixed_lr']['current_best'])
    ax.axvspan(horizon - 9.5, horizon + 0.5, color=COLORS['grid'], alpha=0.6, lw=0)
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Current-best validation mistag (%)')
    ax.set_xlim(0.5, horizon + 0.5)
    ax.legend(loc='upper right')
    fig.tight_layout()
    paths.append(output / 'figures' / 'current_best_trajectories.png')
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)

    comparisons = summary['sealed_test']['comparisons']
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(comparisons) + 1.6))
    positions = list(range(len(comparisons)))[::-1]
    ax.axvline(0, color=COLORS['muted'], lw=1)
    ax.scatter([c['delta_pp'] for c in comparisons.values()], positions, s=55, color=COLORS['blue'],
               edgecolor=COLORS['surface'], linewidth=2, zorder=3)
    ax.set_yticks(positions, [c['label'] for c in comparisons.values()])
    ax.set_xlabel('Windowed PBT - fixed learning rate on the sealed test (pp)')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    paths.append(output / 'figures' / 'sealed_test_differences.png')
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source-root', type=Path, default=PROJECT_DIR,
                        help='checkout containing the completed campaign report (read only)')
    parser.add_argument('--output', type=Path, default=PROJECT_DIR / 'published/results/large_dataset')
    args = parser.parse_args(argv)
    source, output = args.source_root.resolve(), args.output
    data, analysis_sha = load_report(source / REPORT)
    summary = summarize(data)
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    compared = summary['validation_comparison']
    write_csv(output / 'validation_comparison.csv', [
        dict(quantity=name, fixed_lr=row['fixed_lr'], windowed_pbt_v2=row['windowed_pbt_v2'], delta_pp=row['delta_pp'],
             relative_percent=row['relative_percent'], within_selection_noise_margin=row['within_selection_noise_margin'])
        for name, row in [*((k, v) for k, v in compared.items() if k not in ('final_members', 'margin_pp')),
                          *((f'final_member_{k}', v) for k, v in compared['final_members'].items())]])
    write_csv(output / 'sealed_test_checkpoints.csv', [
        dict({k: v for k, v in c.items() if k != 'working_points'},
             **{k[len('validation_'):]: v for k, v in c['working_points'].items()}) for c in summary['sealed_test']['checkpoints']])
    write_csv(output / 'sealed_test_comparisons.csv', [
        dict(comparison=name, label=c['label'], delta_pp=c['delta_pp'], reduction_percent=c['reduction_percent'])
        for name, c in summary['sealed_test']['comparisons'].items()])
    write_csv(output / 'sealed_test_paired_counts.csv', [
        dict(comparison=name, working_point=k[len('validation_'):], **counts)
        for name, c in summary['sealed_test']['comparisons'].items() for k, counts in c['paired_background_counts'].items()])
    write_csv(output / 'current_best_trajectories.csv', [
        dict(arm=arm, training_epoch=i + 1, current_best_validation=value)
        for arm in ARMS for i, value in enumerate(data['summaries'][arm]['current_best'])])
    plots = figures(output, data, summary)
    summary.update(schema_version=1, metric=PRIMARY, lower_is_better=True,
                   sources=[dict(artifact=f'{REPORT}/analysis.json', sha256=analysis_sha)],
                   outputs={str(p.relative_to(output)): sha256(p) for p in sorted(output.glob('*.csv')) + plots})
    text = json.dumps(summary, indent=2, allow_nan=False) + '\n'
    if str(source) in text or str(PROJECT_DIR) in text:
        raise ValueError('machine-specific path in the public summary')
    (output / 'summary.json').write_text(text)
    print(json.dumps(dict(status=summary['status'], output=str(output)), indent=2))


if __name__ == '__main__':
    main()
