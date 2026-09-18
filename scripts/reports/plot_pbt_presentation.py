#!/usr/bin/env python3
"""Three presentation figures from completed, matched full-epoch manifests.

Reads recorded results only. Writes PNGs to the candidate run's plots directory
by default, without invoking training, evaluation, or canonical reporting.
"""
import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports.export_research_result import load_manifest
from training.pbt.reporting.constants import CB_PALETTE
from training.pbt.reporting.style import plot_setup
from training.runtime import sha256
from validation.verify_fixed_lr import load_control


def presentation_data(run, baseline, baseline_member, baseline_prefixes=()):
    manifest, source = load_manifest(run)
    control, control_source = load_manifest(baseline)
    if baseline_prefixes:
        control = load_control(control_source.parent, baseline_prefixes)
    for m in (manifest, control):
        if m['status'] != 'completed' or m['config']['shared']['weaver_epochs_per_generation'] != 1:
            raise ValueError('Presentation requires completed full-epoch runs')
    for key in ('state_sha256', 'optimizer_sha256'):
        if manifest['initial_resume'][key] != control['initial_resume'][key]:
            raise ValueError('Initial checkpoints do not match')
    if manifest['initial_evaluation']['metrics']['validation_data_audit']['consumed'] != control['initial_evaluation']['metrics']['validation_data_audit']['consumed']:
        raise ValueError('Validation events do not match')
    rows = sorted(manifest['generations'], key=lambda g: g['epoch'])
    reference = sorted(control['generations'], key=lambda g: g['epoch'])
    epochs = [g['epoch'] - manifest['config']['shared']['initial_epoch'] for g in rows]
    if epochs != list(range(1, len(rows) + 1)) or len(reference) < len(rows) or len(rows) < 10:
        raise ValueError('Need at least ten contiguous full epochs and a matched control horizon')
    reference = reference[:len(rows)]
    if any(g['status'] != 'completed' for g in rows + reference):
        raise ValueError('Incomplete epoch evidence')
    # Number identities once by initial LR, highest first; never relabel after a copy.
    names = [m['name'] for m in sorted(manifest['config']['population'], key=lambda m: (-m['start_lr'], m['name']))]
    identities = {name: i + 1 for i, name in enumerate(names)}
    metric = manifest['config']['pbt']['metric']
    if metric != control['config']['pbt']['metric']:
        raise ValueError('Metric definitions differ')
    values = {n: [g['workers'][n]['metrics'][metric] for g in rows] for n in names}
    lrs = {n: [g['workers'][n]['lr'] for g in rows] for n in names}
    fixed = [g['workers'][baseline_member]['metrics'][metric] for g in reference]
    fixed_lr = reference[0]['workers'][baseline_member]['lr']
    if any(g['workers'][baseline_member]['lr'] != fixed_lr for g in reference):
        raise ValueError('Reference member must have a fixed LR')
    if any(not math.isfinite(v) for ys in [fixed, *values.values(), *lrs.values()] for v in ys):
        raise ValueError('Non-finite plot data')
    events = []
    for epoch, g in zip(epochs, rows):
        for event in g.get('exploit', []):
            if not event.get('applied'):
                raise ValueError('Unapplied copy in completed run')
            n = event['recipient']
            if epoch >= len(rows) or lrs[n][epoch] != event['new_lr']:
                raise ValueError('Copy does not match next epoch LR')
            events.append(dict(after_epoch=epoch, active_from_epoch=epoch + 1,
                               donor=event['donor'], recipient=n, old_lr=event['recipient_lr'],
                               new_lr=event['new_lr'], factor=event['mutation_factor']))
    winner = min(names, key=lambda n: mean(values[n][-10:]))
    protected_member = manifest['protected_best']['member']
    comparisons = {}
    for label, count in [('final10', 10), ('final5', 5), ('final_checkpoint', 1)]:
        pbt, ref = mean(values[winner][-count:]), mean(fixed[-count:])
        comparisons[label] = dict(pbt=pbt, fixed=ref, improvement_pp=ref - pbt,
                                  reduction_percent=100 * (ref - pbt) / ref)
    sources = [source, *[Path(p) / 'manifest.json' for p in baseline_prefixes], control_source]
    return dict(sources=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in sources],
                epochs=epochs, identities=identities, values=values, lrs=lrs, fixed=fixed, fixed_lr=fixed_lr,
                events=events, winner=winner, protected_member=protected_member, comparisons=comparisons,
                initial_max_lr=max(m['start_lr'] for m in manifest['config']['population']))


def colors(data):
    palette = [CB_PALETTE[k] for k in ('grey', 'blue', 'green', 'vermillion', 'purple')]
    return {n: palette[i % len(palette)] for i, n in enumerate(data['identities'])}


def member_label(data, name):
    return f"Member {data['identities'][name]} ({data['lrs'][name][-1] * 1e6:g}e-6 final)"


def title(fig, heading, subtitle):
    fig.text(.08, .94, heading, fontsize=22, fontweight='bold', color='#172B3A')
    fig.text(.08, .895, subtitle, fontsize=11.5, color='#53616B')


def progression(plt, data):
    fig = plt.figure(figsize=(13.2, 7.5))
    ax = fig.add_axes([.08, .27, .88, .52])
    palette = colors(data)
    winner = data['winner']
    for name, values in data['values'].items():
        leading = name == winner
        ax.plot(data['epochs'], values, color=palette[name], lw=2.7 if leading else 1.1,
                alpha=1 if leading else .40, zorder=4 if leading else 2, label=member_label(data, name))
    ax.plot(data['epochs'], data['fixed'], color='#26343C', ls=(0, (5, 3)), lw=2.1,
            zorder=5, label=f"Fixed {data['fixed_lr'] * 1e6:g}e-6 baseline")
    boundaries = sorted({e['after_epoch'] for e in data['events']})
    for i, epoch in enumerate(boundaries):
        ax.axvline(epoch, color='#78848D', lw=.7, ls=':', alpha=.65, zorder=1)
        group = [e for e in data['events'] if e['after_epoch'] == epoch]
        note = '\n'.join(f"M{data['identities'][e['donor']]} → M{data['identities'][e['recipient']]}: {e['new_lr'] * 1e6:g}e-6" for e in group)
        ax.text(epoch, 1.035 + .10 * (i % 2), note, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, color='#44525B', clip_on=False)
    late = data['epochs'][-10]
    ax.axvspan(late - .5, data['epochs'][-1], color=palette[winner], alpha=.055, zorder=0)
    c = data['comparisons']['final10']
    ax.text(.98, .91, f"FINAL 10 EPOCHS\nPBT {c['pbt']:.6f}%  ·  Fixed {c['fixed']:.6f}%\n"
            f"{c['reduction_percent']:.2f}% relative mistag reduction", transform=ax.transAxes,
            ha='right', va='top', fontsize=12, linespacing=1.6,
            bbox=dict(boxstyle='round,pad=.65', fc='white', ec='#D6DFE5'))
    ax.set(xlim=(1, data['epochs'][-1]), xlabel='Full epoch', ylabel='Composite mistag (%)  ↓ lower is better')
    ax.set_xticks([1, *range(10, data['epochs'][-1] + 1, 10)])
    ax.grid(axis='y', color='#E5E9EC', lw=.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.075, .105), ncol=3,
               frameon=False, fontsize=10.5, handlelength=3, columnspacing=2)
    title(fig, 'Adaptive training improves the late-epoch result',
          'Five persistent PBT members against the matched fixed-LR control')
    role = 'final protected-best member' if winner == data['protected_member'] else 'best final-10 mean'
    fig.text(.08, .055, f"Highlight: Member {data['identities'][winner]}, {role}. Actual epoch values; no winner-envelope stitching.",
             fontsize=10, color='#53616B')
    overlap = [f"Member {data['identities'][n]}" for n, ys in data['values'].items() if ys == data['fixed']]
    overlap_note = (' ' + ', '.join(overlap) + ' overlaps the fixed control.') if overlap else ''
    fig.text(.08, .025, 'Dotted lines mark actual copy boundaries; labels give donor → recipient and new LR.' + overlap_note,
             fontsize=9, color='#53616B')
    return fig


def learning_rates(plt, data):
    fig = plt.figure(figsize=(13.2, 7.5))
    ax = fig.add_axes([.08, .19, .67, .63])
    palette = colors(data)
    ceiling = data['initial_max_lr'] * 1e6
    maximum = max(v * 1e6 for ys in data['lrs'].values() for v in ys)
    ax.axhspan(ceiling, maximum * 1.18, color=CB_PALETTE['blue'], alpha=.04)
    for name, lrs in data['lrs'].items():
        # Epoch k's LR occupies (k-1, k]; a boundary after k changes epoch k+1.
        ax.stairs([v * 1e6 for v in lrs], [0, *data['epochs']], baseline=None,
                  color=palette[name], lw=2.4, zorder=3)
        initial = lrs[0] * 1e6
        ax.text(.7, initial + (-.75 if initial == ceiling else .35),
                f"M{data['identities'][name]}: {initial:g}", color=palette[name], fontsize=9.5)
    ax.axhline(ceiling, color='#26343C', ls=(0, (5, 3)), lw=1.25, zorder=4)
    ax.text(.012, ceiling + .5, f"Initial maximum: {ceiling:g}e-6", transform=ax.get_yaxis_transform(),
            fontsize=10, color='#26343C')
    for event in data['events']:
        n = event['recipient']; x = event['after_epoch']; y = event['new_lr'] * 1e6
        ax.scatter(x, y, s=40, color=palette[n], edgecolors='white', linewidths=.8, zorder=6)
        ax.annotate(f"M{data['identities'][event['donor']]} → M{data['identities'][n]} · {y:g}", (x, y),
                    xytext=(6, 11 if event['factor'] > 1 else -18), textcoords='offset points',
                    fontsize=9, color=palette[n], fontweight='bold')
    position = -math.inf
    for name in sorted(data['lrs'], key=lambda n: data['lrs'][n][-1]):
        y = data['lrs'][name][-1] * 1e6
        position = max(y, position + maximum * .075)
        ax.annotate(member_label(data, name), (data['epochs'][-1], y), xycoords='data',
                    xytext=(1.035, position), textcoords=ax.get_yaxis_transform(),
                    va='center', fontsize=10.5, color=palette[name], annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=palette[name], lw=.8))
    ax.set(xlim=(0, data['epochs'][-1]), ylim=(0, maximum * 1.18), xlabel='Full epoch',
           ylabel='Learning rate (×10⁻⁶)')
    ax.grid(axis='y', color='#E5E9EC', lw=.6)
    title(fig, 'PBT discovers higher learning rates automatically',
          'Persistent member identities; each step uses the LR actually recorded for that epoch')
    first = data['events'][0]['after_epoch'] if data['events'] else None
    timing = f' A change after epoch {first} first trains at the new LR in epoch {first + 1}.' if first else ''
    fig.text(.08, .105, 'Dots mark copies and LR mutations only.' + timing,
             fontsize=10, color='#53616B')
    above = sorted({e['new_lr'] * 1e6 for e in data['events'] if e['new_lr'] > data['initial_max_lr']})
    fig.text(.08, .055, 'Discovered above the initial range: ' + ' / '.join(f'{v:g}e-6' for v in above) + '.',
             fontsize=13, fontweight='bold', color=CB_PALETTE['blue'])
    return fig


def final_comparison(plt, data):
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    fig = plt.figure(figsize=(11.5, 7))
    ax = fig.add_axes([.29, .47, .64, .29])
    c = data['comparisons']['final10']; accent = colors(data)[data['winner']]
    gap = abs(c['fixed'] - c['pbt']) or c['fixed'] * .01
    ax.scatter([c['fixed'], c['pbt']], [1, 0], s=[110, 150], color=['#26343C', accent], zorder=3)
    for key, y, color in [('fixed', 1, '#26343C'), ('pbt', 0, accent)]:
        ax.annotate(f"{c[key]:.6f}%", (c[key], y), xytext=(0, 14), textcoords='offset points',
                    ha='center', fontsize=15, fontweight='bold', color=color)
    ax.set(yticks=[0, 1], yticklabels=[f"PBT Member {data['identities'][data['winner']]}\n({data['lrs'][data['winner']][-1] * 1e6:g}e-6 final)",
                                    f"Fixed {data['fixed_lr'] * 1e6:g}e-6"], ylim=(-.5, 1.7),
           xlim=(min(c['pbt'], c['fixed']) - gap * .6, max(c['pbt'], c['fixed']) + gap * .6),
           xlabel='Final-10 mean composite mistag (%)  ·  lower is better')
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=15)
    ax.grid(axis='x', color='#E5E9EC', lw=.7)
    title(fig, 'Lower mistag, sustained over the final 10 epochs',
          f"Primary comparison: mean over full epochs {data['epochs'][-10]}–{data['epochs'][-1]}")
    fig.text(.29, .35, f"{c['improvement_pp']:.6f} percentage points", fontsize=15, fontweight='bold', color=accent)
    fig.text(.29, .31, 'Absolute improvement', fontsize=10, color='#53616B')
    fig.text(.74, .35, f"{c['reduction_percent']:.2f}%", fontsize=24, fontweight='bold', color=accent)
    fig.text(.74, .31, 'Relative mistag reduction', fontsize=10, color='#53616B')
    for y, label, key in [(.20, 'Final-5 mean', 'final5'), (.15, 'Final checkpoint', 'final_checkpoint')]:
        item = data['comparisons'][key]
        fig.text(.29, y, f"{label}:  {item['fixed']:.6f}% → {item['pbt']:.6f}%   ({item['reduction_percent']:.2f}% reduction)",
                 fontsize=10.5, color='#53616B')
    fig.text(.08, .055, 'Points use a zoomed, labeled axis—not bar lengths. Both runs use the same validation events and training horizon.',
             fontsize=9.5, color='#53616B')
    return fig


def export_figures(run, baseline, baseline_member, output_dir=None, baseline_prefixes=()):
    run = Path(run).resolve()
    run_dir = run if run.is_dir() else run.parent
    output = Path(output_dir).resolve() if output_dir else run_dir / 'plots'
    figures = [('01_performance_progression', progression), ('02_learning_rate_evolution', learning_rates),
               ('03_final_result_comparison', final_comparison)]
    targets = [output / f'{stem}.png' for stem, _ in figures] + [output / 'presentation_source_values.json']
    if any(path.exists() for path in targets):
        raise FileExistsError(f'Refusing to overwrite presentation files in: {output}')
    data = presentation_data(run, baseline, baseline_member, baseline_prefixes)
    plt = plot_setup()
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    output.mkdir(parents=True, exist_ok=True)
    for stem, draw in figures:
        fig = draw(plt, data)
        fig.savefig(output / f'{stem}.png', dpi=300)
        plt.close(fig)
    targets[-1].write_text(json.dumps(data, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-member', required=True)
    parser.add_argument('--baseline-prefix', type=Path, action='append', default=[],
                        help='Earlier fixed-LR control segment; repeat in chronological order')
    parser.add_argument('--output-dir', type=Path, help='Default: <run>/plots')
    args = parser.parse_args()
    print(export_figures(args.run, args.baseline, args.baseline_member, args.output_dir, args.baseline_prefix))


if __name__ == '__main__':
    main()
