#!/usr/bin/env python3
"""Presentation figures from completed, matched full-epoch manifests.

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
    window_epochs = int(manifest['config']['pbt']['windowed_pbt_v2']['window_epochs'])
    if len(rows) % window_epochs:
        raise ValueError('Training horizon is not an integer number of PBT windows')
    pbt_generations = len(rows) // window_epochs
    generation_ranges = [
        dict(generation=i + 1, start_epoch=i * window_epochs + 1, end_epoch=(i + 1) * window_epochs)
        for i in range(pbt_generations)
    ]
    decisions = [g['windowed_pbt_v2'] for g in rows if g.get('windowed_pbt_v2')]
    if len(decisions) != pbt_generations:
        raise ValueError('PBT boundary evidence is incomplete')
    for expected, decision in zip(generation_ranges, decisions):
        if (decision['window_number'] != expected['generation'] or
                decision['full_epochs'] != list(range(expected['start_epoch'], expected['end_epoch'] + 1))):
            raise ValueError('PBT generation-to-epoch mapping is inconsistent')
    events = []
    for epoch, g in zip(epochs, rows):
        for event in g.get('exploit', []):
            if not event.get('applied'):
                raise ValueError('Unapplied copy in completed run')
            n = event['recipient']
            if epoch >= len(rows) or lrs[n][epoch] != event['new_lr']:
                raise ValueError('Copy does not match next epoch LR')
            state_hashes = [event[key]['state']['sha256']
                            for key in ('donor_checkpoint', 'copied_checkpoint', 'post_copy')]
            if len(set(state_hashes)) != 1:
                raise ValueError('Recipient post-copy weights do not match donor weights')
            if event['pre_copy']['state']['sha256'] == state_hashes[0]:
                raise ValueError('Recipient pre-copy weights unexpectedly match donor weights')
            events.append(dict(after_epoch=epoch, active_from_epoch=epoch + 1,
                               donor=event['donor'], recipient=n, old_lr=event['recipient_lr'],
                               new_lr=event['new_lr'], factor=event['mutation_factor'],
                               pbt_generation=g['windowed_pbt_v2']['window_number'],
                               copied_state_sha256=state_hashes[0], copy_state_verified=True,
                               post_copy_evaluation_recorded=False))
    winner = min(names, key=lambda n: mean(values[n][-10:]))
    final_member = min(names, key=lambda n: values[n][-1])
    best_recorded_member, best_recorded_index = min(
        ((name, index) for name in names for index in range(len(epochs))),
        key=lambda item: values[item[0]][item[1]],
    )
    best_recorded_epoch = epochs[best_recorded_index]
    best_recorded_value = values[best_recorded_member][best_recorded_index]
    protected_member = manifest['protected_best']['member']
    comparisons = {}
    for label, count in [('final10', 10), ('final5', 5), ('final_checkpoint', 1)]:
        pbt, ref = mean(values[winner][-count:]), mean(fixed[-count:])
        comparisons[label] = dict(pbt=pbt, fixed=ref, improvement_pp=ref - pbt,
                                  reduction_percent=100 * (ref - pbt) / ref)
    sources = [source, *[Path(p) / 'manifest.json' for p in baseline_prefixes], control_source]
    return dict(sources=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in sources],
                epochs=epochs, identities=identities, values=values, lrs=lrs, fixed=fixed, fixed_lr=fixed_lr,
                events=events, winner=winner, final_member=final_member,
                best_recorded_member=best_recorded_member,
                best_recorded_epoch=best_recorded_epoch,
                best_recorded_value=best_recorded_value,
                protected_member=protected_member, comparisons=comparisons,
                initial_max_lr=max(m['start_lr'] for m in manifest['config']['population']),
                window_epochs=window_epochs, pbt_generations=pbt_generations,
                generation_ranges=generation_ranges)


def colors(data):
    palette = [CB_PALETTE[k] for k in ('grey', 'blue', 'green', 'vermillion', 'purple')]
    return {n: palette[i % len(palette)] for i, n in enumerate(data['identities'])}


def member_label(data, name):
    return f"Member {data['identities'][name]} ({data['lrs'][name][-1] * 1e6:g}e-6 final)"


def title(fig, heading, subtitle):
    fig.text(.08, .94, heading, fontsize=22, fontweight='bold', color='#172B3A')
    fig.text(.08, .895, subtitle, fontsize=11.5, color='#53616B')


def events_by_epoch(data):
    return [(epoch, [event for event in data['events'] if event['after_epoch'] == epoch])
            for epoch in sorted({event['after_epoch'] for event in data['events']})]


def branch_label(data, events):
    lines = []
    for event in events:
        recipient = f"M{data['identities'][event['recipient']]}"
        mapping = f"M{data['identities'][event['donor']]}→{recipient}"
        lines.append(f"{mapping}  ×{event['factor']:g}")
    return '\n'.join(lines)


def linked_epoch_axis(ax, data):
    """Apply the shared epoch scale used by Figures 1–2."""
    horizon = data['epochs'][-1]
    ax.set_xlim(0, horizon)
    ax.set_xticks(range(0, horizon + 1, 10))
    ax.set_xlabel('Training epoch')
    ax.grid(axis='y', color='#E5E9EC', lw=.6)


def progression(plt, data):
    """Figure 1: performance lineage with copies branching from donor states."""
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13.2, 7.5))
    ax = fig.add_axes([.08, .22, .88, .57])
    palette = colors(data)
    highlighted = data['best_recorded_member']
    horizon = data['epochs'][-1]
    linked_epoch_axis(ax, data)
    late_start = data['epochs'][-10]
    ax.axvspan(late_start - .5, horizon, color=palette[data['winner']], alpha=.055, zorder=0)
    copy_marker_handles = []
    grouped_events = events_by_epoch(data)
    event_order = {epoch: group for epoch, group in grouped_events}
    for name, values in data['values'].items():
        leading = name == highlighted
        copies = sorted((event for event in data['events'] if event['recipient'] == name),
                        key=lambda event: event['after_epoch'])
        start = 0
        label = f"Member {data['identities'][name]}"
        for event in [*copies, None]:
            stop = event['after_epoch'] if event else len(data['epochs'])
            if stop > start:
                ax.plot(data['epochs'][start:stop], values[start:stop], color=palette[name],
                        lw=2.9 if leading else 1.15, alpha=1 if leading else .48,
                        zorder=4 if leading else 2, label=label)
                label = None
            if event:
                epoch = event['after_epoch']
                donor_value = data['values'][event['donor']][epoch - 1]
                # The copied state exists after selection at epoch E; the next
                # actual recipient validation is epoch E+1.
                if epoch < horizon:
                    ax.plot([epoch, epoch + 1], [donor_value, values[epoch]],
                            color=palette[name], lw=2.9 if leading else 1.4,
                            alpha=1 if leading else .72, zorder=5)
                siblings = event_order[epoch]
                order = siblings.index(event)
                ax.scatter(epoch, donor_value, marker='s', s=52 + order * 30,
                           facecolors='white', edgecolors=palette[name],
                           linewidths=1.35, zorder=8)
            start = stop
    ax.plot(data['epochs'], data['fixed'], color='#26343C', ls=(0, (5, 3)), lw=2.1,
            zorder=5, label=f"Fixed {data['fixed_lr'] * 1e6:g}e-6 baseline")
    best_epoch = data['best_recorded_epoch']
    best_value = data['best_recorded_value']
    ax.scatter(best_epoch, best_value, marker='*', s=180, color=palette[highlighted],
               edgecolors='white', linewidths=1.0, zorder=8, clip_on=False)
    comparison = data['comparisons']['final10']
    ax.text(.985, .96,
            f"FINAL 10 EPOCHS  ({late_start}–{horizon})\n"
            f"M{data['identities'][data['winner']]} late-window mean:  {comparison['pbt']:.6f}%\n"
            f"Fixed baseline mean:  {comparison['fixed']:.6f}%\n"
            f"{comparison['reduction_percent']:.2f}% relative mistag reduction",
            transform=ax.transAxes, ha='right', va='top', fontsize=10.5, linespacing=1.45,
            color='#26343C', bbox=dict(boxstyle='round,pad=.55', fc='white', ec='#D6DFE5'))
    ax.set_ylabel('Composite mistag (%)')
    handles, labels = ax.get_legend_handles_labels()
    copy_marker_handles.append(Line2D([], [], marker='s', linestyle='none', markersize=6,
                                      markerfacecolor='white', markeredgecolor='#44525B',
                                      label='Copied checkpoint'))
    fig.legend([*handles, *copy_marker_handles], [*labels, 'Copied checkpoint'],
               loc='lower left', bbox_to_anchor=(.075, .105), ncol=7,
               frameon=False, fontsize=9.5, handlelength=2.4, columnspacing=1.15)
    title(fig, 'Performance lineage',
          'Validation outcome versus the matched fixed 14e-6 continuation control')
    fig.text(.08, .055,
             f"★ Global best: Member {data['identities'][highlighted]} at E{best_epoch} "
             f"({best_value:.6f}%).  Shaded: final-10 comparison window.",
             fontsize=9.5, color=palette[highlighted])
    fig.text(.08, .025,
             '□ copied donor checkpoint (not a validation); mutation details are shown in Figure 2.',
             fontsize=9, color='#53616B')
    return fig


def learning_rates(plt, data):
    """Figure 2: LR lineage with recipients branching from donor states."""
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(13.2, 7.5))
    ax = fig.add_axes([.08, .22, .88, .57])
    palette = colors(data)
    maximum = max(v * 1e6 for ys in data['lrs'].values() for v in ys)
    linked_epoch_axis(ax, data)
    grouped_events = events_by_epoch(data)
    event_order = {epoch: group for epoch, group in grouped_events}
    for name, lrs in data['lrs'].items():
        copies = sorted((event for event in data['events'] if event['recipient'] == name),
                        key=lambda event: event['after_epoch'])
        starts = [0, *[event['after_epoch'] for event in copies]]
        stops = [*[event['after_epoch'] for event in copies], data['epochs'][-1]]
        levels = [lrs[0] * 1e6, *[event['new_lr'] * 1e6 for event in copies]]
        label = f"Member {data['identities'][name]}"
        for start, stop, level in zip(starts, stops, levels):
            ax.plot([start, stop], [level, level], color=palette[name], lw=2.5,
                    zorder=3, label=label)
            label = None
        for event in copies:
            epoch = event['after_epoch']
            donor_lr = data['lrs'][event['donor']][epoch - 1] * 1e6
            new_lr = event['new_lr'] * 1e6
            ax.plot([epoch, epoch], [donor_lr, new_lr], color=palette[name], lw=1.7, zorder=5)
            siblings = event_order[epoch]
            order = siblings.index(event)
            ax.scatter(epoch, donor_lr, marker='s', s=52 + order * 30,
                       facecolors='white', edgecolors=palette[name],
                       linewidths=1.35, zorder=7)
            ax.scatter(epoch, new_lr, marker='D', s=43, color=palette[name],
                       edgecolors='white', linewidths=.8, zorder=8)
    for index, (epoch, events) in enumerate(grouped_events):
        donor_lrs = [data['lrs'][event['donor']][epoch - 1] * 1e6 for event in events]
        new_lrs = [event['new_lr'] * 1e6 for event in events]
        anchor = (sum(donor_lrs) + sum(new_lrs)) / (len(donor_lrs) + len(new_lrs))
        near_right = epoch >= data['epochs'][-1] - 5
        dy = -22 if index % 2 == 0 else 18
        ax.annotate(branch_label(data, events), (epoch, anchor),
                    xytext=(-5 if near_right else 5, dy), textcoords='offset points',
                    ha='right' if near_right else 'left',
                    va='top' if dy < 0 else 'bottom', fontsize=7.8,
                    color='#44525B', linespacing=1.35,
                    bbox=dict(boxstyle='round,pad=.18', fc='white', ec='none', alpha=.84),
                    zorder=9)
    ax.set_ylim(0, maximum * 1.18)
    ax.set_ylabel('Learning rate (×10⁻⁶)')
    handles, labels = ax.get_legend_handles_labels()
    copy_handle = Line2D([], [], marker='s', linestyle='none', markersize=6,
                         markerfacecolor='white', markeredgecolor='#44525B')
    mutation_handle = Line2D([], [], marker='D', linestyle='none', markersize=5,
                             markerfacecolor='#44525B', markeredgecolor='white')
    fig.legend([*handles, copy_handle, mutation_handle],
               [*labels, 'Copied checkpoint', 'Mutated LR'],
               loc='lower left', bbox_to_anchor=(.075, .105), ncol=7,
               frameon=False, fontsize=9.5, handlelength=2.4, columnspacing=1.15)
    title(fig, 'Learning-rate lineage',
          'Each recipient branch originates on its donor, then moves to the mutated LR')
    fig.text(.08, .055,
             'Labels give donor→recipient and mutation factor; vertical position gives the resulting LR.',
             fontsize=9.5, color='#53616B')
    fig.text(.08, .025,
             '□ copied donor checkpoint  ·  ◆ mutated LR used from the next training epoch.',
             fontsize=9, color='#53616B')
    return fig


def export_figures(run, baseline, baseline_member, output_dir=None, baseline_prefixes=()):
    run = Path(run).resolve()
    run_dir = run if run.is_dir() else run.parent
    output = Path(output_dir).resolve() if output_dir else run_dir / 'plots'
    figures = [('01_performance_progression', progression),
               ('02_learning_rate_evolution', learning_rates)]
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
