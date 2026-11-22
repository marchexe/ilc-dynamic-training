#!/usr/bin/env python3
"""Figures and values of the train8m follow-ups for the documentation.

    .venv/bin/python scripts/reports/plot_train8m_followups.py [--output published/results/sel1m]

train8m_holdout_overview.png: holdout composite mistag of every train8m run of the standard ParT comparison
and of the prediction ensemble. overfitting.png: training and selection-set loss per epoch of the best standard
ParT member at 800k and at 8M. The values go to train8m_followups.json with their sources. Explanations belong
in the documentation text, not on the figures.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.runtime import PROJECT_DIR  # noqa: E402

PRIMARY = 'validation_total_reference_mistag_geomean_percent'
EVAL = PROJECT_DIR / 'runs/eval'
STANDARD, PBT, ENSEMBLE = '#1baf7a', '#eb6834', '#e87ba4'
# (label, campaign, checkpoint id, colour); holdout values from the frozen holdout campaigns.
HOLDOUT = [
    ('Standard ParT, start LR 1e-3', 'holdout_sel1m_train8m_v1', 'standard_sel1m_8m_s12345', STANDARD),
    ('Standard ParT, start LR 3e-3', 'holdout_sel1m_train8m_lr3e-3_v1', 'standard3e3_sel1m_8m_s12345', STANDARD),
    ('PBT', 'holdout_sel1m_train8m_v1', 'pbt_sel1m_8m_s12345', PBT),
    ('PBT + probes', 'holdout_sel1m_probe_v1', 'pbt_probe_sel1m_8m_s12345', PBT),
    ('PBT + probes + final decay', 'holdout_sel1m_probe_decay_v1', 'pbt_probe_decay_sel1m_8m_s12345', PBT),
]
LOSS_RUNS = [('800k', 'sel1m_standard_part', 'flat_decay_100e-6'),
             ('8M', 'sel1m_train8m_standard_part_3e-3', 'flat_decay_3000e-6')]
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def holdout_values():
    rows = []
    for label, campaign, ident, colour in HOLDOUT:
        result = json.loads((EVAL / campaign / 'evaluations' / ident / 'result.json').read_text())
        rows.append(dict(label=label, value=result['metrics'][PRIMARY], colour=colour,
                         source=f'runs/eval/{campaign}/evaluations/{ident}'))
    ensemble = json.loads((EVAL / 'holdout_ensemble_8m_dev_v1/result.json').read_text())
    rows.append(dict(label='Ensemble of the two standard ParT models', value=ensemble['values']['ensemble'],
                     colour=ENSEMBLE, source='runs/eval/holdout_ensemble_8m_dev_v1 '
                     '(configs/prepared/holdout_ensemble_8m_dev_v1/spec.json)'))
    return rows, ensemble['comparisons']


def losses(run, member):
    out = []
    for log in sorted((PROJECT_DIR / 'runs/pbt' / run / 'logs' / member).glob('generation-[0-9][0-9][0-9].log')):
        text = ANSI.sub('', log.read_text(errors='replace'))
        train = re.findall(r'Train AvgLoss: ([0-9.]+)', text)
        val = re.findall(r'Eval AvgLoss: ([0-9.]+)', text)
        if train and val:
            out.append(dict(epoch=18 + int(log.stem.split('-')[1]), train=float(train[-1]), selection=float(val[-1])))
    return out


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(color='#e6e6e6', linewidth=0.8)
    ax.set_axisbelow(True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=PROJECT_DIR / 'published/results/sel1m')
    args = parser.parse_args(argv)
    figures = args.output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)

    rows, comparisons = holdout_values()
    best_single = rows[1]['value']
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=200)
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [r['value'] for r in rows], color=[r['colour'] for r in rows], height=0.62)
    for yi, r in zip(y, rows):
        change = 100 * (r['value'] / best_single - 1)
        note = '' if r is rows[1] else f'  ({change:+.1f}%)'
        ax.text(r['value'] + 0.0007, yi, f"{r['value']:.3f}{note}", va='center', fontsize=9, color='#333333',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.set_yticks(y, [r['label'] for r in rows], fontsize=9)
    ax.axvline(best_single, color='#888888', linewidth=1, linestyle='--', zorder=0)
    ax.set_xlim(0.14, 0.19)
    ax.set_xlabel('holdout mistag, geometric mean (%) — lower is better')
    ax.set_title('8M training jets, holdout (2.8M jets)', fontsize=11)
    style(ax)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(figures / 'train8m_holdout_overview.png')
    plt.close(fig)

    curves = {name: losses(run, member) for name, run, member in LOSS_RUNS}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=200, sharey=False)
    for ax, (name, run, member) in zip(axes, LOSS_RUNS):
        c = curves[name]
        epochs = [p['epoch'] for p in c]
        ax.plot(epochs, [p['train'] for p in c], color='#555555', linewidth=2, label='training data')
        ax.plot(epochs, [p['selection'] for p in c], color=STANDARD, linewidth=2, label='selection set (unseen)')
        ax.set_title(f'{name} training jets', fontsize=11)
        ax.set_xlabel('weaver epoch')
        style(ax)
    axes[0].set_ylabel('loss (cross-entropy)')
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / 'overfitting.png')
    plt.close(fig)

    summary = dict(schema_version=1, metric=PRIMARY, holdout=[{k: v for k, v in r.items() if k != 'colour'} for r in rows],
                   ensemble_comparisons=comparisons,
                   loss_curves={name: dict(run=run, member=member, epochs=curves[name])
                                for name, run, member in LOSS_RUNS})
    (args.output / 'train8m_followups.json').write_text(json.dumps(summary, indent=1) + '\n')
    for r in rows:
        print(f"{r['label']:45s} {r['value']:.4f}")


if __name__ == '__main__':
    main()
