#!/usr/bin/env python3
"""Standard report figures for the train800k learning-rate-level follow-ups.

    .venv/bin/python scripts/reports/plot_lr_level_followups.py [--output DIR]

01-05 are the Stage 3 report figures (stage3.analysis.figures) for the primary comparison of
holdout_fast_pbt_v1: arm A is the selected fixed-LR grid, arm B the fast windowed PBT, and 05 shows the
per-working-point change on the holdout. 06 shows the holdout metric of every train800k run in the series.
Explanations belong in the report text, not on the figures.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.stage3.analysis import ARM_COLOR, arm_summary, figures  # noqa: E402
from training.runtime import PROJECT_DIR  # noqa: E402

PRIMARY = 'validation_total_reference_mistag_geomean_percent'
EVAL = PROJECT_DIR / 'runs/eval'
FIXED, PBT = 'fixed_lr_top_grid_50epochs', 'windowed_pbt_v2_fast'
CAMPAIGN, PAIR = 'holdout_fast_pbt_v1', 'windowed_fast50_s12345_minus_fixed_top50_s12345'
# (label, holdout campaign, checkpoint id, kind) in the order the runs were made.
RUNS = [
    ('fixed 3e-6 to 1.4e-5', 'stage2a_holdout_v1', 'fixed50_s12345', 'fixed_lr'),
    ('PBT, original', 'stage2a_holdout_v1', 'windowed50_s12345', 'windowed_pbt_v2'),
    ('fixed 1.4e-5 to 3e-5', 'holdout_high_lr_grid_v1', 'fixed_high50_s12345', 'fixed_lr'),
    ('fixed 3e-5 to 8e-5', 'holdout_fixed_lr_ceiling_v1', 'fixed_ceiling50_s12345', 'fixed_lr'),
    ('fixed 8e-5 to 3.2e-4', 'holdout_fast_pbt_v1', 'fixed_top50_s12345', 'fixed_lr'),
    ('PBT, fast', 'holdout_fast_pbt_v1', 'windowed_fast50_s12345', 'windowed_pbt_v2'),
]


def load(path):
    return json.loads(Path(path).read_text())


def holdout_metric(campaign, checkpoint):
    if campaign == 'stage2a_holdout_v1':
        return load(EVAL / campaign / 'evaluations' / checkpoint / 'result.json')['metrics'][PRIMARY]
    return load(EVAL / campaign / 'analysis.json')['metrics'][checkpoint][PRIMARY]


def lollipop(rows, title, path):
    """One row per run (label, value, color), the style of the earlier holdout figures."""
    from training.pbt.reporting.style import plot_setup
    plt = plot_setup()
    values = [value for _, value, _ in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(rows) + 0.6))
    for y, (label, value, color) in enumerate(rows):
        ax.hlines(y, min(values) - 0.004, value, color='#ebeae6', lw=2)
        ax.plot(value, y, 'o', ms=9, color=color, mec='white', mew=1.5)
        ax.annotate(f'{value:.4f}', (value, y), xytext=(7, 0), textcoords='offset points', va='center',
                    fontsize=8, color='#52514e')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(min(values) - 0.004, max(values) + 0.006)
    ax.set(xlabel='holdout primary mistag (%), lower is better', title=title)
    ax.grid(axis='x', color='#ebeae6', lw=0.8)
    ax.tick_params(axis='y', length=0)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def overview(directory):
    rows = [(label, holdout_metric(campaign, checkpoint), ARM_COLOR[kind]) for label, campaign, checkpoint, kind in RUNS]
    return lollipop(rows, 'train800k runs, seed 12345: holdout', Path(directory) / '06_holdout_all_runs.png')


# (seed, fixed-LR run, its holdout id, PBT run, its holdout id, holdout campaign)
SEEDS = [
    ('seed 12345', 'fixed_lr_top_grid_50epochs', 'fixed_top50_s12345', 'windowed_pbt_v2_fast', 'windowed_fast50_s12345',
     'holdout_fast_pbt_seed22345_v1'),
    ('seed 22345', 'fixed_lr_best_grid_50epochs_seed22345', 'fixed_best50_s22345', 'windowed_pbt_v2_fast_seed22345',
     'windowed_fast50_s22345', 'holdout_fast_pbt_seed22345_v1'),
]


def validation_best(run):
    return load(PROJECT_DIR / 'runs/pbt' / run / 'manifest.json')['best']['metrics'][PRIMARY]


def seeds_overview(directory):
    """Fast PBT against the fixed-LR grid for both seeds: holdout scores, and PBT-vs-fixed on validation and holdout."""
    from training.pbt.reporting.style import plot_setup
    plt = plot_setup()
    rows = []
    for seed, fixed, fixed_id, pbt, pbt_id, campaign in SEEDS:
        metrics = load(EVAL / campaign / 'analysis.json')['metrics']
        hf, hp = metrics[fixed_id][PRIMARY], metrics[pbt_id][PRIMARY]
        vf, vp = validation_best(fixed), validation_best(pbt)
        rows.append(dict(seed=seed, hf=hf, hp=hp, dv=100 * (vp / vf - 1), dh=100 * (hp / hf - 1)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for y, r in enumerate(rows):
        left.hlines(y, min(r['hf'], r['hp']), max(r['hf'], r['hp']), color='#c3c2bd', lw=2, zorder=1)
        for value, kind, label in ((r['hf'], 'fixed_lr', 'fixed LR'), (r['hp'], 'windowed_pbt_v2', 'fast PBT')):
            left.plot(value, y, 'o', ms=9, color=ARM_COLOR[kind], mec='white', mew=1.5, zorder=2,
                      label=label if y == 0 else None)
            left.annotate(f'{value:.4f}', (value, y), xytext=(0, 9), textcoords='offset points', ha='center',
                          fontsize=8, color='#52514e')
    left.set_yticks(range(len(rows)), [r['seed'] for r in rows])
    left.set_ylim(len(rows) - 0.4, -0.6)
    lo = min(min(r['hf'], r['hp']) for r in rows); hi = max(max(r['hf'], r['hp']) for r in rows)
    left.set_xlim(lo - 0.0012, hi + 0.0012)
    left.set(xlabel='holdout primary mistag (%), lower is better', title='Holdout score')
    left.legend(loc='lower right', frameon=False, fontsize=8)
    right.axvline(0, color='#8a8984', lw=1)
    for y, r in enumerate(rows):
        right.hlines(y, min(r['dv'], r['dh']), max(r['dv'], r['dh']), color='#c3c2bd', lw=2, zorder=1)
        for value, marker, color, label in ((r['dv'], 's', '#8a8984', 'validation (150k)'),
                                            (r['dh'], 'o', '#2b2a27', 'holdout (2.8M)')):
            right.plot(value, y, marker, ms=8, color=color, mec='white', mew=1.5, zorder=2, label=label if y == 0 else None)
            right.annotate(f'{value:+.1f}%', (value, y), xytext=(0, 9), textcoords='offset points', ha='center',
                           fontsize=8, color='#52514e')
    right.set_yticks(range(len(rows)), [r['seed'] for r in rows])
    right.set_ylim(len(rows) - 0.4, -0.6)
    span = max(abs(v) for r in rows for v in (r['dv'], r['dh'])) + 0.5
    right.set_xlim(-span, span)
    right.set(xlabel='fast PBT vs fixed LR (%), negative = PBT better', title='PBT vs fixed LR: validation and holdout')
    right.legend(loc='lower right', frameon=False, fontsize=8)
    for ax in (left, right):
        ax.grid(axis='x', color='#ebeae6', lw=0.8)
        ax.tick_params(axis='y', length=0)
    fig.tight_layout(w_pad=2.5)
    path = Path(directory) / '07_two_seeds.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


# (seed label, holdout campaign, checkpoint suffix); a seed appears once its campaign has an analysis.
SEL1M_SEEDS = [('seed 12345', 'holdout_sel1m_v1', 's12345'), ('seed 22345', 'holdout_sel1m_v1', 's22345'),
               ('seed 32345', 'holdout_sel1m_seed32345_v1', 's32345')]
SEL1M_ARMS = [('standard', 'standard ParT', '#1baf7a'), ('fixed', 'fixed 6e-5 to 1.6e-4', ARM_COLOR['fixed_lr']),
              ('pbt', 'PBT, fast', ARM_COLOR['windowed_pbt_v2'])]


def sel1m_overview(directory):
    """Holdout score of the three arms of the sel1m comparison, grouped by seed, in the style of figure 06."""
    rows = []
    for seed, campaign, suffix in SEL1M_SEEDS:
        path = EVAL / campaign / 'analysis.json'
        if path.exists():
            metrics = load(path)['metrics']
            rows += [(f'{label}, {seed}', metrics[f'{arm}_sel1m_{suffix}'][PRIMARY], color)
                     for arm, label, color in SEL1M_ARMS]
    return lollipop(rows, 'train800k runs, selection on 1M jets: holdout', Path(directory) / '08_sel1m_holdout.png')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=EVAL / 'lr_level_followups/figures')
    args = parser.parse_args(argv)
    manifests = {arm: load(PROJECT_DIR / 'runs/pbt' / run / 'manifest.json')
                 for arm, run in (('fixed_lr', FIXED), ('windowed_pbt_v2', PBT))}
    summaries = {arm: arm_summary(manifest, None) for arm, manifest in manifests.items()}
    analysis = load(EVAL / CAMPAIGN / 'analysis.json')
    test = dict(metrics=analysis['metrics'], comparisons=dict(selected_best=analysis['comparisons'][PAIR]))
    written = figures(args.output, summaries, test=test,
                      labels=dict(fixed_lr='A fixed LR 8e-5 to 3.2e-4', windowed_pbt_v2='B fast PBT'),
                      text=dict(epoch='train800k epoch', selection_title='Best member so far, validation set',
                                test_title='Holdout: B vs A per working point (negative = B better)'))
    written += [overview(args.output), seeds_overview(args.output), sel1m_overview(args.output)]
    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
