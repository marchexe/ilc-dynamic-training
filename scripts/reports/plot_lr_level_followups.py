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


def overview(directory):
    from training.pbt.reporting.style import plot_setup
    plt = plot_setup()
    values = [holdout_metric(campaign, checkpoint) for _, campaign, checkpoint, _ in RUNS]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for y, ((label, _, _, kind), value) in enumerate(zip(RUNS, values)):
        ax.hlines(y, min(values) - 0.004, value, color='#ebeae6', lw=2)
        ax.plot(value, y, 'o', ms=9, color=ARM_COLOR[kind], mec='white', mew=1.5)
        ax.annotate(f'{value:.4f}', (value, y), xytext=(7, 0), textcoords='offset points', va='center',
                    fontsize=8, color='#52514e')
    ax.set_yticks(range(len(RUNS)), [r[0] for r in RUNS])
    ax.invert_yaxis()
    ax.set_xlim(min(values) - 0.004, max(values) + 0.006)
    ax.set(xlabel='holdout primary mistag (%), lower is better', title='train800k runs, seed 12345: holdout')
    ax.grid(axis='x', color='#ebeae6', lw=0.8)
    ax.tick_params(axis='y', length=0)
    path = Path(directory) / '06_holdout_all_runs.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


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
    written.append(overview(args.output))
    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
