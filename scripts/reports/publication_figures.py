#!/usr/bin/env python3
"""Figures of the research publication docs/_extra/overview.html (written to docs/_extra/pub/*.svg).

    .venv/bin/python scripts/reports/publication_figures.py

Test values are the published values of docs/results.rst (selected checkpoints, Test sample of 2.8M jets
unless stated); learning-rate and Validation trajectories come from the run manifests in runs/pbt; loss curves
from published/results/sel1m/train8m_followups.json. Nothing here is fitted or smoothed.
"""
from datetime import date
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FixedLocator  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _f in Path('/usr/share/fonts/truetype/lato').glob('Lato-*.ttf'):  # system Lato; DejaVu Sans is the fallback
    font_manager.fontManager.addfont(str(_f))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/_extra/pub'
RUNS = ROOT / 'runs/pbt'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'

INK, MUTED, GRID = '#1d1d1f', '#6e6e73', '#e6e6e9'
STD, PBT, ENS, CONST = '#2b5797', '#c4532f', '#2a7d64', '#8a8a8f'
plt.rcParams.update({
    'font.family': ['Lato', 'DejaVu Sans'], 'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 4, 'ytick.major.size': 4,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'svg.fonttype': 'path', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0.08)
    preview = __import__('os').environ.get('FIG_PREVIEW')  # optional PNG copies for review
    if preview:
        fig.savefig(Path(preview) / name.replace('.svg', '.png'), bbox_inches='tight', pad_inches=0.08, dpi=110)
    plt.close(fig)


def grid(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def rows_panel(ax, rows, xlim, ticks, title):
    """rows: (label, values per seed, colour, marker, filled, note). One row per method, one point per seed."""
    n = len(rows)
    for i, (label, vals, c, m, filled, note) in enumerate(rows):
        y = n - 1 - i
        ax.plot(vals, [y] * len(vals), m, ms=8, mew=1.6, color=c, mfc=c if filled else 'white', ls='none', zorder=3)
        if len(vals) > 1:
            mean = sum(vals) / len(vals)
            ax.plot([mean, mean], [y - 0.28, y + 0.28], color=c, lw=2, zorder=2)
        ax.text(xlim[0], y + 0.36, label, ha='left', va='bottom', fontsize=11, color=INK, fontweight='bold')
        if note:
            ax.text(max(vals) + (xlim[1] - xlim[0]) * 0.025, y, note, va='center', fontsize=10.5, color=c)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('mistag on Test (%)  —  lower is better')
    ax.set_title(title, loc='left', fontweight='bold', pad=14)
    grid(ax)


def fig_central():
    """The study in one figure: Test mistag of every method in the two data regimes."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.3), gridspec_kw=dict(wspace=0.18))
    rows_panel(a, [
        ('Standard ParT', [0.3029, 0.3038, 0.3024], STD, 'o', True, 'reference'),
        ('Constant learning rate, tuned', [0.3028, 0.3013, 0.3024], CONST, 'o', False, '−0.28%  (mean of seeds)'),
        ('PBT', [0.3008, 0.3012, 0.2996], PBT, 's', True, '−0.81%  [−1.34, −0.34]'),
    ], (0.2985, 0.3065), [0.300, 0.302, 0.304, 0.306], 'Training on 800k jets  ·  3 seeds')
    rows_panel(b, [
        ('Standard ParT', [0.1611, 0.1597, 0.1623], STD, 'o', True, 'reference'),
        ('PBT', [0.1812], PBT, 's', True, '+12.4%  (1 seed)'),
        ('PBT with probes and final decay', [0.1655], PBT, 's', False, '+2.7%  (1 seed)'),
        ('Ensemble of two standard ParT models', [0.1530, 0.1519, 0.1540], ENS, 'D', True, '−5.0%  [−5.8, −4.2]'),
    ], (0.148, 0.200), [0.150, 0.160, 0.170, 0.180, 0.190, 0.200], 'Training on 8M jets')
    save(fig, 'fig1_central.svg')


def fig_trajectory():
    """How the estimated advantage of PBT changed as the comparison became stricter."""
    pts = [  # (date, PBT advantage %, label, data size, valid comparison, interval)
        (date(2026, 9, 18), 1.4, 'vs. constant LRs\n(too low)', '800k', False, None),
        (date(2026, 11, 2), 7.7, 'vs. constant LRs (too low),\npreregistered, Test 9.0M', '8M', False, None),
        (date(2026, 11, 4), 1.0, 'vs. tuned\nconstant LR', '800k', True, None),
        (date(2026, 11, 9), -0.5, 'same,\nsecond seed', '800k', True, None),
        (date(2026, 11, 13), 0.81, 'vs. standard ParT,\n3 seeds', '800k', True, (0.34, 1.34)),
        (date(2026, 11, 17), -12.4, 'vs. standard ParT', '8M', True, None),
        (date(2026, 11, 21), -5.4, '+ probes', '8M', True, None),
        (date(2026, 11, 23), -2.7, '+ final decay', '8M', True, None),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvspan(date(2026, 9, 12), date(2026, 11, 3), color='#f4f4f6', zorder=0)
    ax.text(date(2026, 9, 14), 10.6, 'Comparator: constant learning rates from a grid\nthat later proved too low',
            fontsize=10, color=MUTED, va='top')
    ax.text(date(2026, 11, 3), 10.6, 'Comparators tuned;\nfrom Nov 10 standard ParT', fontsize=10, color=MUTED, va='top')
    offsets = {0: (8, 10), 1: (-8, 12), 2: (8, 8), 3: (8, -26), 4: (8, 8), 5: (10, -6), 6: (10, -6), 7: (10, -4)}
    for k, (d, v, lab, size, valid, ci) in enumerate(pts):
        c = PBT if size == '8M' else STD
        m = 's' if size == '8M' else 'o'
        if ci:
            ax.plot([d, d], ci, color=c, lw=1.6)
        ax.plot(d, v, m, ms=9, mew=1.6, color=c, mfc=c if valid else 'white', zorder=3)
        dx, dy = offsets[k]
        ax.annotate(lab, (d, v), (dx, dy), textcoords='offset points', fontsize=9.5, color=INK,
                    ha='right' if dx < 0 else 'left', va='bottom' if dy > 0 else 'top', linespacing=1.15)
    ax.set_xlim(date(2026, 9, 12), date(2026, 12, 3))
    ax.set_ylim(-14, 11)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    ax.set_ylabel('PBT advantage over comparator (%)')
    ax.yaxis.set_major_locator(FixedLocator([-12, -8, -4, 0, 4, 8]))
    grid(ax, 'y')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', color=STD, ls='none', ms=8, label='800k training jets'),
                       Line2D([], [], marker='s', color=PBT, ls='none', ms=8, label='8M training jets'),
                       Line2D([], [], marker='o', color=MUTED, mfc='white', ls='none', ms=8,
                              label='comparator later found too weak')],
              loc='lower left', fontsize=10)
    save(fig, 'fig2_trajectory.svg')


def fig_lr():
    """Test mistag against learning rate: constant-LR grids and the two PBT runs (800k, seed 12345)."""
    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    grid_pts = [(1.4e-5, 0.3269), (3e-5, 0.3146), (8e-5, 0.3018), (1.2e-4, 0.3033)]
    ax.axvspan(3e-6, 1.4e-5, color='#f4f4f6', zorder=0)
    ax.text(6.4e-6, 0.3285, 'range of the\nSeptember grid', ha='center', va='top', fontsize=10, color=MUTED)
    ax.plot(*zip(*grid_pts), '-', color=CONST, lw=1.5, zorder=2)
    ax.plot(*zip(*grid_pts), 'o', color=CONST, mfc='white', mew=1.6, ms=8, zorder=3)
    ax.text(3.3e-5, 0.3165, 'constant learning rate\n(best model of each grid)', fontsize=10.5, color=CONST)
    ax.plot([2.02e-5], [0.3225], 's', color=PBT, ms=9, zorder=4)
    ax.annotate('PBT, September', (2.02e-5, 0.3225), (10, 2), textcoords='offset points', fontsize=10.5, color=PBT)
    ax.plot([8.75e-5], [0.3003], 's', color=PBT, ms=9, zorder=4)
    ax.annotate('PBT, November', (8.75e-5, 0.3003), (10, -14), textcoords='offset points', fontsize=10.5, color=PBT)
    ax.set_xscale('log')
    ax.set_xlim(2.5e-6, 3e-4)
    ax.set_ylim(0.298, 0.330)
    ax.set_xlabel('learning rate of the selected model')
    ax.set_ylabel('mistag on Test (%)')
    grid(ax, 'both')
    save(fig, 'fig3_learning_rate.svg')


def trajectory(run, members=None):
    m = json.loads((RUNS / run / 'manifest.json').read_text())
    out, best = [], None
    for g in m['generations']:
        ws = {k: w for k, w in g['workers'].items() if members is None or k in members}
        vals = [w['metrics'][PRIMARY] for w in ws.values() if (w.get('metrics') or {}).get(PRIMARY) is not None]
        if vals:
            best = min(vals) if best is None else min(best, min(vals))
        lrs = sorted(w['lr'] for w in ws.values())
        out.append((g['epoch'] - 17, lrs[0], lrs[-1], best))
    return out


def fig_regimes():
    """Training and Validation loss of the best standard ParT model at 800k and 8M jets."""
    data = json.loads((ROOT / 'published/results/sel1m/train8m_followups.json').read_text())['loss_curves']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw=dict(wspace=0.22))
    for ax, key, title in zip(axes, ('800k', '8M'), ('800k training jets', '8M training jets')):
        pts = data[key]['epochs']
        e = [p['epoch'] - 17 for p in pts]
        tr, va = [p['train'] for p in pts], [p['selection'] for p in pts]
        ax.plot(e, tr, color=INK, lw=1.8)
        ax.plot(e, va, color=STD, lw=1.8)
        ax.text(e[-1] + 0.8, tr[-1], 'training', va='center', fontsize=10, color=INK)
        ax.text(e[-1] + 0.8, va[-1], 'Validation', va='center', fontsize=10, color=STD)
        ax.set_xlim(1, 50)
        ax.set_xlabel('epoch')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        grid(ax, 'y')
    axes[0].set_ylabel('cross-entropy loss')
    axes[0].annotate('training loss falls below\nValidation loss: overfitting', (35, 0.2728), (18, 0.2900),
                     fontsize=10, color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    axes[1].annotate('the two stay together', (45, 0.2452), (24, 0.2700), fontsize=10, color=MUTED,
                     arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    # separate labels when the end points nearly coincide
    tx = axes[1].texts
    tx[0].set_y(tx[0].get_position()[1] - 0.0018)
    tx[1].set_y(tx[1].get_position()[1] + 0.0018)
    save(fig, 'fig4_regimes.svg')


def fig_mechanism():
    """8M, seed 12345: LR schedule and best Validation mistag so far, standard ParT against PBT."""
    std = trajectory('sel1m_train8m_standard_part_3e-3')
    pbt = trajectory('sel1m_train8m_fast_pbt')
    fix = trajectory('sel1m_train8m_fast_pbt_probe_decay')
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(wspace=0.25))
    for ax in (a, b):
        ax.axvspan(36, 50, color='#f4f4f6', zorder=0)
        ax.set_xlim(1, 50)
        ax.set_xlabel('epoch')
    a.plot([p[0] for p in std], [p[1] for p in std], color=STD, lw=2)
    a.fill_between([p[0] for p in pbt], [p[1] for p in pbt], [p[2] for p in pbt], color=PBT, alpha=0.28, lw=0)
    a.text(3, 3.9e-3, 'standard ParT', color=STD, fontsize=10.5)
    a.text(16, 1.5e-4 * 0.55, 'PBT, range over its five models', color=PBT, fontsize=10.5, va='top')
    a.text(37, 4.5e-3, 'final decay', color=MUTED, fontsize=10)
    a.set_yscale('log')
    a.set_ylim(5e-6, 8e-3)
    a.set_ylabel('learning rate')
    a.set_title('Learning rate', loc='left', fontweight='bold')
    b.plot([p[0] for p in std], [p[3] for p in std], color=STD, lw=2)
    b.plot([p[0] for p in pbt], [p[3] for p in pbt], color=PBT, lw=2)
    b.plot([p[0] for p in fix], [p[3] for p in fix], color=PBT, lw=1.5, ls='--')
    b.text(50.8, std[-1][3], 'standard ParT', va='center', color=STD, fontsize=10.5)
    b.text(50.8, pbt[-1][3], 'PBT', va='center', color=PBT, fontsize=10.5)
    b.text(50.8, fix[-1][3] + 0.0016, 'PBT + probes\n+ final decay', va='bottom', color=PBT, fontsize=10)
    b.set_ylim(0.155, 0.225)
    b.set_ylabel('best Validation mistag so far (%)')
    b.set_title('Validation mistag', loc='left', fontweight='bold')
    grid(b, 'y')
    save(fig, 'fig5_mechanism.svg')


def fig_ensemble():
    """(a) Per seed: the two standard ParT models and their ensemble. (b) Disagreement between two models."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.35))
    seeds = {'12345': (0.1635, 0.1611, 0.1530), '22345': (0.1623, 0.1597, 0.1519), '32345': (0.1634, 0.1623, 0.1540)}
    xs = [0, 1, 2]
    for s, vals in seeds.items():
        a.plot(xs, vals, '-', color='#c9c9ce', lw=1.2, zorder=1)
        a.plot(xs[:2], vals[:2], 'o', color=STD, ms=7, zorder=2)
        a.plot(xs[2], vals[2], 'D', color=ENS, ms=7, zorder=2)
        a.text(2.13, vals[2], f'seed {s}', va='center', fontsize=9.5, color=MUTED)
    a.set_xticks(xs, ['start LR\n10⁻³', 'start LR\n3×10⁻³', 'ensemble\nof the two'])
    a.set_xlim(-0.3, 2.75)
    a.set_ylabel('mistag on Test (%)')
    a.set_title('Standard ParT models, 8M jets', loc='left', fontweight='bold')
    grid(a, 'y')
    groups = [  # (label, disagreement low, high, gain text, colour)
        ('two checkpoints of one run, 8M', 0.2, 0.4, 'no gain', CONST),
        ('two models of one PBT run, 8M', 0.2, 0.4, 'no gain', PBT),
        ('two models, different seeds, 800k', 0.6, 0.7, 'no gain', STD),
        ('two separately trained models, 8M', 1.5, 2.3, '3.6–4.7% gain', ENS),
    ]
    for i, (lab, lo, hi, gain, c) in enumerate(groups):
        y = len(groups) - 1 - i
        b.plot([lo, hi], [y, y], color=c, lw=7, solid_capstyle='butt')
        b.text(0, y + 0.3, lab, fontsize=10, color=INK, va='bottom')
        b.text(hi + 0.06, y, gain, fontsize=10, color=c, va='center')
    b.set_xlim(0, 3.1)
    b.set_ylim(-0.6, len(groups) - 0.05)
    b.set_yticks([])
    b.spines['left'].set_visible(False)
    b.set_xlabel('jets on which the two models predict different classes (%)')
    b.set_title('Ensemble gain needs models that disagree', loc='left', fontweight='bold')
    grid(b)
    save(fig, 'fig6_ensemble.svg')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_central, fig_trajectory, fig_lr, fig_regimes, fig_mechanism, fig_ensemble):
        f()
    print(sorted(p.name for p in OUT.iterdir()))
