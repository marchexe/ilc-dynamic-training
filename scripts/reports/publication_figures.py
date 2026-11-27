#!/usr/bin/env python3
"""Figures of the research publication docs/_extra/overview.html (written to docs/_extra/pub/*.svg).

    .venv/bin/python scripts/reports/publication_figures.py

Test values are the published values of docs/results.rst (selected checkpoints, Test sample of 2.8M jets
unless stated); learning-rate and Validation trajectories come from the run manifests in runs/pbt; loss curves
from published/results/sel1m/train8m_followups.json. Nothing here is fitted or smoothed.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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


def central_rows():
    """Rows of Figure 1 from the result files: (group, label, change %, interval or None, seeds, colour)."""
    pub = ROOT / 'published/results/sel1m'
    block = json.loads((pub / 'block100.json').read_text())['comparisons']
    pooled = json.loads((pub / 'pooled_block100.json').read_text())
    const = [c['change_percent'] for c in block if c['left'] == 'standard' and c['right'] == 'fixed']
    t8 = {h['label']: h['value'] for h in json.loads((pub / 'train8m_followups.json').read_text())['holdout']}
    ref = t8['Standard ParT, start LR 3e-3']
    ens = json.loads((ROOT / 'runs/eval/ensemble_seed32345_v1/result.json').read_text())['question_B']
    rel = lambda v: 100 * (v / ref - 1)
    return [
        ('Training (800k)', 'Constant LR, tuned grid', sum(const) / len(const), None, 3, CONST),
        ('Training (800k)', 'PBT', pooled['mean_change_percent'], pooled['mean_interval_percent'], 3, PBT),
        ('Training (8M)', 'PBT', rel(t8['PBT']), None, 1, PBT),
        ('Training (8M)', 'PBT + cool-down probes', rel(t8['PBT + probes']), None, 1, PBT),
        ('Training (8M)', 'PBT + probes + final decay', rel(t8['PBT + probes + final decay']), None, 1, PBT),
        ('Training (8M)', 'Ensemble of two standard ParT models', ens['mean_change_percent'], ens['pooled_interval_percent'], 3, ENS),
    ]


def fig_central():
    """Diverging bars: change of the Test mistag against standard ParT trained on the same sample."""
    rows = central_rows()
    short = {'Constant LR, tuned grid': 'Tuned\nconstant LR', 'PBT': 'PBT', 'PBT + cool-down probes': 'PBT\n+ probes',
             'PBT + probes + final decay': 'PBT + probes\n+ final decay', 'Ensemble of two standard ParT models': 'Ensemble\nof two'}
    good, bad, grey = '#4f8a6a', '#b5553f', '#9a9aa0'
    xs, x, last = [], 0.0, None
    for g, *_ in rows:
        if last is not None:
            x += 1.0 if g == last else 1.7
        last = g
        xs.append(x)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    groups = list(dict.fromkeys(r[0] for r in rows))
    span = {g: (min(xx for r, xx in zip(rows, xs) if r[0] == g), max(xx for r, xx in zip(rows, xs) if r[0] == g)) for g in groups}
    lo, hi = span[groups[1]]
    ax.axvspan(lo - 0.6, hi + 0.6, color='#f5f5f7', zorder=0, lw=0)
    for (g, lab, v, ci, seeds, c), xx in zip(rows, xs):
        col = grey if lab.startswith('Constant') else (good if v < 0 else bad)
        ax.bar(xx, v, width=0.62, color=col, zorder=2)
        if ci:
            ax.plot([xx, xx], ci, color=INK, lw=1.2, zorder=3)
            for e in ci:
                ax.plot([xx - 0.09, xx + 0.09], [e, e], color=INK, lw=1.2, zorder=3)
        end = (min(ci) if ci else v) if v < 0 else v
        val = f'{v:+.2f}%'.replace('-', '−')
        ax.text(xx, end + (-0.6 if v < 0 else 0.45), val, ha='center', va='top' if v < 0 else 'bottom',
                fontsize=12.5, color=INK, fontweight='bold')
        if ci:
            ax.text(xx, end - 1.6, f'[{ci[0]:.2f}, {ci[1]:.2f}]'.replace('-', '−'), ha='center', va='top',
                    fontsize=10.5, color=MUTED)
        ax.text(xx, -9.6, short[lab], ha='center', va='top', fontsize=11.5, color=INK, linespacing=1.15)
        ax.text(xx, -12.3, f'{seeds} seed{"s" if seeds > 1 else ""}', ha='center', va='top', fontsize=10.5, color=MUTED)
    for g in groups:
        a0, a1 = span[g]
        ax.plot([a0 - 0.4, a1 + 0.4], [-13.7, -13.7], color=MUTED, lw=1, clip_on=False)
        ax.text((a0 + a1) / 2, -14.1, g, ha='center', va='top', fontsize=12.5, color=INK, fontweight='bold')
    ax.axhline(0, color=INK, lw=1.1, zorder=3)
    ax.text(xs[-1] + 0.75, 0.25, 'standard ParT', ha='right', va='bottom', fontsize=10.5, color=MUTED)
    ax.set_xlim(xs[0] - 0.75, xs[-1] + 0.75)
    ax.set_ylim(-9, 15)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ticks = list(range(-6, 16, 3))
    ax.set_yticks(ticks, [('+' if t > 0 else '') + f'{t}%'.replace('-', '−') if t else '0' for t in ticks])
    ax.tick_params(axis='y', labelsize=11.5)
    ax.set_ylabel('change in Test mistag against standard ParT', fontsize=11.5)
    ax.text(xs[0] - 0.68, 14.6, 'worse ↑', ha='left', va='top', fontsize=10.5, color=MUTED)
    ax.text(xs[0] - 0.68, -8.6, 'better ↓', ha='left', va='bottom', fontsize=10.5, color=MUTED)
    grid(ax, 'y')
    save(fig, 'fig1_central.svg')


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
    for ax, key, title in zip(axes, ('800k', '8M'), ('Training (800k)', 'Training (8M)')):
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
    """Disagreement between two models and the ensemble gain, for four kinds of model pairs."""
    fig, b = plt.subplots(figsize=(10.4, 2.9))
    groups = [  # (label, disagreement low, high, gain text, colour)
        ('two checkpoints of one run, 8M', 0.2, 0.4, 'no gain', CONST),
        ('two models of one PBT run, 8M', 0.2, 0.4, 'no gain', PBT),
        ('two models of different seeds, 800k', 0.6, 0.7, 'no gain', STD),
        ('two separately trained models, 8M', 1.5, 2.3, 'gain 3.6% (same LR) to 4.7% (different LR)', ENS),
    ]
    for i, (lab, lo, hi, gain, c) in enumerate(groups):
        y = len(groups) - 1 - i
        b.plot([lo, hi], [y, y], color=c, lw=8, solid_capstyle='butt')
        b.text(-0.05, y, lab, fontsize=10.5, color=INK, va='center', ha='right')
        b.text(hi + 0.06, y, gain, fontsize=10.5, color=c, va='center')
    b.set_xlim(0, 4.2)
    b.set_ylim(-0.6, len(groups) - 0.4)
    b.set_yticks([])
    b.spines['left'].set_visible(False)
    b.set_xlabel('Validation jets on which the two models predict different classes (%)')
    grid(b)
    save(fig, 'fig6_ensemble.svg')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_central, fig_lr, fig_regimes, fig_mechanism, fig_ensemble):
        f()
    print(sorted(p.name for p in OUT.iterdir()))
