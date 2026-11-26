#!/usr/bin/env python3
"""Figures of the study report page (docs/_extra/overview.html), as PNG in a plain physics-plot style.

    .venv/bin/python scripts/reports/plot_overview_figs.py

Writes docs/_extra/overview_figs/*.png. Test values are the published values of docs/results.rst; LR and
Validation trajectories come from the run manifests (runs/pbt); loss curves from
published/results/sel1m/train8m_followups.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/_extra/overview_figs'
RUNS = ROOT / 'runs/pbt'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'
STD, PBT, ENS, CONST = '#1f5fa8', '#c4472a', '#6a4fd0', '#555555'
GOOD, BAD = '#2e8b3e', '#c4472a'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelsize': 13, 'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'legend.frameon': False, 'legend.fontsize': 12, 'figure.facecolor': 'white', 'savefig.dpi': 160,
    'mathtext.default': 'regular'})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def delta_bars(name, rows, lim, height=None):
    """rows: (label, change %, colour, interval or None, hatched). Negative = lower mistag = better."""
    fig, ax = plt.subplots(figsize=(9, height or 0.62 * len(rows) + 1.3))
    n = len(rows)
    for i, (label, v, c, ci, hatch) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, height=0.55, color='white' if hatch else c, edgecolor=c, hatch='///' if hatch else None, lw=1.2)
        if ci:
            ax.errorbar(v, y, xerr=[[v - ci[0]], [ci[1] - v]], fmt='none', ecolor='black', capsize=4, lw=1.3)
        edge = (ci[0] if v < 0 else ci[1]) if ci else v
        word = 'better' if v < 0 else 'worse'
        ax.text(edge + (-0.3 if v < 0 else 0.3), y, f'{abs(v):.1f}% {word}', va='center',
                ha='right' if v < 0 else 'left', fontsize=12.5)
    ax.axvline(0, color='black', lw=1.2)
    ax.set_yticks(range(n), [r[0] for r in reversed(rows)])
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_xlim(*lim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xticks(ax.get_xticks(), [f'{abs(t):g}%' for t in ax.get_xticks()])
    ax.set_xlim(*lim)
    ax.set_xlabel('change of the mistag   (← better | worse →)')
    save(fig, name)


def summary():
    delta_bars('summary.png', [
        ('PBT vs standard ParT, 800k jets', -0.81, GOOD, (-1.34, -0.34), False),
        ('PBT vs standard ParT, 8M jets', 12.4, BAD, None, False),
        ('PBT + probes + final LR decay, 8M jets', 2.7, BAD, None, False),
        ('ensemble of two vs one model, 8M jets', -5.01, GOOD, (-5.84, -4.17), False)], (-18, 22))



def stage2():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    grid = [(1.4e-5, 0.3269, '#1'), (3e-5, 0.3146, '#12'), (8e-5, 0.3018, '#13'), (1.2e-4, 0.3033, '#14')]
    ax.axvspan(3e-6, 1.4e-5, color='0.92', zorder=0)
    ax.text(6.5e-6, 0.3008, 'LRs used\nin September', ha='center', fontsize=11, color='0.35')
    ax.plot([g[0] for g in grid], [g[1] for g in grid], '-o', color=CONST, mfc='white', mew=1.6, ms=8, lw=1.4,
            label='constant LR (best model of each range)')
    for x, y, t in grid:
        ax.annotate(t, (x, y), (7, 6), textcoords='offset points', fontsize=10.5, color='0.35')
    ax.plot([2.02e-5], [0.3225], 's', color=PBT, ms=9, label='PBT, September (#3)')
    ax.plot([8.75e-5], [0.3003], 's', color=PBT, mfc='white', mew=2, ms=9, label='PBT, November (#15)')
    ax.set_xscale('log')
    ax.set_xlim(2.5e-6, 2.5e-4)
    ax.set_ylim(0.298, 0.330)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('learning rate of the selected model')
    ax.set_ylabel('mistag (%)')
    ax.legend(loc='upper right')
    save(fig, 'stage2.png')


def dots(name, rows, xlim, xlabel, legend):
    """rows: (label, [values per seed], marker, colour, filled) or None (separator)."""
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.4))
    n = len(rows)
    for i, row in enumerate(rows):
        y = n - 1 - i
        if row is None:
            ax.axhline(y, color='0.75', lw=0.8, ls='--')
            continue
        label, vals, m, c, filled = row
        ax.plot(vals, [y] * len(vals), m, color=c, mfc=c if filled else 'white', mew=1.6, ms=9, ls='none')
    ax.set_yticks([n - 1 - i for i, r in enumerate(rows) if r], [r[0] for r in rows if r])
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(*xlim)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(xlabel)
    ax.legend(handles=legend, loc='center right', fontsize=11)
    save(fig, name)


def stage3():
    dots('stage3.png', [('standard ParT', [0.3029, 0.3038, 0.3024], 'o', STD, True),
                        ('constant LR', [0.3028, 0.3013, 0.3024], 'o', CONST, False),
                        ('PBT', [0.3008, 0.3012, 0.2996], 's', PBT, True)],
         (0.2985, 0.3045), 'mistag (%) of the selected model', [
             Line2D([], [], ls='none', marker='o', color='0.3', mfc='0.3', ms=8, label='one point per seed')])


def stage4():
    delta_bars('stage4.png', [('PBT vs standard ParT', 12.4, BAD, None, False),
                              ('with fix 1 (cool-down probes)', 5.4, BAD, None, False),
                              ('with fixes 1 and 2 (final LR decay)', 2.7, BAD, None, False)], (-2, 21))


def stage5():
    dots('stage5.png', [('standard ParT, start LR 10$^{-3}$', [0.1635, 0.1623, 0.1634], 'o', STD, False),
                        ('standard ParT, start LR 3×10$^{-3}$', [0.1611, 0.1597, 0.1623], 'o', STD, True),
                        ('ensemble of the two', [0.1530, 0.1519, 0.1540], 'D', ENS, True), None,
                        ('PBT', [0.1812], 's', PBT, True), ('PBT with both fixes', [0.1655], 's', PBT, False)],
         (0.148, 0.186), 'mistag (%) of the selected model', [
             Line2D([], [], ls='none', marker='o', color='0.3', ms=8, label='one point per seed (PBT: one seed)')])


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


def why():
    loss = json.loads((ROOT / 'published/results/sel1m/train8m_followups.json').read_text())['loss_curves']
    std = trajectory('sel1m_train8m_standard_part_3e-3')
    pbt = trajectory('sel1m_train8m_fast_pbt')
    fix = trajectory('sel1m_train8m_fast_pbt_probe_decay')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, title in ((axes[0, 0], '800k', '(a) 800k jets: loss per epoch'), (axes[0, 1], '8M', '(b) 8M jets: loss per epoch')):
        pts = loss[key]['epochs']
        e = [p['epoch'] - 17 for p in pts]
        ax.plot(e, [p['train'] for p in pts], color='black', lw=1.8, label='training data')
        ax.plot(e, [p['selection'] for p in pts], color=STD, lw=1.8, label='Validation')
        ax.set_title(title, loc='left', fontsize=13)
        ax.set_ylabel('loss')
        ax.legend(loc='upper right')
    a, b = axes[1]
    a.plot([p[0] for p in std], [p[1] for p in std], color=STD, lw=2, label='standard ParT')
    a.fill_between([p[0] for p in pbt], [p[1] for p in pbt], [p[2] for p in pbt], color=PBT, alpha=0.3, lw=0,
                   label='PBT (range of 5 models)')
    a.set_yscale('log')
    a.set_ylim(5e-6, 8e-3)
    a.set_ylabel('learning rate')
    a.set_title('(c) 8M jets: learning rate', loc='left', fontsize=13)
    a.legend(loc='lower left')
    b.plot([p[0] for p in std], [p[3] for p in std], color=STD, lw=2, label='standard ParT')
    b.plot([p[0] for p in pbt], [p[3] for p in pbt], color=PBT, lw=2, label='PBT')
    b.plot([p[0] for p in fix], [p[3] for p in fix], color=PBT, lw=1.6, ls='--', label='PBT with both fixes')
    b.set_ylim(0.155, 0.225)
    b.set_ylabel('best Validation mistag so far (%)')
    b.set_title('(d) 8M jets: mistag', loc='left', fontsize=13)
    b.legend(loc='upper right')
    for ax in (a, b):
        ax.axvspan(36, 50, color='0.92', zorder=0)
    for ax in axes.flat:
        ax.set_xlim(1, 50)
        ax.set_xlabel('epoch')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ax is not a:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    save(fig, 'why.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (summary, stage2, stage3, stage4, stage5, why):
        f()
