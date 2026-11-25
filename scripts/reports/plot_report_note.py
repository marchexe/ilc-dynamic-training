#!/usr/bin/env python3
"""Figures of the summary note (docs/_extra/overview.html), in a plain journal style.

    .venv/bin/python scripts/reports/plot_report_note.py

Writes docs/_extra/note/{history,fig1,fig3,fig4}.png. Test values are the published values of docs/results.rst;
trajectories come from the run manifests, loss curves from published/results/sel1m/train8m_followups.json.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/_extra/note'
RUNS = ROOT / 'runs/pbt'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'
K, BLUE, RED, GREY = '#000000', '#1f5fa8', '#b2232a', '#8c8c8c'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 13, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.dpi': 200,
                     'mathtext.fontset': 'dejavuserif'})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def fig1():
    """Test mistag against LR at 800k: constant-LR grids and the PBT runs at the LR of their selected model."""
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    grid = [(1.4e-5, 0.3269), (3e-5, 0.3146), (8e-5, 0.3018), (1.2e-4, 0.3033)]
    ax.plot(*zip(*grid), '-o', color=K, ms=5, lw=1, label='constant LR (best model of each grid)')
    ax.plot([2.02e-5], [0.3225], 's', color=RED, ms=7, label='PBT, Sep (decision every 5 epochs)')
    ax.plot([8.75e-5], [0.3003], 'D', color=RED, mfc='white', mew=1.5, ms=7, label='PBT, Nov (decision every 2 epochs)')
    ax.axvspan(3e-6, 1.4e-5, color='0.92', zorder=0)
    ax.text(6.5e-6, 0.3045, 'grid of\nSeptember', ha='center', fontsize=11.5, color='0.3')
    ax.set_xscale('log')
    ax.set_xlim(2.5e-6, 2.5e-4)
    ax.set_ylim(0.298, 0.330)
    ax.set_xlabel('learning rate (of the selected model)')
    ax.set_ylabel('Test (2.8M) mistag (%)')
    ax.legend(loc='upper right', fontsize=11)
    save(fig, 'fig1.png')



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


def fig3():
    """8M: LR per epoch and best Validation value so far, standard ParT against PBT."""
    std = trajectory('sel1m_train8m_standard_part_3e-3')
    pbt = trajectory('sel1m_train8m_fast_pbt')
    fix = trajectory('sel1m_train8m_fast_pbt_probe_decay')
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a.plot([p[0] for p in std], [p[1] for p in std], color=K, lw=1.5, label='standard ParT')
    a.fill_between([p[0] for p in pbt], [p[1] for p in pbt], [p[2] for p in pbt], color=RED, alpha=0.25, lw=0,
                   label='PBT (range of 5 models)')
    a.set_yscale('log')
    a.set_xlabel('epoch')
    a.set_ylabel('learning rate')
    a.set_xlim(1, 50)
    a.set_ylim(2e-6, 6e-3)
    a.legend(loc='lower left', fontsize=11)
    a.set_title('(a)', loc='left', fontsize=13)
    b.plot([p[0] for p in std], [p[3] for p in std], color=K, lw=1.5, label='standard ParT')
    b.plot([p[0] for p in pbt], [p[3] for p in pbt], color=RED, lw=1.5, label='PBT')
    b.plot([p[0] for p in fix], [p[3] for p in fix], color=RED, lw=1.2, ls='--', label='PBT + probes + final decay')
    b.set_xlabel('epoch')
    b.set_ylabel('best Validation mistag (%)')
    b.set_xlim(1, 50)
    b.set_ylim(0.155, 0.232)
    b.legend(loc='upper right', fontsize=11)
    b.set_title('(b)', loc='left', fontsize=13)
    for ax in (a, b):
        ax.axvspan(36, 50, color='0.93', zorder=0)
    save(fig, 'fig3.png')


def fig4():
    """Training and Validation loss of the best standard ParT model, 800k and 8M."""
    loss = json.loads((ROOT / 'published/results/sel1m/train8m_followups.json').read_text())['loss_curves']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, key, tag in zip(axes, ('800k', '8M'), ('(a) Training (800k)', '(b) Training (8M)')):
        pts = loss[key]['epochs']
        e = [p['epoch'] - 17 for p in pts]
        ax.plot(e, [p['train'] for p in pts], color=K, lw=1.4, label='training data')
        ax.plot(e, [p['selection'] for p in pts], color=BLUE, lw=1.4, label='Validation (1M)')
        ax.set_xlabel('epoch')
        ax.set_xlim(1, 50)
        ax.set_title(tag, loc='left', fontsize=13)
    axes[0].set_ylabel('loss (cross-entropy)')
    axes[0].legend(loc='upper right', fontsize=11)
    save(fig, 'fig4.png')



def fig_history():
    """One row per configuration, in time order: Test (2.8M) value of the selected model (one point per seed)."""
    C, S, P, E = ('o', K, 'white'), ('o', K, K), ('s', RED, RED), ('D', BLUE, BLUE)
    k800 = [('1   constant LR 3e-6 to 1.4e-5', [0.3269], C), ('2   constant LR, 100 epochs', [0.3156], C),
            ('3   PBT, every 5 epochs, max 3e-5', [0.3225], P), ('4   PBT, 100 epochs', [0.3111], P),
            ('6   PBT every epoch / every 5 epochs', [0.3177, 0.3162, 0.3167, 0.3163], P), None,
            ('12  constant LR 1.4e-5 to 3e-5', [0.3146], C), ('13  constant LR 3e-5 to 8e-5', [0.3018], C),
            ('14  constant LR 8e-5 to 3.2e-4', [0.3033], C), ('15  fast PBT, every 2 epochs', [0.3003], P),
            ('16  same, seed 22345: constant LR', [0.3027], C), ('16  same, seed 22345: PBT', [0.3041], P),
            ('17  fast PBT, extra copies', [0.3026], P), None,
            ('18  standard ParT (3 seeds)', [0.3029, 0.3038, 0.3024], S),
            ('18  constant LR (3 seeds)', [0.3028, 0.3013, 0.3024], C),
            ('18  PBT (3 seeds)', [0.3008, 0.3012, 0.2996], P), ('25  PBT + probes', [0.3024], P)]
    k8 = [('10  constant LR 3e-6 to 1.4e-5', [0.2530], C), ('10  PBT, every 5 epochs', [0.2359], P), None,
          ('22  standard ParT, start 1e-3', [0.1635, 0.1623, 0.1634], S),
          ('24  standard ParT, start 3e-3', [0.1611, 0.1597, 0.1623], S),
          ('33  standard ParT, start 1e-2', [0.1599], S),
          ('23  PBT', [0.1812], P), ('25  PBT + probes', [0.1699], P), ('26  PBT + probes + final decay', [0.1655], P),
          ('27-28  ensemble of 22 and 24 (3 seeds)', [0.1530, 0.1519, 0.1540], E)]
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.6, 10.5), gridspec_kw=dict(height_ratios=[len(k800), len(k8)]))
    for ax, rows, lim in ((a, k800, (0.297, 0.330)), (b, k8, (0.145, 0.262))):
        n = len(rows)
        labels = []
        for i, row in enumerate(rows):
            y = n - 1 - i
            if row is None:
                ax.axhline(y, color='0.8', lw=0.8)
                labels.append((y, ''))
                continue
            name, vals, (m, c, f) = row
            ax.axhline(y, color='0.94', lw=0.6, zorder=0)
            ax.plot(vals, [y] * len(vals), m, color=c, mfc=f, mew=1.4, ms=7, ls='none')
            labels.append((y, re.sub(r'(\d(?:\.\d)?)e-(\d)', r'$\1\\times10^{-\2}$', name)))
        ax.set_yticks([y for y, _ in labels], [t for _, t in labels], fontsize=11.5)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(*lim)
        ax.set_xlabel('Test (2.8M) mistag (%)   (lower is better)')
    a.set_title('(a) Training (800k)', loc='left', fontsize=13)
    b.set_title('(b) Training (8M)', loc='left', fontsize=13)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, color=c, mfc=f, mew=1.4, ms=7, ls='none', label=l) for (m, c, f), l in
         ((C, 'constant LR'), (S, 'standard ParT'), (P, 'PBT'), (E, 'ensemble of two'))]
    b.legend(handles=h, loc='lower right', fontsize=11)
    save(fig, 'history.png')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_history, fig1, fig3, fig4):
        f()
