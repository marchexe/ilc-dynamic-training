#!/usr/bin/env python3
"""One simple figure per step of docs/report.rst.

    .venv/bin/python scripts/reports/plot_report_story.py

Writes published/results/report/step{1..5}.png. Values are the published values of docs/results.rst.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / 'published/results/report'
INK, INK2, GRID = '#191919', '#666666', '#e6e6e6'
FIXED, PBT, STD, ENS = '#257de0', '#f47230', '#00b894', '#6a4ce2'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def clean(ax, grid='x'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbars(ax, rows, xlabel, fmt, lim):
    """rows: (label, value, colour, text) from top to bottom."""
    n = len(rows)
    for i, (label, v, c, text) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, height=0.6, color=c)
        ax.text(v + lim * 0.01, y, text or fmt.format(v), va='center', fontsize=11, color=INK)
    ax.set_yticks(range(n), [r[0] for r in reversed(rows)], color=INK)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, lim)
    ax.set_xlabel(xlabel)
    clean(ax)


def step1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 2.6))
    hbars(a, [('fixed LR', 0.3269, FIXED, None), ('PBT', 0.3225, PBT, '0.3225  (1.4% better)')],
          'mistag on Test (2.8M), %', '{:.4f}', 0.42)
    a.set_title('800k training jets', loc='left', color=INK)
    hbars(b, [('fixed LR', 0.2572, FIXED, None), ('PBT', 0.2375, PBT, '0.2375  (7.7% better)')],
          'mistag on Test (9.0M), %', '{:.4f}', 0.34)
    b.set_title('8M training jets', loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'step1.png', dpi=150)
    plt.close(fig)


def step2():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axvspan(3e-6, 1.4e-5, color='#e8f1fc')
    ax.text(6.5e-6, 0.3285, 'LRs used in\nstep 1', ha='center', va='top', fontsize=10, color=FIXED)
    pts = [(1.4e-5, 0.3269), (3e-5, 0.3146), (8e-5, 0.3018), (1.2e-4, 0.3033)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=FIXED, linewidth=2, markersize=8,
            label='fixed LR: best model of each LR range')
    ax.axhline(0.3225, color=PBT, linestyle='--', linewidth=1.5)
    ax.text(2.2e-4, 0.3225, 'PBT of step 1', va='bottom', ha='right', fontsize=10, color=PBT)
    ax.axhline(0.3003, color=PBT, linewidth=1.5)
    ax.text(2.2e-4, 0.3003, 'faster PBT, wider LR range', va='bottom', ha='right', fontsize=10, color=PBT)
    ax.set_xscale('log')
    ax.set_xlim(2.5e-6, 2.5e-4)
    ax.set_ylim(0.297, 0.330)
    ax.set_xlabel('learning rate (log scale)')
    ax.set_ylabel('mistag on Test (2.8M), %')
    ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=10)
    ax.set_title('800k training jets: a higher fixed LR is as good as PBT', loc='left', color=INK)
    clean(ax, 'both')
    fig.tight_layout()
    fig.savefig(OUT / 'step2.png', dpi=150)
    plt.close(fig)


def gain_chart(name, rows, mean, ci, title, colour, lim):
    fig, ax = plt.subplots(figsize=(11, 2.9))
    labels = [r[0] for r in rows] + ['mean of 3 seeds']
    values = [r[1] for r in rows] + [mean]
    n = len(labels)
    for i, (label, v) in enumerate(zip(labels, values)):
        y = n - 1 - i
        last = i == n - 1
        ax.barh(y, v, height=0.6, color=colour, alpha=1 if last else 0.55)
        x = ci[1] if last else v
        ax.text(x + lim * 0.012, y, f'{v:.2f}% better' + (f'   (95% interval {ci[0]:.2f} – {ci[1]:.2f}%)' if last else ''),
                va='center', fontsize=11, color=INK, fontweight='bold' if last else 'normal')
    ax.plot(ci, [0, 0], color=INK, linewidth=1.8)
    for c in ci:
        ax.plot([c, c], [-0.15, 0.15], color=INK, linewidth=1.8)
    ax.set_yticks(range(n), list(reversed(labels)), color=INK)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, lim)
    ax.set_xlabel('improvement on Test (2.8M), %')
    ax.set_title(title, loc='left', color=INK)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def step4():
    fig, ax = plt.subplots(figsize=(11, 2.9))
    rows = [('PBT', 12.4), ('PBT + fix 1 (probes)', 5.4), ('PBT + fix 1 + fix 2 (final decay)', 2.7)]
    for i, (label, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.6, color=PBT)
        ax.text(v + 0.15, y, f'{v:.1f}% worse', va='center', fontsize=11, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 15)
    ax.set_xlabel('worse than standard ParT on Test (2.8M), %')
    ax.set_title('8M training jets: PBT against standard ParT (0 = as good as standard ParT)', loc='left', color=INK)
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'step4.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    step1()
    step2()
    gain_chart('step3.png', [('seed 1', 0.67), ('seed 2', 0.85), ('seed 3', 0.91)], 0.81, (0.34, 1.34),
               '800k training jets: PBT against standard ParT', PBT, 2.4)
    step4()
    gain_chart('step5.png', [('seed 1', 5.05), ('seed 2', 4.86), ('seed 3', 5.14)], 5.01, (4.17, 5.84),
               '8M training jets: ensemble of two standard ParT models against the better one', ENS, 9.5)
