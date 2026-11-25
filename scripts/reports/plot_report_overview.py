#!/usr/bin/env python3
"""Overview figures of docs/report.rst (all trials of the project).

    .venv/bin/python scripts/reports/plot_report_overview.py

Writes published/results/report/{timeline,results,progress}.png. All values are copied from the
results pages (docs/results.rst and docs/report.rst); this script reads no run data.
"""
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / 'published/results/report'
INK, INK2, GRID = '#191919', '#666666', '#e6e6e6'
BASE, PBT, OTHER, INFRA = '#257de0', '#f47230', '#ee7fac', '#b4b4b4'
BETTER, WORSE, NEUTRAL = '#00b894', '#f47230', '#b4b4b4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def d(md, year=2026):
    m, day = md.split('-')
    return date(year, int(m), int(day))


# (trial, phase, start, end, kind)
TRIALS = [
    (1, 1, '09-15', '09-16', BASE), (2, 1, '09-16', '09-17', BASE), (3, 1, '09-17', '09-18', PBT),
    (4, 1, '09-18', '09-19', PBT), (5, 1, '09-19', '09-21', INFRA), (6, 1, '09-22', '09-28', PBT),
    (7, 1, '09-28', '09-28', INFRA),
    (8, 2, '10-12', '10-12', INFRA), (9, 2, '10-13', '10-13', INFRA), (10, 2, '10-13', '11-02', PBT),
    (11, 2, '10-17', '11-03', INFRA),
    (12, 3, '11-02', '11-03', BASE), (13, 3, '11-03', '11-03', BASE), (14, 3, '11-03', '11-04', BASE),
    (15, 3, '11-04', '11-04', PBT), (16, 3, '11-08', '11-09', PBT), (17, 3, '11-09', '11-09', PBT),
    (18, 4, '11-09', '11-13', PBT), (19, 4, '11-11', '11-12', PBT), (20, 4, '11-12', '11-12', OTHER),
    (21, 4, '11-12', '11-12', OTHER),
    (22, 5, '11-13', '11-15', BASE), (23, 5, '11-15', '11-16', PBT), (24, 5, '11-18', '11-20', BASE),
    (25, 5, '11-19', '11-21', PBT), (26, 5, '11-20', '11-22', PBT), (27, 5, '11-23', '11-23', OTHER),
    (28, 5, '11-23', '11-25', OTHER), (29, 5, '11-23', '11-23', OTHER), (30, 5, '11-23', '11-23', OTHER),
    (31, 5, '11-23', '11-24', OTHER), (32, 5, '11-24', '11-24', OTHER), (33, 5, '11-23', '11-25', BASE),
]
PHASES = {
    1: ('1  PBT vs fixed LRs', '800k training jets', 'PBT 1.4% better, but the fixed LRs were too low'),
    2: ('2  Large data, first time', '8M training jets', 'PBT 7.7% better, again against too-low LRs'),
    3: ('3  Higher fixed LRs', '800k training jets', 'Good fixed LRs equal PBT; Validation (150k) too small'),
    4: ('4  PBT vs standard ParT', '800k, Validation (1M)', 'PBT 0.8% better on 3 seeds'),
    5: ('5  Large data, standard ParT', '8M, Validation (1M)', 'PBT 2.7–12% worse; ensemble 5.0% better'),
}


def run_dates():
    """{trial: (start, end)} in JST from published/results/report/runs.csv (export_report_trials.py)."""
    import csv
    from datetime import datetime
    out = {}
    with open(OUT / 'runs.csv') as handle:
        for r in csv.DictReader(handle):
            s, e = (datetime.strptime(r[k], '%Y-%m-%d %H:%M') for k in ('start_jst', 'end_jst'))
            t = int(r['trial'])
            out[t] = (min(s, out.get(t, (s, e))[0]), max(e, out.get(t, (s, e))[1]))
    return out


def timeline():
    exact = run_dates()
    fig, ax = plt.subplots(figsize=(14, 6.4))
    y = 0
    ticks, labels = [], []
    for phase in sorted(PHASES, reverse=True):
        rows = [t for t in TRIALS if t[1] == phase]
        lanes = []  # end date per lane
        placed = []
        for t in sorted(rows, key=lambda r: d(r[2])):
            if t[0] in exact:
                s0, e0 = (mdates.date2num(x) for x in exact[t[0]])
            else:
                s0, e0 = mdates.date2num(d(t[2])), mdates.date2num(d(t[3])) + 1
            e0 = max(e0, s0 + 2.0)  # room for the number
            lane = next((i for i, end in enumerate(lanes) if end + 0.3 <= s0), None)
            if lane is None:
                lanes.append(e0)
                lane = len(lanes) - 1
            else:
                lanes[lane] = e0
            placed.append((t, lane, s0, e0))
        n = len(lanes)
        for t, lane, s0, e0 in placed:
            yy = y + (n - 1 - lane) * 0.62
            ax.barh(yy, e0 - s0, left=s0, height=0.5, color=t[4], edgecolor='white', linewidth=1.5)
            ax.text((s0 + e0) / 2, yy, str(t[0]), ha='center', va='center', fontsize=7.5,
                    color='white' if t[4] not in (INFRA, OTHER) else INK)
        mid = y + (n - 1) * 0.31
        ticks.append(mid)
        labels.append(f'{PHASES[phase][0]}\n{PHASES[phase][1]}')
        ax.text(1.01, mid, PHASES[phase][2], transform=ax.get_yaxis_transform(), va='center', fontsize=9,
                color=INK)
        y += n * 0.62 + 0.55
        if phase > 1:
            ax.axhline(y - 0.55 / 2 - 0.31, color=GRID, linewidth=1)
    ax.set_yticks(ticks, labels, fontsize=9.5, color=INK)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(d('09-12'), d('11-28'))
    ax.set_ylim(-0.5, y - 0.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_title('All trials in time order, JST (number = trial in the tables below; phase 0, Jul–Aug, was set-up)',
                 loc='left', fontsize=11, color=INK)
    ax.legend(handles=[Patch(color=BASE, label='baseline (fixed LR, standard ParT)'), Patch(color=PBT, label='PBT'),
                       Patch(color=OTHER, label='other idea (ensemble, distillation, ...)'),
                       Patch(color=INFRA, label='set-up, check, evaluation')],
              loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.15, right=0.72, top=0.93, bottom=0.14)
    fig.savefig(OUT / 'timeline.png', dpi=150)
    plt.close(fig)


# (label, change %, interval or None, colour, note) top to bottom; negative = first named is better
RESULTS = [
    ('800k  PBT vs fixed LRs (LRs too low)', -1.36, None, 'weak'),
    ('800k  PBT vs fixed LR, 100 epochs (LR too low)', -1.44, None, 'weak'),
    ('800k  PBT every epoch vs every 5 epochs (2 seeds)', +0.30, None, 'neutral'),
    ('8M  PBT vs fixed LRs (LRs too low), Test (9.0M)', -7.67, None, 'weak'),
    ('800k  PBT vs best fixed LR, seed 12345', -1.00, None, 'better'),
    ('800k  PBT vs best fixed LR, seed 22345', +0.47, None, 'worse'),
    ('800k  PBT vs standard ParT (3 seeds)', -0.81, (-1.34, -0.34), 'better'),
    ('8M  PBT vs standard ParT', +12.4, None, 'worse'),
    ('8M  PBT + probes vs standard ParT', +5.4, None, 'worse'),
    ('8M  PBT + probes + final decay vs standard ParT', +2.7, None, 'worse'),
    ('8M  ensemble of 2 standard ParT vs best single (3 seeds)', -5.01, (-5.84, -4.17), 'better'),
    ('800k  ensemble of 2 vs best single (Validation)', 0.0, None, 'neutral'),
    ('8M  distillation vs plain (Validation)', +1.0, None, 'worse'),
]


def results():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    colours = {'better': BETTER, 'worse': WORSE, 'neutral': NEUTRAL, 'weak': '#ffffff'}
    n = len(RESULTS)
    for i, (label, v, ci, kind) in enumerate(RESULTS):
        yy = n - 1 - i
        hatch = '////' if kind == 'weak' else None
        ax.barh(yy, v if v else 0.08, left=0 if v else -0.04, height=0.62, color=colours[kind],
                edgecolor=INK2 if kind == 'weak' else 'white', hatch=hatch, linewidth=1)
        if ci:
            ax.plot(ci, [yy, yy], color=INK, linewidth=1.6)
            for c in ci:
                ax.plot([c, c], [yy - 0.15, yy + 0.15], color=INK, linewidth=1.6)
        x = (ci[0] if ci and v < 0 else v) if v else 0
        txt = f'{v:+.2f}%' if v else '≈ 0'
        ax.text(x + (0.25 if v >= 0 else -0.25), yy, txt, va='center', ha='left' if v >= 0 else 'right',
                fontsize=9, color=INK)
    ax.set_yticks(range(n), [r[0] for r in reversed(RESULTS)], fontsize=9.5, color=INK)
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color=INK, linewidth=1)
    ax.axhline(n - 4.5, color=GRID, linewidth=1, linestyle='--')
    ax.axhline(n - 7.5, color=GRID, linewidth=1, linestyle='--')
    ax.set_xlim(-10, 14)
    ax.set_xlabel('change of Test (2.8M) mistag (%)      ← first named is better   |   first named is worse →')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_title('Every comparison, in time order (bars with lines: 95% interval)', loc='left', fontsize=11,
                 color=INK)
    ax.legend(handles=[Patch(color=BETTER, label='better'), Patch(color=WORSE, label='worse'),
                       Patch(color=NEUTRAL, label='no clear difference'),
                       Patch(facecolor='white', edgecolor=INK2, hatch='////', label='comparator LR too low: not valid')],
              loc='lower right', frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.94, bottom=0.10)
    fig.savefig(OUT / 'results.png', dpi=150)
    plt.close(fig)


# Test (2.8M) value of each selected checkpoint at the date its run finished: (date, value, kind, epochs100)
P800 = [('09-16', 0.3269, BASE, 0), ('09-17', 0.3156, BASE, 1), ('09-18', 0.3225, PBT, 0), ('09-19', 0.3111, PBT, 1),
        ('09-23', 0.3177, PBT, 0), ('09-25', 0.3162, PBT, 0), ('09-25', 0.3167, PBT, 0), ('09-28', 0.3163, PBT, 0),
        ('11-03', 0.3146, BASE, 0), ('11-03', 0.3018, BASE, 0), ('11-04', 0.3033, BASE, 0), ('11-04', 0.3003, PBT, 0),
        ('11-08', 0.3041, PBT, 0), ('11-09', 0.3027, BASE, 0), ('11-09', 0.3026, PBT, 0),
        ('11-10', 0.3029, OTHER, 0), ('11-10', 0.3028, BASE, 0), ('11-10', 0.3008, PBT, 0),
        ('11-11', 0.3038, OTHER, 0), ('11-11', 0.3013, BASE, 0), ('11-11', 0.3012, PBT, 0),
        ('11-12', 0.3024, OTHER, 0), ('11-12', 0.3024, BASE, 0), ('11-13', 0.2996, PBT, 0),
        ('11-20', 0.3024, PBT, 0)]
P8M = [('10-19', 0.2530, BASE), ('10-21', 0.2359, PBT),
       ('11-15', 0.1635, OTHER), ('11-16', 0.1812, PBT), ('11-20', 0.1611, OTHER), ('11-21', 0.1699, PBT),
       ('11-22', 0.1655, PBT), ('11-23', 0.1530, '#6a4ce2'), ('11-24', 0.1597, OTHER), ('11-24', 0.1623, OTHER),
       ('11-24', 0.1519, '#6a4ce2'), ('11-25', 0.1623, OTHER), ('11-25', 0.1634, OTHER),
       ('11-25', 0.1540, '#6a4ce2'), ('11-25', 0.1599, OTHER)]


def progress():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, pts, title, lo, hi in ((a, P800, '800k training jets', 0.296, 0.330),
                                   (b, P8M, '8M training jets', 0.145, 0.260)):
        for p in pts:
            hollow = len(p) > 3 and p[3]
            ax.scatter(d(p[0]), p[1], s=46, color='white' if hollow else p[2], edgecolor=p[2], linewidth=1.6,
                       zorder=3)
        best, line = 9, []
        for p in sorted(pts, key=lambda p: d(p[0])):
            if not (len(p) > 3 and p[3]):
                best = min(best, p[1])
            line.append((d(p[0]), best))
        ax.step([x for x, _ in line], [v for _, v in line], where='post', color=INK2, linewidth=1.2, zorder=2)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc='left', fontsize=11, color=INK)
        ax.set_ylabel('Test (2.8M) mistag (%), lower is better')
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    a.set_xlim(d('09-10'), d('11-28'))
    b.set_xlim(d('10-12'), d('11-28'))
    a.annotate('LRs too low', (d('09-18'), 0.3225), (d('09-28'), 0.3265), fontsize=9, color=INK2,
               arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    a.annotate('higher fixed LRs', (d('11-03'), 0.3018), (d('10-05'), 0.3005), fontsize=9, color=INK2,
               arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    b.annotate('LRs too low\n(trial 10)', (d('10-21'), 0.2359), (d('10-25'), 0.215), fontsize=9, color=INK2,
               arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    b.annotate('PBT', (d('11-16'), 0.1812), (d('11-04'), 0.190), fontsize=9, color=INK2,
               arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    b.annotate('ensembles', (d('11-23'), 0.1530), (d('11-05'), 0.150), fontsize=9, color=INK2,
               arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    handles = [Line2D([], [], marker='o', linestyle='', color=c, markersize=7, label=l)
               for c, l in ((BASE, 'fixed LR'), (OTHER, 'standard ParT'), (PBT, 'PBT'), ('#6a4ce2', 'ensemble'))]
    handles += [Line2D([], [], marker='o', linestyle='', markerfacecolor='white', markeredgecolor=INK2, markersize=7,
                       label='100 epochs (not comparable)'),
                Line2D([], [], color=INK2, linewidth=1.2, label='best so far')]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=9)
    fig.suptitle('Test (2.8M) value of every selected model, by date', x=0.01, ha='left', fontsize=12, color=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.17, wspace=0.22)
    fig.savefig(OUT / 'progress.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    timeline()
    results()
    progress()
