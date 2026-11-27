#!/usr/bin/env python3
"""Figures of the research publication docs/_extra/overview.html (written to docs/_extra/pub/*.svg).

    .venv/bin/python scripts/reports/publication_figures.py             # from tracked data only
    .venv/bin/python scripts/reports/publication_figures.py --refresh   # first re-extract the data from runs/

All inputs are tracked in the repository: Test values of the selected checkpoints are those of docs/results.rst
(written below); loss curves come from published/results/sel1m/train8m_followups.json; LR and Validation
trajectories and the model-pair analyses come from published/results/report/figure_data.json, which --refresh
extracts from the local run manifests (runs/pbt) and pair analyses (runs/eval). Nothing is fitted or smoothed.
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
DATA = ROOT / 'published/results/report/figure_data.json'
PRIMARY = 'validation_total_reference_mistag_geomean_percent'

INK, MUTED, GRID = '#1d1d1f', '#6e6e73', '#e6e6e9'
STD, PBT, ENS, CONST = '#2b5797', '#c4532f', '#2a7d64', '#8a8a8f'
plt.rcParams['svg.hashsalt'] = 'ilc-report'  # deterministic element ids: same data, same file
plt.rcParams.update({
    'font.family': ['Lato', 'DejaVu Sans'], 'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 4, 'ytick.major.size': 4,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'svg.fonttype': 'path', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=0.08, metadata={'Date': None})
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
    t = data()['trajectories_8m_seed12345']
    std, pbt, fix = t['standard_part_3e-3 (#24)'], t['pbt (#23)'], t['pbt_probe_decay (#26)']
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


def extract_pairs():
    """Model pairs on Validation (1M): (label, n pairs, disagreement %, improvement % = −(ensemble/best − 1)).
    Sources: runs/eval/diversity_8m_explore_v1, ensemble_800k_explore_v1 and ensemble_seed32345_v1 (all seeds)."""
    ev = ROOT / 'runs/eval'
    p8 = json.loads((ev / 'diversity_8m_explore_v1/pairs.json').read_text())['pairs']
    p800 = json.loads((ev / 'ensemble_800k_explore_v1/pairs.json').read_text())['pairs']
    qa = json.loads((ev / 'ensemble_seed32345_v1/result.json').read_text())['question_A']

    def pick(pairs, test):
        sel = [x for x in pairs if test(x['a'], x['b'])]
        return [x['disagree_percent'] for x in sel], [-x['gain_vs_best_percent'] for x in sel]

    ckpt = pick(p8, lambda a, b: {a, b} == {'std3e-3_e62', 'std3e-3_e67'})
    pbt = pick(p8, lambda a, b: a.startswith('pbtdecay_') and b.startswith('pbtdecay_'))
    k800 = pick(p800, lambda a, b: a.endswith('_std100e-6') and b.endswith('_std100e-6'))
    same = ([x['disagree_percent'] for x in qa['same_lr']], [-x['gain_percent'] for x in qa['same_lr']])
    diff = ([x['disagree_percent'] for x in qa['different_lr']], [-x['gain_percent'] for x in qa['different_lr']])
    return [('Checkpoints of one run, 8M', ckpt, False), ('Models of one PBT run, 8M', pbt, False),
            ('Separate seeds, 800k', k800, False),
            ('Separate runs, 8M: same LR', same, True), ('Separate runs, 8M: different LR', diff, True)]


def extract():
    """Re-extract the figure inputs from the local run records into the tracked figure_data.json."""
    out = dict(
        description='Inputs of the report figures, extracted from runs/pbt manifests and runs/eval pair analyses by '
                    'scripts/reports/publication_figures.py --refresh. Trajectories: (training epoch, lowest LR, '
                    'highest LR, best Validation (1M) mistag so far in %). Pairs: disagreement on Validation (1M) in % '
                    'and ensemble improvement in % (mistag reduction against the better model of the pair).',
        trajectories_8m_seed12345={
            'standard_part_3e-3 (#24)': trajectory('sel1m_train8m_standard_part_3e-3'),
            'pbt (#23)': trajectory('sel1m_train8m_fast_pbt'),
            'pbt_probe_decay (#26)': trajectory('sel1m_train8m_fast_pbt_probe_decay')},
        ensemble_pairs=[dict(label=lab, disagreement_percent=dis, improvement_percent=imp, gain=gain)
                        for lab, (dis, imp), gain in extract_pairs()])
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')


_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = json.loads(DATA.read_text())
    return _DATA


def fig_ensemble():
    """Two aligned panels over the same model pairs: disagreement (left) and ensemble improvement (right)."""
    rows = [(r['label'], (r['disagreement_percent'], r['improvement_percent']), r['gain']) for r in data()['ensemble_pairs']]
    good, grey = '#4f8a6a', '#9a9aa0'
    ys = [4.2, 3.2, 2.2, 1.0, 0.35]  # the two 8M sub-rows sit together
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.3), sharey=True,
                               gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.08))
    for (lab, (dis, imp), gain), y in zip(rows, ys):
        c = good if gain else grey
        n = len(dis)
        a.plot([min(dis), max(dis)], [y, y], color=c, lw=7, solid_capstyle='butt')
        if n == 1:
            a.plot(dis[0], y, '|', color=c, ms=14, mew=3)
        rng = f'{min(dis):.2f}' if n == 1 else f'{min(dis):.2f}–{max(dis):.2f}'
        a.text(max(dis) + 0.07, y, rng, va='center', fontsize=10, color=INK)
        a.text(-0.05, y, f'{lab}', transform=a.get_yaxis_transform(), ha='right', va='center', fontsize=10.5, color=INK)
        a.text(-0.05, y - 0.3, f'{n} pair{"s" if n > 1 else ""}', transform=a.get_yaxis_transform(), ha='right',
               va='center', fontsize=9, color=MUTED)
        lo, hi = min(imp), max(imp)
        b.plot([lo, hi] if n > 1 else [lo - 0.03, lo + 0.03], [y, y], color=c, lw=7, solid_capstyle='butt')
        if gain:
            mean = sum(imp) / n
            b.plot(mean, y, '|', color=INK, ms=14, mew=2)
            txt = f'mean {mean:.2f}%  (range {lo:.1f} to {hi:.1f})'
        else:
            txt = 'no observed gain  (' + (f'{lo:+.2f}%' if n == 1 else f'{lo:+.2f} to {hi:+.2f}%') + ')'
        b.text(max(hi, 0.2) + 0.2, y, txt.replace('-', '−'), va='center', fontsize=10, color=INK if gain else MUTED)
    a.set_xlim(0, 2.6)
    a.set_xlabel('jets with different predicted class (%)')
    a.set_title('Model disagreement', loc='left', fontweight='bold')
    b.axvline(0, color=INK, lw=1)
    b.set_xlim(-1.2, 11)
    ticks = [0, 2, 4, 6]
    b.set_xticks(ticks, ['0'] + [f'+{t}%' for t in ticks[1:]])
    b.set_xlabel('mistag reduction against the better model (%)')
    b.set_title('Ensemble improvement', loc='left', fontweight='bold')
    for ax in (a, b):
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(-0.25, 4.75)
        grid(ax)
    b.spines['left'].set_visible(False)
    save(fig, 'fig6_ensemble.svg')

if __name__ == '__main__':
    import sys
    if '--refresh' in sys.argv:
        extract()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_lr, fig_regimes, fig_mechanism, fig_ensemble):
        f()
    print(sorted(p.name for p in OUT.iterdir()))
