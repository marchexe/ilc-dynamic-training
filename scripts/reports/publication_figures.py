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
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw=dict(wspace=0.3))
    for ax, key, title in zip(axes, ('800k', '8M'), ('Training (800k)', 'Training (8M)')):
        pts = data[key]['epochs']
        e = [p['epoch'] - 17 for p in pts]
        tr, va = [p['train'] for p in pts], [p['selection'] for p in pts]
        ax.plot(e, tr, color=STD, lw=1.8, ls=(0, (4, 2)))
        ax.plot(e, va, color=STD, lw=2)
        ax.text(e[-1] + 0.8, tr[-1], 'training data', va='center', fontsize=10, color=STD)
        ax.text(e[-1] + 0.8, va[-1], 'Validation', va='center', fontsize=10, color=STD, fontweight='bold')
        ax.set_xlim(1, 50)
        ax.set_xlabel('epoch')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        grid(ax, 'y')
    axes[0].set_ylabel('cross-entropy loss')
    axes[0].annotate('training loss falls below the\nValidation loss (overfitting)', (35, 0.2728), (16, 0.2900),
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
    a.text(3, 3.9e-3, 'standard ParT #24', color=STD, fontsize=10.5)
    a.text(12, 1.6e-4 * 0.5, 'PBT #23: range\nof its five models', color=PBT, fontsize=10.5, va='top')
    a.text(37, 4.5e-3, 'final decay', color=MUTED, fontsize=10)
    a.set_yscale('log')
    a.set_ylim(5e-6, 8e-3)
    a.set_ylabel('learning rate')
    a.set_title('Learning rate', loc='left', fontweight='bold')
    b.plot([p[0] for p in std], [p[3] for p in std], color=STD, lw=2)
    b.plot([p[0] for p in pbt], [p[3] for p in pbt], color=PBT, lw=2)
    b.plot([p[0] for p in fix], [p[3] for p in fix], color=PBT, lw=1.5, ls='--')
    b.text(50.8, std[-1][3], 'standard ParT #24', va='center', color=STD, fontsize=10.5)
    b.text(50.8, pbt[-1][3], 'PBT #23', va='center', color=PBT, fontsize=10.5)
    b.text(50.8, fix[-1][3] + 0.0016, 'PBT #26: probes,\nfinal decay,\nhigher LRs', va='bottom', color=PBT, fontsize=10)
    b.set_ylim(0.155, 0.225)
    b.set_ylabel('best Validation (1M) mistag so far (%)')
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


def _test(campaign):
    """Test (2.8M) composite mistag (%) of every checkpoint of one evaluation campaign, full precision."""
    a = json.loads((ROOT / 'runs/eval' / campaign / 'analysis.json').read_text())
    return {k: v[PRIMARY] for k, v in a['metrics'].items()}


def _scan(run):
    """Constant-LR run: (start LR, best Validation mistag of that member over the run), sorted by LR."""
    m = json.loads((RUNS / run / 'manifest.json').read_text())
    best = {}
    for g in m['generations']:
        for k, w in g['workers'].items():
            v = (w.get('metrics') or {}).get(PRIMARY)
            if v is not None:
                best[k] = min(best.get(k, v), v)
    lr0 = {k: w['lr'] for k, w in m['generations'][0]['workers'].items()}
    return sorted((lr0[k], v) for k, v in best.items())


def _selected_lr(run):
    """LR at the selected (Validation global best) member-epoch of a run."""
    m = json.loads((RUNS / run / 'manifest.json').read_text())
    return min((w['metrics'][PRIMARY], w['lr']) for g in m['generations'] for w in g['workers'].values()
               if (w.get('metrics') or {}).get(PRIMARY) is not None)[1]


def extract_overview():
    """Values of the research-overview figure (Figure 1), each from its original record."""
    import csv
    ev = ROOT / 'runs/eval'
    hi, ce, fa, s2 = (_test(c) for c in ('holdout_high_lr_grid_v1', 'holdout_fixed_lr_ceiling_v1',
                                         'holdout_fast_pbt_v1', 'holdout_fast_pbt_seed22345_v1'))
    sealed = {r['id']: float(r['primary']) for r in
              csv.DictReader(open(ROOT / 'published/results/large_dataset/sealed_test_checkpoints.csv'))}
    boot = json.loads((ev / 'holdout_block_bootstrap_v1/block100.json').read_text())['comparisons']
    pooled = json.loads((ev / 'holdout_block_bootstrap_v1/pooled_block100.json').read_text())
    t8, t8b, pr, pd = (_test(c) for c in ('holdout_sel1m_train8m_v1', 'holdout_sel1m_train8m_lr3e-3_v1',
                                          'holdout_sel1m_probe_v1', 'holdout_sel1m_probe_decay_v1'))
    e1 = json.loads((ev / 'holdout_ensemble_8m_dev_v1/result.json').read_text())
    e3 = json.loads((ev / 'ensemble_seed32345_v1/result.json').read_text())
    seeds = ('12345', '22345', '32345')

    def bs(left, right, seed):
        c = next(c for c in boot if c['left'] == left and c['right'] == right and c['seed'] == 's' + seed)
        return dict(left=c['left_primary'], right=c['right_primary'], change=c['change_percent'],
                    interval=c['interval_percent'])
    models = e3['models']
    other = {'12345': e1['values']['standard_sel1m_8m_s12345'],  # Test value of the 10^-3 model of each seed
             '22345': json.loads((ev / 'ensemble_seed22345_v1/holdout/s2_1e-3/done.json').read_text())['metric'],
             '32345': json.loads((ev / 'ensemble_seed32345_v1/holdout/s3_1e-3/done.json').read_text())['metric']}
    ens = []
    for i, seed in enumerate(seeds, 1):
        q = e3['question_B']['per_seed'][f's{i}']
        interval = (q.get('interval_percent') or (e1['comparisons'][0]['interval_percent'] if i == 1 else
                    json.loads((ev / 'ensemble_seed22345_v1/result.json').read_text())['question_B']['interval_percent']))
        ens.append(dict(seed=seed, trial='27' if i == 1 else '28', better=q['left_holdout'], other=other[seed],
                        ensemble=q['ensemble'],
                        change=q['change_percent'], interval=interval,
                        model_3e3={k: models[f's{i}_3e-3'][k] for k in ('run', 'member', 'epoch')},
                        model_1e3={k: models[f's{i}_1e-3'][k] for k in ('run', 'member', 'epoch')}))
    return dict(
        sources='runs/eval/<campaign>/analysis.json (Test (2.8M)); published/results/large_dataset/'
                'sealed_test_checkpoints.csv (Test (9.0M)); runs/eval/holdout_block_bootstrap_v1 (intervals); '
                'runs/eval/holdout_ensemble_8m_dev_v1, ensemble_seed22345_v1, ensemble_seed32345_v1 (ensembles); '
                'runs/pbt/<run>/manifest.json (Validation (150k) per constant-LR model and selected LRs). '
                'Epochs of ensemble models are Weaver epochs (training epoch + 17).',
        initial=dict(const_800k=hi['fixed50_s12345'], pbt_800k=hi['windowed50_s12345'],
                     const_8m=sealed['fixed_lr__selected_best'], pbt_8m=sealed['windowed_pbt_v2__selected_best'],
                     grid=[3e-6, 1.4e-5], const_lr=_selected_lr('foundation_fixed_lr_50epochs_20260916'),
                     pbt_lr_800k=_selected_lr('windowed_pbt_v2'), pbt_lr_8m=_selected_lr('stage3_train8m_windowed_pbt_v2_v1')),
        scan={'1': _scan('foundation_fixed_lr_50epochs_20260916'), '12': _scan('fixed_lr_high_grid_50epochs'),
              '13': _scan('fixed_lr_ceiling_grid_50epochs'), '14': _scan('fixed_lr_top_grid_50epochs'),
              '16': _scan('fixed_lr_best_grid_50epochs_seed22345')},
        baseline=[  # (label, trial line, constant-LR Test, PBT Test, seed)
            ('September', '#1 grid vs PBT #3', hi['fixed50_s12345'], hi['windowed50_s12345'], '12345'),
            ('November', '#13 grid vs fast PBT #15', ce['fixed_ceiling50_s12345'], fa['windowed_fast50_s12345'], '12345'),
            ('November', '#14 grid vs fast PBT #15', fa['fixed_top50_s12345'], fa['windowed_fast50_s12345'], '12345'),
            ('November 9', '#16 new grid vs fast PBT', s2['fixed_best50_s22345'], s2['windowed_fast50_s22345'], '22345')],
        sel800k=dict(seeds=[dict(seed=s, standard=bs('standard', 'pbt', s)['left'], constant=bs('fixed', 'pbt', s)['left'],
                                 pbt=bs('standard', 'pbt', s)['right'], pbt_vs_standard=bs('standard', 'pbt', s),
                                 pbt_vs_constant=bs('fixed', 'pbt', s)) for s in seeds],
                     mean_change=pooled['mean_change_percent'], mean_interval=pooled['mean_interval_percent']),
        train8m=dict(std_1e3=t8['standard_sel1m_8m_s12345'], pbt=t8['pbt_sel1m_8m_s12345'],
                     std_3e3=t8b['standard3e3_sel1m_8m_s12345'], probe=pr['pbt_probe_sel1m_8m_s12345'],
                     probe_decay=pd['pbt_probe_decay_sel1m_8m_s12345']),
        ensemble=dict(seeds=ens, mean_change=e3['question_B']['mean_change_percent'],
                      mean_interval=e3['question_B']['pooled_interval_percent']))


def extract_all_pairs():
    """Every scored model pair on Validation (1M): 36 pairs of nine seed-12345 8M models (#30), 15 pairs of the six
    8M standard ParT models of three seeds (#28/#30), 105 pairs of the fifteen 800k models of #18 (#29). The one pair
    present in two sources (#24 epoch 45 with the #22 10^-3 model) is kept once. gap: how much worse the weaker model
    of the pair is than the better one, in %."""
    ev = ROOT / 'runs/eval'
    run = {'std1e-3_e67': 'std22', 'std3e-4_e67': 'std22', 'std3e-3_e62': 'std24', 'std3e-3_e67': 'std24',
           'pbt_best_e66': 'pbt23', 'pbtprobe_best_e67': 'pbt25'}
    out = []

    def add(sample, group, a, b, single_a, single_b, dis, imp):
        lo, hi = sorted((single_a, single_b))
        out.append(dict(sample=sample, group=group, a=a, b=b, disagreement=dis, improvement=imp,
                        gap=100 * (hi / lo - 1)))
    for x in json.loads((ev / 'diversity_8m_explore_v1/pairs.json').read_text())['pairs']:
        a, b = x['a'], x['b']
        if {a, b} == {'std3e-3_e62', 'std1e-3_e67'}:
            continue  # the same pair is in the three-seed set below
        ra, rb = run.get(a, 'pbt26'), run.get(b, 'pbt26')
        group = ('one_run' if ra == rb and (ra != 'std22') else
                 'separate_standard' if ra.startswith('std') and rb.startswith('std') else 'with_pbt')
        add('8M', group, a, b, x['single_a'], x['single_b'], x['disagree_percent'], -x['gain_vs_best_percent'])
    qa = json.loads((ev / 'ensemble_seed32345_v1/result.json').read_text())['question_A']
    for x in qa['same_lr'] + qa['different_lr']:
        add('8M', 'separate_standard', *x['pair'], x['left'], x['right'], x['disagree_percent'], -x['gain_percent'])
    for x in json.loads((ev / 'ensemble_800k_explore_v1/pairs.json').read_text())['pairs']:
        add('800k', '800k', x['a'], x['b'], x['single_a'], x['single_b'], x['disagree_percent'],
            -x['gain_vs_best_percent'])
    return out


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
                        for lab, (dis, imp), gain in extract_pairs()],
        overview=extract_overview(),
        pairs=extract_all_pairs())
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')


_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = json.loads(DATA.read_text())
    return _DATA


def fig_ensemble():
    """Every scored model pair on Validation (1M): ensemble improvement against disagreement (left) and against the
    quality gap between the two models (right). Exploratory: the pairs were scored after the models were trained."""
    pairs = data()['pairs']
    groups = [  # (group, label, style); drawn in this order
        ('800k', 'Training (800k): two of the 15 models of #18 (#29)',
         dict(marker='o', ms=4.5, color='#b9b9bf', mfc='#b9b9bf', mew=0)),
        ('with_pbt', 'Training (8M): a PBT model + a model of another run (#30)',
         dict(marker='s', ms=6.5, color=PBT, mfc='white', mew=1.4)),
        ('one_run', 'Training (8M): two models of one run (#30)',
         dict(marker='^', ms=7.5, color=INK, mfc='white', mew=1.4)),
        ('separate_standard', 'Training (8M): two separately trained standard ParT (#28, #30)',
         dict(marker='o', ms=7, color=ENS, mfc=ENS, mew=0)),
    ]
    chosen = [{'s1_3e-3', 's1_1e-3'}, {'s2_3e-3', 's2_1e-3'}, {'s3_3e-3', 's3_1e-3'}]  # the #27/#28 ensembles
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    for g, lab, st in groups:
        sel = [x for x in pairs if x['group'] == g]
        cnt = f'{lab} · {len(sel)} pairs'
        for ax, key in ((a, 'disagreement'), (b, 'gap')):
            ax.plot([x[key] for x in sel], [x['improvement'] for x in sel], ls='none', zorder=3 if g != '800k' else 2,
                    label=cnt if ax is a else None, **st)
    ring = [x for x in pairs if {x['a'], x['b']} in chosen]
    for ax, key in ((a, 'disagreement'), (b, 'gap')):
        ax.plot([x[key] for x in ring], [x['improvement'] for x in ring], 'o', ms=12, mfc='none', color=INK, mew=1.2,
                zorder=4)
        ax.axhline(0, color=INK, lw=0.9, zorder=1)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        grid(ax, 'both')
    best = max(ring, key=lambda x: x['gap'])
    b.annotate('rings: the ensembles of #27, #28', (best['gap'], best['improvement']),
               (best['gap'] + 2.2, best['improvement'] + 1.0), fontsize=10, color=INK,
               arrowprops=dict(arrowstyle='-', color=INK, lw=0.8))
    a.set_xlim(0, 2.45)
    a.set_xlabel('jets with a different predicted class (%)')
    a.set_title('Disagreement alone does not predict the gain', loc='left', fontweight='bold')
    b.set_xlim(-0.5, 17)
    b.set_xticks([0, 5, 10, 15], ['0', '5%', '10%', '15%'])
    a.set_xticks([0, 0.5, 1, 1.5, 2], ['0', '0.5%', '1%', '1.5%', '2%'])
    b.set_xlabel('weaker model of the pair: mistag higher by (%)')
    b.set_title('A much weaker partner removes the gain', loc='left', fontweight='bold')
    ticks = [-2, 0, 2, 4, 6]
    a.set_yticks(ticks, [f'{t}% worse'.replace('-', '') if t < 0 else ('0' if t == 0 else f'{t}% better') for t in ticks])
    a.set_ylabel('ensemble against the better model')
    a.set_ylim(-2.7, 7)
    a.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.19), ncol=2, fontsize=10.5, handletextpad=0.3,
             columnspacing=1.0, borderaxespad=0)
    save(fig, 'fig6_ensemble.svg')


# ---------------------------------------------------------------------------------------------------------------
# Figure 1: the research trajectory. One band per stage, top to bottom in time. Within a band the left axes hold
# absolute Test mistag values and the right axes relative changes (in A and B: learning rates). No line connects
# values of different bands, because their training samples, Test samples, seeds and comparators differ.
# ---------------------------------------------------------------------------------------------------------------
from matplotlib.patches import Ellipse  # noqa: E402

MARK = dict(std=dict(marker='o', color=STD, mfc=STD), std2=dict(marker='o', color=STD, mfc='white'),
            const=dict(marker='o', color=CONST, mfc='white'), pbt=dict(marker='s', color=PBT, mfc=PBT),
            ens=dict(marker='D', color=ENS, mfc=ENS))  # one encoding for every figure of the report


def _mk(ax, x, y, kind, size=9, zorder=3):
    ax.plot([x], [y], ls='none', ms=size - (1 if kind == 'ens' else 0), mew=1.7, zorder=zorder, **MARK[kind])


def _legend(ax, items, **kw):
    for kind, lab in items:
        ax.plot([], [], ls='none', ms=8, mew=1.7, label=lab, **MARK[kind])
    ax.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=len(items), fontsize=10, handletextpad=0.35,
              columnspacing=1.2, borderaxespad=0.25, **kw)


def _row_labels(ax, rows, x=-0.025, size=10.5, sub_size=9.5):
    """Labels left of the axes: (y, bold label, muted detail)."""
    for y, lab, sub in rows:
        ax.text(x, y + (0.15 if sub else 0), lab, transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=size, color=INK, fontweight='bold' if sub else 'normal')
        if sub:
            ax.text(x, y - 0.2, sub, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=sub_size,
                    color=MUTED)


def _rows_axes(ax, n):
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    grid(ax)


def _rel_axis(ax, lo, hi, step, label):
    """x axis of relative mistag changes, read as 'better' left of 0 and 'worse' right of 0."""
    ticks = [round(t * step, 6) for t in range(int(-(-lo // step)), int(hi // step) + 1)]
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks, [f'{abs(t):g}%' if t else '0' for t in ticks])
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_xlabel(label)
    tr, d = ax.get_xaxis_transform(), (hi - lo) * 0.015
    if lo < 0:
        ax.text(-d, 1.0, '← better', transform=tr, ha='right', va='bottom', fontsize=10, color=MUTED)
    if hi > 0:
        ax.text(d, 1.0, 'worse →', transform=tr, ha='left', va='bottom', fontsize=10, color=MUTED)


def _word(c, digits=1):
    return f'{abs(c):.{digits}f}% {"better" if c < 0 else "worse"}'


def _intervals(ax, rows, value_x):
    """rows: (y, label, change %, (lo, hi) or None, colour, kind, emphasised). Value text in a column at value_x."""
    for y, lab, c, iv, col, kind, strong in rows:
        if iv:
            ax.plot(iv, [y, y], color=col, lw=2 if strong else 1.5, alpha=1 if strong else 0.55, zorder=2)
        ax.plot(c, y, ls='none', ms=9 if strong else 7, mew=1.6, zorder=3, **dict(MARK[kind], color=col))
        ax.text(value_x, y, _word(c, 2), transform=ax.get_yaxis_transform(), va='center', fontsize=10.5,
                color=INK, fontweight='bold' if strong else 'normal')
        ax.text(-0.025, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=10,
                color=INK if strong else MUTED, fontweight='bold' if strong else 'normal')


def fig_overview():
    o = data()['overview']
    W = 11.0
    plan = [('A', 2.35), ('t', 0.6), ('B', 3.0), ('t', 0.6), ('C', 2.85), ('t', 0.6), ('D', 2.85), ('t', 0.6),
            ('E', 2.45)]
    HEAD = 0.62
    H = 0.12 + sum(h + (HEAD if k != 't' else 0) for k, h in plan) + 0.05
    fig = plt.figure(figsize=(W, H))
    L, LW = 2.55, 3.5          # left axes (absolute values); row labels sit in 0.55-2.5 in
    R0, RW, VX = 7.25, 2.6, 1.03  # right axes; labels in 6.25-7.2 in; values right of the axes (axes fraction VX)
    TL = 0.2                   # x of the timeline rule

    def axes(x, top, w, h):
        return fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H])

    def ftext(x, top, s, **kw):
        fig.text(x / W, 1 - top / H, s, **kw)

    tops, y = {}, 0.12
    for k, h in plan:
        tops.setdefault(k, []).append(y)
        y += h + (HEAD if k != 't' else 0)
    st = {k: tops[k][0] for k in 'ABCDE'}
    fig.add_artist(plt.Line2D([TL / W] * 2, [1 - (st['A'] + 0.16) / H, 1 - (st['E'] + 0.16) / H], color='#c9c9cf',
                              lw=2, zorder=0))
    heads = {
        'A': ('Sep 16 – Oct 21', 'PBT is better than the constant-LR grid',
              'Training (800k) and (8M) · Validation (150k) · seed 12345 · one run per method · #1, #3, #10'),
        'B': ('Nov 3 – Nov 9', 'After tuning the constant-LR baseline, the PBT advantage is smaller and not consistent',
              'Training (800k) · Validation (150k) · Test (2.8M) · seed 12345, except #16 (seed 22345) · #12–#16'),
        'C': ('Nov 10 – Nov 13', 'Controlled comparison: PBT slightly better than standard ParT on every seed',
              'Training (800k) · Validation (1M) · Test (2.8M) · three seeds · each method 5 models × 50 epochs · #18'),
        'D': ('Nov 14 – Nov 23', 'With ten times more training data PBT is worse; changes reduce the deficit',
              'Training (8M) · Validation (1M) · Test (2.8M) · seed 12345 only · PBT 5 models, standard ParT best of 6 · '
              '#22–#26'),
        'E': ('Nov 23 – Nov 25', 'Averaging two separately trained standard ParT models is 5% better',
              'Training (8M) · Validation (1M) · Test (2.8M) · three seeds · two models per seed · #27, #28'),
    }
    for k, (when, title, meta) in heads.items():
        t = st[k] + 0.16
        fig.add_artist(Ellipse((TL / W, 1 - t / H), 0.16 / W, 0.16 / H, transform=fig.transFigure, facecolor='white',
                               edgecolor=INK, lw=1.6, zorder=2))
        ftext(0.45, t, k, fontsize=16, fontweight='bold', color=INK, va='center')
        ftext(0.8, t, when, fontsize=11, color=MUTED, va='center', fontweight='bold')
        ftext(2.3, t, title, fontsize=14, fontweight='bold', color=INK, va='center')
        ftext(2.3, t + 0.27, meta, fontsize=10.5, color=MUTED, va='center')
    trans = [
        ('The best constant LR was the highest LR of its grid, and PBT had moved above it.',
         'Was the gain only a higher LR? Extend the grid upwards.'),
        ('A 0.5% difference changed sign with seed and grid; Validation (150k) and Test disagreed.',
         'Select on Validation (1M); compare with standard ParT on three seeds.'),
        ('A gain below 1% on every seed with Training (800k).', 'Does it hold with ten times more training data?'),
        ('No change of the PBT schedule reached standard ParT with Training (8M).',
         'Look for a gain outside the LR schedule: average two models.'),
    ]
    for t, (a, b) in zip(tops['t'], trans):
        mid = t + 0.3
        fig.add_artist(plt.Polygon([[(TL - 0.075) / W, 1 - (mid - 0.07) / H], [(TL + 0.075) / W, 1 - (mid - 0.07) / H],
                                    [TL / W, 1 - (mid + 0.08) / H]], closed=True, color='#9a9aa0', zorder=1))
        first = fig.text(0.45 / W, 1 - mid / H, a, fontsize=11, color=MUTED, va='center')
        end = first.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi  # measured, in inches
        ftext(end + 0.12, mid, '→  ' + b, fontsize=11, color=INK, va='center', fontweight='bold')

    # --- A ---------------------------------------------------------------------------------------------------
    i = o['initial']
    top = st['A'] + HEAD
    for n, (key, title, xl, ticks) in enumerate([
            ('800k', 'Training (800k) · Test (2.8M)', (0.3205, 0.3290), [0.322, 0.324, 0.326, 0.328]),
            ('8m', 'Training (8M) · Test (9.0M)', (0.232, 0.262), [0.235, 0.245, 0.255])]):
        ax = axes(0.75 + n * 3.05, top + 0.72, 2.65, 1.05)
        c, p = i[f'const_{key}'], i[f'pbt_{key}']
        ax.plot([p, c], [0, 0], color='#d0d0d4', lw=3, zorder=1)
        _mk(ax, c, 0, 'const')
        _mk(ax, p, 0, 'pbt')
        ax.text(c, 0.3, f'{c:.4f}', ha='center', fontsize=10, color=MUTED)
        ax.text(p, 0.3, f'{p:.4f}', ha='center', fontsize=10, color=PBT)
        ax.text((c + p) / 2, -0.45, f'PBT {_word(100 * (p / c - 1))}', ha='center', fontsize=11.5, color=INK,
                fontweight='bold')
        ax.set_ylim(-0.75, 0.7)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(*xl)
        ax.set_xticks(ticks)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=5)
        ax.set_xlabel('Test mistag (%)')
        grid(ax)
        if n == 0:
            _legend(ax, [('const', 'best constant-LR model'), ('pbt', 'PBT')], bbox_transform=ax.transAxes)
            ax.get_legend().set_bbox_to_anchor((-0.01, 1.28))
    ax = axes(R0 - 0.35, top + 0.6, RW + 0.35 + 0.9, 1.17)
    lo_g, hi_g = i['grid']
    ax.axvspan(hi_g, 6e-4, color='#f3f3f5', zorder=0)
    ax.text(5.4e-4, -0.3, 'LRs not tested\nin September', ha='right', va='bottom', fontsize=10, color=MUTED)
    ax.plot([lo_g, hi_g], [1, 1], color=CONST, lw=2)
    for v in (3e-6, 5.75e-6, 8.5e-6, 1.125e-5, 1.4e-5):
        ax.plot([v, v], [0.86, 1.14], color=CONST, lw=1.5)
    _mk(ax, i['const_lr'], 1, 'const', size=10)
    ax.text(lo_g, 1.32, 'constant-LR grid (#1, #10); circle: best model', fontsize=10, color=INK, va='bottom')
    _mk(ax, i['pbt_lr_800k'], 0, 'pbt')
    _mk(ax, i['pbt_lr_8m'], 0, 'pbt')
    ax.text(i['pbt_lr_800k'] / 1.2, 0, '800k', ha='right', va='center', fontsize=10, color=PBT)
    ax.text(i['pbt_lr_8m'] * 1.2, 0, '8M', ha='left', va='center', fontsize=10, color=PBT)
    ax.text(lo_g, 0.32, 'LR of the selected PBT model', fontsize=10, color=INK, va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(2e-6, 6e-4)
    ax.set_ylim(-0.45, 1.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('learning rate')
    ax.set_title('The best constant LR was the highest LR tried', loc='left', fontsize=11, fontweight='bold', pad=5)
    lr_box = ax.get_position()

    # --- B ---------------------------------------------------------------------------------------------------
    top = st['B'] + HEAD
    ax = fig.add_axes([lr_box.x0, 1 - (top + 0.3 + 2.25) / H, lr_box.width, 2.25 / H])
    sc = o['scan']
    lab_at = {'1': (4.2e-6, 0.3635, '#1 (Sep)'), '12': (1.55e-5, 0.3335, '#12'), '13': (3.0e-5, 0.3170, '#13'),
              '14': (2.1e-4, 0.3245, '#14'), '16': (2.6e-5, 0.3045, '#16 (seed 22345)')}
    for tr, (lx, ly, lab) in lab_at.items():
        lrs, vals = zip(*sc[tr])
        dashed = tr == '16'
        ax.plot(lrs, vals, color=CONST, lw=1.3, ls=(0, (3, 2)) if dashed else '-', zorder=2)
        ax.plot(lrs, vals, 'o', ms=5.5, color=CONST, mfc='white' if dashed else CONST, mew=1.3, zorder=3)
        bl, bv = min(zip(lrs, vals), key=lambda t: t[1])
        ax.plot(bl, bv, 'o', ms=10, color=INK, mfc='none', mew=1.5, zorder=4)
        ax.text(lx, ly, lab, ha='left' if tr not in ('14',) else 'center', fontsize=10, color=INK)
    ax.set_xscale('log')
    ax.set_xlim(2e-6, 6e-4)
    ax.set_ylim(0.301, 0.372)
    ax.set_ylabel('Validation (150k)\nmistag (%)', fontsize=10.5)
    ax.set_xlabel('constant learning rate')
    ax.set_title('Every constant-LR model; ring: best of its grid', loc='left', fontsize=11, fontweight='bold', pad=5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    grid(ax, 'y')
    ax.text(5.6e-4, 0.371, 'the best LR of each grid was its highest,\nuntil #14 found 1.2×10$^{-4}$ inside its grid',
            fontsize=10, color=MUTED, va='top', ha='right')
    ax = axes(L, top + 0.3, LW, 2.25)
    rows = o['baseline']
    n = len(rows)
    for k, (lab, sub, c, p, seed) in enumerate(rows):
        y = n - 1 - k
        ax.plot([min(c, p), max(c, p)], [y, y], color='#d0d0d4', lw=3, zorder=1)
        _mk(ax, c, y, 'const')
        _mk(ax, p, y, 'pbt')
        right = max(c, p) < 0.318  # label right of the pair, or left of it when the pair is near the right edge
        ax.text(max(c, p) + 0.0012 if right else min(c, p) - 0.0012, y, 'PBT ' + _word(100 * (p / c - 1)),
                va='center', ha='left' if right else 'right', fontsize=10.5, color=INK, fontweight='bold')
    _row_labels(ax, [(n - 1 - k, r[0], r[1]) for k, r in enumerate(rows)])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.99, 0.56, 'seed 12345', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=9.5,
            color=MUTED)
    ax.text(0.99, 0.44, 'seed and grid both changed', transform=ax.get_yaxis_transform(), ha='right', va='top',
            fontsize=9.5, color=MUTED)
    _rows_axes(ax, n)
    ax.set_xlim(0.2985, 0.3345)
    ax.set_xticks([0.300, 0.310, 0.320, 0.330])
    ax.set_xlabel('Test (2.8M) mistag (%)')
    ax.set_title('PBT against the best constant-LR model', loc='left', fontsize=11, fontweight='bold', pad=5)

    # --- C ---------------------------------------------------------------------------------------------------
    top = st['C'] + HEAD
    s8 = o['sel800k']
    seeds = s8['seeds']
    ax = axes(L, top + 0.42, LW, 1.95)
    for k, s in enumerate(seeds):
        for kind in ('std', 'const', 'pbt'):
            _mk(ax, s[{'std': 'standard', 'const': 'constant', 'pbt': 'pbt'}[kind]], 2 - k, kind)
    _row_labels(ax, [(2 - k, f'seed {s["seed"]}', '') for k, s in enumerate(seeds)])
    _rows_axes(ax, 3)
    ax.set_xlim(0.2990, 0.3045)
    ax.set_xticks([0.300, 0.301, 0.302, 0.303, 0.304], ['0.300', '', '0.302', '', '0.304'])
    ax.set_xlabel('Test (2.8M) mistag (%)')
    _legend(ax, [('std', 'standard ParT'), ('const', 'tuned constant LR'), ('pbt', 'PBT')])
    ax = axes(R0, top + 0.42, RW, 1.95)
    ys = [7.8, 7.0, 6.2, 5.2, 3.0, 2.2, 1.4]
    rows = [(y, f'seed {s["seed"]}', s['pbt_vs_standard']['change'], s['pbt_vs_standard']['interval'], PBT, 'pbt',
             False) for y, s in zip(ys, seeds)]
    rows.append((ys[3], 'mean', s8['mean_change'], s8['mean_interval'], PBT, 'pbt', True))
    rows += [(y, f'seed {s["seed"]}', s['pbt_vs_constant']['change'], s['pbt_vs_constant']['interval'], CONST,
              'const', False) for y, s in zip(ys[4:], seeds)]
    _intervals(ax, rows, VX)
    ax.text(-0.025, 8.75, 'PBT vs standard ParT', transform=ax.get_yaxis_transform(), fontsize=10.5, ha='right',
            fontweight='bold', color=PBT, va='center')
    ax.text(-0.025, 4.0, 'PBT vs tuned\nconstant LR', transform=ax.get_yaxis_transform(), fontsize=10.5, ha='right',
            fontweight='bold', color=INK, va='center')
    ax.set_ylim(0.8, 9.1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    _rel_axis(ax, -2.0, 0.8, 0.5, 'relative change; bar: 95% Test-sample interval')
    grid(ax)

    # --- D ---------------------------------------------------------------------------------------------------
    top = st['D'] + HEAD
    t = o['train8m']
    ref = t['std_3e3']
    rows = [('Standard ParT, start LR 10$^{-3}$', '#22 · Nov 14–15 · best of 5 start LRs', t['std_1e3'], 'std2'),
            ('PBT', '#23 · Nov 15–17', t['pbt'], 'pbt'),
            ('Standard ParT, start LR 3×10$^{-3}$', '#24 · Nov 19–20 · the reference', ref, 'std'),
            ('PBT + probes', '#25 · Nov 19–21 · +12% training', t['probe'], 'pbt'),
            ('PBT + probes, decay, higher LRs', '#26 · Nov 20–23 · +9% training', t['probe_decay'], 'pbt')]
    n = len(rows)
    ax = axes(L, top + 0.3, LW, 2.2)
    ax.axvline(ref, color=STD, lw=1, ls=(0, (3, 3)), zorder=1)
    for k, (lab, sub, v, kind) in enumerate(rows):
        _mk(ax, v, n - 1 - k, kind)
        ax.text(v + 0.0009, n - 1 - k, f'{v:.4f}', va='center', fontsize=10, color=MUTED)
    _row_labels(ax, [(n - 1 - k, r[0], r[1]) for k, r in enumerate(rows)])
    _rows_axes(ax, n)
    ax.set_xlim(0.1585, 0.1885)
    ax.set_xticks([0.160, 0.170, 0.180])
    ax.set_xlabel('Test (2.8M) mistag (%)')
    ax = axes(R0, top + 0.3, RW, 2.2)
    for k, (lab, sub, v, kind) in enumerate(rows):
        y = n - 1 - k
        c = 100 * (v / ref - 1)
        if kind == 'std':
            ax.text(VX, y, 'reference', transform=ax.get_yaxis_transform(), va='center', fontsize=10.5, color=STD)
            continue
        ax.barh(y, c, height=0.45, color=PBT if kind == 'pbt' else STD, alpha=0.85 if kind == 'pbt' else 0.35, zorder=2)
        ax.text(VX, y, _word(c), transform=ax.get_yaxis_transform(), va='center', fontsize=10.5, color=INK,
                fontweight='bold' if kind == 'pbt' else 'normal')
    ax.text(7.0, 1.0, 'deficit reduced,\nnot removed', fontsize=10, color=MUTED, va='center')
    _rows_axes(ax, n)
    _rel_axis(ax, -0.5, 13.5, 2, 'relative change against #24; one seed, no interval')

    # --- E ---------------------------------------------------------------------------------------------------
    top = st['E'] + HEAD
    en = o['ensemble']
    ax = axes(L, top + 0.42, LW, 1.6)
    for k, s in enumerate(en['seeds']):
        y = 2 - k
        ax.plot([s['ensemble'], s['better']], [y, y], color='#d0d0d4', lw=3, zorder=1)
        _mk(ax, s['other'], y, 'std2', size=8)
        _mk(ax, s['better'], y, 'std')
        _mk(ax, s['ensemble'], y, 'ens')
    _row_labels(ax, [(2 - k, f'seed {s["seed"]}', f'#{s["trial"]}') for k, s in enumerate(en['seeds'])])
    _rows_axes(ax, 3)
    ax.set_xlim(0.1505, 0.1650)
    ax.set_xticks([0.152, 0.156, 0.160, 0.164])
    ax.set_xlabel('Test (2.8M) mistag (%)')
    _legend(ax, [('std', 'start LR 3×10$^{-3}$ (better)'), ('std2', '10$^{-3}$'), ('ens', 'average of the two')])
    ax = axes(R0, top + 0.42, RW, 1.6)
    rows = [(3 - k, f'seed {s["seed"]}', s['change'], s['interval'], ENS, 'ens', False)
            for k, s in enumerate(en['seeds'])]
    rows.append((0, 'mean', en['mean_change'], en['mean_interval'], ENS, 'ens', True))
    _intervals(ax, rows, VX)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    _rel_axis(ax, -7.0, 0.6, 1, 'against the better single model; bar: 95% Test-sample interval')
    grid(ax)
    save(fig, 'fig1_overview.svg')

if __name__ == '__main__':
    import sys
    if '--refresh' in sys.argv:
        extract()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_overview, fig_regimes, fig_mechanism, fig_ensemble):
        f()
    print(sorted(p.name for p in OUT.iterdir()))
