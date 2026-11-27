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
from matplotlib.ticker import AutoMinorLocator  # noqa: E402
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
        grid_test={'1': hi['fixed50_s12345'], '12': hi['fixed_high50_s12345'], '13': ce['fixed_ceiling50_s12345'],
                   '14': fa['fixed_top50_s12345']},
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
        trajectories_800k_seed12345={
            'standard_part_1e-4 (#18)': trajectory('sel1m_standard_part', ['flat_decay_100e-6']),
            'pbt (#18)': trajectory('sel1m_fast_pbt')},
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
        ('with_pbt', 'Training (8M): a PBT model + a model trained apart from it (#30)',
         dict(marker='s', ms=6.5, color=PBT, mfc='white', mew=1.4)),
        ('one_run', 'Training (8M): two checkpoints of one model, or two models of one PBT population (#30)',
         dict(marker='^', ms=7.5, color=INK, mfc='white', mew=1.4)),
        ('separate_standard', 'Training (8M): two independently trained standard ParT models (#28, #30)',
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
# Main-text figures: one idea each, drawn at a larger type size for the 780 px figure column of the report.
# Encoding (MARK): standard ParT blue filled circle, constant LR grey open circle, PBT orange square, ensemble green
# diamond. Values of different seeds, samples or comparators are never joined by a line.
# ---------------------------------------------------------------------------------------------------------------
from functools import wraps  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

MARK = dict(std=dict(marker='o', color=STD, mfc=STD), const=dict(marker='o', color=CONST, mfc='white'),
            pbt=dict(marker='s', color=PBT, mfc=PBT), ens=dict(marker='D', color=ENS, mfc=ENS))
LARGE = {'font.size': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5,
         'axes.titlesize': 13.5}


def large(f):
    @wraps(f)
    def run(*a, **k):
        with plt.rc_context(LARGE):
            return f(*a, **k)
    return run


def _word(c, digits=1):
    return f'{abs(c):.{digits}f}% {"better" if c < 0 else "worse"}'


def _rel(a, b):
    """Relative change of a against b, in %."""
    return 100 * (a / b - 1)


@large
def fig_timeline():
    """The results that changed the direction of the study, in time order (equal spacing; dates are labels).
    Colour: which method was ahead at that point. Second date line: when the Test values were measured, where that
    was well after the runs."""
    o = data()['overview']
    i, b, t, e = o['initial'], o['baseline'], o['train8m'], o['ensemble']
    lead = sorted(-_rel(r[3], r[2]) for r in b[1:3])
    pts = [  # (dates, trials, Test date or '', text, colour or None for no result)
        ('Jul–Aug', 'set-up', '', 'First implementation:\ntwo defects, not used', None),
        ('Sep 16–19', '#1–#4', 'Test: Oct 12', f'PBT {_word(_rel(i["pbt_800k"], i["const_800k"]))}\nthan constant LR', PBT),
        ('Oct 14–21', '#10', 'Test: Nov 2', f'Training (8M):\nPBT {_word(_rel(i["pbt_8m"], i["const_8m"]))}', PBT),
        ('Nov 3–4', '#12–#15', '', f'Constant LR tuned:\nPBT ahead by {lead[0]:.1f}–{lead[1]:.1f}%', CONST),
        ('Nov 9', '#16', '', f'New seed and grid:\nPBT {_word(_rel(b[3][3], b[3][2]), 2)}', CONST),
        ('Nov 10–13', '#18', '', f'Three seeds: PBT {abs(o["sel800k"]["mean_change"]):.2f}%\nbetter than standard ParT', PBT),
        ('Nov 14–20', '#22–#24', '', f'Training (8M):\nPBT {_word(_rel(t["pbt"], t["std_3e3"]))}', STD),
        ('Nov 19–23', '#25, #26', '', f'Two changes to PBT:\nstill {_word(_rel(t["probe_decay"], t["std_3e3"]))}', STD),
        ('Nov 23–25', '#27, #28', '', f'Two models averaged:\n{abs(e["mean_change"]):.1f}% better', ENS),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 3.55))
    n = len(pts)
    ax.plot([-0.35, n - 0.65], [0, 0], color='#c8c8cd', lw=2, zorder=1)
    for k, (date, tr, test, text, col) in enumerate(pts):
        ax.plot(k, 0, 'o', ms=11, color=col or '#b4b4ba', mfc=col or 'white', mec=col or '#b4b4ba' if col is None else 'white',
                mew=2, zorder=3)
        ax.text(k, -0.22, date, ha='center', va='top', fontsize=12, color=INK, fontweight='bold')
        ax.text(k, -0.52, tr, ha='center', va='top', fontsize=11, color=MUTED)
        if test:
            ax.text(k, -0.80, test, ha='center', va='top', fontsize=10.5, color=MUTED, style='italic')
        h = 0.45 if k % 2 == 0 else 1.45
        ax.plot([k, k], [0.12, h - 0.08], color='#c8c8cd', lw=1, zorder=1)
        ax.text(k, h, text, ha='center', va='bottom', fontsize=11.5, color=INK if col else MUTED, linespacing=1.3)
    for x0, x1, lab in ((0, 0, 'set-up'), (1, 2, 'PBT against constant LR'), (3, 4, 'tuning the baseline'),
                        (5, 5, 'Training (800k)'), (6, 7, 'Training (8M)'), (8, 8, 'ensembles')):
        ax.plot([x0 - 0.38, x1 + 0.38], [-1.22, -1.22], color=MUTED, lw=1)
        ax.text((x0 + x1) / 2, -1.33, lab, ha='center', va='top', fontsize=11, color=MUTED, style='italic')
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_ylim(-1.7, 2.35)
    ax.axis('off')
    save(fig, 'fig_timeline.svg')


@large
def fig_pbt():
    """How one PBT decision works (windowed PBT, scripts/training/pbt/reference/windowed_v2.py): five models train
    at their own LR; at a decision the models are ranked on Validation; a model that lost two decisions in a row
    copies the weights and optimizer state of the best model and continues at the best model's LR times a factor."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ranks = [2, 1, 4, 5, 3]
    x0, xd, x1 = 0.0, 3.2, 6.0
    for k, rank in enumerate(ranks):
        y = 4 - k
        ax.text(x0 - 0.15, y, f'model {k + 1}', ha='right', va='center', fontsize=12, color=INK)
        ax.plot([x0, xd - 0.35], [y, y], color=PBT, lw=3, alpha=0.5, solid_capstyle='butt')
        ax.plot([xd + 0.55, x1], [y, y], color=PBT, lw=3, alpha=1 if rank == 5 else 0.5, solid_capstyle='butt')
        ax.text(xd + 0.1, y, str(rank), ha='center', va='center', fontsize=11.5, color='white' if rank == 1 else INK,
                fontweight='bold', zorder=4,
                bbox=dict(boxstyle='circle,pad=0.28', fc=INK if rank == 1 else 'white', ec=INK, lw=1.2))
    ax.add_patch(FancyBboxPatch((xd - 0.22, -0.42), 0.64, 4.84, boxstyle='round,pad=0.02,rounding_size=0.25',
                                fc='#f3f3f5', ec='none', zorder=0))
    ax.text(xd + 0.1, -0.62, 'rank on\nValidation', ha='center', va='top', fontsize=11.5, color=INK,
            fontweight='bold', linespacing=1.15)
    ax.text((x0 + xd - 0.35) / 2, 4.55, 'train, each at its own LR', ha='center', va='bottom', fontsize=11.5,
            color=MUTED)
    ax.text((xd + 0.55 + x1) / 2, 4.55, 'next window', ha='center', va='bottom', fontsize=11.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((xd + 0.42, 2.92), (xd + 0.95, 1.12), connectionstyle='arc3,rad=-0.3',
                                 arrowstyle='-|>', mutation_scale=16, lw=1.6, color=INK, zorder=5))
    ax.text(x1 + 0.3, 1.0, 'model 4 lost twice in a row: it copies\nthe weights and optimizer state of model 2',
            fontsize=11.5, color=INK, va='bottom', linespacing=1.2)
    ax.text(x1 + 0.3, 0.88, 'and continues at the LR of model 2 × 0.8 or × 1.2', fontsize=11.5, color=PBT,
            va='top', fontweight='bold')
    ax.text(x1 + 0.3, 3.0, 'the others continue unchanged', fontsize=11.5, color=MUTED, va='center')
    ax.set_xlim(-1.25, x1 + 4.6)
    ax.set_ylim(-1.35, 4.95)
    ax.axis('off')
    save(fig, 'fig_pbt.svg')


@large
def fig_baseline():
    """Best constant-LR model of each grid (seed 12345) against the PBT results of September and November; inset: the
    #16 comparison with seed 22345 and a re-centred grid (both changed together)."""
    o = data()['overview']
    g, b = o['grid_test'], o['baseline']
    test = [('#1', 'Sep 16', '≤ 1.4×10⁻⁵', g['1']), ('#12', 'Nov 3', '≤ 3×10⁻⁵', g['12']),
            ('#13', 'Nov 3–4', '≤ 8×10⁻⁵', g['13']), ('#14', 'Nov 4', '≤ 3.2×10⁻⁴', g['14'])]
    fig, (ax, ins) = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True,
                                  gridspec_kw=dict(width_ratios=[4.4, 1.25], wspace=0.12))
    xs = range(len(test))
    ax.plot(xs, [q[3] for q in test], color=CONST, lw=1.6, zorder=2)
    for x, (tr, date, grid_, v) in zip(xs, test):
        ax.plot(x, v, ls='none', ms=10, mew=2, zorder=3, **MARK['const'])
        ax.text(x + 0.09, v + 0.0007, f'{v:.4f}', fontsize=11.5, color=MUTED)
    ax.set_xticks(list(xs), [f'{q[0]}  {q[1]}\nLRs {q[2]}' for q in test])
    for v, lab, ls, x, ha in ((b[0][3], f'PBT in September (#3)  {b[0][3]:.4f}', (0, (5, 3)), 3.45, 'right'),
                              (b[1][3], f'fast PBT in November (#15)  {b[1][3]:.4f}', '-', -0.3, 'left')):
        ax.axhline(v, color=PBT, lw=1.8, ls=ls, zorder=1)
        ax.text(x, v + 0.0006, lab, ha=ha, va='bottom', fontsize=11.5, color=PBT)
    ax.set_xlim(-0.35, 3.5)
    ax.set_ylim(0.2985, 0.3295)
    ax.set_ylabel('Test mistag (%), lower is better')
    ax.set_title('Best constant-LR model of each grid · seed 12345', loc='left', fontsize=12.5, color=MUTED, pad=10)
    grid(ax, 'y')
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0, pad=8)
    _, _, c16, p16, _ = b[3]
    ins.set_facecolor('#f6f6f8')
    ins.plot(0, c16, ls='none', ms=10, mew=2, zorder=3, **MARK['const'])
    ins.plot(1, p16, ls='none', ms=10, mew=2, zorder=3, **MARK['pbt'])
    ins.text(0, c16 - 0.0011, f'{c16:.4f}', ha='center', va='top', fontsize=11, color=MUTED)
    ins.text(1, p16 + 0.0008, f'{p16:.4f}', ha='center', va='bottom', fontsize=11, color=PBT)
    ins.text(0.5, 0.3105, f'PBT\n{_word(_rel(p16, c16), 2)}', ha='center', va='bottom', fontsize=12.5, color=INK,
             fontweight='bold', linespacing=1.25)
    ins.set_xticks([0, 1], ['constant\nLR', 'fast\nPBT'])
    ins.set_xlim(-0.6, 1.6)
    ins.set_title('#16  Nov 9', loc='left', fontsize=12.5, color=INK, fontweight='bold', pad=40)
    ins.text(0, 1.02, 'seed 22345 and a\nnew grid, changed together', transform=ins.transAxes, fontsize=10.5,
             color=MUTED, va='bottom', linespacing=1.15)
    ins.tick_params(axis='x', length=0, pad=8)
    ins.tick_params(axis='y', length=0)
    for s in ('left', 'bottom'):
        ins.spines[s].set_visible(False)
    save(fig, 'fig_baseline.svg')


@large
def fig_lr():
    """LR per epoch of the best standard ParT model and the range of the five PBT models, seed 12345, with Training
    (800k) (#18) and Training (8M) (#23 against #24)."""
    d, o = data(), data()['overview']
    t8, t1 = d['trajectories_8m_seed12345'], d['trajectories_800k_seed12345']
    s12 = o['sel800k']['seeds'][0]
    panels = ((t1['standard_part_1e-4 (#18)'], t1['pbt (#18)'], 'Training (800k)',
               f'PBT {_word(s12["pbt_vs_standard"]["change"], 2)} than standard ParT (#18)'),
              (t8['standard_part_3e-3 (#24)'], t8['pbt (#23)'], 'Training (8M)',
               f'PBT (#23) {_word(_rel(o["train8m"]["pbt"], o["train8m"]["std_3e3"]))} than standard ParT (#24)'))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (s, p, title, res) in zip(axes, panels):
        ax.axvspan(35.5, 50, color='#f3f3f5', zorder=0)
        ax.fill_between([q[0] for q in p], [q[1] for q in p], [q[2] for q in p], color=PBT, alpha=0.25, lw=0, step='mid')
        ax.plot([q[0] for q in s], [q[1] for q in s], color=STD, lw=2.4)
        ax.axhline(s[0][1], color=STD, lw=1, ls=(0, (2, 2)), xmin=0.7, zorder=1)
        ax.set_yscale('log')
        ax.set_ylim(8e-7, 8e-3)
        ax.set_xlim(1, 50)
        ax.set_xlabel('epoch')
        ax.set_title(f'{title}\n', loc='left', fontweight='bold')
        ax.text(0, 1.03, res, transform=ax.transAxes, fontsize=12, color=MUTED, va='bottom')
        ax.text(36.5, 1.2e-6, 'final\ndecay', fontsize=11, color=MUTED, va='bottom')
        grid(ax, 'y')
    axes[0].set_ylabel('learning rate')
    axes[0].text(3, 1.35e-4, 'standard ParT: start LR 10⁻⁴ (best)', color=STD, fontsize=12, va='bottom')
    axes[0].text(3, 7e-6, 'PBT: range of its five models', color=PBT, fontsize=12, va='top')
    axes[1].text(3, 3.6e-3, 'standard ParT: start LR 3×10⁻³ (best)', color=STD, fontsize=12, va='bottom')
    axes[1].text(14, 8e-6, 'PBT', color=PBT, fontsize=12, va='top')
    save(fig, 'fig_lr.svg')


@large
def fig_changes():
    """How much worse each PBT run with Training (8M) is than standard ParT #24 (seed 12345, Test (2.8M))."""
    t = data()['overview']['train8m']
    ref = t['std_3e3']
    rows = [('PBT (#23)', '', t['pbt']), ('+ cool-down probes (#25)', '+12% training', t['probe']),
            ('+ final decay and higher LRs (#26)', '+9% training', t['probe_decay'])]
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    for k, (lab, cost, v) in enumerate(rows):
        y = len(rows) - 1 - k
        c = _rel(v, ref)
        ax.barh(y, c, height=0.52, color=PBT, alpha=1 - 0.2 * k)
        ax.text(c + 0.25, y, _word(c), va='center', fontsize=12.5, color=INK, fontweight='bold')
        ax.text(-0.3, y, lab, ha='right', va='center', fontsize=12.5, color=INK)
        if cost:
            ax.text(c + 2.6, y, cost, va='center', fontsize=11.5, color=MUTED)
    ax.axvline(0, color=STD, lw=2)
    ax.text(0.15, 2.55, 'standard ParT (#24) = 0', ha='left', va='bottom', fontsize=11.5, color=STD)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.5, 2.9)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 15], ['0', '5%', '10%', '15%'])
    ax.set_xlabel('PBT mistag above standard ParT · Training (8M), Test, one seed')
    ax.spines['left'].set_visible(False)
    grid(ax)
    save(fig, 'fig_changes.svg')


@large
def fig_ensemble_gain():
    """Per seed: the better of the two standard ParT models and the average of the two; best PBT run for comparison."""
    o = data()['overview']
    en, pbt = o['ensemble']['seeds'], o['train8m']['probe_decay']
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    for k, s in enumerate(en):
        y = 2 - k
        ax.plot([s['ensemble'], s['better']], [y, y], color='#d4d4d8', lw=3, zorder=1)
        ax.plot(s['better'], y, ls='none', ms=11, zorder=3, **MARK['std'])
        ax.plot(s['ensemble'], y, ls='none', ms=10, zorder=3, **MARK['ens'])
        ax.text(1.02, y, _word(s['change'], 2), transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=12.5, color=ENS, fontweight='bold')
        ax.text(-0.02, y, f'seed {s["seed"]}', transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=12.5, color=INK)
    ax.plot(pbt, 2, ls='none', ms=10, zorder=3, **MARK['pbt'])
    ax.text(pbt, 2.33, 'best PBT (#26)', ha='center', va='bottom', fontsize=11, color=PBT)
    ax.text(en[0]['better'], 2.33, 'better of the two', ha='center', va='bottom', fontsize=11, color=STD)
    ax.text(en[0]['ensemble'], 2.33, 'average of the two', ha='center', va='bottom', fontsize=11, color=ENS)
    ax.set_xlim(0.1500, 0.1675)
    ax.set_ylim(-0.5, 2.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0.152, 0.156, 0.160, 0.164])
    ax.set_xlabel('Test mistag (%) · Training (8M)')
    grid(ax)
    save(fig, 'fig_ensemble_gain.svg')


FIGURES = (fig_timeline, fig_pbt, fig_baseline, fig_lr, fig_changes, fig_ensemble_gain, fig_regimes, fig_ensemble)

if __name__ == '__main__':
    import sys
    if '--refresh' in sys.argv:
        extract()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIGURES:
        f()
    print(sorted(p.name for p in OUT.glob('*.svg')))
