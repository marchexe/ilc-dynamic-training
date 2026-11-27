#!/usr/bin/env python3
"""Build the supervisor report docs/_extra/overview.html.

    .venv/bin/python scripts/reports/publication_figures.py   # figures, docs/_extra/pub/*.svg
    .venv/bin/python scripts/reports/build_report.py

Every number is taken from docs/results.rst, docs/ongoing.rst or the run manifests (dates in JST); the trial
numbers are those of the experiment log (docs/_extra/log.html, scripts/reports/build_log.py).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lr(m, e):
    return (f'{m}×' if m != '1' else '') + f'10<sup>−{e}</sup>'


# (phase title, what changed against the previous phase, rows); row: (trial, date, configuration, result, interpretation)
PHASES = [
    ('Initial implementation', 'Training (800k).', [
        ('–', 'Jul 22 – Aug 31', 'First PBT versions (exploit-and-mutate, anchor-copy) and LR controllers; older training '
         'set-up (samples per epoch capped, damped optimizer start)',
         'Two defects found and fixed: BatchNorm not frozen; a controller overwrote the PBT LR',
         'Not comparable with later trials; training rebuilt as deterministic full epochs (Sep 16)'),
    ]),
    ('PBT against constant learning rates', 'Training (800k), then (8M); Validation (150k); seed 12345 unless noted.', [
        ('1', 'Sep 16', f'Constant LR, 5 values {lr("3", 6)}–{lr("1.4", 5)}', '0.3269; best at the highest LR',
         'Reference for PBT. The best model at the edge of the grid was not followed up at the time'),
        ('2', 'Sep 17–18', 'The 1.4 and 0.85×10<sup>−5</sup> models of #1 continued to 100 epochs', '0.3156',
         'Longer training helps'),
        ('3', 'Sep 18', f'PBT: start as #1; every 5 epochs the weakest copies the best, LR ×0.8 or ×1.2, max {lr("3", 5)}',
         '0.3225; 1.4% better than #1', 'Apparent PBT gain (later found to be an LR-level effect, #12–#14)'),
        ('4', 'Sep 19', 'PBT as #3, 100 epochs', '0.3111; 1.4% better than #2', 'Same apparent gain at 100 epochs'),
        ('5', 'Sep 20', 'Rule-based LR controller in observe-only mode on a 60k subset of Validation; 8 models, 96 short epochs',
         'Subset ranks models like the full set (Spearman 0.96) but detects an improvement in only 72% of cases; '
         'the controller proposed almost no changes', 'Not pursued'),
        ('6', 'Sep 23–29', 'PBT with decisions every epoch against every 5 epochs; seeds 12345 and 22345',
         'Every epoch 0.47% and 0.13% worse on Test; better on Validation', 'Inconclusive: Validation and Test disagree at this level'),
        ('7', 'Sep 28', 'Restart test: one epoch run as five restarted parts', f'Weights differ (max {lr("7.6", 4)})',
         'Decisions within an epoch impossible: the optimizer (Lookahead) state is not saved'),
        ('8', 'Oct 12', 'First Test (2.8M) evaluation of the selected models of #1–#6',
         'Test 0.010–0.019 lower than Validation; rank correlation 0.70', 'Differences of about 0.3% reorder: Validation (150k) resolves them poorly'),
        ('9', 'Oct 13', 'Training (8M) and Test (9.0M) prepared; one-epoch GPU check', 'Ready', ''),
        ('10', 'Oct 14–21, Test Nov 2', 'Preregistered on Training (8M): constant LR as #1 against PBT as #3; Test (9.0M) opened once',
         '0.2375 against 0.2572: PBT 7.7% better; all 10 validity checks passed',
         'Best constant model again at the highest LR and PBT above the grid: cannot separate adaptation from LR level'),
        ('11', 'Oct 17 – Nov 3', 'Checkpoints with the full optimizer state, for decisions within an epoch',
         'One epoch = five restarted parts, bit for bit', 'Implemented; not used in any PBT run'),
    ]),
    ('Higher constant learning rates', 'Training (800k); Validation (150k).', [
        ('12', 'Nov 3', f'Constant LR {lr("1.4", 5)}–{lr("3", 5)}', '0.3146; best again at the highest LR', 'Grid still too low'),
        ('13', 'Nov 3–4', f'Constant LR {lr("3", 5)}–{lr("8", 5)}', '<b>0.3018</b>; best at 8×10<sup>−5</sup>',
         'Better than every PBT run of #3–#6: the early PBT gains came from the LR level'),
        ('14', 'Nov 4', f'Constant LR {lr("8", 5)}–{lr("3.2", 4)}', f'0.3033; best at {lr("1.2", 4)}', f'Optimum near {lr("8", 5)}–{lr("1.2", 4)}'),
        ('15', 'Nov 4', f'Fast PBT: start {lr("1", 5)}–{lr("3.5", 4)}; every 2 epochs from epoch 4, LR ×0.5 or ×2, '
         f'bounds {lr("5", 6)}–{lr("5", 4)}', '0.3003; 1.0% better than #14', 'PBT about equal to the best constant LR'),
        ('16', 'Nov 9', f'#15 and constant LR {lr("6", 5)}–{lr("1.6", 4)}, seed 22345', '0.3041 against 0.3027: PBT 0.5% worse, '
         'although 1.4% better on Validation', 'Sign reverses; selection noise (~0.5%) as large as the effect'),
        ('17', 'Nov 9', '#15 with fallback copies (×0.7 or ×1.4 when ×0.5 and ×2 are both in use)', '0.3026; 0.8% worse than #15',
         'No improvement'),
    ]),
    ('PBT against standard ParT, 800k', 'Training (800k); Validation (1M); three seeds.', [
        ('18', 'Nov 10–13', f'Standard ParT (start {lr("1", 5)}–{lr("1", 3)}), constant LR {lr("6", 5)}–{lr("1.6", 4)} and PBT as #17; '
         'seeds 12345, 22345, 32345', 'Standard 0.3029 / 0.3038 / 0.3024; constant 0.3028 / 0.3013 / 0.3024; PBT 0.3008 / 0.3012 / '
         '0.2996. <b>PBT 0.81% better</b> (95% interval 0.34–1.34%), at 7 of 8 working points',
         'Small but consistent PBT gain; the constant-LR grid is not stable across seeds'),
        ('19', 'Nov 12', 'PBT as #18 with a final LR decay (no decisions in the last 15 epochs); 2 seeds',
         'Same selected models as #18', 'The decay adds nothing at 800k'),
        ('20', 'Nov 12', 'Standard ParT from random weights', '0.378 on Validation, against about 0.30', 'Pretraining is essential'),
        ('21', 'Nov 12', 'Weight averages of nearby checkpoints and of PBT models', 'No gain (Validation)', ''),
    ]),
    ('PBT against standard ParT, 8M', 'Training (8M); Validation (1M); seed 12345.', [
        ('22', 'Nov 14–15', f'Standard ParT, start {lr("1", 5)}–{lr("1", 3)}', f'0.1635; best at the highest start LR ({lr("1", 3)})',
         'Start LR range extended (#24)'),
        ('23', 'Nov 15–17', f'PBT as #18 with the LR range of standard ParT (max {lr("1", 3)})', '0.1812; <b>12.4% worse</b> than #24',
         f'PBT lowered all LRs early (at most {lr("2.5", 4)} from epoch 11); hypothesis: two-epoch decision horizon'),
        ('24', 'Nov 19–20', f'Standard ParT, start {lr("3", 3)} (one model, fixed in advance)', '<b>0.1611</b>', '8M reference'),
        ('25', 'Nov 19–21', 'PBT with cool-down probes: each model judged on a copy trained ¼ epoch at 0.1× its LR; also on 800k',
         '8M: 0.1699, 5.4% worse than #24. 800k: 0.3024 (no change)', f'LRs stay higher ({lr("1.2", 4)}–{lr("1", 3)}); gap halved'),
        ('26', 'Nov 20–23', f'Probes plus the final decay of standard ParT; max LR {lr("3", 3)}', '0.1655; 2.7% worse than #24',
         '78% of the gap closed; still worse. Extra seeds not run (rule fixed in advance)'),
    ]),
    ('Ensembles and other ways to gain, 8M', 'Training (8M); Validation (1M).', [
        ('27', 'Nov 23', 'Ensemble: mean class probabilities of #22 and #24; chosen among 8 ensembles on Validation, Test check fixed in advance',
         '0.1530; 5.05% better than #24 (interval 3.75–6.36%)', 'Large gain on the development seed'),
        ('28', 'Nov 23–25', f'#27 repeated: new standard ParT runs at {lr("3", 3)} and {lr("1", 3)}, seeds 22345 and 32345',
         '0.1519 and 0.1540 against 0.1597 and 0.1623 (3×10<sup>−3</sup> models): 4.86% and 5.14% better; <b>mean of 3 seeds 5.01%</b> (interval 4.17–5.84%)', 'Confirmed on new seeds'),
        ('29', 'Nov 23', 'Ensembles of any two of 15 models of #18 (800k)', 'No gain beyond chance (Validation)', 'Models too similar at 800k'),
        ('30', 'Nov 23–25', 'Disagreement and gain of model pairs, all 8M models', 'Same run or PBT pair: 0.2–0.4% disagreement, no gain; '
         'separate models: 1.5–2.3%; same LR 3.6%, different LR 4.7% gain (Validation)', 'Gain comes mostly from separate training'),
        ('31', 'Nov 23–24', 'Final decay of #24 without dropout', '0.1730 against 0.1595 at epoch 44 (Validation); stopped',
         'Dropout helps at 8M although there is no visible overfitting'),
        ('32', 'Nov 24', 'Distillation: final decay of #24 trained half on labels, half on the ensemble of #27',
         'Best 0.1600 against 0.1584 (Validation)', 'No gain; the ensemble gain is not kept in one model'),
        ('33', 'Nov 24–25', f'Standard ParT, start {lr("1", 2)}', '0.1631 on Validation (worse than #24); 0.1599 on Test',
         f'Start LR above {lr("1", 3)} hardly matters; #24 stays the reference'),
    ]),
]


FIG = '<figure><div class="fig-scroll"><img src="pub/{src}" alt="{alt}"></div><figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>'
SUP = lambda m, e: lr(m, e)

# per phase: (dates, objective, learned-and-next paragraphs (HTML), figures)
PHASE_TEXT = [
    ('Jul 22 – Aug 31', 'Build the training and PBT tooling.',
     ['<b>Learned.</b> Two defects (BatchNorm not frozen; a controller overwriting the PBT LR) made these runs unreliable, '
      'and their training set-up was not comparable. <b>Next:</b> training rebuilt as deterministic full epochs, used by all later trials.'],
     []),
    ('Sep 16 – Nov 3', 'Compare PBT with constant learning rates on Training (800k), then in a preregistered test on Training (8M).',
     ['<b>Learned.</b> PBT was 1.4% better at 800k (#3, #4) and 7.7% better at 8M on Test (9.0M) (#10). In every comparison the '
      'best constant-LR model was at the top of its grid and PBT ended above the grid, so adaptation and LR level could not be '
      'separated. Validation (150k) did not resolve differences of about 0.3–0.5% (#6, #8). <b>Next:</b> move the constant-LR '
      'grid upwards.'],
     []),
    ('Nov 3 – Nov 9', 'Test whether a higher constant LR explains the PBT gain.',
     ['<b>Learned</b> <span class="ev s">single seed per grid</span>. The mistag falls with the constant LR up to about '
      f'{SUP("1", 4)}, and both PBT runs lie on that curve (Figure 2): the early PBT gains came from the LR level. A faster PBT '
      'equalled the best constant LR (#15), but the sign reversed with a second seed (#16): selection noise of about 0.5% on '
      'Validation (150k). <b>Next:</b> select on Validation (1M), compare with standard ParT (constant LR for 35 epochs, '
      'then ×0.01 over 15; five start LRs), three seeds.'],
     [('fig3_learning_rate.svg', 'Mistag on Test against the learning rate of the selected model',
       'Mistag on Test (2.8M) against the LR of the selected model; Training (800k), seed 12345. Grey: best model of each '
       'constant-LR grid (#1, #12–#14). Red: PBT (#3, #15).')]),
    ('Nov 10 – Nov 13', 'Compare PBT with standard ParT under reliable selection, Training (800k).',
     ['<b>Learned</b> <span class="ev r">replicated, 3 seeds</span>. PBT is 0.81% better than standard ParT (95% interval '
      '0.34–1.34%); the tuned constant LR is not stable across seeds. A final LR decay adds nothing to PBT (#19); pretraining is '
      'essential (#20). <b>Next:</b> the same comparison with ten times more training data.'],
     []),
    ('Nov 13 – Nov 23', 'Repeat the comparison on Training (8M) and correct PBT where it fails.',
     ['<b>Learned</b> <span class="ev s">single seed</span>. PBT is 12.4% worse than standard ParT (#23). It keeps all LRs at '
      f'or below {SUP("2.5", 4)} from epoch 11, leads for about 20 epochs and loses in the final decay of standard ParT '
      '(Figure 3). The training loss falls below the Validation loss after about 20 epochs with Training (800k) but not with '
      'Training (8M) (Figure 4). <span class="ev h">Hypothesis</span> PBT judges each LR two epochs after a change, which '
      'favours lower LRs; this is right only when the model overfits. Cool-down probes and a final decay reduce the loss to '
      '2.7% but not to zero (#25, #26); further seeds were not run under the rule fixed in advance. <b>Next:</b> look for a '
      'gain outside the LR schedule.'],
     [('fig5_mechanism.svg', 'Learning rate and best Validation mistag per epoch, Training (8M)',
       'Training (8M), seed 12345. Left: LR per epoch of standard ParT (#24) and range of the five PBT models (#23). Right: '
       'best Validation (1M) mistag so far; dashed: PBT with probes and final decay (#26). Shaded: final LR decay.'),
      ('fig4_regimes.svg', 'Training and Validation loss per epoch, Training (800k) and Training (8M)',
       f'Loss per epoch of the best standard ParT model (start LR {SUP("1", 4)} at 800k, {SUP("3", 3)} at 8M) on the '
       'training data and on Validation (1M).')]),
    ('Nov 23 – Nov 25', 'Test whether combining separately trained standard ParT models helps on Training (8M).',
     ['<b>Learned</b> <span class="ev r">replicated, 3 seeds</span>. The mean output of the two standard ParT models of a seed is '
      '5.01% better than the better one (95% interval 4.17–5.84%), at twice the inference cost (#27, #28). Only separately '
      'trained 8M models disagree enough to gain (Figure 5). <span class="ev h">Hypothesis</span> PBT models share weights '
      'through copying, and at 800k the low LRs keep all models close to the pretrained start. Distillation into one model, '
      'training without dropout and a start LR of 10<sup>−2</sup> gave no gain (#31–#33).'],
     [('fig6_ensemble.svg', 'Disagreement between two models and ensemble gain for four kinds of model pairs',
       'Share of Validation (1M) jets on which two models predict different classes, and the ensemble gain on Validation, '
       'for four kinds of model pairs (#29, #30).')]),
]


def timeline():
    out, nfig = [], 2
    for k, ((title, changed, rows), (dates, objective, learned, figs)) in enumerate(zip(PHASES, PHASE_TEXT)):
        body = ''.join(f'<tr id="t{n}"><td class="n">{n}</td><td class="d">{d}</td><td>{c}</td><td>{r}</td>'
                       f'<td class="i">{i}</td></tr>' for n, d, c, r, i in rows)
        figs_html = []
        for src, alt, cap in figs:
            figs_html.append(FIG.format(src=src, alt=alt, n=nfig, cap=cap))
            nfig += 1
        out.append(f'<section class="phase" id="phase-{k}"><div class="when">Phase {k} · {dates}</div>'
                   f'<h3>{title}</h3><p class="obj">{objective} <span class="cond">{changed}</span></p>'
                   '<div class="table-wrap"><table class="hist"><colgroup><col class="c-n"><col class="c-d"><col class="c-c">'
                   '<col class="c-r"><col class="c-i"></colgroup><thead><tr><th>#</th><th>Date</th><th>Configuration</th>'
                   f'<th>Result</th><th>Interpretation</th></tr></thead><tbody>{body}</tbody></table></div>'
                   + ''.join(f'<p class="learned">{t}</p>' for t in learned) + ''.join(figs_html) + '</section>')
    return '\n'.join(out)


def main():
    page = (Path(__file__).with_name('report_template.html')).read_text().replace('<!--TIMELINE-->', timeline())
    out = ROOT / 'docs/_extra/overview.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


if __name__ == '__main__':
    main()
