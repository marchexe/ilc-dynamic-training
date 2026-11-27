"""Reader-facing text of the research report (docs/_extra/overview.html), rendered by build_report.py.

Every number is from the run records (docs/results.rst, runs/eval/*/analysis.json, figure_data.json); trial numbers are
those of the experiment log. Wording rules: HANDOVER.md. {fig:<key>} is replaced by the figure number at build time.
Each chapter follows one reading order: question → answer → evidence (figure or table, explanation beside it) →
interpretation → experiment table → supporting experiments (collapsed) → next decision.
"""
GH = 'https://github.com/marchexe/ilc-dynamic-training/blob/main/'

TITLE = 'Adapting the learning rate with PBT when fine-tuning ParT'
DEK = ('What four months of experiments, July to November 2026, show about Population Based Training for jet-flavour '
       'tagging')
META = [('Trials', '33, numbered in time order'), ('Research status', '26 November 2026 (no run in progress)'),
        ('Report revised', '28 November 2026'),
        ('Records', '<a href="log.html">experiment log</a> · <a href="results.html">detailed results</a> · '
                    '<a href="#setup-definitions">definitions</a>')]

QUESTION = ('Does Population Based Training (PBT), which adapts the learning rate (LR) while fine-tuning the pretrained '
            'Particle Transformer (ParT), give a lower mistag than the standard ParT training recipe with the same '
            'budget of five models and 50 epochs?')

FINDINGS = [  # (claim, sentence, evidence)
    ('The initial PBT advantage was reduced after tuning the constant-LR baseline.',
     'Subsequent experiments did not establish a consistent advantage over tuned constant-LR training.',
     'one seed per grid; three in #18'),
    ('With Training (800k), PBT is slightly better than standard ParT.',
     'Better on each of three seeds, by 0.81% on average (95% Test-sample interval 0.34–1.34%).', 'three seeds'),
    ('With Training (8M), PBT is worse.',
     '12.4% worse than standard ParT; two tested changes reduce this to 2.7%, with 9–12% more training.', 'one seed'),
    ('Averaging two independently trained standard ParT models is 5% better than the better of the two.',
     'With Training (8M), on each of three seeds (4.86–5.14%; interval of the mean 4.17–5.84%), at two forward passes '
     'per jet.', 'three seeds'),
]

TIMELINE_CAPTION = (
    'The results that changed the direction of the study. Colour: which method was ahead at that point (orange PBT, '
    'blue standard ParT, green the ensemble, grey no consistent difference). Dates are those of the runs; where the '
    'Test values were measured much later, the date of that measurement is given below (#1–#4: Test (2.8M), 12 '
    'October; #10: Test (9.0M), 2 November). Equal spacing; dates are labels, not a time scale.')

_PBT_TABLE = (
    '<table class="par"><thead><tr><th></th><th>windowed PBT (#3, #4, #10)</th><th>fast PBT (#15 onward)</th></tr></thead>'
    '<tbody><tr><td>decision every</td><td>5 epochs</td><td>2 epochs</td></tr>'
    '<tr><td>ranked on the mean of</td><td>last 3 epochs</td><td>last 2 epochs</td></tr>'
    '<tr><td>new LR</td><td>×0.8 or ×1.2</td><td>×0.5 or ×2</td></tr>'
    '<tr><td>LR bounds</td><td>2×10⁻⁶–3×10⁻⁵</td><td>5×10⁻⁶–5×10⁻⁴*</td></tr></tbody></table>'
    '<p class="src">* Upper bound 10⁻³ in #23 and #25, 3×10⁻³ in #26. From #17: fallback copies; #25–#26: cool-down probes. '
    f'Code: <a href="{GH}scripts/training/pbt/reference/windowed_v2.py">windowed_v2.py</a>; configurations: '
    f'<a href="{GH}configs/experiments/windowed_pbt_v2.yaml">windowed</a>, '
    f'<a href="{GH}configs/experiments/windowed_pbt_v2_fast.yaml">fast</a>.</p>')

FIGURES = {  # key: (file, anchor id, title, caption beside the figure)
    'timeline': ('fig_timeline.svg', 'fig-1', 'Timeline', TIMELINE_CAPTION),
    'pbt': ('fig_pbt.svg', 'fig-pbt', 'How PBT adapts the learning rate',
            'The five models (the population) start from the same pretrained model at different LRs. At each decision '
            'they are ranked by their mean Validation mistag over the last epochs. A model whose value is worse than the '
            'best by more than 0.002 in two decisions in a row copies the weights and optimizer state of the best model and '
            'continues at its LR times a factor. At most two models are replaced per decision; there is no decision '
            'at the end of training. The table gives the values of the two versions used.', _PBT_TABLE),
    'baseline': ('fig_baseline.svg', 'fig-baseline', 'Tuning the constant-LR baseline',
                 'Test (2.8M) mistag of the best constant-LR model of each grid, seed 12345, with the PBT results of '
                 'September (#3) and November (#15). As the grid moved to higher rates, the constant learning rate '
                 'passed the September PBT result and came within 0.5–1.0% of the faster November PBT. Right: #16, in '
                 'which the seed (22345) and the grid (6×10⁻⁵–1.6×10⁻⁴) changed together; there PBT was 0.47% '
                 'worse.'),
    'lr': ('fig_lr.svg', 'fig-2', 'PBT learning rates with Training (800k) and (8M)',
           'Learning rate per epoch, seed 12345. Blue: the best standard ParT model, which keeps its start LR for 35 '
           'epochs and then decays it (shaded); dotted: its start LR. Orange: the range of the five PBT models. With '
           'Training (800k), PBT keeps its learning rates around the best standard ParT start LR, 10⁻⁴. With Training '
           '(8M) the best start LR is 3×10⁻³, and from epoch 11 PBT stays more than ten times below it. The '
           'comparison is with the start LR, not with the decaying LR of epochs 36–50.'),
    'changes': ('fig_changes.svg', 'fig-changes', 'Changes to PBT with Training (8M)',
                'How much higher the Test mistag of each PBT run is than that of standard ParT (#24), seed 12345. Each '
                'change reduced the deficit; none removed it.'),
    'regimes': ('fig4_regimes.svg', 'fig-3', 'Training and Validation loss, Training (800k) and (8M)',
                'Best standard ParT model of seed 12345 (start LR 10⁻⁴ with Training (800k), 3×10⁻³ with Training '
                '(8M)); loss on the training data (dashed) and on Validation (1M) (solid). With Training (800k) the '
                'training loss falls below the Validation loss; with Training (8M) the two stay together. The training '
                'loss is measured with dropout active, which is why it starts above the Validation loss. An '
                'observation; its link to the PBT result is a hypothesis.'),
    'ensemble': ('fig_ensemble_gain.svg', 'fig-ensemble', 'Averaging two standard ParT models',
                 'Test (2.8M) mistag per seed of the better of the two standard ParT models (start LR 3×10⁻³, blue) '
                 'and of the average of the two (green). Each change is against the better of the two models, not the '
                 'weaker one. The best PBT run (#26, seed 12345) for comparison.'),
    'pairs': ('fig6_ensemble.svg', 'fig-4', 'Which pairs of models gain from averaging',
              'Every scored pair, on Validation (1M), scored after the models were trained (#29, #30): 105 pairs of the '
              'fifteen Training (800k) models of #18; 36 pairs of nine Training (8M) models of seed 12345 (#22–#26, '
              'including two checkpoints of #24 and three models of #26); the 15 pairs of the six standard ParT models '
              '(start LRs 3×10⁻³ and 10⁻³) of the three seeds (#22, #24, #28). The pair of #24 with the 10⁻³ model of '
              '#22 is in both sets and counted once. Ringed: the three ensembles of #27 and #28. Left: disagreement '
              'alone does not separate pairs that gain from pairs that do not. Right: pairs with a much weaker model '
              'do not gain. An association, not a causal effect.'),
}

# Compact primary-trial rows: date, experiment, Test mistag (%), result.
ROWS = {
    '1': ('Sep 16', 'Constant-LR grid, 3×10⁻⁶–1.4×10⁻⁵', '0.3269', 'best model at the highest rate'),
    '2': ('Sep 17–18', 'Two grid models trained to 100 epochs', '0.3156', 'reference for #4'),
    '3': ('Sep 18', 'Windowed PBT', '0.3225', '1.4% better than #1'),
    '4': ('Sep 19', 'Windowed PBT, 100 epochs', '0.3111', '1.4% better than #2'),
    '10': ('Oct 14–21', 'Preregistered comparison, Training (8M)', '0.2375*', '7.7% better than the grid (0.2572)'),
    '12': ('Nov 3', 'Constant LRs 1.4×10⁻⁵–3×10⁻⁵', '0.3146', 'best at the highest rate again'),
    '13': ('Nov 3–4', 'Constant LRs 3×10⁻⁵–8×10⁻⁵', '0.3018', 'better than every earlier PBT run'),
    '14': ('Nov 4', 'Constant LRs 8×10⁻⁵–3.2×10⁻⁴', '0.3033', 'best rate inside the grid'),
    '15': ('Nov 4', 'Fast PBT', '0.3003', '0.5% better than #13, 1.0% than #14'),
    '16': ('Nov 9', 'Seed 22345 and a re-centred grid', '0.3041', '0.47% worse than the grid (0.3027)'),
    '22': ('Nov 14–15', 'Standard ParT, start rates 10⁻⁵–10⁻³', '0.1635', 'best at the highest start rate'),
    '23': ('Nov 15–17', 'PBT', '0.1811', '12.4% worse than #24'),
    '24': ('Nov 19–20', 'Standard ParT, start rate 3×10⁻³', '0.1611', 'the reference'),
    '25': ('Nov 19–21', 'PBT with cool-down probes', '0.1699', '5.4% worse than #24'),
    '26': ('Nov 20–23', 'Probes, final decay and higher rates', '0.1655', '2.7% worse than #24'),
}
TABLE_NOTE = 'Test (2.8M) mistag (%) of the model selected on Validation. Each number links to its runs in the experiment log.'

# Blocks of a chapter body, in reading order: ('fig', key), ('p', html), ('table', note), ('seed18', note),
# ('seed28', note), ('note', id, html), ('analysis', summary, figure key, html after the figure).
CHAPTERS = [
    dict(phase=1, when='Sep 16 – Oct 21', trials='#1–#11',
         meta='Training (800k), then (8M) · Validation (150k) · seed 12345',
         q='Is PBT better than training at a constant learning rate?',
         a='In these comparisons, yes: by 1.4% and 7.7%. But the best constant learning rate was always the highest one '
           'tried.',
         primary=['1', '2', '3', '4', '10'], support=['5', '6', '7', '8', '9', '11'],
         body=[
             ('fig', 'pbt'),
             ('p', 'The first comparison used the simplest baseline: five models, each trained for 50 epochs at one '
                   'constant learning rate between 3×10⁻⁶ and 1.4×10⁻⁵. PBT, starting from the same five rates, was '
                   '1.4% better on Test (2.8M), and again 1.4% better when both were trained for 100 epochs; these '
                   'Test values were measured on 12 October (#8). A preregistered comparison with Training (8M), run '
                   'on 14–21 October and measured once on the separate Test (9.0M) on 2 November, gave 7.7%.'),
             ('p', 'The advantage was observed, but against a limited baseline. In every one of these comparisons the '
                   'best constant-LR model was the one with the highest rate of its grid, and the model PBT selected '
                   'had moved above that rate. The gain could therefore come from reaching a higher learning rate '
                   'rather than from adapting it.'),
             ('note', 'phase-0', '<span id="setup"></span><b>Before this, July–August:</b> first PBT implementations '
                      'and LR controllers with an older training set-up. Two defects were found (BatchNorm statistics '
                      'were updated although they should be frozen; an LR controller overwrote the LR set by PBT), and '
                      'the results are not used. From 16 September every trial trains deterministic full epochs.'),
             ('table', TABLE_NOTE + ' #1–#4 were measured on 12 October (#8). * #10: Test (9.0M), measured on 2 '
                       'November.'),
         ]),
    dict(phase=2, when='Nov 3 – Nov 9', trials='#12–#17', meta='Training (800k) · Validation (150k) · seed 12345, except #16',
         q='Was the PBT gain only a higher learning rate?',
         a='After the constant learning rate was tuned, PBT was 0.5–1.0% better on one seed and 0.47% worse on '
           'another.',
         primary=['12', '13', '14', '15', '16'], support=['17'],
         body=[
             ('fig', 'baseline'),
             ('p', 'The constant-LR grid was moved upwards in three steps; only the third grid had its best rate, '
                   '1.2×10⁻⁴, inside its range.'),
             ('p', 'With a second seed and a re-centred grid (#16), PBT was 0.47% worse. The seed and the grid changed '
                   'together, so the reversal is not assigned to either; Validation (150k) and Test also disagreed on '
                   'the order. The initial PBT advantage was reduced after tuning the constant-LR baseline. Subsequent '
                   'experiments did not establish a consistent advantage over tuned constant-LR training.'),
             ('p', '<b>Why fallback copies were kept (#17).</b> In #15 a model that lost could not be replaced when both '
                   'factors, ×0.5 and ×2, gave an LR another model already had. This happened about 24 times, and one '
                   'model stayed at 5×10⁻⁴ with the worst mistag from epoch 5 to 36. With fallback copies such a model '
                   'still copies the best one and uses ×0.7 or ×1.4. The run with fallback copies was 0.78% worse on '
                   'Test and also worse on Validation, on one seed. It was kept in all later PBT runs because it '
                   'removes this failure, not because it improved the mistag; its effect was not tested again. '
                   'Every PBT result from #18 on includes it.'),
             ('table', TABLE_NOTE),
         ]),
    dict(phase=3, when='Nov 10 – Nov 13', trials='#18–#21', meta='Training (800k) · Validation (1M) · three seeds',
         q='Does PBT beat the standard training recipe?',
         a='Slightly: better than standard ParT on each of three seeds, by 0.81% on average.',
         primary=['18'], support=['19', '20', '21'],
         body=[
             ('p', 'The comparison was repeated with three seeds, a larger Validation sample (1M) for selecting models, '
                   'and standard ParT as comparator: a constant rate for 35 epochs, then a decay by ×0.01 over 15, with '
                   'five start rates. All comparisons were fixed before training.'),
             ('seed18', 'Below each change: 95% Test-sample interval. It does not include seed-to-seed variation, which '
                        'the three rows show. Seed 32345 was trained after the comparison was fixed: a replication '
                        'with a new training seed under a fixed protocol, evaluated on the same Test sample.'),
             ('p', 'PBT was better than standard ParT on every seed, by 0.81% on average. Against the tuned constant '
                   'learning rate it was 0.65%, 0.03% and 0.94% better: a numerical improvement on every seed, not an '
                   'established, practically meaningful advantage.'),
         ]),
    dict(phase=4, when='Nov 14 – Nov 23', trials='#22–#26', meta='Training (8M) · Validation (1M) · seed 12345',
         q='Does the small gain hold with Training (8M)?',
         a='No: with Training (8M) PBT is 12.4% worse, on one seed. Two tested changes reduce the deficit to 2.7% but '
           'do not remove it.',
         primary=['22', '23', '24', '25', '26'], support=[],
         body=[
             ('fig', 'lr'),
             ('p', '<b>Hypothesis, not tested.</b> PBT judges each change two epochs later, which favours lower rates. '
                   'That costs little when the model overfits, as it does with Training (800k), and much when it does '
                   'not (training and Validation loss: supporting analysis below).'),
             ('fig', 'changes'),
             ('p', 'Cool-down probes judge each rate on a briefly decayed copy; #26 added a final decay together with '
                   'higher rates. The two changes of #26 were made at once, so its improvement over #25 is not '
                   'assigned to either, and the probes cost 9–12% more training. The reference #24 is a sixth standard '
                   'ParT model, while PBT has five. All of this rests on one seed.'),
             ('table', TABLE_NOTE),
             ('analysis', 'Supporting analysis: training and Validation loss', 'regimes', ''),
         ]),
    dict(phase=5, when='Nov 23 – Nov 25', trials='#27–#33', meta='Training (8M) · Validation (1M) · three seeds for #28',
         q='Can we gain without changing the learning-rate schedule?',
         a='Yes: averaging two independently trained standard ParT models is better than the better of the two on each '
           'of three seeds, by 5.01% on average.',
         primary=['27', '28'], support=['29', '30', '31', '32', '33'],
         body=[
             ('fig', 'ensemble'),
             ('p', 'No change of the schedule matched standard ParT, so we tried a gain outside it: averaging the class '
                   'probabilities of two standard ParT models with start rates 3×10⁻³ and 10⁻³, trained independently '
                   '(no weights copied between them). For seed 12345 the two models come from two runs (#22, #24); for '
                   'seeds 22345 and 32345 they are two members of one computational run (#28), which trains both start '
                   'rates side by side. The pair was chosen on seed 12345; the two other seeds were trained after all '
                   'rules were fixed, and evaluated on the same Test sample. It costs a second forward pass per jet.'),
             ('p', 'Two checkpoints of one model, or two models of one PBT population, which copy weights from each '
                   'other, disagree on few jets and do not gain. Independently trained models gain when both are of '
                   'similar quality (supporting analysis below); this was seen after the fact and is not a tested '
                   'explanation.'),
             ('seed28', 'Epoch: training epoch of the checkpoint with the best Validation (1M) value of that model. The '
                        '3×10⁻³ model was the better one on Validation (the preregistered reference) and on Test, on '
                        'every seed. Run names, members and checkpoint files: <a href="log.html#t28">experiment '
                        'log</a>.'),
             ('analysis', 'Supporting analysis: which pairs of models gain', 'pairs', ''),
         ]),
]

CONCLUSION = ('Within the tested configurations, PBT improved on standard ParT by less than 1% with Training (800k), was '
              'worse with Training (8M), and showed no consistent advantage over a tuned constant learning rate. The '
              'only large gain came from averaging two independently trained standard ParT models.')
LIMITS = [
    '<b>Seeds.</b> All Training (8M) PBT results and every tuning grid rest on one seed. The effect of the decision '
    'frequency (#6) is inconclusive; decisions within an epoch (#11) were never run.',
    '<b>Observations and hypotheses.</b> That the model overfits with Training (800k) and not with Training (8M), and '
    'that PBT lowered its LRs early, are observations. That they explain why PBT helps only with Training (800k), and '
    'why pairs of similar quality gain from averaging, are hypotheses, not tested directly.',
    '<b>Test reuse.</b> Test (2.8M) was evaluated in 15 campaigns and guided the next experiments. Only the new seeds of '
    '#18 (32345) and #28 (22345, 32345) are replications using new training seeds under fixed protocols, and they '
    'reused the same Test sample.',
    '<b>Scope.</b> One pretrained checkpoint, one optimizer, 50 epochs. No result was carried into a physics analysis.',
]

SETUP = [  # (label, html): definitions, budget and uncertainty
    ('Training samples', 'The names give source entries, not jets. Training (800k) holds the first 800 thousand source '
     'entries of each of the three flavour files (b, c, light): 2,392,232 jets after preselection. Training (8M) holds '
     'the first 8 million entries of each file: 23,921,449 jets. Training (800k) is a subset of Training (8M). From 10 '
     'November the last 333,333 rows of each Training (8M) file form Validation (1M) and are not trained on, so '
     'Training (8M) then has 22,921,450 training jets, about ten times Training (800k).'),
    ('Validation and Test', 'Validation (150k, until 9 November) and Validation (1M, from 10 November) select the model '
     'and drive PBT. Test (2.8M), 2,825,125 jets, is evaluated once per selected model. Test (9.0M), 8,970,477 jets, '
     'was used only in #10.'),
    ('Metric', 'The mistag is in % (0.3018 means 0.3018%): the geometric mean of the background mistag rates at b-tag '
     '80% and 90% (against c and light jets) and c-tag 50% and 80% (against b and light jets). Lower is better. “1.4% '
     'better” is a relative change of the mistag. Absolute values of Training (800k) and Training (8M) models differ by '
     'about a factor two and are not compared with each other.'),
    ('Compute budget', 'Each compared method trains five models for 50 epochs from the same pretrained checkpoint: one '
     'model per LR (constant-LR grid), one per start LR (standard ParT), or a population of five (PBT). Exceptions: #2 '
     'and #4 train to 100 epochs, #2 by continuing two models of #1; #24 and #33 each add one standard ParT model, so '
     'the Training (8M) reference is the best of six models while PBT has five; the probe runs #25 and #26 train 12% '
     'and 9% more; in #28 each new seed trains only the two standard ParT models of the ensemble, which needs two '
     'forward passes per jet. GPU time of every run is in the <a href="log.html">experiment log</a>.'),
    ('Uncertainty', 'Intervals are 95%, from a block bootstrap of the Test sample. They do not include seed-to-seed '
     'variation, which the seeds show directly. Results with one seed (the grids of #12–#16, all Training (8M) PBT '
     'runs) have no estimate of seed-to-seed variation.'),
    ('Use of the Test sample', 'Test (2.8M) was evaluated in 15 campaigns between 12 October and 25 November, and its '
     'results guided which experiment came next. Each comparison was fixed before its own Test evaluation, but Test is '
     'not an untouched sample: the PBT changes of #25–#26 and the ensemble of #27 were developed on seed 12345, whose '
     'Test results were known. The new seeds of #18 (32345) and #28 (22345, 32345) were trained after their '
     'comparisons were fixed: replications using new training seeds under fixed protocols, on the same Test sample. '
     'The intervals do not cover this repeated use.'),
    ('Reading the report', 'Each chapter lists its primary experiments; diagnostic checks and secondary modifications '
     'are under “Supporting experiments” and open from any link to them. # is the trial number; it links to the '
     'trial\'s runs, configuration files and full description in the <a href="log.html">experiment log</a>. Dates are '
     'those of the runs, in JST.'),
]
