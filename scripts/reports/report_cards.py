"""Short, scannable descriptions for the research report and the experiment log.

CARD: trial -> (name, key change, headline value, sample, comparison). The full configuration, result and
interpretation of every trial stay in build_report.PHASES; these cards only lead the eye. Values are the mistag in
% of the selected model unless the value itself is a relative change.
PHASE_SUMMARY: phase -> (why, what changed, what we found, why we moved on).
"""


def _lr(m, e):
    return (f'{m}×' if m != '1' else '') + f'10<sup>−{e}</sup>'


T, V, T9 = 'Test', 'Validation', 'Test (9.0M)'

CARD = {
    '–': ('Initial implementation', 'First PBT versions and LR controllers', 'No usable result', '', 'two defects fixed'),
    '1': ('Constant-LR grid', f'Five models, each at one constant LR from {_lr("3", 6)} to {_lr("1.4", 5)}; 50 epochs',
          '0.3269', T, 'reference for windowed PBT (#3); best model at the highest LR of the grid'),
    '2': ('Constant LR, 100 epochs', f'The {_lr("1.4", 5)} and {_lr("8.5", 6)} models of the grid trained on to epoch 100',
          '0.3156', T, 'reference for 100-epoch PBT (#4)'),
    '3': ('Windowed PBT', 'Five models, same start LRs as the grid; every 5 epochs the weakest copies the best and '
          'changes its LR by ×0.8 or ×1.2', '0.3225', T, '1.4% better than the constant-LR grid (#1)'),
    '4': ('Windowed PBT, 100 epochs', 'The same PBT trained for 100 epochs', '0.3111', T,
          '1.4% better than the 100-epoch constant LR (#2)'),
    '5': ('LR controller, observe only', 'Proposals logged, not applied', 'Not evaluable', '',
          'almost no proposals'),
    '6': ('PBT decision frequency', 'Every epoch against every 5 epochs; 2 seeds', '0.47% and 0.13% worse', T,
          'every epoch vs every 5 epochs; Validation shows the opposite'),
    '7': ('Restart within an epoch', 'Five restarted parts against one epoch', 'Weights differ', '',
          'optimizer state not saved'),
    '8': ('First Test evaluation', 'Nine fixed checkpoints of #1–#4 and #6', 'Rank correlation 0.70', '',
          'between Validation and Test'),
    '9': ('8M data preparation', 'Training (8M) and Test (9.0M) built', 'Data checked', '', 'GPU readiness test passed'),
    '10': ('Preregistered 8M comparison', 'The same constant-LR grid and windowed PBT, trained on Training (8M); one '
           'evaluation on the separate Test (9.0M)', '0.2375', T9,
           '7.67% better than the constant-LR grid (0.2572); again the best constant LR was the highest of the grid'),
    '11': ('Full-state checkpoints', 'Lookahead state saved for exact restarts', 'Bit-identical', '',
           'not used in a PBT run'),
    '12': ('Higher constant LRs, step 1', f'Five constant LRs from {_lr("1.4", 5)} to {_lr("3", 5)}', '0.3146', T,
           f'best model at the highest LR ({_lr("3", 5)}) again'),
    '13': ('Higher constant LRs, step 2', f'Five constant LRs from {_lr("3", 5)} to {_lr("8", 5)}', '0.3018', T,
           f'better than every earlier PBT run; best model at the highest LR ({_lr("8", 5)}) again'),
    '14': ('Higher constant LRs, step 3', f'Five constant LRs from {_lr("8", 5)} to {_lr("3.2", 4)}', '0.3033', T,
           f'best model at {_lr("1.2", 4)}, inside the grid for the first time'),
    '15': ('Fast PBT', f'Decisions every 2 epochs instead of 5, LR ×0.5 or ×2, start LRs {_lr("1", 5)}–{_lr("3.5", 4)}, '
           f'upper bound {_lr("5", 4)}', '0.3003', T,
           '1.0% better than the step-3 grid (#14, the preregistered reference) and 0.5% better than the step-2 grid (#13)'),
    '16': ('Second seed, re-centred grid', f'Seed 22345: fast PBT unchanged; constant-LR grid moved to {_lr("6", 5)}–'
           f'{_lr("1.6", 4)} around the optimum', '0.3041', T,
           '0.47% worse than the constant-LR grid (0.3027); seed and grid changed together'),
    '17': ('Fast PBT with fallback copies', 'A losing member is always replaced', '0.3026', T, '0.78% worse than #15'),
    '18': ('PBT against standard ParT', f'Selection on Validation (1M); standard ParT (start LRs {_lr("1", 5)}–'
           f'{_lr("1", 3)}), the constant-LR grid and fast PBT, each with seeds 12345, 22345, 32345',
           '0.81% better', T, 'PBT than standard ParT, mean of three seeds (better on each: 0.67%, 0.85%, 0.91%); '
           'against the tuned constant LR 0.65%, 0.03%, 0.94% better. Per seed in the table below'),
    '19': ('PBT with final LR decay', 'No decisions in the last 15 epochs', 'Same models as #18', T, ''),
    '20': ('No pretraining', 'Standard ParT from random weights', '0.378', V, 'against about 0.30 from the pretrained '
           'model'),
    '21': ('Weight averaging', 'Averages of nearby checkpoints and of members', 'At most 0.30% better', V,
           'standard ParT only; not evaluated on Test'),
    '22': ('Standard ParT, 8M', f'Standard ParT on Training (8M), start LRs {_lr("1", 5)} to {_lr("1", 3)}',
           '0.1635', T, f'best model at the highest start LR ({_lr("1", 3)})'),
    '23': ('PBT, 8M', f'Fast PBT as in #18 on Training (8M), start LRs and upper bound as standard ParT '
           f'({_lr("1", 5)}–{_lr("1", 3)})', '0.1811', T, '12.4% worse than standard ParT at 3×10<sup>−3</sup> (#24)'),
    '24': (f'Standard ParT, start LR {_lr("3", 3)}', 'One more standard ParT model, added by a rule fixed in advance',
           '0.1611', T, 'the 8M reference: better than #22 on Validation'),
    '25': ('PBT with cool-down probes', 'PBT as #23, but each LR is judged on a copy trained a quarter epoch at '
           '0.1× that LR', '0.1699', T, '5.4% worse than standard ParT (#24)'),
    '26': ('PBT with probes, final decay and higher LRs', f'Probe PBT (#25) plus the final LR decay of standard ParT, '
           f'and start LRs moved up to {_lr("3", 5)}–{_lr("3", 3)}', '0.1655', T,
           '2.7% worse than standard ParT (#24); the two changes are not separated'),
    '27': ('Ensemble of two standard ParT models', f'Average output of the {_lr("1", 3)} model (#22) and the '
           f'{_lr("3", 3)} model (#24), seed 12345', '0.1530', T,
           '5.05% better than the better of the two models (#24, 0.1611)'),
    '28': ('Ensemble on two new seeds', f'The same pair ({_lr("3", 3)} and {_lr("1", 3)}) trained with seeds 22345 '
           'and 32345, all rules fixed before training', '5.01% better', T, 'than the better of the two models, mean of three '
           'seeds (5.05%, 4.86%, 5.14%); interval 4.17–5.84%. Models and checkpoints in the table below'),
    '29': ('Ensembles, Training (800k)', 'Any two of the 15 models of #18 (105 pairs)', 'No consistent gain', V,
           'the best pairs 0.9% better than the best single model'),
    '30': ('Source of the ensemble gain', 'Disagreement and gain of 51 pairs of 8M models', 'Similar quality', V,
           'independently trained pairs gain if both models are of similar quality; two checkpoints of one model, or two '
           'models of one PBT population, never gain'),
    '31': ('No dropout in the final decay', 'Decay of #24 repeated with dropout 0', '0.1730', V,
           'against 0.1595 with dropout'),
    '32': ('Distillation', 'One model trained on the ensemble output', '0.1600', V, 'against 0.1584 without'),
    '33': (f'Standard ParT, start LR {_lr("1", 2)}', 'One more model, by a rule fixed in advance', '0.1631', V,
           'worse than #24 (0.1584); 0.1599 on Test'),
}


PHASE_SUMMARY = [  # None: the entry adds nothing to the trial table below it and is not shown
    ('Build the tools.', None, 'Two defects; the results are not usable.',
     'Rebuild training as deterministic full epochs.'),
    ('Test whether PBT beats constant learning rates.', None, 'PBT is 1.4% better (Training 800k) and 7.7% better '
     '(Training 8M). But the best constant LR is always the highest LR of its grid.',
     'Check whether higher constant LRs explain the gain.'),
    ('Was the PBT gain only a higher LR?', 'The constant-LR grid moved up in three steps to 3.2×10<sup>−4</sup>; a faster '
     'PBT.', 'A tuned constant LR comes within 0.5–1.0% of PBT. With a new seed and a new grid, PBT is 0.47% worse.',
     'Select on a larger Validation sample (1M), compare with standard ParT, use three seeds.'),
    ('A controlled test against the usual training recipe.', 'Validation (1M); comparator standard ParT (constant LR for '
     '35 epochs, then ×0.01 over 15; five start LRs); three seeds.', 'PBT is better than standard ParT on every seed, by '
     '0.81% on average. Against the tuned constant LR the gain is not consistent.',
     'Repeat with ten times more training data.'),
    ('Does the small gain hold with ten times more training data?', 'Training (8M), one seed; then cool-down probes '
     '(#25), then a final LR decay together with higher LRs (#26).', 'PBT is 12.4% worse; with these changes still 2.7% '
     'worse.', 'Look for a gain outside the LR schedule.'),
    ('Can we gain without changing the LR schedule?', 'Average the class probabilities of two separately trained standard '
     'ParT models.', 'The average is better than the better model on each of three seeds, by 5.01% on average. No single-'
     'model tried (#31–#33) reached it.', None),
]
