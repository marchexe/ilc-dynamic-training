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
    '1': ('Constant-LR grid', f'Five constant LRs, {_lr("3", 6)} to {_lr("1.4", 5)}', '0.3269', T,
          'reference for #3; best LR at the top of the grid'),
    '2': ('Constant LR, 100 epochs', 'Two models of #1 trained on to epoch 100', '0.3156', T, 'reference for #4'),
    '3': ('Windowed PBT', 'LR changed every 5 epochs, ×0.8 or ×1.2', '0.3225', T, '1.4% better than #1'),
    '4': ('Windowed PBT, 100 epochs', '#3 trained for 100 epochs', '0.3111', T, '1.4% better than #2'),
    '5': ('LR controller, observe only', 'Proposals logged, not applied', 'Not evaluable', '',
          'almost no proposals'),
    '6': ('PBT decision frequency', 'Every epoch against every 5 epochs; 2 seeds', '0.47% and 0.13% worse', T,
          'every epoch vs every 5 epochs; Validation shows the opposite'),
    '7': ('Restart within an epoch', 'Five restarted parts against one epoch', 'Weights differ', '',
          'optimizer state not saved'),
    '8': ('First Test evaluation', 'Nine fixed checkpoints of #1–#4 and #6', 'Rank correlation 0.70', '',
          'between Validation and Test'),
    '9': ('8M data preparation', 'Training (8M) and Test (9.0M) built', 'Data checked', '', 'GPU readiness test passed'),
    '10': ('Preregistered 8M comparison', 'Grid #1 against PBT #3 on Training (8M)', '0.2375', T9,
           '7.67% better than the constant-LR grid (0.2572)'),
    '11': ('Full-state checkpoints', 'Lookahead state saved for exact restarts', 'Bit-identical', '',
           'not used in a PBT run'),
    '12': ('Higher constant LRs, step 1', f'{_lr("1.4", 5)} to {_lr("3", 5)}', '0.3146', T,
           'best LR again at the top of the grid'),
    '13': ('Higher constant LRs, step 2', f'{_lr("3", 5)} to {_lr("8", 5)}', '0.3018', T,
           'better than every earlier PBT run; best LR again at the top'),
    '14': ('Higher constant LRs, step 3', f'{_lr("8", 5)} to {_lr("3.2", 4)}', '0.3033', T,
           f'best LR {_lr("1.2", 4)}, inside the grid'),
    '15': ('Fast PBT', 'Decisions every 2 epochs, ×0.5 or ×2, wider LR range', '0.3003', T,
           '1.0% better than #14 · 0.5% better than #13'),
    '16': ('Second seed', 'Seed 22345 for fast PBT and a constant-LR grid', '0.3041', T,
           '0.47% worse than the constant-LR grid (0.3027)'),
    '17': ('Fast PBT with fallback copies', 'A losing member is always replaced', '0.3026', T, '0.78% worse than #15'),
    '18': ('PBT against standard ParT', 'Validation (1M); three methods; three seeds', '0.81% better', T,
           'PBT than standard ParT, mean of 3 seeds (interval 0.34–1.34%)'),
    '19': ('PBT with final LR decay', 'No decisions in the last 15 epochs', 'Same models as #18', T, ''),
    '20': ('No pretraining', 'Standard ParT from random weights', '0.378', V, 'against about 0.30 with pretraining'),
    '21': ('Weight averaging', 'Averages of nearby checkpoints and of members', 'At most 0.30% better', V,
           'standard ParT only; not evaluated on Test'),
    '22': ('Standard ParT, 8M', f'Start LRs {_lr("1", 5)} to {_lr("1", 3)}', '0.1635', T,
           'best start LR at the top of the range'),
    '23': ('PBT, 8M', 'PBT as #18 with the LR range of standard ParT', '0.1812', T, '12.4% worse than #24'),
    '24': (f'Standard ParT, start LR {_lr("3", 3)}', 'One more model, by a rule fixed in advance', '0.1611', T,
           '8M reference'),
    '25': ('PBT with cool-down probes', 'Members ranked after a short low-LR probe', '0.1699', T,
           '5.4% worse than #24'),
    '26': ('PBT with probes, final decay and higher LRs', 'Two changes at once', '0.1655', T, '2.7% worse than #24'),
    '27': ('Ensemble of two standard ParT models', 'Average output of #22 and #24', '0.1530', T,
           '5.05% better than #24'),
    '28': ('Ensemble on two new seeds', 'Seeds 22345 and 32345', '5.01% better', T,
           'than the better single model, mean of 3 seeds (interval 4.17–5.84%)'),
    '29': ('Ensembles at 800k', 'Any two of 15 models of #18', 'No gain', V, 'beyond chance among 105 pairs'),
    '30': ('Source of the ensemble gain', 'Disagreement and gain of 8M model pairs', 'Separate training', V,
           'separately trained standard ParT pairs gain; same-run pairs do not'),
    '31': ('No dropout in the final decay', 'Decay of #24 repeated with dropout 0', '0.1730', V,
           'against 0.1595 with dropout'),
    '32': ('Distillation', 'One model trained on the ensemble output', '0.1600', V, 'against 0.1584 without'),
    '33': (f'Standard ParT, start LR {_lr("1", 2)}', 'One more model, by a rule fixed in advance', '0.1631', V,
           'worse than #24 (0.1584); 0.1599 on Test'),
}

KEY = {'10', '13', '16', '18', '23', '26', '28'}          # decisive results: emphasised
BASE = {'1', '2', '12', '14', '22', '24'}                 # reference runs: subordinate

PHASE_SUMMARY = [
    ('Build the tools.', 'First PBT versions and LR controllers.', 'Two defects; the results are not usable.',
     'Rebuild training as deterministic full epochs.'),
    ('Test whether PBT beats constant learning rates.', 'PBT against a grid of five constant LRs, on 800k and then 8M '
     'training jets.', 'PBT is 1.4% better (800k) and 7.7% better (8M). But the best constant LR is always the top of '
     'its grid.', 'Check whether higher constant LRs explain the gain.'),
    ('Was the PBT gain only a higher LR?', 'The constant-LR grid moved up to 3.2×10<sup>−4</sup>; a faster PBT.',
     'A constant LR near 10<sup>−4</sup> matches PBT. With a second seed the order reverses.',
     'Select on a larger Validation sample (1M), compare with standard ParT, use three seeds.'),
    ('A fair test against the usual training recipe.', 'Validation (1M); comparator standard ParT (constant LR for 35 '
     'epochs, then ×0.01 over 15; five start LRs); three seeds.', 'PBT is 0.81% better on every seed.',
     'Repeat with ten times more training data.'),
    ('Does the small gain hold with more data?', 'Training (8M); one seed; then two corrections to PBT.',
     'PBT is 12.4% worse; the corrections reduce this to 2.7%.', 'Look for a gain outside the LR schedule.'),
    ('Can we gain without changing the LR schedule?', 'Average the outputs of two separately trained standard ParT '
     'models.', 'The ensemble is 5.0% better on three seeds. No single model keeps the gain.', 'Current state.'),
]
