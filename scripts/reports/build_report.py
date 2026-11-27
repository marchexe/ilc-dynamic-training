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
    ('Set-up', 'Training (800k).', [
        ('–', 'Jul 22 – Aug 31', 'First PBT versions (exploit-and-mutate, anchor-copy) and LR controllers; older training '
         'set-up (samples per epoch capped, damped optimizer start)',
         'Two defects found and fixed: BatchNorm not frozen; a controller overwrote the PBT LR',
         'Not comparable with later trials; training rebuilt as deterministic full epochs (Sep 16)'),
    ]),
    ('PBT against constant learning rates', 'Training (800k), Validation (150k), seed 12345 unless noted. From here: '
     'deterministic full epochs.', [
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
    ('Higher constant learning rates', 'Changed: constant-LR grid moved upwards; faster PBT. Training (800k), Validation (150k).', [
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
    ('PBT against standard ParT, 800k', 'Changed: Validation (1M); comparator standard ParT; three seeds.', [
        ('18', 'Nov 10–13', f'Standard ParT (start {lr("1", 5)}–{lr("1", 3)}), constant LR {lr("6", 5)}–{lr("1.6", 4)} and PBT as #17; '
         'seeds 12345, 22345, 32345', 'Standard 0.3029 / 0.3038 / 0.3024; constant 0.3028 / 0.3013 / 0.3024; PBT 0.3008 / 0.3012 / '
         '0.2996. <b>PBT 0.81% better</b> (95% interval 0.34–1.34%), at 7 of 8 working points',
         'Small but consistent PBT gain; the constant-LR grid is not stable across seeds'),
        ('19', 'Nov 12', 'PBT as #18 with a final LR decay (no decisions in the last 15 epochs); 2 seeds',
         'Same selected models as #18', 'The decay adds nothing at 800k'),
        ('20', 'Nov 12', 'Standard ParT from random weights', '0.378 on Validation, against about 0.30', 'Pretraining is essential'),
        ('21', 'Nov 12', 'Weight averages of nearby checkpoints and of PBT models', 'No gain (Validation)', ''),
    ]),
    ('PBT against standard ParT, 8M', 'Changed: Training (8M); one seed (12345).', [
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
    ('Ensembles and other ways to gain, 8M', 'Changed: from the LR schedule to combining models.', [
        ('27', 'Nov 23', 'Ensemble: mean class probabilities of #22 and #24; chosen among 8 ensembles on Validation, Test check fixed in advance',
         '0.1530; 5.05% better than #24 (interval 3.75–6.36%)', 'Large gain on the development seed'),
        ('28', 'Nov 23–25', f'#27 repeated: new standard ParT runs at {lr("3", 3)} and {lr("1", 3)}, seeds 22345 and 32345',
         '4.86% and 5.14% better; <b>mean of 3 seeds 5.01%</b> (interval 4.17–5.84%)', 'Confirmed on new seeds'),
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


def history():
    """One chronological table; each phase is a separator row with what changed against the previous phase."""
    body = []
    for k, (title, changed, rows) in enumerate(PHASES):
        body.append(f'<tr class="phase" id="phase-{k}"><td colspan="4"><b>Phase {k} · {title}</b>'
                    f'<span class="chg">{changed}</span></td></tr>')
        body += [f'<tr id="t{n}"><td class="d"><span class="tn">{"#" + n if n != "–" else ""}</span>{d}</td><td>{c}</td>'
                 f'<td>{r}</td><td class="i">{i}</td></tr>' for n, d, c, r, i in rows]
    return ('<div class="table-wrap"><table class="hist"><colgroup><col class="c-d"><col class="c-c"><col class="c-r">'
            '<col class="c-i"></colgroup><thead><tr><th>Date</th><th>Configuration</th><th>Result</th>'
            f'<th>Interpretation</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def main():
    page = (Path(__file__).with_name('report_template.html')).read_text().replace('<!--HISTORY-->', history())
    out = ROOT / 'docs/_extra/overview.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


if __name__ == '__main__':
    main()
