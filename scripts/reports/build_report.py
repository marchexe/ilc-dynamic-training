#!/usr/bin/env python3
"""Build the supervisor report docs/_extra/overview.html.

    .venv/bin/python scripts/reports/publication_figures.py   # figures, docs/_extra/pub/*.svg
    .venv/bin/python scripts/reports/build_report.py

Every number is taken from docs/results.rst, docs/ongoing.rst or the run manifests (dates in JST); the trial
numbers are those of the experiment log (docs/_extra/log.html, scripts/reports/build_log.py).
"""
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_cards import CARD, PHASE_SUMMARY  # noqa: E402
import report_text as X  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def lr(m, e):
    return (f'{m}×' if m != '1' else '') + f'10<sup>−{e}</sup>'


E = lambda m, e: lr(m, e)
G1 = f'{E("3", 6)}, {E("5.75", 6)}, {E("8.5", 6)}, {E("1.125", 5)}, {E("1.4", 5)}'
WIN = (f'Windowed PBT: every 5 epochs the members are ranked by their mean Validation mistag over the last 3 epochs; a '
       'member that loses in two windows by more than 0.002 copies the weights and optimizer state of the best member '
       f'and continues at its LR ×0.8 or ×1.2 (at most two copies per decision; LR bounds {E("2", 6)}–{E("3", 5)})')
FAST = (f'Fast PBT: as windowed PBT, but decisions every 2 epochs on the mean of the last 2 epochs, LR ×0.5 or ×2, bounds '
        f'{E("5", 6)}–{E("5", 4)}; start LRs {E("1", 5)}, {E("2.5", 5)}, {E("6", 5)}, {E("1.5", 4)}, {E("3.5", 4)}')
V = ' <i>(Validation)</i>'
SUP = lambda m, e: lr(m, e)  # noqa: E731

# (phase title, conditions, rows); row: (trial, date, configuration, result, interpretation)
# Results are mistag (%) of the selected model on Test (2.8M) unless marked (Validation); "x% better/worse than #n" is
# the relative change of that value against trial #n.
PHASES = [
    ('Initial implementation: two defects, no usable result', 'Training (800k)', [
        ('–', 'Jul 22 – Aug 31', 'First PBT implementations (exploit-and-mutate, anchor-copy) and LR controllers, with an '
         'older training set-up: a capped number of training jets per epoch and a damped optimizer start',
         'Two defects found and fixed: BatchNorm statistics were updated although they should be frozen, and an LR '
         'controller overwrote the LR set by PBT',
         'Results not comparable with later trials and not used. Training was rebuilt as deterministic full epochs '
         '(from Sep 16), which all later trials use'),
    ]),
    ('PBT is better than the constant-LR grid, but the best constant LR is always the highest LR tried', 'Training (800k), then (8M) · Validation (150k) · seed 12345 unless noted', [
        ('1', 'Sep 16', f'Constant-LR grid: five models, each at one LR for 50 epochs: {G1}',
         '0.3269; selected model: the highest LR, 1.4×10<sup>−5</sup>, epoch 44',
         'First comparator for PBT. That the best LR was the top of the grid was noted but not followed up until #12'),
        ('2', 'Sep 17–18', f'The {E("1.4", 5)} and {E("8.5", 6)} models of #1 continued from epoch 50 to 100 at the same LR',
         f'0.3156 ({E("1.4", 5)}, epoch 99)', 'Longer training at the same LR helps; comparator for #4'),
        ('3', 'Sep 18', f'{WIN}; start LRs as #1', '0.3225; 1.4% better than #1',
         'Apparent gain of PBT. Its selected model had reached 2.0×10<sup>−5</sup>, above the grid of #1'),
        ('4', 'Sep 19', 'PBT as #3, 100 epochs', '0.3111; 1.4% better than #2',
         'Same apparent gain at 100 epochs. The comparator #2 is one model, not a population'),
        ('5', 'Sep 20', f'Rule-based LR controller in observe-only mode (it logs proposals, ×1.05 or ×0.95, but does not apply '
         f'them), reading a 60k-jet subset of Validation; eight constant-LR models {E("3", 6)}–{E("1.4", 5)}, 96 short '
         'epochs of 120k jets each (older set-up)',
         'The 60k subset ranks checkpoints like the full Validation (150k) (rank correlation 0.96) but detects an '
         'improvement between epochs in only 72% of cases. The controller proposed two LR increases, both blocked by its '
         'LR bound, and no decrease',
         'The controller could not be evaluated; not pursued'),
        ('6', 'Sep 23–29', 'Cadenced PBT: after 2 warm-up epochs the best member replaces the worst (at most one per '
         'decision), LR ×0.8 or ×1.2, bounds as #3; decisions every epoch against every 5 epochs; seeds 12345 and 22345 '
         '(four runs). The rules differ from #3 in more than the frequency',
         'Every-epoch PBT 0.47% (seed 12345) and 0.13% (seed 22345) worse than every-5-epochs PBT on Test, '
         'but better on Validation in both seeds',
         'Inconclusive: Validation (150k) and Test disagree on the order at this size of difference'),
        ('7', 'Sep 23–28 (exact date not recorded)', 'Restart test: one epoch trained as five parts, each restarted from '
         'the checkpoint of the previous part, against one uninterrupted epoch',
         f'Model weights differ (largest difference {E("7.6", 4)})',
         'PBT decisions within an epoch are not possible with these checkpoints, because the state of the Lookahead '
         'optimizer is not saved (fixed in #11)'),
        ('8', 'Oct 12', 'First evaluation on Test (2.8M): the pretrained start and the selected models of #1–#4 and #6 '
         '(nine checkpoints), fixed in advance',
         'Every checkpoint 0.010–0.019 lower on Test than on Validation; rank correlation 0.70 between the two samples',
         'Large differences keep their order; differences of about 0.3% reorder. Validation (150k) cannot resolve them'),
        ('9', 'Oct 13', 'Preparation of Training (8M, 23.9M jets) and of the separate Test (9.0M), kept closed until #10 had '
         'finished; short GPU readiness test of 8M training',
         'Data files built and checked; the readiness test passed', 'Prerequisite for #10'),
        ('10', 'Oct 14–21; Test opened Nov 2', 'Preregistered comparison on Training (8M): constant-LR grid as #1 against PBT as '
         '#3, seed 12345, 50 epochs; selected models evaluated once on Test (9.0M). Before the arms, a one-epoch gate run '
         'failed on a defect in its own check (gradient norms of skipped overflow steps counted as failures) and passed '
         'after the check was corrected; the epoch reproduced bit for bit',
         'PBT 0.2375 against 0.2572 on Test (9.0M): 7.67% better (6.74% on Test (2.8M)). All ten preregistered validity '
         'checks passed',
         'The protocol was followed correctly. As a scientific comparison it is limited: the best constant LR was again the '
         'highest of the grid and PBT ended above it (3×10<sup>−5</sup>), so adapting the LR and reaching a higher LR are '
         'not separated'),
        ('11', 'Oct 17 – Nov 3', 'Checkpoints that also save the Lookahead optimizer state, so that a run can be restarted '
         'within an epoch without changing its result; GPU check on Nov 3',
         'One epoch trained as five restarted parts equals one uninterrupted epoch bit for bit',
         'Implemented; not used in any PBT run'),
    ]),
    ('After tuning the constant-LR baseline, the PBT advantage is smaller and not consistent', 'Training (800k) · Validation (150k) · seed 12345 unless noted', [
        ('12', 'Nov 3', f'Constant-LR grid {E("1.4", 5)}, {E("1.8", 5)}, {E("2.2", 5)}, {E("2.6", 5)}, {E("3", 5)}, '
         'because the best LR of #1 and #10 was the top of the grid',
         f'0.3146; selected LR {E("3", 5)}, again the top of the grid', 'The optimum lies higher still'),
        ('13', 'Nov 3–4', f'Constant-LR grid {E("3", 5)}, {E("4", 5)}, {E("5", 5)}, {E("6", 5)}, {E("8", 5)}',
         f'0.3018; selected LR {E("8", 5)}, again the top of the grid (Validation 0.3166)',
         'Better than every PBT run of #3–#6, so those runs had been compared with a constant-LR baseline that was not tuned. '
         'Grid extended once more'),
        ('14', 'Nov 4', f'Constant-LR grid {E("8", 5)}, {E("1.2", 4)}, {E("1.6", 4)}, {E("2.4", 4)}, {E("3.2", 4)}',
         f'0.3033; selected LR {E("1.2", 4)}, inside the grid (Validation 0.3141, better than #13)',
         f'For this seed the best constant LR lies in {E("8", 5)}–{E("1.2", 4)}. #14 is better than #13 on Validation '
         'but worse on Test'),
        ('15', 'Nov 4', FAST,
         '0.3003: 1.0% better than #14 (the preregistered comparator: of #13 and #14, the one with the better Validation '
         'mistag) and 0.5% better than #13 (the better one on Test)',
         'The gap to the best constant LR is 0.5% or 1.0%, depending on whether #13 or #14 is the reference'),
        ('16', 'Nov 9', f'Second seed (22345): fast PBT exactly as #15, and a new constant-LR grid {E("6", 5)}, {E("8", 5)}, '
         f'{E("1", 4)}, {E("1.2", 4)}, {E("1.6", 4)}, centred on the optimum of #13–#14 (not the grid of #14)',
         'PBT 0.3041 against constant LR 0.3027: 0.47% worse on Test, although 1.4% better on Validation',
         'The order reverses; the seed and the constant-LR grid changed together, so the cause is not separated. Validation '
         'and Test also disagree: selection noise of about 0.5% is as large '
         'as the effect'),
        ('17', 'Nov 9', 'Fast PBT as #15 (seed 12345) with fallback copies. In #15 a losing member was not replaced when '
         'both ×0.5 and ×2 gave an LR another member already had (about 24 times; one member stayed at 5×10<sup>−4</sup> '
         'with the worst mistag from epoch 5 to 36). With fallback copies it still copies the best member and uses ×0.7 '
         'or ×1.4, whichever LR is farthest from the others',
         '0.3026: 0.78% worse than #15 on Test; also worse on Validation (0.3166 against 0.3156)',
         'No improvement; kept in later PBT runs because it removes the stuck member. Next: Validation (1M), standard '
         'ParT as comparator, three seeds'),
    ]),
    ('Training (800k): PBT 0.81% better than standard ParT on three seeds', 'Training (800k) · Validation (1M) · 3 seeds', [
        ('18', 'Nov 10–13', f'Three methods, each with seeds 12345, 22345, 32345: standard ParT (constant LR for 35 epochs, '
         f'then ×0.01 over 15; start LRs {E("1", 5)}, {E("3", 5)}, {E("1", 4)}, {E("3", 4)}, {E("1", 3)}); constant-LR '
         f'grid as #16 ({E("6", 5)}–{E("1.6", 4)}); PBT as #17. Comparisons fixed before training',
         'Per seed in the table below. Standard ParT selected start LR 10<sup>−4</sup> on every seed. PBT better than '
         'standard ParT by 0.67%, 0.85% and 0.91%: <b>mean 0.81%</b>, Test-sample interval 0.34–1.34%; better at 7 of 8 '
         'working points',
         'PBT has a lower mistag than standard ParT on every seed. Against the tuned constant-LR grid, PBT is numerically '
         'lower on every seed (0.65%, 0.03%, 0.94%), but negligibly so for seed 22345 (interval 0.74% better to 0.57% worse): not an '
         'established, practically meaningful advantage over tuned constant-LR training'),
        ('19', 'Nov 12', 'PBT as #18 with the final decay of standard ParT: no decisions in the last 15 epochs, all LRs '
         '×0.01 over them; seeds 12345 and 22345',
         'The same models are selected as in #18 (epochs 31 and 33, before the decay starts), so the same Test values',
         'The decay does not change the result with Training (800k)'),
        ('20', 'Nov 12', f'Standard ParT from random weights instead of the pretrained model, start LRs '
         f'{E("2.5", 4)}–{E("4", 3)}, 50 epochs',
         f'0.378{V} (start LR {E("4", 3)}, epoch 14), against about 0.30 from the pretrained model; not evaluated on Test',
         'Without the pretrained start the model is far worse after 50 epochs'),
        ('21', 'Nov 12', 'Weight averages of the selected checkpoint with nearby epochs (3–10 epochs) and of population '
         'members, for the six runs of seeds 12345 and 22345 in #18',
         f'Standard ParT: averages 0.09% and 0.30% better than the selected checkpoint{V}; constant LR and PBT: no '
         'average better than the selected checkpoint. The planned Test evaluation was not run',
         'At most a small change on Validation; not pursued'),
    ]),
    ('Training (8M): PBT is worse than standard ParT; changes reduce the deficit', 'Training (8M) · Validation (1M) · seed 12345', [
        ('22', 'Nov 14–15', f'Standard ParT as #18, Training (8M), start LRs {E("1", 5)}–{E("1", 3)}',
         f'0.1635; selected start LR {E("1", 3)}, the top of the range', 'The range was extended in #24'),
        ('23', 'Nov 15–17', f'PBT as #18 with the start LRs and upper LR bound of standard ParT ({E("1", 5)}–{E("1", 3)})',
         '0.1811: 12.4% worse than #24 (10.8% worse than #22, the first reported value)',
         f'PBT lowered all LRs to at most {E("2.5", 4)} from epoch 11'),
        ('24', 'Nov 19–20', f'Standard ParT, one model at start LR {E("3", 3)}, added under a rule fixed before it started: '
         'the 8M reference is whichever of #22 and #24 is better on Validation',
         '0.1611 (Validation 0.1584, better than #22): the 8M reference', 'Reference for #23, #25, #26 and #27'),
        ('25', 'Nov 19–21', 'PBT as #23 with cool-down probes: at each decision a copy of every member trains a quarter '
         'epoch at 0.1× its LR, and the members are ranked on these copies (120 probes, about 12% more training). Also run '
         'on Training (800k)',
         '8M: 0.1699, 5.4% worse than #24. 800k: 0.3024, against 0.3008 for PBT (#18) and 0.3029 for standard ParT, seed 12345',
         f'The LRs stay higher ({E("1.2", 4)}–{E("1", 3)}); the gap to standard ParT halves'),
        ('26', 'Nov 20–23', f'PBT with probes as #25, changed in two ways at once: (a) the final decay of #19 and (b) a '
         f'higher LR range, start LRs {E("3", 5)}, {E("1", 4)}, {E("3", 4)}, {E("1", 3)}, {E("3", 3)} and upper bound '
         f'{E("3", 3)} (#25: {E("1", 5)}–{E("1", 3)}). 85 probes, about 9% more training',
         '0.1655: 2.7% worse than #24, against 5.4% for #25',
         '78% of the gap to standard ParT closed; still worse. The two changes were made together, so the improvement '
         'over #25 cannot be assigned to either. Two further seeds were prepared, with the rule that they run only if '
         'this run is not clearly worse than #24; they were not run'),
    ]),
    ('Training (8M): averaging two standard ParT models is 5% better', 'Training (8M) · Validation (1M) · 3 seeds for #28', [
        ('27', 'Nov 23', 'Ensemble: average of the class probabilities of the selected models of #22 (start LR '
         f'{E("1", 3)}) and #24 ({E("3", 3)}), seed 12345; chosen as the best of eight candidate ensembles on Validation, '
         'Test comparison fixed before evaluation',
         '0.1530: 5.05% better than #24 (Test interval 3.75–6.36%)', 'Large gain on the seed on which it was chosen'),
        ('28', 'Nov 23–25', f'Test of #27 on new seeds: standard ParT at {E("3", 3)} and {E("1", 3)} with seeds 22345 and '
         '32345; models, ensemble and the three-seed claim fixed before training',
         f'Ensembles 0.1519 and 0.1540 against 0.1597 and 0.1623 (the {E("3", 3)} models): 4.86% and 5.14% better. '
         '<b>Mean of three seeds 5.01%</b>, Test interval 4.17–5.84%',
         'Replicated on two new seeds. Cost: two model forward passes per jet'),
        ('29', 'Nov 23', 'Ensembles of any two of the 15 selected 800k models of #18 (105 pairs), on Validation',
         f'The best pairs about 0.9% better than the best single model{V}; two standard ParT {E("1", 4)} models of different '
         'seeds 0.0% on average',
         'No consistent gain with Training (800k); the best pairs were picked after the fact (Figure 4)'),
        ('30', 'Nov 23–25', 'Disagreement (share of jets with a different predicted class) and ensemble change for pairs '
         'of 8M models, on Validation: all pairs of nine seed-12345 models (#22–#26), and the 15 pairs of the '
         f'{E("3", 3)} and {E("1", 3)} standard ParT models of the three seeds (#22, #24, #28)',
         'Two models with a shared training history (two checkpoints of one model, or two models of one PBT population): '
         'disagreement 0.22–0.40%, ensemble 0.04–0.81% worse than the better model. Independently trained standard ParT '
         'models (19 pairs): 1.49–2.14%, from 0.14% worse to 6.7% '
         'better; the 15 three-seed pairs gain 3.6% on average with the same start LR and 4.7% with different start LRs. '
         f'A PBT model with a model trained apart from it (27 pairs): 1.02–2.32%, from 2.0% worse to 4.7% better{V}',
         'Disagreement alone does not predict the gain. Every pair that gains more than 1% has two models of similar '
         'quality (the weaker at most 6% worse); no pair whose weaker model is more than 9% worse gains more than 1%, '
         'with or without PBT. Exploratory, scored after the fact (Figure 4)'),
        ('31', 'Nov 23–24', 'The final decay of #24 (epochs 36–50) repeated from its epoch-35 checkpoint with dropout 0',
         f'0.1730 against 0.1595 with dropout at epoch 44{V}; stopped at epoch 45 when the process was lost on the node and '
         'not restarted', 'Without dropout this final decay was much worse; one run, not repeated'),
        ('32', 'Nov 24', 'Distillation: the final decay of #24 repeated from epoch 35 with a loss of half the usual '
         'cross-entropy and half the cross-entropy against the ensemble probabilities of #27',
         f'Best 0.1600 against 0.1584 without distillation{V}; 0.1604 against 0.1597 at the last epoch. The full run was '
         'not started', 'The ensemble gain was not transferred to one model'),
        ('33', 'Nov 24–25', f'Standard ParT, one model at start LR {E("1", 2)}, added under a rule fixed before it started',
         f'0.1631{V}, worse than #24 (0.1584), so #24 stays the reference; 0.1599 on Test',
         f'The Test value lies within the seed spread of {E("3", 3)} (0.1597–0.1623): in this one run a higher start LR '
         'gave no gain'),
    ]),
]


PUB = ROOT / 'docs/_extra/pub'


def asset(name):
    """pub/<name>?v=<content hash>: a changed file gets a new URL, so no browser serves a stale copy."""
    import hashlib
    return f'pub/{name}?v={hashlib.sha256((PUB / name).read_bytes()).hexdigest()[:12]}'


def xref(html):
    """Link trial references such as #18 to their row on this page."""
    return re.sub(r'(?<![&\w])#(\d+)', r'<a class="tref" href="#t\1">#\1</a>', html)


FIGDATA = ROOT / 'published/results/report/figure_data.json'


def _iv(lo, hi):
    """95% interval of a relative change (negative = lower mistag), in words."""
    if hi < 0:
        return f'{-hi:.2f}–{-lo:.2f}% better'
    if lo > 0:
        return f'{lo:.2f}–{hi:.2f}% worse'
    return f'{-lo:.2f}% better to {hi:.2f}% worse'


def _ch(c):
    return f'{abs(c):.2f}% {"better" if c < 0 else "worse"}'


def table_18(d):
    """#18 per seed: absolute Test values of the three methods and PBT's relative change against each comparator."""
    o = d['overview']['sel800k']
    rows = ''.join(
        f'<tr><td>{s["seed"]}</td><td class="num">{s["standard"]:.4f}</td><td class="num">{s["constant"]:.4f}</td>'
        f'<td class="num"><b>{s["pbt"]:.4f}</b></td><td>{_ch(s["pbt_vs_standard"]["change"])}'
        f'<span class="iv">{_iv(*s["pbt_vs_standard"]["interval"])}</span></td>'
        f'<td>{_ch(s["pbt_vs_constant"]["change"])}<span class="iv">{_iv(*s["pbt_vs_constant"]["interval"])}</span></td></tr>'
        for s in o['seeds'])
    rows += (f'<tr class="mean"><td>mean</td><td></td><td></td><td></td><td><b>{_ch(o["mean_change"])}</b>'
             f'<span class="iv">{_iv(*o["mean_interval"])}</span></td><td>not computed</td></tr>')
    return ('<div class="table-wrap"><table class="cmp"><caption><a class="tref" href="#t18">#18</a> per seed: Test (2.8M) '
            'mistag (%) of the selected model of each method, and PBT against each comparator</caption><thead><tr>'
            '<th>Seed</th><th class="num">Standard ParT</th><th class="num">Tuned constant LR</th><th class="num">PBT</th>'
            '<th>PBT vs standard ParT</th><th>PBT vs tuned constant LR</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div><p class="tnote">Below each change: 95% Test-sample interval. It does '
            'not include seed-to-seed variation, which the three rows show. PBT is numerically better than the tuned '
            'constant LR on every seed, but by 0.03% on seed 22345: not an established, practically meaningful '
            'advantage.</p>')


def table_28(d):
    """#27/#28 per seed: provenance (trial, training epoch) and Test value of both models, the ensemble and its change."""
    o = d['overview']['ensemble']
    src = {'sel1m_train8m_standard_part_3e-3': '#24', 'sel1m_train8m_standard_part': '#22'}

    def model(m, value):
        where = src.get(m['run'], f'#28 run')
        return f'<td>{where}, epoch {m["epoch"] - 17}</td><td class="num">{value:.4f}</td>'
    rows = ''.join(
        f'<tr><td>{s["seed"]} <a class="tref" href="#t{s["trial"]}">#{s["trial"]}</a></td>{model(s["model_3e3"], s["better"])}'
        f'{model(s["model_1e3"], s["other"])}<td class="num"><b>{s["ensemble"]:.4f}</b></td><td>{_ch(s["change"])}'
        f'<span class="iv">{_iv(*s["interval"])}</span></td></tr>' for s in o['seeds'])
    rows += (f'<tr class="mean"><td>mean</td><td></td><td></td><td></td><td></td><td></td><td><b>{_ch(o["mean_change"])}'
             f'</b><span class="iv">{_iv(*o["mean_interval"])}</span></td></tr>')
    return ('<div class="table-wrap"><table class="cmp"><caption><a class="tref" href="#t27">#27</a> and '
            '<a class="tref" href="#t28">#28</a> per seed: the two standard ParT models, their average and its change '
            'against the better model; Test (2.8M) mistag (%)</caption><thead><tr><th rowspan="2">Seed</th>'
            f'<th colspan="2">Start LR {SUP("3", 3)} (better model)</th><th colspan="2">Start LR {SUP("1", 3)}</th>'
            '<th rowspan="2" class="num">Ensemble</th><th rowspan="2">Ensemble vs better model</th></tr><tr>'
            '<th>model, epoch</th><th class="num">Test</th><th>model, epoch</th><th class="num">Test</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div><p class="tnote">Epoch: training epoch of the checkpoint with the best '
            'Validation (1M) value of that model. The 3×10<sup>−3</sup> model was the better one on Validation (the '
            'preregistered reference) and on Test, on every seed. For seeds 22345 and 32345 the two models are '
            'independently trained members of one #28 run, which trains only these two start LRs side by side. Run names, members and checkpoint files are in the '
            '<a href="log.html#t28">experiment log</a>.</p>')


# ---------------------------------------------------------------------------------------------------------------
# Rendering (layout approved 2026-11-28, "P4"): a wide grid with a 780 px figure/table column and the explanation
# beside it; prose in a 720 px column. Text: report_text.py. Anchors kept from earlier versions: summary, timeline,
# setup, setup-definitions, conclusions, phase-0 … phase-5, t1 … t33, and fig-1 … fig-4 on the figures that carry the
# same content as before (fig-1 research overview → timeline, fig-2 PBT learning rate, fig-3 loss curves, fig-4
# model-pair analysis); new figures have named anchors (fig-pbt, fig-baseline, fig-changes, fig-ensemble).
# ---------------------------------------------------------------------------------------------------------------
import html as _html  # noqa: E402



def _figure_numbers():
    """Figure numbers in reading order: the timeline, then every figure of the chapters (supporting ones included)."""
    order = ['timeline'] + [b[1] if b[0] == 'fig' else b[2] for ch in X.CHAPTERS for b in ch['body']
                            if b[0] in ('fig', 'analysis')]
    return {k: n for n, k in enumerate(order, 1)}


FIGNO = _figure_numbers()


def text(s):
    """Report prose: figure numbers filled in, trial numbers linked to their rows on this page."""
    return xref(re.sub(r'\{fig:(\w+)\}', lambda m: str(FIGNO[m.group(1)]), s))


def figure(key, cls='fr'):
    name, fid, title, cap, *below = X.FIGURES[key]  # below: optional HTML under the image (figure column)
    n = FIGNO[key]
    return (f'<figure class="{cls}" id="{fid}"><div><img src="{asset(name)}" alt="Figure {n}: {_html.escape(title)}">'
            f'{text(below[0]) if below else ""}</div><figcaption><b data-anchor="{fid}">Figure {n}. {title}</b>{text(cap)}</figcaption></figure>')


def side(main, note, cls=''):
    return f'<div class="fr tr{cls}"><div class="scroll">{main}</div><div class="cap">{note}</div></div>'


def primary_table(ch, note):
    rows = ''.join(
        f'<tr id="t{n}"><td class="n" data-anchor="t{n}"><a href="log.html#t{n}" title="Runs and configuration files of '
        f'trial {n} in the experiment log">{n}</a></td><td class="d">{X.ROWS[n][0]}</td><td>{X.ROWS[n][1]}</td>'
        f'<td class="v">{X.ROWS[n][2]}</td><td>{xref(X.ROWS[n][3])}</td></tr>' for n in ch['primary'])
    table = ('<table class="trials"><thead><tr><th class="n">#</th><th class="d">Date</th><th>Experiment</th>'
             f'<th class="v">Test</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table>')
    return side(table, f'<b>Primary experiments</b>{text(note)}')


def seed_table(html, note, ids):
    """A per-seed table of table_18/table_28 with its note beside it; trial anchors on its rows."""
    html = re.sub(r'<p class="tnote">.*?</p>', '', html, flags=re.S).replace('<div class="table-wrap">', '').replace('</table></div>', '</table>')
    for seed, tid in ids.items():
        html = html.replace(f'<tr><td>{seed} ', f'<tr id="{tid}"><td>{seed} ', 1)
    return side(html, text(note), ' seed')


def supporting(ch):
    if not ch['support']:
        return ''
    items = []
    for n in ch['support']:
        name, change, value, sample, versus = CARD[n]
        items.append(f'<li id="t{n}"><a href="log.html#t{n}" data-anchor="t{n}">#{n}</a> <b>{name}.</b> {xref(change)}: '
                     f'{value}{" (" + sample + ")" if sample else ""}{", " + xref(versus) if versus else ""}.</li>')
    return (f'<details class="more"><summary>Supporting experiments ({len(items)})</summary><ul>{"".join(items)}</ul>'
            '</details>')


def chapter(k, ch, nxt, figdata):
    p = ch['phase']
    kick = ' <span class="dot">·</span> '.join([ch['when'], f'Trials {ch["trials"]}'] + ch['meta'].split(' · '))
    out = [f'<section class="ch" id="phase-{p}"><p class="kick"><span class="no">{k}</span>{kick}</p>',
           f'<h2 data-anchor="phase-{p}">{ch["q"]}</h2><p class="ans">{text(ch["a"])}</p>']
    first = True
    for block in ch['body']:
        kind = block[0]
        if kind == 'fig':
            out.append(figure(block[1]))
        elif kind == 'p':
            out.append(f'<div class="prose{" first" if first else ""}"><p>{text(block[1])}</p></div>')
        elif kind == 'note':
            out.append(f'<p class="note" id="{block[1]}">{text(block[2])}</p>')
        elif kind == 'table':
            out.append(primary_table(ch, block[1]))
        elif kind == 'seed18':
            out.append(f'<div id="t18">{seed_table(table_18(figdata), block[1], {})}</div>')
        elif kind == 'seed28':
            out.append(seed_table(table_28(figdata), block[1], {'12345': 't27', '22345': 't28'}))
        elif kind == 'analysis':
            out.append(f'<details class="analysis"><summary>{block[1]}</summary>{figure(block[2], "wide")}'
                       f'{block[3]}</details>')
        first = kind not in ('p', 'note')
    out.append(supporting(ch))
    nx = PHASE_SUMMARY[p][3]
    if nx:
        out.append(f'<p class="next"><b>Next decision</b>{nx}'
                   + (f'<a href="#phase-{nxt["phase"]}">→ {nxt["q"]}</a>' if nxt else '') + '</p>')
    out.append('</section>')
    return ''.join(out)


def body():
    import json
    figdata = json.loads(FIGDATA.read_text())
    meta = ''.join(f'<div><b>{k}</b>{v}</div>' for k, v in X.META)
    findings = ''.join(f'<li><b>{c}</b> {s} <span class="ev">({e})</span></li>' for c, s, e in X.FINDINGS)
    toc = ''.join(f'<a href="#phase-{ch["phase"]}">{k}. {ch["q"]}<span>{ch["when"]}</span></a>'
                  for k, ch in enumerate(X.CHAPTERS, 1))
    out = [f'<header><h1>{X.TITLE}</h1><p class="dek">{X.DEK}</p></header><div class="byline">{meta}</div>',
           f'<div class="top" id="summary"><div><p class="lead">{X.QUESTION}</p><ol class="findings">{findings}</ol>'
           '<p class="fnote">“Better” and “worse” are relative changes of the mistag on Test (2.8M) unless stated; lower '
           'mistag is better. Samples, metric, budget and intervals: <a href="#setup-definitions">setup and definitions</a>.</p></div><nav class="toc" aria-label="Contents"><div class="h">The study in five questions'
           f'</div>{toc}<a href="#conclusions">Conclusions and limits</a><a href="#setup-definitions">Setup and '
           'definitions</a></nav></div>',
           f'<div class="timeline" id="timeline">{figure("timeline", "wide")}</div>']
    for k, ch in enumerate(X.CHAPTERS, 1):
        out.append(chapter(k, ch, X.CHAPTERS[k] if k < len(X.CHAPTERS) else None, figdata))
    limits = ''.join(f'<li>{text(x)}</li>' for x in X.LIMITS)
    setup = ''.join(f'<div><h3>{lab}</h3><p>{text(v)}</p></div>' for lab, v in X.SETUP)
    out.append(f'<section class="end" id="conclusions"><h2 data-anchor="conclusions">Conclusions</h2><div class="prose">'
               f'<p>{X.CONCLUSION}</p></div><h3 class="lim">Limits</h3><ul class="limits">{limits}</ul></section>'
               f'<section class="end" id="setup-definitions"><h2 data-anchor="setup-definitions">Setup and definitions'
               f'</h2><div class="defs">{setup}</div></section>')
    return '\n'.join(out)


def main():
    page = Path(__file__).with_name('report_template.html').read_text().replace('<!--BODY-->', body())
    page = page.replace('pub/anchors.js', asset('anchors.js'))
    out = ROOT / 'docs/_extra/overview.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


if __name__ == '__main__':
    main()
