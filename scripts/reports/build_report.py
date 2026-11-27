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
from report_cards import BASE, CARD, KEY, PHASE_SUMMARY  # noqa: E402

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
REF = 'Reference: '
V = ' <i>(Validation)</i>'

# (phase title, conditions, rows); row: (trial, date, configuration, result, interpretation)
# Results are mistag (%) of the selected model on Test (2.8M) unless marked (Validation); "x% better/worse than #n" is
# the relative change of that value against trial #n.
PHASES = [
    ('Initial implementation', 'Training (800k)', [
        ('–', 'Jul 22 – Aug 31', 'First PBT implementations (exploit-and-mutate, anchor-copy) and LR controllers, with an '
         'older training set-up: a capped number of training jets per epoch and a damped optimizer start',
         'Two defects found and fixed: BatchNorm statistics were updated although they should be frozen, and an LR '
         'controller overwrote the LR set by PBT',
         'Results not comparable with later trials and not used. Training was rebuilt as deterministic full epochs '
         '(from Sep 16), which all later trials use'),
    ]),
    ('PBT against constant learning rates', 'Training (800k), then (8M) · Validation (150k) · seed 12345 unless noted', [
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
    ('Higher constant learning rates', 'Training (800k) · Validation (150k) · seed 12345 unless noted', [
        ('12', 'Nov 3', f'Constant-LR grid {E("1.4", 5)}, {E("1.8", 5)}, {E("2.2", 5)}, {E("2.6", 5)}, {E("3", 5)}, '
         'because the best LR of #1 and #10 was the top of the grid',
         f'0.3146; selected LR {E("3", 5)}, again the top of the grid', 'The optimum lies higher still'),
        ('13', 'Nov 3–4', f'Constant-LR grid {E("3", 5)}, {E("4", 5)}, {E("5", 5)}, {E("6", 5)}, {E("8", 5)}',
         f'0.3018; selected LR {E("8", 5)}, again the top of the grid (Validation 0.3166)',
         'Better than every PBT run of #3–#6: the early PBT gains came from the LR level. Grid extended once more'),
        ('14', 'Nov 4', f'Constant-LR grid {E("8", 5)}, {E("1.2", 4)}, {E("1.6", 4)}, {E("2.4", 4)}, {E("3.2", 4)}',
         f'0.3033; selected LR {E("1.2", 4)}, inside the grid (Validation 0.3141, better than #13)',
         f'Optimum near {E("8", 5)}–{E("1.2", 4)}. #14 is better than #13 on Validation but worse on Test'),
        ('15', 'Nov 4', FAST,
         '0.3003: 1.0% better than #14 (the preregistered comparator: of #13 and #14, the one with the better Validation '
         'mistag) and 0.5% better than #13 (the better one on Test)',
         'PBT about equal to the best constant LR; the order depends on which constant-LR model is the reference'),
        ('16', 'Nov 9', f'Second seed (22345): fast PBT exactly as #15, and a new constant-LR grid {E("6", 5)}, {E("8", 5)}, '
         f'{E("1", 4)}, {E("1.2", 4)}, {E("1.6", 4)}, centred on the optimum of #13–#14 (not the grid of #14)',
         'PBT 0.3041 against constant LR 0.3027: 0.47% worse on Test, although 1.4% better on Validation',
         'The order reverses with the seed, and Validation and Test disagree: selection noise of about 0.5% is as large '
         'as the effect'),
        ('17', 'Nov 9', 'Fast PBT as #15 (seed 12345) with fallback copies. In #15 a losing member was not replaced when '
         'both ×0.5 and ×2 gave an LR another member already had (about 24 times; one member stayed at 5×10<sup>−4</sup> '
         'with the worst mistag from epoch 5 to 36). With fallback copies it still copies the best member and uses ×0.7 '
         'or ×1.4, whichever LR is farthest from the others',
         '0.3026: 0.78% worse than #15 on Test; also worse on Validation (0.3166 against 0.3156)',
         'No improvement; kept in later PBT runs because it removes the stuck member. Next: Validation (1M), standard '
         'ParT as comparator, three seeds'),
    ]),
    ('PBT against standard ParT, 800k', 'Training (800k) · Validation (1M) · 3 seeds', [
        ('18', 'Nov 10–13', f'Three methods, each with seeds 12345, 22345, 32345: standard ParT (constant LR for 35 epochs, '
         f'then ×0.01 over 15; start LRs {E("1", 5)}, {E("3", 5)}, {E("1", 4)}, {E("3", 4)}, {E("1", 3)}); constant-LR '
         f'grid as #16 ({E("6", 5)}–{E("1.6", 4)}); PBT as #17. Comparisons fixed before training',
         'Standard ParT 0.3029 / 0.3038 / 0.3024 (always start LR 10<sup>−4</sup>); constant LR 0.3028 / 0.3013 / 0.3024; '
         'PBT 0.3008 / 0.3012 / 0.2996. PBT better than standard ParT by 0.67%, 0.85%, 0.91%: <b>mean 0.81%</b>, Test '
         'interval 0.34–1.34%; better at 7 of 8 working points',
         'A small PBT gain on every seed. The constant-LR grid equals PBT for seed 22345 and standard ParT for the other two'),
        ('19', 'Nov 12', 'PBT as #18 with the final decay of standard ParT: no decisions in the last 15 epochs, all LRs '
         '×0.01 over them; seeds 12345 and 22345',
         'The same models are selected as in #18 (epochs 31 and 33, before the decay starts), so the same Test values',
         'The decay does not change the result with Training (800k)'),
        ('20', 'Nov 12', f'Standard ParT from random weights instead of the pretrained model, start LRs '
         f'{E("2.5", 4)}–{E("4", 3)}, 50 epochs',
         f'0.378{V} (start LR {E("4", 3)}, epoch 14), against about 0.30 from the pretrained model; not evaluated on Test',
         'Pretraining is essential at this training budget'),
        ('21', 'Nov 12', 'Weight averages of the selected checkpoint with nearby epochs (3–10 epochs) and of population '
         'members, for the six runs of seeds 12345 and 22345 in #18',
         f'Standard ParT: averages 0.09% and 0.30% better than the selected checkpoint{V}; constant LR and PBT: no '
         'average better than the selected checkpoint. The planned Test evaluation was not run',
         'At most a small change on Validation; not pursued'),
    ]),
    ('PBT against standard ParT, 8M', 'Training (8M) · Validation (1M) · seed 12345', [
        ('22', 'Nov 14–15', f'Standard ParT as #18, Training (8M), start LRs {E("1", 5)}–{E("1", 3)}',
         f'0.1635; selected start LR {E("1", 3)}, the top of the range', 'The range was extended in #24'),
        ('23', 'Nov 15–17', f'PBT as #18 with the start LRs and upper LR bound of standard ParT ({E("1", 5)}–{E("1", 3)})',
         '0.1812: 12.4% worse than #24 (10.8% worse than #22, the first reported value)',
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
    ('Ensembles and other ways to gain, 8M', 'Training (8M) · Validation (1M)', [
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
         'No gain beyond what choosing among 105 pairs gives by chance'),
        ('30', 'Nov 23–25', 'Disagreement (share of jets with a different predicted class) and ensemble change for pairs '
         'of 8M models, on Validation: all pairs of nine seed-12345 models (#22–#26), and the 15 pairs of the '
         f'{E("3", 3)} and {E("1", 3)} standard ParT models of the three seeds (#22, #24, #28)',
         'Two checkpoints of one run or two models of one PBT run: 0.22–0.40%, ensemble 0.04–0.81% worse. Separately '
         f'trained standard ParT models: 1.49–2.14%; the 15 three-seed pairs gain 3.6% on average with the same start LR and '
         f'4.7% with different start LRs. Pairs with a PBT model of another run: 1.02–2.32%, change from 2.0% worse to '
         f'4.7% better{V}',
         'Separately trained standard ParT models gain; disagreement alone does not predict the gain (pairs with PBT '
         'models disagree as much but often do not gain). Figure 4 shows the three-seed pairs only'),
        ('31', 'Nov 23–24', 'The final decay of #24 (epochs 36–50) repeated from its epoch-35 checkpoint with dropout 0',
         f'0.1730 against 0.1595 with dropout at epoch 44{V}; stopped at epoch 45 when the process was lost on the node and '
         'not restarted', 'Dropout helps with Training (8M) although there is no visible overfitting'),
        ('32', 'Nov 24', 'Distillation: the final decay of #24 repeated from epoch 35 with a loss of half the usual '
         'cross-entropy and half the cross-entropy against the ensemble probabilities of #27',
         f'Best 0.1600 against 0.1584 without distillation{V}; 0.1604 against 0.1597 at the last epoch. The full run was '
         'not started', 'The ensemble gain was not transferred to one model'),
        ('33', 'Nov 24–25', f'Standard ParT, one model at start LR {E("1", 2)}, added under a rule fixed before it started',
         f'0.1631{V}, worse than #24 (0.1584), so #24 stays the reference; 0.1599 on Test',
         f'The Test value lies within the seed spread of {E("3", 3)} (0.1597–0.1623): above {E("1", 3)} the start LR hardly '
         'matters'),
    ]),
]


PUB = ROOT / 'docs/_extra/pub'


def asset(name):
    """pub/<name>?v=<content hash>: a changed file gets a new URL, so no browser serves a stale copy."""
    import hashlib
    return f'pub/{name}?v={hashlib.sha256((PUB / name).read_bytes()).hexdigest()[:12]}'


FIG = ('<figure id="fig-{n}"><div class="fig-scroll"><img src="{src}" alt="{alt}"></div>'
       '<figcaption data-anchor="fig-{n}"><b>Figure {n}.</b> {cap}</figcaption></figure>')


def xref(html):
    """Link trial references such as #18 to their row on this page."""
    return re.sub(r'(?<![&\w])#(\d+)', r'<a class="tref" href="#t\1">#\1</a>', html)


def trial_row(n, d, c, r, i):
    """Name, key change and headline result first; the full configuration and result under 'Details'."""
    name, change, value, sample, versus = CARD[n]
    cls = ' class="key"' if n in KEY else ' class="base"' if n in BASE else ''
    smp = f' <span class="smp">{sample}</span>' if sample else ''
    cfg = (f'<div class="nm">{name}</div><div class="chg">{xref(change)}</div>'
           f'<details class="more"><summary>Details</summary><p><b>Configuration.</b> {xref(c)}</p>'
           f'<p><b>Result.</b> {xref(r)}</p></details>')
    res = f'<div class="val"><b>{value}</b>{smp}</div><div class="vs">{xref(versus)}</div>'
    if n == '–':  # the initial implementation period has no trial number
        return f'<tr id="setup"{cls}><td class="n"></td><td class="d">{d}</td><td>{cfg}</td><td>{res}</td><td class="i">{i}</td></tr>'
    return (f'<tr id="t{n}"{cls}><td class="n" data-anchor="t{n}"><a href="log.html#t{n}" '
            f'title="Runs and configuration files of trial {n} in the experiment log">{n}</a></td>'
            f'<td class="d">{d}</td><td>{cfg}</td><td>{res}</td><td class="i">{xref(i)}</td></tr>')


SUP = lambda m, e: lr(m, e)

# per phase: (dates, objective, learned-and-next paragraphs (HTML), figures)
PHASE_TEXT = [  # (dates, objective, outcome, interpretation and reason for the next phase, figures)
    ('Jul 22 – Aug 31', 'Build the training and PBT tooling.', 'No usable comparison.',
     '<b>Interpretation.</b> The defects and the set-up made these runs neither reliable nor comparable. '
     '<b>Next:</b> rebuild training as deterministic full epochs, so that runs are reproducible and comparable.',
     []),
    ('Sep 16 – Nov 3', 'Compare PBT with constant LRs, on Training (800k) and in a preregistered test on Training (8M).',
     'PBT ahead of the constant LRs on both training samples.',
     '<b>Interpretation.</b> In every comparison the best constant LR was at the top of its grid and PBT ended above it, so '
     'the gain could come from the LR level rather than from adapting the LR; Validation (150k) did not resolve differences '
     'of about 0.5%. <b>Next:</b> extend the constant-LR grid upwards to separate the two.',
     []),
    ('Nov 3 – Nov 9', 'Test whether a higher constant LR explains the PBT gain.',
     f'A constant LR near {SUP("1", 4)} matches PBT.',
     '<b>Interpretation</b> <span class="ev s">single seed per grid</span>. Both PBT runs lie on the curve of the constant '
     'LRs (Figure 1), so the earlier gains were an LR-level effect. The sign reversed with a second seed: at this precision '
     'the comparison needs a larger Validation sample and several seeds. <b>Next:</b> Validation (1M), standard ParT as '
     'comparator (the usual recipe: constant LR for 35 epochs, then ×0.01 over 15; five start LRs), three seeds.',
     [('fig3_learning_rate.svg', 'Mistag on Test against the learning rate of the selected model',
       'Mistag on Test (2.8M) against the LR of the selected model; Training (800k), seed 12345. Grey: best model of each '
       'constant-LR grid (#1, #12–#14). Red: PBT (#3, #15).')]),
    ('Nov 10 – Nov 13', 'Compare PBT with standard ParT under the new selection, Training (800k).',
     'PBT better on all three seeds, by less than 1%.',
     '<b>Interpretation</b> <span class="ev r">replicated, 3 seeds</span>. With Training (800k), PBT finds a slightly better '
     'schedule than standard ParT. The interval (0.34–1.34%) is the Test-sample uncertainty; the seed-to-seed spread is '
     '0.67–0.91%. <b>Next:</b> repeat with ten times more training data, to see whether the gain holds.',
     []),
    ('Nov 14 – Nov 23', 'Repeat the comparison on Training (8M) and try to correct PBT.',
     'PBT worse than standard ParT; the corrections reduce but do not remove the deficit.',
     '<b>Observations</b> <span class="ev s">single seed</span>. PBT lowers all LRs early and falls behind during the final '
     'decay of standard ParT (Figure 2). With Training (800k) the training loss falls below the Validation loss after about '
     '20 epochs; with Training (8M) it does not (Figure 3). <b>Hypothesis</b> <span class="ev h">not tested</span>. Deciding '
     'two epochs after each change favours lower LRs, which costs little when the model overfits and much when it does not. '
     '#26 changed the LR range and added the final decay at once, so its improvement over #25 is not assigned to either. '
     'The probe runs used 9–12% more training than standard ParT, and the reference #24 is a sixth standard ParT model. '
     '<b>Next:</b> no LR-schedule change beat standard ParT, so test a gain that does not depend on the schedule: '
     'combining separately trained models.',
     [('fig5_mechanism.svg', 'Learning rate and best Validation mistag per epoch, Training (8M)',
       'Training (8M), seed 12345. Left: LR per epoch of standard ParT (#24) and range of the five PBT models (#23). Right: '
       'best Validation (1M) mistag so far; dashed: PBT with probes, final decay and higher LR range (#26). Shaded: final LR decay.'),
      ('fig4_regimes.svg', 'Training and Validation loss per epoch, Training (800k) and Training (8M)',
       f'Loss per epoch of the best standard ParT model (start LR {SUP("1", 4)} at 800k, {SUP("3", 3)} at 8M) on the '
       'training data and on Validation (1M); one seed each.')]),
    ('Nov 23 – Nov 25', 'Test whether combining separately trained standard ParT models helps on Training (8M).',
     'The ensemble gain is replicated on two new seeds; no single-model alternative keeps it.',
     '<b>Observation.</b> Pairs from one run disagree on at most 0.4% of the jets and do not gain; separately trained '
     'standard ParT models disagree on 1.5–2.1% and gain (Figure 4); pairs involving PBT models of other runs disagree '
     'as much but gain between −2.0% and +4.7% (#30). <b>Hypothesis</b> '
     '<span class="ev h">not tested</span>. The gain needs models that make different errors; PBT copies weights between '
     'its models, and at 800k the low LRs keep all models close to the pretrained start. The ensemble needs two forward '
     'passes per jet; its inference cost was not measured. Interval 4.17–5.84%: Test-sample uncertainty; seed-to-seed '
     'spread 4.86–5.14%.',
     [('fig6_ensemble.svg', 'Disagreement and ensemble improvement for five kinds of model pairs on Validation (1M)',
       'Pairs of models evaluated on Validation (1M) (#29, #30). Left: share of jets on which the two models predict '
       'different classes. Right: mistag reduction of the two-model ensemble against the better model of the pair; '
       'grey rows show the measured range where no gain was observed. Rows: checkpoints at epochs 45 and 50 of standard ParT '
       f'{SUP("3", 3)}; three models of the PBT run with probes and final decay (#26); standard ParT {SUP("1", 4)} of three seeds '
       f'(800k); standard ParT models at {SUP("3", 3)} and {SUP("1", 3)} of three seeds, paired with the same or a '
       'different start LR. Bars span the pairs; black ticks mark the mean. The panels show an association across these '
       'pairs, not a causal effect.')]),
]


SUPPORTING = {  # trial -> supporting role; all other trials are primary (they carry the main line of the study)
    '5', '6', '7', '8', '9', '11', '17', '19', '20', '21', '29', '30', '31', '32', '33'}
SUPPORT_TITLE = {  # phase -> what its supporting trials investigated
    1: 'Diagnostics and preparation: LR controller in observe mode (#5), decision frequency of PBT (#6), restart within '
       'an epoch (#7, #11), first Test evaluation (#8), preparation of Training (8M) (#9)',
    2: 'PBT modification: fallback copies when no new LR is free (#17), used in all later PBT runs',
    3: 'Side checks at 800k: final LR decay in PBT (#19), training without pretraining (#20), weight averaging (#21)',
    5: 'Further checks at 8M: ensembles at 800k (#29), source of the ensemble gain (#30), dropout (#31), distillation '
       '(#32), start LR 10<sup>−2</sup> (#33)',
}
THEAD = ('<colgroup><col class="c-n"><col class="c-d"><col class="c-c"><col class="c-r"><col class="c-i"></colgroup>'
         '<thead><tr><th>#</th><th>Date</th><th>Configuration</th><th>Result</th><th>Interpretation</th></tr></thead>')


def timeline():
    out, nfig = [], 1
    for k, ((title, cond, rows), (dates, objective, outcome, learned, figs)) in enumerate(zip(PHASES, PHASE_TEXT)):
        main = ''.join(trial_row(*row) for row in rows if row[0] not in SUPPORTING)
        support = [row for row in rows if row[0] in SUPPORTING]
        figs_html = []
        for src, alt, cap in figs:
            figs_html.append(FIG.format(src=asset(src), alt=alt, n=nfig, cap=xref(cap)))
            nfig += 1
        extra = ''
        if support:
            extra = (f'<details class="supporting"><summary><span class="sup-label">Supporting experiments '
                     f'({len(support)})</span> {SUPPORT_TITLE[k]}</summary><div class="table-wrap"><table class="hist">'
                     f'{THEAD}<tbody>{"".join(trial_row(*row) for row in support)}</tbody></table></div></details>')
        why, changed, found, nxt = PHASE_SUMMARY[k]
        learned = re.sub(r'\s*<b>Next:</b>.*$', '', learned, flags=re.S)  # the reason to move on is in the summary
        out.append(f'<section class="phase" id="phase-{k}"><div class="when">Phase {k} · {dates} · {cond}</div>'
                   f'<h3 data-anchor="phase-{k}">{title}</h3><dl class="psum"><dt>Why</dt><dd>{why}</dd>'
                   f'<dt>Changed</dt><dd>{changed}</dd><dt>Found</dt><dd class="found">{found}</dd>'
                   f'<dt>Next</dt><dd>{nxt}</dd></dl>'
                   f'<div class="table-wrap"><table class="hist">{THEAD}<tbody>{main}</tbody></table></div>'
                   f'{extra}<p class="learned">{xref(learned)}</p>' + ''.join(figs_html) + '</section>')
    return '\n'.join(out)


def main():
    page = (Path(__file__).with_name('report_template.html')).read_text().replace('<!--TIMELINE-->', timeline()).replace('pub/style.css', asset('style.css')).replace('pub/anchors.js', asset('anchors.js'))
    out = ROOT / 'docs/_extra/overview.html'
    out.write_text(page)
    print(f'{out.relative_to(ROOT)}: {len(page) // 1024} kB')


if __name__ == '__main__':
    main()
