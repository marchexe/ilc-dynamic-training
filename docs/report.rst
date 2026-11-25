.. _report:

Report: All Trials and Results
==============================

Status on 2026-11-26. This page lists every trial of the project, in time order, with its
configuration and a short result. The last section gives an interpretation of the outcome.
No run is in progress.

.. _report-terms:

Terms
-----

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Term
     - Meaning
   * - Training (800k)
     - 2.4M jets (source entries below 800k of each flavour file). Used until 2026-11-13.
   * - Training (8M)
     - 24M jets (source entries below 8M). From 2026-11-13 without the Validation (1M) rows.
   * - Validation (150k)
     - 150,000 jets. Until 2026-11-09 it ranks the members, drives the PBT decisions and selects
       the checkpoint.
   * - Validation (1M)
     - 999,999 jets from the training files, not used for training. It has the same role as
       Validation (150k), from 2026-11-10. We changed to it because 150k was too small to select
       checkpoints reliably (phase 3).
   * - Test (2.8M)
     - 2,825,125 jets. Each selected checkpoint is evaluated on it once. It never selects a
       model. Before this report it was called "holdout".
   * - Test (9.0M)
     - 8,970,477 jets. Used one time only, for trial 10. Before this report it was called
       "sealed test".
   * - Composite mistag
     - The metric of all tables (%). It is the geometric mean of the background mistag rates at
       eight working points (b-tag at 80% and 90%, c-tag at 50% and 80%, each against two
       backgrounds). Lower is better.
   * - Member
     - One model of a run. Each member trains on one GPU. Most runs have five members.
   * - Standard ParT
     - The learning-rate (LR) schedule of the pretrained model: a constant LR for 35 epochs,
       then a decay by ×0.01 over the last 15 epochs. One run trains five members with different
       start LRs; the best member on the validation set is the result.
   * - PBT
     - Population Based Training. At each decision, a weak member copies the weights of a
       strong member and changes its LR (for example ×0.5 or ×2).
   * - Change (%)
     - (A − B) / B, where B is the comparator. Negative means that A is better.

All trials start from the same pretrained checkpoint (epoch 17) and train for 50 epochs, unless
the table gives a different number. Change (%) and the intervals are on Test (2.8M) unless the
table says "Validation". An interval is a 95% interval for the Test sample only; it does not
include seed-to-seed variation.

.. _report-trials:

List of trials
--------------

Phase 0: set-up, 2026-07-22 to 08-31
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 5 14 40 41

   * - #
     - Dates
     - Configuration
     - Result
   * - 0
     - 07-22 – 08-31
     - First PBT versions (exploit-and-mutate, anchor-copy), a bandit LR controller, controller
       on/off pairs, and a study of LR against mistag. Short epochs (samples per epoch limited),
       damped optimizer start.
     - Set-up only. We found and fixed two defects: BatchNorm was not frozen, and a controller
       overwrote the PBT LR. The training contract changed after this phase, so no result of
       this phase is comparable with the later trials.

Phase 1: PBT against fixed LRs, Training (800k), Validation (150k), 2026-09-16 to 10-01
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 5 14 40 41

   * - #
     - Dates
     - Configuration
     - Result (Test 2.8M)
   * - 1
     - 09-15 – 09-16
     - Fixed-LR grid: five members, constant LRs 3×10⁻⁶ – 1.4×10⁻⁵.
     - 0.3269. Best member: the highest LR.
   * - 2
     - 09-16 – 09-17
     - Two fixed-LR members (1.4×10⁻⁵, 8.5×10⁻⁶) continued to 100 epochs.
     - 0.3156 (1.4×10⁻⁵).
   * - 3
     - 09-17 – 09-18
     - Windowed PBT: start as trial 1, decisions every 5 epochs, ×0.8 or ×1.2, upper limit
       3×10⁻⁵.
     - 0.3225; −1.36% against trial 1.
   * - 4
     - 09-18 – 09-19
     - Windowed PBT, 100 epochs.
     - 0.3111; −1.44% against trial 2.
   * - 5
     - 09-19 – 09-21
     - Smaller validation subset (60k) and an LR controller in observe-only mode.
     - The subset ranks checkpoints like the full set (rank correlation 0.96). The controller
       proposed almost no changes. Not used further.
   * - 6
     - 09-22 – 09-28
     - Cadenced PBT: decisions every epoch against every five epochs; two seeds each.
     - Every epoch is better on Validation, but worse on Test (+0.47% and +0.13%).
       Inconclusive.
   * - 7
     - 09-28
     - Restart test: one epoch in five restarted parts.
     - Result changes, because the Lookahead state of the optimizer is not saved. Sub-epoch PBT
       is not possible with this contract (see trial 11).
   * - —
     - 10-01
     - Interim report.
     -

Phase 2: Test (2.8M) and the large-data comparison, 2026-10-12 to 11-03
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 5 14 40 41

   * - #
     - Dates
     - Configuration
     - Result
   * - 8
     - 10-12
     - First evaluation of the selected checkpoints of trials 1–6 on Test (2.8M).
     - Test values follow Validation for large differences (rank correlation 0.70). Differences
       of about 0.3% reorder.
   * - 9
     - 10-13
     - Preparation of Training (8M) and of Test (9.0M); one-epoch GPU check.
     - Ready.
   * - 10
     - 10-13 – 11-02
     - Large-data comparison, Training (8M), preregistered: fixed-LR grid as trial 1 (arm A)
       against windowed PBT as trial 3 (arm B), one seed.
     - Test (9.0M): **−7.67%** for PBT. All ten validity checks passed. Later found to be a
       comparison against LRs that are too low (trial 13).
   * - 11
     - 10-17 – 11-03
     - Full optimizer state in checkpoints, for sub-epoch PBT.
     - Implemented. GPU check: one epoch equals five restarted parts, bit for bit. No PBT
       experiment with it.

Phase 3: higher LRs, Training (800k), Validation (150k), 2026-11-02 to 11-09
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trial 10 left one question open: the best fixed member was the highest LR of the grid, and PBT
ended above the grid. These trials ask whether a higher fixed LR gives the same gain.

.. list-table::
   :header-rows: 1
   :widths: 5 14 40 41

   * - #
     - Dates
     - Configuration
     - Result (Test 2.8M)
   * - 12
     - 11-02 – 11-03
     - Fixed-LR grid 1.4×10⁻⁵ – 3×10⁻⁵.
     - 0.3146. Best member: the highest LR again.
   * - 13
     - 11-03
     - Fixed-LR grid 3×10⁻⁵ – 8×10⁻⁵.
     - 0.3018. Better than all PBT runs of phase 1.
   * - 14
     - 11-03 – 11-04
     - Fixed-LR grid 8×10⁻⁵ – 3.2×10⁻⁴.
     - 0.3033. Best member 1.2×10⁻⁴.
   * - 15
     - 11-04
     - Fast windowed PBT: start 1×10⁻⁵ – 3.5×10⁻⁴, decisions every 2 epochs, ×0.5 or ×2,
       upper limit 5×10⁻⁴.
     - 0.3003; −1.00% against trial 14.
   * - 16
     - 11-08 – 11-09
     - Trials 15 and 14 again with seed 22345 (grid 6×10⁻⁵ – 1.6×10⁻⁴).
     - PBT +0.47% (worse), although 1.4% better on Validation (150k).
   * - 17
     - 11-09
     - Trial 15 with fallback copies (×0.7 or ×1.4 when ×0.5 and ×2 are both in use).
     - +0.78% against trial 15.
   * - —
     - 11-09
     - Conclusion of phase 3.
     - With Validation (150k), the selection noise (about 0.5%) is as large as the effect.
       Change to Validation (1M), and compare with standard ParT.

Phase 4: PBT against standard ParT, Training (800k), Validation (1M), 2026-11-09 to 11-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 5 14 40 41

   * - #
     - Dates
     - Configuration
     - Result (Test 2.8M)
   * - 18
     - 11-09 – 11-13
     - Three seeds (12345, 22345, 32345), three runs each: standard ParT (start LRs 1×10⁻⁵ –
       1×10⁻³), fixed-LR grid (6×10⁻⁵ – 1.6×10⁻⁴), fast PBT with fallback copies.
     - **PBT −0.81% against standard ParT** (−0.67%, −0.85%, −0.91%; interval −1.34 to −0.34).
       PBT is better on every seed. The fixed grid equals PBT on one seed and standard ParT on
       two.
   * - 19
     - 11-11 – 11-12
     - Fast PBT with a final decay (no decisions in the last 15 epochs), two seeds.
     - Same checkpoints as trial 18. No gain.
   * - 20
     - 11-12
     - Standard ParT from random weights (no pretraining).
     - 0.378 on Validation, against about 0.30 from the pretrained checkpoint. Not continued.
   * - 21
     - 11-12
     - Weight averages of nearby checkpoints and of members.
     - No gain (Validation only).

Phase 5: Training (8M), Validation (1M), 2026-11-13 to 11-25
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All trials of this phase have seed 12345 unless the table gives a different seed.

.. list-table::
   :header-rows: 1
   :widths: 5 14 40 41

   * - #
     - Dates
     - Configuration
     - Result (Test 2.8M)
   * - 22
     - 11-13 – 11-15
     - Standard ParT, start LRs 1×10⁻⁵ – 1×10⁻³.
     - 0.1635. Best member: the highest LR (1×10⁻³).
   * - 23
     - 11-15 – 11-16
     - Fast PBT as trial 18, with the LR range of standard ParT (upper limit 1×10⁻³).
     - 0.1812. **+12.4% (worse)** against trial 24.
   * - 24
     - 11-18 – 11-20
     - Standard ParT, start LR 3×10⁻³ (one member).
     - **0.1611**, the 8M reference. 1.4% better than trial 22.
   * - 25
     - 11-19 – 11-21
     - PBT with cool-down probes: at each decision, a copy of each member trains a quarter epoch
       at 0.1× its LR, and the decision uses these copies.
     - 0.1699; +5.4% against trial 24. At 800k: no change.
   * - 26
     - 11-20 – 11-22
     - Trial 25 with a final decay and the LR range up to 3×10⁻³.
     - 0.1655; +2.7% against trial 24. The two seeds planned for this case were not started,
       by the rule fixed before the run.
   * - 27
     - 11-23
     - Ensemble: mean of the class probabilities of trials 22 and 24 (two models). Selected
       among eight ensembles on Validation, then fixed before the Test evaluation.
     - 0.1530; −5.05% against trial 24.
   * - 28
     - 11-23 – 11-25
     - Trial 27 repeated with seeds 22345 and 32345 (new standard ParT runs, start LRs 3×10⁻³
       and 1×10⁻³). Rules fixed before the runs.
     - −4.86% and −5.14%. **Mean of three seeds −5.01%** (interval −5.84 to −4.17).
   * - 29
     - 11-23
     - Same ensemble check on Training (800k), three seeds, fifteen models.
     - No gain (Validation only). The models are too similar.
   * - 30
     - 11-23
     - Why the ensemble gains: pairs with the same LR and different seeds, against pairs with
       different LRs.
     - Same LR: −3.56%; different LRs: −4.65% (Validation). Most of the gain comes from two
       separately trained models; a different LR adds about 1 point. Two members of one PBT run
       give no gain, because PBT copies weights between them.
   * - 31
     - 11-23 – 11-24
     - Final decay of trial 24 (epochs 53–67) with dropout 0.
     - Much worse (0.1730 against 0.1595 at epoch 61, Validation). Stopped.
   * - 32
     - 11-24
     - Distillation: final decay of trial 24, trained half on the true labels and half on the
       ensemble probabilities of trial 27.
     - Best 0.1600 against 0.1584 without it (Validation). No gain; the full run was not
       started.
   * - 33
     - 11-23 – 11-25
     - Standard ParT, start LR 1×10⁻² (one member).
     - Worse on Validation (0.1631), 0.76% better on Test (0.1599). Above 1×10⁻³ the start LR
       has almost no effect.

.. _report-corrections:

Statements that changed
-----------------------

* **1.3–1.7% (phase 1) and 7.7% (trial 10) for PBT.** These gains are against fixed LRs that
  were too low. Against good fixed LRs (trial 13) the gain is about 0.
* **10.8% worse at 8M (first value of trial 23).** It is 12.4% against the better standard ParT
  member of trial 24.
* **"Holdout" and "sealed test".** These are now Test (2.8M) and Test (9.0M). File and directory
  names in the repository keep the old words, because frozen records contain them.

.. _report-interpretation:

Interpretation
--------------

1. **PBT gives a small gain only with small training data.** At 800k jets PBT is 0.8% better
   than standard ParT on three seeds. The model overfits at 800k (the training loss falls, the
   validation loss stops), so a lower LR at the correct time helps, and PBT finds this time.
2. **PBT is worse with large training data.** At 8M jets the model does not overfit. The best
   schedule is a long phase at a high LR and then a decay. PBT compares members two epochs after
   each decision. Over two epochs a lower LR almost always looks better, so PBT lowers the LR
   too early. The cool-down probes and the final decay reduce the loss from 12.4% to 2.7%, but
   PBT is still worse than standard ParT.
3. **The earlier positive results came from weak comparators.** The gains of phases 1 and 2 are
   mostly the gain from a higher LR, not from the adaptation.
4. **The usable result is the ensemble, not PBT.** At 8M, the mean of two standard ParT models
   of one LR scan is 5.0% better than the better model, on three seeds. Standard ParT trains
   these models anyway; the cost is two models at inference time. A single model that keeps
   this gain was not found (distillation, weight averaging: no gain). At 800k the ensemble
   gives no gain.
5. **Limits.** The 8M PBT trials have one seed. All trials start from one pretrained checkpoint
   and use one optimizer contract. Test (2.8M) was also used in diagnostics before trial 18, so
   it is not fully unseen. No result was propagated to a physics analysis.

.. _report-open:

Open items
----------

* Sub-epoch PBT (trial 11) is implemented but was not run.
* The effect of the decision frequency (trial 6) is inconclusive.
* Possible next steps, not started: PBT decisions that look further ahead than two epochs; more
  than two models in the ensemble; other pretrained start checkpoints.
