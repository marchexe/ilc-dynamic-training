.. _report:

Report: All Trials and Results
==============================

Status on 2026-11-26. No run is in progress.

In one minute
-------------

* **PBT helps only with little data.** On 800k training jets, PBT is **0.8% better** than
  standard ParT on three seeds.
* **PBT is worse with much data.** On 8M training jets, PBT is **12.4% worse**. Two fixes reduce
  this to **2.7% worse**.
* **Most early gains were not real.** Before November, PBT was compared with fixed learning
  rates (LRs) that were too low. Good fixed LRs give the same result as PBT.
* **The useful result is an ensemble.** On 8M jets, the mean of two standard ParT models is
  **5.0% better** than the better single model, on three seeds. PBT is not necessary for it.

.. figure:: ../published/results/report/results.png
   :alt: Change of the Test (2.8M) mistag for every comparison of the project
   :width: 100%

   Every comparison of the project. Left of zero: the first named method is better. Hatched:
   the comparison is not valid, because the comparator's LRs were too low.

.. figure:: ../published/results/report/progress.png
   :alt: Test (2.8M) value of every selected model, by date, for 800k and 8M training jets
   :width: 100%

   The best Test (2.8M) value by date. Each dot is one selected model. The large improvements
   came from higher LRs (800k, early November) and from the change to standard ParT and
   ensembles (8M), not from PBT.

.. figure:: ../published/results/report/timeline.png
   :alt: Time line of all trials, by phase, coloured by kind of run
   :width: 100%

   All trials by phase. The numbers are the trial numbers of the tables below.

.. _report-terms:

Words used
----------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Word
     - Meaning
   * - Training (800k), (8M)
     - Training data: 2.4M and 24M jets (source entries below 800k and 8M of each file).
   * - Validation (150k), (1M)
     - Data evaluated after each epoch. It selects the best model and drives PBT. 150k until
       2026-11-09, then 1M, because 150k was too small.
   * - Test (2.8M)
     - Data used only once per selected model, after the selection. Earlier name: "holdout".
   * - Test (9.0M)
     - Data used one time only, in trial 10. Earlier name: "sealed test".
   * - Mistag
     - The metric of all results (%): a mean of the background mistag rates at eight working
       points. Lower is better.
   * - Standard ParT
     - The usual LR schedule: constant for 35 epochs, then a decay by ×0.01 over 15 epochs.
   * - PBT
     - Population Based Training. Five models train in parallel. A weak model copies the
       weights of a strong model and changes its LR.
   * - Ensemble
     - The mean of the outputs of two models.

All trials start from the same pretrained model and train for 50 epochs, unless the table gives
a different number.

.. _report-trials:

List of trials
--------------

Outcome: ✅ better · ❌ worse · ➖ no clear difference · ⚠️ not valid (comparator LRs too low) ·
🔧 set-up or check. Values are on Test (2.8M) unless the cell says otherwise.

Phase 0 (Jul 22 – Aug 31) was set-up. We tested first PBT versions and LR controllers and fixed
two defects. These runs used an older training set-up and are not comparable.

**Phase 1: PBT against fixed LRs** · 800k · Validation (150k) · Sep 15 – Sep 28

.. list-table::
   :header-rows: 1
   :widths: 5 12 45 28 10

   * - #
     - Dates
     - What
     - Result
     - Outcome
   * - 1
     - Sep 15–16
     - Fixed LRs 3×10⁻⁶ – 1.4×10⁻⁵
     - 0.3269
     - 🔧
   * - 2
     - Sep 16–17
     - Fixed LRs, 100 epochs
     - 0.3156
     - 🔧
   * - 3
     - Sep 17–18
     - PBT, decision every 5 epochs
     - 0.3225, −1.4% vs 1
     - ⚠️
   * - 4
     - Sep 18–19
     - PBT, 100 epochs
     - 0.3111, −1.4% vs 2
     - ⚠️
   * - 5
     - Sep 19–21
     - Small Validation subset, LR controller
     - controller did nothing
     - ➖
   * - 6
     - Sep 22–28
     - PBT every epoch vs every 5 epochs, 2 seeds
     - +0.5%, +0.1%
     - ➖
   * - 7
     - Sep 28
     - PBT decisions inside an epoch
     - not possible (optimizer state lost)
     - 🔧

**Phase 2: first large-data comparison** · 8M · Validation (150k) · Oct 12 – Nov 3

.. list-table::
   :header-rows: 1
   :widths: 5 12 45 28 10

   * - #
     - Dates
     - What
     - Result
     - Outcome
   * - 8
     - Oct 12
     - First Test (2.8M) evaluation of trials 1–6
     - agrees with Validation for large differences
     - 🔧
   * - 9
     - Oct 13
     - Prepare 8M training data and Test (9.0M)
     - ready
     - 🔧
   * - 10
     - Oct 13 – Nov 2
     - PBT vs fixed LRs as in trial 1, preregistered
     - **−7.7%** on Test (9.0M)
     - ⚠️
   * - 11
     - Oct 17 – Nov 3
     - Save the full optimizer state
     - works; not used yet
     - 🔧

**Phase 3: higher fixed LRs** · 800k · Validation (150k) · Nov 2 – Nov 9

.. list-table::
   :header-rows: 1
   :widths: 5 12 45 28 10

   * - #
     - Dates
     - What
     - Result
     - Outcome
   * - 12
     - Nov 2–3
     - Fixed LRs 1.4×10⁻⁵ – 3×10⁻⁵
     - 0.3146
     - 🔧
   * - 13
     - Nov 3
     - Fixed LRs 3×10⁻⁵ – 8×10⁻⁵
     - **0.3018**, better than all PBT so far
     - 🔧
   * - 14
     - Nov 3–4
     - Fixed LRs 8×10⁻⁵ – 3.2×10⁻⁴
     - 0.3033
     - 🔧
   * - 15
     - Nov 4
     - Fast PBT: decision every 2 epochs, ×0.5 or ×2
     - 0.3003, −1.0% vs 14
     - ✅
   * - 16
     - Nov 8–9
     - Trials 14 and 15 with a second seed
     - PBT +0.5%
     - ❌
   * - 17
     - Nov 9
     - Fast PBT with extra copies
     - +0.8% vs 15
     - ❌

Conclusion of phase 3: Validation (150k) is too small; the selection noise (about 0.5%) is as
large as the effect. Next: Validation (1M), and standard ParT as the comparator.

**Phase 4: PBT against standard ParT** · 800k · Validation (1M) · Nov 9 – Nov 20

.. list-table::
   :header-rows: 1
   :widths: 5 12 45 28 10

   * - #
     - Dates
     - What
     - Result
     - Outcome
   * - 18
     - Nov 9–13
     - Standard ParT, fixed LRs and fast PBT; 3 seeds
     - **PBT −0.8%** (−0.7, −0.9, −0.9)
     - ✅
   * - 19
     - Nov 11–12
     - PBT with a final decay
     - same models as 18
     - ➖
   * - 20
     - Nov 12
     - Standard ParT without pretraining
     - 0.378 (Validation), far worse
     - ❌
   * - 21
     - Nov 12
     - Weight averages of checkpoints
     - no gain (Validation)
     - ➖

**Phase 5: large data against standard ParT** · 8M · Validation (1M) · Nov 13 – Nov 25

.. list-table::
   :header-rows: 1
   :widths: 5 12 45 28 10

   * - #
     - Dates
     - What
     - Result
     - Outcome
   * - 22
     - Nov 13–15
     - Standard ParT, start LRs up to 1×10⁻³
     - 0.1635
     - 🔧
   * - 23
     - Nov 15–16
     - Fast PBT
     - 0.1812, **+12.4%** vs 24
     - ❌
   * - 24
     - Nov 18–20
     - Standard ParT, start LR 3×10⁻³
     - **0.1611**, the reference
     - 🔧
   * - 25
     - Nov 19–21
     - PBT with cool-down probes
     - 0.1699, +5.4% vs 24
     - ❌
   * - 26
     - Nov 20–22
     - PBT with probes and a final decay
     - 0.1655, +2.7% vs 24
     - ❌
   * - 27
     - Nov 23
     - Ensemble of trials 22 and 24
     - 0.1530, **−5.1%** vs 24
     - ✅
   * - 28
     - Nov 23–25
     - Trial 27 with two new seeds
     - −4.9%, −5.1%; **mean −5.0%**
     - ✅
   * - 29
     - Nov 23
     - Ensembles at 800k
     - no gain (Validation)
     - ➖
   * - 30
     - Nov 23
     - Where the ensemble gain comes from
     - mostly from separate training
     - 🔧
   * - 31
     - Nov 23–24
     - Final decay without dropout
     - much worse (Validation)
     - ❌
   * - 32
     - Nov 24
     - Distillation of the ensemble into one model
     - +1.0% (Validation)
     - ❌
   * - 33
     - Nov 23–25
     - Standard ParT, start LR 1×10⁻²
     - 0.1599, as 24
     - ➖

Details of each trial: :doc:`results`.

.. _report-corrections:

What changed in earlier statements
----------------------------------

* **PBT 1.4% better (phase 1) and 7.7% better (trial 10):** both against fixed LRs that were too
  low. Against good fixed LRs (trial 13), the gain is about zero.
* **PBT 10.8% worse at 8M (first value):** it is 12.4% against the better standard ParT model
  (trial 24).

.. _report-interpretation:

Interpretation
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - 800k training jets
     - 8M training jets
   * - The model
     - overfits (training loss falls, validation loss stops)
     - does not overfit
   * - Best LR schedule
     - lower the LR at the correct time
     - high LR for a long time, then a decay
   * - PBT
     - finds the correct time: **0.8% better**
     - lowers the LR too early: **2.7–12.4% worse**
   * - Ensemble of two models
     - no gain: the models are too similar
     - **5.0% better**

PBT judges each decision after two epochs. Over two epochs, a lower LR almost always looks
better. With much data this is wrong, because the high LR gives its benefit only after the
final decay.

Limits: the 8M PBT trials have one seed. All trials start from one pretrained model. Test (2.8M)
was used in diagnostics before trial 18, so it is not fully unseen.

.. _report-open:

Open items
----------

* PBT decisions inside an epoch (trial 11): implemented, not run.
* Decision frequency (trial 6): inconclusive.
* Possible next steps, not started: PBT decisions that look further ahead; ensembles of more
  than two models; other pretrained start models.
