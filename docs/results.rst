Results and Limitations
=======================

Composite mistag in percent; lower is better. Endpoints and splits are defined in
:doc:`evaluation`. All values are generated from the run manifests and the holdout
evaluation by :repo:`scripts/publish/export_results_summary.py`; tables, trajectories
and source hashes are in :repo:`published/results`.

.. _main-comparisons:

Main comparisons
----------------

Treatment − comparator; negative favours the treatment. The validation column is the
population endpoint (final-10 current-best); the holdout column is one selected
checkpoint per run (:ref:`endpoints`). The two describe different quantities.

.. list-table::
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - Comparison
     - Validation final-10
     - Difference
     - Holdout, selected checkpoint
     - Difference
   * - Windowed PBT vs fixed-LR population, 50 epochs
     - 0.341219 / 0.345568
     - −0.004350 (−1.26%)
     - 0.322484 / 0.326934
     - −0.004450 (−1.36%)
   * - Windowed PBT vs fixed LR 1.4 × 10⁻⁵, 100 epochs [#asym]_
     - 0.326944 / 0.332571
     - −0.005627 (−1.69%)
     - 0.311050 / 0.315582
     - −0.004533 (−1.44%)
   * - Every-epoch vs every-five-epochs PBT, seed 12345
     - 0.332391 / 0.333758
     - −0.001366 (−0.41%)
     - 0.317738 / 0.316246
     - **+0.001492 (+0.47%)**
   * - Every-epoch vs every-five-epochs PBT, seed 22345
     - 0.332398 / 0.334040
     - −0.001642 (−0.49%)
     - 0.316731 / 0.316314
     - **+0.000418 (+0.13%)**
   * - Every-epoch vs windowed PBT, 50 epochs [#policy]_
     - 0.332391 / 0.341219
     - −0.008827 (−2.59%)
     - 0.317738 / 0.322484
     - −0.004746 (−1.47%)

.. [#asym] Asymmetric: one fixed-LR member, continued to 100 epochs after it was the best
   member at epoch 50, against a five-member population.
.. [#policy] The policies differ in several rules, not only cadence.

.. figure:: ../published/results/figures/matched_differences.png
   :alt: Treatment minus comparator on validation and on the independent holdout
   :width: 100%

.. _windowed-vs-fixed:

Windowed PBT versus fixed learning rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both evaluations favour windowed PBT by 1.3–1.7%.
At 50 epochs the current-best trajectories are identical through epoch 15; afterwards PBT
is lower in 28 of 35 epochs. Its selected checkpoint mistags fewer holdout background jets
at seven of eight working points and the same number at the eighth. Learning rates moved
above the initial grid, reaching the upper bound in the 100-epoch run.

.. figure:: ../published/results/figures/current_best_trajectories.png
   :alt: Current-best validation trajectories, windowed PBT and fixed learning rate
   :width: 100%

   Current-best validation value per epoch; shaded: final-10 window.

.. note::

   Earlier reports gave 0.327181% and −1.62% for 100-epoch windowed PBT. That is the
   final-10 mean of the ``lr_3e-6`` member alone. The population endpoint is 0.326944%
   (−1.69%).

.. _adaptation-frequency:

Adaptation frequency: ranking reversed on the holdout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every-epoch and every-five-epochs PBT are otherwise identical (:ref:`learning-rate-policies`)
and were run with two training seeds each. On validation, every-epoch PBT is better for
both seeds, by 0.0014 and 0.0016 pp in the population endpoint and by 0.0004 and
0.0008 pp in its selected checkpoint. On the holdout, the same selected checkpoints rank
the other way for both seeds: every-epoch PBT is worse by 0.0015 pp (seed 12345) and
0.0004 pp (seed 22345). In the paired counts, every-epoch PBT mistags more background jets
at seven of eight working points for seed 12345; for seed 22345 the working points are
split (four worse, three better, one equal).

The data do not identify a cause. Three properties of the design are compatible with the
reversal and are not separated by these runs:

* every-epoch PBT has five times as many decision opportunities and applied more copies
  (32 and 25 against 9 and 8), each chosen on the same validation set used for the
  endpoint and for checkpoint selection, so its validation values can be more optimistic;
* the differences are of the size by which checkpoints of similar quality reorder between
  validation and holdout (below);
* one seed pair per setting gives no estimate of seed-to-seed variation.

Adaptation frequency is therefore reported as inconclusive.

.. _independent-holdout:

Independent holdout results
---------------------------

Selected checkpoints evaluated once on the 2,825,125-jet holdout (:ref:`holdout-protocol`).
The final-10 column is the population endpoint of the same run, given for context; it is
not a property of the selected checkpoint.

.. list-table::
   :header-rows: 1
   :widths: 36 12 17 17 18

   * - Selected checkpoint
     - Training epoch
     - Validation, selected
     - Validation, final-10
     - Holdout
   * - Pretrained start
     - 0
     - 0.456705
     - —
     - 0.438151
   * - Fixed-LR population, 50 epochs
     - 44
     - 0.344561
     - 0.345568
     - 0.326934
   * - Windowed PBT, 50 epochs
     - 50
     - 0.337727
     - 0.341219
     - 0.322484
   * - Windowed PBT, 100 epochs
     - 89
     - 0.321381
     - 0.326944
     - 0.311050
   * - Fixed LR 1.4 × 10⁻⁵, 100 epochs (single member)
     - 99
     - 0.331445
     - 0.332571
     - 0.315582
   * - Every-epoch PBT, seed 12345
     - 42
     - 0.330931
     - 0.332391
     - 0.317738
   * - Every-five-epochs PBT, seed 12345
     - 47
     - 0.331369
     - 0.333758
     - 0.316246
   * - Every-epoch PBT, seed 22345
     - 44
     - 0.330665
     - 0.332398
     - 0.316731
   * - Every-five-epochs PBT, seed 22345
     - 49
     - 0.331417
     - 0.334040
     - 0.316314

.. _validation-vs-holdout:

**Validation versus holdout.** Every selected checkpoint scores 0.010–0.019 pp lower on
the holdout, a different sample. Rank correlation across the nine evaluated checkpoints
is 0.70: the best and the three weakest keep their ranks, the five in between (holdout
0.3156–0.3177%) reorder. Differences of about 0.001 pp are below what the validation set
resolves.

.. figure:: ../published/results/figures/holdout_selected_checkpoints.png
   :alt: Validation and holdout value of each selected checkpoint
   :width: 80%

Per-checkpoint values, eight working points and paired counts:
:repo:`published/results/holdout_checkpoints.csv`,
:repo:`published/results/holdout_paired_counts.csv`. The evaluation protocol is frozen in
:repo:`configs/prepared/stage2a_holdout_v1/protocol.json`.

.. _supporting-studies:

Supporting studies
------------------

**Validation proxy.** Of nine frozen subsets of the selection validation set, tested on
38 checkpoints, a 60,000-jet representative subset ranks checkpoints like the full set
(Spearman 0.961) at 5.3 times lower cost, but detects whether one trajectory improved
between epochs in only 72% of cases. Summary:
:download:`qualification_summary.json <_static/proxy_validation/qualification_summary.json>`.

**Adaptive controller, shadow mode: not evaluable.** A rule-based controller read that
proxy after each of 96 generations of eight fixed-LR members and logged, without
applying, ×1.05 / ×0.95 proposals. It proposed two increases, both cancelled by a
configured bound, and no decreases. This run used the earlier contract (120,000 training
jets per generation, damped optimizer start). Audit:
:repo:`published/studies/controller-shadow-audit/report.md`.

.. _sub-epoch-restart-test:

**Sub-epoch adaptation: not testable under the current contract.** Restarting five times
within an epoch changed the result (maximum parameter difference 7.6 × 10⁻⁴ against an
uninterrupted epoch) because Lookahead state is not checkpointed
(:ref:`restart-semantics`). Development status: :ref:`sub-epoch-adaptation`.

**Training from random initialization.** Implemented and smoke-tested only; no
evaluated run.

.. _cost:

Cost
----

A 50-epoch, five-member run takes about 16 GPU-hours (median 225–233 s per member-epoch,
including about 44 s validation). PBT adds no training or validation work. Two runs had
slower medians (321 and 351 s); this is not a policy cost, since the matched every-epoch
run with the other seed took 226 s.

.. _limitations:

Limitations
-----------

* One seed per policy (two for cadence); differences of 0.001–0.002 pp changed sign
  between evaluations.
* No event-level confidence intervals.
* PBT decisions, checkpoint selection and the primary endpoint share one validation set.
* The holdout was inspected in earlier diagnostics.
* The 100-epoch comparator is one selected member, not a population.
* All results use the epoch-restarted Ranger contract.
* The common start, pretrained epoch 17, ranks only 4th of 20 pretrained epochs on the
  selection metric (:doc:`experiments/pretrained-0-19`).
* Effects of 1–2% have not been propagated to a physics analysis.

.. _open-questions:

Open questions
--------------

Does the advantage hold with ten times more data and on a sealed test
(:ref:`large-dataset-status`)? How large is seed-to-seed variation? Does adaptation frequency matter
once selection optimism is controlled? Does sub-epoch adaptation help with full
optimizer state?
