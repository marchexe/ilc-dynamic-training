Research History and Provenance
===============================

This page records how the study developed, which earlier approaches were superseded,
and how internal labels that still appear in file and directory names map to the
terms used in this documentation. Dates are UTC and refer to the completion of the
recorded work.

Timeline
--------

July–August 2026: exploratory phase
   Several adaptive schemes were tried on the pretrained tagger: a guarded
   smooth-learning-rate population, a contextual-bandit (LinUCB) learning-rate
   controller with an active/observe comparison, exploit-and-mutate PBT with tiered
   validation, anchor-copy PBT with learning-rate re-centring (a 100-generation
   validation completed on 2026-08-31), and a learning-rate/mistag correlation study
   across seeds, validation sizes and generation lengths (2026-08-20). These runs used
   an earlier training contract (see below) and are not comparable with the results
   in :doc:`results`.

2026-09-16: deterministic training foundation
   Full-epoch traversal with audited data loading and exact model, optimizer and
   loss-scaler continuation was verified; the 50-epoch fixed-learning-rate population
   completed.

2026-09-17
   The 1.4 × 10\ :sup:`-5` and 8.5 × 10\ :sup:`-6` fixed-learning-rate members were
   continued to 100 epochs; the pretrained checkpoints of epochs 0–19 were
   re-evaluated.

2026-09-18 to 2026-09-19
   The windowed-PBT policy was frozen; its 50-epoch run and the 100-epoch
   continuation completed.

2026-09-20 to 2026-09-21
   The validation-proxy qualification completed; the adaptive controller was replayed
   offline and then audited in shadow mode (:doc:`exploratory_studies`).

2026-09-23 to 2026-09-28
   Every-epoch PBT and the every-five-epochs control completed for two seeds. A
   design test showed that sub-epoch adaptation is blocked by the optimizer restart
   contract. An interim technical report followed on 2026-10-01
   (:doc:`archive/index`).

2026-10-12
   The independent holdout evaluation of nine selected checkpoints completed.

2026-10-13
   The ten-times larger training set was converted, verified and smoke-tested on GPU.
   The large-dataset comparison was preregistered and launched; its full-epoch gate
   passed on the second attempt after a documented verifier correction
   (:doc:`ongoing`).

2026-10-14
   The documentation was reorganised. The 100-epoch windowed-PBT endpoint was
   corrected from a single-member value to the population definition
   (:doc:`results`).

Superseded training contract
----------------------------

Experiments before mid-September 2026 differ from the current contract in ways that
affect performance: training used capped samples per epoch rather than full epochs,
the restored optimizer state was damped, and the earliest runs did not freeze
BatchNorm statistics (a diagnostic pair of runs established that freezing removes
a degradation in the first epoch). Validation also used smaller or different sets.
Their configurations are kept for reproducibility in ``configs/experiments/`` and
``configs/experiments/archive/``, and their run records remain in the local run
archive, but none of their numbers are used in the current results.

Internal labels
---------------

Development labels remain in directory names, configuration names and frozen
records, which cannot be renamed without breaking recorded hashes. They map to the
documentation's terms as follows.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Internal label (paths and records)
     - Meaning in this documentation
   * - Stage 1, "foundation"
     - deterministic training contract and the completed 50/100-epoch comparisons
   * - Stage 2A, ``stage2a_holdout_v1``
     - :doc:`holdout_evaluation`
   * - Stage 2B, ``stage2b_train8m_v1``
     - preparation and GPU smoke test of the ``train8m`` training set
   * - Stage 3, ``stage3_train8m_v1``
     - :doc:`ongoing`
   * - Variant A; ``cadenced_pbt_v1`` (interval 1)
     - every-epoch PBT
   * - cadence5 control; ``cadenced_pbt_v1`` (interval 5)
     - every-five-epochs PBT
   * - Variant B
     - sub-epoch (0.2-epoch) adaptation; not implemented
   * - ``windowed_pbt_v2``
     - windowed PBT
   * - ``fixed_lr_grid``
     - fixed-learning-rate population
   * - "control proxy" or "full reference" (older pages) applied to ``val50k_tail``
     - selection validation set (150,000 jets)
   * - ``val_holdout``
     - independent holdout
   * - ``test3000k``
     - sealed test set
   * - generation
     - one training epoch in all current experiments

Run directories
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Run directory (``runs/pbt/``)
     - Experiment
   * - ``foundation_fixed_lr_50epochs_20260916``
     - fixed-LR population, 50 epochs
   * - ``fixed_lr_continuation_20260917_lr_14e-6`` and ``..._lr_8_5e-6``
     - fixed-LR members continued to 100 epochs
   * - ``windowed_pbt_v2``, ``windowed_pbt_v2_100epochs``
     - windowed PBT, 50 and 100 epochs
   * - ``cadenced_pbt_v1_50epochs``, ``cadenced_pbt_v1_seed22345_50epochs``
     - every-epoch PBT, seeds 12345 and 22345
   * - ``cadenced_pbt_v1_cadence5_50epochs``, ``cadenced_pbt_v1_cadence5_seed22345_50epochs``
     - every-five-epochs PBT, seeds 12345 and 22345
   * - ``pretrained_fixed_lr_8gpu_representative60k_shadow``
     - shadow audit of the adaptive controller
