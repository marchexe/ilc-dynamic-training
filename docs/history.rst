Research History
================

Timeline
--------

* **Jul–Aug 2026, exploratory phase:** smooth-LR populations, a LinUCB controller,
  exploit-and-mutate and anchor-copy PBT, a learning-rate/mistag correlation study.
  These runs used an earlier contract (samples capped per epoch, damped optimizer
  start, BatchNorm not frozen at first) and are not comparable with :doc:`results`.
  Their configurations remain in Git history only.
* **2026-09-16 to 09-19:** deterministic full-epoch contract; fixed-LR and windowed-PBT
  runs to 50 and 100 epochs.
* **2026-09-20 to 09-21:** validation-proxy qualification; adaptive controller replay and
  shadow audit.
* **2026-09-23 to 09-28:** every-epoch versus every-five-epochs PBT, two seeds; sub-epoch
  adaptation found blocked. :doc:`Interim report <archive/index>`, 2026-10-01.
* **2026-10-12:** independent holdout evaluation.
* **2026-10-13:** large-dataset study preregistered and launched (:doc:`ongoing`).
* **2026-10-14:** documentation reorganised; 100-epoch endpoint corrected.

Internal names
--------------

Directory, configuration and record names keep development labels, because frozen
records hash them.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name in paths and records
     - Meaning
   * - Stage 2A, ``stage2a_holdout_v1``
     - independent holdout evaluation
   * - Stage 2B, ``stage2b_train8m_v1``
     - ``train8m`` preparation and GPU smoke test
   * - Stage 3, ``stage3_train8m_v1``
     - large-dataset study
   * - Variant A / cadence5 control
     - every-epoch / every-five-epochs PBT
   * - Variant B
     - sub-epoch adaptation (not implemented)
   * - "control proxy", "full reference" for ``val50k_tail``
     - selection validation set
   * - generation
     - one training epoch in all current experiments

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Run directory (``runs/pbt/``, local)
     - Experiment
   * - ``foundation_fixed_lr_50epochs_20260916``
     - fixed-LR population, 50 epochs
   * - ``fixed_lr_continuation_20260917_lr_14e-6``, ``..._lr_8_5e-6``
     - fixed-LR members, epochs 51–100
   * - ``windowed_pbt_v2``, ``windowed_pbt_v2_100epochs``
     - windowed PBT, 50 and 100 epochs
   * - ``cadenced_pbt_v1_50epochs``, ``cadenced_pbt_v1_seed22345_50epochs``
     - every-epoch PBT, seeds 12345, 22345
   * - ``cadenced_pbt_v1_cadence5_50epochs``, ``cadenced_pbt_v1_cadence5_seed22345_50epochs``
     - every-five-epochs PBT, seeds 12345, 22345
   * - ``pretrained_fixed_lr_8gpu_representative60k_shadow``
     - controller shadow audit

.. toctree::
   :hidden:

   archive/index
