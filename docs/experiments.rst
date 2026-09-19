Experiments
===========

Canonical runs
--------------

Display names are used for readability; manifests and provenance retain the real
filesystem paths shown below.

.. list-table:: Current and reproducibility-critical runs
   :header-rows: 1
   :widths: 22 38 40

   * - Display name
     - Filesystem path
     - Role
   * - Windowed PBT v2 — 50 epochs
     - ``runs/pbt/windowed_pbt_v2``
     - Verified reference and immutable continuation ancestor.
   * - Windowed PBT v2 — 100 epochs
     - ``runs/pbt/windowed_pbt_v2_100epochs``
     - Verified continuation and current primary result.
   * - Fixed-LR grid — 50 epochs
     - ``runs/pbt/foundation_fixed_lr_50epochs_20260916``
     - Matched five-member control and source of the 100-epoch controls.
   * - Fixed LR 14e-6 — 100 epochs
     - ``runs/pbt/fixed_lr_continuation_20260917_lr_14e-6``
     - Single-member control covering full epochs 51–100 after continuation.
   * - Fixed LR 8.5e-6 — 100 epochs
     - ``runs/pbt/fixed_lr_continuation_20260917_lr_8_5e-6``
     - Single-member control covering full epochs 51–100 after continuation.
   * - Deterministic foundation check
     - ``runs/pbt/foundation_correctness_20260916``
     - Matched data order, optimizer steps, model, optimizer and scaler state.
   * - Pretrained baseline sweep — epochs 0–19
     - ``runs/eval/original_pretrained_0_19_20260917``
     - Re-evaluation used to select and document the epoch-17 baseline.

Meaningful milestones
---------------------

Historical display names below are labels only. The completed run directories
keep their original names and paths.

.. list-table:: Concise experiment timeline
   :header-rows: 1
   :widths: 14 30 56

   * - Date
     - Milestone
     - Result
   * - 2026-08-04
     - checkpoint ownership fix
     - PBT recipient LR ownership was separated from the dynamic controller;
       the retained verification run records the regression evidence.
   * - 2026-08-20
     - LR/mistag comparison matrix
     - Seed, validation-size and generation-length studies showed variable and
       often uncertain within-generation LR correlations; they remain historical
       motivation rather than the current algorithm.
   * - 2026-09-16
     - deterministic foundation
     - Full-epoch traversal and model/optimizer/scaler parity were verified; a
       50-epoch fixed-LR control was completed.
   * - 2026-09-18
     - windowed PBT v2
     - Frozen window scoring and deterministic exploration were introduced, then
       completed for 50 full epochs.
   * - 2026-09-19
     - 100-epoch continuation
     - The verified 50-epoch population was continued without rewriting its
       ancestry; matched fixed-LR controls reached the same total horizon.

Historical ``exploit_mutate``, ``population_lr_policy`` and
``anchor_copy_lr_recenter`` studies remain useful provenance, but they are not
the current method. Smoke runs, failed launches, temporary preparations and
derived plots are not scientific milestones.

.. list-table:: Selected historical studies
   :header-rows: 1
   :widths: 30 45 25

   * - Display name
     - Filesystem path
     - Historical role
   * - LR/mistag sensitivity study
     - ``runs/showcase/lr_mistag_correlation_matrix``
     - Seed, validation-size and generation-length comparison.
   * - Anchor recenter v2 — 100 generations
     - ``runs/showcase/anchor_copy_lr_recenter_v2_rewrite/attempt2_momentum002_20260830_100gen``
     - Completed legacy-strategy comparison.
   * - Tiered exploit/mutate pilot
     - ``runs/showcase/pretrained_exploit_mutate_8gpu_10m_tiered_pilot_20260804_091343``
     - Early exploit/mutate milestone.
   * - Night controller study
     - ``runs/showcase/pretrained_pbt_4gpu_night_controller_active_20260805_015318``
     - Early controller-active milestone.
   * - LR ownership regression check
     - ``runs/pbt/ownership_fix_verify``
     - Checkpoint/controller ownership evidence.
