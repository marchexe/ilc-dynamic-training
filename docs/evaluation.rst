Experimental Design and Evaluation
==================================

The definitions on this page are used throughout the documentation.

.. _data-splits:

Data splits
-----------

Windows of source entries in each flavour file of the 2025-07 SGV sample; jet counts
after preselection.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Split
     - Jets
     - Role
   * - Training ``train800k``
     - 2,392,232
     - completed experiments; prefix of ``train8m``
   * - Training ``train8m``
     - 23,921,449
     - :ref:`large-dataset comparison <large-dataset-comparison>`; source entries below 8 M
   * - Validation (150k) ``val50k_tail``
     - 150,000
     - evaluated after every epoch; drives PBT and checkpoint selection, until 2026-11-09
   * - Validation (1M) ``sel1m_tail``
     - 999,999
     - the same role, from 2026-11-10; the last 333,333 rows of each ``train8m`` file, removed
       from training
   * - Test (2.8M) ``val_holdout``
     - 2,825,125
     - one-time evaluation of selected checkpoints; never used for selection
   * - Test (9.0M) ``test3000k``
     - 8,970,477
     - large-dataset comparison only; opened once, after both arms had completed

Validation (150k) and Test (2.8M) are disjoint windows of source entries 16–17 M; Test (9.0M)
covers 17–20 M. Training shares no identical jet record with Validation (150k), Test (2.8M) or
Test (9.0M); Test (2.8M) and Test (9.0M) also share none with the pretraining sample
(:repo:`configs/prepared/stage2b_train8m_v1/PROVENANCE.md`,
:repo:`configs/prepared/stage3_train8m_v1/TEST3000K_PROVENANCE.md`). Source entries are
jets; with no event identifier, isolation holds at the jet level only.

Earlier versions of this site called Test (2.8M) the "holdout" and Test (9.0M) the "sealed test".
File, directory and configuration names keep these words, because frozen records contain them.

.. _metric:

Metric
------

Background mistag rate (%) at fixed signal efficiency, at eight working points: b-tag
at 80% and 90% against c and d; c-tag at 50% and 80% against b and d. The **composite
mistag** (``validation_total_reference_mistag_geomean_percent``) is the geometric mean of
the b-tag and c-tag scores, each the geometric mean of its four rates. Lower is better.
Differences are given in percentage points (pp) and relative to the comparator.

.. _endpoints:

Endpoints
---------

**Final-10 current-best** (primary)
   After each epoch take the population's lowest validation value, then average over the
   last ten epochs (:repo:`scripts/training/pbt/reporting/endpoints.py`). Describes the
   population, not one checkpoint.

**Selected checkpoint**
   The single member-epoch with the lowest validation value over the run (global best).
   Its validation value is optimistic: it is the minimum of many noisy measurements.

**Test (2.8M) performance**
   The selected checkpoint evaluated once on Test (2.8M), on identical jets
   for every checkpoint.

PBT's validation values are additionally optimistic because the same validation set
drives its decisions; Test (2.8M) values are not.

.. _statistical-rules:

Statistical rules
-----------------

* Earlier comparisons: one training seed per policy (two for cadence); differences describe
  these runs and no significance is claimed. The comparison with standard ParT training has
  three seeds.
* Test (2.8M) files have no event identifier, but the jets of one event are adjacent rows
  (the mass of two adjacent jets peaks at the Higgs mass; that of two random jets does not).
  Intervals for the comparison with standard ParT training resample blocks of 100 adjacent
  rows of one file, with shared draws for the compared checkpoints; they cover Test (2.8M)
  sample only, not seed-to-seed variation. The earlier comparisons report point differences
  and per-working-point counts of background jets mistagged by only one model.
* The checkpoints for Test (2.8M) are fixed by the validation rule before evaluation. Test (2.8M)
  was inspected in earlier diagnostics, so it is not pristine; the large-dataset
  comparison uses Test (9.0M) instead.

.. _holdout-protocol:

Test (2.8M) evaluation
------------------------------

* **Checkpoints.** The pretrained start and, for each of eight completed runs, its
  selected checkpoint (validation global best). The rule was fixed and the checkpoints
  frozen by hash before inference; Test (2.8M) was never used to re-select a model.
* **Data.** All 2,825,125 test jets (b 943,645; c 941,995; d 939,485), without
  reweighting. Every checkpoint is evaluated on identical jets, so comparisons are paired
  at the jet level.
* **Integrity.** Inference only (no optimizer step), complete and non-repeating row
  coverage, finite values at all eight working points, and metrics recomputed from the
  predictions. Protocol: :repo:`configs/prepared/stage2a_holdout_v1/protocol.json`;
  verifier: :repo:`scripts/validation/verify_stage2a_provenance.py`.
* **Publication.** Curated tables only; raw predictions and logs are not published.

.. _checkpoint-identity:

Determinism and checkpoint identity
-----------------------------------

Two training states are called identical when their restart bundles
(:ref:`restart-semantics`) agree bit for bit in content: every tensor in dtype, shape and
bytes, every scalar exactly, and every key of the model, optimizer and loss-scaler state.
Bundles are loaded on the CPU, so the only information not compared is the device on
which each tensor was stored. File hashes are not a test of identity: the serialized
archive records that device, and a checksum over the archive depends on it, so identical
states saved by workers on different GPU indices are different files. File hashes are used
only to show that a given file has not changed (checkpoint ledgers, archives, frozen
references). The comparison is implemented in
:repo:`scripts/experiments/stage3/checkpoint_contents.py`.

Matched runs share data order, so a member evolves identically across runs until it is
first copied. Verified for the completed experiments: the never-copied 1.4 × 10⁻⁵ member
has identical weights in the 50-epoch windowed and fixed runs at epochs 1, 23 and 50.
