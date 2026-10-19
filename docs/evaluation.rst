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
     - :doc:`ongoing study <ongoing>`; source entries below 8 M
   * - Selection validation ``val50k_tail``
     - 150,000
     - evaluated after every epoch; drives PBT and checkpoint selection
   * - Independent holdout ``val_holdout``
     - 2,825,125
     - one-time evaluation of selected checkpoints; never used for selection
   * - Sealed test ``test3000k``
     - 8,970,477
     - ongoing study only; opened after training is complete

Validation and holdout are disjoint windows of source entries 16–17 M; the sealed test
covers 17–20 M. Training shares no identical jet record with validation, holdout or
sealed test; holdout and sealed test also share none with the pretraining sample
(:repo:`configs/prepared/stage2b_train8m_v1/PROVENANCE.md`,
:repo:`configs/prepared/stage3_train8m_v1/TEST3000K_PROVENANCE.md`). Source entries are
jets; with no event identifier, isolation holds at the jet level only.

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

**Holdout performance**
   The selected checkpoint evaluated once on the independent holdout, on identical jets
   for every checkpoint.

PBT's validation values are additionally optimistic because the same validation set
drives its decisions; holdout values are not.

.. _statistical-rules:

Statistical rules
-----------------

* One training seed per policy (two for cadence): differences describe these runs; no
  significance is claimed.
* No event-level confidence intervals, because no event identifier exists. Paired
  holdout comparisons report point differences and per-working-point counts of
  background jets mistagged by only one model.
* Holdout checkpoints are fixed by the validation rule before evaluation. The holdout
  was inspected in earlier diagnostics, so it is not pristine; the ongoing study uses a
  sealed test set instead.

.. _holdout-protocol:

Independent holdout evaluation
------------------------------

* **Checkpoints.** The pretrained start and, for each of eight completed runs, its
  selected checkpoint (validation global best). The rule was fixed and the checkpoints
  frozen by hash before inference; the holdout was never used to re-select a model.
* **Data.** All 2,825,125 holdout jets (b 943,645; c 941,995; d 939,485), without
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
