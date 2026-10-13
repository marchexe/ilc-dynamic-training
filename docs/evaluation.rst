Experimental Design and Evaluation
==================================

This page is the single reference for the data splits, metrics, endpoints and
statistical conventions used throughout the documentation.

Data splits
-----------

All splits come from the 2025-07 SGV sample (b, c and d jets). They are defined
as windows of source entries in each flavour file; the counts below are jets after
the fixed preselection.

.. list-table::
   :header-rows: 1
   :widths: 22 16 30 32

   * - Split
     - Jets
     - Source window
     - Role
   * - Training (``train800k``)
     - 2,392,232
     - prefix of the training window
     - training data of all completed experiments
   * - Training (``train8m``)
     - 23,921,449
     - entries below 8 M
     - training data of the :doc:`ongoing study <ongoing>`
   * - Selection validation (``val50k_tail``)
     - 150,000 (50,000 per class)
     - tail of the 16 M–17 M window
     - validated after every epoch; drives PBT decisions and checkpoint selection
   * - Independent holdout (``val_holdout``)
     - 2,825,125
     - rest of the 16 M–17 M window
     - evaluation of selected checkpoints; never used for training or selection
   * - Sealed test (``test3000k``)
     - 8,970,477
     - 17 M–20 M window
     - reserved for the ongoing study; opened only after both arms are complete

Isolation was verified by content: the training windows share no identical
scalar jet record with the 16 M–17 M validation window, and the sealed test shares
none with any training or validation split or with the pretraining sample. The
selection validation set, a 15,000-jet tail control set and the holdout are exact,
disjoint windows of the 16 M–17 M range. Two older validation sets (``val50k`` and
``val5k``) lie inside the large training window and are not used for selection
with ``train8m``; ``val5k`` also overlaps ``train800k``. Evidence:
``configs/prepared/stage2b_train8m_v1/PROVENANCE.md`` and
``configs/prepared/stage3_train8m_v1/TEST3000K_PROVENANCE.md``.

The source entries are jets, not physical events, and the files store no event
identifier. Isolation is therefore established at the jet-record level; whether
the two jets of one event can fall into different splits cannot be verified.

Performance metrics
-------------------

Performance is measured as background mistag rates (in percent) at fixed signal
efficiencies, at eight working points:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Tagger
     - Signal efficiency
     - Backgrounds
   * - b-tag
     - 80% and 90%
     - c jets, d jets
   * - c-tag
     - 50% and 80%
     - b jets, d jets

The **composite mistag** (metric key
``validation_total_reference_mistag_geomean_percent``) is the geometric mean of the
b-tag score and the c-tag score, each being the geometric mean of its four mistag
rates. It is finite only when all eight working points are measured. Lower is
better. The same definition is used on the selection validation set, the holdout
and the sealed test.

Differences are given in **percentage points (pp)** of composite mistag, and as
relative differences (difference divided by the comparator's value).

Endpoints
---------

Three distinct quantities are reported. They answer different questions and must
not be compared with each other directly.

**Final-10 current-best (population validation endpoint).**
   After every epoch, take the lowest validation composite mistag among the
   population's members (the *current best*), then average this value over the last
   ten epochs of the run. It summarises the population's late-training level rather
   than one lucky checkpoint. Implementation:
   ``current_best_trajectory`` and ``final_window_mean`` in
   ``scripts/training/pbt/reporting/endpoints.py``. This is the primary endpoint of
   all comparisons.

**Global best (selected checkpoint).**
   The single member-epoch with the lowest validation composite mistag over the
   whole run. It is the checkpoint a user would select, and it is the checkpoint
   evaluated on the independent holdout. Because it is the minimum of many noisy
   measurements, its validation value is optimistic.

**Independent holdout performance.**
   The composite mistag of the selected checkpoint on the holdout, measured once,
   with identical rows for all checkpoints. It is unaffected by selection on
   validation.

The validation value of PBT runs is additionally optimistic because the same
validation set drives PBT's copy decisions. Holdout values do not have this bias.

Comparisons
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Comparison
     - Matched
     - Not matched
   * - Windowed PBT vs fixed-LR population, 50 epochs
     - checkpoint, data, population, initial learning rates, seed schedule,
       training contract
     - —
   * - Windowed PBT vs fixed LR 1.4 × 10\ :sup:`-5`, 100 epochs
     - checkpoint, data, contract, horizon
     - comparator is one member, continued after it was identified at epoch 50
       as the best fixed-LR member; population size and compute differ
   * - Every-epoch vs every-five-epochs PBT, two seeds
     - everything except the adaptation interval
     - —
   * - Every-epoch PBT vs windowed PBT, 50 epochs
     - checkpoint, data, population, seed, contract
     - several policy rules, not only cadence

Members of the same seed schedule consume identical training rows in every epoch,
so a same-named member evolves identically across matched runs until it is first
copied. In the completed 50-epoch runs, the never-copied 1.4 × 10\ :sup:`-5`
member of windowed PBT has model weights identical, tensor for tensor, to the
fixed-LR member at epochs 1, 23 and 50 (checked directly), and the same final-10
value (0.346416%). Differences between matched runs therefore arise from the
policies, not from data order or numerical non-determinism.

Statistical conventions
-----------------------

* **No significance claims.** Each policy was run with one training seed (two
  for the cadence comparison). Differences describe these realizations, not a
  distribution over seeds.
* **No event-level confidence intervals.** Without physical event identifiers,
  resampling at the event level is impossible, and resampling jet rows would
  ignore correlations between jets of one event. Paired holdout comparisons
  therefore report point differences and per-working-point counts of background
  jets that pass for one model only.
* **Operational margins are not tests.** The 0.002 pp PBT decision margin is a
  policy parameter.
* **Selection hygiene.** Checkpoints evaluated on the holdout were fixed by a rule
  on validation (global best) before inference. The holdout was not used to choose
  any model, but it was inspected in earlier diagnostic studies and is therefore
  not a pristine test set. The ongoing study uses a sealed test set for this reason.

Experiment states
-----------------

The documentation uses the following states: *configured* (inputs frozen),
*running*, *completed* (all epochs trained and verified), *evaluated* (independent
evaluation performed) and *interpreted* (included in the results). Only completed
and evaluated experiments contribute to :doc:`results`.

Glossary
--------

.. glossary::

   Member
      One training branch of a population, identified by a stable name.

   Copy (exploit) and mutation (explore)
      A PBT replacement: the recipient takes the donor's model, optimizer and
      loss-scaler state, then trains with a perturbed learning rate.

   Training epoch
      One full pass over the training split; epochs are numbered from the pretrained
      starting checkpoint.

   Current best
      The lowest validation composite mistag among the members at one epoch.

   pp
      Percentage points of composite mistag.
