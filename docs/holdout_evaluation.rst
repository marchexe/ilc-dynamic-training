Independent Holdout Evaluation
==============================

This supporting page documents the evaluation of nine previously selected
checkpoints on the independent holdout. The summary of its findings is in
:doc:`results`.

Design
------

* **Checkpoints.** The pretrained starting checkpoint and, for each of eight
  completed runs, its validation global best (the lowest validation composite
  mistag over all member-epochs). The choice was made by this fixed rule before
  inference and frozen with checkpoint hashes; the holdout was never used to
  re-select a model.
* **Data.** The holdout split (:doc:`evaluation`): 2,825,125 jets (b 943,645;
  c 941,995; d 939,485), evaluated in full without reweighting. Every checkpoint saw
  identical rows, so comparisons are paired at the jet level.
* **Integrity.** Inference only (zero optimizer steps), full traversal, complete
  and non-repeating row coverage, finite values at all eight working points, and
  metrics recomputed from the predictions. The frozen evaluation protocol is
  ``configs/prepared/stage2a_holdout_v1/protocol.json`` (SHA256
  ``cabf6703…``); ``scripts/validation/verify_stage2a_provenance.py`` re-verifies
  its inputs and results.
* **Publication.** Raw predictions and logs are not published. Curated tables:
  :download:`holdout_checkpoints.csv <../published/results/holdout_checkpoints.csv>`,
  :download:`holdout_comparisons.csv <../published/results/holdout_comparisons.csv>`,
  :download:`holdout_paired_counts.csv <../published/results/holdout_paired_counts.csv>`.

Selected checkpoints
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 12 18 18 18

   * - Checkpoint
     - Training epoch
     - Selection validation
     - Population final-10 (validation)
     - Independent holdout
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
   * - Fixed LR 1.4 × 10\ :sup:`-5`, 100 epochs (single member)
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

The population final-10 column is taken from the training trajectories and is shown
for context only; it is not a property of the selected checkpoint.

Working points on the holdout
-----------------------------

Mistag rates in percent. b-tag columns give the background at 80% and 90% b-jet
efficiency; c-tag columns at 50% and 80% c-jet efficiency.

.. list-table::
   :header-rows: 1
   :widths: 24 9 9 10 10 9 10 9 10

   * - Checkpoint
     - b: c @80
     - b: d @80
     - b: c @90
     - b: d @90
     - c: b @50
     - c: d @50
     - c: b @80
     - c: d @80
   * - Pretrained start
     - 0.1926
     - 0.0623
     - 3.6329
     - 0.2098
     - 0.6022
     - 0.0644
     - 3.0166
     - 1.2704
   * - Fixed-LR population, 50 ep.
     - 0.1376
     - 0.0476
     - 2.6034
     - 0.1744
     - 0.4004
     - 0.0477
     - 2.3154
     - 0.9937
   * - Windowed PBT, 50 ep.
     - 0.1350
     - 0.0476
     - 2.5442
     - 0.1726
     - 0.3911
     - 0.0473
     - 2.2865
     - 0.9806
   * - Windowed PBT, 100 ep.
     - 0.1278
     - 0.0463
     - 2.4294
     - 0.1693
     - 0.3791
     - 0.0452
     - 2.2160
     - 0.9471
   * - Fixed LR 1.4e-5, 100 ep.
     - 0.1316
     - 0.0461
     - 2.4807
     - 0.1708
     - 0.3807
     - 0.0463
     - 2.2497
     - 0.9650
   * - Every-epoch PBT, s12345
     - 0.1335
     - 0.0462
     - 2.5026
     - 0.1719
     - 0.3883
     - 0.0458
     - 2.2692
     - 0.9706
   * - Every-5-epochs PBT, s12345
     - 0.1331
     - 0.0455
     - 2.4981
     - 0.1712
     - 0.3854
     - 0.0458
     - 2.2645
     - 0.9681
   * - Every-epoch PBT, s22345
     - 0.1332
     - 0.0459
     - 2.4942
     - 0.1707
     - 0.3850
     - 0.0462
     - 2.2646
     - 0.9662
   * - Every-5-epochs PBT, s22345
     - 0.1314
     - 0.0461
     - 2.4847
     - 0.1711
     - 0.3840
     - 0.0465
     - 2.2555
     - 0.9662

Paired background counts
------------------------

For a paired comparison, each background jet at each working point is either
mistagged by both models, by neither, or by only one of them. Only the last two
categories carry the difference. The table shows them for the matched 50-epoch
comparison of windowed PBT (treatment) and the fixed-learning-rate population
(comparator). Background jets per working point: about 940,000.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Working point
     - Mistagged by fixed LR only
     - Mistagged by windowed PBT only
   * - b-tag 80%, c background
     - 55
     - 31
   * - b-tag 80%, d background
     - 15
     - 15
   * - b-tag 90%, c background
     - 1315
     - 757
   * - b-tag 90%, d background
     - 48
     - 32
   * - c-tag 50%, b background
     - 182
     - 95
   * - c-tag 50%, d background
     - 10
     - 6
   * - c-tag 80%, b background
     - 715
     - 442
   * - c-tag 80%, d background
     - 297
     - 174

These counts describe jets, not independent events, and are not converted into a
significance statement (:doc:`evaluation`). Counts for all five comparisons are in
the CSV file above.

Validation versus holdout ranking
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Rank
     - Selection validation
     - Independent holdout
   * - 1
     - Windowed PBT, 100 ep.
     - Windowed PBT, 100 ep.
   * - 2
     - Every-epoch PBT, seed 22345
     - Fixed LR 1.4e-5, 100 ep.
   * - 3
     - Every-epoch PBT, seed 12345
     - Every-5-epochs PBT, seed 12345
   * - 4
     - Every-5-epochs PBT, seed 12345
     - Every-5-epochs PBT, seed 22345
   * - 5
     - Every-5-epochs PBT, seed 22345
     - Every-epoch PBT, seed 22345
   * - 6
     - Fixed LR 1.4e-5, 100 ep.
     - Every-epoch PBT, seed 12345
   * - 7
     - Windowed PBT, 50 ep.
     - Windowed PBT, 50 ep.
   * - 8
     - Fixed-LR population, 50 ep.
     - Fixed-LR population, 50 ep.
   * - 9
     - Pretrained start
     - Pretrained start

Spearman rank correlation: 0.70. Rank comparisons across different horizons are
descriptive; only the matched comparisons in :doc:`results` are interpreted.
