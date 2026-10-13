Experiment Records
==================

Each record documents one completed run: its result summary, figures, learning-rate
and copy histories, model identities and provenance. Records were generated from the
recorded run evidence by ``scripts/publish/export_run.py`` and later annotated with the
endpoints and corrections used in :doc:`results`. Downloads are path-free copies of the
publication bundles in ``published/experiments/``.

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Record
     - Role
     - Note
   * - :doc:`Windowed PBT — 50 epochs <experiments/pbt-v2-50>`
     - PBT, matched comparison
     - comparator: fixed-learning-rate population, 50 epochs
   * - :doc:`Windowed PBT — 100 epochs <experiments/pbt-v2-100>`
     - PBT, longer horizon
     - continuation of the 50-epoch run; corrected final-10 endpoint
   * - :doc:`Every-epoch PBT — 50 epochs <experiments/cadenced-pbt-v1-50>`
     - adaptation-frequency study
     - seed 12345; the other three runs of the study are summarised in :doc:`results`
   * - :doc:`Fixed learning rate — 50 epochs <experiments/fixed-lr-50>`
     - comparator
     - five-member population
   * - :doc:`Fixed learning rate — 100 epochs <experiments/fixed-lr-100>`
     - comparator (asymmetric)
     - two single members continued after epoch 50
   * - :doc:`Pretrained checkpoints — epochs 0–19 <experiments/pretrained-0-19>`
     - starting point
     - re-evaluation on the selection validation set

The complete set of validation endpoints, including the runs without a record page, is
in ``published/results/validation_endpoints.csv``; the independent holdout evaluation
is documented in :doc:`holdout_evaluation`.

.. toctree::
   :hidden:

   experiments/pbt-v2-50
   experiments/pbt-v2-100
   experiments/cadenced-pbt-v1-50
   experiments/fixed-lr-50
   experiments/fixed-lr-100
   experiments/pretrained-0-19
