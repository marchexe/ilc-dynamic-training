Results
=======

Current reference results
-------------------------

All values below are recorded in the run manifests and summaries. The selection
metric is ``validation_total_reference_mistag_geomean_percent`` on the fixed
``val50k_tail`` control proxy; lower is better.

.. list-table:: Best recorded selection metric
   :header-rows: 1
   :widths: 31 13 15 19 22

   * - Experiment
     - Horizon
     - Best score (%)
     - Selected member
     - Interpretation
   * - pretrained epoch 17
     - baseline
     - 0.456705
     - —
     - measured starting checkpoint
   * - windowed PBT v2
     - 50 epochs
     - 0.337727
     - ``lr_11_25e-6``
     - 26.05% below the measured start
   * - fixed-LR grid
     - 50 epochs
     - 0.344561
     - ``lr_14e-6``
     - matched control population
   * - windowed PBT v2
     - 100 epochs
     - 0.321381
     - ``lr_3e-6``
     - 29.63% below the measured start
   * - fixed LR ``14e-6``
     - 100 epochs
     - 0.331445
     - ``lr_14e-6``
     - continuation control
   * - fixed LR ``8.5e-6``
     - 100 epochs
     - 0.339059
     - ``lr_8_5e-6``
     - continuation control

Member labels identify persistent branches; they do not encode the current LR.
The 100-epoch best checkpoint came from the branch named ``lr_3e-6`` at
generation 88, after PBT had changed learning rates.

Late-epoch comparison
---------------------

The purpose-built comparison uses the mean over full epochs 91–100, avoiding a
single-checkpoint comparison. It records 0.327181% for PBT and 0.332571% for the
fixed-``14e-6`` control: 0.005390 percentage points, or 1.62% relative reduction.

.. image:: _static/results/pbt_100_performance.png
   :alt: Windowed PBT trajectories and fixed-learning-rate control over 100 epochs
   :width: 100%

The PBT population executed 11 copy/mutation actions at 20 five-epoch decision
boundaries. Its learning rates moved beyond the initial 3–14 × 10⁻⁶ range.

.. image:: _static/results/pbt_100_learning_rates.png
   :alt: Learning-rate trajectories and copy boundaries for the 100-epoch PBT run
   :width: 100%

Interpretation and limits
-------------------------

The completed evidence supports continued work on adaptive LR selection: PBT
improved the late-epoch control-proxy result relative to the matched fixed-LR
control and explored useful rates above the initial grid. It does not establish
causality for individual mutations or full-validation generalization.

The 50/100-epoch reports used only the control proxy and scheduled no independent
monitor/full corroboration tier. The global-best selection checkpoint can also
differ from the separately computed best-physics-summary checkpoint. Those roles
must remain distinct when interpreting or exporting a model.
