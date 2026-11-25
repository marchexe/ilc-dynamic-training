Ongoing Research
================

The large-dataset comparison against fixed learning rates is complete. Its design, execution record and results are in
:ref:`large-dataset-comparison`.
The comparison of PBT with standard ParT training on 8M jets, the fixes tried for it and the
ensemble study are complete: :ref:`pbt-vs-standard-part-8m`, :ref:`probe-scored-pbt`,
:ref:`prediction-ensemble`.

.. raw:: html

   <span id="large-dataset-comparison"></span><span id="large-dataset-status"></span>
   <span id="large-dataset-design"></span><span id="full-epoch-gate"></span>
   <span id="execution-record"></span><span id="determinism-verification"></span>
   <span id="pbt-vs-standard-part-8m"></span><span id="probe-scored-pbt"></span>
   <span id="ensemble-diversity"></span><span id="distillation"></span>

.. _dropout-branch:

Final decay without dropout (stopped, negative)
-----------------------------------------------

The model does not overfit at 8M (:ref:`no-overfitting-at-8m`), so its dropout (0.1 in the
particle-attention blocks) might only slow it down. A test repeated the final decay of standard
ParT at 3×10⁻³ from its saved epoch-52 checkpoint with dropout 0 and otherwise the recorded
commands (Validation (1M) only). Without dropout the model almost stopped improving during the
decay: 0.1730 at epoch 61, against 0.1595 with dropout. The test ended at epoch 62 when the
process was lost on the node and was not restarted, since the result is clear. Dropout helps at 8M
even without visible overfitting.

.. _sub-epoch-adaptation:

Sub-epoch adaptation
--------------------

A proposed scenario copies members every 0.2 epoch (five generations per epoch) and
changes learning rates on a slower cadence. Under the epoch-restarted contract it is not
testable: restarting within an epoch changes the optimizer trajectory
(:ref:`restart-semantics`; measured effect: :ref:`sub-epoch restart test
<sub-epoch-restart-test>`).

The missing parts are implemented and were merged after the large-dataset comparison had
completed. They are configuration options, off by default; no reported result and no
completed run uses them.

* A versioned checkpoint format that also stores Lookahead's slow weights, step
  counter, k and α; checkpoints without it load with the epoch-restarted behaviour
  (:repo:`scripts/training/optimizer_contract.py`).
* An epoch cursor that continues an epoch's data stream at a batch offset and restores
  the process-local random-number and preprocessing state
  (:repo:`scripts/training/data_cursor.py`).
* Generations shorter than an epoch, and for the cadenced policy an optional
  learning-rate interval *M*: copies may occur at every generation, while learning rates
  change only after every *M*-th generation (for example *M* = 5 with five generations per
  epoch).

Verified so far: unit tests, and with the real Weaver training loop on the CPU, one
epoch trained as three restarted segments equals one uninterrupted epoch bit for bit, and
two epochs trained as four half-epoch generations equal two one-epoch generations bit for
bit. The changes to execution files pinned by the large-dataset protocol are recorded in
:repo:`configs/prepared/stage3_train8m_v1/execution_revision_1.json`. Not yet done: an
equivalence test on GPU at production scale and any experiment with the scenario.
