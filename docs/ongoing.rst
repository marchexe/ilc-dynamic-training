Ongoing Research
================

The large-dataset comparison against fixed learning rates is complete. Its design, execution record and results are in
:ref:`large-dataset-comparison`.
The comparison of PBT with standard ParT training on 8M jets and the fixes tried for it are
complete: :ref:`pbt-vs-standard-part-8m`, :ref:`probe-scored-pbt`.

.. raw:: html

   <span id="large-dataset-comparison"></span><span id="large-dataset-status"></span>
   <span id="large-dataset-design"></span><span id="full-epoch-gate"></span>
   <span id="execution-record"></span><span id="determinism-verification"></span>
   <span id="pbt-vs-standard-part-8m"></span><span id="probe-scored-pbt"></span>

.. _ensemble-diversity:

Where the ensemble gain comes from
----------------------------------

The prediction ensemble of the two standard ParT models is 5.05% better on the holdout than the
better single model (:ref:`prediction-ensemble`). The selection-set predictions of nine finished
8M models were compared in all 36 pairs (selection set only, existing checkpoints only):

* Two checkpoints of the same run, or two members of the same PBT run, disagree on 0.2–0.4% of
  the jets and give no ensemble gain.
* Two separately trained models disagree on 1.5–2.3% of the jets. If both are strong, the
  ensemble is 1.5–6.5% better than the better of the two; a weak partner (selection value above
  0.17) gives little or makes it worse.
* The largest gains need the standard ParT model at 1×10⁻³: with standard ParT at 3×10⁻³ 4.4%
  (6.5% with the last epoch of both), with members of the last PBT run 4.0–4.7%. Standard ParT
  at 3×10⁻³ with the same PBT members gives only 1.5–2.9%.

Separate training is therefore necessary, and a different learning rate seems to add to it, but
with one seed the two effects cannot be separated. A second standard ParT run at 3×10⁻³ with a
new seed would separate them. If the learning rate matters, a population method that keeps its
members different, instead of copying them, could give a better ensemble at the same cost.

.. _dropout-branch:

Final decay without dropout
---------------------------

The model does not overfit at 8M (:ref:`no-overfitting-at-8m`), so its dropout (0.1 in the
particle-attention blocks) may only slow it down. A test repeats the final decay of standard ParT
at 3×10⁻³ (epochs 53–67) from its saved epoch-52 checkpoint with dropout 0 and otherwise the
recorded commands. This is a quarter of a full run; a full run follows only if the decay alone
shows a gain.

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
