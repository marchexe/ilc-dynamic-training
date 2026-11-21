Ongoing Research
================

The large-dataset comparison against fixed learning rates is complete. Its design, execution record and results are in
:ref:`large-dataset-comparison`.
The comparison of PBT with standard ParT training on 8M jets is also complete:
:ref:`pbt-vs-standard-part-8m`. A fix for its result is in progress: :ref:`probe-scored-pbt`.

.. raw:: html

   <span id="large-dataset-comparison"></span><span id="large-dataset-status"></span>
   <span id="large-dataset-design"></span><span id="full-epoch-gate"></span>
   <span id="execution-record"></span><span id="determinism-verification"></span>
   <span id="pbt-vs-standard-part-8m"></span>

.. _probe-scored-pbt:

PBT with cool-down probes on 8M jets
------------------------------------

On 8M jets PBT lowers every learning rate within ten epochs (:ref:`pbt-vs-standard-part-8m`).
A cool-down probe tests a fix: at each decision, a copy of every member trains a quarter epoch
at 0.1× its learning rate and is validated, and the decision ranks the members on these probe
values. Probe weights never return to the population and are never selected
(:repo:`scripts/training/pbt/planning/window_probe.py`). The probe settings were chosen after
the 8M result and are not tuned.

All values below are on the selection set; holdout evaluation follows once each run has
completed (:repo:`configs/prepared/holdout_sel1m_probe_v1`,
:repo:`configs/prepared/holdout_sel1m_probe_decay_v1`).

* **800k, probes** (:repo:`configs/experiments/sel1m_fast_pbt_probe.yaml`): completed;
  best 0.2989, against 0.2982 for PBT without probes and 0.3006 for standard ParT. The probes
  change little where PBT already wins.
* **8M, probes** (:repo:`configs/experiments/sel1m_train8m_fast_pbt_probe.yaml`): the
  learning rates stay at 1.2×10⁻⁴–1×10⁻³ instead of collapsing, and at epoch 45 the run is 8%
  better than PBT without probes. Without a final decay it then stops improving (0.1684 from
  epoch 61), behind standard ParT (0.1584 at start learning rate 3×10⁻³).
* **8M, probes and final decay** (:repo:`configs/experiments/sel1m_train8m_fast_pbt_probe_decay.yaml`):
  in progress. As above, plus the final decay of standard ParT (last 15 of 50 epochs, ×0.01)
  and the learning-rate range of standard ParT including 3×10⁻³.

These runs use seed 12345, on which three PBT variants have now been tried on 8M, so they are
development runs. A claim needs new seeds with the configuration fixed beforehand: the
configurations and holdout specification for seeds 22345 and 32345 were committed before the
result of the last run was known (:repo:`configs/prepared/holdout_sel1m_train8m_seeds_v1`).

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
