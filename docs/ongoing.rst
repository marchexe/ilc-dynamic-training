Ongoing Research
================

.. _large-dataset-comparison:

Large-dataset comparison
------------------------

.. important::

   In progress. The fixed-learning-rate arm is complete; the windowed-PBT arm is
   training. No endpoint of this study is reported before both arms complete and the
   preregistered analysis has run. Status verified 2026-10-19, 21:35 UTC.

**Question.** Does windowed PBT still beat a matched fixed-learning-rate population when
both train on ``train8m`` (ten times more data), and does the difference hold on the
sealed test set?

.. _large-dataset-design:

Design
~~~~~~

Preregistered before training (prose:
:repo:`configs/prepared/stage3_train8m_v1/PROTOCOL.md`; binding:
:repo:`configs/prepared/stage3_train8m_v1/protocol_v2.json`):

* training on ``train8m``, 23,921,449 jets; selection on ``val50k_tail``; sealed test
  ``test3000k``, 8,970,477 jets (:ref:`data-splits`);
* arm A fixed learning rate, arm B windowed PBT, both as in :ref:`learning-rate-policies`:
  five members each, 50 epochs (250 member-epochs per arm), identical start, seed
  schedule and accepted training rows; the policy is the only difference;
* primary endpoint: final-10 current-best, B − A; a difference below the 0.002 pp policy
  margin is reported as within selection noise;
* sealed test opened only after both arms complete and pass the validity checks; the
  evaluated checkpoints are frozen by hash beforehand;
* determinism checks: same-named members identical across arms until a member's first
  copy in arm B, and the gate's state after training epoch 1 reproduced by both arms
  (:ref:`checkpoint-identity`); a failure is reported as "determinism not confirmed" and
  qualifies, but does not invalidate, the comparison;
* every launch re-verifies the hashes of the protocol, configurations, data manifests,
  starting checkpoint and 29 execution files; every resume first checks the bundle it
  resumes from against a ledger of committed checkpoint hashes.

.. _large-dataset-status:

Execution status
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Stage
     - State
   * - Full-epoch gate
     - passed (attempt 2)
   * - Arm A, fixed learning rate
     - complete: 50 of 50 epochs; ledger, run verifier and final evaluations verified
   * - Arm B, windowed PBT
     - training since 2026-10-19; no generation completed at the time of verification
   * - Analysis, sealed test, holdout, report
     - after arm B completes

The protocol, its hash and the pinned execution files are unchanged since
preregistration; no launch has reported drift.

.. _full-epoch-gate:

Full-epoch gate
~~~~~~~~~~~~~~~

One real ``train8m`` epoch of the highest-learning-rate member must pass preregistered
checks (finite losses and metrics, step and row accounting, memory, timing, clean
termination) before the arms start. Attempt 1 failed because of a verifier defect.
Weaver's maximum gradient norm is taken over all steps, including the steps skipped by
the loss scaler for overflow (7 of 21,012); after unscaling, a skipped step's gradient is
non-finite, so the logged maximum was ``inf`` and the check rejected the epoch. The
check now accepts ``inf`` only if the number of skipped steps agrees between the log, the
run manifest and the data-audit accounting and is at least one; NaN, other non-finite
values and every other criterion fail as before, and no threshold changed
(:repo:`configs/prepared/stage3_train8m_v1/gate_revision_1.json`). Attempt 2 passed, and
its epoch reproduced attempt 1 bit for bit (model, optimizer and loss-scaler files).
Measured 2,473–2,496 s per member-epoch; projected campaign cost about 460 GPU-hours.

.. _execution-record:

Interruptions and resumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The campaign runs on shared GPUs without a batch scheduler. GPUs are leased
cooperatively, and the campaign stops its own processes (and only those) when a process
outside the campaign allocates memory on a leased GPU, so no measured epoch shares a GPU.
Three interruptions occurred:

.. list-table::
   :header-rows: 1
   :widths: 40 26 34

   * - Interruption
     - Committed at the time
     - Handling
   * - Campaign processes terminated by the host's memory-pressure service
     - arm A, 18 epochs
     - launcher changed to run outside that service's scope; epoch 19 recomputed
   * - Processes outside the campaign allocated memory on GPUs leased to arm A
     - arm A, 47 epochs
     - campaign stopped; epoch 48 recomputed
   * - A process outside the campaign allocated memory on a GPU selected for arm B
       during its initial evaluation
     - arm A complete; arm B none
     - campaign stopped; initial evaluation repeated

Each resumption continued from the last committed epoch after the ledger check
(:ref:`restart-semantics`); an interrupted epoch is recomputed from its starting bundle
with the same seed and data order, and completed epochs are not retrained. Matched
compute is unaffected: both arms consist of the same 250 member-epochs, and recomputed
work is reported separately as consumed, not effective, GPU-hours.

.. _determinism-verification:

Correction of the determinism checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two preregistered determinism checks originally compared SHA-256 hashes of the
model files. Identical states saved on different GPU indices are different files
(:ref:`checkpoint-identity`): the gate trained on one GPU, while the gate member in arm A
ran on another device index, so the gate check would have reported a mismatch for
identical states, and the lineage check would do so for members placed on different
devices in the two arms. Both checks now compare the full restart bundle in content —
model, optimizer and loss-scaler state — which also makes them stricter, since the
optimizer and loss scaler were not compared before. The correction was made after arm A
completed and before any arm-B generation, and does not depend on any PBT outcome.
Applied to the completed arm A, the gate member's bundle after training epoch 1 equals
the gate's in content although the files differ. Record:
:repo:`configs/prepared/stage3_train8m_v1/analysis_revision_1.json`.

.. _sub-epoch-adaptation:

Sub-epoch adaptation (development)
----------------------------------

A proposed scenario copies members every 0.2 epoch (five generations per epoch) and
changes learning rates on a slower cadence. Under the epoch-restarted contract it is not
testable: restarting within an epoch changes the optimizer trajectory
(:ref:`restart-semantics`; measured effect: :ref:`sub-epoch restart test
<sub-epoch-restart-test>`).

A development branch,
`full-lookahead-state <https://github.com/marchexe/ilc-dynamic-training/tree/full-lookahead-state>`_,
implements the missing parts. It is not merged and is not used by any reported result or
by the large-dataset comparison.

* A versioned checkpoint format that also stores Lookahead's slow weights, step
  counter, k and α; checkpoints without it load with the epoch-restarted behaviour.
* An epoch cursor that continues an epoch's data stream at a batch offset and restores
  the process-local random-number and preprocessing state.
* Generations shorter than an epoch, and for the cadenced policy an optional
  learning-rate interval *M*: copies may occur at every generation, while learning rates
  change only after every *M*-th generation (for example *M* = 5 with five generations per
  epoch).

Verified so far: unit tests, and with the real Weaver training loop on the CPU, one
epoch trained as three restarted segments equals one uninterrupted epoch bit for bit, and
two epochs trained as four half-epoch generations equal two one-epoch generations bit for
bit. Not yet done: an equivalence test on GPU at production scale, the merge (the branch
changes protocol-pinned execution files and is therefore merged only after the
large-dataset comparison completes, with a recorded execution revision), and any
experiment with the scenario.
