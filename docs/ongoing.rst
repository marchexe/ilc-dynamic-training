Ongoing Research: Large-Dataset Comparison
==========================================

.. important::

   This study is **running**. No validation or test result from it is reported, and
   none of the findings in :doc:`results` depend on it. Status verified on
   2026-10-13 at 19:23 UTC.

Question
--------

Does the frozen windowed-PBT policy still improve on a matched fixed-learning-rate
population when both are trained on a training set ten times larger (``train8m``,
23,921,449 jets), and does any difference hold on a sealed test set that was never
inspected?

The study addresses three limitations of the completed experiments
(:doc:`results`): it trains on more data, reserves a sealed test set, and fixes every
analysis decision before training.

Preregistered design
--------------------

The protocol was frozen before any training of this study.
``configs/prepared/stage3_train8m_v1/PROTOCOL.md`` describes it in prose; the binding
machine-readable version is ``protocol_v2.json`` (SHA256 ``b696ff63…``), an
administrative revision of ``protocol.json`` that changed only portable path records
and left the scientific sections verbatim. Every launch and resume recomputes the
identities of 29 execution files, all configurations, the data manifests and the
starting checkpoint, and refuses to run on any change.

* **Arm A:** a fixed learning rate per member.
* **Arm B:** windowed PBT, frozen as described in :doc:`method`.
* **Shared by both arms:** the pretrained epoch-17 start with its optimizer state;
  five members with the initial learning rates of :doc:`method`; the same seed
  schedule, so all ten workers see identical training rows in every epoch; full
  epochs over ``train8m``; selection on the 150,000-jet validation set; 50 epochs.
  The only intended difference is arm B's copy and mutation behaviour.

**Horizon.** Fixed at 50 epochs for both arms. A shorter run continued later would
not be equivalent for windowed PBT, because the policy treats the last boundary of a
run as terminal and makes no copy there.

**Primary endpoint.** Final-10 current-best validation composite mistag (epochs
41–50) per arm, reported as the difference B − A in pp and in percent. A difference
below the policy's 0.002 pp decision margin is reported as within the selection-noise
margin. Arm B's validation values are optimistic because the same set drives its
decisions.

**Sealed test.** ``test3000k`` (8,970,477 jets), isolated from all training and
validation data and from the pretraining sample. It is opened only after both arms
are complete and valid. For each arm, the validation global best and all five final
members are frozen with hashes before inference. The comparisons are the selected
checkpoint, the final best member, the final members paired by identity, and the
final population mean, each for the composite metric and all eight working points.
The previously inspected holdout is evaluated descriptively for the two selected
checkpoints only.

**Validity.** A comparison is reported only if both arms complete 50 epochs, pass the
run verifier, consume identical training rows in every epoch, start from identical
initial evaluations, end with complete finite evaluations, and match their checkpoint
ledgers. Otherwise the outcome is reported as invalid or incomplete and the sealed
test stays closed.

**Stated in advance.** One seed schedule, so no significance claim; no event-level
intervals; epochs on ``train8m`` are not comparable units to epochs on the smaller
training set.

Full-epoch gate
---------------

Before the arms may start, one real, uncapped ``train8m`` epoch of the highest
learning-rate member must pass preregistered checks: full traversal of every row,
the expected number of accepted rows, optimizer-step accounting, a bounded fraction
of mixed-precision steps skipped for gradient overflow (at most 1%), finite metrics
and model state, memory, throughput, and a projected campaign cost within budget.

* **Attempt 1 failed** on one check, finite gradient norm. Weaver logs the epoch
  maximum gradient norm over all steps, including the 7 of 21,012 steps that the
  loss scaler skipped because their gradients overflowed. On such a step the norm
  is infinite by construction, so the logged maximum was ``inf``, which the metric
  parser does not record. All other checks passed, and the loss, validation metrics
  and model state were finite. The same pattern occurs in 1,369 accepted epochs of
  the earlier experiments, always together with skipped steps.
* **Correction.** The gate check was corrected to read the logged value directly and
  to accept an infinite maximum only when the log, the run record and the data-loading
  audit agree on at least one skipped step and the step-accounting and skipped-step
  checks pass. NaN and every other non-finite or missing value still fail, and no
  threshold was changed. The correction, the hash of the failed verdict and the
  reasoning are recorded in ``configs/prepared/stage3_train8m_v1/gate_revision_1.json``.
  The protocol and its pinned files were not modified. The failed attempt is
  preserved.
* **Attempt 2 passed** all 19 checks. It reproduced attempt 1 exactly: identical
  model state after the epoch (SHA256 ``ffba7672…``), training loss, validation
  metric and 7 skipped steps.

Measured cost
-------------

The gate epoch took 2,473–2,496 s per member-epoch (9,150–9,190 training jets per
second). Extrapolated to 500 member-epochs with a safety factor of 1.3, the campaign
needs about 460 GPU-hours, within the preregistered budget of 800. With both arms
running concurrently this is about two days of wall time, or about four days if the
arms run one after the other. These figures are projections, not measurements of the
campaign.

Status
------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Stage
     - State
     - Note
   * - Protocol and inputs
     - frozen
     - no drift at launch
   * - Sealed test conversion and isolation
     - verified
     - not opened
   * - Full-epoch gate
     - passed (attempt 2)
     - attempt 1 failed; corrected verifier
   * - Arm A (fixed learning rate)
     - running
     - started 2026-10-13, 18:49 UTC
   * - Arm B (windowed PBT)
     - waiting
     - starts when five GPUs are free; the arms need not run concurrently
   * - Analysis, sealed test, report
     - not started
     - only after both arms are complete and valid

Interim snapshots of the arms may be produced for monitoring. They are descriptive
and are not published as results.
