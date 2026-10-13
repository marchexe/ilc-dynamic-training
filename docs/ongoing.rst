Ongoing Research: Large-Dataset Comparison
==========================================

.. important::

   Running. No result of this study is reported. Status checked 2026-10-13, 20:33 UTC.

**Question.** Does windowed PBT still beat a matched fixed-learning-rate population when
both train on ``train8m`` (ten times more data), and does the difference hold on the
sealed test set?

**Design**, preregistered before training (prose:
:repo:`configs/prepared/stage3_train8m_v1/PROTOCOL.md`; binding:
:repo:`configs/prepared/stage3_train8m_v1/protocol_v2.json`):

* arm A fixed learning rate, arm B windowed PBT, both as in :doc:`method`, 50 epochs each;
  identical start, population, seed schedule and training rows; the policy is the only
  difference;
* primary endpoint: final-10 current-best, B − A; a difference below the 0.002 pp policy
  margin is reported as within selection noise;
* sealed test opened only after both arms complete and pass validity checks; evaluated
  checkpoints frozen by hash beforehand;
* every launch re-verifies the hashes of the protocol, configurations, data manifests,
  starting checkpoint and 29 execution files.

**Full-epoch gate.** One real ``train8m`` epoch must pass preregistered checks before the
arms start. Attempt 1 failed on a verifier defect: Weaver's maximum gradient norm includes
the steps skipped for overflow (7 of 21,012), so it was ``inf``, which the check rejected.
The check now accepts ``inf`` only when skipped steps are independently confirmed; NaN and
other non-finite values still fail, and no threshold changed
(:repo:`configs/prepared/stage3_train8m_v1/gate_revision_1.json`). Attempt 2 passed and
reproduced attempt 1 bit for bit. Measured 2,473–2,496 s per member-epoch; projected
about 460 GPU-hours.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Stage
     - State
   * - Gate
     - passed (attempt 2)
   * - Arm A, fixed learning rate
     - running since 2026-10-13, 18:49 UTC
   * - Arm B, windowed PBT
     - waiting for five free GPUs
   * - Analysis, sealed test, report
     - after both arms complete
