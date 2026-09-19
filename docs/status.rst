Status
======

Completed
---------

* Deterministic full-epoch training and validation audits.
* Coherent model/optimizer/AMP-scaler copy and continuation behavior.
* Frozen, replayable ``windowed_pbt_v2`` decisions.
* Verified 50- and 100-full-epoch PBT runs.
* Matched 50-epoch fixed-LR population and two 100-epoch continuation controls.
* Read-only reporting that preserves manifest hashes and continuation evidence.

Current work
------------

The current task is scientific validation design: decide how a decision-driving
proxy and an independent corroboration tier should be sampled, scheduled and
compared. The existing 50/100-epoch evidence uses the fixed control proxy only.

Next research steps
-------------------

1. Define and audit disjoint proxy/corroboration datasets and their evaluation
   cadence.
2. Measure ranking agreement and stability without allowing the corroboration
   tier to influence training decisions.
3. Specify a new rule-based adaptive LR controller as a separately versioned
   policy.
4. Compare it against frozen ``windowed_pbt_v2`` and matched fixed-LR controls.

Open questions
--------------

* How small can the decision proxy be while preserving checkpoint ranking?
* Which trend/window rules respond to real changes rather than tail noise?
* How should controller actions be gated when proxy and corroboration disagree?
* Do the late-epoch gains persist on an independent validation tier and across
  additional seeds?
