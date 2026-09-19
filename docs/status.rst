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
* Offline qualification of nine proxy designs over 38 SHA-deduplicated
  checkpoints; ``representative_60k`` selected for shadow testing.

Current work
------------

The current task is offline controller replay: test smoothing, short-window
trend, patience and cooldown rules against recorded measurements before any
policy is allowed to change a live learning rate. The qualified proxy is a
measurement input, not evidence of controller performance.

Next research steps
-------------------

1. Replay candidate controller rules offline over recorded trajectories.
2. Run the selected ``representative_60k`` signal in shadow mode without LR
   changes.
3. Specify the rule-based adaptive LR controller as a separately versioned
   policy only after replay and shadow checks pass.
4. Compare it against frozen ``windowed_pbt_v2`` and matched fixed-LR controls,
   with an independent validation tier that never drives selection.

Open questions
--------------

* Which trend/window rules respond to real changes rather than tail noise?
* How should controller actions be gated when proxy and corroboration disagree?
* Do the late-epoch gains persist on an independent validation tier and across
  additional seeds?
