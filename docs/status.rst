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
* Offline replay of a noise-gated controller over the qualified trajectories;
  the base rule proposed 1 UP, 37 KEEP and 0 DOWN actions with no reversals or
  reference-action disagreements.

Current work
------------

The prepared next experiment is an eight-member fixed-LR shadow run. It logs
controller proposals from ``representative_60k`` but ``mode: shadow`` prevents
the only LR-application path from changing optimizer learning rates. The 150k
reference tier is observational and never enters controller decisions.

Next research steps
-------------------

1. Run the selected ``representative_60k`` signal in shadow mode without LR
   changes.
2. Review proposal stability, including DOWN behavior that the sparse
   historical replay did not exercise.
3. Specify the rule-based adaptive LR controller as a separately versioned
   policy only after replay and shadow checks pass.
4. Compare it against frozen ``windowed_pbt_v2`` and matched fixed-LR controls,
   with an independent validation tier that never drives selection.

Open questions
--------------

* Does the replay-selected trend rule remain stable at the denser shadow cadence?
* How should controller actions be gated when proxy and corroboration disagree?
* Do the late-epoch gains persist on an independent validation tier and across
  additional seeds?
