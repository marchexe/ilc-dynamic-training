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
* Completed the 96-generation ``representative_60k`` shadow audit. The run was
  intact, but only two UP proposals fired, no DOWN proposal was exercised, and
  the cumulative-factor bound clamped both proposed LR changes to no-ops.

Current work
------------

The first eight-member fixed-LR shadow run is complete. Its evidence is not
sufficient for a live adaptive-LR run: six members never proposed a change,
DOWN behavior remains untested, and ``max_cumulative_lr_factor_per_epoch: 1.0``
made the two UP proposals mechanically ineffective.

Next research steps
-------------------

1. Run a pre-registered continuation shadow with a non-zero cumulative LR
   allowance while retaining shadow mode and the qualified measurement tiers.
2. Require real DOWN coverage, non-clamped proposals and no meaningful
   opposite reference direction before considering live control.
3. Specify the rule-based adaptive LR controller as a separately versioned
   policy only after replay and shadow checks pass.
4. Compare it against frozen ``windowed_pbt_v2`` and matched fixed-LR controls,
   with an independent validation tier that never drives selection.

Open questions
--------------

* Can a continuation shadow exercise stable DOWN behavior across multiple members?
* How should controller actions be gated when proxy and corroboration disagree?
* Do the late-epoch gains persist on an independent validation tier and across
  additional seeds?
