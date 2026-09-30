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
* Completed two matched-seed pretrained comparisons of supervisor Variant A
  (one-epoch generations, adapt every epoch) against the additional cadence5
  control (one-epoch generations, adapt every five epochs). Variant A recorded
  the lower final-10 current-best mean for both deterministic seeds.
* Implemented and tested scratch initialization semantics and completed a real
  three-generation scratch smoke. No full scratch production comparison has
  run.

Current work
------------

The :doc:`supervisor PBT milestone <pbt_interim_status>` is ready for review.
The true 0.2-epoch supervisor Variant B is blocked before implementation:
historical Ranger checkpoints restore RAdam but omit Lookahead slow weights and
the step counter, so restarting at five sub-epoch boundaries changes the
optimizer trajectory. The deterministic continuation-equivalence gate failed.

Separately, the first eight-member fixed-LR shadow run is complete. Its evidence
is not sufficient for a live adaptive-LR run: six members never proposed a
change, DOWN behavior remains untested, and
``max_cumulative_lr_factor_per_epoch: 1.0`` made the two UP proposals
mechanically ineffective.

Next research steps
-------------------

1. Design a new versioned Ranger/Lookahead checkpoint contract and prove exact
   restarted continuation before implementing supervisor Variant B.
2. Compare Variant B only with a newly matched corrected Variant A; do not use
   historical ``cadenced_pbt_v1`` as the corrected comparator.
3. Treat full scratch production runs as a separate initialization-regime
   robustness track.
4. Run a pre-registered continuation shadow with a non-zero cumulative LR
   allowance while retaining shadow mode and the qualified measurement tiers.
5. Require real DOWN coverage, non-clamped proposals and no meaningful
   opposite reference direction before considering live control.
6. Specify the rule-based adaptive LR controller as a separately versioned
   policy only after replay and shadow checks pass.
7. Compare it against frozen ``windowed_pbt_v2`` and matched fixed-LR controls,
   with an independent validation tier that never drives selection.

Open questions
--------------

* Can a continuation shadow exercise stable DOWN behavior across multiple members?
* How should controller actions be gated when proxy and corroboration disagree?
* Do the late-epoch gains persist on an independent validation tier and across
  additional seeds?
