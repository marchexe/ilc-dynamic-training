Research changelog
==================

2026-09-21
   Audited the completed 96-generation ``representative_60k`` shadow run.
   Recorded two clamped UP proposals, 766 KEEP decisions and no DOWN coverage;
   deferred any live adaptive-LR experiment pending a corrected continuation
   shadow.

2026-09-20
   Completed the offline proxy-validation qualification: 342 cached evaluations
   across 38 checkpoints and nine candidates, selecting ``representative_60k``
   for shadow adaptive-LR testing.

   Added deterministic offline controller replay and a log-only fixed-LR
   shadow configuration. The base replay produced 1 UP, 37 KEEP and 0 DOWN
   proposals with zero reversals and zero reference-action disagreements.

2026-09-19
   Completed and verified the 100-full-epoch ``windowed_pbt_v2`` continuation;
   produced the matched late-epoch fixed-LR comparison.

2026-09-18
   Added the frozen ``windowed_pbt_v2`` reference strategy and completed its
   50-full-epoch run.

2026-09-17
   Completed two fixed-LR continuations to 100 full epochs and evaluated the
   original pretrained epoch series.

2026-09-16
   Verified deterministic full-epoch traversal and exact model, optimizer and
   scaler parity; completed the 50-epoch fixed-LR control.

2026-08-31
   Completed the historical 100-generation ``anchor_copy_lr_recenter`` v2
   validation after reducing recenter momentum.

2026-08-20
   Completed the historical LR/mistag correlation matrix across seeds,
   validation sizes and generation lengths.

2026-08-04
   Established tiered validation/reporting and fixed LR ownership across PBT
   checkpoint copies and the dynamic controller.
