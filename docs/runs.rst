Run inventory and future names
==============================

Download the :download:`run inventory <runs_inventory.json>` (snapshot dated
2026-09-19). Paths are repository-relative. Entries describe runs or collections,
not every checkpoint file; the most specific matching path takes precedence.
Classifications describe research relevance, not process liveness or permission
to remove data. Missing manifests are recorded as unknown, not as failed runs.

Preserve these together
-----------------------

* ``runs/pbt/windowed_pbt_v2``: completed 50-epoch reference and required ancestor.
* ``runs/pbt/windowed_pbt_v2_100epochs``: current completed 100-epoch reference.
* ``runs/pbt/foundation_correctness_20260916``: reproducibility and parity evidence.
* ``runs/pbt/foundation_fixed_lr_50epochs_20260916`` and both
  ``runs/pbt/fixed_lr_continuation_20260917_lr_*`` arms: matched fixed-LR controls.
* ``runs/eval/original_pretrained_0_19_20260917``: pretrained-checkpoint evaluation.
* ``runs/showcase/lr_mistag_correlation_matrix`` and the completed anchor-rewrite
  attempt: historical comparison evidence, distinct from the current algorithm.

Prepared configurations and derived plots are identified separately. A prepared
directory may belong to a run that has since completed; its category describes
the preparation artifacts, not an assertion that the experiment is unlaunched.

Proposed convention (future runs only)
--------------------------------------

``YYYYMMDDTHHMMSSZ_<study>_s<seed>_e<total>``

Example: ``20260919T120000Z_wpbtv2_s22345_e50``. Use a UTC creation timestamp,
a short lowercase study token (``wpbtv2`` or ``fixedlr14u``), the configured base
seed, and the target total number of full epochs. For continuation, ``e100``
means total ancestry horizon, not 100 additional epochs. The manifest records
the source and exact strategy/configuration. Do not encode GPU allocation,
mutable LR, or eventual classification in the name.

This is a proposal, not a launcher change. Existing paths stay untouched.
Archival moves, renames, deletion, and execution of prepared experiments require
separate approval; classification alone authorizes none of them.
