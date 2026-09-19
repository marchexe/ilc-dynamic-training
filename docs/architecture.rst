Compatibility boundaries
========================

Condensed from ``docs/pbt_architecture_cleanup.md`` and the README's continuation
guide. Existing manifests and experiment configurations remain authoritative.

Frozen reference and member identity
------------------------------------

``scripts/training/pbt/reference/windowed_v2.py`` preserves decision semantics
from revision ``a4f750793911508812b5b09296afe5306306c930``. The legacy planning
module re-exports the functions and retains checkpoint transitions. Algorithm
changes require a separate strategy version.

``MemberState(member_id, current_lr)`` is a command-boundary snapshot. Historical
names such as ``lr_3e-6`` are opaque identities, not current learning rates.
Schema-v1 manifests retain ``name``/``lr`` and their lineage fields; the adapter's
``to_legacy()`` projection must not replace an entire manifest member record.

State and reporting ownership
-----------------------------

* Training owns ``manifest.json`` persistence. Reports consume snapshots without
  changing the caller or the persisted manifest. Report selection and artifact
  metadata live in ``summary.json`` and reporting return values.
* ``scripts/training/checkpoints.py`` owns bundle paths, copying, optional AMP
  scaler companions, hashes, and identity checks. Legacy imports remain valid.
* Model/optimizer staging precedes sequential replacement; the scaler follows.
  This historical copy order is not a filesystem transaction. Optimizer
  transformations remain in ``state/optimizer_state.py``.
* ``scripts/validation/results.py`` shares finite-metric and final-result checks;
  the standalone verifier retains its stricter evidence checks.

Completed-run continuation
--------------------------

Continue into a new run, preserving the training contract and population
identities while extending the horizon by complete windows. The continuation
configuration pins the source manifest SHA-256 and checkpoint-state digest.
Historical decisions are replayed under their original horizons.

Source runs and all ancestors are read-only dependencies. Preserve their paths,
manifests, checkpoints, optimizer states, scalers, and evidence. Never edit a
completed run's horizon or rename it to reflect a later result.

The runner and planners still use legacy dictionaries, and training event
writers remain in the reporting package. Proxy validation and a new adaptive
LR controller are deferred.
