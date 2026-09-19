# Compatibility boundaries for PBT

PBT is the primary method; fixed-LR runs are controls. This cleanup adds no
algorithm, controller, validation definition, or experiment configuration.

## Frozen reference

`scripts/training/pbt/reference/windowed_v2.py` contains the reference decision
functions from revision `a4f750793911508812b5b09296afe5306306c930`, without changes
to scoring, ties, mutations, counters, or terminal decisions. The old
`planning/windowed_pbt_v2.py` import path re-exports them and retains checkpoint
transitions. Future algorithm changes need a separate strategy version.

## Identity and mutable learning rate

`scripts/training/members.py` defines `MemberState(member_id, current_lr)` for
command construction. A historical name such as `lr_3e-6` is an opaque identity,
even when the current LR is `1.344e-5`. Local and Ray command builders accept
either this snapshot or a legacy record. The schema-v1 manifest still stores
`name`/`lr`, plus all lineage and extension fields. The adapter's `to_legacy()`
projects only identity/LR; it must not replace a full manifest record.

## Reporting ownership

Training code owns persistence of `manifest.json`. `write_canonical_outputs()`
accepts an in-memory snapshot, writes derived artifacts, and returns their
metadata. It neither creates nor rewrites the training manifest and leaves the
caller's nested data unchanged, including when a renderer fails.

Report selection metadata and artifact metadata are stored in `summary.json`
under `checkpoint_selection` and `canonical_artifacts`. The legacy manifest
fields remain readable and are preserved if present, but reports no longer
refresh them. Callers that previously inspected newly added manifest fields
should use the function return value or summary instead. Markdown renderers use
a private enriched view so existing plot details and warnings remain available.

The two physics plot APIs retain their path-based interface and also accept a
keyword-only `manifest` snapshot. Standalone checkpoint report generation also
leaves its source manifest untouched. Regeneration therefore cannot invalidate
a continuation's pinned source-manifest hash.

## Shared operations

`scripts/training/checkpoints.py` owns file copying, optional scaler companions,
bundle paths, hashes, and identity checks. Old PBT imports remain compatibility
exports. Optimizer transformations remain in `state/optimizer_state.py`.
Copy order, temporary suffixes, stale-scaler removal, and hash checks are
unchanged. Staging precedes model/optimizer replacement; replacements are
sequential and the scaler follows afterward. This is not a filesystem
transaction; changing crash semantics belongs in a separately reviewed slice.

`scripts/validation/results.py` owns the existing finite-metric acceptance and
final checkpoint-result checks. Historical caller acceptance policies and the
standalone verifier's stricter evidence checks remain unchanged.

## Remaining work and next slice

The runner and planners still use legacy dictionaries internally. Training
event writers also remain in the `reporting` package even though they belong
to state persistence. The next safe slice is to move those event writers behind
a state-owned module with compatibility exports and unchanged event payloads.
Avoid combining that move with algorithm or resume changes. Proxy validation
and a new adaptive LR controller remain deferred.

## Files in this slice

- `docs/pbt_architecture_cleanup.md`
- `scripts/reports/plot_background_efficiency_curves.py`
- `scripts/reports/plot_physics_performance.py`
- `scripts/training/checkpoints.py`
- `scripts/training/members.py`
- `scripts/training/pbt/execution/backend.py`
- `scripts/training/pbt/execution/weaver_command.py`
- `scripts/training/pbt/planning/windowed_pbt_v2.py`
- `scripts/training/pbt/reference/__init__.py`
- `scripts/training/pbt/reference/windowed_v2.py`
- `scripts/training/pbt/reporting/canonical.py`
- `scripts/training/pbt/reporting/markdown_report.py`
- `scripts/training/pbt/reporting/plots.py`
- `scripts/training/pbt/runner.py`
- `scripts/training/pbt/state/checkpointing.py`
- `scripts/training/pbt/state/continuation.py`
- `scripts/training/pbt/state/optimizer_state.py`
- `scripts/validation/evaluate_checkpoint_fixed_wp.py`
- `scripts/validation/results.py`
- `tests/test_pbt_artifacts.py`
- `tests/test_training_compatibility.py`

## Verification on 2026-09-19

- `.venv/bin/python -m unittest discover -s tests -t .`: 447 tests,
  successful, one skip (Ray is installed, so its missing-dependency test skips).
- Targeted compatibility/artifacts/windowed-PBT/continuation run: 42 tests
  passed. The completed compatibility module, including standalone checkpoint
  reporting, then passed all 9 tests and is included in the final full suite.
- `scripts/validation/verify_fixed_lr.py runs/pbt/windowed_pbt_v2`: passed
  through 50 full epochs, with no failures.
- `scripts/validation/verify_fixed_lr.py runs/pbt/windowed_pbt_v2_100epochs`:
  passed through 100 full epochs, with no failures.
- Both verifier JSON results were identical before and after cleanup. Original
  manifest SHA-256 values and modification timestamps were also unchanged.
- Reports rebuilt successfully from temporary copies of both completed runs;
  input dictionaries and copied manifest bytes/timestamps stayed unchanged.
- AST comparison against the pinned reference confirmed all three decision
  functions and both checkpoint-transition functions are identical.
- `git diff --check`: passed. No experiment configuration or historical run
  files changed. No training was launched.

The initial test-discovery command omitted `-t .` and encountered four import
errors because `tests/research` shadowed `scripts/research`. Specifying the
repository top level resolved the collision; no source fix was needed.
