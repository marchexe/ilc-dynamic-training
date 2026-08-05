# Nightly plan — 50k control-proxy vs. full-validation audit

Date: 2026-08-05. Budget: ≤3h inspection/implementation, ≤8h unattended execution
(≥45min reserved for post-processing/reporting).

## Research question

Does the fixed 50,000-events/class control proxy (`val50k_tail`) reproduce
full-validation metrics and checkpoint ranking well enough to be trusted for
PBT control and intermediate checkpoint ranking?

## Current validation flow (as found)

The repo already implements a **four-tier** validation hierarchy, not the
two tiers the task assumes. Confirmed directly from
`datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json`
(row counts are per class × 3 classes):

| tier (code name) | suffix | rows/class → total events | cadence | role today |
|---|---|---|---|---|
| `control` | `val5k_tail` | 5,000 → 15,000 | every generation | **drives** controller + PBT ranking |
| `monitor` | `val50k_tail` | 50,000 → 150,000 | every 4 generations | read-only diagnostic |
| `full` | `val1000k` | ~996,708 avg → 2,990,125 | every 16 generations | read-only, **overlaps** control+monitor |
| `full_holdout` | `val_holdout` | ~941,708 avg → 2,825,125 | alongside `full` | read-only, **zero overlap** with control+monitor by construction |

All four are physically disjoint slices of one 10M-event ROOT-derived parquet
conversion (`train800k` is a separate, non-overlapping training split — no
train/validation leakage). `control`+`monitor` are the tail of `val1000k`;
`full_holdout` is everything before that tail, i.e. `val1000k` minus
`control`+`monitor`.

- Dataset selection: `scripts/validation/tail_proxy_subsets.py` — deterministic
  last-N-rows tail slicing, not random resampling. Fixed and stable across a
  run once built (built once, offline, checked into `datasets/`).
- Metrics: `scripts/training/runtime.py::read_metrics()` parses Weaver's
  `--run-mode test` stdout — background-rejection curves, working-point mistag
  %, Wilson uncertainty. Same code path for every tier.
- Only aggregated metrics are stored anywhere in this repo's active code —
  Weaver's `--predict-output` (event-level predictions) exists in the vendored
  framework but is never passed by any command builder here. Paired bootstrap
  is therefore not available; any uncertainty must be reported as unavailable,
  not fabricated.
- Correlation/ranking-agreement/corroboration diagnostics already exist:
  `scripts/training/pbt/reporting/statistics.py` (`tier_correlation`,
  `ranking_agreement`, `best_checkpoint_by_tier`, `proxy_overfitting_cases`,
  `corroboration_status`) — built for a live run's `manifest.json`, reused
  by feeding them a synthetic manifest shaped the same way.

## Current checkpoint flow

- Canonical pretrained base: `checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
  (sha256 `ae4928aa...`) — every experiment resumes from this, never random init.
- Per-run: `runs/{pbt,showcase,archive,dev}/<experiment>/<member>/net_epoch-<N>_state.pt`
  and a run-level `checkpoints/global_best_{state,optimizer}.pt` +
  `global_best_metadata.json` (member, generation, epoch, lr, metric_value).
- 8 genuinely distinct checkpoints found on disk by SHA256 (no LR sweep
  needed — see table below), spanning the pretrained base, 4
  controller-on/off arms, a BN-freeze pilot, a legacy guarded-resume run, and
  a BN-freeze-diagnostic frozen variant.

Confirmed by actually running the audit's provenance extraction
(`scripts/research/run_proxy_audit.py::provenance_for`) against every
checkpoint's real `global_best_metadata.json` / `manifest.json` — not
assumed from file-naming convention (an earlier draft of this table wrongly
assumed all `global_best` checkpoints were epoch 17; global_best is
whatever epoch/generation actually produced the best control-tier metric,
which for most of these runs is well past epoch 17):

| hash prefix | source | epoch | generation | lr |
|---|---|---|---|---|
| `ae4928aa` | pretrained base | 17 | n/a | n/a |
| `e4d35539` | `pretrained_pbt_4gpu_long_controller_active` (global_best) | 36 | 18 | 1.4e-5 |
| `ab9501a3` | `pretrained_pbt_4gpu_long_controller_off` (global_best) | 34 | 16 | 1.4e-5 |
| `e737b7cd` | `pretrained_pbt_4gpu_night_controller_off` (global_best) | 29 | 11 | 3.0e-6 |
| `a3f38002` | `pretrained_pbt_4gpu_night_controller_active` (global_best, showcase) | 85 | 67 | 7.92e-6 |
| `419ce473` | `pretrained_exploit_mutate_8gpu_10m_tiered_pilot_bnfreeze` (global_best) | 29 | 11 | 1.08e-5 |
| `4401df43` | legacy `guarded_resume_8gpu` / `smooth_lr_8gpu` (global_best) | 32 | -1 | 1.56e-5 |
| `6bb6a036` | `bn_freeze_diag_frozen_test` (global_best) | 18 | 0 | 9.0e-6 |
| `e3063887` | `pretrained_pbt_4gpu_long_controller_active`, member `lr_14e-6` | 42 | 24 | 1.4e-5 |
| `d3bc156e` | `pretrained_pbt_4gpu_long_controller_off`, member `lr_14e-6` | 43 | 25 | 1.4e-5 |
| `abc512fe` | `pretrained_pbt_4gpu_night_controller_off`, member `lr_14e-6` | 51 | 33 | 1.4e-5 |
| `ea4fa51b` | `pretrained_pbt_4gpu_night_controller_active`, member `lr_14e-6` | 107 | 89 | 7.35e-6 |

LR and generation provenance for each comes from that run's
`manifest.json`/`global_best_metadata.json`, extracted automatically (not
hand-typed) and recorded per-checkpoint in `checkpoint_metrics.csv`.
`legacy_guarded_resume_global_best`'s generation of -1 means its manifest
predates per-generation `global_best` bookkeeping (it was seeded from the
initial evaluation) — recorded as-is, not fabricated. 12 checkpoints total
(top of the task's suggested 6-12 range) — 24 more distinct checkpoints
were available (4 runs × 4 members at their latest epoch) but were not all
used, to keep the checkpoint set to a reviewable, clearly-justified size;
the 4 chosen `lr_14e-6` arms pair with their run's own global_best entry,
giving both "what PBT selected" and "a different, meaningfully-diverged
arm" per run.

## Runtime benchmark (measured, iutgpu01 GPU 0/1, pretrained-base checkpoint)

| tier | events | wall time | inference rate |
|---|---|---|---|
| control_proxy_50k (`val50k_tail`) | 150,000 | 31.5s | ~12,857 events/s |
| full_validation (`val_holdout`) | 2,825,125 | 3m43s (223s) | comparable rate, ~19x more events |

Per-checkpoint cost (both tiers): ~254.5s (~4.2 min). For 12 checkpoints:
- Fully serial on 1 GPU: 12 × 254.5s ≈ 51 minutes.
- Parallelized across iutgpu01's 8 free GPUs (confirmed free via
  `scripts/cluster/check.sh`; all other nodes were busy or lower-scored):
  24 evaluations / 8 slots ≈ 3 rounds ≈ 10-20 minutes including per-job
  startup overhead.

Both estimates are far inside the ~7h15m compute budget, so all 12
checkpoints are used and nothing needs to be trimmed. The live-PBT smoke run
(`nightly_proxy_control50k_smoke.yaml`, 2 generations, 2 members,
deliberately small `samples_per_epoch_val: 4500` for speed) completed in
1m13s, confirmed a finite `control` metric (1.1255 initial, 0.8969/0.9529
per-member), confirmed `monitor` and `full` (`val1000k`) never appear
anywhere in its manifest or log, and confirmed the controller takes only
safe `keep` actions (`flat`/`noisy` classifications) — no crash, no
unhandled error.

## NaN/Inf defense-in-depth finding

Writing the required NaN/Inf controller test surfaced a real (if
currently unreachable in production) gap: `classify_observation`
(`scripts/training/pbt/controller/decision.py`) compares a NaN metric
safely (all `<`/`>` comparisons against NaN are `False` in Python, so it
always fell through to `"flat"`/`"keep"`), but an **Inf** metric does not —
a finite baseline compared against a +inf current metric produces a
genuine `-inf` `baseline_delta`, and `-inf < -tolerance` is a real `True`,
which classified as `"unsafe"` and selected a real LR-decrease action from
a garbage signal. `finite_metric_ok` (execution/backend.py) already
prevents any non-finite metric from reaching this function in production
today, so this was not an active bug, but the task requires the controller
itself to never act on a non-finite metric. Fixed with a 6-line guard at
the top of `classify_observation`: if the current metric isn't finite,
return `"flat"` immediately, before any comparison. Covered by
`tests/test_pbt_controller.py::test_nan_or_inf_metric_never_produces_an_active_controller_action`.

## Reusable code (do not duplicate)

- `scripts/validation/evaluate_checkpoint_fixed_wp.py` — the exact
  checkpoint-vs-one-validation-suffix eval primitive (`build_test_command`,
  `read_metrics`). Reused by the new audit orchestrator instead of
  reinventing the Weaver invocation.
- `scripts/training/pbt/execution/backend.py::finite_metric_ok` — the
  existing NaN/Inf/missing-metric guard, reused for per-cell audit failure
  handling.
- `scripts/training/pbt/reporting/statistics.py` — Pearson/Spearman
  correlation, ranking agreement, best-checkpoint-by-tier, corroboration
  status. Reused as-is by feeding a synthetic manifest; only Kendall tau and
  explicit pairwise-direction agreement are new.
- `scripts/reports/plot_physics_performance.py`,
  `plot_background_efficiency_curves.py`, `write_metrics_summary.py` —
  existing per-checkpoint physics plotting/reporting, reused where the shape
  fits.

## Blocking problems

None. One compatibility constraint: the `population_lr_policy` strategy
(`configs/presets/pbt/population_lr_policy_monitor.yaml`) depends on the
`monitor` tier by name for its own (different) decisions. Nothing in this
plan touches the shared preset or that strategy preset, so it is unaffected.

## Tier-count mismatch and resolution

The task describes exactly two validation levels; the repo has four. Per
user decision: the controller's decision tier is switched from the 5k
`control` proxy to the existing 50k `monitor`-sized proxy, **scoped to a new
experiment-specific preset override only** — the shared preset
(`configs/presets/shared/pretrained_epoch17_ranger_10m_parquet_proxy_control.yaml`)
and the strategy preset (`configs/presets/pbt/exploit_mutate_significance_tiered.yaml`)
are both left unmodified, so the other 11 experiment configs that compose
them (including the in-progress night controller A/B pair) are unaffected.

Report/audit vocabulary for tonight's work:
- **`control_proxy_50k`** = code tier `control`, repointed to `val50k_tail`
  (50,000 events/class, 150,000 total) via the new override preset.
- **`full_validation`** = code tier `full_holdout` (`val_holdout`,
  2,825,125 events, disjoint from `control_proxy_50k` by construction).
- The legacy overlapping `full` tier (`val1000k`) is explicitly **not**
  scheduled in the new experiment config (its dataset/suffix are nulled in
  the override), and is never used by the standalone audit script.

## Planned minimal changes

1. `configs/presets/shared/proxy_control_50k_override.yaml` (new) — repoints
   `control` to `val50k_tail`, and critically also raises
   `samples_per_epoch_val` to 150,000 so Weaver evaluates the *entire* fixed
   file every generation rather than a random subsample of it (see below).
2. `configs/experiments/nightly_proxy_control50k_smoke.yaml` (new).
3. Two new config-resolution tests + one NaN/Inf controller
   defense-in-depth test (which found and fixed a real, if
   currently-unreachable, Inf-handling bug in `classify_observation`).
4. `scripts/research/run_proxy_audit.py`, `proxy_statistics.py` (new, thin,
   reuse-first).
5. `scripts/reports/build_proxy_audit_report.py` (new).
6. `configs/research/nightly_proxy_audit.yaml` (new).
7. `tests/research/` (new).
8. No changes to `runner.py`, `planning/`, or any shared preset/strategy
   preset file. One 8-line surgical fix in `controller/decision.py`
   (finiteness guard, see above) — everything else in `controller/` is
   unchanged.

## Config detail: samples_per_epoch_val must move with the control suffix

Weaver only evaluates an entire validation file per epoch when
`samples_per_epoch_val` is unset or `>=` the file's row count
(`weaver-core/weaver/train.py:142-148`); otherwise each epoch caps to a
random `steps_per_epoch_val = samples_per_epoch_val // batch_size_val`
subsample. The base preset sets `samples_per_epoch_val: 15000` to exactly
match `val5k_tail`'s full 15,000 events. Left unchanged after switching
`control_suffix` to `val50k_tail`, every generation would have silently
evaluated only a ~10% random slice of the new 150,000-event proxy instead
of the whole fixed set — caught before the smoke run, fixed by setting
`samples_per_epoch_val: 150000` in the same override preset.

## Estimated runtime

See the benchmark table above: ~51 minutes fully serial, ~10-20 minutes
parallelized across iutgpu01's 8 free GPUs, for all 12 checkpoints × 2
tiers. Comfortably inside the ~7h15m compute budget — no checkpoint
reduction needed.
