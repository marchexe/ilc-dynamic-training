# Prepared fixed-LR baseline — NOT LAUNCHED

Exact source config: `configs/experiments/foundation_fixed_lr_20epochs.yaml` (self-contained; no preset inheritance).
Resolved runtime config: `resolved_config.yaml`. All 100 future worker commands: `commands.json`.
Preflight result and initial input hashes: `preflight.json`. Verification source: `preflight_check.py`.
The foundation implementation is unchanged. Production now uses raw epoch-17 optimizer continuation; the historical correctness pilot used damping 0.25.

## Contract

- Five independent arms: [3.0e-6, 5.75e-6, 8.5e-6, 11.25e-6, 1.4e-5].
- Same pretrained epoch-17 model and untouched source optimizer in every arm (`initial_optimizer_mode: raw`). Five temporary bootstraps matched the original model and optimizer byte-for-byte and tensor-for-tensor, including moments, counters, and Ranger slow buffers. The damping field was removed from the source config; a schema default may remain in resolved metadata but raw mode does not apply it. Per-arm LR is the intentional optimizer parameter-group difference on load. Initial scaler is fresh/common; subsequent scaler state resumes from each arm's checkpoint.
- 20 additional full finite training epochs; checkpoint epoch numbers 18–37. Each epoch traverses 2,392,232 raw source rows once; stochastic downsampling accepts approximately 2.15M rows. Final partial batch is retained.
- Seed = 12345 + generation (0–19), identical for all arms; dataset seed additionally depends on checkpoint epoch and worker ID through the unchanged iterator. One subprocess per full epoch, using that arm's immediately preceding checkpoint. Same schedule as the pilot; no mid-epoch cursor continuation is claimed.
- Fixed val50k_tail: 150,000 total events, deterministic full evaluation after every epoch. Initial standalone evaluation plus final evaluations of five final arms and the selected best are enabled. No additional validation tiers.
- Ranger, batch 1024, FP16 AMP, frozen BatchNorm, one data-loader worker, prefetch factor 4, fetch step 0.01.

## Adaptive-path inspection

- Strategy dispatch chooses fixed_lr_grid_plan, which returns an empty exploit plan. Existing ranking_and_plan computes candidate plans internally, but fixed_lr_grid discards them before they reach execution.
- exploit_interval_generations=21 makes should_apply_exploit_for_strategy false for all configured generations 0–19. Final-generation execution is also suppressed. The interval is additional protection; the empty fixed-grid plan remains the main contract.
- run_generation_controller exits for mode=disabled; apply_controller_actions_to_members has the same inactive guard. No training_controller is passed to Weaver.
- No lr_radius, lr_controller, population_lr_policy, anchor_copy_lr_recenter, or tiered_validation policy. No anchor planner is dispatched, so recenter, rewind, and plateau escape are unreachable.
- rollback_fraction=0 and fixed-grid excludes population rollback; baseline_guard_action=observe; early_stop_degraded_generations=0.
- Weaver receives --lr-scheduler none; optimizer_setup returns no scheduler. Every generation reloads only its own prior model/optimizer/scaler and reapplies its original configured LR.
- Reporting still ranks arms and archives a global-best checkpoint. Those copies never replace an arm's training state.
- Verification exercised the real generation-decision path for all 20 generations with deliberately degraded metrics, retaining the real planner/controller gates while substituting reporting/health bookkeeping. All plans remained empty, all LRs unchanged, early stopping false. Inspected all 100 generated commands; all passed.

## Proposed resources

Host: iutgpu01. Allocation in population order: GPU 0 → 3e-6; GPU 1 → 5.75e-6; GPU 2 → 8.5e-6; GPU 3 → 11.25e-6; GPU 4 → 1.4e-5. Availability is checked by the user-run smoke launcher; no GPU is reserved.

Existing full-epoch pilot measurements (two concurrent arms, not the old short generations):

| Arm / checkpoint | Training seconds | Validation seconds | Whole subprocess seconds |
|---|---:|---:|---:|
| identical_a / 18 | 172.236 | 35.669 | 250.875 |
| identical_b / 18 | 173.673 | 35.387 | 251.378 |
| identical_a / 19 | 168.725 | 38.722 | 237.846 |
| identical_b / 19 | 172.538 | 35.692 | 238.375 |

Each traversed 2,392,232 source rows. These are provisional reference measurements, not a five-way 20-epoch runtime commitment. Recalculate definitive wall time after the user-run five-arm smoke; the earlier 90–120 minute estimate is withdrawn. Pilot peak allocation was 21,539.7 MiB per 40-GiB A100; the launcher requires at least 26 GiB free on every target GPU.

Measured checkpoint bundle: 10.236 MiB per epoch (model + optimizer + scaler). Retain all 20 epoch bundles plus initial state and per-arm best aliases: approximately 225 MiB/arm, 1.11 GiB aggregate including the global-best bundle. Logs, JSON audits/manifests, CSVs and plots add tens to hundreds of MiB. Reserve 2–3 GiB output space. Existing dataset is shared, not copied; no prediction arrays are requested by the training/final-evaluation commands. Filesystem currently has approximately 16 TiB available.

## Launch commands (not executed)

```sh
ssh iutgpu01 nvidia-smi
bash configs/prepared/foundation_fixed_lr_20epochs_20260916/launch.sh
```

Launch script contains the exact nohup command and writes launcher log/PID under logs/experiments. Output root will be `runs/pbt/foundation_fixed_lr_20epochs_20260916`, with separate lr_3e-6, lr_5_75e-6, lr_8_5e-6, lr_11_25e-6 and lr_14e-6 directories. Actual experiment output directory is absent; no worker or launcher was started.

Remaining concerns: GPU availability can change before launch; runtime estimate is extrapolated from two-way pilot concurrency. Retain the per-epoch data/AMP audit to check identical schedules and actual applied optimizer steps as the different LRs diverge. No adaptive-execution blocker remains.

## User-run full-data smoke (prepared, not executed)

Config: `configs/experiments/foundation_fixed_lr_smoke.yaml`, inheriting production with only one generation and separate output paths. It does not use the runner's reduced-data `--smoke` flag. Initial/final standalone evaluations remain enabled.

Run directly on iutgpu01 from the repository:

```sh
.venv/bin/python scripts/launch/experiment.py start --config configs/experiments/foundation_fixed_lr_smoke.yaml
.venv/bin/python scripts/launch/experiment.py status --config configs/experiments/foundation_fixed_lr_smoke.yaml
.venv/bin/python scripts/launch/experiment.py stop --config configs/experiments/foundation_fixed_lr_smoke.yaml
.venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/foundation_fixed_lr_smoke
```

Current reusable smoke output: `runs/pbt/foundation_fixed_lr_smoke/`; launcher log and identity files: `logs/experiments/foundation_fixed_lr_smoke/`. Worker logs and the manifest live in the run directory. The stop tool matches an inherited per-launch token and the current UID and uses pidfds.

Historical completed smoke artifacts remain at `runs/pbt/foundation_fixed_lr_smoke_20260916/study/`; pass that path to the generic verifier to inspect them. Historical preflight reports below describe the original preparation.

Raw preflight passed: production 100 commands/20 no-action decisions/five exact source bootstraps; smoke five commands/one no-action decision/five exact source bootstraps. Smoke preparation preflight evidence is under `smoke/`. No training, smoke, or checkpoint inference was executed during preparation.
