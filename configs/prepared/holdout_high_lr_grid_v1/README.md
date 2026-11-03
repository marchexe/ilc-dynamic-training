# Extended fixed-LR grid and full-state GPU test: operations

Two train800k follow-ups to the large-dataset comparison.

* **Extended fixed-LR grid** (`configs/experiments/fixed_lr_high_grid_50epochs.yaml`). In the large-dataset
  comparison the best fixed-LR member was the top of the grid (1.4e-5), while windowed PBT ended at
  1.9–3.0e-5. This run repeats the 50-epoch train800k control with learning rates 1.4, 1.8, 2.2, 2.6 and
  3.0e-5; everything else (epoch-17 start, raw optimizer, seed 12345, data, Weaver command) is unchanged. Its
  validation-selected checkpoint is then compared on the frozen Stage 2A holdout with the windowed-PBT and
  fixed-LR checkpoints evaluated there (`spec.json`, fixed before the run). If it matches windowed PBT, the
  gain is explained by the learning-rate level; if not, by adaptation.
* **Full-state GPU equivalence** (`full_state_gpu_one_epoch.yaml`, `full_state_gpu_fifths.yaml`). One epoch of
  member lr_14e-6 under the full Lookahead contract, trained uninterrupted and as five restarted generations
  (420 × 4 + 422 batches). Verified so far on CPU only; the 0.2-epoch scenario depends on it holding on GPU.

## Commands (from the repository root on the login node)

`scripts/experiments/train800k.py` picks one node with one idle GPU per member, all of one model, starts the run
there with `scripts/launch/experiment.py` (which checks the GPUs again immediately before launching), and stops it
if the runner landed under systemd-oomd management. `--dry-run` only probes and plans.

```bash
py=.venv/bin/python
# 1. Full-state test, about 20 minutes on one GPU. Start the second run after the first has finished:
#    a runner takes its GPU only when its worker starts, so two launches within a minute can pick the same GPU.
$py scripts/experiments/train800k.py start  --config configs/experiments/full_state_gpu_one_epoch.yaml
$py scripts/experiments/train800k.py status --config configs/experiments/full_state_gpu_one_epoch.yaml
$py scripts/experiments/train800k.py start  --config configs/experiments/full_state_gpu_fifths.yaml
$py scripts/experiments/reproduction_checks.py full-state      # exit 1 if not bit-identical

# 2. Extended grid, five GPUs, about 3.5 hours (start once step 1 holds no GPU).
$py scripts/experiments/train800k.py start  --config configs/experiments/fixed_lr_high_grid_50epochs.yaml
$py scripts/experiments/train800k.py status --config configs/experiments/fixed_lr_high_grid_50epochs.yaml
$py scripts/experiments/reproduction_checks.py anchor          # lr_14e-6 against the original control

# 3. After the grid has completed: freeze the selection, then evaluate on the holdout (3 GPUs, about 25 minutes).
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_high_lr_grid_v1 --freeze
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_high_lr_grid_v1 --dry-run
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_high_lr_grid_v1
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_high_lr_grid_v1 --status
# After a failed or interrupted launch (the plain launch refuses then):
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_high_lr_grid_v1 --retry
# To stop a running launch deliberately:
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_high_lr_grid_v1 --stop
```

Follow the holdout evaluation with `--status` (batches done per checkpoint out of `expected_batches`). The
`attach` command in `launch.json` opens the tmux session read-only; detach with Ctrl-b d. The first launch
(2026-11-03) was stopped after 26 minutes by Ctrl-C typed into the attached pane, which reached the supervisor as
SIGINT. The supervisor now ignores SIGINT and logs it; SIGTERM and SIGHUP (`--stop`) still stop it, and the
signals it received are recorded in `status.json`. `--retry` copies the earlier launch records to
`failed_launches/NNN`; each evaluation attempt keeps its own `evaluations/<id>/attempt-NNN` logs, and completed
evaluations are not repeated. The 14,400 s evaluation timeout is part of the frozen protocol and unchanged.

`resume` and `stop` work like `status`. Results: `runs/eval/reproduction_checks/*.json` and
`runs/eval/holdout_high_lr_grid_v1/report.md`.

## What the checks mean

* `reproduction_checks.py full-state` requires the same restart bundle (model, RAdam and Lookahead state, AMP
  scaler) after the uninterrupted and the restarted epoch, slices that partition the epoch, and the same
  validation metric. It also reports the restart cost (`restarted_over_uninterrupted`): each generation starts
  Weaver and validates once, which the 0.2-epoch scenario pays five times per epoch.
* `reproduction_checks.py anchor` is descriptive. The original control ran on A100 40GB SXM4 GPUs; the extended
  grid will probably run on A100 80GB PCIe GPUs. If lr_14e-6 is not reproduced bit for bit, the per-epoch metric
  deltas show the run-to-run variation that the comparison with the original runs carries.
* The holdout addendum re-evaluates the two Stage 2A checkpoints it compares against. The report states whether
  they reproduce their Stage 2A holdout metrics bit for bit. The holdout never selects a checkpoint.
  Differences from the frozen Stage 2A launcher, which is unchanged: parallelism equals the idle GPUs found on the
  chosen node (at most one per checkpoint), and the tmux session starts outside systemd-oomd management.
