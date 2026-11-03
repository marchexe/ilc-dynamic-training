# Learning-rate level follow-ups: plan and operations

## Why

On the frozen Stage 2A holdout (`runs/eval/holdout_high_lr_grid_v1/report.md`, selection frozen in
`holdout_high_lr_grid_v1/protocol.json` before evaluation):

| train800k checkpoint (seed 12345) | holdout primary (%) | vs original fixed-LR |
|---|---|---|
| fixed-LR grid 3.0e-6 to 1.4e-5 | 0.326934 | — |
| windowed PBT (LR bounds 2e-6 to 3e-5) | 0.322484 | −1.36% |
| fixed-LR grid 1.4e-5 to 3.0e-5, best 3.0e-5 | 0.314628 | −3.76% |

The higher fixed learning rate beats windowed PBT by 2.5%, with the same sign at all eight working points. On
train800k the gain of windowed PBT over the original control is explained by the learning-rate level, not by
adaptation. Both fixed grids selected their top learning rate, so the fixed-LR optimum may lie higher still.

The same confound applies to Stage 3 on train8m: its fixed-LR arm stopped at 1.4e-5 (selected member: the top
one), while windowed PBT could reach 3.0e-5 and its selected member ended there. The −7.7% on the sealed test
therefore does not show that adaptation helps on train8m.

## Plan

| # | Question | Run(s) | GPUs × time | Holdout spec |
|---|---|---|---|---|
| 2 | Where is the fixed-LR optimum on train800k? | `fixed_lr_ceiling_grid_50epochs` (3.0–8.0e-5) | 5 × ~4.7 h | `holdout_fixed_lr_ceiling_v1` |
| 3 | Does the result hold for a second seed? | `fixed_lr_high_grid_50epochs_seed22345`, `windowed_pbt_v2_seed22345` | 2 × 5 × ~4.7 h | `holdout_seed22345_v1` |
| 4 | Does adaptation help on train8m once fixed LR is high enough? | `train8m_fixed_lr_high_grid` (1.4–5.0e-5) | 5 × ~40–50 h | `holdout_train8m_fixed_lr_high_v1` |
| 5 | Sub-epoch (0.2-epoch) adaptation pilot | deferred, see below | — | — |

Every run starts from the epoch-17 checkpoint with a raw optimizer and changes only what its question needs
(learning rates or seed; checked by `tests/test_train800k_followups.py`). Item 2 and item 4 each repeat one
member of an earlier run (`lr_30e-6` of the extended grid; `lr_14e-6` of Stage 3 arm A), so
`reproduction_checks.py anchor` shows whether the comparison with that run is exact.

Each holdout spec is committed before its runs train. After a run completes, `--freeze` records its
validation-selected checkpoint (the Stage 2A rule); the holdout never selects. References are re-evaluated and
must reproduce their earlier holdout metric bit for bit. The Stage 3 sealed test is not used: it was opened once
for the Stage 3 comparison, and any further use would need its own decision.

Order: items 2 and 3 are cheap and independent; start them first (sequentially, one launch at a time, at least a
minute apart, so two launches cannot pick the same idle GPU). Item 4 is the decisive and expensive one. Its grid
(1.4–5.0e-5) can still be changed before it starts, if item 2 puts the train800k optimum well above 5e-5; change
the configuration and its spec in one commit before launching.

### Item 5: why the 0.2-epoch pilot is deferred

The pilot asks whether adapting more often than once per epoch helps. It costs about three times an ordinary run
(the GPU equivalence test measured 3.06× per epoch: each restart re-reads the skipped batches, starts Weaver and
validates). Until items 2–4 show that adaptation helps at all once the fixed learning rate is tuned, its result
could not be interpreted. Reconsider it if windowed PBT beats the tuned fixed-LR grid on train8m (item 4), and
first make the data cursor skip batches without preprocessing them.

## Commands (from the repository root on the login node)

```bash
py=.venv/bin/python
# Training: one node with one idle GPU per member. Status, resume and stop work the same way.
$py scripts/experiments/train800k.py start  --config configs/experiments/fixed_lr_ceiling_grid_50epochs.yaml
$py scripts/experiments/train800k.py start  --config configs/experiments/fixed_lr_high_grid_50epochs_seed22345.yaml
$py scripts/experiments/train800k.py start  --config configs/experiments/windowed_pbt_v2_seed22345.yaml
$py scripts/experiments/train800k.py start  --config configs/experiments/train8m_fixed_lr_high_grid.yaml
$py scripts/experiments/train800k.py status --config configs/experiments/<config>.yaml

# Anchors, after the run completes:
$py scripts/experiments/reproduction_checks.py anchor --run fixed_lr_ceiling_grid_50epochs \
    --control fixed_lr_high_grid_50epochs --member lr_30e-6
$py scripts/experiments/reproduction_checks.py anchor --run train8m_fixed_lr_high_grid_v1 \
    --control stage3_train8m_fixed_lr_v1 --member lr_14e-6

# Holdout, after all runs of a spec complete (3-4 GPUs, 1-2 hours each):
p=configs/prepared/holdout_fixed_lr_ceiling_v1   # or holdout_seed22345_v1, holdout_train8m_fixed_lr_high_v1
$py scripts/experiments/holdout_addendum.py --prepared $p --freeze
git add $p/protocol.json $p/protocol.sha256 && git commit -m "Freeze the $(basename $p) selection" && git push
$py scripts/experiments/holdout_addendum.py --prepared $p --dry-run
$py scripts/experiments/holdout_addendum.py --prepared $p
$py scripts/experiments/holdout_addendum.py --prepared $p --status
```

`train800k.py` is not specific to train800k: it places any population run. The train8m run needs the converted
train8m dataset of Stage 3 and about 2 GiB of disk per member. Do not type Ctrl-C into an attached session; use
`--stop` (holdout) or `train800k.py stop` (training).

## Reading the results

* Item 2: if a learning rate above 3.0e-5 is selected and is better on the holdout, the fixed-LR optimum is higher
  and the gap to windowed PBT on train800k is larger than 2.5%.
* Item 3: the seed-22345 pair should show the same sign as the seed-12345 pair. If it flips, the train800k
  comparison is within seed-to-seed variation and needs more seeds before any conclusion.
* Item 4: if the high fixed-LR grid matches or beats windowed PBT (holdout 0.235949), the Stage 3 gain is also a
  learning-rate-level effect. If windowed PBT stays clearly better, adaptation helps on train8m, and the
  sub-epoch pilot (item 5) becomes worth its cost.
* There are no confidence intervals (jets of one physical event cannot be linked). Use the delta, the sign across
  the eight working points and, for item 3, the agreement between seeds.
