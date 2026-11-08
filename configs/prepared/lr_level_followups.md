# Learning-rate level follow-ups: results, plan and operations

## Terms

* **Member**: one model in a population run. It trains on one GPU with its own learning rate (LR).
* **Fixed-LR grid**: five members. Each member keeps one constant LR for all 50 epochs.
* **Windowed PBT**: five members. At the end of a window (a set of epochs), a member that loses for two windows
  copies the weights of the best member and multiplies its LR by a mutation factor.
* **Validation set**: 150,000 jets. The runs use it after each epoch. It selects the best epoch and member.
* **Holdout set**: 2,825,125 jets (the frozen Stage 2A holdout). Nothing is selected on it. Each selected
  checkpoint is evaluated on it once, for the final comparison.
* **Metric**: the primary metric. It is the geometric mean of the mistag rates (%) at the eight reference
  working points. Lower is better.
* **Anchor**: a member that repeats a member of an earlier run. If the two are bit-identical, the comparison
  between the runs is exact.

## Results on train800k (seed 12345, start at pretrained epoch 17, 50 epochs)

All selections were frozen and committed before the holdout evaluation. Every anchor was bit-identical over all
50 epochs. Every reference reproduced its earlier holdout metric bit for bit.

| Run | Configuration | LRs | Selected | Validation | Holdout |
|---|---|---|---|---|---|
| original fixed-LR grid | `foundation_fixed_lr_50epochs` | 3e-6 to 1.4e-5 | 1.4e-5 | 0.3446 | 0.3269 |
| original windowed PBT | `windowed_pbt_v2` | start 3e-6 to 1.4e-5, ×0.8/×1.2, 5-epoch windows, cap 3e-5 | ~2e-5 | 0.3377 | 0.3225 |
| extended grid | `fixed_lr_high_grid_50epochs` | 1.4e-5 to 3e-5 | 3e-5 | 0.3303 | 0.3146 |
| ceiling grid | `fixed_lr_ceiling_grid_50epochs` | 3e-5 to 8e-5 | 8e-5 | 0.3167 | 0.3018 |
| top grid | `fixed_lr_top_grid_50epochs` | 8e-5 to 3.2e-4 | 1.2e-4 | 0.3142 | 0.3033 |
| fast windowed PBT | `windowed_pbt_v2_fast` | start 1e-5 to 3.5e-4, ×0.5/×2, 2-epoch windows, cap 5e-4 | 8.75e-5 | 0.3156 | **0.3003** |

Reports: `runs/eval/holdout_high_lr_grid_v1`, `holdout_fixed_lr_ceiling_v1`, `holdout_fast_pbt_v1`.

* The original PBT lost because it could not reach the useful LRs. It made four decisions in 50 epochs, each
  changed the LR by 20% or less, and the cap was 3e-5.
* The fixed-LR optimum is near 8e-5 to 1.2e-4. Above it, members reach their best epoch early and then degrade.
* The fast PBT is 1.0% better than the top grid (the primary comparison), with fewer background jets at six of
  eight working points. It is 0.5% better than the ceiling grid. It increased the low LRs early and decreased
  all LRs to 0.5e-5–2e-5 near the end.
* Selection noise is about 0.5%: the top grid was better than the ceiling grid on validation but worse on the
  holdout. With one seed, read the fast PBT result as "equal to the best fixed LR, possibly better".

## Plan

Run one five-GPU job at a time.

| Step | Runs | GPUs × time | Holdout spec | Question |
|---|---|---|---|---|
| 1 | `windowed_pbt_v2_fast_seed22345`, then `fixed_lr_best_grid_50epochs_seed22345` (6e-5 to 1.6e-4) | 2 × 5 × ~5 h | `holdout_fast_pbt_seed22345_v1` | Does the fast PBT result hold with seed 22345? |
| 2 | `train8m_windowed_pbt_v2_fast`, then `train8m_fixed_lr_high_grid` (1.4e-5 to 2.4e-4) | 2 × 5 × ~40–50 h | `holdout_train8m_fast_pbt_v1` | Does it hold on train8m? |

Step 1b, after step 1: `windowed_pbt_v2_fast_fallback` (seed 12345, 5 × ~5 h). In the fast PBT a selected loser
was skipped whenever both mutation factors gave an LR that another member already had (reason
`no_distinct_mutation`; about 24 times in seed 12345, 20 in seed 22345; one member stayed at 5e-4 with the worst
metric from full epoch 5 to 36). The run is `windowed_pbt_v2_fast` plus `fallback_factors: [0.7, 1.4]`: such a
loser still copies the donor and gets the fallback factor that keeps its LR farthest from the others. Without the
key the decisions are the frozen reference decisions (Stage 3 `execution_revision_2.json`). Its holdout spec is
prepared after step 1; if it is not worse, step 2 uses it.

* Step 1 changes only the seed. The fixed grid covers the LRs that the seed-12345 grids selected.
* Step 2 uses the Stage 3 training input and contract. The train8m fast PBT is `windowed_pbt_v2_fast` without
  changes. The train8m fixed grid has `lr_14e-6` as its anchor to Stage 3 arm A. The Stage 3 sealed test is
  not used.
* The sub-epoch (0.2-epoch) pilot stays deferred. The 2-epoch windows already make PBT fast enough to match the
  best fixed LR.
* A member with a non-finite metric stops its run. The pretrained schedule started at 1e-3, so all LRs here are
  below the LRs that the model already trained at.

## Commands (from the repository root on the login node)

```bash
py=.venv/bin/python
# Training: one node with one idle GPU per member. status, resume and stop work the same way.
$py scripts/experiments/train800k.py start  --config configs/experiments/<config>.yaml
$py scripts/experiments/train800k.py status --config configs/experiments/<config>.yaml

# Step 2 anchor, after the train8m fixed grid completes:
$py scripts/experiments/reproduction_checks.py anchor --run train8m_fixed_lr_high_grid_v1 \
    --control stage3_train8m_fixed_lr_v1 --member lr_14e-6

# Holdout, after all runs of a spec complete:
p=configs/prepared/<spec directory>
$py scripts/experiments/holdout_addendum.py --prepared $p --freeze
git add $p/protocol.json $p/protocol.sha256 && git commit -m "Freeze the $(basename $p) selection" && git push
$py scripts/experiments/holdout_addendum.py --prepared $p --dry-run
$py scripts/experiments/holdout_addendum.py --prepared $p
$py scripts/experiments/holdout_addendum.py --prepared $p --status   # --retry after a failure, --stop to stop
```

`train800k.py` places any population run, not only train800k runs. Do not type Ctrl-C into an attached
session. Use `--stop` for a holdout evaluation and `train800k.py stop` for a training run.
