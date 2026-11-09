# Dynamic training against standard ParT, with a 1M-jet selection set

## Why

Over two seeds on train800k, the fast windowed PBT and the best constant LR were equal on the holdout set
(−1.0% and +0.5%), and in both seeds the validation set (val50k_tail, 150k jets) and the holdout set disagreed on
which arm was better. Member ranking, PBT decisions and checkpoint selection all use that validation set, so the
selection noise is as large as the effect. The comparison was also against tuned constant-LR grids, not against
standard ParT training.

## Terms

* **Selection set (sel1m_tail)**: 999,999 jets, the last 333,333 rows of each train8m flavor file. It ranks
  members, drives PBT decisions and selects checkpoints, as val50k_tail did before.
* **train8m_head**: train8m without the sel1m_tail rows, the 8M training input of step 3.
* **Standard ParT**: the recipe of the pretrained model: Ranger, start LR 1e-3, Weaver `flat+decay` (constant
  for 70% of the epochs, then ×0.01^(1/n) per epoch over the last n = 30%). The epoch-17 optimizer LR,
  2.15e-5, is exactly epoch 18 of a 20-epoch schedule from 1e-3. Here the schedule restarts from the epoch-17
  checkpoint for 50 epochs, at five start LRs (1e-5 to 1e-3).
* **Holdout set**: the frozen Stage 2A holdout, 2,825,125 jets, used once per selected checkpoint.

## Data

`scripts/validation/build_selection_set.py` writes `datasets/20250711_ilc_nnqq_sgv_10m_3cat_sel1m_parquet` and
the manifest `datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_sel1m_v1.json` (row ranges, file hashes, checks).
train800k is the first ~799k rows of each train8m file (checked row by row on the scalar columns), so train800k
runs never see sel1m_tail. train8m and val1000k/test3000k are different source files, so sel1m_tail does not
overlap the holdout set, val50k_tail, val5k_tail or the sealed Stage 3 test.

## Steps

| Step | Runs | GPUs × time | Holdout spec |
|---|---|---|---|
| 1 | build sel1m_tail and train8m_head | CPU | — |
| 2 | `sel1m_standard_part`, `sel1m_fixed_lr_grid`, `sel1m_fast_pbt`, each with seed 12345 and `_seed22345` | 6 × 5 × ~6–7 h | `holdout_sel1m_v1` |
| 3 | `sel1m_train8m_standard_part`, `sel1m_train8m_fast_pbt` | 2 × 5 × ~40–50 h | `holdout_sel1m_train8m_v1` |

* Step 2 asks: with reliable selection, does dynamic training beat standard ParT, and does it match a tuned
  constant-LR grid at a fifth of the cost? The fast PBT includes the fallback copies (`fallback_factors`).
* `sel1m_fixed_lr_grid_seed22345` trains exactly as `fixed_lr_best_grid_50epochs_seed22345` (training does not
  read the validation set), so every member is an anchor:
  `reproduction_checks.py anchor --run sel1m_fixed_lr_grid_seed22345 --control fixed_lr_best_grid_50epochs_seed22345 --member lr_120e-6`.
* Step 3 starts only if step 2 favours dynamic training, and after the supervisor agrees to the GPU time.
* LR schedules are the opt-in population key `lr_schedule` (`scripts/training/pbt/planning/lr_schedule.py`);
  the Stage 3 pinned-file change is recorded in `stage3_train8m_v1/execution_revision_3.json`.

## Commands

```bash
py=.venv/bin/python
setsid nohup bash scripts/experiments/run_queue.sh sel1m \
    configs/experiments/sel1m_standard_part.yaml configs/experiments/sel1m_fixed_lr_grid.yaml \
    configs/experiments/sel1m_fast_pbt.yaml configs/experiments/sel1m_standard_part_seed22345.yaml \
    configs/experiments/sel1m_fixed_lr_grid_seed22345.yaml configs/experiments/sel1m_fast_pbt_seed22345.yaml \
    > /dev/null 2>&1 < /dev/null &
# then, as in lr_level_followups.md: --freeze, commit, --dry-run, launch, --status
$py scripts/experiments/holdout_addendum.py --prepared configs/prepared/holdout_sel1m_v1 --freeze
```
