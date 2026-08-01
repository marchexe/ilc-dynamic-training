# ILC Dynamic Training

Continuation experiments for a pretrained SGV `pp` ParticleTransformer using Weaver.
The repo is organized around one active pretrained checkpoint, parallel training runs,
and Population Based Training experiments.

## Active Inputs

```text
model:      networks/pretrained_sgv_particle_transformer.py
checkpoint: checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt
data cfg:   /data/suehara/part/data/ilc_nnqq_sgvnew_3cat_cut.217feb3dc9ed1ee6978db1c04604f81b.auto.yaml
dataset:    datasets/20250218_ilc_nnqq_sgvnew_parquet
format:     parquet
```

`checkpoints/.../source.txt` records where the local checkpoint symlinks came from.
Large checkpoint/data/run artifacts are not tracked by Git.

## Experiments

Active presets:

```text
configs/experiments/parallel_baseline_vs_controller.yaml
configs/experiments/pbt_smoke.yaml
configs/experiments/pbt_anchored_lr_sweep.yaml
configs/experiments/pbt_anchored_lr_sweep_ray.yaml        # full Ray executor sweep
configs/experiments/pbt_anchored_lr_sweep_ray_trial.yaml  # epoch-wise Ray LR sweep preset
configs/experiments/pbt_ray_smoke.yaml                    # Ray executor smoke preset
configs/experiments/pbt_no_controller.yaml
configs/experiments/pbt_observe_controller.yaml
configs/experiments/pbt_control_fixed_lr.yaml
```

Archived old presets live in:

```text
configs/experiments/archive/
```

Controller presets:

```text
configs/controllers/linucb_lr_pp_active.yaml
configs/controllers/linucb_lr_pp_observe.yaml
```

## Commands

Install/update the Python 3.10 GPU-node environment:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && python3 -m venv .venv'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python -m pip install --upgrade pip'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt'
```

Inspect commands without launching training:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_comparison.py --dry-run'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/pbt_anchored_lr_sweep_ray_trial.yaml --gpus 0,1,2,3 --experiment-name ray_anchored_lr_sweep_epochwise --dry-run'
```

Run the current Ray PBT trial:

```bash
ssh iutgpu02
cd /data/suehara/part/march
tmux new -s ray_pbt
source .venv/bin/activate
.venv/bin/python scripts/training/run_pbt.py \
  --config configs/experiments/pbt_anchored_lr_sweep_ray_trial.yaml \
  --gpus 0,1,2,3 \
  --experiment-name ray_anchored_lr_sweep_epochwise
```

Run smoke checks on a GPU node:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_comparison.py --smoke --gpus 0,2'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --smoke --gpus 0,2'
```

Validate the pretrained checkpoint:

```bash
scripts/validation/validate_pretrained_sgv_3cat.sh 0
```

Fine-tuning architecture for the pretrained best checkpoint:

```bash
# 1. Low-LR fixed grid, AdamW: tests whether smaller LR alone prevents drift.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_fixed_lr_grid_adamw.yaml --gpus 0,1,2,3'

# 1a. Weaker low-LR grid, AdamW: isolates whether full fine-tuning needs much smaller steps.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_weak_lr_grid_adamw.yaml --gpus 0,1,2,3'

# 1b. Weaker low-LR grid, Ranger: same weak LR range with the historically steadier optimizer.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_weak_lr_grid_ranger.yaml --gpus 0,1,2,3'

# 2. Same fixed grid, Ranger: optimizer comparison against the historically stronger setup.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_fixed_lr_grid_ranger.yaml --gpus 0,1,2,3'

# 3. Adaptive rescue: anchored LR sweep with global-best rollback and early stop.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_adaptive_lr_rescue.yaml --gpus 0,1,2,3'

# 3a. Weaker adaptive rescue: smart LR around 1.5e-5 with radius shrink, rollback, and early stop.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_weak_adaptive_lr_rescue.yaml --gpus 0,1,2,3'

# 4. Head warmup: freeze backbone for two generations, then unfreeze for the same low-LR grid.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_head_warmup_fixed_lr_grid.yaml --gpus 0,1,2,3'

# 5. Safe optimizer-resume tail grid: damp epoch-17 Ranger momentum and rollback any baseline-regressing branch.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_damped_optimizer_tail_lr_grid_ranger.yaml --gpus 0,1,2,3'

# 5-control. Raw optimizer-resume tail grid: keep only as a control for reproducing the unsafe start behavior.
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/training/run_pbt.py --config configs/experiments/finetune_resume_optimizer_tail_lr_grid_ranger.yaml --gpus 0,1,2,3'
```

The fine-tuning runs rank workers by `validation_working_point_mistag_percent`, the average mistag at b-tag 80/90% and c-tag 50/80% reference working points. The fixed-grid runs keep each LR fixed; the adaptive rescue run is the one that mutates/rolls back workers.

PBT runs write `metrics_summary.json` and plot PNGs under `plots/` automatically after completion. The clean showcase set is intentionally small:

- `plots/report/physics_performance.png`: the main HEP-style result, combining fixed working-point mistag tables with compact mistag [%] bar charts.
- `plots/report/training_diagnostics.png`: compact training/PBT diagnostic for understanding whether the run improved or drifted.

Machine-readable fixed working-point tables are also written as `plots/report/ctag_mistag_tables.csv` and `plots/report/btag_mistag_tables.csv`. Default diagnostic PNGs are `plots/diagnostics/background_efficiency_curves.png`, `plots/diagnostics/btag_background_efficiency_vs_training_size.png`, and `plots/diagnostics/selection_timeline.png`; other report scripts are available for manual debugging but are not generated as part of the standard report.

Plot reports from an existing PBT run:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_bgrej_curves.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_summary.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_physics_performance.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_fixed_b_efficiency.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_bgrej_evolution.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json --tag b --quantity mistag'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_bgrej_evolution.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json --tag c --quantity mistag'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_lr_response.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_mistag_tables.py run=runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json --tag c --eff 0.5,0.8'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_mistag_tables.py run=runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json --tag b --eff 0.8,0.9'
# Optional ROOT style, after loading ROOT/PyROOT on the host:
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_fixed_b_efficiency_root.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
```

Run tests:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python -m unittest discover -v'
```

## Project Layout

```text
configs/       experiment and controller presets
networks/      checkpoint-compatible pretrained SGV model
scripts/       launchers, PBT backend, reports, validation, cluster helpers
tests/         unit and compatibility tests
weaver-core/   editable local Weaver checkout
```

## Git Policy

Tracked:

```text
source code
experiment/controller configs
requirements.txt
requirements-lock.txt
checkpoint provenance text
```

Ignored:

```text
.venv/
datasets/
checkpoints/* except source.txt
runs/
results/
*.pt, *.root, *.onnx, *.log, *.auto.yaml
```
