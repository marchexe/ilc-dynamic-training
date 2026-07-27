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

PBT runs write `metrics_summary.json` and plot PNGs under `plots/` automatically after completion. The main physics plots are:

- `plots/btag_mistag_evolution.png`: c/d mistag [%] vs b-tag efficiency across generation winners.
- `plots/btag_rejection_evolution.png`: c/d background rejection vs b-tag efficiency across generation winners.
- `plots/pbt_lr_response.png`: learning-rate response at b-tag working points.
- `plots/working_point_mistag_history.png`: mistag history at fixed b/c efficiencies.
- `plots/global_best_all_pair_rejection_curves.png`: all-pair rejection curves for the global-best checkpoint.
- `plots/pbt_objective_diagnostics.png`: internal PBT objective used only for worker ranking.

Plot reports from an existing PBT run:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_bgrej_curves.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_summary.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_fixed_b_efficiency.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_bgrej_evolution.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json --quantity mistag'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_pbt_lr_response.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json'
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python scripts/reports/plot_mistag_tables.py run=runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json --tag c --eff 0.5,0.8'
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
