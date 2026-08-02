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

Active experiment configs:

```text
configs/experiments/pretrained_guarded_8gpu_smooth_lr.yaml  # canonical full run
configs/experiments/pretrained_guarded_4gpu_smooth_lr.yaml  # shorter test run
configs/experiments/pbt_smoke.yaml                          # local unit/smoke PBT
configs/experiments/pbt_ray_smoke.yaml                      # Ray executor smoke PBT
configs/experiments/pbt_anchored_lr_sweep.yaml              # legacy local sweep fixture
configs/experiments/pbt_anchored_lr_sweep_ray.yaml          # legacy Ray sweep fixture
configs/experiments/pbt_anchored_lr_sweep_ray_trial.yaml    # Ray trial fixture
configs/experiments/parallel_baseline_vs_controller.yaml    # comparison runner fixture
configs/experiments/pbt_control_fixed_lr.yaml               # comparison test fixture
configs/experiments/pbt_no_controller.yaml                  # controller fixture
configs/experiments/pbt_observe_controller.yaml             # controller fixture
```

Reusable preset blocks live in:

```text
configs/presets/shared/pretrained_epoch17_ranger_parquet.yaml
configs/presets/resources/local_8gpu.yaml
configs/presets/resources/local_4gpu.yaml
configs/presets/population/members_8.yaml
configs/presets/population/members_4.yaml
configs/presets/pbt/guarded_smooth_lr_mistag.yaml
```

Archived old presets live in:

```text
configs/experiments/archive/
configs/experiments/archive/legacy_finetune/
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
# Full 8GPU guarded continuation from pretrained epoch17 + optimizer state.
.venv/bin/python scripts/training/run_pbt.py \
  --config configs/experiments/pretrained_guarded_8gpu_smooth_lr.yaml \
  --slots iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3,iutgpu01:4,iutgpu01:5,iutgpu01:6,iutgpu01:7 \
  --experiment-name pretrained_guarded_8gpu_iutgpu01_$(date +%Y%m%d)

# Shorter 4GPU guarded test run using the same pretrained/safety presets.
.venv/bin/python scripts/training/run_pbt.py \
  --config configs/experiments/pretrained_guarded_4gpu_smooth_lr.yaml \
  --gpus 0,1,2,3 \
  --experiment-name pretrained_guarded_4gpu_test_$(date +%Y%m%d)
```

The guarded fine-tuning runs rank workers by `validation_working_point_mistag_percent`, the average mistag at b-tag 80/90% and c-tag 50/80% reference working points. They resume the canonical pretrained epoch-17 state and optimizer from the self-contained `checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/` bundle, seed the initial checkpoint as global best, and reject any worse global-best replacement. Selection is confidence-aware when exact background pass/total counters are available, so near-ties inside the configured uncertainty margin keep the previous/stable ordering instead of over-selecting noisy tails. The guarded presets use `anchored_weight_source: self`: LR is still centered on the best branch, but each member keeps its own weights across generations. This preserves long-horizon branch diversity; `anchored_weight_source: anchor` remains available for aggressive exploit-and-copy sweeps. Historical fixed-grid and rescue experiments are kept under `configs/experiments/archive/legacy_finetune/` for reproducibility only.

Datasets are not committed; the expected local parquet layout is recorded in `datasets/manifests/`. PBT runs write `metrics_summary.json` and plot PNGs under `plots/` automatically after completion. Lightweight Git-trackable research evidence lives in `results/research/` and can be produced with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/reports/export_research_result.py \
  runs/pbt/<run>/manifest.json \
  --output results/research/<result>.json \
  --csv-output results/research/<result>.csv
```

The clean showcase set is intentionally small:

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
