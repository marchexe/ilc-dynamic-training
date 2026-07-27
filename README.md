# ILC Dynamic Training

Continuation experiments for a pretrained SGV `pp` ParticleTransformer using Weaver.
The repo is organized around one active pretrained checkpoint, parallel training runs,
and Population Based Training experiments.

## Active Inputs

```text
model:      networks/pretrained_sgv_particle_transformer.py
checkpoint: checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt
data cfg:   /data/suehara/part/data/ilc_nnqq_sgvnew_3cat_cut.217feb3dc9ed1ee6978db1c04604f81b.auto.yaml
dataset:    datasets/20250218_ilc_nnqq_sgvnew
```

`checkpoints/.../source.txt` records where the local checkpoint symlinks came from.
Large checkpoint/data/run artifacts are not tracked by Git.

## Experiments

Active presets:

```text
configs/experiments/parallel_baseline_vs_controller.yaml
configs/experiments/pbt_smoke.yaml
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

Install/update the local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c requirements-lock.txt
```

Inspect commands without launching training:

```bash
.venv/bin/python scripts/training/run_comparison.py --dry-run
.venv/bin/python scripts/training/run_pbt.py --dry-run
```

Run smoke checks:

```bash
.venv/bin/python scripts/training/run_comparison.py --smoke --gpus 0,2
.venv/bin/python scripts/training/run_pbt.py --smoke --gpus 0,2
```

Validate the pretrained checkpoint:

```bash
scripts/validation/validate_pretrained_sgv_3cat.sh 0
```

Plot reports from an existing PBT run:

```bash
.venv/bin/python scripts/reports/plot_bgrej_curves.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json
.venv/bin/python scripts/reports/plot_pbt_summary.py runs/pbt/parquet_pbt_bkg_rejection_best/manifest.json
```

Run tests:

```bash
.venv/bin/python -m unittest discover -v
```

## Project Layout

```text
configs/       experiment and controller presets
networks/      checkpoint-compatible pretrained SGV model
scripts/       launchers, reports, validation, cluster helpers
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
