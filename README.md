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
configs/experiments/pretrained_guarded_8gpu_smooth_lr.yaml  # labeled canonical full run, but no run/ has ever used it -- audit as of 2026-08-04
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
configs/experiments/pretrained_exploit_mutate_8gpu_10m_tiered_pilot.yaml            # 8-member exploit_mutate + tiered-validation pilot, 10m proxy-control dataset
configs/experiments/pretrained_fixed_lr_8gpu_10m_proxy_control.yaml                 # fixed-LR grid (no exploit), same 8-member ladder/dataset as the tiered pilot
configs/experiments/pretrained_guarded_8gpu_smooth_lr_10m_proxy_control.yaml        # guarded smooth-LR sweep on the 10m proxy-control dataset (see line ~132)
configs/experiments/pretrained_exploit_mutate_smoke_tiered.yaml                     # architecture-complete smoke of the exploit_mutate/tiered-validation stack (4 members, 6 generations)
configs/experiments/pretrained_exploit_mutate_smoke_tiered_ownership_fix_verify.yaml # same smoke, shortened to 4 generations -- verification run for the PBT/dynamic-controller LR-ownership fix
configs/experiments/bn_freeze_diag_baseline.yaml            # single-gen BatchNorm-drift diagnostic (freeze_batch_norm: false)
configs/experiments/bn_freeze_diag_frozen.yaml              # paired diagnostic confirming --freeze-batch-norm fixes the gen-0 mistag regression
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

Datasets are not committed; the expected local parquet layout is recorded in `datasets/manifests/`.

The main proxy-validation dataset is a local 3-category subset of `/data/suehara/mldata/flavortag/20250711_ilc_nnqq_sgv_10m`, restricted to `bb/cc/dd` to match the current pretrained 3cat head and working-point metrics. Rebuild its parquet form with:

```bash
bash scripts/data/convert_20250711_sgv10m_3cat_to_parquet.sh
```

Its tracked base manifest is `datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_parquet.json`; the active proxy manifest is `datasets/manifests/20250711_ilc_nnqq_sgv_10m_3cat_tail_proxy_v1.json`. Use `configs/experiments/pretrained_guarded_8gpu_smooth_lr_10m_proxy_control.yaml` for an adaptive run that trains on `train800k` parquet while validating on the fixed `val5k_tail` control proxy. The `val5k_tail` control proxy uses the final rows of each `val1000k` file; `val50k_tail` monitor uses the immediately preceding rows, so monitor/control are disjoint and both come from the dataset tail.

PBT runs write canonical artifacts automatically after completion. Rebuild those artifacts for an existing completed run without training with:

```bash
.venv/bin/python scripts/training/pbt/rebuild_artifacts.py runs/pbt/<run>
```

Lightweight Git-trackable research evidence lives in `results/research/` and can be produced with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/reports/export_research_result.py \
  runs/pbt/<run>/manifest.json \
  --output results/research/<result>.json \
  --csv-output results/research/<result>.csv
```

The final PBT artifact set is intentionally small:

- `plots/training_evolution.png`
- `plots/working_point_evolution.png`
- `plots/baseline_vs_selected.png` (only when a measured baseline and a global-best checkpoint both exist)
- `plots/report/physics_performance.png`
- `plots/diagnostics/background_efficiency_curves.png`
- `plots/report/btag_mistag_tables.csv`
- `plots/report/ctag_mistag_tables.csv`
- `plots/report/exploit_table.csv`

Run tests:

```bash
ssh iutgpu02 'cd /data/suehara/part/march && .venv/bin/python -m unittest discover -v'
```

## Project Layout

```text
configs/               experiment and controller presets (see configs/presets/README.md)
networks/               checkpoint-compatible pretrained SGV model
scripts/
  training/
    pbt/                 the PBT engine, split by concern:
      models/              pydantic schema for config/manifest/controller/exploit-events
      planning/            population ranking + one module per pbt.strategy (exploit_mutate,
                            anchored_lr_sweep, fixed_lr_grid, population_lr_policy) + rollback injection
      controller/           the dynamic (fine-grained LR) controller: observation/decision/apply
      execution/            backend/ray_backend/weaver_command -- runs the actual Weaver subprocesses
      state/                 checkpoint paths, optimizer-state transforms, exploit-application
      reporting/            canonical run-directory artifacts: events, CSVs, plots, report.md
      runner.py             the generation-loop orchestrator (this package's entrypoint)
    comparison/            independent (non-PBT) baseline-vs-controller runner
    runtime.py             shared utilities (paths, data layout, Weaver log/metric parsing) used by
                            pbt/, comparison/, and validation/ alike
    weaver.py               generic Weaver train/val command builder, wrapped by pbt/execution/weaver_command.py
  validation/              offline proxy-validation dataset construction + standalone checkpoint evaluation
  reports/                 plotting/summary library shared by pbt/reporting/ and validation/
  cluster/, data/          GPU-fleet status and dataset-conversion shell scripts (no Python coupling)
tests/                   unit and compatibility tests (python -m unittest discover)
weaver-core/             editable local Weaver checkout (vendored, but actively patched -- see git log)
checkpoints/, datasets/  pretrained checkpoint and dataset manifests (heavy artifacts are gitignored)
runs/                    full experiment output trees (gitignored except runs/showcase/)
results/research/        lightweight, Git-tracked JSON/CSV run summaries (scripts/reports/export_research_result.py)
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
