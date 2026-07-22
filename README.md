# ILC Dynamic Training

This project trains an ee-specific ParticleTransformer on ILC SGV fast-simulation
data and experiments with ML-based learning-rate control inside an epoch.

Start with `observe` mode. It runs normal training and records the actions that
the controller would propose, but does not change the learning rate.

## Quick start

The environment is already prepared in `.venv` on `iutgpu05`.

Run a short one-epoch check on GPU 2:

```bash
cd /data/suehara/part/march

EPOCHS=1 \
SAMPLES_PER_EPOCH=76800 \
SAMPLES_PER_EPOCH_VAL=15000 \
./scripts/train_sgv_3cat.sh 2 observe
```

Follow the training log:

```bash
tail -f runs/sgv_3cat/observe/train.log
```

After the run, inspect the controller decisions:

```bash
sed -n '1,10p' runs/sgv_3cat/observe/net_controller.jsonl
```

The short run contains 300 training batches. The controller warms up for 10%
of the epoch and then produces a decision every 5% of the epoch.

To show only the useful controller and metric lines:

```bash
rg '\[training-control\]|Train AvgLoss|Eval AvgLoss|Current validation metric' \
  runs/sgv_3cat/active/train.log
```

## Verified matched-seed demo

A fixed-LR run and an active run were completed with the same seed, training
budget, and train/validation worker seeds:

```text
                         fixed LR   LinUCB
validation accuracy      0.81964    0.81782
validation ROC AUC       0.93598    0.93886
```

The controller made 19 decisions without producing NaN. The mixed result is an
integration milestone, not evidence of an overall performance gain: AUC rose
while accuracy fell, and only one seed has been tested.

Committed figure: [`results/training_control_comparison_seed12345.png`](results/training_control_comparison_seed12345.png).

## What happens during training

```text
ROOT files
    ↓
fixed Weaver preprocessing
    ↓
ee-specific ParticleTransformer
    ↓
cross-entropy loss
    ↓
AdamW optimizer updates the model
    ↓
LinUCB observes the training state
    ↓
logs or applies a learning-rate action
```

The classification task has three classes:

- `nnbb` — b jets
- `nncc` — c jets
- `nndd` — d jets

The model uses the current official ParticleTransformer implementation with
`pair_input_type="ee"`. It does not use the old local
`ParticleTransformer_test.py` implementation.

## ML-based learning-rate controller

AdamW remains the optimizer that updates the network weights. LinUCB is a
separate controller that changes one AdamW hyperparameter: the learning rate.

The controller observes:

- exponentially smoothed training loss;
- loss change over the previous decision window;
- gradient norm;
- progress through the epoch;
- current learning rate.

After a warmup covering 10% of an epoch, it makes a decision every 5%:

```text
LR × 0.9
LR × 1.0
LR × 1.1
```

The learning rate is restricted to `5e-4 ... 2e-3` for the current `1e-3`
baseline.

### Observe mode

```bash
./scripts/train_sgv_3cat.sh 2 observe
```

The proposed actions are logged, but the learning rate is unchanged. No reward
is attributed to an action that was not actually applied. Use this mode first
to validate the training and logging pipeline.

### Active mode

```bash
./scripts/train_sgv_3cat.sh 2 active
```

The selected action changes the learning rate for the next training window.
At the following decision point, LinUCB receives a reward based on the relative
improvement of the smoothed loss and updates its internal model.

Do not use `active` mode for the first check.

### Fixed-LR baseline and comparison plot

```bash
SEED=12345 RUN_NAME=baseline_seed12345_v2 \
  EPOCHS=1 SAMPLES_PER_EPOCH=153600 SAMPLES_PER_EPOCH_VAL=15000 \
  ./scripts/train_sgv_3cat.sh 2 baseline

SEED=12345 RUN_NAME=active_seed12345_v2 \
  EPOCHS=1 SAMPLES_PER_EPOCH=153600 SAMPLES_PER_EPOCH_VAL=15000 \
  ./scripts/train_sgv_3cat.sh 2 active

python scripts/plot_training_control.py \
  --baseline-dir runs/sgv_3cat/baseline_seed12345_v2 \
  --active-dir runs/sgv_3cat/active_seed12345_v2 \
  --output runs/sgv_3cat/training_control_comparison_seed12345.png
```

The figure shows the controller's LR decisions and a matched-seed validation
comparison with fixed LR.

## Data

The default dataset is:

```text
/data/suehara/mldata/flavortag/20250218_ilc_nnqq_sgvnew
```

To use another copy of the same dataset layout:

```bash
ILC_FASTSIM_DIR=/path/to/dataset ./scripts/train_sgv_3cat.sh 2 observe
```

The committed data configuration already contains preprocessing and reweighting
statistics. They are not recalculated for every checkout.

## Output files

An observe run writes to `runs/sgv_3cat/observe/`; an active run writes to
`runs/sgv_3cat/active/`.

```text
train.log                         training log
net_controller.jsonl             controller observations and decisions
net_epoch-N_state.pt             ParticleTransformer weights
net_epoch-N_optimizer.pt         AdamW state
net_epoch-N_controller.pt        LinUCB state
```

The model, optimizer, and controller states are saved separately so an
interrupted experiment can later resume consistently.

## Project layout

```text
.
├── configs/
│   ├── controllers/              observe and active LinUCB settings
│   └── data/                     fixed SGV preprocessing configuration
├── networks/
│   └── particle_transformer_ee.py
├── scripts/
│   └── train_sgv_3cat.sh         main entry point
├── weaver-core/                  Weaver plus the controller integration
├── .venv/
├── requirements.txt
└── README.md
```

Relevant implementation files:

- `weaver-core/weaver/utils/training_control/` — controller API and LinUCB;
- `weaver-core/weaver/utils/nn/tools.py` — per-batch controller hook;
- `weaver-core/weaver/train.py` — CLI setup and controller checkpoints;
- `weaver-core/docs/training-controller.md` — lower-level Weaver documentation.

## Current status and limitation

- Weaver base: official commit `154db693565c69fabbc3fd80923fb8c2724fbf7b`.
- Controller integration source: commits `bb7dbeb`, `02d9ae6`, and `ca30c63`
  on branch `feature/ml-training-controller`, imported here as `weaver-core/`
  with Git subtree.
- The controller currently learns from training-loss improvement only.
- It is therefore loss-aware, not yet physics-aware.

The next development step is a fixed proxy-validation set whose AUC or partial
AUC becomes the controller reward. That must be validated against full
validation before making physics-performance claims.

## Environment setup on another machine

Install a CUDA-compatible PyTorch build first. Then create the environment and
install the local Weaver checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
