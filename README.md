# ILC Dynamic Training

This repository continues training a checkpoint-compatible ParticleTransformer
on ILC SGV data and provides the parallel execution layer needed for population
experiments.

## Current objective

The current comparison starts from one pretrained `pp` checkpoint:

```text
checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt
```

The checkpoint is the best state from the completed 20-epoch training (epoch
17). On the project-local validation data it gives:

```text
accuracy: 0.88936
ROC AUC:  0.97336
```

The historical training report lists `0.88992` accuracy and `0.97364` AUC. This
small difference is the reference tolerance; continuation experiments should
be compared with the historical best, not with a model trained from scratch.

Two matched workers are currently defined:

- `baseline`: fixed learning rate;
- `linucb`: the same continuation with the experimental LinUCB controller.

LinUCB is an integration trial, not the assumed final optimization algorithm.
The parallel launcher is intentionally controller-agnostic so it can serve as
the execution layer for a later PBT coordinator.

## Data and model contract

The local dataset is expected at:

```text
datasets/20250218_ilc_nnqq_sgvnew
```

The data configuration contains separate charged and neutral inputs:
`pf_features/pf_vectors/pf_mask` and
`neu_features/neu_vectors/neu_mask`. The checkpoint-compatible network consumes
both groups separately and uses the original `pair_input_type="pp"` path.

The model is self-contained in:

```text
networks/particle_transformer_pp_pretrained.py
```

It does not import code from another checkout under `/data/suehara/weaver`.
Do not substitute `particle_transformer_ee.py`: the modern `ee` implementation
is not numerically compatible with this checkpoint even where parameter shapes
match.

Validate the untouched checkpoint on one GPU:

```bash
./scripts/validate_pretrained_sgv_3cat.sh 0
```

## Parallel matched training

The reproducible experiment contract is:

```text
configs/experiments/pp_matched.yaml
```

It defines the shared checkpoint, dataset, seed, optimizer, learning rate,
batch size, epoch/sample budgets, network, and workers. Shared values are
constructed once and passed to every worker; only the worker GPU and optional
controller differ.

Inspect commands without starting training:

```bash
.venv/bin/python scripts/run_parallel_training.py --dry-run
```

Run the required two-GPU smoke test:

```bash
.venv/bin/python scripts/run_parallel_training.py \
  --smoke \
  --gpus 0,2 \
  --experiment-name pp_smoke_01
```

`--smoke` overrides the budget to one epoch, 7680 training samples, and 3000
validation samples. Its metrics only verify integration; short training changes
BatchNorm statistics and is not a performance measurement.

After the smoke test is accepted, run the configured two-epoch continuation:

```bash
.venv/bin/python scripts/run_parallel_training.py \
  --gpus 0,2 \
  --experiment-name pp_continuation_01
```

Resume an interrupted run with the same config and overrides:

```bash
.venv/bin/python scripts/run_parallel_training.py \
  --gpus 0,2 \
  --experiment-name pp_continuation_01 \
  --resume
```

Resume is rejected if the saved experiment fingerprint differs from the
resolved training contract. A worker resumes only from an epoch that has model
and optimizer state, plus controller state where applicable.

## Outputs and failure behavior

Each experiment is isolated under:

```text
runs/parallel/<experiment>/
├── manifest.json
├── resolved_config.yaml
├── baseline/
│   ├── console.log
│   ├── train.log
│   └── net_*.pt
└── linucb/
    ├── console.log
    ├── train.log
    ├── net_controller.jsonl
    └── net_*.pt
```

The manifest records the exact commands, checkpoint path/target/SHA-256, Git
revision and dirty state, worker status, timestamps, and final validation
metrics. If one worker fails, the launcher terminates the other workers and
marks the experiment failed instead of leaving a partial comparison running.

With FP16, PyTorch can reject an optimizer step while adjusting dynamic loss
scaling. Such steps are counted in the log as `AMP skipped optimizer steps`;
the per-step scheduler and training controller also skip them.

## Population Based Training

The PBT coordinator is configured in:

```text
configs/experiments/pp_pbt.yaml
```

The default population contains four fixed-LR members. Two GPU slots execute
the population in parallel waves. After every generation the coordinator:

1. ranks members by the configured validation metric;
2. replaces the bottom fraction with model and optimizer state from the top;
3. mutates the donors' learning rates within configured bounds;
4. resumes every member from the resulting epoch checkpoint;
5. records ranking, metrics, parentage, mutations and exact commands.

Run a two-member, two-generation integration test:

```bash
.venv/bin/python scripts/run_pbt.py \
  --smoke \
  --gpus 0,2 \
  --experiment-name pp_pbt_smoke_01
```

Inspect the full four-member commands without training:

```bash
.venv/bin/python scripts/run_pbt.py --dry-run
```

Run the configured population:

```bash
.venv/bin/python scripts/run_pbt.py \
  --gpus 0,2 \
  --experiment-name pp_pbt_01
```

Use the same options plus `--resume` after an interruption. Exploit copying is
atomic and idempotent, so resume can safely finish a partially applied
generation. On resume, Weaver restores model and optimizer state and then
rescales optimizer parameter-group learning rates to the PBT-selected value.

The default ranking objective is validation accuracy, matching the historical
`0.88992` reference. It can be changed to validation AUC or loss in the YAML.
The current minimal implementation mutates only learning rate; additional
hyperparameters can be added after this execution path is accepted.
Smoke-test metrics are not evidence of PBT performance.

## Previous ee prototype

`networks/particle_transformer_ee.py`, `scripts/train_sgv_3cat.sh`, and the
matched-seed results under `results/` belong to the earlier LinUCB integration
prototype. They are retained for provenance but are not used by the current
pretrained `pp` continuation.

## Project layout

```text
.
├── configs/
│   ├── controllers/          LinUCB settings
│   ├── data/                 charged/neutral SGV preprocessing
│   └── experiments/          reproducible parallel run contracts
├── networks/
│   ├── particle_transformer_pp_pretrained.py
│   └── particle_transformer_ee.py
├── scripts/
│   ├── run_parallel_training.py
│   ├── run_pbt.py
│   ├── validate_pretrained_sgv_3cat.sh
│   └── train_sgv_3cat.sh
├── tests/                    coordinator unit tests
├── weaver-core/              Weaver and controller integration
├── checkpoints/              local artifacts, ignored by Git
├── datasets/                 local data, ignored by Git
└── runs/                     generated experiments, ignored by Git
```

## Environment

The prepared environment is `.venv`. To recreate it on a compatible CUDA
machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
