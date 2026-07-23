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

## Path toward PBT

The implemented launcher solves the first required layer: multiple independent,
reproducible trainings running concurrently on assigned GPUs.

The next PBT layer should:

1. define a population of workers and an evaluation interval;
2. rank workers using a chosen validation objective;
3. copy model and optimizer state from stronger to weaker workers;
4. mutate selected hyperparameters;
5. persist lineage and mutations in the experiment manifest;
6. resume the whole population after interruption.

That coordinator should reuse this launcher/manifest contract. It should be
implemented before drawing performance conclusions from LinUCB or revisiting
`ee` pairwise variables.

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
│   ├── validate_pretrained_sgv_3cat.sh
│   └── train_sgv_3cat.sh
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
