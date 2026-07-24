# ILC Dynamic Training

This project studies whether adaptive training strategies can improve an
existing ParticleTransformer for ILC flavour tagging.

The starting point is not a randomly initialized model. It is the best `pp`
checkpoint from a completed 20-epoch training run. The current work builds a
reproducible continuation pipeline around that checkpoint:

```text
pretrained pp model
        ↓
parallel independent training
        ↓
matched baseline/controller comparison
        ↓
Population Based Training
        ↓
full-budget performance evaluation
```

The immediate goal is infrastructure and experiment correctness. No PBT
performance improvement is claimed yet.

## Why this direction

The project originally focused on an `ee` ParticleTransformer and a LinUCB
learning-rate controller. The current direction follows these observations:

- the supervisor requested improvement from the best existing model, rather
  than another from-scratch result;
- previous work did not show a clear advantage from the `ee` pairwise
  variables;
- LinUCB is useful as an integration trial, but should not be assumed to be the
  final algorithm;
- parallel multiple training is required before meaningful PBT experiments.

Therefore:

- current pretrained path: `pp`;
- current optimization direction: PBT;
- LinUCB: optional comparison worker;
- old `ee` path: retained for provenance, not used by PBT.

## Current state

Implemented and verified:

- checkpoint-compatible `pp` model;
- local copy of the original dataset;
- separate charged and neutral particle inputs;
- pretrained checkpoint validation;
- parallel multi-GPU launcher;
- isolated logs, checkpoints and experiment manifests;
- failure handling and resume;
- epoch-level PBT with ranking, exploit, LR mutation and lineage;
- two-worker/two-generation PBT smoke-test.

Not completed:

- full-budget continuation;
- full four-member PBT run;
- multi-seed performance comparison;
- final choice of ranking metric;
- decision on initial optimizer state.

## Pretrained reference

Checkpoint:

```text
checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt
```

Model:

```text
networks/particle_transformer_pp_pretrained.py
```

The checkpoint is the best state, epoch 17, from the original 20-epoch run.

| Evaluation | Accuracy | ROC AUC |
|---|---:|---:|
| Historical report | `0.88992` | `0.97364` |
| Local reproduction | `0.88936` | `0.97336` |

Validation command:

```bash
./scripts/validate_pretrained_sgv_3cat.sh 0
```

The small difference from the historical result is treated as the baseline
reproduction tolerance.

## Charged and neutral particles

Data configuration:

```text
configs/data/ilc_nnqq_sgvnew_3cat.yaml
```

The two particle types are represented separately:

```text
charged → pf_features  + pf_vectors  + pf_mask
neutral → neu_features + neu_vectors + neu_mask
```

The model applies separate embeddings before the common transformer:

```text
charged embedding ─┐
                   ├→ ParticleTransformer → nnbb / nncc / nndd
neutral embedding ─┘
```

The active model uses `pair_input_type="pp"`. The modern `ee` implementation is
not numerically compatible with the pretrained checkpoint, even where tensor
shapes match.

## Parallel matched training

Purpose: compare two continuations under the same experimental conditions.

Configuration:

```text
configs/experiments/pp_matched.yaml
```

Workers:

```text
baseline → fixed learning rate
linucb   → experimental LinUCB learning-rate controller
```

Both workers receive the same:

- pretrained weights;
- dataset and preprocessing;
- random seed;
- optimizer and initial LR;
- batch size;
- epoch and sample budgets;
- AMP settings.

Only the GPU, output directory and optional controller differ.

Inspect the resolved commands:

```bash
.venv/bin/python scripts/run_parallel_training.py --dry-run
```

Run a short integration test:

```bash
.venv/bin/python scripts/run_parallel_training.py \
  --smoke \
  --gpus 0,2 \
  --experiment-name pp_matched_smoke
```

Resume with the same options plus:

```bash
--resume
```

## Population Based Training

Purpose: train a population, keep useful states from stronger members, and
explore learning-rate variants without restarting models from scratch.

Configuration:

```text
configs/experiments/pp_pbt.yaml
```

Default population:

| Member | Initial LR |
|---|---:|
| `member_00` | `7.5e-5` |
| `member_01` | `1.0e-4` |
| `member_02` | `1.25e-4` |
| `member_03` | `1.5e-4` |

One PBT generation:

```text
train every member
        ↓
evaluate validation metric
        ↓
rank population
        ↓
bottom 50% copy model + optimizer from top 50%
        ↓
mutate copied learning rate by ×0.8 or ×1.2
        ↓
resume next generation
```

The population may be larger than the available GPU count. With four members
and GPU slots `0,2`, execution happens in two parallel waves.

Inspect the full PBT commands:

```bash
.venv/bin/python scripts/run_pbt.py --dry-run
```

Run the integration smoke-test:

```bash
.venv/bin/python scripts/run_pbt.py \
  --smoke \
  --gpus 0,2 \
  --experiment-name pp_pbt_smoke
```

The smoke-test uses two members, two generations, 7680 training samples and
3000 validation samples per generation.

## What the smoke-test proves

Verified control flow:

```text
parallel training
→ validation
→ ranking
→ model copy
→ optimizer copy
→ LR mutation
→ epoch resume
```

Example:

```text
generation 0:
member_01 outperformed member_00

exploit:
member_00 model     ← member_01 model
member_00 optimizer ← member_01 optimizer
member_00 LR        ← 1e-4 × 0.8 = 8e-5

generation 1:
both members resumed from epoch 0
```

Smoke-test metrics are not performance results. The short run changes
BatchNorm statistics and uses only a small validation subset; it must not be
compared directly with `0.88992`.

## Reproducibility and recovery

Each experiment writes:

```text
manifest.json
resolved_config.yaml
per-worker training logs
per-worker console logs
model checkpoints
optimizer checkpoints
optional controller state
```

The manifest records:

- resolved experiment configuration;
- checkpoint path and SHA-256;
- Git revision and dirty state;
- exact commands;
- seeds and GPU assignments;
- worker exit status;
- validation loss, accuracy and AUC;
- PBT ranking, mutations and parentage.

Resume is rejected if the resolved experiment contract differs from the saved
fingerprint.

PBT exploit copying is atomic and idempotent. A partially completed exploit can
be safely applied again after resume.

## AMP behavior

FP16 can produce non-finite gradients while dynamic loss scaling is being
adjusted.

On a rejected AMP step:

```text
optimizer step skipped
scheduler step skipped
training-controller observation skipped
event counted in the training log
```

LinUCB additionally sanitizes non-finite context values so they cannot corrupt
its internal matrices.

## Important open decisions

Before a full run:

1. Initial optimizer:
   - fresh Ranger optimizer per member; or
   - original epoch-17 optimizer state.
2. Final PBT ranking target:
   - combined `validation_bkg_rejection_score`;
   - b-tag focused `validation_b_tag_rejection_score`;
   - c-tag focused `validation_c_tag_rejection_score`;
   - individual `bc` / `bd` / `cb` / `cd` rejection scores.
3. Population size and number of generations.
4. Learning-rate range and mutation factors.
5. Final comparison protocol and number of seeds.

The current implementation starts every initial member from pretrained model
weights with a fresh optimizer. After the first PBT generation, model and
optimizer states are transferred together.

The active LinUCB controller can also use high-frequency proxy validation:
every configured interval it evaluates a small validation subset, computes the
physics-aligned `bkg_rejection_score`, and uses that signal as its online
reward for learning-rate actions. Full validation is still kept as the end-of-
epoch reference.

## Main files

```text
configs/
├── data/ilc_nnqq_sgvnew_3cat.yaml
├── experiments/pp_matched.yaml
├── experiments/pp_pbt.yaml
└── controllers/linucb_lr_pp_active.yaml

networks/
├── particle_transformer_pp_pretrained.py   # active path
└── particle_transformer_ee.py              # previous prototype

scripts/
├── validate_pretrained_sgv_3cat.sh
├── run_parallel_training.py
└── run_pbt.py

tests/
└── test_pbt.py

weaver-core/weaver/
├── train.py                                # checkpoint/resume logic
└── utils/
    ├── nn/tools.py                         # training and AMP handling
    └── training_control/                   # LinUCB integration
```

## Local artifacts

Ignored by Git:

```text
.venv/
datasets/
checkpoints/
runs/
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Covered:

- PBT generation resume command;
- deterministic ranking and exploit planning;
- model and optimizer copying;
- lineage updates;
- LinUCB protection from non-finite gradient norms.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Previous prototype

The following files belong to the earlier `ee`/LinUCB prototype:

```text
networks/particle_transformer_ee.py
scripts/train_sgv_3cat.sh
scripts/plot_training_control.py
results/
```

They remain in the repository for provenance and are not used by the current
pretrained `pp` PBT pipeline.
