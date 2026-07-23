# ILC Dynamic Training

Internal training infrastructure for ILC SGV flavour tagging.

## Status

| Component | State |
|---|---|
| Pretrained `pp` validation | Passed |
| Charged/neutral split | Verified |
| Two-GPU matched launcher | Passed smoke-test |
| LinUCB integration | Trial only |
| Epoch-level PBT | Passed 2-generation smoke-test |
| Full continuation | Not run |
| Full PBT benchmark | Not run |

Reference metrics:

| Source | Accuracy | ROC AUC |
|---|---:|---:|
| Historical best | `0.88992` | `0.97364` |
| Local checkpoint validation | `0.88936` | `0.97336` |

## Quick commands

```bash
# Validate untouched pretrained weights
./scripts/validate_pretrained_sgv_3cat.sh 0

# Inspect matched baseline/LinUCB commands
.venv/bin/python scripts/run_parallel_training.py --dry-run

# Matched two-GPU smoke-test
.venv/bin/python scripts/run_parallel_training.py \
  --smoke --gpus 0,2 --experiment-name pp_matched_smoke

# Inspect four-member PBT commands
.venv/bin/python scripts/run_pbt.py --dry-run

# Two-member, two-generation PBT smoke-test
.venv/bin/python scripts/run_pbt.py \
  --smoke --gpus 0,2 --experiment-name pp_pbt_smoke

# Unit tests
.venv/bin/python -m unittest discover -s tests -v
```

No full-budget command should be started before reviewing:

- optimizer initialization;
- ranking metric;
- population size;
- LR range;
- compute budget.

## Required local artifacts

Ignored by Git:

| Artifact | Path |
|---|---|
| Dataset | `datasets/20250218_ilc_nnqq_sgvnew/` |
| Pretrained weights | `checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt` |
| Virtual environment | `.venv/` |
| Run outputs | `runs/` |

Checkpoint:

- source run: 20 epochs;
- selected best state: epoch 17;
- SHA-256: `ae4928aa088b73538597f23b78b51678298e59d23552c9cd2c2e849fb3ced501`;
- initial continuation: model weights + fresh optimizer;
- original epoch-17 optimizer: available externally, not wired into current contract.

## Model/data contract

| Item | Value |
|---|---|
| Task | `nnbb` / `nncc` / `nndd` |
| Network | `networks/particle_transformer_pp_pretrained.py` |
| Pair input | `pp` |
| Data config | `configs/data/ilc_nnqq_sgvnew_3cat.yaml` |
| Optimizer | Ranger |
| AMP | FP16 |

Particle inputs:

```text
charged → pf_features  + pf_vectors  + pf_mask
neutral → neu_features + neu_vectors + neu_mask
```

Model flow:

```text
charged embedding ─┐
                   ├→ checkpoint-compatible ParticleTransformer → 3 classes
neutral embedding ─┘
```

Constraints:

- charged and neutral inputs remain separate;
- pretrained weights require the checkpoint-compatible `pp` implementation;
- `particle_transformer_ee.py` is not checkpoint-equivalent;
- no runtime import from `/data/suehara/weaver`.

## Matched parallel training

Config:

```text
configs/experiments/pp_matched.yaml
```

Workers:

| Worker | Controller |
|---|---|
| `baseline` | none |
| `linucb` | `configs/controllers/linucb_lr_pp_active.yaml` |

Matched fields:

```text
checkpoint
dataset
data/network config
seed
optimizer
start LR
batch size
epoch/sample budget
AMP settings
```

Per-worker fields:

```text
name
GPU
controller
output directory
```

Execution:

```text
resolve YAML
  → validate paths and samples
  → compute contract fingerprint
  → launch workers
  → monitor exit codes
  → parse validation metrics
  → write manifest
```

Failure policy:

- one worker fails → terminate remaining workers;
- experiment status → `failed`;
- incompatible resume contract → reject;
- resume checkpoint → model + optimizer + optional controller state.

Output:

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

## Population Based Training

Config:

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

Default policy:

| Field | Value |
|---|---|
| GPU slots | `0,2` |
| Generations | `2` |
| Epochs/generation | `1` |
| Ranking metric | `validation_accuracy` |
| Ranking mode | `max` |
| Exploit fraction | `0.5` |
| LR mutations | `×0.8`, `×1.2` |
| LR bounds | `5e-5 ... 2e-4` |

Generation loop:

```text
train population
  → validate population
  → rank members
  → bottom 50% copy model + optimizer from top 50%
  → mutate copied LR
  → persist lineage
  → resume next generation
```

GPU scheduling with four members/two slots:

```text
wave 1: member_00@GPU0 + member_01@GPU2
wave 2: member_02@GPU0 + member_03@GPU2
```

Exploit state:

```text
donor net_epoch-N_state.pt     → recipient
donor net_epoch-N_optimizer.pt → recipient
mutated LR                     → applied after optimizer load
```

Safety:

- atomic checkpoint replacement;
- exploit plan persisted before copy;
- idempotent partial-exploit resume;
- config fingerprint validation;
- generation/member status in manifest;
- exact commands and lineage recorded.

Supported ranking fields:

```text
validation_accuracy
validation_auc
validation_loss
```

Current mutation space:

```text
learning rate only
```

Output:

```text
runs/pbt/<experiment>/
├── manifest.json
├── resolved_config.yaml
├── member_00/
│   ├── generation-000.log
│   ├── generation-000.console.log
│   └── net_epoch-*.pt
└── member_*/
```

## AMP/controller behavior

FP16 non-finite gradient:

```text
GradScaler rejects optimizer step
  → scheduler step skipped
  → training-controller observation skipped
  → counter logged as "AMP skipped optimizer steps"
```

LinUCB fallback:

- non-finite context values sanitized;
- no NaN/Inf propagation into LinUCB matrices;
- still experimental;
- not part of current PBT.

## Reproducibility manifest

Recorded fields:

```text
resolved config
contract fingerprint
checkpoint path + SHA-256
Git commit/branch/dirty state
exact worker commands
seeds and GPU assignment
timestamps and return codes
validation loss/accuracy/AUC
PBT ranking
donor/recipient lineage
LR mutations
resume state
```

## Known limitations

- Full-budget results unavailable.
- PBT performance improvement unproven.
- Smoke metrics not comparable with `0.88992`.
- Initial population uses fresh optimizers.
- PBT mutates LR only.
- Default ranking metric not yet physics-approved.
- Multi-seed statistical comparison not run.
- `ee` pairwise path retained only as an earlier prototype.

## Repository map

```text
configs/
├── controllers/linucb_lr_pp_active.yaml
├── data/ilc_nnqq_sgvnew_3cat.yaml
└── experiments/
    ├── pp_matched.yaml
    └── pp_pbt.yaml

networks/
├── particle_transformer_pp_pretrained.py   # active pretrained path
└── particle_transformer_ee.py              # previous prototype

scripts/
├── run_parallel_training.py
├── run_pbt.py
├── validate_pretrained_sgv_3cat.sh
└── train_sgv_3cat.sh                       # previous ee prototype

tests/
└── test_pbt.py

weaver-core/
├── weaver/train.py                         # checkpoint/LR resume
└── weaver/utils/
    ├── nn/tools.py                         # AMP/controller hook
    └── training_control/                   # controller implementation
```

## Internal documentation

- Weaver controller internals: [`weaver-core/docs/training-controller.md`](weaver-core/docs/training-controller.md)

## Environment bootstrap

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
