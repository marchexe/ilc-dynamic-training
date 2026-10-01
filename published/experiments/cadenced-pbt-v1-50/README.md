# Cadenced PBT v1 — 50 epochs

A completed pretrained population-based training run with one-epoch generations, copy and learning-rate mutation opportunities after every eligible epoch, a two-epoch warm-up, and five population members.

- Status: `completed`
- Method: `cadenced_pbt_v1`
- Training seed: `12345`
- Canonical server path: `runs/pbt/cadenced_pbt_v1_50epochs`
- Generation length: `1 epoch`
- Copy opportunity: `every epoch after warm-up`
- LR mutation opportunity: `every epoch after warm-up`
- Warm-up: `2 epochs`
- Population: `5`
- Initialization: `pretrained epoch 17`

## Main results

- Final-10 current-best mean: `0.3323913294200741`
- Global best: `0.3309306510368319`
- Global-best completed / Weaver epoch: `42` / `59`
- Global-best member / LR: `lr_5_75e-6` / `1.9200000000000003e-05`

## PBT dynamics

- Exploit events: `32`
- LR mutations: `28`
- Copy-only events: `4`
- Mutation factors: `×0.8` on `17` events; `×1.2` on `11` events
- Initial-lineage collapse: completed epoch `5`
- Five unique live LRs remained present at every completed epoch.

After completed epoch 5, every live member descended from the initial `lr_14e-6` ancestry. The plots describe recorded temporal relationships; they do not claim that an individual LR mutation caused a later performance change.

## Checkpoint selection

The global-best checkpoint is selected by the primary full-reference optimization metric. The final-best checkpoint is the top-ranked member at completed epoch 50; the two meanings are kept separate.

The supplementary physics report's best-physics checkpoint is not the primary PBT global-best checkpoint. The figures are retained as recorded and do not redefine the primary selection.

## Contents

`metrics.csv`, `lr_history.csv`, `decision_history.csv`, and `exploit_history.csv` are generated or copied from recorded evidence. `plots/` contains standalone presentation figures and verified physics outputs. `models/` contains explicitly selected release assets.

## Provenance

Source manifest SHA256: `57dcee6b08b7e429adcee6c7c88e2af633fa1a932875bd340a1c88d577ab82a5`
