# Cadenced PBT paired-seed replications

Status: prepared and CPU-preflighted; not launched.

The two additional training seeds are `22345` and `32345`, following the
repository's existing `12345`, `22345`, `32345`, ... replicate convention.
For zero-based generation `g`, the worker training seed is
`shared.seed + g`, producing disjoint ranges `22345..22394` and
`32345..32394`. `pbt.seed` remains `2026`; it is not the Weaver/data-order
training seed.

The four resolved configurations and their pairwise unified diffs are in
`resolved/` and `diffs/`. Regenerate and verify them without launching work:

```bash
cd /data/suehara/part/march
.venv/bin/python configs/prepared/cadenced_pbt_v1_seed_replicates_20260926/preflight_check.py
```

Run the production commands below sequentially, after checking that all five
GPUs are free. Do not overlap runs on the same GPU set.

```bash
ssh iutgpu01 'cd /data/suehara/part/march && .venv/bin/python scripts/launch/experiment.py start --config configs/experiments/cadenced_pbt_v1_seed22345.yaml --gpus 0,1,2,3,4'
ssh iutgpu01 'cd /data/suehara/part/march && .venv/bin/python scripts/launch/experiment.py start --config configs/experiments/cadenced_pbt_v1_cadence5_seed22345.yaml --gpus 0,1,2,3,4'
ssh iutgpu01 'cd /data/suehara/part/march && .venv/bin/python scripts/launch/experiment.py start --config configs/experiments/cadenced_pbt_v1_seed32345.yaml --gpus 0,1,2,3,4'
ssh iutgpu01 'cd /data/suehara/part/march && .venv/bin/python scripts/launch/experiment.py start --config configs/experiments/cadenced_pbt_v1_cadence5_seed32345.yaml --gpus 0,1,2,3,4'
```
