# Presets

Reusable configuration blocks, listed under `presets:` in an experiment file with paths
relative to that file. Experiment files override preset fields recursively; lists, such
as `population`, are replaced whole. Configurations may also inherit another experiment
file directly (for example `windowed_pbt_v2.yaml` inherits `foundation_fixed_lr_50epochs.yaml`).

| Preset | Content |
|---|---|
| `shared/pretrained_epoch17_ranger_10m_parquet_proxy_control.yaml` | pretrained epoch-17 start, Ranger, parquet data, proxy validation tiers |
| `resources/local_8gpu.yaml` | eight local GPUs |
| `population/fixed_lr_8_tail.yaml` | eight fixed learning rates, 3e-6 to 1.4e-5 |
| `pbt/fixed_lr_grid_mistag.yaml` | fixed-learning-rate policy ranked by composite mistag |

Used by `configs/experiments/pretrained_fixed_lr_8gpu_representative60k_shadow.yaml`.
