# PBT Presets

Reusable YAML presets keep future runs from copy-pasting checkpoint, optimizer, population, and safety policy blocks.

Use them from `configs/experiments/*.yaml` with paths relative to the experiment file, for example:

```yaml
schema_version: 1
presets:
  - ../presets/shared/pretrained_epoch17_ranger_parquet.yaml
  - ../presets/resources/local_8gpu.yaml
  - ../presets/population/members_8.yaml
  - ../presets/pbt/guarded_smooth_lr_mistag.yaml

experiment:
  name: my_run
```

Experiment files override preset fields recursively. Lists, including `population`, are replaced as whole lists.
