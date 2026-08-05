# pretrained_pbt_4gpu_night_controller_active_20260805_015318

## Results
- Evaluation type: `proxy`
- Validation dataset: `/data/suehara/part/march/datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet`
- Validation suffix: `val5k_tail`
- Validation sample count: 15000
- Controller objective: mean predefined fixed-WP mistag percent (lower is better; not a HEP metric)
- Configured PBT selection metric: `validation_working_point_mistag_percent` (min)
- Measured baseline: 1.14
- Configured reference: 1.14
- Final checkpoint controller objective: 1.00095 by `lr_3e-6`
- Global best configured metric: 0.954244 by `lr_6e-6`
- Delta vs measured baseline: 16.2944%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/pretrained_pbt_4gpu_night_controller_active_20260805_015318/checkpoints/global_best_state.pt`

## Training Evolution
- [Training evolution](plots/training_evolution.png)
- [Working-point evolution](plots/working_point_evolution.png)
- `lr_14e-6` samples_seen:LR = 20000:1.4e-05, 40000:1.4e-05, 60000:1.4e-05, 80000:1.4e-05, 100000:1.26e-05, 120000:1.26e-05, 140000:1.26e-05, 160000:1.26e-05, 180000:1.26e-05, 200000:1.26e-05, 220000:1.26e-05, 240000:1.26e-05, 260000:1.26e-05, 280000:1.2e-05, 300000:1.2e-05, 320000:1.2e-05, 340000:1.2e-05, 360000:1.2e-05, 380000:1.2e-05, 400000:1.2e-05, 420000:1.2e-05, 440000:1.2e-05, 460000:1.2e-05, 480000:1.2e-05, 500000:1.2e-05, 520000:1.2e-05, 540000:1.2e-05, 560000:1.2e-05, 580000:1.2e-05, 600000:1.2e-05, 620000:1.2e-05, 640000:1.14e-05, 660000:1.14e-05, 680000:1.14e-05, 700000:1.14e-05, 720000:1.14e-05, 740000:1.08e-05, 760000:1.08e-05, 780000:1.08e-05, 800000:1.03e-05, 820000:1.03e-05, 840000:1.03e-05, 860000:1.03e-05, 880000:1.03e-05, 900000:1.03e-05, 920000:1.11e-05, 940000:1.11e-05, 960000:1.05e-05, 980000:1.05e-05, 1000000:1.05e-05, 1020000:1.05e-05, 1040000:1e-05, 1060000:1e-05, 1080000:1e-05, 1100000:1e-05, 1120000:1e-05, 1140000:1e-05, 1160000:9.5e-06, 1180000:9.5e-06, 1200000:9.5e-06, 1220000:9.5e-06, 1240000:9.5e-06, 1260000:9.5e-06, 1280000:9.03e-06, 1300000:9.03e-06, 1320000:9.03e-06, 1340000:9.03e-06, 1360000:8.58e-06, 1380000:8.58e-06, 1400000:8.58e-06, 1420000:8.58e-06, 1440000:8.58e-06, 1460000:8.58e-06, 1480000:8.58e-06, 1500000:8.58e-06, 1520000:8.58e-06, 1540000:8.15e-06, 1560000:8.15e-06, 1580000:8.15e-06, 1600000:8.15e-06, 1620000:8.15e-06, 1640000:8.15e-06, 1660000:8.15e-06, 1680000:8.15e-06, 1700000:8.15e-06, 1720000:8.15e-06, 1740000:7.74e-06, 1760000:7.74e-06, 1780000:7.35e-06, 1800000:7.35e-06
- `lr_3e-6` samples_seen:LR = 20000:3e-06, 40000:3e-06, 60000:3e-06, 80000:3e-06, 100000:3e-06, 120000:3e-06, 140000:3e-06, 160000:3e-06, 180000:3e-06, 200000:3e-06, 220000:3e-06, 240000:3e-06, 260000:9.41e-06, 280000:9.41e-06, 300000:9.41e-06, 320000:9.41e-06, 340000:9.41e-06, 360000:9.41e-06, 380000:8.93e-06, 400000:8.93e-06, 420000:8.93e-06, 440000:8.93e-06, 460000:8.93e-06, 480000:8.93e-06, 500000:8.49e-06, 520000:8.49e-06, 540000:8.49e-06, 560000:8.49e-06, 580000:8.49e-06, 600000:8.49e-06, 620000:8.49e-06, 640000:8.06e-06, 660000:8.06e-06, 680000:8.06e-06, 700000:8.06e-06, 720000:8.06e-06, 740000:8.06e-06, 760000:8.06e-06, 780000:8.06e-06, 800000:8.06e-06, 820000:8.06e-06, 840000:7.66e-06, 860000:7.66e-06, 880000:7.66e-06, 900000:7.66e-06, 920000:7.28e-06, 940000:7.28e-06, 960000:7.28e-06, 980000:6.91e-06, 1000000:6.91e-06, 1020000:6.91e-06, 1040000:6.91e-06, 1060000:6.91e-06, 1080000:6.91e-06, 1100000:6.91e-06, 1120000:6.57e-06, 1140000:6.57e-06, 1160000:6.57e-06, 1180000:6.57e-06, 1200000:6.57e-06, 1220000:6.57e-06, 1240000:6.57e-06, 1260000:6.24e-06, 1280000:6.24e-06, 1300000:6.24e-06, 1320000:6.24e-06, 1340000:6.24e-06, 1360000:6.24e-06, 1380000:6.24e-06, 1400000:5.93e-06, 1420000:5.93e-06, 1440000:5.93e-06, 1460000:5.93e-06, 1480000:5.93e-06, 1500000:5.93e-06, 1520000:5.93e-06, 1540000:5.93e-06, 1560000:5.93e-06, 1580000:6.34e-06, 1600000:6.34e-06, 1620000:6.34e-06, 1640000:6.34e-06, 1660000:6.34e-06, 1680000:6.34e-06, 1700000:6.34e-06, 1720000:6.34e-06, 1740000:6.34e-06, 1760000:6.34e-06, 1780000:6.02e-06, 1800000:6.02e-06
- `lr_6e-6` samples_seen:LR = 20000:6e-06, 40000:6e-06, 60000:6e-06, 80000:6e-06, 100000:6e-06, 120000:6e-06, 140000:6e-06, 160000:6e-06, 180000:6e-06, 200000:6e-06, 220000:6e-06, 240000:5.4e-06, 260000:5.4e-06, 280000:5.4e-06, 300000:5.4e-06, 320000:5.4e-06, 340000:5.4e-06, 360000:5.13e-06, 380000:5.13e-06, 400000:5.13e-06, 420000:5.13e-06, 440000:5.13e-06, 460000:5.13e-06, 480000:5.13e-06, 500000:5.13e-06, 520000:5.13e-06, 540000:5.13e-06, 560000:5.13e-06, 580000:5.13e-06, 600000:5.13e-06, 620000:5.13e-06, 640000:4.87e-06, 660000:4.87e-06, 680000:1.02e-05, 700000:1.02e-05, 720000:1.02e-05, 740000:1.02e-05, 760000:1.02e-05, 780000:1.02e-05, 800000:1.02e-05, 820000:1.02e-05, 840000:9.72e-06, 860000:9.72e-06, 880000:9.72e-06, 900000:9.24e-06, 920000:9.24e-06, 940000:8.77e-06, 960000:8.77e-06, 980000:8.77e-06, 1000000:8.77e-06, 1020000:8.77e-06, 1040000:8.34e-06, 1060000:8.34e-06, 1080000:8.34e-06, 1100000:8.34e-06, 1120000:8.34e-06, 1140000:8.34e-06, 1160000:8.34e-06, 1180000:8.34e-06, 1200000:8.34e-06, 1220000:8.34e-06, 1240000:8.34e-06, 1260000:8.34e-06, 1280000:8.34e-06, 1300000:7.92e-06, 1320000:7.92e-06, 1340000:7.92e-06, 1360000:7.92e-06, 1380000:7.92e-06, 1400000:7.92e-06, 1420000:7.92e-06, 1440000:7.92e-06, 1460000:7.92e-06, 1480000:7.92e-06, 1500000:7.92e-06, 1520000:7.92e-06, 1540000:7.92e-06, 1560000:7.92e-06, 1580000:7.92e-06, 1600000:7.92e-06, 1620000:7.92e-06, 1640000:7.52e-06, 1660000:7.52e-06, 1680000:7.52e-06, 1700000:7.52e-06, 1720000:7.52e-06, 1740000:7.52e-06, 1760000:7.52e-06, 1780000:7.52e-06, 1800000:7.52e-06
- `lr_9e-6` samples_seen:LR = 20000:9e-06, 40000:9e-06, 60000:9e-06, 80000:9e-06, 100000:9e-06, 120000:9e-06, 140000:9e-06, 160000:9e-06, 180000:9e-06, 200000:9e-06, 220000:9e-06, 240000:8.55e-06, 260000:8.55e-06, 280000:8.55e-06, 300000:8.12e-06, 320000:8.12e-06, 340000:8.12e-06, 360000:8.12e-06, 380000:8.12e-06, 400000:8.12e-06, 420000:8.12e-06, 440000:8.12e-06, 460000:8.12e-06, 480000:7.72e-06, 500000:7.72e-06, 520000:7.72e-06, 540000:7.72e-06, 560000:7.72e-06, 580000:7.72e-06, 600000:7.72e-06, 620000:1.08e-05, 640000:1.08e-05, 660000:1.08e-05, 680000:1.08e-05, 700000:1.08e-05, 720000:1.08e-05, 740000:1.08e-05, 760000:1.08e-05, 780000:1.02e-05, 800000:1.02e-05, 820000:1.02e-05, 840000:1.02e-05, 860000:1.02e-05, 880000:1.02e-05, 900000:1.02e-05, 920000:1.02e-05, 940000:1.02e-05, 960000:1.02e-05, 980000:1.02e-05, 1000000:1.02e-05, 1020000:9.72e-06, 1040000:9.72e-06, 1060000:9.72e-06, 1080000:9.72e-06, 1100000:9.72e-06, 1120000:9.72e-06, 1140000:9.72e-06, 1160000:9.24e-06, 1180000:9.24e-06, 1200000:8.77e-06, 1220000:8.77e-06, 1240000:8.77e-06, 1260000:8.77e-06, 1280000:8.77e-06, 1300000:8.77e-06, 1320000:8.77e-06, 1340000:8.77e-06, 1360000:8.34e-06, 1380000:8.34e-06, 1400000:8.34e-06, 1420000:8.34e-06, 1440000:8.34e-06, 1460000:8.34e-06, 1480000:8.34e-06, 1500000:8.34e-06, 1520000:8.34e-06, 1540000:8.34e-06, 1560000:8.34e-06, 1580000:8.34e-06, 1600000:8.34e-06, 1620000:8.34e-06, 1640000:7.92e-06, 1660000:7.92e-06, 1680000:7.92e-06, 1700000:7.52e-06, 1720000:7.52e-06, 1740000:7.52e-06, 1760000:7.52e-06, 1780000:7.52e-06, 1800000:7.52e-06

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 11: `lr_9e-6` -> `lr_3e-6`, donor metric 1.05431, recipient metric 1.15275, LR 3e-06 -> 9.41e-06, mutation `1.1`, weight `population`, optimizer `population`
- generation 29: `lr_14e-6` -> `lr_9e-6`, donor metric 1.00258, recipient metric 1.07741, LR 7.72e-06 -> 1.08e-05, mutation `0.9`, weight `population`, optimizer `population`
- generation 32: `lr_14e-6` -> `lr_6e-6`, donor metric 0.996981, recipient metric 1.1144, LR 4.87e-06 -> 1.02e-05, mutation `0.9`, weight `population`, optimizer `population`
- generation 44: `lr_6e-6` -> `lr_14e-6`, donor metric 0.96022, recipient metric 1.06981, LR 1.03e-05 -> 1.11e-05, mutation `1.2`, weight `population`, optimizer `population`
- generation 77: `lr_6e-6` -> `lr_3e-6`, donor metric 1.02098, recipient metric 1.14629, LR 5.93e-06 -> 6.34e-06, mutation `0.8`, weight `population`, optimizer `population`
- [Skipped exploits (significance gating)](plots/report/skipped_exploits.csv) -- 80 donor->recipient replacement(s) declined for insufficient significance

## Proxy Validation Diagnostics
- [Proxy validation diagnostics](plots/proxy_diagnostics.png)
- control vs. monitor correlation: n=180, Pearson r=0.674, Spearman rho=0.556
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=60, Pearson r=0.560, Spearman rho=0.450
- Best checkpoint by tier: monitor: `lr_3e-6` gen 87 (1.0135), full: `lr_14e-6` gen 89 (1.0313), control: `lr_6e-6` gen 67 (0.954244), full_holdout: `lr_14e-6` gen 89 (1.03229)
- Best-checkpoint agreement across tiers: DISAGREE
- Control-selected global best (`lr_6e-6`, gen 67) measured on other tiers: monitor: 1.02275
- Corroboration status: **monitor-corroborated**
  - monitor: baseline 1.12575 -> selected 1.02275 (improved, 9.14946% relative change)
  - full: not available (baseline or selected checkpoint not evaluated on this tier)
- **20 proxy-overfitting case(s) detected** (control improved, monitor did not):
  - `lr_6e-6` gen 9->11: control 1.08268->0.988039, monitor 1.07975->1.08125
  - `lr_14e-6` gen 25->27: control 1.06919->1.06692, monitor 1.032->1.03425
  - `lr_9e-6` gen 31->33: control 1.13469->1.03713, monitor 1.027->1.0275
  - `lr_9e-6` gen 33->35: control 1.03713->0.988182, monitor 1.0275->1.02825
  - `lr_9e-6` gen 37->39: control 1.05239->0.998196, monitor 1.02475->1.02675
  - `lr_3e-6` gen 39->41: control 1.02471->1.01257, monitor 1.031->1.03225
  - `lr_14e-6` gen 41->43: control 0.996891->0.965091, monitor 1.02225->1.02275
  - `lr_3e-6` gen 41->43: control 1.01257->0.998627, monitor 1.03225->1.0325
  - `lr_6e-6` gen 41->43: control 1.00834->1.00506, monitor 1.023->1.0235
  - `lr_9e-6` gen 47->49: control 1.08021->1.06167, monitor 1.02025->1.0215
  - ... and 10 more (see tiered_metrics.csv)

## Physics Performance
- [Physics performance](plots/report/physics_performance.png)
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- [B-tag mistag CSV](plots/report/btag_mistag_tables.csv)
- [C-tag mistag CSV](plots/report/ctag_mistag_tables.csv)

## Baseline vs. Selected Model
- [Baseline vs. selected mistag](plots/baseline_vs_selected.png)

## Method
- Method: `exploit_mutate`
- Population: 4 trials
- Training interval: 20000 samples/trial chunk (1x samples_per_epoch)
- Evaluation interval: every 1 training chunk(s), 15000 validation samples
- Exploit interval: every 3 training chunk(s)
- Exploit significance gating: 1.0 sigma (combined uncertainty) required before a donor replaces a recipient
- Burn-in: 2 generation(s) (observe-only, no exploit/controller LR action applied)
- Monitor-tier cadence: 2 generation(s), all population members, read-only
- Full-tier cadence: 6 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `6d3084f0f08068116d729719a65b4126be6ef65e`
- Git dirty: `True`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/pretrained_pbt_4gpu_night_controller_active.yaml', '--slots', 'iutgpu01:4,iutgpu01:5,iutgpu01:6,iutgpu01:7', '--experiment-name', 'pretrained_pbt_4gpu_night_controller_active_20260805_015318']`
- [manifest.json](manifest.json)
- [resolved_config.yaml](resolved_config.yaml)
- [events.jsonl](events.jsonl)
- [metrics.csv](metrics.csv)
- [tiered_metrics.csv](tiered_metrics.csv)
- [summary.json](summary.json)

## Caveats
- Proxy, smoke, and full validation results are reported as distinct evaluation types and should not be mixed in one scorecard.
- Configured reference values are not treated as measured baselines unless a successful runtime initial evaluation exists.
- Control-tier evidence alone is 'provisional' -- see Proxy Validation Diagnostics above. It is never a substitute for monitor/full corroboration.
- No data-loader shutdown-race warnings observed across 843 evaluation(s).
