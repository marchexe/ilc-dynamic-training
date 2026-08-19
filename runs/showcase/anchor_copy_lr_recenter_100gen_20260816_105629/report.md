# anchor_copy_lr_recenter_100gen_20260816_105629

## Results
- Evaluation type: `proxy`
- Validation dataset: `/data/suehara/part/march/datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet`
- Validation suffix: `val50k_tail`
- Validation sample count: 150000
- Controller objective: mean predefined fixed-WP mistag percent (lower is better; not a HEP metric)
- Configured PBT selection metric: `validation_total_reference_mistag_geomean_percent` (min)
- **`total_mistag_score` (sqrt(ctag_score * btag_score)) is this run's PBT ranking metric** -- ctag_score/btag_score are its two components, shown for diagnosis, never used for ranking on their own.
- Measured baseline: n/a
- Configured reference: n/a
- Final checkpoint controller objective: 1.00154 by `lr_9e-6`
- Global best configured metric: 0.388359 by `lr_3e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_100gen_20260816_105629/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_3e-6`, generation 69), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_9e-6`, generation 90) -- these are two distinct selection criteria, not the same checkpoint.
  - Validation: `/data/suehara/part/march/datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet` (`val50k_tail`), 150000 samples
- [Physics performance](plots/report/physics_performance.png)
- [C-tag mistag CSV](plots/report/ctag_mistag_tables.csv)
- [B-tag mistag CSV](plots/report/btag_mistag_tables.csv)

## PBT Population and Selection
- [Population and selection](plots/pbt_population_selection.png)
- Ranking metric: `validation_total_reference_mistag_geomean_percent` (min); winner = each generation's authoritative decision winner (the member that actually drove that generation's exploit/anchor/global-best outcome), never re-derived from total_mistag_score.
- Winner-timeline decision markers: `^` accepted_new_anchor, `o` reused_previous_anchor, `v` rewound_to_previous_anchor.

## Mistag Score Evolution
- [Mistag score evolution](plots/mistag_score_evolution.png)
- **`total_mistag_score` (sqrt(ctag_score * btag_score)) is this run's PBT ranking metric** -- the thick line above is the ranking metric itself, not just a diagnostic summary.
- Baseline point: not available for this run.

## Learning-Rate Lineage
- [Learning-rate lineage](plots/learning_rate_lineage.png)
- Heavy edge = an applied donor->recipient checkpoint copy (events.jsonl, applied=True only); light edge = a member continuing its own branch.

## Learning Rate vs. Mistag Score Correlation
- [Training dynamics and within-generation LR analysis](plots/learning_rate_mistag_correlation.png)
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=400, Pearson r=-0.023 (95% CI -0.098 to 0.057), Spearman rho=-0.028 (95% CI -0.111 to 0.055)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=-0.057, Spearman rho=-0.284
- Best checkpoint by tier: control: `lr_9e-6` gen 99 (0.397588), full_holdout: `lr_14e-6` gen 99 (0.387791)
- Best-checkpoint agreement across tiers: DISAGREE
- Control-selected global best has not been evaluated on monitor/full yet.
- Corroboration status: **provisional**
  - monitor: not available (baseline or selected checkpoint not evaluated on this tier)
  - full: not available (baseline or selected checkpoint not evaluated on this tier)
- No proxy-overfitting cases detected (control improved while monitor did not) in the paired generations evaluated so far.

## Model Selection Scores
- Final generation: 99
- All mistag/score values in percent (lower is better); status marks the generation's winner and/or the persisted anchor member.

| member | bc@0.8 | bd@0.8 | bc@0.9 | bd@0.9 | cb@0.5 | cd@0.5 | cb@0.8 | cd@0.8 | ctag_score | btag_score | total_mistag_score | LR | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_14e-6 | 0.1826 | 0.07619 | 3.021 | 0.1985 | 0.5223 | 0.06817 | 2.708 | 1.179 | 0.5806 | 0.3022 | 0.4189 | 1.34e-05 | - |
| lr_3e-6 | 0.1826 | 0.07619 | 3.027 | 0.1985 | 0.5203 | 0.06817 | 2.706 | 1.175 | 0.5795 | 0.3024 | 0.4186 | 8.92e-06 | anchor |
| lr_6e-6 | 0.1746 | 0.07219 | 3.179 | 0.1905 | 0.5119 | 0.05816 | 2.668 | 1.163 | 0.5513 | 0.2956 | 0.4037 | 1e-05 | - |
| lr_9e-6 | 0.1886 | 0.06607 | 3.157 | 0.1822 | 0.513 | 0.05406 | 2.678 | 1.173 | 0.5433 | 0.291 | 0.3976 | 1.11e-05 | winner |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_14e-6 | 0.4419 | 0.6095 | 0.3204 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 1 | lr_14e-6 | 0.4414 | 0.613 | 0.3179 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 2 | lr_9e-6 | 0.4373 | 0.601 | 0.3181 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 3 | lr_3e-6 | 0.4263 | 0.5942 | 0.3058 | 1.12e-05 | 1.4e-05 | 1.12e-05 | accepted_new_anchor | no |
| 4 | lr_9e-6 | 0.4172 | 0.5832 | 0.2985 | 1.12e-05 | 1.12e-05 | 1.12e-05 | accepted_new_anchor | no |
| 5 | lr_6e-6 | 0.4183 | 0.5704 | 0.3068 | 1.008e-05 | 1.12e-05 | 1.12e-05 | rewound_to_previous_anchor | no |
| 6 | lr_9e-6 | 0.4188 | 0.5784 | 0.3033 | 1.12e-05 | 1.12e-05 | 1.12e-05 | rewound_to_previous_anchor | no |
| 7 | lr_9e-6 | 0.4131 | 0.577 | 0.2958 | 1.12e-05 | 1.12e-05 | 1.12e-05 | accepted_new_anchor | no |
| 8 | lr_3e-6 | 0.4123 | 0.5808 | 0.2927 | 8.96e-06 | 1.12e-05 | 8.96e-06 | accepted_new_anchor | no |
| 9 | lr_14e-6 | 0.4123 | 0.5689 | 0.2988 | 1.075e-05 | 8.96e-06 | 8.96e-06 | rewound_to_previous_anchor | no |
| 10 | lr_9e-6 | 0.4215 | 0.5748 | 0.309 | 8.96e-06 | 8.96e-06 | 8.96e-06 | rewound_to_previous_anchor | no |
| 11 | lr_6e-6 | 0.4176 | 0.5769 | 0.3022 | 8.064e-06 | 8.96e-06 | 8.96e-06 | rewound_to_previous_anchor | no |
| 12 | lr_14e-6 | 0.4093 | 0.5613 | 0.2984 | 1.075e-05 | 8.96e-06 | 1.075e-05 | accepted_new_anchor | no |
| 13 | lr_14e-6 | 0.4125 | 0.573 | 0.297 | 1.29e-05 | 1.075e-05 | 1.075e-05 | rewound_to_previous_anchor | no |
| 14 | lr_14e-6 | 0.4096 | 0.5702 | 0.2942 | 1.29e-05 | 1.075e-05 | 1.075e-05 | rewound_to_previous_anchor | no |
| 15 | lr_6e-6 | 0.4098 | 0.5517 | 0.3044 | 9.677e-06 | 1.075e-05 | 1.075e-05 | rewound_to_previous_anchor | no |
| 16 | lr_9e-6 | 0.4147 | 0.5678 | 0.3029 | 1.075e-05 | 1.075e-05 | 1.075e-05 | rewound_to_previous_anchor | no |
| 17 | lr_9e-6 | 0.4138 | 0.5699 | 0.3005 | 1.075e-05 | 1.075e-05 | 1.075e-05 | rewound_to_previous_anchor | no |
| 18 | lr_14e-6 | 0.4127 | 0.5663 | 0.3007 | 1.29e-05 | 1.075e-05 | 1.075e-05 | rewound_to_previous_anchor | no |
| 19 | lr_6e-6 | 0.4053 | 0.5568 | 0.2951 | 9.677e-06 | 1.075e-05 | 9.677e-06 | accepted_new_anchor | no |
| 20 | lr_14e-6 | 0.4112 | 0.5609 | 0.3014 | 1.161e-05 | 9.677e-06 | 9.677e-06 | rewound_to_previous_anchor | no |
| 21 | lr_6e-6 | 0.4132 | 0.5664 | 0.3015 | 8.709e-06 | 9.677e-06 | 9.677e-06 | rewound_to_previous_anchor | no |
| 22 | lr_3e-6 | 0.4102 | 0.5763 | 0.2919 | 7.741e-06 | 9.677e-06 | 9.677e-06 | rewound_to_previous_anchor | no |
| 23 | lr_9e-6 | 0.4052 | 0.5605 | 0.293 | 9.677e-06 | 9.677e-06 | 9.677e-06 | accepted_new_anchor | no |
| 24 | lr_14e-6 | 0.3983 | 0.5449 | 0.2912 | 1.161e-05 | 9.677e-06 | 1.161e-05 | accepted_new_anchor | no |
| 25 | lr_14e-6 | 0.409 | 0.566 | 0.2955 | 1.393e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 26 | lr_9e-6 | 0.4011 | 0.5617 | 0.2864 | 1.161e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 27 | lr_3e-6 | 0.4083 | 0.5637 | 0.2957 | 9.29e-06 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 28 | lr_14e-6 | 0.4159 | 0.5696 | 0.3037 | 1.393e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 29 | lr_9e-6 | 0.4099 | 0.5563 | 0.302 | 1.161e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 30 | lr_14e-6 | 0.4124 | 0.5666 | 0.3002 | 1.393e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 31 | lr_6e-6 | 0.4004 | 0.5266 | 0.3044 | 1.045e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 32 | lr_6e-6 | 0.4072 | 0.5594 | 0.2965 | 1.045e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 33 | lr_14e-6 | 0.4031 | 0.56 | 0.2902 | 1.393e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 34 | lr_6e-6 | 0.4049 | 0.5483 | 0.299 | 1.045e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 35 | lr_9e-6 | 0.4031 | 0.5586 | 0.2909 | 1.161e-05 | 1.161e-05 | 1.161e-05 | rewound_to_previous_anchor | no |
| 36 | lr_14e-6 | 0.3934 | 0.551 | 0.2808 | 1.393e-05 | 1.161e-05 | 1.393e-05 | accepted_new_anchor | no |
| 37 | lr_6e-6 | 0.4102 | 0.5532 | 0.3041 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 38 | lr_9e-6 | 0.4022 | 0.5381 | 0.3007 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 39 | lr_14e-6 | 0.4002 | 0.5658 | 0.2831 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 40 | lr_3e-6 | 0.4171 | 0.5801 | 0.2999 | 1.115e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 41 | lr_9e-6 | 0.4104 | 0.5597 | 0.3009 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 42 | lr_6e-6 | 0.4098 | 0.5541 | 0.3031 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 43 | lr_14e-6 | 0.4107 | 0.5608 | 0.3007 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 44 | lr_3e-6 | 0.4069 | 0.5528 | 0.2994 | 1.115e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 45 | lr_3e-6 | 0.4109 | 0.5576 | 0.3028 | 1.115e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 46 | lr_14e-6 | 0.4082 | 0.5541 | 0.3007 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 47 | lr_3e-6 | 0.4029 | 0.5484 | 0.2961 | 1.115e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 48 | lr_14e-6 | 0.399 | 0.5507 | 0.2891 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 49 | lr_9e-6 | 0.416 | 0.5558 | 0.3114 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 50 | lr_9e-6 | 0.4119 | 0.5427 | 0.3126 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 51 | lr_3e-6 | 0.4041 | 0.5467 | 0.2987 | 1.115e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 52 | lr_14e-6 | 0.4064 | 0.5438 | 0.3036 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 53 | lr_14e-6 | 0.4089 | 0.5622 | 0.2974 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 54 | lr_9e-6 | 0.3904 | 0.5356 | 0.2846 | 1.393e-05 | 1.393e-05 | 1.393e-05 | accepted_new_anchor | no |
| 55 | lr_6e-6 | 0.4089 | 0.5567 | 0.3003 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 56 | lr_14e-6 | 0.4135 | 0.5572 | 0.3069 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 57 | lr_6e-6 | 0.4104 | 0.5535 | 0.3044 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 58 | lr_9e-6 | 0.396 | 0.5483 | 0.2861 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 59 | lr_9e-6 | 0.4037 | 0.5536 | 0.2944 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 60 | lr_6e-6 | 0.4131 | 0.5596 | 0.305 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 61 | lr_14e-6 | 0.4032 | 0.5537 | 0.2937 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 62 | lr_3e-6 | 0.4035 | 0.5507 | 0.2956 | 1.115e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 63 | lr_6e-6 | 0.4059 | 0.5561 | 0.2963 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 64 | lr_9e-6 | 0.4057 | 0.5429 | 0.3031 | 1.393e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 65 | lr_14e-6 | 0.4018 | 0.5391 | 0.2995 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 66 | lr_14e-6 | 0.4174 | 0.567 | 0.3073 | 1.4e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 67 | lr_6e-6 | 0.4058 | 0.5558 | 0.2962 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 68 | lr_6e-6 | 0.4118 | 0.5635 | 0.3009 | 1.254e-05 | 1.393e-05 | 1.393e-05 | rewound_to_previous_anchor | no |
| 69 | lr_3e-6 | 0.3884 | 0.5462 | 0.2761 | 1.115e-05 | 1.393e-05 | 1.115e-05 | accepted_new_anchor | no |
| 70 | lr_9e-6 | 0.3991 | 0.5584 | 0.2852 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 71 | lr_3e-6 | 0.3972 | 0.5479 | 0.288 | 8.918e-06 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 72 | lr_14e-6 | 0.3997 | 0.5362 | 0.2979 | 1.338e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 73 | lr_6e-6 | 0.4046 | 0.5476 | 0.2989 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 74 | lr_3e-6 | 0.4091 | 0.5469 | 0.3061 | 8.918e-06 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 75 | lr_9e-6 | 0.4044 | 0.5547 | 0.2948 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 76 | lr_14e-6 | 0.404 | 0.542 | 0.3011 | 1.338e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 77 | lr_9e-6 | 0.4035 | 0.5541 | 0.2938 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 78 | lr_9e-6 | 0.4093 | 0.5529 | 0.3029 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 79 | lr_6e-6 | 0.4078 | 0.5517 | 0.3014 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 80 | lr_6e-6 | 0.3998 | 0.5367 | 0.2978 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 81 | lr_6e-6 | 0.3981 | 0.5337 | 0.2969 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 82 | lr_3e-6 | 0.3934 | 0.5312 | 0.2914 | 8.918e-06 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 83 | lr_9e-6 | 0.3979 | 0.5433 | 0.2914 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 84 | lr_9e-6 | 0.4073 | 0.5573 | 0.2977 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 85 | lr_14e-6 | 0.3994 | 0.5497 | 0.2902 | 1.338e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 86 | lr_14e-6 | 0.4096 | 0.5634 | 0.2978 | 1.338e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 87 | lr_6e-6 | 0.4014 | 0.5525 | 0.2917 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 88 | lr_6e-6 | 0.4106 | 0.561 | 0.3006 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 89 | lr_6e-6 | 0.4152 | 0.5632 | 0.3062 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 90 | lr_3e-6 | 0.4006 | 0.5498 | 0.292 | 8.918e-06 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 91 | lr_9e-6 | 0.4152 | 0.5693 | 0.3027 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 92 | lr_9e-6 | 0.4077 | 0.5631 | 0.2952 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 93 | lr_14e-6 | 0.4022 | 0.5612 | 0.2882 | 1.338e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 94 | lr_9e-6 | 0.4057 | 0.5442 | 0.3024 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 95 | lr_6e-6 | 0.4011 | 0.5455 | 0.2949 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 96 | lr_6e-6 | 0.4015 | 0.5411 | 0.298 | 1.003e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 97 | lr_3e-6 | 0.4025 | 0.536 | 0.3023 | 8.918e-06 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 98 | lr_14e-6 | 0.4025 | 0.5502 | 0.2945 | 1.338e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |
| 99 | lr_9e-6 | 0.3976 | 0.5433 | 0.291 | 1.115e-05 | 1.115e-05 | 1.115e-05 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_14e-6` -> `lr_3e-6`, donor metric 0.441945, recipient metric 0.448545, LR 3e-06 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_6e-6`, donor metric 0.441945, recipient metric 0.446311, LR 6e-06 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_9e-6`, donor metric 0.441945, recipient metric 0.450596, LR 9e-06 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_14e-6`, donor metric 0.441945, recipient metric 0.441945, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_14e-6` -> `lr_3e-6`, donor metric 0.441431, recipient metric 0.442405, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_14e-6` -> `lr_6e-6`, donor metric 0.441431, recipient metric 0.445492, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_14e-6` -> `lr_9e-6`, donor metric 0.441431, recipient metric 0.452814, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_14e-6` -> `lr_14e-6`, donor metric 0.441431, recipient metric 0.441431, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_3e-6`, donor metric 0.437251, recipient metric 0.440924, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_6e-6`, donor metric 0.437251, recipient metric 0.439924, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_9e-6`, donor metric 0.437251, recipient metric 0.437251, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_14e-6`, donor metric 0.437251, recipient metric 0.437833, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_3e-6`, donor metric 0.426259, recipient metric 0.426259, LR 1.12e-05 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_6e-6`, donor metric 0.426259, recipient metric 0.433854, LR 1.26e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_9e-6`, donor metric 0.426259, recipient metric 0.431111, LR 1.4e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_14e-6`, donor metric 0.426259, recipient metric 0.440737, LR 1.4e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417188, recipient metric 0.418604, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417188, recipient metric 0.423915, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417188, recipient metric 0.417188, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417188, recipient metric 0.426962, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_3e-6`, donor metric 0.420241, recipient metric 0.419125, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_6e-6`, donor metric 0.420241, recipient metric 0.418304, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_9e-6`, donor metric 0.420241, recipient metric 0.420241, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_14e-6`, donor metric 0.420241, recipient metric 0.4206, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_3e-6`, donor metric 0.418848, recipient metric 0.42316, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_6e-6`, donor metric 0.418848, recipient metric 0.423237, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_9e-6`, donor metric 0.418848, recipient metric 0.418848, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_14e-6`, donor metric 0.418848, recipient metric 0.419897, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_3e-6`, donor metric 0.413124, recipient metric 0.413927, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_6e-6`, donor metric 0.413124, recipient metric 0.41319, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_9e-6`, donor metric 0.413124, recipient metric 0.413124, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_14e-6`, donor metric 0.413124, recipient metric 0.413235, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_3e-6` -> `lr_3e-6`, donor metric 0.412278, recipient metric 0.412278, LR 8.96e-06 -> 7.17e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_3e-6` -> `lr_6e-6`, donor metric 0.412278, recipient metric 0.42136, LR 1.01e-05 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_3e-6` -> `lr_9e-6`, donor metric 0.412278, recipient metric 0.414638, LR 1.12e-05 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_3e-6` -> `lr_14e-6`, donor metric 0.412278, recipient metric 0.424653, LR 1.34e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_3e-6` -> `lr_3e-6`, donor metric 0.421229, recipient metric 0.421229, LR 7.17e-06 -> 7.17e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_3e-6` -> `lr_6e-6`, donor metric 0.421229, recipient metric 0.415251, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_3e-6` -> `lr_9e-6`, donor metric 0.421229, recipient metric 0.423622, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_3e-6` -> `lr_14e-6`, donor metric 0.421229, recipient metric 0.41231, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_3e-6` -> `lr_3e-6`, donor metric 0.423003, recipient metric 0.423003, LR 7.17e-06 -> 7.17e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_3e-6` -> `lr_6e-6`, donor metric 0.423003, recipient metric 0.429503, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_3e-6` -> `lr_9e-6`, donor metric 0.423003, recipient metric 0.421455, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_3e-6` -> `lr_14e-6`, donor metric 0.423003, recipient metric 0.422902, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_3e-6` -> `lr_3e-6`, donor metric 0.428046, recipient metric 0.428046, LR 7.17e-06 -> 7.17e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_3e-6` -> `lr_6e-6`, donor metric 0.428046, recipient metric 0.417576, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_3e-6` -> `lr_9e-6`, donor metric 0.428046, recipient metric 0.419675, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_3e-6` -> `lr_14e-6`, donor metric 0.428046, recipient metric 0.423049, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409271, recipient metric 0.410798, LR 7.17e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409271, recipient metric 0.415243, LR 8.06e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409271, recipient metric 0.416583, LR 8.96e-06 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409271, recipient metric 0.409271, LR 1.08e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_3e-6`, donor metric 0.412523, recipient metric 0.413408, LR 8.6e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_6e-6`, donor metric 0.412523, recipient metric 0.413124, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_9e-6`, donor metric 0.412523, recipient metric 0.426529, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_14e-6`, donor metric 0.412523, recipient metric 0.412523, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409593, recipient metric 0.412114, LR 8.6e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409593, recipient metric 0.410697, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409593, recipient metric 0.415072, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409593, recipient metric 0.409593, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_3e-6`, donor metric 0.426722, recipient metric 0.426984, LR 8.6e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_6e-6`, donor metric 0.426722, recipient metric 0.409771, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_9e-6`, donor metric 0.426722, recipient metric 0.42665, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_14e-6`, donor metric 0.426722, recipient metric 0.426722, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415328, recipient metric 0.415576, LR 8.6e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415328, recipient metric 0.416531, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415328, recipient metric 0.414708, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415328, recipient metric 0.415328, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_3e-6`, donor metric 0.41434, recipient metric 0.420141, LR 8.6e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_6e-6`, donor metric 0.41434, recipient metric 0.414786, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_9e-6`, donor metric 0.41434, recipient metric 0.413826, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_14e-6`, donor metric 0.41434, recipient metric 0.41434, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_3e-6`, donor metric 0.412702, recipient metric 0.421014, LR 8.6e-06 -> 8.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_6e-6`, donor metric 0.412702, recipient metric 0.421318, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_9e-6`, donor metric 0.412702, recipient metric 0.422864, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_14e-6`, donor metric 0.412702, recipient metric 0.412702, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_3e-6`, donor metric 0.405347, recipient metric 0.411211, LR 8.6e-06 -> 7.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_6e-6`, donor metric 0.405347, recipient metric 0.405347, LR 9.68e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_9e-6`, donor metric 0.405347, recipient metric 0.413968, LR 1.08e-05 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_14e-6`, donor metric 0.405347, recipient metric 0.41491, LR 1.29e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_3e-6`, donor metric 0.428078, recipient metric 0.42879, LR 7.74e-06 -> 7.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_6e-6`, donor metric 0.428078, recipient metric 0.428078, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_9e-6`, donor metric 0.428078, recipient metric 0.413847, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_14e-6`, donor metric 0.428078, recipient metric 0.411182, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41321, recipient metric 0.41342, LR 7.74e-06 -> 7.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41321, recipient metric 0.41321, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41321, recipient metric 0.414078, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41321, recipient metric 0.417129, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416186, recipient metric 0.410163, LR 7.74e-06 -> 7.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416186, recipient metric 0.416186, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416186, recipient metric 0.41482, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416186, recipient metric 0.41268, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_3e-6`, donor metric 0.405243, recipient metric 0.411141, LR 7.74e-06 -> 7.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_6e-6`, donor metric 0.405243, recipient metric 0.413678, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_9e-6`, donor metric 0.405243, recipient metric 0.405243, LR 9.68e-06 -> 9.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_14e-6`, donor metric 0.405243, recipient metric 0.428406, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_3e-6`, donor metric 0.398315, recipient metric 0.407715, LR 7.74e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_6e-6`, donor metric 0.398315, recipient metric 0.411157, LR 8.71e-06 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_9e-6`, donor metric 0.398315, recipient metric 0.413976, LR 9.68e-06 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_14e-6`, donor metric 0.398315, recipient metric 0.398315, LR 1.16e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409015, recipient metric 0.409182, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409015, recipient metric 0.415935, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409015, recipient metric 0.412731, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409015, recipient metric 0.409015, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415098, recipient metric 0.413663, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415098, recipient metric 0.418704, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415098, recipient metric 0.401106, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415098, recipient metric 0.415098, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_3e-6`, donor metric 0.41365, recipient metric 0.408272, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_6e-6`, donor metric 0.41365, recipient metric 0.420312, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_9e-6`, donor metric 0.41365, recipient metric 0.410176, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_14e-6`, donor metric 0.41365, recipient metric 0.41365, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415893, recipient metric 0.41606, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415893, recipient metric 0.416099, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415893, recipient metric 0.41601, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415893, recipient metric 0.415893, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_3e-6`, donor metric 0.412447, recipient metric 0.415322, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_6e-6`, donor metric 0.412447, recipient metric 0.417908, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_9e-6`, donor metric 0.412447, recipient metric 0.409895, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_14e-6`, donor metric 0.412447, recipient metric 0.412447, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_3e-6`, donor metric 0.412435, recipient metric 0.418418, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_6e-6`, donor metric 0.412435, recipient metric 0.412997, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_9e-6`, donor metric 0.412435, recipient metric 0.412753, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_14e-6`, donor metric 0.412435, recipient metric 0.412435, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_3e-6`, donor metric 0.420229, recipient metric 0.404177, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_6e-6`, donor metric 0.420229, recipient metric 0.400388, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_9e-6`, donor metric 0.420229, recipient metric 0.415055, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_14e-6`, donor metric 0.420229, recipient metric 0.420229, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411752, recipient metric 0.416492, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411752, recipient metric 0.407249, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411752, recipient metric 0.409418, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411752, recipient metric 0.411752, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_3e-6`, donor metric 0.403101, recipient metric 0.415275, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_6e-6`, donor metric 0.403101, recipient metric 0.416702, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_9e-6`, donor metric 0.403101, recipient metric 0.417029, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_14e-6`, donor metric 0.403101, recipient metric 0.403101, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408554, recipient metric 0.415774, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_6e-6`, donor metric 0.408554, recipient metric 0.404891, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_9e-6`, donor metric 0.408554, recipient metric 0.414881, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408554, recipient metric 0.408554, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_3e-6`, donor metric 0.42154, recipient metric 0.404277, LR 9.29e-06 -> 9.29e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_6e-6`, donor metric 0.42154, recipient metric 0.413086, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_9e-6`, donor metric 0.42154, recipient metric 0.403076, LR 1.16e-05 -> 1.16e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_14e-6`, donor metric 0.42154, recipient metric 0.42154, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_3e-6`, donor metric 0.393362, recipient metric 0.410563, LR 9.29e-06 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_6e-6`, donor metric 0.393362, recipient metric 0.4209, LR 1.05e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_9e-6`, donor metric 0.393362, recipient metric 0.422613, LR 1.16e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_14e-6`, donor metric 0.393362, recipient metric 0.393362, LR 1.39e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417601, recipient metric 0.410799, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_6e-6`, donor metric 0.417601, recipient metric 0.410178, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_9e-6`, donor metric 0.417601, recipient metric 0.415513, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417601, recipient metric 0.417601, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408662, recipient metric 0.405149, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_6e-6`, donor metric 0.408662, recipient metric 0.417268, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_9e-6`, donor metric 0.408662, recipient metric 0.402249, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408662, recipient metric 0.408662, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_3e-6`, donor metric 0.40024, recipient metric 0.406033, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_6e-6`, donor metric 0.40024, recipient metric 0.416275, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_9e-6`, donor metric 0.40024, recipient metric 0.409178, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_14e-6`, donor metric 0.40024, recipient metric 0.40024, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417296, recipient metric 0.417144, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_6e-6`, donor metric 0.417296, recipient metric 0.418127, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_9e-6`, donor metric 0.417296, recipient metric 0.417286, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417296, recipient metric 0.417296, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_3e-6`, donor metric 0.419746, recipient metric 0.420614, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_6e-6`, donor metric 0.419746, recipient metric 0.419785, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_9e-6`, donor metric 0.419746, recipient metric 0.410405, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_14e-6`, donor metric 0.419746, recipient metric 0.419746, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_3e-6`, donor metric 0.424683, recipient metric 0.413118, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_6e-6`, donor metric 0.424683, recipient metric 0.409821, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_9e-6`, donor metric 0.424683, recipient metric 0.418028, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_14e-6`, donor metric 0.424683, recipient metric 0.424683, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_3e-6`, donor metric 0.410667, recipient metric 0.411109, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_6e-6`, donor metric 0.410667, recipient metric 0.420395, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_9e-6`, donor metric 0.410667, recipient metric 0.413678, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_14e-6`, donor metric 0.410667, recipient metric 0.410667, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417051, recipient metric 0.406853, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_6e-6`, donor metric 0.417051, recipient metric 0.41678, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_9e-6`, donor metric 0.417051, recipient metric 0.417057, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417051, recipient metric 0.417051, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_3e-6`, donor metric 0.4149, recipient metric 0.410907, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_6e-6`, donor metric 0.4149, recipient metric 0.42133, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_9e-6`, donor metric 0.4149, recipient metric 0.411087, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_14e-6`, donor metric 0.4149, recipient metric 0.4149, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408206, recipient metric 0.409229, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_6e-6`, donor metric 0.408206, recipient metric 0.408435, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_9e-6`, donor metric 0.408206, recipient metric 0.422509, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408206, recipient metric 0.408206, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_3e-6`, donor metric 0.41362, recipient metric 0.402933, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_6e-6`, donor metric 0.41362, recipient metric 0.414247, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_9e-6`, donor metric 0.41362, recipient metric 0.406328, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_14e-6`, donor metric 0.41362, recipient metric 0.41362, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_14e-6` -> `lr_3e-6`, donor metric 0.399014, recipient metric 0.399867, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_14e-6` -> `lr_6e-6`, donor metric 0.399014, recipient metric 0.404863, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_14e-6` -> `lr_9e-6`, donor metric 0.399014, recipient metric 0.408202, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_14e-6` -> `lr_14e-6`, donor metric 0.399014, recipient metric 0.399014, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_14e-6` -> `lr_3e-6`, donor metric 0.42063, recipient metric 0.419473, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_14e-6` -> `lr_6e-6`, donor metric 0.42063, recipient metric 0.420076, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_14e-6` -> `lr_9e-6`, donor metric 0.42063, recipient metric 0.416048, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_14e-6` -> `lr_14e-6`, donor metric 0.42063, recipient metric 0.42063, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_14e-6` -> `lr_3e-6`, donor metric 0.413291, recipient metric 0.413322, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_14e-6` -> `lr_6e-6`, donor metric 0.413291, recipient metric 0.413739, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_14e-6` -> `lr_9e-6`, donor metric 0.413291, recipient metric 0.411853, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_14e-6` -> `lr_14e-6`, donor metric 0.413291, recipient metric 0.413291, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_14e-6` -> `lr_3e-6`, donor metric 0.412139, recipient metric 0.404091, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_14e-6` -> `lr_6e-6`, donor metric 0.412139, recipient metric 0.407733, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_14e-6` -> `lr_9e-6`, donor metric 0.412139, recipient metric 0.412792, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_14e-6` -> `lr_14e-6`, donor metric 0.412139, recipient metric 0.412139, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_14e-6` -> `lr_3e-6`, donor metric 0.406373, recipient metric 0.406798, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_14e-6` -> `lr_6e-6`, donor metric 0.406373, recipient metric 0.407529, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_14e-6` -> `lr_9e-6`, donor metric 0.406373, recipient metric 0.407838, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_14e-6` -> `lr_14e-6`, donor metric 0.406373, recipient metric 0.406373, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408912, recipient metric 0.409634, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_14e-6` -> `lr_6e-6`, donor metric 0.408912, recipient metric 0.409424, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_14e-6` -> `lr_9e-6`, donor metric 0.408912, recipient metric 0.410254, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408912, recipient metric 0.408912, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_3e-6`, donor metric 0.390432, recipient metric 0.419549, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_6e-6`, donor metric 0.390432, recipient metric 0.420827, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_9e-6`, donor metric 0.390432, recipient metric 0.390432, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_14e-6`, donor metric 0.390432, recipient metric 0.41955, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_3e-6`, donor metric 0.419872, recipient metric 0.421126, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_6e-6`, donor metric 0.419872, recipient metric 0.408878, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_9e-6`, donor metric 0.419872, recipient metric 0.419872, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_14e-6`, donor metric 0.419872, recipient metric 0.413713, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417146, recipient metric 0.415268, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417146, recipient metric 0.415416, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417146, recipient metric 0.417146, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417146, recipient metric 0.413538, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_3e-6`, donor metric 0.423676, recipient metric 0.424391, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_6e-6`, donor metric 0.423676, recipient metric 0.410445, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_9e-6`, donor metric 0.423676, recipient metric 0.423676, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_14e-6`, donor metric 0.423676, recipient metric 0.413906, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_3e-6`, donor metric 0.39603, recipient metric 0.415109, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_6e-6`, donor metric 0.39603, recipient metric 0.396838, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_9e-6`, donor metric 0.39603, recipient metric 0.39603, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_14e-6`, donor metric 0.39603, recipient metric 0.407911, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_3e-6`, donor metric 0.40373, recipient metric 0.408815, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_6e-6`, donor metric 0.40373, recipient metric 0.419024, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_9e-6`, donor metric 0.40373, recipient metric 0.40373, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_14e-6`, donor metric 0.40373, recipient metric 0.408413, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_3e-6`, donor metric 0.416019, recipient metric 0.413963, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_6e-6`, donor metric 0.416019, recipient metric 0.413125, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_9e-6`, donor metric 0.416019, recipient metric 0.416019, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_14e-6`, donor metric 0.416019, recipient metric 0.416502, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_9e-6` -> `lr_3e-6`, donor metric 0.412242, recipient metric 0.421305, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_9e-6` -> `lr_6e-6`, donor metric 0.412242, recipient metric 0.422528, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_9e-6` -> `lr_9e-6`, donor metric 0.412242, recipient metric 0.412242, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_9e-6` -> `lr_14e-6`, donor metric 0.412242, recipient metric 0.403232, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_9e-6` -> `lr_3e-6`, donor metric 0.408742, recipient metric 0.403489, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_9e-6` -> `lr_6e-6`, donor metric 0.408742, recipient metric 0.41045, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_9e-6` -> `lr_9e-6`, donor metric 0.408742, recipient metric 0.408742, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_9e-6` -> `lr_14e-6`, donor metric 0.408742, recipient metric 0.408133, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_9e-6` -> `lr_3e-6`, donor metric 0.409758, recipient metric 0.410221, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_9e-6` -> `lr_6e-6`, donor metric 0.409758, recipient metric 0.405908, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_9e-6` -> `lr_9e-6`, donor metric 0.409758, recipient metric 0.409758, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_9e-6` -> `lr_14e-6`, donor metric 0.409758, recipient metric 0.409579, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_9e-6` -> `lr_3e-6`, donor metric 0.405654, recipient metric 0.414029, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_9e-6` -> `lr_6e-6`, donor metric 0.405654, recipient metric 0.412869, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_9e-6` -> `lr_9e-6`, donor metric 0.405654, recipient metric 0.405654, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_9e-6` -> `lr_14e-6`, donor metric 0.405654, recipient metric 0.41371, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_9e-6` -> `lr_3e-6`, donor metric 0.414848, recipient metric 0.410368, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_9e-6` -> `lr_6e-6`, donor metric 0.414848, recipient metric 0.402428, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_9e-6` -> `lr_9e-6`, donor metric 0.414848, recipient metric 0.414848, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_9e-6` -> `lr_14e-6`, donor metric 0.414848, recipient metric 0.401805, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_9e-6` -> `lr_3e-6`, donor metric 0.42067, recipient metric 0.419054, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_9e-6` -> `lr_6e-6`, donor metric 0.42067, recipient metric 0.418289, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_9e-6` -> `lr_9e-6`, donor metric 0.42067, recipient metric 0.42067, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_9e-6` -> `lr_14e-6`, donor metric 0.42067, recipient metric 0.417386, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_9e-6` -> `lr_3e-6`, donor metric 0.407547, recipient metric 0.410065, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_9e-6` -> `lr_6e-6`, donor metric 0.407547, recipient metric 0.40575, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_9e-6` -> `lr_9e-6`, donor metric 0.407547, recipient metric 0.407547, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_9e-6` -> `lr_14e-6`, donor metric 0.407547, recipient metric 0.409162, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_9e-6` -> `lr_3e-6`, donor metric 0.413153, recipient metric 0.415305, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_9e-6` -> `lr_6e-6`, donor metric 0.413153, recipient metric 0.411795, LR 1.25e-05 -> 1.25e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_9e-6` -> `lr_9e-6`, donor metric 0.413153, recipient metric 0.413153, LR 1.39e-05 -> 1.39e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_9e-6` -> `lr_14e-6`, donor metric 0.413153, recipient metric 0.412063, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_3e-6`, donor metric 0.388359, recipient metric 0.388359, LR 1.11e-05 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_6e-6`, donor metric 0.388359, recipient metric 0.422142, LR 1.25e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_9e-6`, donor metric 0.388359, recipient metric 0.422534, LR 1.39e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_14e-6`, donor metric 0.388359, recipient metric 0.410331, LR 1.4e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_3e-6` -> `lr_3e-6`, donor metric 0.399221, recipient metric 0.399221, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_3e-6` -> `lr_6e-6`, donor metric 0.399221, recipient metric 0.4143, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_3e-6` -> `lr_9e-6`, donor metric 0.399221, recipient metric 0.399069, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_3e-6` -> `lr_14e-6`, donor metric 0.399221, recipient metric 0.405512, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_3e-6` -> `lr_3e-6`, donor metric 0.397201, recipient metric 0.397201, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_3e-6` -> `lr_6e-6`, donor metric 0.397201, recipient metric 0.398367, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_3e-6` -> `lr_9e-6`, donor metric 0.397201, recipient metric 0.399722, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_3e-6` -> `lr_14e-6`, donor metric 0.397201, recipient metric 0.402367, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_3e-6` -> `lr_3e-6`, donor metric 0.406144, recipient metric 0.406144, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_3e-6` -> `lr_6e-6`, donor metric 0.406144, recipient metric 0.404926, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_3e-6` -> `lr_9e-6`, donor metric 0.406144, recipient metric 0.40248, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_3e-6` -> `lr_14e-6`, donor metric 0.406144, recipient metric 0.399664, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_3e-6` -> `lr_3e-6`, donor metric 0.410504, recipient metric 0.410504, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_3e-6` -> `lr_6e-6`, donor metric 0.410504, recipient metric 0.404616, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_3e-6` -> `lr_9e-6`, donor metric 0.410504, recipient metric 0.408564, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_3e-6` -> `lr_14e-6`, donor metric 0.410504, recipient metric 0.40632, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_3e-6` -> `lr_3e-6`, donor metric 0.409138, recipient metric 0.409138, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_3e-6` -> `lr_6e-6`, donor metric 0.409138, recipient metric 0.410907, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_3e-6` -> `lr_9e-6`, donor metric 0.409138, recipient metric 0.414346, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_3e-6` -> `lr_14e-6`, donor metric 0.409138, recipient metric 0.414039, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_3e-6` -> `lr_3e-6`, donor metric 0.404718, recipient metric 0.404718, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_3e-6` -> `lr_6e-6`, donor metric 0.404718, recipient metric 0.404577, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_3e-6` -> `lr_9e-6`, donor metric 0.404718, recipient metric 0.404392, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_3e-6` -> `lr_14e-6`, donor metric 0.404718, recipient metric 0.410116, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_3e-6` -> `lr_3e-6`, donor metric 0.406312, recipient metric 0.406312, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_3e-6` -> `lr_6e-6`, donor metric 0.406312, recipient metric 0.406546, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_3e-6` -> `lr_9e-6`, donor metric 0.406312, recipient metric 0.414464, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_3e-6` -> `lr_14e-6`, donor metric 0.406312, recipient metric 0.403993, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_3e-6` -> `lr_3e-6`, donor metric 0.418241, recipient metric 0.418241, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_3e-6` -> `lr_6e-6`, donor metric 0.418241, recipient metric 0.403504, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_3e-6` -> `lr_9e-6`, donor metric 0.418241, recipient metric 0.403471, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_3e-6` -> `lr_14e-6`, donor metric 0.418241, recipient metric 0.418615, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_3e-6` -> `lr_3e-6`, donor metric 0.415317, recipient metric 0.415317, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_3e-6` -> `lr_6e-6`, donor metric 0.415317, recipient metric 0.414956, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_3e-6` -> `lr_9e-6`, donor metric 0.415317, recipient metric 0.409259, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_3e-6` -> `lr_14e-6`, donor metric 0.415317, recipient metric 0.410727, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_3e-6` -> `lr_3e-6`, donor metric 0.414635, recipient metric 0.414635, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_3e-6` -> `lr_6e-6`, donor metric 0.414635, recipient metric 0.407777, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_3e-6` -> `lr_9e-6`, donor metric 0.414635, recipient metric 0.417841, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_3e-6` -> `lr_14e-6`, donor metric 0.414635, recipient metric 0.416297, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_3e-6` -> `lr_3e-6`, donor metric 0.420446, recipient metric 0.420446, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_3e-6` -> `lr_6e-6`, donor metric 0.420446, recipient metric 0.399827, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_3e-6` -> `lr_9e-6`, donor metric 0.420446, recipient metric 0.409059, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_3e-6` -> `lr_14e-6`, donor metric 0.420446, recipient metric 0.413685, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_3e-6` -> `lr_3e-6`, donor metric 0.424341, recipient metric 0.424341, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_3e-6` -> `lr_6e-6`, donor metric 0.424341, recipient metric 0.398078, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_3e-6` -> `lr_9e-6`, donor metric 0.424341, recipient metric 0.424267, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_3e-6` -> `lr_14e-6`, donor metric 0.424341, recipient metric 0.417366, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_3e-6` -> `lr_3e-6`, donor metric 0.393413, recipient metric 0.393413, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_3e-6` -> `lr_6e-6`, donor metric 0.393413, recipient metric 0.41192, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_3e-6` -> `lr_9e-6`, donor metric 0.393413, recipient metric 0.397819, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_3e-6` -> `lr_14e-6`, donor metric 0.393413, recipient metric 0.410983, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_3e-6` -> `lr_3e-6`, donor metric 0.402557, recipient metric 0.402557, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_3e-6` -> `lr_6e-6`, donor metric 0.402557, recipient metric 0.405643, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_3e-6` -> `lr_9e-6`, donor metric 0.402557, recipient metric 0.397891, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_3e-6` -> `lr_14e-6`, donor metric 0.402557, recipient metric 0.398656, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_3e-6` -> `lr_3e-6`, donor metric 0.41261, recipient metric 0.41261, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_3e-6` -> `lr_6e-6`, donor metric 0.41261, recipient metric 0.412416, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_3e-6` -> `lr_9e-6`, donor metric 0.41261, recipient metric 0.407317, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_3e-6` -> `lr_14e-6`, donor metric 0.41261, recipient metric 0.412582, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_3e-6` -> `lr_3e-6`, donor metric 0.407823, recipient metric 0.407823, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_3e-6` -> `lr_6e-6`, donor metric 0.407823, recipient metric 0.399781, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_3e-6` -> `lr_9e-6`, donor metric 0.407823, recipient metric 0.422065, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_3e-6` -> `lr_14e-6`, donor metric 0.407823, recipient metric 0.399373, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_3e-6` -> `lr_3e-6`, donor metric 0.41421, recipient metric 0.41421, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_3e-6` -> `lr_6e-6`, donor metric 0.41421, recipient metric 0.411738, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_3e-6` -> `lr_9e-6`, donor metric 0.41421, recipient metric 0.411776, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_3e-6` -> `lr_14e-6`, donor metric 0.41421, recipient metric 0.409599, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_3e-6` -> `lr_3e-6`, donor metric 0.410789, recipient metric 0.410789, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_3e-6` -> `lr_6e-6`, donor metric 0.410789, recipient metric 0.401425, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_3e-6` -> `lr_9e-6`, donor metric 0.410789, recipient metric 0.409646, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_3e-6` -> `lr_14e-6`, donor metric 0.410789, recipient metric 0.40901, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_3e-6` -> `lr_3e-6`, donor metric 0.415768, recipient metric 0.415768, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_3e-6` -> `lr_6e-6`, donor metric 0.415768, recipient metric 0.410619, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_3e-6` -> `lr_9e-6`, donor metric 0.415768, recipient metric 0.411987, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_3e-6` -> `lr_14e-6`, donor metric 0.415768, recipient metric 0.412956, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_3e-6` -> `lr_3e-6`, donor metric 0.415383, recipient metric 0.415383, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_3e-6` -> `lr_6e-6`, donor metric 0.415383, recipient metric 0.415237, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_3e-6` -> `lr_9e-6`, donor metric 0.415383, recipient metric 0.415513, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_3e-6` -> `lr_14e-6`, donor metric 0.415383, recipient metric 0.41545, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_3e-6` -> `lr_3e-6`, donor metric 0.400649, recipient metric 0.400649, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_3e-6` -> `lr_6e-6`, donor metric 0.400649, recipient metric 0.408052, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_3e-6` -> `lr_9e-6`, donor metric 0.400649, recipient metric 0.403547, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_3e-6` -> `lr_14e-6`, donor metric 0.400649, recipient metric 0.41774, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_3e-6` -> `lr_3e-6`, donor metric 0.418176, recipient metric 0.418176, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_3e-6` -> `lr_6e-6`, donor metric 0.418176, recipient metric 0.418003, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_3e-6` -> `lr_9e-6`, donor metric 0.418176, recipient metric 0.415161, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_3e-6` -> `lr_14e-6`, donor metric 0.418176, recipient metric 0.419444, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_3e-6` -> `lr_3e-6`, donor metric 0.407765, recipient metric 0.407765, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_3e-6` -> `lr_6e-6`, donor metric 0.407765, recipient metric 0.410167, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_3e-6` -> `lr_9e-6`, donor metric 0.407765, recipient metric 0.407719, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_3e-6` -> `lr_14e-6`, donor metric 0.407765, recipient metric 0.4101, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_3e-6` -> `lr_3e-6`, donor metric 0.413998, recipient metric 0.413998, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_3e-6` -> `lr_6e-6`, donor metric 0.413998, recipient metric 0.410165, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_3e-6` -> `lr_9e-6`, donor metric 0.413998, recipient metric 0.412404, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_3e-6` -> `lr_14e-6`, donor metric 0.413998, recipient metric 0.402187, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_3e-6` -> `lr_3e-6`, donor metric 0.413365, recipient metric 0.413365, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_3e-6` -> `lr_6e-6`, donor metric 0.413365, recipient metric 0.413698, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_3e-6` -> `lr_9e-6`, donor metric 0.413365, recipient metric 0.405665, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_3e-6` -> `lr_14e-6`, donor metric 0.413365, recipient metric 0.414664, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_3e-6` -> `lr_3e-6`, donor metric 0.403757, recipient metric 0.403757, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_3e-6` -> `lr_6e-6`, donor metric 0.403757, recipient metric 0.40111, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_3e-6` -> `lr_9e-6`, donor metric 0.403757, recipient metric 0.412608, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_3e-6` -> `lr_14e-6`, donor metric 0.403757, recipient metric 0.401753, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_3e-6` -> `lr_3e-6`, donor metric 0.413104, recipient metric 0.413104, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_3e-6` -> `lr_6e-6`, donor metric 0.413104, recipient metric 0.401539, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_3e-6` -> `lr_9e-6`, donor metric 0.413104, recipient metric 0.405575, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_3e-6` -> `lr_14e-6`, donor metric 0.413104, recipient metric 0.412601, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_3e-6` -> `lr_3e-6`, donor metric 0.402542, recipient metric 0.402542, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_3e-6` -> `lr_6e-6`, donor metric 0.402542, recipient metric 0.408661, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_3e-6` -> `lr_9e-6`, donor metric 0.402542, recipient metric 0.410157, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_3e-6` -> `lr_14e-6`, donor metric 0.402542, recipient metric 0.409692, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_3e-6`, donor metric 0.414656, recipient metric 0.414656, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_6e-6`, donor metric 0.414656, recipient metric 0.40605, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_9e-6`, donor metric 0.414656, recipient metric 0.412218, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_14e-6`, donor metric 0.414656, recipient metric 0.402496, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_3e-6` -> `lr_3e-6`, donor metric 0.418577, recipient metric 0.418577, LR 8.92e-06 -> 8.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_3e-6` -> `lr_6e-6`, donor metric 0.418577, recipient metric 0.403691, LR 1e-05 -> 1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_3e-6` -> `lr_9e-6`, donor metric 0.418577, recipient metric 0.397588, LR 1.11e-05 -> 1.11e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_3e-6` -> `lr_14e-6`, donor metric 0.418577, recipient metric 0.418892, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- [Skipped exploits (significance gating)](plots/report/skipped_exploits.csv) -- 0 donor->recipient replacement(s) declined for insufficient significance

## Method
- Method: `anchor_copy_lr_recenter`
- Population: 4 trials
- Training interval: 120000 samples/trial chunk (1x samples_per_epoch)
- Evaluation interval: every 1 training chunk(s), 150000 validation samples
- Exploit interval: every 1 training chunk(s)
- Exploit significance gating: disabled (nominal rank order only)
- Burn-in: 0 generation(s) (observe-only, no exploit/controller LR action applied)
- Monitor-tier cadence: disabled generation(s), all population members, read-only
- Full-tier cadence: 20 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `d87fbe39f71b00f616fbe550e95f97942636b66f`
- Git dirty: `False`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_100gen.yaml', '--slots', 'iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3', '--experiment-name', 'anchor_copy_lr_recenter_100gen_20260816_105629']`
- [manifest.json](manifest.json)
- [resolved_config.yaml](resolved_config.yaml)
- [events.jsonl](events.jsonl)
- [metrics.csv](metrics.csv)
- [tiered_metrics.csv](tiered_metrics.csv)
- [summary.json](summary.json)

## Caveats
- Proxy, smoke, and full validation results are reported as distinct evaluation types and should not be mixed in one scorecard.
- Configured reference values are not treated as measured baselines unless a successful runtime initial evaluation exists.
- Control-tier evidence alone is 'provisional' -- see Proxy Validation above. It is never a substitute for monitor/full corroboration.
- No data-loader shutdown-race warnings observed across 440 evaluation(s).
