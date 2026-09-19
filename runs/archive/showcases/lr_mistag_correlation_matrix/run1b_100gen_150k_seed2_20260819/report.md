# anchor_copy_lr_recenter_100gen_seed2_20260819_111523

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
- Final checkpoint controller objective: 1.00403 by `lr_3e-6`
- Global best configured metric: 0.391532 by `lr_6e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_100gen_seed2_20260819_111523/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_6e-6`, generation 37), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_14e-6`, generation 47) -- these are two distinct selection criteria, not the same checkpoint.
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=400, Pearson r=-0.043 (95% CI -0.122 to 0.034), Spearman rho=-0.042 (95% CI -0.111 to 0.026)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=0.388, Spearman rho=0.111
- Best checkpoint by tier: control: `lr_3e-6` gen 39 (0.40583), full_holdout: `lr_14e-6` gen 99 (0.39229)
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
| lr_14e-6 | 0.1809 | 0.07825 | 3.237 | 0.1986 | 0.5048 | 0.06621 | 2.67 | 1.174 | 0.5689 | 0.3089 | 0.4192 | 1.09e-05 | - |
| lr_3e-6 | 0.1943 | 0.07224 | 3.176 | 0.1926 | 0.5124 | 0.06221 | 2.691 | 1.132 | 0.5582 | 0.3044 | 0.4122 | 7.26e-06 | winner |
| lr_6e-6 | 0.179 | 0.07618 | 3.231 | 0.1965 | 0.5011 | 0.06816 | 2.672 | 1.181 | 0.5729 | 0.305 | 0.418 | 8.16e-06 | anchor |
| lr_9e-6 | 0.1788 | 0.07807 | 3.184 | 0.2022 | 0.5143 | 0.06606 | 2.672 | 1.147 | 0.5681 | 0.3079 | 0.4182 | 9.07e-06 | - |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_14e-6 | 0.4432 | 0.6259 | 0.3138 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 1 | lr_6e-6 | 0.436 | 0.6086 | 0.3124 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 2 | lr_14e-6 | 0.4231 | 0.5938 | 0.3014 | 1.4e-05 | 1.26e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 3 | lr_6e-6 | 0.4276 | 0.6058 | 0.3017 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 4 | lr_14e-6 | 0.4278 | 0.592 | 0.3092 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 5 | lr_14e-6 | 0.4256 | 0.585 | 0.3096 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 6 | lr_9e-6 | 0.4292 | 0.5899 | 0.3123 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 7 | lr_14e-6 | 0.4272 | 0.5903 | 0.3091 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 8 | lr_9e-6 | 0.4271 | 0.5969 | 0.3057 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 9 | lr_14e-6 | 0.4246 | 0.5946 | 0.3032 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 10 | lr_9e-6 | 0.4288 | 0.5949 | 0.3091 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 11 | lr_9e-6 | 0.4242 | 0.59 | 0.3051 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 12 | lr_14e-6 | 0.4322 | 0.593 | 0.315 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 13 | lr_6e-6 | 0.4335 | 0.594 | 0.3163 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 14 | lr_14e-6 | 0.417 | 0.5801 | 0.2998 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 15 | lr_14e-6 | 0.4146 | 0.573 | 0.3 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 16 | lr_14e-6 | 0.4098 | 0.5599 | 0.3 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 17 | lr_3e-6 | 0.4215 | 0.5815 | 0.3056 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 18 | lr_14e-6 | 0.4156 | 0.5716 | 0.3021 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 19 | lr_9e-6 | 0.4131 | 0.5626 | 0.3033 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 20 | lr_6e-6 | 0.4094 | 0.5717 | 0.2932 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 21 | lr_6e-6 | 0.4188 | 0.569 | 0.3082 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 22 | lr_14e-6 | 0.4119 | 0.5611 | 0.3023 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 23 | lr_14e-6 | 0.4244 | 0.5812 | 0.3099 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 24 | lr_9e-6 | 0.4079 | 0.563 | 0.2955 | 1.26e-05 | 1.26e-05 | 1.26e-05 | accepted_new_anchor | no |
| 25 | lr_3e-6 | 0.4117 | 0.5558 | 0.305 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 26 | lr_9e-6 | 0.411 | 0.5669 | 0.2979 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 27 | lr_9e-6 | 0.4107 | 0.5755 | 0.293 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 28 | lr_6e-6 | 0.4172 | 0.5631 | 0.3091 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 29 | lr_3e-6 | 0.4027 | 0.5457 | 0.2972 | 1.008e-05 | 1.26e-05 | 1.008e-05 | accepted_new_anchor | no |
| 30 | lr_6e-6 | 0.4056 | 0.5561 | 0.2958 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 31 | lr_9e-6 | 0.4102 | 0.5635 | 0.2986 | 1.008e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 32 | lr_6e-6 | 0.403 | 0.5647 | 0.2876 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 33 | lr_3e-6 | 0.4089 | 0.5652 | 0.2958 | 8.064e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 34 | lr_14e-6 | 0.4029 | 0.5633 | 0.2881 | 1.21e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 35 | lr_3e-6 | 0.4036 | 0.5574 | 0.2922 | 8.064e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 36 | lr_9e-6 | 0.4076 | 0.5768 | 0.288 | 1.008e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 37 | lr_6e-6 | 0.3915 | 0.5388 | 0.2845 | 9.072e-06 | 1.008e-05 | 9.072e-06 | accepted_new_anchor | no |
| 38 | lr_3e-6 | 0.4068 | 0.5577 | 0.2968 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 39 | lr_3e-6 | 0.4058 | 0.5454 | 0.302 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 40 | lr_3e-6 | 0.4103 | 0.5622 | 0.2994 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 41 | lr_6e-6 | 0.4039 | 0.5493 | 0.2969 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 42 | lr_14e-6 | 0.4018 | 0.5435 | 0.2971 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 43 | lr_14e-6 | 0.4028 | 0.5417 | 0.2995 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 44 | lr_9e-6 | 0.4075 | 0.5632 | 0.2948 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 45 | lr_9e-6 | 0.4114 | 0.5617 | 0.3014 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 46 | lr_14e-6 | 0.4127 | 0.5718 | 0.2978 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 47 | lr_14e-6 | 0.4017 | 0.5491 | 0.2939 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 48 | lr_3e-6 | 0.403 | 0.5532 | 0.2936 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 49 | lr_3e-6 | 0.4044 | 0.5521 | 0.2962 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 50 | lr_9e-6 | 0.4163 | 0.5638 | 0.3073 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 51 | lr_6e-6 | 0.4065 | 0.5647 | 0.2926 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 52 | lr_9e-6 | 0.4077 | 0.544 | 0.3055 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 53 | lr_9e-6 | 0.4022 | 0.5483 | 0.295 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 54 | lr_3e-6 | 0.4128 | 0.5614 | 0.3035 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 55 | lr_14e-6 | 0.3975 | 0.5567 | 0.2839 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 56 | lr_6e-6 | 0.4095 | 0.5545 | 0.3024 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 57 | lr_9e-6 | 0.4089 | 0.5517 | 0.3031 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 58 | lr_3e-6 | 0.4032 | 0.5442 | 0.2987 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 59 | lr_14e-6 | 0.4101 | 0.5692 | 0.2955 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 60 | lr_14e-6 | 0.407 | 0.5556 | 0.2981 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 61 | lr_3e-6 | 0.3994 | 0.5483 | 0.2909 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 62 | lr_3e-6 | 0.4016 | 0.5442 | 0.2963 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 63 | lr_3e-6 | 0.4004 | 0.5449 | 0.2942 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 64 | lr_14e-6 | 0.4008 | 0.5557 | 0.2891 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 65 | lr_14e-6 | 0.3997 | 0.5474 | 0.2918 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 66 | lr_6e-6 | 0.4059 | 0.5588 | 0.2948 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 67 | lr_6e-6 | 0.4058 | 0.5466 | 0.3012 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 68 | lr_9e-6 | 0.4084 | 0.5454 | 0.3058 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 69 | lr_14e-6 | 0.3946 | 0.5158 | 0.3018 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 70 | lr_14e-6 | 0.4083 | 0.5578 | 0.2988 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 71 | lr_3e-6 | 0.4085 | 0.5476 | 0.3047 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 72 | lr_9e-6 | 0.407 | 0.5584 | 0.2967 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 73 | lr_3e-6 | 0.4125 | 0.5717 | 0.2976 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 74 | lr_6e-6 | 0.409 | 0.5599 | 0.2988 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 75 | lr_9e-6 | 0.4051 | 0.5436 | 0.3018 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 76 | lr_14e-6 | 0.3991 | 0.5411 | 0.2944 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 77 | lr_9e-6 | 0.4119 | 0.5612 | 0.3023 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 78 | lr_6e-6 | 0.4044 | 0.5445 | 0.3004 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 79 | lr_9e-6 | 0.4115 | 0.574 | 0.2951 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 80 | lr_14e-6 | 0.4091 | 0.5526 | 0.3029 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 81 | lr_9e-6 | 0.4087 | 0.5772 | 0.2894 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 82 | lr_3e-6 | 0.3958 | 0.5426 | 0.2887 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 83 | lr_14e-6 | 0.4044 | 0.5288 | 0.3092 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 84 | lr_14e-6 | 0.3973 | 0.5402 | 0.2923 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 85 | lr_9e-6 | 0.4127 | 0.5742 | 0.2967 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 86 | lr_14e-6 | 0.4086 | 0.5641 | 0.296 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 87 | lr_14e-6 | 0.4077 | 0.5495 | 0.3025 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 88 | lr_14e-6 | 0.406 | 0.5589 | 0.2949 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 89 | lr_14e-6 | 0.4084 | 0.5504 | 0.303 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 90 | lr_6e-6 | 0.4042 | 0.5508 | 0.2967 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 91 | lr_6e-6 | 0.4087 | 0.5523 | 0.3024 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 92 | lr_6e-6 | 0.4004 | 0.5363 | 0.2989 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 93 | lr_3e-6 | 0.4056 | 0.5404 | 0.3044 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 94 | lr_14e-6 | 0.4024 | 0.5487 | 0.2951 | 1.089e-05 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 95 | lr_6e-6 | 0.4023 | 0.5415 | 0.2989 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 96 | lr_6e-6 | 0.4061 | 0.5485 | 0.3007 | 8.165e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 97 | lr_3e-6 | 0.4047 | 0.5479 | 0.2989 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 98 | lr_9e-6 | 0.405 | 0.5514 | 0.2974 | 9.072e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |
| 99 | lr_3e-6 | 0.4122 | 0.5582 | 0.3044 | 7.258e-06 | 9.072e-06 | 9.072e-06 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_14e-6` -> `lr_3e-6`, donor metric 0.443164, recipient metric 0.451513, LR 3e-06 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_6e-6`, donor metric 0.443164, recipient metric 0.446394, LR 6e-06 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_9e-6`, donor metric 0.443164, recipient metric 0.443542, LR 9e-06 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_14e-6`, donor metric 0.443164, recipient metric 0.443164, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_3e-6`, donor metric 0.436009, recipient metric 0.43896, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_6e-6`, donor metric 0.436009, recipient metric 0.436009, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_9e-6`, donor metric 0.436009, recipient metric 0.446345, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_14e-6`, donor metric 0.436009, recipient metric 0.438955, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_3e-6`, donor metric 0.423064, recipient metric 0.428575, LR 1.01e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_6e-6`, donor metric 0.423064, recipient metric 0.437235, LR 1.13e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_9e-6`, donor metric 0.423064, recipient metric 0.428652, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_14e-6`, donor metric 0.423064, recipient metric 0.423064, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_14e-6` -> `lr_3e-6`, donor metric 0.428758, recipient metric 0.428167, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_14e-6` -> `lr_6e-6`, donor metric 0.428758, recipient metric 0.427554, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_14e-6` -> `lr_9e-6`, donor metric 0.428758, recipient metric 0.434465, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_14e-6` -> `lr_14e-6`, donor metric 0.428758, recipient metric 0.428758, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_3e-6`, donor metric 0.427828, recipient metric 0.431634, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_6e-6`, donor metric 0.427828, recipient metric 0.429074, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_9e-6`, donor metric 0.427828, recipient metric 0.435417, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_14e-6`, donor metric 0.427828, recipient metric 0.427828, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_3e-6`, donor metric 0.425603, recipient metric 0.439815, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_6e-6`, donor metric 0.425603, recipient metric 0.443607, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_9e-6`, donor metric 0.425603, recipient metric 0.443158, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_14e-6`, donor metric 0.425603, recipient metric 0.425603, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_3e-6`, donor metric 0.431955, recipient metric 0.443551, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_6e-6`, donor metric 0.431955, recipient metric 0.435188, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_9e-6`, donor metric 0.431955, recipient metric 0.429209, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_14e-6`, donor metric 0.431955, recipient metric 0.431955, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_3e-6`, donor metric 0.427184, recipient metric 0.436221, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_6e-6`, donor metric 0.427184, recipient metric 0.432672, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_9e-6`, donor metric 0.427184, recipient metric 0.431693, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_14e-6`, donor metric 0.427184, recipient metric 0.427184, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_3e-6`, donor metric 0.433343, recipient metric 0.432728, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_6e-6`, donor metric 0.433343, recipient metric 0.427172, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_9e-6`, donor metric 0.433343, recipient metric 0.427131, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_14e-6`, donor metric 0.433343, recipient metric 0.433343, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_3e-6`, donor metric 0.424615, recipient metric 0.433205, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_6e-6`, donor metric 0.424615, recipient metric 0.433332, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_9e-6`, donor metric 0.424615, recipient metric 0.4338, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_14e-6`, donor metric 0.424615, recipient metric 0.424615, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_3e-6`, donor metric 0.437766, recipient metric 0.43424, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_6e-6`, donor metric 0.437766, recipient metric 0.430532, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_9e-6`, donor metric 0.437766, recipient metric 0.428807, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_14e-6`, donor metric 0.437766, recipient metric 0.437766, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_3e-6`, donor metric 0.427028, recipient metric 0.432658, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_6e-6`, donor metric 0.427028, recipient metric 0.432753, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_9e-6`, donor metric 0.427028, recipient metric 0.424232, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_14e-6`, donor metric 0.427028, recipient metric 0.427028, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_3e-6`, donor metric 0.432238, recipient metric 0.43596, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_6e-6`, donor metric 0.432238, recipient metric 0.436579, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_9e-6`, donor metric 0.432238, recipient metric 0.43536, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_14e-6`, donor metric 0.432238, recipient metric 0.432238, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_3e-6`, donor metric 0.435152, recipient metric 0.437377, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_6e-6`, donor metric 0.435152, recipient metric 0.433463, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_9e-6`, donor metric 0.435152, recipient metric 0.436988, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_14e-6`, donor metric 0.435152, recipient metric 0.435152, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417026, recipient metric 0.444085, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_6e-6`, donor metric 0.417026, recipient metric 0.437012, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_9e-6`, donor metric 0.417026, recipient metric 0.427175, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417026, recipient metric 0.417026, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_3e-6`, donor metric 0.414607, recipient metric 0.427986, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_6e-6`, donor metric 0.414607, recipient metric 0.41796, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_9e-6`, donor metric 0.414607, recipient metric 0.421481, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_14e-6`, donor metric 0.414607, recipient metric 0.414607, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409815, recipient metric 0.418014, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409815, recipient metric 0.417034, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409815, recipient metric 0.425997, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409815, recipient metric 0.409815, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_3e-6`, donor metric 0.424642, recipient metric 0.421523, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_6e-6`, donor metric 0.424642, recipient metric 0.421966, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_9e-6`, donor metric 0.424642, recipient metric 0.42719, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_14e-6`, donor metric 0.424642, recipient metric 0.424642, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415588, recipient metric 0.417603, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415588, recipient metric 0.416609, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415588, recipient metric 0.426252, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415588, recipient metric 0.415588, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_3e-6`, donor metric 0.421829, recipient metric 0.426248, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_6e-6`, donor metric 0.421829, recipient metric 0.422618, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_9e-6`, donor metric 0.421829, recipient metric 0.413096, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_14e-6`, donor metric 0.421829, recipient metric 0.421829, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_3e-6`, donor metric 0.409411, recipient metric 0.417178, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_6e-6`, donor metric 0.409411, recipient metric 0.409411, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_9e-6`, donor metric 0.409411, recipient metric 0.424989, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_14e-6`, donor metric 0.409411, recipient metric 0.415036, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41877, recipient metric 0.419239, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41877, recipient metric 0.41877, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41877, recipient metric 0.424824, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41877, recipient metric 0.425037, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416831, recipient metric 0.415452, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416831, recipient metric 0.416831, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416831, recipient metric 0.423746, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416831, recipient metric 0.411869, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_3e-6`, donor metric 0.428887, recipient metric 0.424726, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_6e-6`, donor metric 0.428887, recipient metric 0.428887, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_9e-6`, donor metric 0.428887, recipient metric 0.429384, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_14e-6`, donor metric 0.428887, recipient metric 0.424402, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_3e-6`, donor metric 0.407915, recipient metric 0.408924, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_6e-6`, donor metric 0.407915, recipient metric 0.425157, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_9e-6`, donor metric 0.407915, recipient metric 0.407915, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_14e-6`, donor metric 0.407915, recipient metric 0.427909, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417289, recipient metric 0.411713, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417289, recipient metric 0.417442, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417289, recipient metric 0.417289, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417289, recipient metric 0.417295, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_9e-6` -> `lr_3e-6`, donor metric 0.410968, recipient metric 0.411537, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_9e-6` -> `lr_6e-6`, donor metric 0.410968, recipient metric 0.415837, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_9e-6` -> `lr_9e-6`, donor metric 0.410968, recipient metric 0.410968, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_9e-6` -> `lr_14e-6`, donor metric 0.410968, recipient metric 0.415041, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_9e-6` -> `lr_3e-6`, donor metric 0.410654, recipient metric 0.41901, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_9e-6` -> `lr_6e-6`, donor metric 0.410654, recipient metric 0.418451, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_9e-6` -> `lr_9e-6`, donor metric 0.410654, recipient metric 0.410654, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_9e-6` -> `lr_14e-6`, donor metric 0.410654, recipient metric 0.411422, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_9e-6` -> `lr_3e-6`, donor metric 0.418627, recipient metric 0.418574, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_9e-6` -> `lr_6e-6`, donor metric 0.418627, recipient metric 0.417166, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_9e-6` -> `lr_9e-6`, donor metric 0.418627, recipient metric 0.418627, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_9e-6` -> `lr_14e-6`, donor metric 0.418627, recipient metric 0.417696, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_3e-6`, donor metric 0.402721, recipient metric 0.402721, LR 1.01e-05 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_6e-6`, donor metric 0.402721, recipient metric 0.410666, LR 1.13e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_9e-6`, donor metric 0.402721, recipient metric 0.413772, LR 1.26e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_14e-6`, donor metric 0.402721, recipient metric 0.40453, LR 1.4e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_3e-6`, donor metric 0.407955, recipient metric 0.407955, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_6e-6`, donor metric 0.407955, recipient metric 0.405582, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_9e-6`, donor metric 0.407955, recipient metric 0.40741, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_14e-6`, donor metric 0.407955, recipient metric 0.426001, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_3e-6`, donor metric 0.414654, recipient metric 0.414654, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_6e-6`, donor metric 0.414654, recipient metric 0.410265, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_9e-6`, donor metric 0.414654, recipient metric 0.410194, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_14e-6`, donor metric 0.414654, recipient metric 0.41187, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_3e-6`, donor metric 0.412608, recipient metric 0.412608, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_6e-6`, donor metric 0.412608, recipient metric 0.403027, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_9e-6`, donor metric 0.412608, recipient metric 0.413105, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_14e-6`, donor metric 0.412608, recipient metric 0.412073, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_3e-6`, donor metric 0.408925, recipient metric 0.408925, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_6e-6`, donor metric 0.408925, recipient metric 0.409121, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_9e-6`, donor metric 0.408925, recipient metric 0.412057, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_14e-6`, donor metric 0.408925, recipient metric 0.409877, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_3e-6`, donor metric 0.415465, recipient metric 0.415465, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_6e-6`, donor metric 0.415465, recipient metric 0.410714, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_9e-6`, donor metric 0.415465, recipient metric 0.415617, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_14e-6`, donor metric 0.415465, recipient metric 0.402876, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_3e-6`, donor metric 0.403613, recipient metric 0.403613, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_6e-6`, donor metric 0.403613, recipient metric 0.40786, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_9e-6`, donor metric 0.403613, recipient metric 0.409064, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_14e-6`, donor metric 0.403613, recipient metric 0.409032, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_3e-6`, donor metric 0.419923, recipient metric 0.419923, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_6e-6`, donor metric 0.419923, recipient metric 0.409325, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_9e-6`, donor metric 0.419923, recipient metric 0.407587, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_14e-6`, donor metric 0.419923, recipient metric 0.408195, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_3e-6`, donor metric 0.391532, recipient metric 0.404871, LR 8.06e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_6e-6`, donor metric 0.391532, recipient metric 0.391532, LR 9.07e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_9e-6`, donor metric 0.391532, recipient metric 0.396055, LR 1.01e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_14e-6`, donor metric 0.391532, recipient metric 0.396462, LR 1.21e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_3e-6`, donor metric 0.411475, recipient metric 0.406825, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_6e-6`, donor metric 0.411475, recipient metric 0.411475, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_9e-6`, donor metric 0.411475, recipient metric 0.40856, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_14e-6`, donor metric 0.411475, recipient metric 0.41821, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_3e-6`, donor metric 0.411105, recipient metric 0.40583, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_6e-6`, donor metric 0.411105, recipient metric 0.411105, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_9e-6`, donor metric 0.411105, recipient metric 0.406105, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_14e-6`, donor metric 0.411105, recipient metric 0.420065, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_3e-6`, donor metric 0.427395, recipient metric 0.410268, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_6e-6`, donor metric 0.427395, recipient metric 0.427395, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_9e-6`, donor metric 0.427395, recipient metric 0.427567, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_14e-6`, donor metric 0.427395, recipient metric 0.415705, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_3e-6`, donor metric 0.403856, recipient metric 0.414301, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_6e-6`, donor metric 0.403856, recipient metric 0.403856, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_9e-6`, donor metric 0.403856, recipient metric 0.415315, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_14e-6`, donor metric 0.403856, recipient metric 0.412679, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416632, recipient metric 0.40854, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416632, recipient metric 0.416632, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416632, recipient metric 0.402777, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416632, recipient metric 0.401835, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_3e-6`, donor metric 0.408775, recipient metric 0.414222, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_6e-6`, donor metric 0.408775, recipient metric 0.408775, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_9e-6`, donor metric 0.408775, recipient metric 0.413956, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_14e-6`, donor metric 0.408775, recipient metric 0.402753, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_3e-6`, donor metric 0.412759, recipient metric 0.412995, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_6e-6`, donor metric 0.412759, recipient metric 0.412759, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_9e-6`, donor metric 0.412759, recipient metric 0.407451, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_14e-6`, donor metric 0.412759, recipient metric 0.416976, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_3e-6`, donor metric 0.412624, recipient metric 0.414078, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_6e-6`, donor metric 0.412624, recipient metric 0.412624, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_9e-6`, donor metric 0.412624, recipient metric 0.411439, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_14e-6`, donor metric 0.412624, recipient metric 0.41305, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_3e-6`, donor metric 0.414035, recipient metric 0.414112, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_6e-6`, donor metric 0.414035, recipient metric 0.414035, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_9e-6`, donor metric 0.414035, recipient metric 0.41495, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_14e-6`, donor metric 0.414035, recipient metric 0.412657, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_3e-6`, donor metric 0.415758, recipient metric 0.405161, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_6e-6`, donor metric 0.415758, recipient metric 0.415758, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_9e-6`, donor metric 0.415758, recipient metric 0.419618, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_14e-6`, donor metric 0.415758, recipient metric 0.401673, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416826, recipient metric 0.403001, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416826, recipient metric 0.416826, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416826, recipient metric 0.420454, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416826, recipient metric 0.41151, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_3e-6`, donor metric 0.424504, recipient metric 0.404364, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_6e-6`, donor metric 0.424504, recipient metric 0.424504, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_9e-6`, donor metric 0.424504, recipient metric 0.416621, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_14e-6`, donor metric 0.424504, recipient metric 0.406541, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418133, recipient metric 0.421764, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418133, recipient metric 0.418133, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418133, recipient metric 0.416292, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418133, recipient metric 0.418388, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_3e-6`, donor metric 0.406516, recipient metric 0.406973, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_6e-6`, donor metric 0.406516, recipient metric 0.406516, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_9e-6`, donor metric 0.406516, recipient metric 0.428894, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_14e-6`, donor metric 0.406516, recipient metric 0.409679, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41064, recipient metric 0.409975, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41064, recipient metric 0.41064, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41064, recipient metric 0.407668, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41064, recipient metric 0.409959, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_3e-6`, donor metric 0.414882, recipient metric 0.411426, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_6e-6`, donor metric 0.414882, recipient metric 0.414882, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_9e-6`, donor metric 0.414882, recipient metric 0.402207, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_14e-6`, donor metric 0.414882, recipient metric 0.411114, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_3e-6`, donor metric 0.414981, recipient metric 0.412781, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_6e-6`, donor metric 0.414981, recipient metric 0.414981, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_9e-6`, donor metric 0.414981, recipient metric 0.415984, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_14e-6`, donor metric 0.414981, recipient metric 0.415782, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41294, recipient metric 0.401141, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41294, recipient metric 0.41294, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41294, recipient metric 0.421674, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41294, recipient metric 0.397534, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_3e-6`, donor metric 0.409505, recipient metric 0.430518, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_6e-6`, donor metric 0.409505, recipient metric 0.409505, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_9e-6`, donor metric 0.409505, recipient metric 0.427392, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_14e-6`, donor metric 0.409505, recipient metric 0.429537, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_3e-6`, donor metric 0.410036, recipient metric 0.41292, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_6e-6`, donor metric 0.410036, recipient metric 0.410036, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_9e-6`, donor metric 0.410036, recipient metric 0.408941, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_14e-6`, donor metric 0.410036, recipient metric 0.410588, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_3e-6`, donor metric 0.422028, recipient metric 0.403185, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_6e-6`, donor metric 0.422028, recipient metric 0.422028, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_9e-6`, donor metric 0.422028, recipient metric 0.409645, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_14e-6`, donor metric 0.422028, recipient metric 0.418394, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_3e-6`, donor metric 0.415135, recipient metric 0.412824, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_6e-6`, donor metric 0.415135, recipient metric 0.415135, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_9e-6`, donor metric 0.415135, recipient metric 0.423637, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_14e-6`, donor metric 0.415135, recipient metric 0.410109, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_3e-6`, donor metric 0.415347, recipient metric 0.414619, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_6e-6`, donor metric 0.415347, recipient metric 0.415347, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_9e-6`, donor metric 0.415347, recipient metric 0.425263, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_14e-6`, donor metric 0.415347, recipient metric 0.406956, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_3e-6`, donor metric 0.421997, recipient metric 0.399372, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_6e-6`, donor metric 0.421997, recipient metric 0.421997, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_9e-6`, donor metric 0.421997, recipient metric 0.416128, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_14e-6`, donor metric 0.421997, recipient metric 0.416581, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_3e-6`, donor metric 0.407822, recipient metric 0.401555, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_6e-6`, donor metric 0.407822, recipient metric 0.407822, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_9e-6`, donor metric 0.407822, recipient metric 0.410713, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_14e-6`, donor metric 0.407822, recipient metric 0.413889, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_3e-6`, donor metric 0.407766, recipient metric 0.400405, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_6e-6`, donor metric 0.407766, recipient metric 0.407766, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_9e-6`, donor metric 0.407766, recipient metric 0.412727, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_14e-6`, donor metric 0.407766, recipient metric 0.405294, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416776, recipient metric 0.421675, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416776, recipient metric 0.416776, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416776, recipient metric 0.42119, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416776, recipient metric 0.40077, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_3e-6`, donor metric 0.4078, recipient metric 0.411136, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_6e-6`, donor metric 0.4078, recipient metric 0.4078, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_9e-6`, donor metric 0.4078, recipient metric 0.411755, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_14e-6`, donor metric 0.4078, recipient metric 0.399667, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_3e-6`, donor metric 0.405896, recipient metric 0.411924, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_6e-6`, donor metric 0.405896, recipient metric 0.405896, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_9e-6`, donor metric 0.405896, recipient metric 0.406638, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_14e-6`, donor metric 0.405896, recipient metric 0.40801, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_3e-6`, donor metric 0.405754, recipient metric 0.409774, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_6e-6`, donor metric 0.405754, recipient metric 0.405754, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_9e-6`, donor metric 0.405754, recipient metric 0.408428, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_14e-6`, donor metric 0.405754, recipient metric 0.424334, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_3e-6`, donor metric 0.410555, recipient metric 0.409007, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_6e-6`, donor metric 0.410555, recipient metric 0.410555, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_9e-6`, donor metric 0.410555, recipient metric 0.408368, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_14e-6`, donor metric 0.410555, recipient metric 0.417676, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_3e-6`, donor metric 0.411006, recipient metric 0.410528, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_6e-6`, donor metric 0.411006, recipient metric 0.411006, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_9e-6`, donor metric 0.411006, recipient metric 0.394619, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_14e-6`, donor metric 0.411006, recipient metric 0.394551, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_3e-6`, donor metric 0.428202, recipient metric 0.415587, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_6e-6`, donor metric 0.428202, recipient metric 0.428202, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_9e-6`, donor metric 0.428202, recipient metric 0.418476, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_14e-6`, donor metric 0.428202, recipient metric 0.408271, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_3e-6`, donor metric 0.408641, recipient metric 0.408501, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_6e-6`, donor metric 0.408641, recipient metric 0.408641, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_9e-6`, donor metric 0.408641, recipient metric 0.416399, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_14e-6`, donor metric 0.408641, recipient metric 0.411733, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_3e-6`, donor metric 0.420416, recipient metric 0.419177, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_6e-6`, donor metric 0.420416, recipient metric 0.420416, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_9e-6`, donor metric 0.420416, recipient metric 0.406999, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_14e-6`, donor metric 0.420416, recipient metric 0.421687, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418586, recipient metric 0.412466, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418586, recipient metric 0.418586, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418586, recipient metric 0.412706, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418586, recipient metric 0.417931, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_3e-6`, donor metric 0.40903, recipient metric 0.413786, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_6e-6`, donor metric 0.40903, recipient metric 0.40903, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_9e-6`, donor metric 0.40903, recipient metric 0.411347, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_14e-6`, donor metric 0.40903, recipient metric 0.414423, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_3e-6`, donor metric 0.413556, recipient metric 0.405223, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_6e-6`, donor metric 0.413556, recipient metric 0.413556, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_9e-6`, donor metric 0.413556, recipient metric 0.405064, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_14e-6`, donor metric 0.413556, recipient metric 0.405172, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_3e-6`, donor metric 0.40513, recipient metric 0.408071, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_6e-6`, donor metric 0.40513, recipient metric 0.40513, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_9e-6`, donor metric 0.40513, recipient metric 0.412621, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_14e-6`, donor metric 0.40513, recipient metric 0.399107, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_3e-6`, donor metric 0.412564, recipient metric 0.412224, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_6e-6`, donor metric 0.412564, recipient metric 0.412564, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_9e-6`, donor metric 0.412564, recipient metric 0.411855, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_14e-6`, donor metric 0.412564, recipient metric 0.414083, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_3e-6`, donor metric 0.404403, recipient metric 0.42496, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_6e-6`, donor metric 0.404403, recipient metric 0.404403, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_9e-6`, donor metric 0.404403, recipient metric 0.414675, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_14e-6`, donor metric 0.404403, recipient metric 0.405855, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416977, recipient metric 0.423999, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416977, recipient metric 0.416977, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416977, recipient metric 0.411547, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416977, recipient metric 0.419005, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_3e-6`, donor metric 0.422443, recipient metric 0.411979, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_6e-6`, donor metric 0.422443, recipient metric 0.422443, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_9e-6`, donor metric 0.422443, recipient metric 0.423839, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_14e-6`, donor metric 0.422443, recipient metric 0.409145, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41237, recipient metric 0.417524, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41237, recipient metric 0.41237, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41237, recipient metric 0.408695, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41237, recipient metric 0.412047, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_3e-6`, donor metric 0.398078, recipient metric 0.395795, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_6e-6`, donor metric 0.398078, recipient metric 0.398078, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_9e-6`, donor metric 0.398078, recipient metric 0.409544, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_14e-6`, donor metric 0.398078, recipient metric 0.408317, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_3e-6`, donor metric 0.409347, recipient metric 0.410324, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_6e-6`, donor metric 0.409347, recipient metric 0.409347, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_9e-6`, donor metric 0.409347, recipient metric 0.406253, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_14e-6`, donor metric 0.409347, recipient metric 0.404357, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_3e-6`, donor metric 0.409476, recipient metric 0.416303, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_6e-6`, donor metric 0.409476, recipient metric 0.409476, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_9e-6`, donor metric 0.409476, recipient metric 0.415514, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_14e-6`, donor metric 0.409476, recipient metric 0.397343, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_3e-6`, donor metric 0.422583, recipient metric 0.417767, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_6e-6`, donor metric 0.422583, recipient metric 0.422583, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_9e-6`, donor metric 0.422583, recipient metric 0.412747, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_14e-6`, donor metric 0.422583, recipient metric 0.414902, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_3e-6`, donor metric 0.420367, recipient metric 0.412526, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_6e-6`, donor metric 0.420367, recipient metric 0.420367, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_9e-6`, donor metric 0.420367, recipient metric 0.41406, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_14e-6`, donor metric 0.420367, recipient metric 0.40864, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418041, recipient metric 0.413554, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418041, recipient metric 0.418041, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418041, recipient metric 0.409908, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418041, recipient metric 0.407712, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_3e-6`, donor metric 0.408706, recipient metric 0.419267, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_6e-6`, donor metric 0.408706, recipient metric 0.408706, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_9e-6`, donor metric 0.408706, recipient metric 0.419437, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_14e-6`, donor metric 0.408706, recipient metric 0.405983, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_3e-6`, donor metric 0.417809, recipient metric 0.408574, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_6e-6`, donor metric 0.417809, recipient metric 0.417809, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_9e-6`, donor metric 0.417809, recipient metric 0.420991, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_14e-6`, donor metric 0.417809, recipient metric 0.408365, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_3e-6`, donor metric 0.404233, recipient metric 0.415625, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_6e-6`, donor metric 0.404233, recipient metric 0.404233, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_9e-6`, donor metric 0.404233, recipient metric 0.412768, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_14e-6`, donor metric 0.404233, recipient metric 0.414312, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_3e-6`, donor metric 0.408693, recipient metric 0.413548, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_6e-6`, donor metric 0.408693, recipient metric 0.408693, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_9e-6`, donor metric 0.408693, recipient metric 0.413889, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_14e-6`, donor metric 0.408693, recipient metric 0.409188, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_3e-6`, donor metric 0.400404, recipient metric 0.421688, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_6e-6`, donor metric 0.400404, recipient metric 0.400404, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_9e-6`, donor metric 0.400404, recipient metric 0.418398, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_14e-6`, donor metric 0.400404, recipient metric 0.421129, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_3e-6`, donor metric 0.414514, recipient metric 0.405559, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_6e-6`, donor metric 0.414514, recipient metric 0.414514, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_9e-6`, donor metric 0.414514, recipient metric 0.411522, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_14e-6`, donor metric 0.414514, recipient metric 0.406215, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_3e-6`, donor metric 0.410462, recipient metric 0.415041, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_6e-6`, donor metric 0.410462, recipient metric 0.410462, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_9e-6`, donor metric 0.410462, recipient metric 0.413715, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_14e-6`, donor metric 0.410462, recipient metric 0.402407, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_3e-6`, donor metric 0.40228, recipient metric 0.407975, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_6e-6`, donor metric 0.40228, recipient metric 0.40228, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_9e-6`, donor metric 0.40228, recipient metric 0.406312, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_14e-6`, donor metric 0.40228, recipient metric 0.408042, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_3e-6`, donor metric 0.406094, recipient metric 0.414794, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_6e-6`, donor metric 0.406094, recipient metric 0.406094, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_9e-6`, donor metric 0.406094, recipient metric 0.415951, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_14e-6`, donor metric 0.406094, recipient metric 0.415079, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41939, recipient metric 0.404667, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41939, recipient metric 0.41939, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41939, recipient metric 0.414425, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41939, recipient metric 0.416587, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_3e-6`, donor metric 0.411042, recipient metric 0.411436, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_6e-6`, donor metric 0.411042, recipient metric 0.411042, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_9e-6`, donor metric 0.411042, recipient metric 0.404963, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_14e-6`, donor metric 0.411042, recipient metric 0.418368, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418029, recipient metric 0.41223, LR 7.26e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418029, recipient metric 0.418029, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418029, recipient metric 0.418222, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418029, recipient metric 0.419197, LR 1.09e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
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
- Git commit: `b3c69dd88163c6f447475860d145ce77d9308cc7`
- Git dirty: `False`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_100gen_seed2.yaml', '--slots', 'iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3', '--experiment-name', 'anchor_copy_lr_recenter_100gen_seed2_20260819_111523']`
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
