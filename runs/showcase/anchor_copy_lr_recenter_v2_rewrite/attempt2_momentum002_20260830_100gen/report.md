# anchor_copy_lr_recenter_v2_validation

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
- Final checkpoint controller objective: 0.993759 by `lr_8_5e-6`
- Global best configured metric: 0.391014 by `lr_11_25e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_v2_validation/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_11_25e-6`, generation 82), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_5_75e-6`, generation 96) -- these are two distinct selection criteria, not the same checkpoint.
  - Validation: `/data/suehara/part/march/datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet` (`val50k_tail`), 150000 samples
- [Physics performance](plots/report/physics_performance.png)
- [C-tag mistag CSV](plots/report/ctag_mistag_tables.csv)
- [B-tag mistag CSV](plots/report/btag_mistag_tables.csv)

## PBT Population and Selection
- [Population and selection](plots/pbt_population_selection.png)
- Ranking metric: `validation_total_reference_mistag_geomean_percent` (min); winner = each generation's authoritative decision winner (the member that actually drove that generation's exploit/anchor/global-best outcome), never re-derived from total_mistag_score.
- Winner-timeline decision markers: `^` accepted_new_anchor, `o` reused_previous_anchor, `v` rewound_to_previous_anchor, `P` plateau_escape_accepted.

## Mistag Score Evolution
- [Mistag score evolution](plots/mistag_score_evolution.png)
- **`total_mistag_score` (sqrt(ctag_score * btag_score)) is this run's PBT ranking metric** -- the thick line above is the ranking metric itself, not just a diagnostic summary.
- Baseline point: not available for this run.

## Learning-Rate Lineage
- [Learning-rate lineage](plots/learning_rate_lineage.png)
- Heavy edge = an applied donor->recipient checkpoint copy (events.jsonl, applied=True only); light edge = a member continuing its own branch.

## Learning Rate vs. Mistag Score Correlation
- [Training dynamics and within-generation LR analysis](plots/learning_rate_mistag_correlation.png)
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=500, Pearson r=-0.060 (95% CI -0.148 to 0.023), Spearman rho=-0.054 (95% CI -0.133 to 0.020)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=25, Pearson r=0.830, Spearman rho=0.721
- Best checkpoint by tier: control: `lr_8_5e-6` gen 99 (0.396378), full_holdout: `lr_11_25e-6` gen 99 (0.385416)
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
| lr_11_25e-6 | 0.1786 | 0.07417 | 3.076 | 0.1924 | 0.5264 | 0.06815 | 2.698 | 1.163 | 0.5792 | 0.2975 | 0.4151 | 4.13e-06 | - |
| lr_14e-6 | 0.1768 | 0.0681 | 3.122 | 0.1843 | 0.51 | 0.06009 | 2.671 | 1.176 | 0.5569 | 0.2885 | 0.4008 | 5.25e-06 | - |
| lr_3e-6 | 0.1785 | 0.0782 | 3.26 | 0.2025 | 0.5062 | 0.06216 | 2.66 | 1.143 | 0.5561 | 0.3098 | 0.4151 | 3e-06 | - |
| lr_5_75e-6 | 0.1768 | 0.0681 | 3.126 | 0.1843 | 0.51 | 0.06009 | 2.673 | 1.174 | 0.5568 | 0.2886 | 0.4009 | 3e-06 | - |
| lr_8_5e-6 | 0.1866 | 0.06607 | 3.111 | 0.1822 | 0.517 | 0.05406 | 2.66 | 1.173 | 0.5434 | 0.2891 | 0.3964 | 3e-06 | winner, anchor |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_5_75e-6 | 0.4446 | 0.6203 | 0.3187 | 5.75e-06 | 5.75e-06 | 5.75e-06 | accepted_new_anchor | no |
| 1 | lr_3e-6 | 0.4457 | 0.6113 | 0.325 | 3e-06 | 5.75e-06 | 5.75e-06 | rewound_to_previous_anchor | no |
| 2 | lr_5_75e-6 | 0.4527 | 0.6284 | 0.3262 | 4.312e-06 | 5.75e-06 | 5.75e-06 | rewound_to_previous_anchor | no |
| 3 | lr_11_25e-6 | 0.4464 | 0.621 | 0.3209 | 7.188e-06 | 5.75e-06 | 5.75e-06 | rewound_to_previous_anchor | no |
| 4 | lr_5_75e-6 | 0.4493 | 0.6278 | 0.3215 | 4.312e-06 | 5.75e-06 | 5.75e-06 | rewound_to_previous_anchor | no |
| 5 | lr_3e-6 | 0.4433 | 0.6284 | 0.3127 | 3e-06 | 5.75e-06 | 3e-06 | accepted_new_anchor | yes |
| 6 | lr_14e-6 | 0.4388 | 0.6123 | 0.3144 | 4.5e-06 | 3e-06 | 4.59e-06 | accepted_new_anchor | no |
| 7 | lr_14e-6 | 0.441 | 0.6208 | 0.3132 | 6.885e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 8 | lr_14e-6 | 0.4404 | 0.6173 | 0.3142 | 6.885e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 9 | lr_3e-6 | 0.4492 | 0.6173 | 0.3268 | 3e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 10 | lr_5_75e-6 | 0.45 | 0.6066 | 0.3338 | 3.443e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 11 | lr_3e-6 | 0.4484 | 0.6293 | 0.3196 | 3e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 12 | lr_11_25e-6 | 0.4396 | 0.6097 | 0.3169 | 5.738e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 13 | lr_5_75e-6 | 0.4427 | 0.618 | 0.317 | 3.443e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 14 | lr_5_75e-6 | 0.4422 | 0.6205 | 0.3152 | 3.443e-06 | 4.59e-06 | 3.443e-06 | plateau_escape_accepted | yes |
| 15 | lr_8_5e-6 | 0.437 | 0.6186 | 0.3087 | 3.443e-06 | 3.443e-06 | 3.443e-06 | accepted_new_anchor | yes |
| 16 | lr_14e-6 | 0.4424 | 0.5992 | 0.3266 | 5.164e-06 | 3.443e-06 | 3.443e-06 | rewound_to_previous_anchor | yes |
| 17 | lr_14e-6 | 0.4354 | 0.6045 | 0.3136 | 5.164e-06 | 3.443e-06 | 5.267e-06 | accepted_new_anchor | no |
| 18 | lr_11_25e-6 | 0.4385 | 0.5976 | 0.3218 | 6.584e-06 | 5.267e-06 | 5.267e-06 | rewound_to_previous_anchor | no |
| 19 | lr_14e-6 | 0.4271 | 0.5882 | 0.3102 | 7.901e-06 | 5.267e-06 | 8.059e-06 | accepted_new_anchor | no |
| 20 | lr_8_5e-6 | 0.429 | 0.5931 | 0.3103 | 8.059e-06 | 8.059e-06 | 8.059e-06 | rewound_to_previous_anchor | no |
| 21 | lr_14e-6 | 0.4318 | 0.5871 | 0.3175 | 1.209e-05 | 8.059e-06 | 8.059e-06 | rewound_to_previous_anchor | no |
| 22 | lr_11_25e-6 | 0.4291 | 0.5942 | 0.3099 | 1.007e-05 | 8.059e-06 | 8.059e-06 | rewound_to_previous_anchor | no |
| 23 | lr_5_75e-6 | 0.4312 | 0.5999 | 0.3099 | 6.044e-06 | 8.059e-06 | 8.059e-06 | rewound_to_previous_anchor | no |
| 24 | lr_14e-6 | 0.4193 | 0.575 | 0.3058 | 1.209e-05 | 8.059e-06 | 1.233e-05 | accepted_new_anchor | yes |
| 25 | lr_8_5e-6 | 0.4241 | 0.5849 | 0.3075 | 1.233e-05 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 26 | lr_8_5e-6 | 0.4096 | 0.5783 | 0.2901 | 1.233e-05 | 1.233e-05 | 1.233e-05 | accepted_new_anchor | yes |
| 27 | lr_3e-6 | 0.4163 | 0.5858 | 0.2958 | 6.165e-06 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 28 | lr_8_5e-6 | 0.4227 | 0.583 | 0.3065 | 1.233e-05 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 29 | lr_11_25e-6 | 0.4124 | 0.5781 | 0.2943 | 1.4e-05 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 30 | lr_14e-6 | 0.4136 | 0.5792 | 0.2953 | 1.4e-05 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 31 | lr_5_75e-6 | 0.4131 | 0.5546 | 0.3077 | 9.247e-06 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 32 | lr_8_5e-6 | 0.4159 | 0.5787 | 0.2989 | 1.233e-05 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 33 | lr_14e-6 | 0.4134 | 0.5839 | 0.2927 | 1.4e-05 | 1.233e-05 | 1.233e-05 | rewound_to_previous_anchor | yes |
| 34 | lr_11_25e-6 | 0.4226 | 0.5868 | 0.3044 | 1.4e-05 | 1.233e-05 | 1.4e-05 | plateau_escape_accepted | yes |
| 35 | lr_11_25e-6 | 0.409 | 0.5696 | 0.2936 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 36 | lr_3e-6 | 0.4012 | 0.5679 | 0.2834 | 7e-06 | 1.4e-05 | 6.86e-06 | accepted_new_anchor | no |
| 37 | lr_3e-6 | 0.4127 | 0.5576 | 0.3055 | 3.43e-06 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 38 | lr_3e-6 | 0.4132 | 0.5629 | 0.3033 | 3.43e-06 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 39 | lr_11_25e-6 | 0.4086 | 0.5736 | 0.291 | 8.575e-06 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 40 | lr_3e-6 | 0.4211 | 0.5828 | 0.3043 | 3.43e-06 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 41 | lr_14e-6 | 0.419 | 0.5857 | 0.2998 | 1.029e-05 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 42 | lr_3e-6 | 0.4158 | 0.5708 | 0.3029 | 3.43e-06 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 43 | lr_8_5e-6 | 0.4219 | 0.5731 | 0.3106 | 6.86e-06 | 6.86e-06 | 6.86e-06 | rewound_to_previous_anchor | no |
| 44 | lr_14e-6 | 0.4173 | 0.5687 | 0.3061 | 1.029e-05 | 6.86e-06 | 1.029e-05 | plateau_escape_accepted | yes |
| 45 | lr_8_5e-6 | 0.4035 | 0.5617 | 0.2899 | 1.029e-05 | 1.029e-05 | 1.029e-05 | accepted_new_anchor | no |
| 46 | lr_8_5e-6 | 0.4072 | 0.5618 | 0.2952 | 1.029e-05 | 1.029e-05 | 1.029e-05 | rewound_to_previous_anchor | no |
| 47 | lr_14e-6 | 0.4112 | 0.561 | 0.3014 | 1.4e-05 | 1.029e-05 | 1.029e-05 | rewound_to_previous_anchor | no |
| 48 | lr_3e-6 | 0.3957 | 0.5458 | 0.2869 | 5.145e-06 | 1.029e-05 | 5.042e-06 | accepted_new_anchor | no |
| 49 | lr_8_5e-6 | 0.4024 | 0.5604 | 0.2889 | 5.042e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 50 | lr_5_75e-6 | 0.4122 | 0.546 | 0.3111 | 3.782e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 51 | lr_8_5e-6 | 0.4019 | 0.569 | 0.2838 | 5.042e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 52 | lr_5_75e-6 | 0.4105 | 0.549 | 0.3069 | 3.782e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 53 | lr_14e-6 | 0.405 | 0.5598 | 0.293 | 7.563e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 54 | lr_5_75e-6 | 0.4131 | 0.5615 | 0.304 | 3.782e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 55 | lr_14e-6 | 0.4097 | 0.557 | 0.3014 | 7.563e-06 | 5.042e-06 | 5.042e-06 | rewound_to_previous_anchor | no |
| 56 | lr_14e-6 | 0.4087 | 0.5543 | 0.3013 | 7.563e-06 | 5.042e-06 | 7.563e-06 | plateau_escape_accepted | no |
| 57 | lr_3e-6 | 0.4125 | 0.556 | 0.306 | 3e-06 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 58 | lr_8_5e-6 | 0.3981 | 0.5479 | 0.2892 | 7.563e-06 | 7.563e-06 | 7.563e-06 | accepted_new_anchor | no |
| 59 | lr_14e-6 | 0.4052 | 0.5566 | 0.2951 | 1.134e-05 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 60 | lr_5_75e-6 | 0.4155 | 0.5604 | 0.3081 | 5.672e-06 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 61 | lr_14e-6 | 0.3987 | 0.5289 | 0.3006 | 1.134e-05 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 62 | lr_3e-6 | 0.407 | 0.5537 | 0.2992 | 3.782e-06 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 63 | lr_14e-6 | 0.4072 | 0.5578 | 0.2973 | 1.134e-05 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 64 | lr_3e-6 | 0.4062 | 0.5417 | 0.3045 | 3.782e-06 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 65 | lr_8_5e-6 | 0.4034 | 0.5443 | 0.299 | 7.563e-06 | 7.563e-06 | 7.563e-06 | rewound_to_previous_anchor | no |
| 66 | lr_11_25e-6 | 0.4173 | 0.5682 | 0.3065 | 9.454e-06 | 7.563e-06 | 9.454e-06 | plateau_escape_accepted | no |
| 67 | lr_3e-6 | 0.4023 | 0.5562 | 0.291 | 3e-06 | 9.454e-06 | 3e-06 | accepted_new_anchor | yes |
| 68 | lr_14e-6 | 0.4125 | 0.564 | 0.3016 | 4.5e-06 | 3e-06 | 3e-06 | rewound_to_previous_anchor | yes |
| 69 | lr_11_25e-6 | 0.4083 | 0.5522 | 0.302 | 3.75e-06 | 3e-06 | 3e-06 | rewound_to_previous_anchor | yes |
| 70 | lr_14e-6 | 0.4009 | 0.564 | 0.2849 | 4.5e-06 | 3e-06 | 4.59e-06 | accepted_new_anchor | no |
| 71 | lr_8_5e-6 | 0.3975 | 0.5476 | 0.2885 | 4.59e-06 | 4.59e-06 | 4.59e-06 | accepted_new_anchor | no |
| 72 | lr_14e-6 | 0.3975 | 0.536 | 0.2948 | 6.885e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 73 | lr_3e-6 | 0.4106 | 0.5542 | 0.3042 | 3e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 74 | lr_11_25e-6 | 0.4063 | 0.547 | 0.3018 | 5.738e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 75 | lr_14e-6 | 0.4055 | 0.5547 | 0.2963 | 6.885e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 76 | lr_14e-6 | 0.3992 | 0.5419 | 0.2941 | 6.885e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 77 | lr_3e-6 | 0.4038 | 0.5536 | 0.2945 | 3e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 78 | lr_11_25e-6 | 0.4097 | 0.5519 | 0.3041 | 5.738e-06 | 4.59e-06 | 4.59e-06 | rewound_to_previous_anchor | no |
| 79 | lr_5_75e-6 | 0.4064 | 0.5371 | 0.3074 | 3.443e-06 | 4.59e-06 | 3.443e-06 | plateau_escape_accepted | yes |
| 80 | lr_14e-6 | 0.4005 | 0.541 | 0.2965 | 6.024e-06 | 3.443e-06 | 6.145e-06 | accepted_new_anchor | no |
| 81 | lr_8_5e-6 | 0.3959 | 0.5425 | 0.2889 | 6.145e-06 | 6.145e-06 | 6.145e-06 | accepted_new_anchor | no |
| 82 | lr_11_25e-6 | 0.391 | 0.5286 | 0.2892 | 7.681e-06 | 6.145e-06 | 7.835e-06 | accepted_new_anchor | no |
| 83 | lr_14e-6 | 0.3987 | 0.5467 | 0.2907 | 1.175e-05 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 84 | lr_8_5e-6 | 0.4083 | 0.5603 | 0.2975 | 7.835e-06 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 85 | lr_14e-6 | 0.3962 | 0.5424 | 0.2894 | 1.175e-05 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 86 | lr_14e-6 | 0.4135 | 0.5641 | 0.3032 | 1.175e-05 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 87 | lr_14e-6 | 0.4026 | 0.5495 | 0.295 | 1.175e-05 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 88 | lr_5_75e-6 | 0.4109 | 0.5534 | 0.3051 | 5.876e-06 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 89 | lr_11_25e-6 | 0.4051 | 0.5515 | 0.2975 | 9.793e-06 | 7.835e-06 | 7.835e-06 | rewound_to_previous_anchor | no |
| 90 | lr_5_75e-6 | 0.3998 | 0.5446 | 0.2935 | 5.876e-06 | 7.835e-06 | 5.876e-06 | plateau_escape_accepted | no |
| 91 | lr_11_25e-6 | 0.4062 | 0.561 | 0.2941 | 8.08e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 92 | lr_11_25e-6 | 0.4079 | 0.5645 | 0.2947 | 7.345e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 93 | lr_3e-6 | 0.4002 | 0.56 | 0.286 | 3e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 94 | lr_14e-6 | 0.4037 | 0.5445 | 0.2994 | 8.814e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 95 | lr_3e-6 | 0.4017 | 0.5458 | 0.2956 | 3e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 96 | lr_5_75e-6 | 0.4017 | 0.56 | 0.2881 | 4.407e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 97 | lr_5_75e-6 | 0.4004 | 0.5349 | 0.2997 | 4.407e-06 | 5.876e-06 | 5.876e-06 | rewound_to_previous_anchor | no |
| 98 | lr_3e-6 | 0.4025 | 0.55 | 0.2946 | 3e-06 | 5.876e-06 | 3e-06 | plateau_escape_accepted | yes |
| 99 | lr_8_5e-6 | 0.3964 | 0.5434 | 0.2891 | 3e-06 | 3e-06 | 3e-06 | accepted_new_anchor | yes |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.44464, recipient metric 0.463505, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.44464, recipient metric 0.44464, LR 5.75e-06 -> 4.31e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.44464, recipient metric 0.44828, LR 8.5e-06 -> 5.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.44464, recipient metric 0.454377, LR 1.13e-05 -> 7.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.44464, recipient metric 0.460618, LR 1.4e-05 -> 8.62e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.451616, recipient metric 0.445719, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.451616, recipient metric 0.451616, LR 4.31e-06 -> 4.31e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.451616, recipient metric 0.451658, LR 5.75e-06 -> 5.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.451616, recipient metric 0.451629, LR 7.19e-06 -> 7.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.451616, recipient metric 0.450795, LR 8.62e-06 -> 8.62e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.452743, recipient metric 0.457988, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.452743, recipient metric 0.452743, LR 4.31e-06 -> 4.31e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.452743, recipient metric 0.453301, LR 5.75e-06 -> 5.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.452743, recipient metric 0.455968, LR 7.19e-06 -> 7.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.452743, recipient metric 0.453961, LR 8.62e-06 -> 8.62e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.463548, recipient metric 0.44975, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.463548, recipient metric 0.463548, LR 4.31e-06 -> 4.31e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.463548, recipient metric 0.455046, LR 5.75e-06 -> 5.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.463548, recipient metric 0.446443, LR 7.19e-06 -> 7.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.463548, recipient metric 0.447959, LR 8.62e-06 -> 8.62e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.449298, recipient metric 0.449411, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.449298, recipient metric 0.449298, LR 4.31e-06 -> 4.31e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.449298, recipient metric 0.450701, LR 5.75e-06 -> 5.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.449298, recipient metric 0.449598, LR 7.19e-06 -> 7.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.449298, recipient metric 0.452736, LR 8.62e-06 -> 8.62e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_3e-6`, donor metric 0.443272, recipient metric 0.443272, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.443272, recipient metric 0.445695, LR 4.31e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.443272, recipient metric 0.449587, LR 5.75e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.443272, recipient metric 0.449903, LR 7.19e-06 -> 3.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_14e-6`, donor metric 0.443272, recipient metric 0.45218, LR 8.62e-06 -> 4.5e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_3e-6`, donor metric 0.438804, recipient metric 0.450342, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.438804, recipient metric 0.450375, LR 3e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.438804, recipient metric 0.451855, LR 3e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.438804, recipient metric 0.439846, LR 3.75e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_14e-6`, donor metric 0.438804, recipient metric 0.438804, LR 4.5e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_3e-6`, donor metric 0.44096, recipient metric 0.457369, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.44096, recipient metric 0.454211, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.44096, recipient metric 0.452431, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.44096, recipient metric 0.451633, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_14e-6`, donor metric 0.44096, recipient metric 0.44096, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_3e-6`, donor metric 0.440387, recipient metric 0.46051, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.440387, recipient metric 0.440722, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.440387, recipient metric 0.445883, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.440387, recipient metric 0.455751, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_14e-6`, donor metric 0.440387, recipient metric 0.440387, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_3e-6`, donor metric 0.453595, recipient metric 0.449182, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.453595, recipient metric 0.450012, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.453595, recipient metric 0.452084, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.453595, recipient metric 0.453025, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_14e-6`, donor metric 0.453595, recipient metric 0.453595, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_3e-6`, donor metric 0.450738, recipient metric 0.454995, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.450738, recipient metric 0.450028, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.450738, recipient metric 0.454348, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.450738, recipient metric 0.452278, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_14e-6`, donor metric 0.450738, recipient metric 0.450738, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_3e-6`, donor metric 0.450332, recipient metric 0.448444, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.450332, recipient metric 0.453279, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.450332, recipient metric 0.453603, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.450332, recipient metric 0.451159, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_14e-6`, donor metric 0.450332, recipient metric 0.450332, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_3e-6`, donor metric 0.446671, recipient metric 0.43965, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.446671, recipient metric 0.450464, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.446671, recipient metric 0.450381, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.446671, recipient metric 0.439604, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_14e-6`, donor metric 0.446671, recipient metric 0.446671, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_3e-6`, donor metric 0.448517, recipient metric 0.455669, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.448517, recipient metric 0.442656, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.448517, recipient metric 0.450018, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.448517, recipient metric 0.453849, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_14e-6`, donor metric 0.448517, recipient metric 0.448517, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.442206, recipient metric 0.44376, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.442206, recipient metric 0.442206, LR 3.44e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.442206, recipient metric 0.44484, LR 4.59e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.442206, recipient metric 0.448357, LR 5.74e-06 -> 4.73e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.442206, recipient metric 0.444499, LR 6.89e-06 -> 6.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.436989, recipient metric 0.451863, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.436989, recipient metric 0.452925, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.436989, recipient metric 0.436989, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.436989, recipient metric 0.449042, LR 4.73e-06 -> 4.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.436989, recipient metric 0.447732, LR 6.02e-06 -> 5.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.443127, recipient metric 0.443454, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.443127, recipient metric 0.445204, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.443127, recipient metric 0.443127, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.443127, recipient metric 0.443935, LR 4.3e-06 -> 4.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.443127, recipient metric 0.442387, LR 5.16e-06 -> 5.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_3e-6`, donor metric 0.435387, recipient metric 0.440798, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.435387, recipient metric 0.444417, LR 3e-06 -> 3.95e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.435387, recipient metric 0.443688, LR 3.44e-06 -> 5.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.435387, recipient metric 0.449822, LR 4.3e-06 -> 6.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_14e-6`, donor metric 0.435387, recipient metric 0.435387, LR 5.16e-06 -> 7.9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_3e-6`, donor metric 0.449335, recipient metric 0.444374, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.449335, recipient metric 0.455674, LR 3.95e-06 -> 3.95e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.449335, recipient metric 0.438885, LR 5.27e-06 -> 5.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.449335, recipient metric 0.438507, LR 6.58e-06 -> 6.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_14e-6`, donor metric 0.449335, recipient metric 0.449335, LR 7.9e-06 -> 7.9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_3e-6`, donor metric 0.427143, recipient metric 0.429037, LR 3e-06 -> 4.03e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.427143, recipient metric 0.439614, LR 3.95e-06 -> 6.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.427143, recipient metric 0.437041, LR 5.27e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.427143, recipient metric 0.4373, LR 6.58e-06 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_14e-6`, donor metric 0.427143, recipient metric 0.427143, LR 7.9e-06 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_3e-6`, donor metric 0.446159, recipient metric 0.450102, LR 4.03e-06 -> 4.03e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.446159, recipient metric 0.432913, LR 6.04e-06 -> 6.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.446159, recipient metric 0.428971, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.446159, recipient metric 0.431928, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_14e-6`, donor metric 0.446159, recipient metric 0.446159, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_3e-6`, donor metric 0.431775, recipient metric 0.432253, LR 4.03e-06 -> 4.03e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.431775, recipient metric 0.435227, LR 6.04e-06 -> 6.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.431775, recipient metric 0.438252, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.431775, recipient metric 0.436623, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_14e-6`, donor metric 0.431775, recipient metric 0.431775, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_3e-6`, donor metric 0.431486, recipient metric 0.434174, LR 4.03e-06 -> 4.03e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.431486, recipient metric 0.432319, LR 6.04e-06 -> 6.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.431486, recipient metric 0.432439, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.431486, recipient metric 0.429078, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_14e-6`, donor metric 0.431486, recipient metric 0.431486, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_3e-6`, donor metric 0.43163, recipient metric 0.437788, LR 4.03e-06 -> 4.03e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.43163, recipient metric 0.431189, LR 6.04e-06 -> 6.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.43163, recipient metric 0.433346, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.43163, recipient metric 0.431995, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_14e-6`, donor metric 0.43163, recipient metric 0.43163, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_3e-6`, donor metric 0.419334, recipient metric 0.433843, LR 4.03e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.419334, recipient metric 0.437855, LR 6.04e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.419334, recipient metric 0.432188, LR 8.06e-06 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.419334, recipient metric 0.43077, LR 1.01e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_14e-6`, donor metric 0.419334, recipient metric 0.419334, LR 1.21e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_3e-6`, donor metric 0.425027, recipient metric 0.427952, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.425027, recipient metric 0.425112, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.425027, recipient metric 0.424139, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.425027, recipient metric 0.425326, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_14e-6`, donor metric 0.425027, recipient metric 0.425027, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.40959, recipient metric 0.429336, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.40959, recipient metric 0.418233, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.40959, recipient metric 0.40959, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.40959, recipient metric 0.41909, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.40959, recipient metric 0.420771, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.419845, recipient metric 0.416303, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.419845, recipient metric 0.420008, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.419845, recipient metric 0.419845, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.419845, recipient metric 0.427361, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.419845, recipient metric 0.427361, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.422681, recipient metric 0.425028, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.422681, recipient metric 0.425105, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.422681, recipient metric 0.422681, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.422681, recipient metric 0.424908, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.422681, recipient metric 0.423981, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.41822, recipient metric 0.419544, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.41822, recipient metric 0.413289, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.41822, recipient metric 0.41822, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.41822, recipient metric 0.412447, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.41822, recipient metric 0.424398, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.415455, recipient metric 0.427146, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.415455, recipient metric 0.423161, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.415455, recipient metric 0.415455, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.415455, recipient metric 0.422975, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.415455, recipient metric 0.413596, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.428829, recipient metric 0.42999, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.428829, recipient metric 0.413081, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.428829, recipient metric 0.428829, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.428829, recipient metric 0.428342, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.428829, recipient metric 0.428342, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.415881, recipient metric 0.420621, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.415881, recipient metric 0.418991, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.415881, recipient metric 0.415881, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.415881, recipient metric 0.426271, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.415881, recipient metric 0.426271, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.429464, recipient metric 0.414025, LR 6.16e-06 -> 6.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.429464, recipient metric 0.418847, LR 9.25e-06 -> 9.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.429464, recipient metric 0.429464, LR 1.23e-05 -> 1.23e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.429464, recipient metric 0.413705, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.429464, recipient metric 0.413432, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.42259, recipient metric 0.424278, LR 6.16e-06 -> 3.5e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.42259, recipient metric 0.424401, LR 9.25e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.42259, recipient metric 0.425442, LR 1.23e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.42259, recipient metric 0.42259, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.42259, recipient metric 0.425316, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.408959, recipient metric 0.424883, LR 3.5e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.408959, recipient metric 0.409613, LR 8.75e-06 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.408959, recipient metric 0.428079, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.408959, recipient metric 0.408959, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.408959, recipient metric 0.423479, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_3e-6`, donor metric 0.40115, recipient metric 0.40115, LR 7e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.40115, recipient metric 0.416353, LR 1.05e-05 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.40115, recipient metric 0.4151, LR 1.4e-05 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.40115, recipient metric 0.419275, LR 1.4e-05 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_14e-6`, donor metric 0.40115, recipient metric 0.401214, LR 1.4e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_3e-6`, donor metric 0.412712, recipient metric 0.412712, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.412712, recipient metric 0.423554, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.412712, recipient metric 0.419024, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.412712, recipient metric 0.42331, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_14e-6`, donor metric 0.412712, recipient metric 0.424565, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_3e-6`, donor metric 0.413161, recipient metric 0.413161, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.413161, recipient metric 0.421642, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.413161, recipient metric 0.4226, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.413161, recipient metric 0.433977, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_14e-6`, donor metric 0.413161, recipient metric 0.41889, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_3e-6`, donor metric 0.42182, recipient metric 0.42182, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.42182, recipient metric 0.409274, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.42182, recipient metric 0.415633, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.42182, recipient metric 0.408559, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_14e-6`, donor metric 0.42182, recipient metric 0.420911, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_3e-6`, donor metric 0.421136, recipient metric 0.421136, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.421136, recipient metric 0.425921, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.421136, recipient metric 0.427039, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.421136, recipient metric 0.427434, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_14e-6`, donor metric 0.421136, recipient metric 0.424953, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_3e-6`, donor metric 0.421358, recipient metric 0.421358, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.421358, recipient metric 0.427823, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.421358, recipient metric 0.421705, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.421358, recipient metric 0.420037, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_14e-6`, donor metric 0.421358, recipient metric 0.418987, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_3e-6` -> `lr_3e-6`, donor metric 0.415795, recipient metric 0.415795, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.415795, recipient metric 0.417494, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.415795, recipient metric 0.418464, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.415795, recipient metric 0.425771, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_3e-6` -> `lr_14e-6`, donor metric 0.415795, recipient metric 0.418037, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_3e-6` -> `lr_3e-6`, donor metric 0.427793, recipient metric 0.427793, LR 3.43e-06 -> 3.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.427793, recipient metric 0.431273, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.427793, recipient metric 0.421911, LR 6.86e-06 -> 6.86e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.427793, recipient metric 0.430693, LR 8.58e-06 -> 8.58e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_3e-6` -> `lr_14e-6`, donor metric 0.427793, recipient metric 0.430189, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417256, recipient metric 0.419151, LR 3.43e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.417256, recipient metric 0.425282, LR 5.14e-06 -> 6.43e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.417256, recipient metric 0.41914, LR 6.86e-06 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.417256, recipient metric 0.417784, LR 8.58e-06 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417256, recipient metric 0.417256, LR 1.03e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.403519, recipient metric 0.424243, LR 3e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.403519, recipient metric 0.405574, LR 6.43e-06 -> 7.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.403519, recipient metric 0.403519, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.403519, recipient metric 0.416636, LR 1.4e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.403519, recipient metric 0.41072, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.407226, recipient metric 0.421156, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.407226, recipient metric 0.407937, LR 7.72e-06 -> 7.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.407226, recipient metric 0.407226, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.407226, recipient metric 0.408402, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.407226, recipient metric 0.408176, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.413203, recipient metric 0.425455, LR 5.14e-06 -> 5.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.413203, recipient metric 0.414996, LR 7.72e-06 -> 7.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.413203, recipient metric 0.413203, LR 1.03e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.413203, recipient metric 0.411602, LR 1.29e-05 -> 1.29e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.413203, recipient metric 0.411231, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_3e-6` -> `lr_3e-6`, donor metric 0.39574, recipient metric 0.39574, LR 5.14e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.39574, recipient metric 0.408535, LR 7.72e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.39574, recipient metric 0.407549, LR 1.03e-05 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.39574, recipient metric 0.397518, LR 1.29e-05 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_3e-6` -> `lr_14e-6`, donor metric 0.39574, recipient metric 0.397601, LR 1.4e-05 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_3e-6` -> `lr_3e-6`, donor metric 0.419495, recipient metric 0.419495, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.419495, recipient metric 0.418675, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.419495, recipient metric 0.402381, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.419495, recipient metric 0.412408, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_3e-6` -> `lr_14e-6`, donor metric 0.419495, recipient metric 0.41616, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_3e-6` -> `lr_3e-6`, donor metric 0.41363, recipient metric 0.41363, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.41363, recipient metric 0.412168, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.41363, recipient metric 0.417586, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.41363, recipient metric 0.414144, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_3e-6` -> `lr_14e-6`, donor metric 0.41363, recipient metric 0.413478, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_3e-6` -> `lr_3e-6`, donor metric 0.407624, recipient metric 0.407624, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.407624, recipient metric 0.407722, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.407624, recipient metric 0.401868, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.407624, recipient metric 0.413065, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_3e-6` -> `lr_14e-6`, donor metric 0.407624, recipient metric 0.412709, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_3e-6` -> `lr_3e-6`, donor metric 0.418992, recipient metric 0.418992, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.418992, recipient metric 0.410463, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.418992, recipient metric 0.418502, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.418992, recipient metric 0.421046, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_3e-6` -> `lr_14e-6`, donor metric 0.418992, recipient metric 0.419319, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_3e-6` -> `lr_3e-6`, donor metric 0.410598, recipient metric 0.410598, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.410598, recipient metric 0.407344, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.410598, recipient metric 0.409714, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.410598, recipient metric 0.408157, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_3e-6` -> `lr_14e-6`, donor metric 0.410598, recipient metric 0.40498, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_3e-6` -> `lr_3e-6`, donor metric 0.41861, recipient metric 0.41861, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.41861, recipient metric 0.413125, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.41861, recipient metric 0.418345, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.41861, recipient metric 0.419167, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_3e-6` -> `lr_14e-6`, donor metric 0.41861, recipient metric 0.419369, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_3e-6` -> `lr_3e-6`, donor metric 0.410639, recipient metric 0.410639, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.410639, recipient metric 0.41122, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.410639, recipient metric 0.410902, LR 5.04e-06 -> 5.04e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.410639, recipient metric 0.411144, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_3e-6` -> `lr_14e-6`, donor metric 0.410639, recipient metric 0.409729, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408668, recipient metric 0.410095, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.408668, recipient metric 0.41035, LR 3.78e-06 -> 4.73e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.408668, recipient metric 0.415103, LR 5.04e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.408668, recipient metric 0.409856, LR 6.3e-06 -> 1.04e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408668, recipient metric 0.408668, LR 7.56e-06 -> 1.32e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_14e-6` -> `lr_3e-6`, donor metric 0.425236, recipient metric 0.412477, LR 3e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.425236, recipient metric 0.414405, LR 4.73e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.425236, recipient metric 0.419882, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.425236, recipient metric 0.414134, LR 1.04e-05 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_14e-6` -> `lr_14e-6`, donor metric 0.425236, recipient metric 0.425236, LR 1.32e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.398065, recipient metric 0.416824, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.398065, recipient metric 0.409076, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.398065, recipient metric 0.398065, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.398065, recipient metric 0.408794, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.398065, recipient metric 0.407965, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.4054, recipient metric 0.423206, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.4054, recipient metric 0.423585, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.4054, recipient metric 0.4054, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.4054, recipient metric 0.409539, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.4054, recipient metric 0.405247, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.419967, recipient metric 0.421472, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.419967, recipient metric 0.415503, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.419967, recipient metric 0.419967, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.419967, recipient metric 0.415896, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.419967, recipient metric 0.427697, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.424742, recipient metric 0.421065, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.424742, recipient metric 0.409013, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.424742, recipient metric 0.424742, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.424742, recipient metric 0.423729, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.424742, recipient metric 0.398707, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.410635, recipient metric 0.407046, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.410635, recipient metric 0.410786, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.410635, recipient metric 0.410635, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.410635, recipient metric 0.412994, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.410635, recipient metric 0.412124, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.409302, recipient metric 0.412159, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.409302, recipient metric 0.407767, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.409302, recipient metric 0.409302, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.409302, recipient metric 0.411943, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.409302, recipient metric 0.40721, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.408162, recipient metric 0.406156, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.408162, recipient metric 0.407507, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.408162, recipient metric 0.408162, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.408162, recipient metric 0.41259, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.408162, recipient metric 0.414066, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.403411, recipient metric 0.412827, LR 3.78e-06 -> 3.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.403411, recipient metric 0.403776, LR 5.67e-06 -> 5.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.403411, recipient metric 0.403411, LR 7.56e-06 -> 7.56e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.403411, recipient metric 0.413121, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.403411, recipient metric 0.413194, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.417332, recipient metric 0.417527, LR 3.78e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.417332, recipient metric 0.424601, LR 5.67e-06 -> 5.91e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.417332, recipient metric 0.421544, LR 7.56e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.417332, recipient metric 0.417332, LR 9.45e-06 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.417332, recipient metric 0.420668, LR 1.13e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_3e-6` -> `lr_3e-6`, donor metric 0.402333, recipient metric 0.402333, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.402333, recipient metric 0.412628, LR 5.91e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.402333, recipient metric 0.410192, LR 9.45e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.402333, recipient metric 0.415296, LR 1.3e-05 -> 3.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_3e-6` -> `lr_14e-6`, donor metric 0.402333, recipient metric 0.409157, LR 1.4e-05 -> 4.5e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_3e-6` -> `lr_3e-6`, donor metric 0.412842, recipient metric 0.412842, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.412842, recipient metric 0.424385, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.412842, recipient metric 0.413787, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.412842, recipient metric 0.420393, LR 3.75e-06 -> 3.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_3e-6` -> `lr_14e-6`, donor metric 0.412842, recipient metric 0.412459, LR 4.5e-06 -> 4.5e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_3e-6`, donor metric 0.410346, recipient metric 0.410346, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.410346, recipient metric 0.423822, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.410346, recipient metric 0.410346, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.410346, recipient metric 0.408344, LR 3.75e-06 -> 3.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_3e-6` -> `lr_14e-6`, donor metric 0.410346, recipient metric 0.423829, LR 4.5e-06 -> 4.5e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_3e-6`, donor metric 0.400868, recipient metric 0.410368, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.400868, recipient metric 0.404486, LR 3e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.400868, recipient metric 0.408103, LR 3e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.400868, recipient metric 0.401958, LR 3.75e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_14e-6`, donor metric 0.400868, recipient metric 0.400868, LR 4.5e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.397499, recipient metric 0.408575, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.397499, recipient metric 0.405796, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.397499, recipient metric 0.397499, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.397499, recipient metric 0.404493, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.397499, recipient metric 0.403039, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.40202, recipient metric 0.407445, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.40202, recipient metric 0.407622, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.40202, recipient metric 0.40202, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.40202, recipient metric 0.407038, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.40202, recipient metric 0.39751, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.410743, recipient metric 0.410617, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.410743, recipient metric 0.410985, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.410743, recipient metric 0.410743, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.410743, recipient metric 0.416515, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.410743, recipient metric 0.41363, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.410008, recipient metric 0.416566, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.410008, recipient metric 0.419521, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.410008, recipient metric 0.410008, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.410008, recipient metric 0.406274, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.410008, recipient metric 0.415415, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.409439, recipient metric 0.406648, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.409439, recipient metric 0.41843, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.409439, recipient metric 0.409439, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.409439, recipient metric 0.405648, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.409439, recipient metric 0.405451, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.399307, recipient metric 0.405561, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.399307, recipient metric 0.405203, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.399307, recipient metric 0.399307, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.399307, recipient metric 0.414297, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.399307, recipient metric 0.399245, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.418115, recipient metric 0.403807, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.418115, recipient metric 0.404484, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.418115, recipient metric 0.418115, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.418115, recipient metric 0.411495, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.418115, recipient metric 0.404538, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.417461, recipient metric 0.410816, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.417461, recipient metric 0.414977, LR 3.44e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.417461, recipient metric 0.417461, LR 4.59e-06 -> 4.59e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.417461, recipient metric 0.409682, LR 5.74e-06 -> 5.74e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.417461, recipient metric 0.414241, LR 6.89e-06 -> 6.89e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.406373, recipient metric 0.415224, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.406373, recipient metric 0.406373, LR 3.44e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.406373, recipient metric 0.406639, LR 4.59e-06 -> 3.44e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.406373, recipient metric 0.411894, LR 5.74e-06 -> 4.73e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.406373, recipient metric 0.414775, LR 6.89e-06 -> 6.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_3e-6`, donor metric 0.400523, recipient metric 0.40948, LR 3e-06 -> 3.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_5_75e-6`, donor metric 0.400523, recipient metric 0.406301, LR 3e-06 -> 4.61e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_8_5e-6`, donor metric 0.400523, recipient metric 0.40678, LR 3.44e-06 -> 6.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_11_25e-6`, donor metric 0.400523, recipient metric 0.418758, LR 4.73e-06 -> 7.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_14e-6`, donor metric 0.400523, recipient metric 0.400523, LR 6.02e-06 -> 9.22e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.395884, recipient metric 0.411065, LR 3.07e-06 -> 3.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.395884, recipient metric 0.417496, LR 4.61e-06 -> 4.61e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.395884, recipient metric 0.395884, LR 6.14e-06 -> 6.14e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.395884, recipient metric 0.417755, LR 7.68e-06 -> 7.68e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.395884, recipient metric 0.39646, LR 9.22e-06 -> 9.22e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.391014, recipient metric 0.392802, LR 3.07e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.391014, recipient metric 0.412966, LR 4.61e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.391014, recipient metric 0.414156, LR 6.14e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.391014, recipient metric 0.391014, LR 7.68e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.391014, recipient metric 0.415657, LR 9.22e-06 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.403383, recipient metric 0.406707, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.403383, recipient metric 0.399625, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.403383, recipient metric 0.405087, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.403383, recipient metric 0.403383, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.403383, recipient metric 0.398686, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.410672, recipient metric 0.413952, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.410672, recipient metric 0.413754, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.410672, recipient metric 0.408251, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.410672, recipient metric 0.410672, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.410672, recipient metric 0.411816, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.421745, recipient metric 0.397483, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.421745, recipient metric 0.407035, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.421745, recipient metric 0.397026, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.421745, recipient metric 0.421745, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.421745, recipient metric 0.396212, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.416518, recipient metric 0.414191, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.416518, recipient metric 0.415985, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.416518, recipient metric 0.420873, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.416518, recipient metric 0.416518, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.416518, recipient metric 0.413536, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.41203, recipient metric 0.409658, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.41203, recipient metric 0.410125, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.41203, recipient metric 0.403195, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.41203, recipient metric 0.41203, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.41203, recipient metric 0.402604, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.411589, recipient metric 0.41854, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.411589, recipient metric 0.410898, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.411589, recipient metric 0.411572, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.411589, recipient metric 0.411589, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.411589, recipient metric 0.413242, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_11_25e-6` -> `lr_3e-6`, donor metric 0.405055, recipient metric 0.41455, LR 3.92e-06 -> 3.92e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_11_25e-6` -> `lr_5_75e-6`, donor metric 0.405055, recipient metric 0.413554, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_11_25e-6` -> `lr_8_5e-6`, donor metric 0.405055, recipient metric 0.406597, LR 7.83e-06 -> 7.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_11_25e-6` -> `lr_11_25e-6`, donor metric 0.405055, recipient metric 0.405055, LR 9.79e-06 -> 9.79e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_11_25e-6` -> `lr_14e-6`, donor metric 0.405055, recipient metric 0.406112, LR 1.18e-05 -> 1.18e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.399833, recipient metric 0.419171, LR 3.92e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.399833, recipient metric 0.399833, LR 5.88e-06 -> 3.67e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.399833, recipient metric 0.402761, LR 7.83e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.399833, recipient metric 0.407702, LR 9.79e-06 -> 8.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.399833, recipient metric 0.407282, LR 1.18e-05 -> 1.03e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.420748, recipient metric 0.415631, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.420748, recipient metric 0.420748, LR 3.67e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.420748, recipient metric 0.419699, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.420748, recipient metric 0.406199, LR 8.08e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.420748, recipient metric 0.406239, LR 1.03e-05 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.411743, recipient metric 0.418352, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.411743, recipient metric 0.411743, LR 4.41e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.411743, recipient metric 0.410786, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.411743, recipient metric 0.407885, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.411743, recipient metric 0.417948, LR 8.81e-06 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.409499, recipient metric 0.400227, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.409499, recipient metric 0.409499, LR 4.41e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.409499, recipient metric 0.40398, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.409499, recipient metric 0.414801, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.409499, recipient metric 0.415736, LR 8.81e-06 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.41092, recipient metric 0.409679, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.41092, recipient metric 0.41092, LR 4.41e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.41092, recipient metric 0.411021, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.41092, recipient metric 0.41061, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.41092, recipient metric 0.403742, LR 8.81e-06 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.403351, recipient metric 0.401715, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.403351, recipient metric 0.403351, LR 4.41e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.403351, recipient metric 0.406162, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.403351, recipient metric 0.406161, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.403351, recipient metric 0.40638, LR 8.81e-06 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.401692, recipient metric 0.406365, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.401692, recipient metric 0.401692, LR 4.41e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.401692, recipient metric 0.412791, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.401692, recipient metric 0.414077, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.401692, recipient metric 0.404583, LR 8.81e-06 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_5_75e-6` -> `lr_3e-6`, donor metric 0.400386, recipient metric 0.413623, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_5_75e-6` -> `lr_5_75e-6`, donor metric 0.400386, recipient metric 0.400386, LR 4.41e-06 -> 4.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_5_75e-6` -> `lr_8_5e-6`, donor metric 0.400386, recipient metric 0.413829, LR 5.88e-06 -> 5.88e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_5_75e-6` -> `lr_11_25e-6`, donor metric 0.400386, recipient metric 0.407361, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_5_75e-6` -> `lr_14e-6`, donor metric 0.400386, recipient metric 0.409743, LR 8.81e-06 -> 8.81e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_3e-6`, donor metric 0.402549, recipient metric 0.402549, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_5_75e-6`, donor metric 0.402549, recipient metric 0.405291, LR 4.41e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_8_5e-6`, donor metric 0.402549, recipient metric 0.415005, LR 5.88e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_11_25e-6`, donor metric 0.402549, recipient metric 0.414992, LR 7.35e-06 -> 4.13e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_3e-6` -> `lr_14e-6`, donor metric 0.402549, recipient metric 0.413614, LR 8.81e-06 -> 5.25e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_8_5e-6` -> `lr_3e-6`, donor metric 0.396378, recipient metric 0.415102, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_8_5e-6` -> `lr_5_75e-6`, donor metric 0.396378, recipient metric 0.400857, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_8_5e-6` -> `lr_8_5e-6`, donor metric 0.396378, recipient metric 0.396378, LR 3e-06 -> 3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_8_5e-6` -> `lr_11_25e-6`, donor metric 0.396378, recipient metric 0.415136, LR 4.13e-06 -> 3.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_8_5e-6` -> `lr_14e-6`, donor metric 0.396378, recipient metric 0.40084, LR 5.25e-06 -> 4.5e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- [Skipped exploits (significance gating)](plots/report/skipped_exploits.csv) -- 0 donor->recipient replacement(s) declined for insufficient significance

## Method
- Method: `anchor_copy_lr_recenter`
- Population: 5 trials
- Training interval: 120000 samples/trial chunk (1x samples_per_epoch)
- Evaluation interval: every 1 training chunk(s), 150000 validation samples
- Exploit interval: every 1 training chunk(s)
- Exploit significance gating: disabled (nominal rank order only)
- Burn-in: 0 generation(s) (observe-only, no exploit/controller LR action applied)
- Monitor-tier cadence: disabled generation(s), all population members, read-only
- Full-tier cadence: 20 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `cc14890f6ab6b250f0ade5d6a5bfd7fb0759d805`
- Git dirty: `True`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_v2_validation.yaml', '--slots', 'iutgpu05:1,iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3', '--experiment-name', 'anchor_copy_lr_recenter_v2_validation']`
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
- No data-loader shutdown-race warnings observed across 550 evaluation(s).
