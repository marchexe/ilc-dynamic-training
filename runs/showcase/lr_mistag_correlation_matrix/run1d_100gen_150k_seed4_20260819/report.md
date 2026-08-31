# anchor_copy_lr_recenter_100gen_seed4_20260819_111523

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
- Final checkpoint controller objective: 1.01492 by `lr_6e-6`
- Global best configured metric: 0.386866 by `lr_14e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_100gen_seed4_20260819_111523/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_14e-6`, generation 61), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_6e-6`, generation 63) -- these are two distinct selection criteria, not the same checkpoint.
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=400, Pearson r=-0.092 (95% CI -0.179 to -0.000), Spearman rho=-0.089 (95% CI -0.192 to 0.009)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=0.853, Spearman rho=0.702
- Best checkpoint by tier: control: `lr_9e-6` gen 59 (0.401581), full_holdout: `lr_9e-6` gen 79 (0.391326)
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
| lr_14e-6 | 0.1886 | 0.07425 | 3.117 | 0.1886 | 0.5158 | 0.06622 | 2.695 | 1.116 | 0.5661 | 0.3012 | 0.413 | 1.4e-05 | anchor |
| lr_3e-6 | 0.1747 | 0.06818 | 3.289 | 0.1925 | 0.5157 | 0.06216 | 2.709 | 1.135 | 0.5603 | 0.2947 | 0.4063 | 1.01e-05 | - |
| lr_6e-6 | 0.1805 | 0.06211 | 3.271 | 0.1823 | 0.5207 | 0.06412 | 2.712 | 1.126 | 0.5651 | 0.286 | 0.402 | 1.13e-05 | winner |
| lr_9e-6 | 0.1747 | 0.07018 | 3.285 | 0.1925 | 0.5157 | 0.06216 | 2.701 | 1.135 | 0.5599 | 0.2967 | 0.4076 | 1.26e-05 | - |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_9e-6 | 0.4386 | 0.6165 | 0.312 | 9e-06 | 9e-06 | 9e-06 | accepted_new_anchor | no |
| 1 | lr_3e-6 | 0.4465 | 0.622 | 0.3206 | 7.2e-06 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 2 | lr_9e-6 | 0.4527 | 0.6199 | 0.3306 | 9e-06 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 3 | lr_14e-6 | 0.4509 | 0.6138 | 0.3312 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 4 | lr_14e-6 | 0.4454 | 0.6023 | 0.3293 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 5 | lr_9e-6 | 0.4435 | 0.6227 | 0.3159 | 9e-06 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 6 | lr_3e-6 | 0.4408 | 0.6064 | 0.3204 | 7.2e-06 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 7 | lr_14e-6 | 0.4412 | 0.619 | 0.3144 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 8 | lr_6e-6 | 0.4534 | 0.6212 | 0.3309 | 8.1e-06 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 9 | lr_14e-6 | 0.4467 | 0.6206 | 0.3215 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 10 | lr_14e-6 | 0.4403 | 0.6057 | 0.3201 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 11 | lr_9e-6 | 0.4373 | 0.6178 | 0.3095 | 9e-06 | 9e-06 | 9e-06 | accepted_new_anchor | no |
| 12 | lr_14e-6 | 0.4368 | 0.6006 | 0.3177 | 1.08e-05 | 9e-06 | 1.08e-05 | accepted_new_anchor | no |
| 13 | lr_9e-6 | 0.4184 | 0.5886 | 0.2975 | 1.08e-05 | 1.08e-05 | 1.08e-05 | accepted_new_anchor | no |
| 14 | lr_6e-6 | 0.425 | 0.5844 | 0.3091 | 9.72e-06 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 15 | lr_9e-6 | 0.424 | 0.6037 | 0.2978 | 1.08e-05 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 16 | lr_9e-6 | 0.4265 | 0.6125 | 0.297 | 1.08e-05 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 17 | lr_6e-6 | 0.4236 | 0.5786 | 0.3101 | 9.72e-06 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 18 | lr_9e-6 | 0.4252 | 0.5916 | 0.3057 | 1.08e-05 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 19 | lr_14e-6 | 0.4274 | 0.5877 | 0.3109 | 1.296e-05 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 20 | lr_6e-6 | 0.4289 | 0.5773 | 0.3186 | 9.72e-06 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 21 | lr_6e-6 | 0.4309 | 0.5818 | 0.3192 | 9.72e-06 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 22 | lr_14e-6 | 0.4278 | 0.5761 | 0.3177 | 1.296e-05 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 23 | lr_6e-6 | 0.4229 | 0.5963 | 0.2999 | 9.72e-06 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 24 | lr_14e-6 | 0.4234 | 0.5887 | 0.3045 | 1.296e-05 | 1.08e-05 | 1.08e-05 | rewound_to_previous_anchor | no |
| 25 | lr_6e-6 | 0.4151 | 0.5805 | 0.2969 | 9.72e-06 | 1.08e-05 | 9.72e-06 | accepted_new_anchor | no |
| 26 | lr_14e-6 | 0.4219 | 0.5782 | 0.3078 | 1.166e-05 | 9.72e-06 | 9.72e-06 | rewound_to_previous_anchor | no |
| 27 | lr_9e-6 | 0.422 | 0.5973 | 0.2981 | 9.72e-06 | 9.72e-06 | 9.72e-06 | rewound_to_previous_anchor | no |
| 28 | lr_14e-6 | 0.4253 | 0.585 | 0.3092 | 1.166e-05 | 9.72e-06 | 9.72e-06 | rewound_to_previous_anchor | no |
| 29 | lr_9e-6 | 0.4143 | 0.5676 | 0.3023 | 9.72e-06 | 9.72e-06 | 9.72e-06 | accepted_new_anchor | no |
| 30 | lr_9e-6 | 0.4228 | 0.5875 | 0.3043 | 9.72e-06 | 9.72e-06 | 9.72e-06 | rewound_to_previous_anchor | no |
| 31 | lr_6e-6 | 0.4135 | 0.5805 | 0.2946 | 8.748e-06 | 9.72e-06 | 8.748e-06 | accepted_new_anchor | no |
| 32 | lr_3e-6 | 0.4191 | 0.5901 | 0.2977 | 6.998e-06 | 8.748e-06 | 8.748e-06 | rewound_to_previous_anchor | no |
| 33 | lr_3e-6 | 0.4182 | 0.5769 | 0.3031 | 6.998e-06 | 8.748e-06 | 8.748e-06 | rewound_to_previous_anchor | no |
| 34 | lr_3e-6 | 0.4172 | 0.5798 | 0.3002 | 6.998e-06 | 8.748e-06 | 8.748e-06 | rewound_to_previous_anchor | no |
| 35 | lr_9e-6 | 0.4151 | 0.5746 | 0.2999 | 8.748e-06 | 8.748e-06 | 8.748e-06 | rewound_to_previous_anchor | no |
| 36 | lr_14e-6 | 0.4133 | 0.5759 | 0.2966 | 1.05e-05 | 8.748e-06 | 1.05e-05 | accepted_new_anchor | no |
| 37 | lr_9e-6 | 0.4054 | 0.5631 | 0.2918 | 1.05e-05 | 1.05e-05 | 1.05e-05 | accepted_new_anchor | no |
| 38 | lr_3e-6 | 0.4183 | 0.5744 | 0.3046 | 8.398e-06 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 39 | lr_9e-6 | 0.4125 | 0.5812 | 0.2927 | 1.05e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 40 | lr_9e-6 | 0.4051 | 0.5624 | 0.2918 | 1.05e-05 | 1.05e-05 | 1.05e-05 | accepted_new_anchor | no |
| 41 | lr_3e-6 | 0.4152 | 0.5731 | 0.3008 | 8.398e-06 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 42 | lr_9e-6 | 0.4041 | 0.5576 | 0.2929 | 1.05e-05 | 1.05e-05 | 1.05e-05 | accepted_new_anchor | no |
| 43 | lr_9e-6 | 0.3938 | 0.5659 | 0.274 | 1.05e-05 | 1.05e-05 | 1.05e-05 | accepted_new_anchor | no |
| 44 | lr_6e-6 | 0.4059 | 0.5517 | 0.2986 | 9.448e-06 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 45 | lr_14e-6 | 0.4018 | 0.5502 | 0.2935 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 46 | lr_14e-6 | 0.413 | 0.5457 | 0.3125 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 47 | lr_6e-6 | 0.4046 | 0.5541 | 0.2954 | 9.448e-06 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 48 | lr_9e-6 | 0.4105 | 0.5558 | 0.3032 | 1.05e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 49 | lr_9e-6 | 0.4081 | 0.5533 | 0.3011 | 1.05e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 50 | lr_3e-6 | 0.4099 | 0.5549 | 0.3027 | 8.398e-06 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 51 | lr_9e-6 | 0.4088 | 0.5658 | 0.2953 | 1.05e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 52 | lr_9e-6 | 0.4136 | 0.5692 | 0.3004 | 1.05e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 53 | lr_14e-6 | 0.4158 | 0.5716 | 0.3024 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 54 | lr_14e-6 | 0.4081 | 0.5436 | 0.3064 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 55 | lr_14e-6 | 0.4107 | 0.5552 | 0.3038 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 56 | lr_14e-6 | 0.4083 | 0.5466 | 0.305 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 57 | lr_3e-6 | 0.3975 | 0.5368 | 0.2943 | 8.398e-06 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 58 | lr_14e-6 | 0.4069 | 0.5592 | 0.2961 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 59 | lr_9e-6 | 0.4016 | 0.5393 | 0.299 | 1.05e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 60 | lr_14e-6 | 0.4085 | 0.5531 | 0.3017 | 1.26e-05 | 1.05e-05 | 1.05e-05 | rewound_to_previous_anchor | no |
| 61 | lr_14e-6 | 0.3869 | 0.5363 | 0.2791 | 1.26e-05 | 1.05e-05 | 1.26e-05 | accepted_new_anchor | no |
| 62 | lr_6e-6 | 0.399 | 0.5477 | 0.2907 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 63 | lr_6e-6 | 0.4001 | 0.5408 | 0.296 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 64 | lr_3e-6 | 0.4093 | 0.5568 | 0.3009 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 65 | lr_9e-6 | 0.3882 | 0.5296 | 0.2845 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 66 | lr_6e-6 | 0.3986 | 0.5455 | 0.2913 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 67 | lr_9e-6 | 0.408 | 0.5565 | 0.2991 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 68 | lr_6e-6 | 0.4092 | 0.5645 | 0.2967 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 69 | lr_6e-6 | 0.401 | 0.5527 | 0.2909 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 70 | lr_6e-6 | 0.4069 | 0.5489 | 0.3016 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 71 | lr_9e-6 | 0.4103 | 0.561 | 0.3 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 72 | lr_6e-6 | 0.4003 | 0.5524 | 0.2902 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 73 | lr_14e-6 | 0.4056 | 0.5387 | 0.3053 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 74 | lr_3e-6 | 0.408 | 0.564 | 0.2952 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 75 | lr_14e-6 | 0.4023 | 0.5542 | 0.292 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 76 | lr_6e-6 | 0.4061 | 0.5567 | 0.2963 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 77 | lr_6e-6 | 0.41 | 0.5696 | 0.2951 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 78 | lr_9e-6 | 0.4063 | 0.568 | 0.2906 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 79 | lr_6e-6 | 0.4108 | 0.5603 | 0.3012 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 80 | lr_3e-6 | 0.4204 | 0.5696 | 0.3103 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 81 | lr_9e-6 | 0.4024 | 0.5548 | 0.2918 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 82 | lr_14e-6 | 0.3978 | 0.5488 | 0.2883 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 83 | lr_6e-6 | 0.4053 | 0.5542 | 0.2964 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 84 | lr_9e-6 | 0.4077 | 0.5429 | 0.3061 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 85 | lr_3e-6 | 0.4042 | 0.559 | 0.2923 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 86 | lr_9e-6 | 0.4062 | 0.5518 | 0.2991 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 87 | lr_14e-6 | 0.4065 | 0.5631 | 0.2935 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 88 | lr_14e-6 | 0.3974 | 0.5403 | 0.2923 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 89 | lr_6e-6 | 0.3962 | 0.5461 | 0.2875 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 90 | lr_14e-6 | 0.4049 | 0.5556 | 0.2951 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 91 | lr_6e-6 | 0.4105 | 0.5661 | 0.2977 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 92 | lr_9e-6 | 0.4117 | 0.553 | 0.3065 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 93 | lr_9e-6 | 0.4088 | 0.557 | 0.3 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 94 | lr_3e-6 | 0.4088 | 0.5683 | 0.2941 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 95 | lr_14e-6 | 0.3986 | 0.5414 | 0.2934 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 96 | lr_14e-6 | 0.4095 | 0.5578 | 0.3006 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 97 | lr_14e-6 | 0.4002 | 0.5351 | 0.2993 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 98 | lr_6e-6 | 0.4065 | 0.5459 | 0.3026 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 99 | lr_6e-6 | 0.402 | 0.5651 | 0.286 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_9e-6` -> `lr_3e-6`, donor metric 0.438559, recipient metric 0.448184, LR 3e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_6e-6`, donor metric 0.438559, recipient metric 0.456956, LR 6e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_9e-6`, donor metric 0.438559, recipient metric 0.438559, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_14e-6`, donor metric 0.438559, recipient metric 0.441729, LR 1.4e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_3e-6`, donor metric 0.458278, recipient metric 0.446546, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_6e-6`, donor metric 0.458278, recipient metric 0.446918, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_9e-6`, donor metric 0.458278, recipient metric 0.458278, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_14e-6`, donor metric 0.458278, recipient metric 0.457363, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_3e-6`, donor metric 0.452723, recipient metric 0.462161, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_6e-6`, donor metric 0.452723, recipient metric 0.452762, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_9e-6`, donor metric 0.452723, recipient metric 0.452723, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_14e-6`, donor metric 0.452723, recipient metric 0.454643, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_3e-6`, donor metric 0.456138, recipient metric 0.453294, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_6e-6`, donor metric 0.456138, recipient metric 0.461793, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_9e-6`, donor metric 0.456138, recipient metric 0.456138, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_14e-6`, donor metric 0.456138, recipient metric 0.450871, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_3e-6`, donor metric 0.453915, recipient metric 0.450184, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_6e-6`, donor metric 0.453915, recipient metric 0.449224, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_9e-6`, donor metric 0.453915, recipient metric 0.453915, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_14e-6`, donor metric 0.453915, recipient metric 0.445352, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_3e-6`, donor metric 0.443516, recipient metric 0.450597, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_6e-6`, donor metric 0.443516, recipient metric 0.449065, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_9e-6`, donor metric 0.443516, recipient metric 0.443516, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_9e-6` -> `lr_14e-6`, donor metric 0.443516, recipient metric 0.449653, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_3e-6`, donor metric 0.446529, recipient metric 0.44082, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_6e-6`, donor metric 0.446529, recipient metric 0.444447, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_9e-6`, donor metric 0.446529, recipient metric 0.446529, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_9e-6` -> `lr_14e-6`, donor metric 0.446529, recipient metric 0.448686, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_3e-6`, donor metric 0.451757, recipient metric 0.441598, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_6e-6`, donor metric 0.451757, recipient metric 0.449832, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_9e-6`, donor metric 0.451757, recipient metric 0.451757, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_9e-6` -> `lr_14e-6`, donor metric 0.451757, recipient metric 0.441187, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_9e-6` -> `lr_3e-6`, donor metric 0.453723, recipient metric 0.454515, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_9e-6` -> `lr_6e-6`, donor metric 0.453723, recipient metric 0.453399, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_9e-6` -> `lr_9e-6`, donor metric 0.453723, recipient metric 0.453723, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_9e-6` -> `lr_14e-6`, donor metric 0.453723, recipient metric 0.455073, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_9e-6` -> `lr_3e-6`, donor metric 0.45279, recipient metric 0.449158, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_9e-6` -> `lr_6e-6`, donor metric 0.45279, recipient metric 0.45047, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_9e-6` -> `lr_9e-6`, donor metric 0.45279, recipient metric 0.45279, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_9e-6` -> `lr_14e-6`, donor metric 0.45279, recipient metric 0.446651, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_3e-6`, donor metric 0.446981, recipient metric 0.441661, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_6e-6`, donor metric 0.446981, recipient metric 0.447205, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_9e-6`, donor metric 0.446981, recipient metric 0.446981, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_14e-6`, donor metric 0.446981, recipient metric 0.440346, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_3e-6`, donor metric 0.437266, recipient metric 0.451441, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_6e-6`, donor metric 0.437266, recipient metric 0.444257, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_9e-6`, donor metric 0.437266, recipient metric 0.437266, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_14e-6`, donor metric 0.437266, recipient metric 0.449728, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_3e-6`, donor metric 0.43678, recipient metric 0.443425, LR 7.2e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_6e-6`, donor metric 0.43678, recipient metric 0.443962, LR 8.1e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_9e-6`, donor metric 0.43678, recipient metric 0.443754, LR 9e-06 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_14e-6`, donor metric 0.43678, recipient metric 0.43678, LR 1.08e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_9e-6` -> `lr_3e-6`, donor metric 0.418429, recipient metric 0.430196, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_9e-6` -> `lr_6e-6`, donor metric 0.418429, recipient metric 0.433724, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_9e-6` -> `lr_9e-6`, donor metric 0.418429, recipient metric 0.418429, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_9e-6` -> `lr_14e-6`, donor metric 0.418429, recipient metric 0.431595, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_3e-6`, donor metric 0.426763, recipient metric 0.435719, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_6e-6`, donor metric 0.426763, recipient metric 0.425, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_9e-6`, donor metric 0.426763, recipient metric 0.426763, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_14e-6`, donor metric 0.426763, recipient metric 0.429663, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_3e-6`, donor metric 0.423995, recipient metric 0.428634, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_6e-6`, donor metric 0.423995, recipient metric 0.424862, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_9e-6`, donor metric 0.423995, recipient metric 0.423995, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_14e-6`, donor metric 0.423995, recipient metric 0.431468, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_3e-6`, donor metric 0.42653, recipient metric 0.431027, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_6e-6`, donor metric 0.42653, recipient metric 0.428374, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_9e-6`, donor metric 0.42653, recipient metric 0.42653, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_14e-6`, donor metric 0.42653, recipient metric 0.427559, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_3e-6`, donor metric 0.433583, recipient metric 0.435182, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_6e-6`, donor metric 0.433583, recipient metric 0.423588, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_9e-6`, donor metric 0.433583, recipient metric 0.433583, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_14e-6`, donor metric 0.433583, recipient metric 0.425021, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_3e-6`, donor metric 0.425222, recipient metric 0.437222, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_6e-6`, donor metric 0.425222, recipient metric 0.435205, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_9e-6`, donor metric 0.425222, recipient metric 0.425222, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_14e-6`, donor metric 0.425222, recipient metric 0.432082, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_3e-6`, donor metric 0.428869, recipient metric 0.431056, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_6e-6`, donor metric 0.428869, recipient metric 0.437339, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_9e-6`, donor metric 0.428869, recipient metric 0.428869, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_14e-6`, donor metric 0.428869, recipient metric 0.427448, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_3e-6`, donor metric 0.435771, recipient metric 0.43208, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_6e-6`, donor metric 0.435771, recipient metric 0.428902, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_9e-6`, donor metric 0.435771, recipient metric 0.435771, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_14e-6`, donor metric 0.435771, recipient metric 0.442323, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_3e-6`, donor metric 0.436071, recipient metric 0.434877, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_6e-6`, donor metric 0.436071, recipient metric 0.430939, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_9e-6`, donor metric 0.436071, recipient metric 0.436071, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_14e-6`, donor metric 0.436071, recipient metric 0.43385, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_3e-6`, donor metric 0.428829, recipient metric 0.433624, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_6e-6`, donor metric 0.428829, recipient metric 0.43749, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_9e-6`, donor metric 0.428829, recipient metric 0.428829, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_14e-6`, donor metric 0.428829, recipient metric 0.427838, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_3e-6`, donor metric 0.425576, recipient metric 0.425589, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_6e-6`, donor metric 0.425576, recipient metric 0.422912, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_9e-6`, donor metric 0.425576, recipient metric 0.425576, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_14e-6`, donor metric 0.425576, recipient metric 0.422981, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_3e-6`, donor metric 0.431107, recipient metric 0.435286, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_6e-6`, donor metric 0.431107, recipient metric 0.431515, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_9e-6`, donor metric 0.431107, recipient metric 0.431107, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_14e-6`, donor metric 0.431107, recipient metric 0.423408, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_3e-6`, donor metric 0.415146, recipient metric 0.417349, LR 8.64e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_6e-6`, donor metric 0.415146, recipient metric 0.415146, LR 9.72e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_9e-6`, donor metric 0.415146, recipient metric 0.426009, LR 1.08e-05 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_14e-6`, donor metric 0.415146, recipient metric 0.426142, LR 1.3e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_3e-6`, donor metric 0.427652, recipient metric 0.424069, LR 7.78e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_6e-6`, donor metric 0.427652, recipient metric 0.427652, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_9e-6`, donor metric 0.427652, recipient metric 0.424203, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_14e-6`, donor metric 0.427652, recipient metric 0.42189, LR 1.17e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_3e-6`, donor metric 0.42938, recipient metric 0.42412, LR 7.78e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_6e-6`, donor metric 0.42938, recipient metric 0.42938, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_9e-6`, donor metric 0.42938, recipient metric 0.421954, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_14e-6`, donor metric 0.42938, recipient metric 0.427231, LR 1.17e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_3e-6`, donor metric 0.428066, recipient metric 0.427681, LR 7.78e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_6e-6`, donor metric 0.428066, recipient metric 0.428066, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_9e-6`, donor metric 0.428066, recipient metric 0.426823, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_14e-6`, donor metric 0.428066, recipient metric 0.425297, LR 1.17e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_9e-6` -> `lr_3e-6`, donor metric 0.414264, recipient metric 0.417959, LR 7.78e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_9e-6` -> `lr_6e-6`, donor metric 0.414264, recipient metric 0.417304, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_9e-6` -> `lr_9e-6`, donor metric 0.414264, recipient metric 0.414264, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_9e-6` -> `lr_14e-6`, donor metric 0.414264, recipient metric 0.418118, LR 1.17e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_9e-6` -> `lr_3e-6`, donor metric 0.422792, recipient metric 0.42497, LR 7.78e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_9e-6` -> `lr_6e-6`, donor metric 0.422792, recipient metric 0.436178, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_9e-6` -> `lr_9e-6`, donor metric 0.422792, recipient metric 0.422792, LR 9.72e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_9e-6` -> `lr_14e-6`, donor metric 0.422792, recipient metric 0.438297, LR 1.17e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_3e-6`, donor metric 0.413534, recipient metric 0.420077, LR 7.78e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_6e-6`, donor metric 0.413534, recipient metric 0.413534, LR 8.75e-06 -> 7.87e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_9e-6`, donor metric 0.413534, recipient metric 0.415924, LR 9.72e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_14e-6`, donor metric 0.413534, recipient metric 0.415535, LR 1.17e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_3e-6`, donor metric 0.419893, recipient metric 0.41911, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_6e-6`, donor metric 0.419893, recipient metric 0.419893, LR 7.87e-06 -> 7.87e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_9e-6`, donor metric 0.419893, recipient metric 0.420735, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_14e-6`, donor metric 0.419893, recipient metric 0.420996, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_3e-6`, donor metric 0.419692, recipient metric 0.418184, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_6e-6`, donor metric 0.419692, recipient metric 0.419692, LR 7.87e-06 -> 7.87e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_9e-6`, donor metric 0.419692, recipient metric 0.425218, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_14e-6`, donor metric 0.419692, recipient metric 0.423503, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_3e-6`, donor metric 0.423451, recipient metric 0.4172, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_6e-6`, donor metric 0.423451, recipient metric 0.423451, LR 7.87e-06 -> 7.87e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_9e-6`, donor metric 0.423451, recipient metric 0.419087, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_14e-6`, donor metric 0.423451, recipient metric 0.417928, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_3e-6`, donor metric 0.42976, recipient metric 0.421551, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_6e-6`, donor metric 0.42976, recipient metric 0.42976, LR 7.87e-06 -> 7.87e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_9e-6`, donor metric 0.42976, recipient metric 0.415095, LR 8.75e-06 -> 8.75e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_14e-6`, donor metric 0.42976, recipient metric 0.431092, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_3e-6`, donor metric 0.413287, recipient metric 0.414223, LR 7e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_6e-6`, donor metric 0.413287, recipient metric 0.427914, LR 7.87e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_9e-6`, donor metric 0.413287, recipient metric 0.413807, LR 8.75e-06 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_14e-6`, donor metric 0.413287, recipient metric 0.413287, LR 1.05e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_3e-6`, donor metric 0.405375, recipient metric 0.422326, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_6e-6`, donor metric 0.405375, recipient metric 0.422019, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_9e-6`, donor metric 0.405375, recipient metric 0.405375, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_14e-6`, donor metric 0.405375, recipient metric 0.415837, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_9e-6` -> `lr_3e-6`, donor metric 0.421481, recipient metric 0.418251, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_9e-6` -> `lr_6e-6`, donor metric 0.421481, recipient metric 0.421558, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_9e-6` -> `lr_9e-6`, donor metric 0.421481, recipient metric 0.421481, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_9e-6` -> `lr_14e-6`, donor metric 0.421481, recipient metric 0.426395, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_9e-6` -> `lr_3e-6`, donor metric 0.412464, recipient metric 0.429978, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_9e-6` -> `lr_6e-6`, donor metric 0.412464, recipient metric 0.420993, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_9e-6` -> `lr_9e-6`, donor metric 0.412464, recipient metric 0.412464, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_9e-6` -> `lr_14e-6`, donor metric 0.412464, recipient metric 0.415337, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_9e-6` -> `lr_3e-6`, donor metric 0.405097, recipient metric 0.423327, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_9e-6` -> `lr_6e-6`, donor metric 0.405097, recipient metric 0.417848, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_9e-6` -> `lr_9e-6`, donor metric 0.405097, recipient metric 0.405097, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_9e-6` -> `lr_14e-6`, donor metric 0.405097, recipient metric 0.421897, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_9e-6` -> `lr_3e-6`, donor metric 0.418485, recipient metric 0.415189, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_9e-6` -> `lr_6e-6`, donor metric 0.418485, recipient metric 0.415813, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_9e-6` -> `lr_9e-6`, donor metric 0.418485, recipient metric 0.418485, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_9e-6` -> `lr_14e-6`, donor metric 0.418485, recipient metric 0.419149, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_9e-6` -> `lr_3e-6`, donor metric 0.404098, recipient metric 0.41933, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_9e-6` -> `lr_6e-6`, donor metric 0.404098, recipient metric 0.418791, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_9e-6` -> `lr_9e-6`, donor metric 0.404098, recipient metric 0.404098, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_9e-6` -> `lr_14e-6`, donor metric 0.404098, recipient metric 0.412885, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_9e-6` -> `lr_3e-6`, donor metric 0.393766, recipient metric 0.395292, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_9e-6` -> `lr_6e-6`, donor metric 0.393766, recipient metric 0.40708, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_9e-6` -> `lr_9e-6`, donor metric 0.393766, recipient metric 0.393766, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_9e-6` -> `lr_14e-6`, donor metric 0.393766, recipient metric 0.404582, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_9e-6` -> `lr_3e-6`, donor metric 0.420102, recipient metric 0.421159, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_9e-6` -> `lr_6e-6`, donor metric 0.420102, recipient metric 0.405855, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_9e-6` -> `lr_9e-6`, donor metric 0.420102, recipient metric 0.420102, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_9e-6` -> `lr_14e-6`, donor metric 0.420102, recipient metric 0.418344, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417199, recipient metric 0.417886, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417199, recipient metric 0.417424, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417199, recipient metric 0.417199, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417199, recipient metric 0.40182, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_9e-6` -> `lr_3e-6`, donor metric 0.413315, recipient metric 0.418649, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_9e-6` -> `lr_6e-6`, donor metric 0.413315, recipient metric 0.413719, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_9e-6` -> `lr_9e-6`, donor metric 0.413315, recipient metric 0.413315, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_9e-6` -> `lr_14e-6`, donor metric 0.413315, recipient metric 0.41296, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417835, recipient metric 0.409547, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417835, recipient metric 0.404591, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417835, recipient metric 0.417835, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417835, recipient metric 0.412584, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_9e-6` -> `lr_3e-6`, donor metric 0.410506, recipient metric 0.410733, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_9e-6` -> `lr_6e-6`, donor metric 0.410506, recipient metric 0.421894, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_9e-6` -> `lr_9e-6`, donor metric 0.410506, recipient metric 0.410506, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_9e-6` -> `lr_14e-6`, donor metric 0.410506, recipient metric 0.426169, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_9e-6` -> `lr_3e-6`, donor metric 0.408146, recipient metric 0.413333, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_9e-6` -> `lr_6e-6`, donor metric 0.408146, recipient metric 0.411079, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_9e-6` -> `lr_9e-6`, donor metric 0.408146, recipient metric 0.408146, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_9e-6` -> `lr_14e-6`, donor metric 0.408146, recipient metric 0.409193, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417374, recipient metric 0.409871, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417374, recipient metric 0.41119, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417374, recipient metric 0.417374, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417374, recipient metric 0.413372, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_9e-6` -> `lr_3e-6`, donor metric 0.40875, recipient metric 0.409658, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_9e-6` -> `lr_6e-6`, donor metric 0.40875, recipient metric 0.409508, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_9e-6` -> `lr_9e-6`, donor metric 0.40875, recipient metric 0.40875, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_9e-6` -> `lr_14e-6`, donor metric 0.40875, recipient metric 0.424036, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_9e-6` -> `lr_3e-6`, donor metric 0.413551, recipient metric 0.426495, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_9e-6` -> `lr_6e-6`, donor metric 0.413551, recipient metric 0.419675, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_9e-6` -> `lr_9e-6`, donor metric 0.413551, recipient metric 0.413551, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_9e-6` -> `lr_14e-6`, donor metric 0.413551, recipient metric 0.426646, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_9e-6` -> `lr_3e-6`, donor metric 0.427438, recipient metric 0.427412, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_9e-6` -> `lr_6e-6`, donor metric 0.427438, recipient metric 0.422126, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_9e-6` -> `lr_9e-6`, donor metric 0.427438, recipient metric 0.427438, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_9e-6` -> `lr_14e-6`, donor metric 0.427438, recipient metric 0.415775, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_3e-6`, donor metric 0.413265, recipient metric 0.40885, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_6e-6`, donor metric 0.413265, recipient metric 0.411384, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_9e-6`, donor metric 0.413265, recipient metric 0.413265, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_9e-6` -> `lr_14e-6`, donor metric 0.413265, recipient metric 0.408115, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_3e-6`, donor metric 0.413387, recipient metric 0.420871, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_6e-6`, donor metric 0.413387, recipient metric 0.422109, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_9e-6`, donor metric 0.413387, recipient metric 0.413387, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_9e-6` -> `lr_14e-6`, donor metric 0.413387, recipient metric 0.410659, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_3e-6`, donor metric 0.409236, recipient metric 0.41122, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_6e-6`, donor metric 0.409236, recipient metric 0.41149, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_9e-6`, donor metric 0.409236, recipient metric 0.409236, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_9e-6` -> `lr_14e-6`, donor metric 0.409236, recipient metric 0.408294, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_3e-6`, donor metric 0.397628, recipient metric 0.397514, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_6e-6`, donor metric 0.397628, recipient metric 0.42084, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_9e-6`, donor metric 0.397628, recipient metric 0.397628, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_9e-6` -> `lr_14e-6`, donor metric 0.397628, recipient metric 0.409341, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_3e-6`, donor metric 0.410935, recipient metric 0.411069, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_6e-6`, donor metric 0.410935, recipient metric 0.410364, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_9e-6`, donor metric 0.410935, recipient metric 0.410935, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_9e-6` -> `lr_14e-6`, donor metric 0.410935, recipient metric 0.406881, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_3e-6`, donor metric 0.401581, recipient metric 0.412948, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_6e-6`, donor metric 0.401581, recipient metric 0.418937, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_9e-6`, donor metric 0.401581, recipient metric 0.401581, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_9e-6` -> `lr_14e-6`, donor metric 0.401581, recipient metric 0.409332, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417084, recipient metric 0.413904, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417084, recipient metric 0.414868, LR 9.45e-06 -> 9.45e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417084, recipient metric 0.417084, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417084, recipient metric 0.408506, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_14e-6` -> `lr_3e-6`, donor metric 0.386866, recipient metric 0.390082, LR 8.4e-06 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_14e-6` -> `lr_6e-6`, donor metric 0.386866, recipient metric 0.416534, LR 9.45e-06 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_14e-6` -> `lr_9e-6`, donor metric 0.386866, recipient metric 0.420563, LR 1.05e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_14e-6` -> `lr_14e-6`, donor metric 0.386866, recipient metric 0.386866, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_14e-6` -> `lr_3e-6`, donor metric 0.403966, recipient metric 0.416907, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_14e-6` -> `lr_6e-6`, donor metric 0.403966, recipient metric 0.399023, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_14e-6` -> `lr_9e-6`, donor metric 0.403966, recipient metric 0.399175, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_14e-6` -> `lr_14e-6`, donor metric 0.403966, recipient metric 0.403966, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_14e-6` -> `lr_3e-6`, donor metric 0.413207, recipient metric 0.40531, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_14e-6` -> `lr_6e-6`, donor metric 0.413207, recipient metric 0.400124, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_14e-6` -> `lr_9e-6`, donor metric 0.413207, recipient metric 0.40842, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_14e-6` -> `lr_14e-6`, donor metric 0.413207, recipient metric 0.413207, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_14e-6` -> `lr_3e-6`, donor metric 0.410103, recipient metric 0.409336, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_14e-6` -> `lr_6e-6`, donor metric 0.410103, recipient metric 0.411729, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_14e-6` -> `lr_9e-6`, donor metric 0.410103, recipient metric 0.409536, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_14e-6` -> `lr_14e-6`, donor metric 0.410103, recipient metric 0.410103, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415793, recipient metric 0.410739, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415793, recipient metric 0.411365, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415793, recipient metric 0.388181, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415793, recipient metric 0.415793, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_14e-6` -> `lr_3e-6`, donor metric 0.406117, recipient metric 0.400385, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_14e-6` -> `lr_6e-6`, donor metric 0.406117, recipient metric 0.398626, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_14e-6` -> `lr_9e-6`, donor metric 0.406117, recipient metric 0.399526, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_14e-6` -> `lr_14e-6`, donor metric 0.406117, recipient metric 0.406117, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411057, recipient metric 0.408062, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411057, recipient metric 0.410129, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411057, recipient metric 0.407962, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411057, recipient metric 0.411057, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409477, recipient metric 0.409923, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409477, recipient metric 0.409222, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409477, recipient metric 0.416736, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409477, recipient metric 0.409477, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_14e-6` -> `lr_3e-6`, donor metric 0.416174, recipient metric 0.411057, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_14e-6` -> `lr_6e-6`, donor metric 0.416174, recipient metric 0.400974, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_14e-6` -> `lr_9e-6`, donor metric 0.416174, recipient metric 0.402533, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_14e-6` -> `lr_14e-6`, donor metric 0.416174, recipient metric 0.416174, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_3e-6`, donor metric 0.407053, recipient metric 0.410009, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_6e-6`, donor metric 0.407053, recipient metric 0.406897, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_9e-6`, donor metric 0.407053, recipient metric 0.424244, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_14e-6` -> `lr_14e-6`, donor metric 0.407053, recipient metric 0.407053, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411492, recipient metric 0.41183, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411492, recipient metric 0.411197, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411492, recipient metric 0.410271, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411492, recipient metric 0.411492, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_14e-6` -> `lr_3e-6`, donor metric 0.410131, recipient metric 0.402863, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_14e-6` -> `lr_6e-6`, donor metric 0.410131, recipient metric 0.400348, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_14e-6` -> `lr_9e-6`, donor metric 0.410131, recipient metric 0.400379, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_14e-6` -> `lr_14e-6`, donor metric 0.410131, recipient metric 0.410131, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_14e-6` -> `lr_3e-6`, donor metric 0.405574, recipient metric 0.409048, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_14e-6` -> `lr_6e-6`, donor metric 0.405574, recipient metric 0.408223, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_14e-6` -> `lr_9e-6`, donor metric 0.405574, recipient metric 0.407299, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_14e-6` -> `lr_14e-6`, donor metric 0.405574, recipient metric 0.405574, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_14e-6` -> `lr_3e-6`, donor metric 0.410924, recipient metric 0.408045, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_14e-6` -> `lr_6e-6`, donor metric 0.410924, recipient metric 0.416687, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_14e-6` -> `lr_9e-6`, donor metric 0.410924, recipient metric 0.411474, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_14e-6` -> `lr_14e-6`, donor metric 0.410924, recipient metric 0.410924, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_14e-6` -> `lr_3e-6`, donor metric 0.402252, recipient metric 0.4074, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_14e-6` -> `lr_6e-6`, donor metric 0.402252, recipient metric 0.411414, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_14e-6` -> `lr_9e-6`, donor metric 0.402252, recipient metric 0.421111, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_14e-6` -> `lr_14e-6`, donor metric 0.402252, recipient metric 0.402252, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_14e-6` -> `lr_3e-6`, donor metric 0.419462, recipient metric 0.410348, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_14e-6` -> `lr_6e-6`, donor metric 0.419462, recipient metric 0.406128, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_14e-6` -> `lr_9e-6`, donor metric 0.419462, recipient metric 0.408028, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_14e-6` -> `lr_14e-6`, donor metric 0.419462, recipient metric 0.419462, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_14e-6` -> `lr_3e-6`, donor metric 0.414325, recipient metric 0.416478, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_14e-6` -> `lr_6e-6`, donor metric 0.414325, recipient metric 0.409963, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_14e-6` -> `lr_9e-6`, donor metric 0.414325, recipient metric 0.413764, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_14e-6` -> `lr_14e-6`, donor metric 0.414325, recipient metric 0.414325, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409309, recipient metric 0.419426, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409309, recipient metric 0.409051, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409309, recipient metric 0.406265, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409309, recipient metric 0.409309, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_14e-6` -> `lr_3e-6`, donor metric 0.41314, recipient metric 0.41245, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_14e-6` -> `lr_6e-6`, donor metric 0.41314, recipient metric 0.410838, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_14e-6` -> `lr_9e-6`, donor metric 0.41314, recipient metric 0.413438, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_14e-6` -> `lr_14e-6`, donor metric 0.41314, recipient metric 0.41314, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_3e-6`, donor metric 0.428023, recipient metric 0.420385, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_6e-6`, donor metric 0.428023, recipient metric 0.42368, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_9e-6`, donor metric 0.428023, recipient metric 0.428963, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_14e-6` -> `lr_14e-6`, donor metric 0.428023, recipient metric 0.428023, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_14e-6` -> `lr_3e-6`, donor metric 0.405021, recipient metric 0.409715, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_14e-6` -> `lr_6e-6`, donor metric 0.405021, recipient metric 0.416026, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_14e-6` -> `lr_9e-6`, donor metric 0.405021, recipient metric 0.40237, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_14e-6` -> `lr_14e-6`, donor metric 0.405021, recipient metric 0.405021, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_14e-6` -> `lr_3e-6`, donor metric 0.397769, recipient metric 0.417101, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_14e-6` -> `lr_6e-6`, donor metric 0.397769, recipient metric 0.415124, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_14e-6` -> `lr_9e-6`, donor metric 0.397769, recipient metric 0.414917, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_14e-6` -> `lr_14e-6`, donor metric 0.397769, recipient metric 0.397769, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_14e-6` -> `lr_3e-6`, donor metric 0.413264, recipient metric 0.411568, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_14e-6` -> `lr_6e-6`, donor metric 0.413264, recipient metric 0.405277, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_14e-6` -> `lr_9e-6`, donor metric 0.413264, recipient metric 0.408204, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_14e-6` -> `lr_14e-6`, donor metric 0.413264, recipient metric 0.413264, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411226, recipient metric 0.423616, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411226, recipient metric 0.417018, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411226, recipient metric 0.407691, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411226, recipient metric 0.411226, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408176, recipient metric 0.404207, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_14e-6` -> `lr_6e-6`, donor metric 0.408176, recipient metric 0.408131, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_14e-6` -> `lr_9e-6`, donor metric 0.408176, recipient metric 0.40526, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408176, recipient metric 0.408176, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411684, recipient metric 0.415817, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411684, recipient metric 0.415687, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411684, recipient metric 0.406247, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411684, recipient metric 0.411684, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_14e-6` -> `lr_3e-6`, donor metric 0.406511, recipient metric 0.414462, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_14e-6` -> `lr_6e-6`, donor metric 0.406511, recipient metric 0.414252, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_14e-6` -> `lr_9e-6`, donor metric 0.406511, recipient metric 0.411871, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_14e-6` -> `lr_14e-6`, donor metric 0.406511, recipient metric 0.406511, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_14e-6` -> `lr_3e-6`, donor metric 0.397377, recipient metric 0.406323, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_14e-6` -> `lr_6e-6`, donor metric 0.397377, recipient metric 0.406238, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_14e-6` -> `lr_9e-6`, donor metric 0.397377, recipient metric 0.409444, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_14e-6` -> `lr_14e-6`, donor metric 0.397377, recipient metric 0.397377, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417847, recipient metric 0.424424, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_14e-6` -> `lr_6e-6`, donor metric 0.417847, recipient metric 0.396212, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_14e-6` -> `lr_9e-6`, donor metric 0.417847, recipient metric 0.419498, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417847, recipient metric 0.417847, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_14e-6` -> `lr_3e-6`, donor metric 0.404947, recipient metric 0.41405, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_14e-6` -> `lr_6e-6`, donor metric 0.404947, recipient metric 0.412388, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_14e-6` -> `lr_9e-6`, donor metric 0.404947, recipient metric 0.418338, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_14e-6` -> `lr_14e-6`, donor metric 0.404947, recipient metric 0.404947, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_14e-6` -> `lr_3e-6`, donor metric 0.4126, recipient metric 0.412118, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_14e-6` -> `lr_6e-6`, donor metric 0.4126, recipient metric 0.410546, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_14e-6` -> `lr_9e-6`, donor metric 0.4126, recipient metric 0.414626, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_14e-6` -> `lr_14e-6`, donor metric 0.4126, recipient metric 0.4126, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_14e-6` -> `lr_3e-6`, donor metric 0.414087, recipient metric 0.412874, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_14e-6` -> `lr_6e-6`, donor metric 0.414087, recipient metric 0.414545, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_14e-6` -> `lr_9e-6`, donor metric 0.414087, recipient metric 0.411688, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_14e-6` -> `lr_14e-6`, donor metric 0.414087, recipient metric 0.414087, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_14e-6` -> `lr_3e-6`, donor metric 0.421095, recipient metric 0.417067, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_14e-6` -> `lr_6e-6`, donor metric 0.421095, recipient metric 0.417712, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_14e-6` -> `lr_9e-6`, donor metric 0.421095, recipient metric 0.408813, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_14e-6` -> `lr_14e-6`, donor metric 0.421095, recipient metric 0.421095, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409382, recipient metric 0.408797, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409382, recipient metric 0.412, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409382, recipient metric 0.410718, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409382, recipient metric 0.409382, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_14e-6` -> `lr_3e-6`, donor metric 0.398576, recipient metric 0.401427, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_14e-6` -> `lr_6e-6`, donor metric 0.398576, recipient metric 0.412026, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_14e-6` -> `lr_9e-6`, donor metric 0.398576, recipient metric 0.410061, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_14e-6` -> `lr_14e-6`, donor metric 0.398576, recipient metric 0.398576, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409495, recipient metric 0.413154, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409495, recipient metric 0.413013, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409495, recipient metric 0.410912, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409495, recipient metric 0.409495, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_14e-6` -> `lr_3e-6`, donor metric 0.400174, recipient metric 0.404926, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_14e-6` -> `lr_6e-6`, donor metric 0.400174, recipient metric 0.407344, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_14e-6` -> `lr_9e-6`, donor metric 0.400174, recipient metric 0.407285, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_14e-6` -> `lr_14e-6`, donor metric 0.400174, recipient metric 0.400174, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_14e-6` -> `lr_3e-6`, donor metric 0.406554, recipient metric 0.412324, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_14e-6` -> `lr_6e-6`, donor metric 0.406554, recipient metric 0.406452, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_14e-6` -> `lr_9e-6`, donor metric 0.406554, recipient metric 0.410612, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_14e-6` -> `lr_14e-6`, donor metric 0.406554, recipient metric 0.406554, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_14e-6` -> `lr_3e-6`, donor metric 0.412966, recipient metric 0.406345, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_14e-6` -> `lr_6e-6`, donor metric 0.412966, recipient metric 0.401996, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_14e-6` -> `lr_9e-6`, donor metric 0.412966, recipient metric 0.407607, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_14e-6` -> `lr_14e-6`, donor metric 0.412966, recipient metric 0.412966, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
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
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_100gen_seed4.yaml', '--slots', 'iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3', '--experiment-name', 'anchor_copy_lr_recenter_100gen_seed4_20260819_111523']`
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
