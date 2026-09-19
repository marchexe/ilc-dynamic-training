# anchor_copy_lr_recenter_100gen_seed3_20260819_111523

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
- Final checkpoint controller objective: 1.0003 by `lr_3e-6`
- Global best configured metric: 0.398004 by `lr_9e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_100gen_seed3_20260819_111523/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_9e-6`, generation 71), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_9e-6`, generation 60) -- these are two distinct selection criteria, not the same checkpoint.
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=400, Pearson r=0.013 (95% CI -0.058 to 0.081), Spearman rho=0.014 (95% CI -0.058 to 0.079)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=0.638, Spearman rho=0.633
- Best checkpoint by tier: control: `lr_3e-6` gen 99 (0.410223), full_holdout: `lr_14e-6` gen 79 (0.397867)
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
| lr_14e-6 | 0.1763 | 0.07006 | 3.158 | 0.1942 | 0.5197 | 0.06806 | 2.671 | 1.139 | 0.5728 | 0.295 | 0.4111 | 6.77e-06 | - |
| lr_3e-6 | 0.1763 | 0.06806 | 3.158 | 0.1962 | 0.5197 | 0.06806 | 2.677 | 1.139 | 0.5731 | 0.2936 | 0.4102 | 4.51e-06 | winner |
| lr_6e-6 | 0.1602 | 0.07642 | 3.216 | 0.2011 | 0.5378 | 0.06636 | 2.717 | 1.132 | 0.5756 | 0.2983 | 0.4144 | 5.08e-06 | - |
| lr_9e-6 | 0.1732 | 0.07005 | 3.213 | 0.1961 | 0.5293 | 0.06805 | 2.683 | 1.143 | 0.5765 | 0.2957 | 0.4129 | 5.64e-06 | anchor |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_14e-6 | 0.4438 | 0.6199 | 0.3177 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 1 | lr_3e-6 | 0.4418 | 0.6045 | 0.323 | 1.12e-05 | 1.4e-05 | 1.12e-05 | accepted_new_anchor | no |
| 2 | lr_9e-6 | 0.4438 | 0.5924 | 0.3324 | 1.12e-05 | 1.12e-05 | 1.12e-05 | rewound_to_previous_anchor | no |
| 3 | lr_6e-6 | 0.4264 | 0.5816 | 0.3127 | 1.008e-05 | 1.12e-05 | 1.008e-05 | accepted_new_anchor | no |
| 4 | lr_6e-6 | 0.4359 | 0.6081 | 0.3125 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 5 | lr_3e-6 | 0.4349 | 0.6044 | 0.3129 | 8.064e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 6 | lr_3e-6 | 0.4325 | 0.5972 | 0.3132 | 8.064e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 7 | lr_9e-6 | 0.4322 | 0.6006 | 0.311 | 1.008e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 8 | lr_6e-6 | 0.4355 | 0.6052 | 0.3135 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 9 | lr_9e-6 | 0.4281 | 0.5989 | 0.306 | 1.008e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 10 | lr_6e-6 | 0.426 | 0.5897 | 0.3077 | 9.072e-06 | 1.008e-05 | 9.072e-06 | accepted_new_anchor | no |
| 11 | lr_14e-6 | 0.4195 | 0.5787 | 0.3041 | 1.089e-05 | 9.072e-06 | 1.089e-05 | accepted_new_anchor | no |
| 12 | lr_3e-6 | 0.4134 | 0.5711 | 0.2992 | 8.709e-06 | 1.089e-05 | 8.709e-06 | accepted_new_anchor | no |
| 13 | lr_14e-6 | 0.4237 | 0.582 | 0.3085 | 1.045e-05 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 14 | lr_9e-6 | 0.4226 | 0.572 | 0.3122 | 8.709e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 15 | lr_3e-6 | 0.4177 | 0.5738 | 0.304 | 6.967e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 16 | lr_14e-6 | 0.4255 | 0.5795 | 0.3125 | 1.045e-05 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 17 | lr_14e-6 | 0.4148 | 0.5691 | 0.3023 | 1.045e-05 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 18 | lr_14e-6 | 0.417 | 0.5644 | 0.308 | 1.045e-05 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 19 | lr_3e-6 | 0.4178 | 0.58 | 0.301 | 6.967e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 20 | lr_14e-6 | 0.4168 | 0.5791 | 0.2999 | 1.045e-05 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 21 | lr_6e-6 | 0.4222 | 0.5742 | 0.3105 | 7.838e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 22 | lr_6e-6 | 0.4149 | 0.563 | 0.3058 | 7.838e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 23 | lr_6e-6 | 0.4208 | 0.5694 | 0.311 | 7.838e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 24 | lr_9e-6 | 0.4181 | 0.5764 | 0.3033 | 8.709e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 25 | lr_9e-6 | 0.4141 | 0.5768 | 0.2972 | 8.709e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 26 | lr_6e-6 | 0.4192 | 0.5856 | 0.3 | 7.838e-06 | 8.709e-06 | 8.709e-06 | rewound_to_previous_anchor | no |
| 27 | lr_6e-6 | 0.4127 | 0.5756 | 0.2959 | 7.838e-06 | 8.709e-06 | 7.838e-06 | accepted_new_anchor | no |
| 28 | lr_6e-6 | 0.4236 | 0.5795 | 0.3097 | 7.054e-06 | 7.838e-06 | 7.838e-06 | rewound_to_previous_anchor | no |
| 29 | lr_3e-6 | 0.4077 | 0.574 | 0.2896 | 6.271e-06 | 7.838e-06 | 6.271e-06 | accepted_new_anchor | no |
| 30 | lr_14e-6 | 0.4131 | 0.5734 | 0.2976 | 7.525e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 31 | lr_6e-6 | 0.4216 | 0.5832 | 0.3048 | 5.644e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 32 | lr_14e-6 | 0.4194 | 0.5834 | 0.3015 | 7.525e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 33 | lr_6e-6 | 0.4104 | 0.5631 | 0.2992 | 5.644e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 34 | lr_14e-6 | 0.4195 | 0.5706 | 0.3084 | 7.525e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 35 | lr_6e-6 | 0.4195 | 0.5653 | 0.3114 | 5.644e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 36 | lr_14e-6 | 0.4184 | 0.5892 | 0.2972 | 7.525e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 37 | lr_3e-6 | 0.4196 | 0.5657 | 0.3113 | 5.016e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 38 | lr_9e-6 | 0.4132 | 0.5713 | 0.2989 | 6.271e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 39 | lr_6e-6 | 0.4188 | 0.5775 | 0.3037 | 5.644e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 40 | lr_6e-6 | 0.4175 | 0.5756 | 0.3028 | 5.644e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 41 | lr_9e-6 | 0.4186 | 0.5803 | 0.302 | 6.271e-06 | 6.271e-06 | 6.271e-06 | rewound_to_previous_anchor | no |
| 42 | lr_6e-6 | 0.4036 | 0.5496 | 0.2964 | 5.644e-06 | 6.271e-06 | 5.644e-06 | accepted_new_anchor | no |
| 43 | lr_6e-6 | 0.4048 | 0.5574 | 0.2939 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 44 | lr_14e-6 | 0.4187 | 0.5748 | 0.305 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 45 | lr_9e-6 | 0.4149 | 0.5762 | 0.2987 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 46 | lr_9e-6 | 0.4146 | 0.5673 | 0.3029 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 47 | lr_9e-6 | 0.4133 | 0.5703 | 0.2995 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 48 | lr_9e-6 | 0.4277 | 0.5879 | 0.3112 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 49 | lr_6e-6 | 0.4076 | 0.5565 | 0.2985 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 50 | lr_9e-6 | 0.4179 | 0.5687 | 0.3071 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 51 | lr_6e-6 | 0.4065 | 0.571 | 0.2894 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 52 | lr_3e-6 | 0.4096 | 0.5671 | 0.2959 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 53 | lr_9e-6 | 0.4173 | 0.5708 | 0.3051 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 54 | lr_3e-6 | 0.4099 | 0.588 | 0.2858 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 55 | lr_14e-6 | 0.4156 | 0.5757 | 0.3001 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 56 | lr_14e-6 | 0.415 | 0.58 | 0.297 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 57 | lr_3e-6 | 0.412 | 0.5505 | 0.3084 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 58 | lr_6e-6 | 0.4182 | 0.5914 | 0.2957 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 59 | lr_9e-6 | 0.4137 | 0.5651 | 0.3028 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 60 | lr_9e-6 | 0.409 | 0.5553 | 0.3012 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 61 | lr_6e-6 | 0.4207 | 0.5714 | 0.3098 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 62 | lr_6e-6 | 0.4251 | 0.5743 | 0.3147 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 63 | lr_9e-6 | 0.4122 | 0.5704 | 0.2978 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 64 | lr_9e-6 | 0.4213 | 0.5809 | 0.3056 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 65 | lr_3e-6 | 0.4248 | 0.5865 | 0.3077 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 66 | lr_9e-6 | 0.4165 | 0.5696 | 0.3046 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 67 | lr_3e-6 | 0.4101 | 0.5721 | 0.2939 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 68 | lr_6e-6 | 0.4226 | 0.5769 | 0.3095 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 69 | lr_14e-6 | 0.4153 | 0.5825 | 0.2961 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 70 | lr_14e-6 | 0.4195 | 0.568 | 0.3098 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 71 | lr_9e-6 | 0.398 | 0.5653 | 0.2802 | 5.644e-06 | 5.644e-06 | 5.644e-06 | accepted_new_anchor | no |
| 72 | lr_3e-6 | 0.4109 | 0.5658 | 0.2984 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 73 | lr_6e-6 | 0.4168 | 0.571 | 0.3042 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 74 | lr_6e-6 | 0.414 | 0.5607 | 0.3056 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 75 | lr_9e-6 | 0.4201 | 0.5789 | 0.3048 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 76 | lr_6e-6 | 0.4131 | 0.5758 | 0.2963 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 77 | lr_6e-6 | 0.4079 | 0.5573 | 0.2985 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 78 | lr_6e-6 | 0.4182 | 0.5618 | 0.3113 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 79 | lr_3e-6 | 0.4126 | 0.5678 | 0.2998 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 80 | lr_6e-6 | 0.4219 | 0.568 | 0.3134 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 81 | lr_14e-6 | 0.4084 | 0.5776 | 0.2887 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 82 | lr_9e-6 | 0.4121 | 0.564 | 0.3012 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 83 | lr_9e-6 | 0.4193 | 0.5694 | 0.3088 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 84 | lr_6e-6 | 0.4122 | 0.5735 | 0.2963 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 85 | lr_9e-6 | 0.4016 | 0.5476 | 0.2945 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 86 | lr_3e-6 | 0.4045 | 0.5659 | 0.2891 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 87 | lr_14e-6 | 0.4117 | 0.5593 | 0.303 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 88 | lr_14e-6 | 0.4131 | 0.5724 | 0.2981 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 89 | lr_6e-6 | 0.418 | 0.5771 | 0.3027 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 90 | lr_3e-6 | 0.4078 | 0.5587 | 0.2977 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 91 | lr_9e-6 | 0.4123 | 0.5605 | 0.3033 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 92 | lr_14e-6 | 0.4149 | 0.5849 | 0.2942 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 93 | lr_14e-6 | 0.4223 | 0.5721 | 0.3117 | 6.772e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 94 | lr_3e-6 | 0.4233 | 0.5724 | 0.313 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 95 | lr_3e-6 | 0.4171 | 0.5873 | 0.2962 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 96 | lr_6e-6 | 0.4079 | 0.5722 | 0.2907 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 97 | lr_6e-6 | 0.4167 | 0.5651 | 0.3072 | 5.079e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 98 | lr_9e-6 | 0.4164 | 0.5681 | 0.3052 | 5.644e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |
| 99 | lr_3e-6 | 0.4102 | 0.5731 | 0.2936 | 4.515e-06 | 5.644e-06 | 5.644e-06 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_14e-6` -> `lr_3e-6`, donor metric 0.443798, recipient metric 0.45186, LR 3e-06 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_6e-6`, donor metric 0.443798, recipient metric 0.451943, LR 6e-06 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_9e-6`, donor metric 0.443798, recipient metric 0.445661, LR 9e-06 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_14e-6`, donor metric 0.443798, recipient metric 0.443798, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_3e-6`, donor metric 0.441849, recipient metric 0.441849, LR 1.12e-05 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_6e-6`, donor metric 0.441849, recipient metric 0.450138, LR 1.26e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_9e-6`, donor metric 0.441849, recipient metric 0.446154, LR 1.4e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_14e-6`, donor metric 0.441849, recipient metric 0.446154, LR 1.4e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_3e-6`, donor metric 0.443879, recipient metric 0.443879, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_6e-6`, donor metric 0.443879, recipient metric 0.449649, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_9e-6`, donor metric 0.443879, recipient metric 0.443753, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_14e-6`, donor metric 0.443879, recipient metric 0.446374, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_3e-6`, donor metric 0.426448, recipient metric 0.445898, LR 8.96e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_6e-6`, donor metric 0.426448, recipient metric 0.426448, LR 1.01e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_9e-6`, donor metric 0.426448, recipient metric 0.430251, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_14e-6`, donor metric 0.426448, recipient metric 0.427916, LR 1.34e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_3e-6`, donor metric 0.435891, recipient metric 0.440233, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_6e-6`, donor metric 0.435891, recipient metric 0.435891, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_9e-6`, donor metric 0.435891, recipient metric 0.437447, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_14e-6`, donor metric 0.435891, recipient metric 0.437718, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_3e-6`, donor metric 0.440611, recipient metric 0.434913, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_6e-6`, donor metric 0.440611, recipient metric 0.440611, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_9e-6`, donor metric 0.440611, recipient metric 0.435994, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_14e-6`, donor metric 0.440611, recipient metric 0.434999, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_3e-6`, donor metric 0.439569, recipient metric 0.432505, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_6e-6`, donor metric 0.439569, recipient metric 0.439569, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_9e-6`, donor metric 0.439569, recipient metric 0.441378, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_14e-6`, donor metric 0.439569, recipient metric 0.44041, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_3e-6`, donor metric 0.434829, recipient metric 0.432874, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_6e-6`, donor metric 0.434829, recipient metric 0.434829, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_9e-6`, donor metric 0.434829, recipient metric 0.432232, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_14e-6`, donor metric 0.434829, recipient metric 0.437944, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_3e-6`, donor metric 0.435542, recipient metric 0.450172, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_6e-6`, donor metric 0.435542, recipient metric 0.435542, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_9e-6`, donor metric 0.435542, recipient metric 0.443662, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_14e-6`, donor metric 0.435542, recipient metric 0.442959, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_3e-6`, donor metric 0.429355, recipient metric 0.43881, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_6e-6`, donor metric 0.429355, recipient metric 0.429355, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_9e-6`, donor metric 0.429355, recipient metric 0.428111, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_14e-6`, donor metric 0.429355, recipient metric 0.439142, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_3e-6`, donor metric 0.425955, recipient metric 0.443131, LR 8.06e-06 -> 7.26e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_6e-6`, donor metric 0.425955, recipient metric 0.425955, LR 9.07e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_9e-6`, donor metric 0.425955, recipient metric 0.441945, LR 1.01e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_14e-6`, donor metric 0.425955, recipient metric 0.442031, LR 1.21e-05 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_3e-6`, donor metric 0.419535, recipient metric 0.427149, LR 7.26e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_6e-6`, donor metric 0.419535, recipient metric 0.42865, LR 8.16e-06 -> 9.8e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_9e-6`, donor metric 0.419535, recipient metric 0.43383, LR 9.07e-06 -> 1.09e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_14e-6`, donor metric 0.419535, recipient metric 0.419535, LR 1.09e-05 -> 1.31e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_3e-6`, donor metric 0.413366, recipient metric 0.413366, LR 8.71e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_6e-6`, donor metric 0.413366, recipient metric 0.424643, LR 9.8e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_9e-6`, donor metric 0.413366, recipient metric 0.429212, LR 1.09e-05 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_14e-6`, donor metric 0.413366, recipient metric 0.429421, LR 1.31e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_3e-6`, donor metric 0.429075, recipient metric 0.429075, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_6e-6`, donor metric 0.429075, recipient metric 0.42712, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_9e-6`, donor metric 0.429075, recipient metric 0.42534, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_14e-6`, donor metric 0.429075, recipient metric 0.423732, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_3e-6` -> `lr_3e-6`, donor metric 0.427905, recipient metric 0.427905, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_3e-6` -> `lr_6e-6`, donor metric 0.427905, recipient metric 0.433966, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_3e-6` -> `lr_9e-6`, donor metric 0.427905, recipient metric 0.422557, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_3e-6` -> `lr_14e-6`, donor metric 0.427905, recipient metric 0.434667, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_3e-6` -> `lr_3e-6`, donor metric 0.417682, recipient metric 0.417682, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_3e-6` -> `lr_6e-6`, donor metric 0.417682, recipient metric 0.422312, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_3e-6` -> `lr_9e-6`, donor metric 0.417682, recipient metric 0.432698, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_3e-6` -> `lr_14e-6`, donor metric 0.417682, recipient metric 0.433515, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_3e-6` -> `lr_3e-6`, donor metric 0.427787, recipient metric 0.427787, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_3e-6` -> `lr_6e-6`, donor metric 0.427787, recipient metric 0.427709, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_3e-6` -> `lr_9e-6`, donor metric 0.427787, recipient metric 0.428874, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_3e-6` -> `lr_14e-6`, donor metric 0.427787, recipient metric 0.425535, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_3e-6` -> `lr_3e-6`, donor metric 0.41811, recipient metric 0.41811, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_3e-6` -> `lr_6e-6`, donor metric 0.41811, recipient metric 0.422501, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_3e-6` -> `lr_9e-6`, donor metric 0.41811, recipient metric 0.423266, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_3e-6` -> `lr_14e-6`, donor metric 0.41811, recipient metric 0.414766, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_3e-6` -> `lr_3e-6`, donor metric 0.418722, recipient metric 0.418722, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_3e-6` -> `lr_6e-6`, donor metric 0.418722, recipient metric 0.422235, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_3e-6` -> `lr_9e-6`, donor metric 0.418722, recipient metric 0.421822, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_3e-6` -> `lr_14e-6`, donor metric 0.418722, recipient metric 0.416952, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_3e-6` -> `lr_3e-6`, donor metric 0.417808, recipient metric 0.417808, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_3e-6` -> `lr_6e-6`, donor metric 0.417808, recipient metric 0.432639, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_3e-6` -> `lr_9e-6`, donor metric 0.417808, recipient metric 0.433202, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_3e-6` -> `lr_14e-6`, donor metric 0.417808, recipient metric 0.432145, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_3e-6` -> `lr_3e-6`, donor metric 0.428943, recipient metric 0.428943, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_3e-6` -> `lr_6e-6`, donor metric 0.428943, recipient metric 0.417897, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_3e-6` -> `lr_9e-6`, donor metric 0.428943, recipient metric 0.418059, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_3e-6` -> `lr_14e-6`, donor metric 0.428943, recipient metric 0.416786, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_3e-6` -> `lr_3e-6`, donor metric 0.432121, recipient metric 0.432121, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_3e-6` -> `lr_6e-6`, donor metric 0.432121, recipient metric 0.42223, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_3e-6` -> `lr_9e-6`, donor metric 0.432121, recipient metric 0.424777, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_3e-6` -> `lr_14e-6`, donor metric 0.432121, recipient metric 0.431238, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_3e-6` -> `lr_3e-6`, donor metric 0.426996, recipient metric 0.426996, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_3e-6` -> `lr_6e-6`, donor metric 0.426996, recipient metric 0.414947, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_3e-6` -> `lr_9e-6`, donor metric 0.426996, recipient metric 0.423957, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_3e-6` -> `lr_14e-6`, donor metric 0.426996, recipient metric 0.424965, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_3e-6` -> `lr_3e-6`, donor metric 0.424741, recipient metric 0.424741, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_3e-6` -> `lr_6e-6`, donor metric 0.424741, recipient metric 0.420783, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_3e-6` -> `lr_9e-6`, donor metric 0.424741, recipient metric 0.425589, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_3e-6` -> `lr_14e-6`, donor metric 0.424741, recipient metric 0.421482, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_3e-6` -> `lr_3e-6`, donor metric 0.418407, recipient metric 0.418407, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_3e-6` -> `lr_6e-6`, donor metric 0.418407, recipient metric 0.41933, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_3e-6` -> `lr_9e-6`, donor metric 0.418407, recipient metric 0.418076, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_3e-6` -> `lr_14e-6`, donor metric 0.418407, recipient metric 0.418173, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_3e-6` -> `lr_3e-6`, donor metric 0.423346, recipient metric 0.423346, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_3e-6` -> `lr_6e-6`, donor metric 0.423346, recipient metric 0.420754, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_3e-6` -> `lr_9e-6`, donor metric 0.423346, recipient metric 0.414058, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_3e-6` -> `lr_14e-6`, donor metric 0.423346, recipient metric 0.427919, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_3e-6` -> `lr_3e-6`, donor metric 0.429667, recipient metric 0.429667, LR 6.97e-06 -> 6.97e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_3e-6` -> `lr_6e-6`, donor metric 0.429667, recipient metric 0.419169, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_3e-6` -> `lr_9e-6`, donor metric 0.429667, recipient metric 0.423892, LR 8.71e-06 -> 8.71e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_3e-6` -> `lr_14e-6`, donor metric 0.429667, recipient metric 0.421606, LR 1.05e-05 -> 1.05e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_3e-6`, donor metric 0.412719, recipient metric 0.421823, LR 6.97e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_6e-6`, donor metric 0.412719, recipient metric 0.412719, LR 7.84e-06 -> 7.05e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_9e-6`, donor metric 0.412719, recipient metric 0.413118, LR 8.71e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_14e-6`, donor metric 0.412719, recipient metric 0.424368, LR 1.05e-05 -> 9.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_3e-6`, donor metric 0.423594, recipient metric 0.42624, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_6e-6`, donor metric 0.423594, recipient metric 0.423594, LR 7.05e-06 -> 7.05e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_9e-6`, donor metric 0.423594, recipient metric 0.426221, LR 7.84e-06 -> 7.84e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_14e-6`, donor metric 0.423594, recipient metric 0.426169, LR 9.41e-06 -> 9.41e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_3e-6`, donor metric 0.407717, recipient metric 0.407717, LR 6.27e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_6e-6`, donor metric 0.407717, recipient metric 0.408344, LR 7.05e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_9e-6`, donor metric 0.407717, recipient metric 0.413862, LR 7.84e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_3e-6` -> `lr_14e-6`, donor metric 0.407717, recipient metric 0.413537, LR 9.41e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_3e-6`, donor metric 0.421319, recipient metric 0.421319, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_6e-6`, donor metric 0.421319, recipient metric 0.421765, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_9e-6`, donor metric 0.421319, recipient metric 0.41369, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_3e-6` -> `lr_14e-6`, donor metric 0.421319, recipient metric 0.413128, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_3e-6`, donor metric 0.42182, recipient metric 0.42182, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_6e-6`, donor metric 0.42182, recipient metric 0.421636, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_9e-6`, donor metric 0.42182, recipient metric 0.421716, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_3e-6` -> `lr_14e-6`, donor metric 0.42182, recipient metric 0.424336, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_3e-6`, donor metric 0.423424, recipient metric 0.423424, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_6e-6`, donor metric 0.423424, recipient metric 0.425343, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_9e-6`, donor metric 0.423424, recipient metric 0.426237, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_3e-6` -> `lr_14e-6`, donor metric 0.423424, recipient metric 0.419415, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_3e-6`, donor metric 0.416492, recipient metric 0.416492, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_6e-6`, donor metric 0.416492, recipient metric 0.410429, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_9e-6`, donor metric 0.416492, recipient metric 0.412884, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_3e-6` -> `lr_14e-6`, donor metric 0.416492, recipient metric 0.426751, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_3e-6`, donor metric 0.423516, recipient metric 0.423516, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_6e-6`, donor metric 0.423516, recipient metric 0.424811, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_9e-6`, donor metric 0.423516, recipient metric 0.425471, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_3e-6` -> `lr_14e-6`, donor metric 0.423516, recipient metric 0.419494, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_3e-6`, donor metric 0.419615, recipient metric 0.419615, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_6e-6`, donor metric 0.419615, recipient metric 0.419544, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_9e-6`, donor metric 0.419615, recipient metric 0.431589, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_3e-6` -> `lr_14e-6`, donor metric 0.419615, recipient metric 0.429815, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_3e-6`, donor metric 0.422974, recipient metric 0.422974, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_6e-6`, donor metric 0.422974, recipient metric 0.428646, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_9e-6`, donor metric 0.422974, recipient metric 0.428771, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_3e-6` -> `lr_14e-6`, donor metric 0.422974, recipient metric 0.418435, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_3e-6`, donor metric 0.419633, recipient metric 0.419633, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_6e-6`, donor metric 0.419633, recipient metric 0.42377, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_9e-6`, donor metric 0.419633, recipient metric 0.433585, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_3e-6` -> `lr_14e-6`, donor metric 0.419633, recipient metric 0.421192, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_3e-6`, donor metric 0.415406, recipient metric 0.415406, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_6e-6`, donor metric 0.415406, recipient metric 0.41671, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_9e-6`, donor metric 0.415406, recipient metric 0.413243, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_3e-6` -> `lr_14e-6`, donor metric 0.415406, recipient metric 0.418923, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_3e-6`, donor metric 0.425632, recipient metric 0.425632, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_6e-6`, donor metric 0.425632, recipient metric 0.418756, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_9e-6`, donor metric 0.425632, recipient metric 0.429494, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_3e-6` -> `lr_14e-6`, donor metric 0.425632, recipient metric 0.429788, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_3e-6`, donor metric 0.419297, recipient metric 0.419297, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_6e-6`, donor metric 0.419297, recipient metric 0.417458, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_9e-6`, donor metric 0.419297, recipient metric 0.419555, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_3e-6` -> `lr_14e-6`, donor metric 0.419297, recipient metric 0.432395, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_3e-6`, donor metric 0.421968, recipient metric 0.421968, LR 5.02e-06 -> 5.02e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_6e-6`, donor metric 0.421968, recipient metric 0.419198, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_9e-6`, donor metric 0.421968, recipient metric 0.41862, LR 6.27e-06 -> 6.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_3e-6` -> `lr_14e-6`, donor metric 0.421968, recipient metric 0.422151, LR 7.52e-06 -> 7.52e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_3e-6`, donor metric 0.403634, recipient metric 0.418701, LR 5.02e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_6e-6`, donor metric 0.403634, recipient metric 0.403634, LR 5.64e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_9e-6`, donor metric 0.403634, recipient metric 0.418216, LR 6.27e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_14e-6`, donor metric 0.403634, recipient metric 0.412933, LR 7.52e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_3e-6`, donor metric 0.40476, recipient metric 0.426513, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_6e-6`, donor metric 0.40476, recipient metric 0.40476, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_9e-6`, donor metric 0.40476, recipient metric 0.404951, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_14e-6`, donor metric 0.40476, recipient metric 0.413732, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_3e-6`, donor metric 0.419391, recipient metric 0.41908, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_6e-6`, donor metric 0.419391, recipient metric 0.419391, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_9e-6`, donor metric 0.419391, recipient metric 0.419174, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_14e-6`, donor metric 0.419391, recipient metric 0.418732, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_3e-6`, donor metric 0.423565, recipient metric 0.426587, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_6e-6`, donor metric 0.423565, recipient metric 0.423565, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_9e-6`, donor metric 0.423565, recipient metric 0.414877, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_14e-6`, donor metric 0.423565, recipient metric 0.422855, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416718, recipient metric 0.416656, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416718, recipient metric 0.416718, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416718, recipient metric 0.414573, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416718, recipient metric 0.415713, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_3e-6`, donor metric 0.420507, recipient metric 0.421377, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_6e-6`, donor metric 0.420507, recipient metric 0.420507, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_9e-6`, donor metric 0.420507, recipient metric 0.413299, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_14e-6`, donor metric 0.420507, recipient metric 0.420039, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_3e-6`, donor metric 0.429741, recipient metric 0.429375, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_6e-6`, donor metric 0.429741, recipient metric 0.429741, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_9e-6`, donor metric 0.429741, recipient metric 0.427738, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_14e-6`, donor metric 0.429741, recipient metric 0.42989, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_3e-6`, donor metric 0.407596, recipient metric 0.418169, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_6e-6`, donor metric 0.407596, recipient metric 0.407596, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_9e-6`, donor metric 0.407596, recipient metric 0.425731, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_14e-6`, donor metric 0.407596, recipient metric 0.420512, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_3e-6`, donor metric 0.426562, recipient metric 0.419379, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_6e-6`, donor metric 0.426562, recipient metric 0.426562, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_9e-6`, donor metric 0.426562, recipient metric 0.417893, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_14e-6`, donor metric 0.426562, recipient metric 0.425895, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_3e-6`, donor metric 0.406496, recipient metric 0.406818, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_6e-6`, donor metric 0.406496, recipient metric 0.406496, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_9e-6`, donor metric 0.406496, recipient metric 0.426064, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_14e-6`, donor metric 0.406496, recipient metric 0.421252, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_3e-6`, donor metric 0.429208, recipient metric 0.40965, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_6e-6`, donor metric 0.429208, recipient metric 0.429208, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_9e-6`, donor metric 0.429208, recipient metric 0.410084, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_14e-6`, donor metric 0.429208, recipient metric 0.411437, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_3e-6`, donor metric 0.417484, recipient metric 0.418384, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_6e-6`, donor metric 0.417484, recipient metric 0.417484, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_9e-6`, donor metric 0.417484, recipient metric 0.41729, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_14e-6`, donor metric 0.417484, recipient metric 0.417985, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_3e-6`, donor metric 0.430844, recipient metric 0.409901, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_6e-6`, donor metric 0.430844, recipient metric 0.430844, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_9e-6`, donor metric 0.430844, recipient metric 0.416312, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_14e-6`, donor metric 0.430844, recipient metric 0.430126, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_3e-6`, donor metric 0.4164, recipient metric 0.416589, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_6e-6`, donor metric 0.4164, recipient metric 0.4164, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_9e-6`, donor metric 0.4164, recipient metric 0.428002, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_14e-6`, donor metric 0.4164, recipient metric 0.415645, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_3e-6`, donor metric 0.423773, recipient metric 0.415185, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_6e-6`, donor metric 0.423773, recipient metric 0.423773, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_9e-6`, donor metric 0.423773, recipient metric 0.426939, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_14e-6`, donor metric 0.423773, recipient metric 0.415047, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_3e-6`, donor metric 0.417384, recipient metric 0.412046, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_6e-6`, donor metric 0.417384, recipient metric 0.417384, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_9e-6`, donor metric 0.417384, recipient metric 0.413211, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_14e-6`, donor metric 0.417384, recipient metric 0.413086, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418204, recipient metric 0.427574, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418204, recipient metric 0.418204, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418204, recipient metric 0.430113, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418204, recipient metric 0.420225, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_3e-6`, donor metric 0.413683, recipient metric 0.422638, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_6e-6`, donor metric 0.413683, recipient metric 0.413683, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_9e-6`, donor metric 0.413683, recipient metric 0.413676, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_14e-6`, donor metric 0.413683, recipient metric 0.415811, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_3e-6`, donor metric 0.417399, recipient metric 0.41542, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_6e-6`, donor metric 0.417399, recipient metric 0.417399, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_9e-6`, donor metric 0.417399, recipient metric 0.408963, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_14e-6`, donor metric 0.417399, recipient metric 0.422037, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_3e-6`, donor metric 0.420746, recipient metric 0.425422, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_6e-6`, donor metric 0.420746, recipient metric 0.420746, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_9e-6`, donor metric 0.420746, recipient metric 0.422554, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_14e-6`, donor metric 0.420746, recipient metric 0.422383, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_3e-6`, donor metric 0.4251, recipient metric 0.426554, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_6e-6`, donor metric 0.4251, recipient metric 0.4251, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_9e-6`, donor metric 0.4251, recipient metric 0.429897, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_14e-6`, donor metric 0.4251, recipient metric 0.425597, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_3e-6`, donor metric 0.422673, recipient metric 0.418843, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_6e-6`, donor metric 0.422673, recipient metric 0.422673, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_9e-6`, donor metric 0.422673, recipient metric 0.412158, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_14e-6`, donor metric 0.422673, recipient metric 0.418086, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_3e-6`, donor metric 0.421667, recipient metric 0.423395, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_6e-6`, donor metric 0.421667, recipient metric 0.421667, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_9e-6`, donor metric 0.421667, recipient metric 0.421311, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_14e-6`, donor metric 0.421667, recipient metric 0.421922, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_3e-6`, donor metric 0.425486, recipient metric 0.424849, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_6e-6`, donor metric 0.425486, recipient metric 0.425486, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_9e-6`, donor metric 0.425486, recipient metric 0.428973, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_14e-6`, donor metric 0.425486, recipient metric 0.425281, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_3e-6`, donor metric 0.422721, recipient metric 0.42692, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_6e-6`, donor metric 0.422721, recipient metric 0.422721, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_9e-6`, donor metric 0.422721, recipient metric 0.41652, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_14e-6`, donor metric 0.422721, recipient metric 0.422289, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_3e-6`, donor metric 0.420204, recipient metric 0.410065, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_6e-6`, donor metric 0.420204, recipient metric 0.420204, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_9e-6`, donor metric 0.420204, recipient metric 0.411462, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_14e-6`, donor metric 0.420204, recipient metric 0.418689, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_3e-6`, donor metric 0.422578, recipient metric 0.427723, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_6e-6`, donor metric 0.422578, recipient metric 0.422578, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_9e-6`, donor metric 0.422578, recipient metric 0.424424, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_14e-6`, donor metric 0.422578, recipient metric 0.424305, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_3e-6`, donor metric 0.42594, recipient metric 0.415924, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_6e-6`, donor metric 0.42594, recipient metric 0.42594, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_9e-6`, donor metric 0.42594, recipient metric 0.415301, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_14e-6`, donor metric 0.42594, recipient metric 0.415269, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_3e-6`, donor metric 0.419948, recipient metric 0.422542, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_6e-6`, donor metric 0.419948, recipient metric 0.419948, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_9e-6`, donor metric 0.419948, recipient metric 0.424092, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_14e-6`, donor metric 0.419948, recipient metric 0.419467, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_9e-6` -> `lr_3e-6`, donor metric 0.398004, recipient metric 0.410037, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_9e-6` -> `lr_6e-6`, donor metric 0.398004, recipient metric 0.40739, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_9e-6` -> `lr_9e-6`, donor metric 0.398004, recipient metric 0.398004, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_9e-6` -> `lr_14e-6`, donor metric 0.398004, recipient metric 0.417245, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_9e-6` -> `lr_3e-6`, donor metric 0.422653, recipient metric 0.410882, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_9e-6` -> `lr_6e-6`, donor metric 0.422653, recipient metric 0.424241, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_9e-6` -> `lr_9e-6`, donor metric 0.422653, recipient metric 0.422653, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_9e-6` -> `lr_14e-6`, donor metric 0.422653, recipient metric 0.417788, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_9e-6` -> `lr_3e-6`, donor metric 0.416851, recipient metric 0.418136, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_9e-6` -> `lr_6e-6`, donor metric 0.416851, recipient metric 0.416773, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_9e-6` -> `lr_9e-6`, donor metric 0.416851, recipient metric 0.416851, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_9e-6` -> `lr_14e-6`, donor metric 0.416851, recipient metric 0.41897, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_9e-6` -> `lr_3e-6`, donor metric 0.428255, recipient metric 0.415113, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_9e-6` -> `lr_6e-6`, donor metric 0.428255, recipient metric 0.413966, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_9e-6` -> `lr_9e-6`, donor metric 0.428255, recipient metric 0.428255, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_9e-6` -> `lr_14e-6`, donor metric 0.428255, recipient metric 0.428615, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_9e-6` -> `lr_3e-6`, donor metric 0.420066, recipient metric 0.425981, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_9e-6` -> `lr_6e-6`, donor metric 0.420066, recipient metric 0.427998, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_9e-6` -> `lr_9e-6`, donor metric 0.420066, recipient metric 0.420066, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_9e-6` -> `lr_14e-6`, donor metric 0.420066, recipient metric 0.430371, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_9e-6` -> `lr_3e-6`, donor metric 0.424658, recipient metric 0.423932, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_9e-6` -> `lr_6e-6`, donor metric 0.424658, recipient metric 0.413064, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_9e-6` -> `lr_9e-6`, donor metric 0.424658, recipient metric 0.424658, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_9e-6` -> `lr_14e-6`, donor metric 0.424658, recipient metric 0.427375, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_9e-6` -> `lr_3e-6`, donor metric 0.418502, recipient metric 0.418432, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_9e-6` -> `lr_6e-6`, donor metric 0.418502, recipient metric 0.407891, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_9e-6` -> `lr_9e-6`, donor metric 0.418502, recipient metric 0.418502, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_9e-6` -> `lr_14e-6`, donor metric 0.418502, recipient metric 0.412285, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_9e-6` -> `lr_3e-6`, donor metric 0.429368, recipient metric 0.418683, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_9e-6` -> `lr_6e-6`, donor metric 0.429368, recipient metric 0.418225, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_9e-6` -> `lr_9e-6`, donor metric 0.429368, recipient metric 0.429368, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_9e-6` -> `lr_14e-6`, donor metric 0.429368, recipient metric 0.423733, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_9e-6` -> `lr_3e-6`, donor metric 0.422188, recipient metric 0.41262, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_9e-6` -> `lr_6e-6`, donor metric 0.422188, recipient metric 0.423381, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_9e-6` -> `lr_9e-6`, donor metric 0.422188, recipient metric 0.422188, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_9e-6` -> `lr_14e-6`, donor metric 0.422188, recipient metric 0.421791, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_9e-6` -> `lr_3e-6`, donor metric 0.422833, recipient metric 0.424289, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_9e-6` -> `lr_6e-6`, donor metric 0.422833, recipient metric 0.421939, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_9e-6` -> `lr_9e-6`, donor metric 0.422833, recipient metric 0.422833, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_9e-6` -> `lr_14e-6`, donor metric 0.422833, recipient metric 0.425664, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_9e-6` -> `lr_3e-6`, donor metric 0.412928, recipient metric 0.412462, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_9e-6` -> `lr_6e-6`, donor metric 0.412928, recipient metric 0.413754, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_9e-6` -> `lr_9e-6`, donor metric 0.412928, recipient metric 0.412928, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_9e-6` -> `lr_14e-6`, donor metric 0.412928, recipient metric 0.408385, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_9e-6` -> `lr_3e-6`, donor metric 0.412145, recipient metric 0.417421, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_9e-6` -> `lr_6e-6`, donor metric 0.412145, recipient metric 0.414527, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_9e-6` -> `lr_9e-6`, donor metric 0.412145, recipient metric 0.412145, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_9e-6` -> `lr_14e-6`, donor metric 0.412145, recipient metric 0.413626, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_9e-6` -> `lr_3e-6`, donor metric 0.419276, recipient metric 0.425551, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_9e-6` -> `lr_6e-6`, donor metric 0.419276, recipient metric 0.426202, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_9e-6` -> `lr_9e-6`, donor metric 0.419276, recipient metric 0.419276, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_9e-6` -> `lr_14e-6`, donor metric 0.419276, recipient metric 0.42024, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_9e-6` -> `lr_3e-6`, donor metric 0.424599, recipient metric 0.422152, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_9e-6` -> `lr_6e-6`, donor metric 0.424599, recipient metric 0.412215, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_9e-6` -> `lr_9e-6`, donor metric 0.424599, recipient metric 0.424599, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_9e-6` -> `lr_14e-6`, donor metric 0.424599, recipient metric 0.423214, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_9e-6` -> `lr_3e-6`, donor metric 0.401586, recipient metric 0.409194, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_9e-6` -> `lr_6e-6`, donor metric 0.401586, recipient metric 0.424427, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_9e-6` -> `lr_9e-6`, donor metric 0.401586, recipient metric 0.401586, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_9e-6` -> `lr_14e-6`, donor metric 0.401586, recipient metric 0.423792, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_9e-6` -> `lr_3e-6`, donor metric 0.423082, recipient metric 0.404497, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_9e-6` -> `lr_6e-6`, donor metric 0.423082, recipient metric 0.420155, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_9e-6` -> `lr_9e-6`, donor metric 0.423082, recipient metric 0.423082, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_9e-6` -> `lr_14e-6`, donor metric 0.423082, recipient metric 0.410842, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_9e-6` -> `lr_3e-6`, donor metric 0.432009, recipient metric 0.421653, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_9e-6` -> `lr_6e-6`, donor metric 0.432009, recipient metric 0.426703, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_9e-6` -> `lr_9e-6`, donor metric 0.432009, recipient metric 0.432009, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_9e-6` -> `lr_14e-6`, donor metric 0.432009, recipient metric 0.41168, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_9e-6` -> `lr_3e-6`, donor metric 0.42256, recipient metric 0.422787, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_9e-6` -> `lr_6e-6`, donor metric 0.42256, recipient metric 0.419419, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_9e-6` -> `lr_9e-6`, donor metric 0.42256, recipient metric 0.42256, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_9e-6` -> `lr_14e-6`, donor metric 0.42256, recipient metric 0.413097, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_9e-6` -> `lr_3e-6`, donor metric 0.421611, recipient metric 0.426525, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_9e-6` -> `lr_6e-6`, donor metric 0.421611, recipient metric 0.417956, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_9e-6` -> `lr_9e-6`, donor metric 0.421611, recipient metric 0.421611, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_9e-6` -> `lr_14e-6`, donor metric 0.421611, recipient metric 0.425454, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_9e-6` -> `lr_3e-6`, donor metric 0.415244, recipient metric 0.407833, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_9e-6` -> `lr_6e-6`, donor metric 0.415244, recipient metric 0.407842, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_9e-6` -> `lr_9e-6`, donor metric 0.415244, recipient metric 0.415244, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_9e-6` -> `lr_14e-6`, donor metric 0.415244, recipient metric 0.409575, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_9e-6` -> `lr_3e-6`, donor metric 0.412325, recipient metric 0.414634, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_9e-6` -> `lr_6e-6`, donor metric 0.412325, recipient metric 0.415342, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_9e-6` -> `lr_9e-6`, donor metric 0.412325, recipient metric 0.412325, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_9e-6` -> `lr_14e-6`, donor metric 0.412325, recipient metric 0.414504, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_9e-6` -> `lr_3e-6`, donor metric 0.420171, recipient metric 0.415193, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_9e-6` -> `lr_6e-6`, donor metric 0.420171, recipient metric 0.415572, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_9e-6` -> `lr_9e-6`, donor metric 0.420171, recipient metric 0.420171, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_9e-6` -> `lr_14e-6`, donor metric 0.420171, recipient metric 0.414852, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_9e-6` -> `lr_3e-6`, donor metric 0.4248, recipient metric 0.424943, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_9e-6` -> `lr_6e-6`, donor metric 0.4248, recipient metric 0.424983, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_9e-6` -> `lr_9e-6`, donor metric 0.4248, recipient metric 0.4248, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_9e-6` -> `lr_14e-6`, donor metric 0.4248, recipient metric 0.422295, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_9e-6` -> `lr_3e-6`, donor metric 0.434421, recipient metric 0.423287, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_9e-6` -> `lr_6e-6`, donor metric 0.434421, recipient metric 0.434086, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_9e-6` -> `lr_9e-6`, donor metric 0.434421, recipient metric 0.434421, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_9e-6` -> `lr_14e-6`, donor metric 0.434421, recipient metric 0.431084, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_9e-6` -> `lr_3e-6`, donor metric 0.423853, recipient metric 0.417061, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_9e-6` -> `lr_6e-6`, donor metric 0.423853, recipient metric 0.417082, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_9e-6` -> `lr_9e-6`, donor metric 0.423853, recipient metric 0.423853, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_9e-6` -> `lr_14e-6`, donor metric 0.423853, recipient metric 0.418548, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_9e-6` -> `lr_3e-6`, donor metric 0.409277, recipient metric 0.407919, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_9e-6` -> `lr_6e-6`, donor metric 0.409277, recipient metric 0.407881, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_9e-6` -> `lr_9e-6`, donor metric 0.409277, recipient metric 0.409277, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_9e-6` -> `lr_14e-6`, donor metric 0.409277, recipient metric 0.419388, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417324, recipient metric 0.419392, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417324, recipient metric 0.416681, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417324, recipient metric 0.417324, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417324, recipient metric 0.421578, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_9e-6` -> `lr_3e-6`, donor metric 0.416358, recipient metric 0.417491, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_9e-6` -> `lr_6e-6`, donor metric 0.416358, recipient metric 0.417026, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_9e-6` -> `lr_9e-6`, donor metric 0.416358, recipient metric 0.416358, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_9e-6` -> `lr_14e-6`, donor metric 0.416358, recipient metric 0.416429, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_9e-6` -> `lr_3e-6`, donor metric 0.412872, recipient metric 0.410223, LR 4.51e-06 -> 4.51e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_9e-6` -> `lr_6e-6`, donor metric 0.412872, recipient metric 0.414362, LR 5.08e-06 -> 5.08e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_9e-6` -> `lr_9e-6`, donor metric 0.412872, recipient metric 0.412872, LR 5.64e-06 -> 5.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_9e-6` -> `lr_14e-6`, donor metric 0.412872, recipient metric 0.411068, LR 6.77e-06 -> 6.77e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
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
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_100gen_seed3.yaml', '--slots', 'iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3', '--experiment-name', 'anchor_copy_lr_recenter_100gen_seed3_20260819_111523']`
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
