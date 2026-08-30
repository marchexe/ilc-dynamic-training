# anchor_copy_lr_recenter_25gen_50kval_20260820_032015

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
- Final checkpoint controller objective: 0.978523 by `lr_6e-6`
- Global best configured metric: 0.391642 by `lr_9e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_25gen_50kval_20260820_032015/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_9e-6`, generation 14), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_9e-6`, generation 18) -- these are two distinct selection criteria, not the same checkpoint.
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=100, Pearson r=-0.077 (95% CI -0.195 to 0.072), Spearman rho=-0.097 (95% CI -0.280 to 0.085)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=0.771, Spearman rho=0.477
- Best checkpoint by tier: control: `lr_9e-6` gen 14 (0.391642), full_holdout: `lr_14e-6` gen 24 (0.382824)
- Best-checkpoint agreement across tiers: DISAGREE
- Control-selected global best (`lr_9e-6`, gen 14) measured on other tiers: full_holdout: 0.386104
- Corroboration status: **provisional**
  - monitor: not available (baseline or selected checkpoint not evaluated on this tier)
  - full: not available (baseline or selected checkpoint not evaluated on this tier)
- No proxy-overfitting cases detected (control improved while monitor did not) in the paired generations evaluated so far.

## Model Selection Scores
- Final generation: 24
- All mistag/score values in percent (lower is better); status marks the generation's winner and/or the persisted anchor member.

| member | bc@0.8 | bd@0.8 | bc@0.9 | bd@0.9 | cb@0.5 | cd@0.5 | cb@0.8 | cd@0.8 | ctag_score | btag_score | total_mistag_score | LR | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_14e-6 | 0.1766 | 0.06416 | 3.099 | 0.1965 | 0.502 | 0.07218 | 2.657 | 1.097 | 0.57 | 0.2882 | 0.4053 | 8.4e-06 | - |
| lr_3e-6 | 0.1888 | 0.06824 | 3.076 | 0.1846 | 0.505 | 0.06623 | 2.681 | 1.13 | 0.5642 | 0.2925 | 0.4062 | 5.6e-06 | - |
| lr_6e-6 | 0.1824 | 0.06233 | 3.053 | 0.181 | 0.5132 | 0.06635 | 2.695 | 1.076 | 0.5605 | 0.2815 | 0.3972 | 6.3e-06 | winner |
| lr_9e-6 | 0.1909 | 0.06825 | 3.082 | 0.1847 | 0.5049 | 0.06624 | 2.681 | 1.122 | 0.5632 | 0.2934 | 0.4065 | 7e-06 | anchor |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_9e-6 | 0.4262 | 0.5959 | 0.3049 | 9e-06 | 9e-06 | 9e-06 | accepted_new_anchor | no |
| 1 | lr_3e-6 | 0.4098 | 0.5751 | 0.2921 | 7.2e-06 | 9e-06 | 7.2e-06 | accepted_new_anchor | no |
| 2 | lr_14e-6 | 0.4122 | 0.5871 | 0.2894 | 8.64e-06 | 7.2e-06 | 7.2e-06 | rewound_to_previous_anchor | no |
| 3 | lr_9e-6 | 0.4101 | 0.573 | 0.2936 | 7.2e-06 | 7.2e-06 | 7.2e-06 | rewound_to_previous_anchor | no |
| 4 | lr_14e-6 | 0.4159 | 0.567 | 0.305 | 8.64e-06 | 7.2e-06 | 7.2e-06 | rewound_to_previous_anchor | no |
| 5 | lr_3e-6 | 0.4121 | 0.5729 | 0.2965 | 5.76e-06 | 7.2e-06 | 7.2e-06 | rewound_to_previous_anchor | no |
| 6 | lr_14e-6 | 0.4135 | 0.5707 | 0.2995 | 8.64e-06 | 7.2e-06 | 7.2e-06 | rewound_to_previous_anchor | no |
| 7 | lr_6e-6 | 0.4037 | 0.5595 | 0.2912 | 6.48e-06 | 7.2e-06 | 6.48e-06 | accepted_new_anchor | no |
| 8 | lr_14e-6 | 0.3985 | 0.5606 | 0.2833 | 7.776e-06 | 6.48e-06 | 7.776e-06 | accepted_new_anchor | no |
| 9 | lr_14e-6 | 0.4 | 0.5485 | 0.2917 | 9.331e-06 | 7.776e-06 | 7.776e-06 | rewound_to_previous_anchor | no |
| 10 | lr_6e-6 | 0.3975 | 0.554 | 0.2852 | 6.998e-06 | 7.776e-06 | 6.998e-06 | accepted_new_anchor | no |
| 11 | lr_9e-6 | 0.4083 | 0.556 | 0.2999 | 6.998e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 12 | lr_6e-6 | 0.4026 | 0.5602 | 0.2894 | 6.299e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 13 | lr_6e-6 | 0.4012 | 0.5451 | 0.2952 | 6.299e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 14 | lr_9e-6 | 0.3916 | 0.5304 | 0.2892 | 6.998e-06 | 6.998e-06 | 6.998e-06 | accepted_new_anchor | no |
| 15 | lr_9e-6 | 0.393 | 0.5457 | 0.283 | 6.998e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 16 | lr_9e-6 | 0.4058 | 0.5595 | 0.2943 | 6.998e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 17 | lr_3e-6 | 0.4002 | 0.5526 | 0.2898 | 5.599e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 18 | lr_9e-6 | 0.3919 | 0.5399 | 0.2845 | 6.998e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 19 | lr_3e-6 | 0.3971 | 0.5475 | 0.288 | 5.599e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 20 | lr_14e-6 | 0.3962 | 0.5616 | 0.2795 | 8.398e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 21 | lr_14e-6 | 0.3957 | 0.5571 | 0.2811 | 8.398e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 22 | lr_3e-6 | 0.4001 | 0.5449 | 0.2938 | 5.599e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 23 | lr_3e-6 | 0.3972 | 0.5365 | 0.294 | 5.599e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |
| 24 | lr_6e-6 | 0.3972 | 0.5605 | 0.2815 | 6.299e-06 | 6.998e-06 | 6.998e-06 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_9e-6` -> `lr_3e-6`, donor metric 0.426241, recipient metric 0.435359, LR 3e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_6e-6`, donor metric 0.426241, recipient metric 0.44573, LR 6e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_9e-6`, donor metric 0.426241, recipient metric 0.426241, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_14e-6`, donor metric 0.426241, recipient metric 0.433977, LR 1.4e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_3e-6`, donor metric 0.409838, recipient metric 0.409838, LR 7.2e-06 -> 5.76e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_6e-6`, donor metric 0.409838, recipient metric 0.416872, LR 8.1e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_9e-6`, donor metric 0.409838, recipient metric 0.414201, LR 9e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_3e-6` -> `lr_14e-6`, donor metric 0.409838, recipient metric 0.412101, LR 1.08e-05 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_3e-6`, donor metric 0.416119, recipient metric 0.416119, LR 5.76e-06 -> 5.76e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_6e-6`, donor metric 0.416119, recipient metric 0.41279, LR 6.48e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_9e-6`, donor metric 0.416119, recipient metric 0.426726, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_3e-6` -> `lr_14e-6`, donor metric 0.416119, recipient metric 0.412193, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_3e-6`, donor metric 0.421362, recipient metric 0.421362, LR 5.76e-06 -> 5.76e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_6e-6`, donor metric 0.421362, recipient metric 0.41114, LR 6.48e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_9e-6`, donor metric 0.421362, recipient metric 0.410129, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_3e-6` -> `lr_14e-6`, donor metric 0.421362, recipient metric 0.423493, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_3e-6` -> `lr_3e-6`, donor metric 0.428516, recipient metric 0.428516, LR 5.76e-06 -> 5.76e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_3e-6` -> `lr_6e-6`, donor metric 0.428516, recipient metric 0.421393, LR 6.48e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_3e-6` -> `lr_9e-6`, donor metric 0.428516, recipient metric 0.417399, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_3e-6` -> `lr_14e-6`, donor metric 0.428516, recipient metric 0.415905, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_3e-6`, donor metric 0.412145, recipient metric 0.412145, LR 5.76e-06 -> 5.76e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_6e-6`, donor metric 0.412145, recipient metric 0.416533, LR 6.48e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_9e-6`, donor metric 0.412145, recipient metric 0.422104, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_3e-6` -> `lr_14e-6`, donor metric 0.412145, recipient metric 0.421017, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_3e-6` -> `lr_3e-6`, donor metric 0.413894, recipient metric 0.413894, LR 5.76e-06 -> 5.76e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_3e-6` -> `lr_6e-6`, donor metric 0.413894, recipient metric 0.414309, LR 6.48e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_3e-6` -> `lr_9e-6`, donor metric 0.413894, recipient metric 0.414588, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_3e-6` -> `lr_14e-6`, donor metric 0.413894, recipient metric 0.413478, LR 8.64e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_3e-6`, donor metric 0.403664, recipient metric 0.416109, LR 5.76e-06 -> 5.18e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_6e-6`, donor metric 0.403664, recipient metric 0.403664, LR 6.48e-06 -> 5.83e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_9e-6`, donor metric 0.403664, recipient metric 0.412817, LR 7.2e-06 -> 6.48e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_14e-6`, donor metric 0.403664, recipient metric 0.407335, LR 8.64e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_3e-6`, donor metric 0.39853, recipient metric 0.403765, LR 5.18e-06 -> 6.22e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_6e-6`, donor metric 0.39853, recipient metric 0.402572, LR 5.83e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_9e-6`, donor metric 0.39853, recipient metric 0.405989, LR 6.48e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_14e-6`, donor metric 0.39853, recipient metric 0.39853, LR 7.78e-06 -> 9.33e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_3e-6`, donor metric 0.399974, recipient metric 0.40273, LR 6.22e-06 -> 6.22e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_6e-6`, donor metric 0.399974, recipient metric 0.402143, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_9e-6`, donor metric 0.399974, recipient metric 0.408373, LR 7.78e-06 -> 7.78e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_14e-6`, donor metric 0.399974, recipient metric 0.399974, LR 9.33e-06 -> 9.33e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_3e-6`, donor metric 0.397501, recipient metric 0.418669, LR 6.22e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_6e-6`, donor metric 0.397501, recipient metric 0.397501, LR 7e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_9e-6`, donor metric 0.397501, recipient metric 0.407972, LR 7.78e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_14e-6`, donor metric 0.397501, recipient metric 0.417908, LR 9.33e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_3e-6`, donor metric 0.411255, recipient metric 0.411529, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_6e-6`, donor metric 0.411255, recipient metric 0.411255, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_9e-6`, donor metric 0.411255, recipient metric 0.408339, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_14e-6`, donor metric 0.411255, recipient metric 0.410202, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_3e-6`, donor metric 0.402607, recipient metric 0.416939, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_6e-6`, donor metric 0.402607, recipient metric 0.402607, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_9e-6`, donor metric 0.402607, recipient metric 0.404495, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_14e-6`, donor metric 0.402607, recipient metric 0.410568, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_3e-6`, donor metric 0.401182, recipient metric 0.402182, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_6e-6`, donor metric 0.401182, recipient metric 0.401182, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_9e-6`, donor metric 0.401182, recipient metric 0.414444, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_14e-6`, donor metric 0.401182, recipient metric 0.411502, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_3e-6`, donor metric 0.391642, recipient metric 0.400577, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_6e-6`, donor metric 0.391642, recipient metric 0.400971, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_9e-6`, donor metric 0.391642, recipient metric 0.391642, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_9e-6` -> `lr_14e-6`, donor metric 0.391642, recipient metric 0.401268, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_3e-6`, donor metric 0.393, recipient metric 0.412283, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_6e-6`, donor metric 0.393, recipient metric 0.404144, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_9e-6`, donor metric 0.393, recipient metric 0.393, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_9e-6` -> `lr_14e-6`, donor metric 0.393, recipient metric 0.404643, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_3e-6`, donor metric 0.405808, recipient metric 0.406058, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_6e-6`, donor metric 0.405808, recipient metric 0.412515, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_9e-6`, donor metric 0.405808, recipient metric 0.405808, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_9e-6` -> `lr_14e-6`, donor metric 0.405808, recipient metric 0.405815, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_3e-6`, donor metric 0.411122, recipient metric 0.400198, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_6e-6`, donor metric 0.411122, recipient metric 0.413216, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_9e-6`, donor metric 0.411122, recipient metric 0.411122, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_9e-6` -> `lr_14e-6`, donor metric 0.411122, recipient metric 0.406551, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_3e-6`, donor metric 0.391907, recipient metric 0.393598, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_6e-6`, donor metric 0.391907, recipient metric 0.395554, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_9e-6`, donor metric 0.391907, recipient metric 0.391907, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_9e-6` -> `lr_14e-6`, donor metric 0.391907, recipient metric 0.396224, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_3e-6`, donor metric 0.404228, recipient metric 0.397104, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_6e-6`, donor metric 0.404228, recipient metric 0.412602, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_9e-6`, donor metric 0.404228, recipient metric 0.404228, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_14e-6`, donor metric 0.404228, recipient metric 0.39819, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_3e-6`, donor metric 0.402305, recipient metric 0.403433, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_6e-6`, donor metric 0.402305, recipient metric 0.398368, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_9e-6`, donor metric 0.402305, recipient metric 0.402305, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_14e-6`, donor metric 0.402305, recipient metric 0.396154, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_3e-6`, donor metric 0.410232, recipient metric 0.396184, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_6e-6`, donor metric 0.410232, recipient metric 0.401588, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_9e-6`, donor metric 0.410232, recipient metric 0.410232, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_14e-6`, donor metric 0.410232, recipient metric 0.395706, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_3e-6`, donor metric 0.401127, recipient metric 0.400136, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_6e-6`, donor metric 0.401127, recipient metric 0.407272, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_9e-6`, donor metric 0.401127, recipient metric 0.401127, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_14e-6`, donor metric 0.401127, recipient metric 0.402667, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_3e-6`, donor metric 0.403898, recipient metric 0.397157, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_6e-6`, donor metric 0.403898, recipient metric 0.407848, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_9e-6`, donor metric 0.403898, recipient metric 0.403898, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_14e-6`, donor metric 0.403898, recipient metric 0.402486, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_3e-6`, donor metric 0.406516, recipient metric 0.406227, LR 5.6e-06 -> 5.6e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_6e-6`, donor metric 0.406516, recipient metric 0.397233, LR 6.3e-06 -> 6.3e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_9e-6`, donor metric 0.406516, recipient metric 0.406516, LR 7e-06 -> 7e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_9e-6` -> `lr_14e-6`, donor metric 0.406516, recipient metric 0.405313, LR 8.4e-06 -> 8.4e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- [Skipped exploits (significance gating)](plots/report/skipped_exploits.csv) -- 0 donor->recipient replacement(s) declined for insufficient significance

## Method
- Method: `anchor_copy_lr_recenter`
- Population: 4 trials
- Training interval: 480000 samples/trial chunk (4x samples_per_epoch)
- Evaluation interval: every 1 training chunk(s), 150000 validation samples
- Exploit interval: every 1 training chunk(s)
- Exploit significance gating: disabled (nominal rank order only)
- Burn-in: 0 generation(s) (observe-only, no exploit/controller LR action applied)
- Monitor-tier cadence: disabled generation(s), all population members, read-only
- Full-tier cadence: 5 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `b3c69dd88163c6f447475860d145ce77d9308cc7`
- Git dirty: `False`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_25gen_50kval.yaml', '--slots', 'iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3', '--experiment-name', 'anchor_copy_lr_recenter_25gen_50kval_20260820_032015']`
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
- No data-loader shutdown-race warnings observed across 140 evaluation(s).
