# anchor_copy_lr_recenter_50gen_50kval_20260820_032015

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
- Final checkpoint controller objective: 0.972785 by `lr_9e-6`
- Global best configured metric: 0.381266 by `lr_6e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_50gen_50kval_20260820_032015/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_6e-6`, generation 44), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=200, Pearson r=-0.029 (95% CI -0.145 to 0.086), Spearman rho=-0.069 (95% CI -0.191 to 0.065)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=0.805, Spearman rho=0.821
- Best checkpoint by tier: control: `lr_9e-6` gen 49 (0.384861), full_holdout: `lr_14e-6` gen 49 (0.373333)
- Best-checkpoint agreement across tiers: DISAGREE
- Control-selected global best has not been evaluated on monitor/full yet.
- Corroboration status: **provisional**
  - monitor: not available (baseline or selected checkpoint not evaluated on this tier)
  - full: not available (baseline or selected checkpoint not evaluated on this tier)
- No proxy-overfitting cases detected (control improved while monitor did not) in the paired generations evaluated so far.

## Model Selection Scores
- Final generation: 49
- All mistag/score values in percent (lower is better); status marks the generation's winner and/or the persisted anchor member.

| member | bc@0.8 | bd@0.8 | bc@0.9 | bd@0.9 | cb@0.5 | cd@0.5 | cb@0.8 | cd@0.8 | ctag_score | btag_score | total_mistag_score | LR | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_14e-6 | 0.1744 | 0.06832 | 3.003 | 0.1969 | 0.4915 | 0.06229 | 2.676 | 1.103 | 0.5483 | 0.2897 | 0.3986 | 1.1e-05 | - |
| lr_3e-6 | 0.1751 | 0.06011 | 3.023 | 0.1843 | 0.5029 | 0.05611 | 2.711 | 1.074 | 0.5354 | 0.2767 | 0.3849 | 7.35e-06 | - |
| lr_6e-6 | 0.1823 | 0.05811 | 2.959 | 0.1884 | 0.4991 | 0.05811 | 2.689 | 1.084 | 0.5392 | 0.2772 | 0.3866 | 8.27e-06 | anchor |
| lr_9e-6 | 0.1751 | 0.06011 | 3.015 | 0.1843 | 0.5009 | 0.05611 | 2.711 | 1.08 | 0.5356 | 0.2765 | 0.3849 | 9.19e-06 | winner |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_9e-6 | 0.4346 | 0.6012 | 0.3142 | 9e-06 | 9e-06 | 9e-06 | accepted_new_anchor | no |
| 1 | lr_14e-6 | 0.439 | 0.5819 | 0.3312 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 2 | lr_14e-6 | 0.443 | 0.5816 | 0.3373 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 3 | lr_14e-6 | 0.4396 | 0.5986 | 0.3229 | 1.08e-05 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 4 | lr_6e-6 | 0.4377 | 0.5916 | 0.3238 | 8.1e-06 | 9e-06 | 9e-06 | rewound_to_previous_anchor | no |
| 5 | lr_14e-6 | 0.4288 | 0.5924 | 0.3104 | 1.08e-05 | 9e-06 | 1.08e-05 | accepted_new_anchor | no |
| 6 | lr_14e-6 | 0.4177 | 0.5607 | 0.3112 | 1.296e-05 | 1.08e-05 | 1.296e-05 | accepted_new_anchor | no |
| 7 | lr_6e-6 | 0.4179 | 0.5578 | 0.3131 | 1.166e-05 | 1.296e-05 | 1.296e-05 | rewound_to_previous_anchor | no |
| 8 | lr_14e-6 | 0.4132 | 0.5587 | 0.3055 | 1.4e-05 | 1.296e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 9 | lr_6e-6 | 0.4154 | 0.544 | 0.3171 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 10 | lr_9e-6 | 0.4056 | 0.5662 | 0.2905 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 11 | lr_14e-6 | 0.409 | 0.5634 | 0.2969 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 12 | lr_3e-6 | 0.3967 | 0.5486 | 0.2868 | 1.12e-05 | 1.4e-05 | 1.12e-05 | accepted_new_anchor | no |
| 13 | lr_6e-6 | 0.397 | 0.5552 | 0.2839 | 1.008e-05 | 1.12e-05 | 1.12e-05 | rewound_to_previous_anchor | no |
| 14 | lr_14e-6 | 0.3962 | 0.5474 | 0.2868 | 1.344e-05 | 1.12e-05 | 1.344e-05 | accepted_new_anchor | no |
| 15 | lr_14e-6 | 0.3961 | 0.5376 | 0.2918 | 1.4e-05 | 1.344e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 16 | lr_6e-6 | 0.3973 | 0.5444 | 0.29 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 17 | lr_14e-6 | 0.3947 | 0.5344 | 0.2915 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 18 | lr_6e-6 | 0.4072 | 0.5455 | 0.304 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 19 | lr_14e-6 | 0.4017 | 0.5492 | 0.2938 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 20 | lr_9e-6 | 0.3992 | 0.5466 | 0.2915 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 21 | lr_14e-6 | 0.4012 | 0.526 | 0.3061 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 22 | lr_3e-6 | 0.3987 | 0.5418 | 0.2935 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 23 | lr_14e-6 | 0.3946 | 0.5308 | 0.2934 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 24 | lr_3e-6 | 0.4004 | 0.534 | 0.3003 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 25 | lr_3e-6 | 0.402 | 0.5416 | 0.2984 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 26 | lr_6e-6 | 0.3917 | 0.5465 | 0.2808 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 27 | lr_6e-6 | 0.3994 | 0.5383 | 0.2963 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 28 | lr_14e-6 | 0.406 | 0.5511 | 0.2991 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 29 | lr_14e-6 | 0.4014 | 0.5504 | 0.2927 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 30 | lr_6e-6 | 0.3911 | 0.5344 | 0.2862 | 1.134e-05 | 1.26e-05 | 1.134e-05 | accepted_new_anchor | no |
| 31 | lr_14e-6 | 0.4054 | 0.556 | 0.2956 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 32 | lr_6e-6 | 0.3928 | 0.5434 | 0.2839 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 33 | lr_14e-6 | 0.4005 | 0.5475 | 0.293 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 34 | lr_6e-6 | 0.3923 | 0.5377 | 0.2862 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 35 | lr_9e-6 | 0.3942 | 0.5292 | 0.2937 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 36 | lr_9e-6 | 0.3898 | 0.5471 | 0.2778 | 1.134e-05 | 1.134e-05 | 1.134e-05 | accepted_new_anchor | no |
| 37 | lr_9e-6 | 0.3888 | 0.5354 | 0.2823 | 1.134e-05 | 1.134e-05 | 1.134e-05 | accepted_new_anchor | no |
| 38 | lr_6e-6 | 0.3831 | 0.5269 | 0.2785 | 1.021e-05 | 1.134e-05 | 1.021e-05 | accepted_new_anchor | no |
| 39 | lr_3e-6 | 0.3909 | 0.5487 | 0.2784 | 8.165e-06 | 1.021e-05 | 1.021e-05 | rewound_to_previous_anchor | no |
| 40 | lr_9e-6 | 0.3907 | 0.5369 | 0.2844 | 1.021e-05 | 1.021e-05 | 1.021e-05 | rewound_to_previous_anchor | no |
| 41 | lr_14e-6 | 0.3922 | 0.5473 | 0.281 | 1.225e-05 | 1.021e-05 | 1.021e-05 | rewound_to_previous_anchor | no |
| 42 | lr_3e-6 | 0.3942 | 0.5501 | 0.2824 | 8.165e-06 | 1.021e-05 | 1.021e-05 | rewound_to_previous_anchor | no |
| 43 | lr_6e-6 | 0.3856 | 0.5311 | 0.28 | 9.185e-06 | 1.021e-05 | 1.021e-05 | rewound_to_previous_anchor | no |
| 44 | lr_6e-6 | 0.3813 | 0.5266 | 0.2761 | 9.185e-06 | 1.021e-05 | 9.185e-06 | accepted_new_anchor | no |
| 45 | lr_3e-6 | 0.3867 | 0.5392 | 0.2774 | 7.348e-06 | 9.185e-06 | 9.185e-06 | rewound_to_previous_anchor | no |
| 46 | lr_6e-6 | 0.3871 | 0.5308 | 0.2823 | 8.267e-06 | 9.185e-06 | 9.185e-06 | rewound_to_previous_anchor | no |
| 47 | lr_6e-6 | 0.387 | 0.5342 | 0.2803 | 8.267e-06 | 9.185e-06 | 9.185e-06 | rewound_to_previous_anchor | no |
| 48 | lr_3e-6 | 0.3983 | 0.5518 | 0.2875 | 7.348e-06 | 9.185e-06 | 9.185e-06 | rewound_to_previous_anchor | no |
| 49 | lr_9e-6 | 0.3849 | 0.5356 | 0.2765 | 9.185e-06 | 9.185e-06 | 9.185e-06 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_9e-6` -> `lr_3e-6`, donor metric 0.434584, recipient metric 0.435418, LR 3e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_6e-6`, donor metric 0.434584, recipient metric 0.450178, LR 6e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_9e-6`, donor metric 0.434584, recipient metric 0.434584, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_9e-6` -> `lr_14e-6`, donor metric 0.434584, recipient metric 0.444275, LR 1.4e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_3e-6`, donor metric 0.440558, recipient metric 0.445215, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_6e-6`, donor metric 0.440558, recipient metric 0.444835, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_9e-6`, donor metric 0.440558, recipient metric 0.440558, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_14e-6`, donor metric 0.440558, recipient metric 0.439009, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_3e-6`, donor metric 0.443381, recipient metric 0.448814, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_6e-6`, donor metric 0.443381, recipient metric 0.448742, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_9e-6`, donor metric 0.443381, recipient metric 0.443381, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_9e-6` -> `lr_14e-6`, donor metric 0.443381, recipient metric 0.44295, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_3e-6`, donor metric 0.442077, recipient metric 0.4442, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_6e-6`, donor metric 0.442077, recipient metric 0.443627, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_9e-6`, donor metric 0.442077, recipient metric 0.442077, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_14e-6`, donor metric 0.442077, recipient metric 0.439647, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_3e-6`, donor metric 0.446332, recipient metric 0.440162, LR 7.2e-06 -> 7.2e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_6e-6`, donor metric 0.446332, recipient metric 0.437654, LR 8.1e-06 -> 8.1e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_9e-6`, donor metric 0.446332, recipient metric 0.446332, LR 9e-06 -> 9e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_9e-6` -> `lr_14e-6`, donor metric 0.446332, recipient metric 0.446334, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_3e-6`, donor metric 0.428823, recipient metric 0.440029, LR 7.2e-06 -> 8.64e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_6e-6`, donor metric 0.428823, recipient metric 0.429797, LR 8.1e-06 -> 9.72e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_9e-6`, donor metric 0.428823, recipient metric 0.447246, LR 9e-06 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_14e-6`, donor metric 0.428823, recipient metric 0.428823, LR 1.08e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_3e-6`, donor metric 0.417685, recipient metric 0.428461, LR 8.64e-06 -> 1.04e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_6e-6`, donor metric 0.417685, recipient metric 0.417866, LR 9.72e-06 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_9e-6`, donor metric 0.417685, recipient metric 0.419484, LR 1.08e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_14e-6`, donor metric 0.417685, recipient metric 0.417685, LR 1.3e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_3e-6`, donor metric 0.419735, recipient metric 0.420465, LR 1.04e-05 -> 1.04e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_6e-6`, donor metric 0.419735, recipient metric 0.417901, LR 1.17e-05 -> 1.17e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_9e-6`, donor metric 0.419735, recipient metric 0.420199, LR 1.3e-05 -> 1.3e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_14e-6` -> `lr_14e-6`, donor metric 0.419735, recipient metric 0.419735, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_3e-6`, donor metric 0.413158, recipient metric 0.421639, LR 1.04e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_6e-6`, donor metric 0.413158, recipient metric 0.416245, LR 1.17e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_9e-6`, donor metric 0.413158, recipient metric 0.419499, LR 1.3e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_14e-6`, donor metric 0.413158, recipient metric 0.413158, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_3e-6`, donor metric 0.416804, recipient metric 0.419142, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_6e-6`, donor metric 0.416804, recipient metric 0.415361, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_9e-6`, donor metric 0.416804, recipient metric 0.416901, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_14e-6`, donor metric 0.416804, recipient metric 0.416804, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_3e-6`, donor metric 0.405593, recipient metric 0.413915, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_6e-6`, donor metric 0.405593, recipient metric 0.416948, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_9e-6`, donor metric 0.405593, recipient metric 0.405593, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_9e-6` -> `lr_14e-6`, donor metric 0.405593, recipient metric 0.411, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_3e-6`, donor metric 0.417272, recipient metric 0.422065, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_6e-6`, donor metric 0.417272, recipient metric 0.416346, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_9e-6`, donor metric 0.417272, recipient metric 0.417272, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_9e-6` -> `lr_14e-6`, donor metric 0.417272, recipient metric 0.40898, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_3e-6`, donor metric 0.396667, recipient metric 0.396667, LR 1.12e-05 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_6e-6`, donor metric 0.396667, recipient metric 0.405807, LR 1.26e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_9e-6`, donor metric 0.396667, recipient metric 0.407552, LR 1.4e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_3e-6` -> `lr_14e-6`, donor metric 0.396667, recipient metric 0.412156, LR 1.4e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_3e-6`, donor metric 0.39746, recipient metric 0.39746, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_6e-6`, donor metric 0.39746, recipient metric 0.397047, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_9e-6`, donor metric 0.39746, recipient metric 0.404776, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_3e-6` -> `lr_14e-6`, donor metric 0.39746, recipient metric 0.403681, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_3e-6`, donor metric 0.396218, recipient metric 0.404133, LR 8.96e-06 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_6e-6`, donor metric 0.396218, recipient metric 0.41324, LR 1.01e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_9e-6`, donor metric 0.396218, recipient metric 0.405086, LR 1.12e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_14e-6`, donor metric 0.396218, recipient metric 0.396218, LR 1.34e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_3e-6`, donor metric 0.396055, recipient metric 0.401937, LR 1.08e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_6e-6`, donor metric 0.396055, recipient metric 0.416779, LR 1.21e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_9e-6`, donor metric 0.396055, recipient metric 0.401873, LR 1.34e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_14e-6`, donor metric 0.396055, recipient metric 0.396055, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_3e-6`, donor metric 0.402517, recipient metric 0.407075, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_6e-6`, donor metric 0.402517, recipient metric 0.397338, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_9e-6`, donor metric 0.402517, recipient metric 0.409603, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_14e-6` -> `lr_14e-6`, donor metric 0.402517, recipient metric 0.402517, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_3e-6`, donor metric 0.394688, recipient metric 0.408854, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_6e-6`, donor metric 0.394688, recipient metric 0.409902, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_9e-6`, donor metric 0.394688, recipient metric 0.399071, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_14e-6`, donor metric 0.394688, recipient metric 0.394688, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_3e-6`, donor metric 0.407676, recipient metric 0.407499, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_6e-6`, donor metric 0.407676, recipient metric 0.407241, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_9e-6`, donor metric 0.407676, recipient metric 0.411146, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_14e-6`, donor metric 0.407676, recipient metric 0.407676, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_3e-6`, donor metric 0.401684, recipient metric 0.409396, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_6e-6`, donor metric 0.401684, recipient metric 0.403505, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_9e-6`, donor metric 0.401684, recipient metric 0.409612, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_14e-6` -> `lr_14e-6`, donor metric 0.401684, recipient metric 0.401684, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_3e-6`, donor metric 0.40889, recipient metric 0.401426, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_6e-6`, donor metric 0.40889, recipient metric 0.413349, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_9e-6`, donor metric 0.40889, recipient metric 0.399208, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_14e-6` -> `lr_14e-6`, donor metric 0.40889, recipient metric 0.40889, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_3e-6`, donor metric 0.401232, recipient metric 0.408143, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_6e-6`, donor metric 0.401232, recipient metric 0.408006, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_9e-6`, donor metric 0.401232, recipient metric 0.416461, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_14e-6` -> `lr_14e-6`, donor metric 0.401232, recipient metric 0.401232, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409147, recipient metric 0.398745, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409147, recipient metric 0.401842, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409147, recipient metric 0.402683, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409147, recipient metric 0.409147, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_3e-6`, donor metric 0.394633, recipient metric 0.410886, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_6e-6`, donor metric 0.394633, recipient metric 0.411618, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_9e-6`, donor metric 0.394633, recipient metric 0.402584, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_14e-6` -> `lr_14e-6`, donor metric 0.394633, recipient metric 0.394633, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_3e-6`, donor metric 0.409884, recipient metric 0.400415, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_6e-6`, donor metric 0.409884, recipient metric 0.403597, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_9e-6`, donor metric 0.409884, recipient metric 0.408202, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_14e-6` -> `lr_14e-6`, donor metric 0.409884, recipient metric 0.409884, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411885, recipient metric 0.402, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411885, recipient metric 0.404244, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411885, recipient metric 0.404923, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411885, recipient metric 0.411885, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_3e-6`, donor metric 0.391733, recipient metric 0.400629, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_6e-6`, donor metric 0.391733, recipient metric 0.391733, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_9e-6`, donor metric 0.391733, recipient metric 0.412357, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_14e-6`, donor metric 0.391733, recipient metric 0.391986, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_3e-6`, donor metric 0.399377, recipient metric 0.404369, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_6e-6`, donor metric 0.399377, recipient metric 0.399377, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_9e-6`, donor metric 0.399377, recipient metric 0.403978, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_14e-6`, donor metric 0.399377, recipient metric 0.409725, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_3e-6`, donor metric 0.406229, recipient metric 0.412314, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_6e-6`, donor metric 0.406229, recipient metric 0.406229, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_9e-6`, donor metric 0.406229, recipient metric 0.410006, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_14e-6`, donor metric 0.406229, recipient metric 0.406004, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_3e-6`, donor metric 0.409042, recipient metric 0.407439, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_6e-6`, donor metric 0.409042, recipient metric 0.409042, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_9e-6`, donor metric 0.409042, recipient metric 0.408961, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_14e-6`, donor metric 0.409042, recipient metric 0.401401, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_3e-6`, donor metric 0.391085, recipient metric 0.403492, LR 1.01e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_6e-6`, donor metric 0.391085, recipient metric 0.391085, LR 1.13e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_9e-6`, donor metric 0.391085, recipient metric 0.403831, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_14e-6`, donor metric 0.391085, recipient metric 0.392564, LR 1.4e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_3e-6`, donor metric 0.412183, recipient metric 0.406906, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_6e-6`, donor metric 0.412183, recipient metric 0.412183, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_9e-6`, donor metric 0.412183, recipient metric 0.406106, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_14e-6`, donor metric 0.412183, recipient metric 0.405406, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_3e-6`, donor metric 0.392776, recipient metric 0.407373, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_6e-6`, donor metric 0.392776, recipient metric 0.392776, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_9e-6`, donor metric 0.392776, recipient metric 0.4076, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_14e-6`, donor metric 0.392776, recipient metric 0.394243, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_3e-6`, donor metric 0.41062, recipient metric 0.403115, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_6e-6`, donor metric 0.41062, recipient metric 0.41062, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_9e-6`, donor metric 0.41062, recipient metric 0.404859, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_14e-6`, donor metric 0.41062, recipient metric 0.4005, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_3e-6`, donor metric 0.392327, recipient metric 0.411543, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_6e-6`, donor metric 0.392327, recipient metric 0.392327, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_9e-6`, donor metric 0.392327, recipient metric 0.399932, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_14e-6`, donor metric 0.392327, recipient metric 0.399819, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_3e-6`, donor metric 0.398664, recipient metric 0.401501, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_6e-6`, donor metric 0.398664, recipient metric 0.398664, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_9e-6`, donor metric 0.398664, recipient metric 0.394243, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_14e-6`, donor metric 0.398664, recipient metric 0.396606, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_9e-6` -> `lr_3e-6`, donor metric 0.389846, recipient metric 0.405725, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_9e-6` -> `lr_6e-6`, donor metric 0.389846, recipient metric 0.396396, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_9e-6` -> `lr_9e-6`, donor metric 0.389846, recipient metric 0.389846, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_9e-6` -> `lr_14e-6`, donor metric 0.389846, recipient metric 0.392313, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_3e-6`, donor metric 0.388783, recipient metric 0.397943, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_6e-6`, donor metric 0.388783, recipient metric 0.401677, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_9e-6`, donor metric 0.388783, recipient metric 0.388783, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_9e-6` -> `lr_14e-6`, donor metric 0.388783, recipient metric 0.396251, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_3e-6`, donor metric 0.383073, recipient metric 0.40529, LR 9.07e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_6e-6`, donor metric 0.383073, recipient metric 0.383073, LR 1.02e-05 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_9e-6`, donor metric 0.383073, recipient metric 0.394092, LR 1.13e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_14e-6`, donor metric 0.383073, recipient metric 0.395442, LR 1.36e-05 -> 1.22e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_3e-6`, donor metric 0.39384, recipient metric 0.390885, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_6e-6`, donor metric 0.39384, recipient metric 0.39384, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_9e-6`, donor metric 0.39384, recipient metric 0.40072, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_14e-6`, donor metric 0.39384, recipient metric 0.404255, LR 1.22e-05 -> 1.22e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_3e-6`, donor metric 0.396552, recipient metric 0.396999, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_6e-6`, donor metric 0.396552, recipient metric 0.396552, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_9e-6`, donor metric 0.396552, recipient metric 0.390731, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_14e-6`, donor metric 0.396552, recipient metric 0.396695, LR 1.22e-05 -> 1.22e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_3e-6`, donor metric 0.393152, recipient metric 0.399449, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_6e-6`, donor metric 0.393152, recipient metric 0.393152, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_9e-6`, donor metric 0.393152, recipient metric 0.392499, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_14e-6`, donor metric 0.393152, recipient metric 0.392156, LR 1.22e-05 -> 1.22e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_3e-6`, donor metric 0.397383, recipient metric 0.394157, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_6e-6`, donor metric 0.397383, recipient metric 0.397383, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_9e-6`, donor metric 0.397383, recipient metric 0.399039, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_14e-6`, donor metric 0.397383, recipient metric 0.397144, LR 1.22e-05 -> 1.22e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_3e-6`, donor metric 0.385634, recipient metric 0.400709, LR 8.16e-06 -> 8.16e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_6e-6`, donor metric 0.385634, recipient metric 0.385634, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_9e-6`, donor metric 0.385634, recipient metric 0.396389, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_14e-6`, donor metric 0.385634, recipient metric 0.399708, LR 1.22e-05 -> 1.22e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381266, recipient metric 0.381429, LR 8.16e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381266, recipient metric 0.381266, LR 9.19e-06 -> 8.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381266, recipient metric 0.395289, LR 1.02e-05 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381266, recipient metric 0.400276, LR 1.22e-05 -> 1.1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_3e-6`, donor metric 0.392875, recipient metric 0.386741, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_6e-6`, donor metric 0.392875, recipient metric 0.392875, LR 8.27e-06 -> 8.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_9e-6`, donor metric 0.392875, recipient metric 0.398076, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_14e-6`, donor metric 0.392875, recipient metric 0.39373, LR 1.1e-05 -> 1.1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_3e-6`, donor metric 0.387095, recipient metric 0.388576, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_6e-6`, donor metric 0.387095, recipient metric 0.387095, LR 8.27e-06 -> 8.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_9e-6`, donor metric 0.387095, recipient metric 0.396003, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_14e-6`, donor metric 0.387095, recipient metric 0.387172, LR 1.1e-05 -> 1.1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_3e-6`, donor metric 0.386967, recipient metric 0.399738, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_6e-6`, donor metric 0.386967, recipient metric 0.386967, LR 8.27e-06 -> 8.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_9e-6`, donor metric 0.386967, recipient metric 0.397542, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_14e-6`, donor metric 0.386967, recipient metric 0.397153, LR 1.1e-05 -> 1.1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_3e-6`, donor metric 0.408855, recipient metric 0.398304, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_6e-6`, donor metric 0.408855, recipient metric 0.408855, LR 8.27e-06 -> 8.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_9e-6`, donor metric 0.408855, recipient metric 0.40544, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_14e-6`, donor metric 0.408855, recipient metric 0.408593, LR 1.1e-05 -> 1.1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_3e-6`, donor metric 0.386624, recipient metric 0.384913, LR 7.35e-06 -> 7.35e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_6e-6`, donor metric 0.386624, recipient metric 0.386624, LR 8.27e-06 -> 8.27e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_9e-6`, donor metric 0.386624, recipient metric 0.384861, LR 9.19e-06 -> 9.19e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_14e-6`, donor metric 0.386624, recipient metric 0.398561, LR 1.1e-05 -> 1.1e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- [Skipped exploits (significance gating)](plots/report/skipped_exploits.csv) -- 0 donor->recipient replacement(s) declined for insufficient significance

## Method
- Method: `anchor_copy_lr_recenter`
- Population: 4 trials
- Training interval: 240000 samples/trial chunk (2x samples_per_epoch)
- Evaluation interval: every 1 training chunk(s), 150000 validation samples
- Exploit interval: every 1 training chunk(s)
- Exploit significance gating: disabled (nominal rank order only)
- Burn-in: 0 generation(s) (observe-only, no exploit/controller LR action applied)
- Monitor-tier cadence: disabled generation(s), all population members, read-only
- Full-tier cadence: 10 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `b3c69dd88163c6f447475860d145ce77d9308cc7`
- Git dirty: `False`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_50gen_50kval.yaml', '--slots', 'iutgpu02:0,iutgpu02:1,iutgpu02:2,iutgpu02:3', '--experiment-name', 'anchor_copy_lr_recenter_50gen_50kval_20260820_032015']`
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
- No data-loader shutdown-race warnings observed across 240 evaluation(s).
