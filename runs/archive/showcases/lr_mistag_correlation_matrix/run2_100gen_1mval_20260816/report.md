# anchor_copy_lr_recenter_100gen_1mval_20260816_105629

## Results
- Evaluation type: `proxy`
- Validation dataset: `/data/suehara/part/march/datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet`
- Validation suffix: `val1000k`
- Validation sample count: 3000000
- Controller objective: mean predefined fixed-WP mistag percent (lower is better; not a HEP metric)
- Configured PBT selection metric: `validation_total_reference_mistag_geomean_percent` (min)
- **`total_mistag_score` (sqrt(ctag_score * btag_score)) is this run's PBT ranking metric** -- ctag_score/btag_score are its two components, shown for diagnosis, never used for ranking on their own.
- Measured baseline: n/a
- Configured reference: n/a
- Final checkpoint controller objective: 0.986491 by `lr_14e-6`
- Global best configured metric: 0.378275 by `lr_6e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_100gen_1mval_20260816_105629/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_6e-6`, generation 31), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
  - Differs from the separate best-physics-score checkpoint (`lr_3e-6`, generation 35) -- these are two distinct selection criteria, not the same checkpoint.
  - Validation: `/data/suehara/part/march/datasets/20250711_ilc_nnqq_sgv_10m_3cat_parquet` (`val1000k`), 3000000 samples
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=400, Pearson r=-0.068 (95% CI -0.226 to 0.102), Spearman rho=-0.024 (95% CI -0.125 to 0.083)
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=20, Pearson r=0.671, Spearman rho=0.767
- Best checkpoint by tier: control: `lr_14e-6` gen 59 (0.37926), full_holdout: `lr_14e-6` gen 99 (0.381087)
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
| lr_14e-6 | 0.1707 | 0.05497 | 3.158 | 0.1909 | 0.4924 | 0.05207 | 2.64 | 1.133 | 0.5262 | 0.2742 | 0.3799 | 1.36e-05 | winner |
| lr_3e-6 | 0.1686 | 0.05638 | 3.166 | 0.1908 | 0.5013 | 0.05367 | 2.659 | 1.129 | 0.5331 | 0.2753 | 0.3831 | 9.07e-06 | - |
| lr_6e-6 | 0.1706 | 0.0567 | 3.147 | 0.1919 | 0.4989 | 0.05229 | 2.658 | 1.137 | 0.5299 | 0.2765 | 0.3828 | 1.02e-05 | anchor |
| lr_9e-6 | 0.1695 | 0.05632 | 3.145 | 0.1907 | 0.4967 | 0.05141 | 2.647 | 1.134 | 0.5261 | 0.275 | 0.3804 | 1.13e-05 | - |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_14e-6 | 0.4278 | 0.6048 | 0.3026 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 1 | lr_6e-6 | 0.4183 | 0.5894 | 0.2969 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 2 | lr_14e-6 | 0.4144 | 0.5834 | 0.2944 | 1.4e-05 | 1.26e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 3 | lr_9e-6 | 0.4064 | 0.5667 | 0.2915 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 4 | lr_14e-6 | 0.4011 | 0.5606 | 0.287 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 5 | lr_14e-6 | 0.3985 | 0.5562 | 0.2855 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 6 | lr_14e-6 | 0.3954 | 0.5521 | 0.2831 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 7 | lr_6e-6 | 0.3944 | 0.5485 | 0.2836 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 8 | lr_14e-6 | 0.3905 | 0.5415 | 0.2816 | 1.4e-05 | 1.26e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 9 | lr_14e-6 | 0.3885 | 0.5343 | 0.2825 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 10 | lr_6e-6 | 0.3907 | 0.5379 | 0.2839 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 11 | lr_3e-6 | 0.3888 | 0.5383 | 0.2808 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 12 | lr_3e-6 | 0.3903 | 0.5405 | 0.2818 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 13 | lr_14e-6 | 0.3903 | 0.5361 | 0.2842 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 14 | lr_3e-6 | 0.3895 | 0.5397 | 0.2811 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 15 | lr_14e-6 | 0.3883 | 0.5376 | 0.2805 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 16 | lr_6e-6 | 0.3879 | 0.5368 | 0.2802 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 17 | lr_14e-6 | 0.3872 | 0.5346 | 0.2805 | 1.4e-05 | 1.26e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 18 | lr_14e-6 | 0.3867 | 0.5324 | 0.2809 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 19 | lr_9e-6 | 0.3833 | 0.5296 | 0.2775 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 20 | lr_14e-6 | 0.3842 | 0.5319 | 0.2775 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 21 | lr_3e-6 | 0.3836 | 0.5329 | 0.2761 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 22 | lr_9e-6 | 0.3821 | 0.5291 | 0.276 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 23 | lr_9e-6 | 0.3829 | 0.5318 | 0.2757 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 24 | lr_6e-6 | 0.3814 | 0.53 | 0.2745 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 25 | lr_14e-6 | 0.3811 | 0.5304 | 0.2738 | 1.4e-05 | 1.26e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 26 | lr_6e-6 | 0.3786 | 0.5282 | 0.2714 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 27 | lr_14e-6 | 0.3803 | 0.5273 | 0.2743 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 28 | lr_6e-6 | 0.3815 | 0.5285 | 0.2754 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 29 | lr_3e-6 | 0.381 | 0.5262 | 0.2759 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 30 | lr_3e-6 | 0.3789 | 0.5244 | 0.2737 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 31 | lr_6e-6 | 0.3783 | 0.5223 | 0.274 | 1.134e-05 | 1.26e-05 | 1.134e-05 | accepted_new_anchor | no |
| 32 | lr_14e-6 | 0.3814 | 0.5284 | 0.2754 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 33 | lr_3e-6 | 0.3803 | 0.5277 | 0.2741 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 34 | lr_9e-6 | 0.3789 | 0.5253 | 0.2733 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 35 | lr_3e-6 | 0.3783 | 0.5242 | 0.273 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 36 | lr_3e-6 | 0.3796 | 0.5266 | 0.2736 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 37 | lr_6e-6 | 0.3823 | 0.5289 | 0.2763 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 38 | lr_14e-6 | 0.3835 | 0.5282 | 0.2784 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 39 | lr_9e-6 | 0.3813 | 0.5315 | 0.2736 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 40 | lr_3e-6 | 0.38 | 0.5259 | 0.2746 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 41 | lr_3e-6 | 0.3793 | 0.5221 | 0.2756 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 42 | lr_9e-6 | 0.3808 | 0.5291 | 0.2741 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 43 | lr_6e-6 | 0.3807 | 0.5276 | 0.2747 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 44 | lr_3e-6 | 0.3795 | 0.5261 | 0.2737 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 45 | lr_9e-6 | 0.3793 | 0.527 | 0.2729 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 46 | lr_3e-6 | 0.3814 | 0.5256 | 0.2767 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 47 | lr_6e-6 | 0.38 | 0.5266 | 0.2742 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 48 | lr_14e-6 | 0.381 | 0.5273 | 0.2752 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 49 | lr_6e-6 | 0.3791 | 0.5222 | 0.2752 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 50 | lr_9e-6 | 0.3804 | 0.5246 | 0.2758 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 51 | lr_9e-6 | 0.3814 | 0.5293 | 0.2748 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 52 | lr_3e-6 | 0.3805 | 0.5264 | 0.2751 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 53 | lr_9e-6 | 0.3788 | 0.5281 | 0.2717 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 54 | lr_14e-6 | 0.3807 | 0.5285 | 0.2743 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 55 | lr_9e-6 | 0.38 | 0.528 | 0.2735 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 56 | lr_14e-6 | 0.3785 | 0.5259 | 0.2725 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 57 | lr_9e-6 | 0.3811 | 0.526 | 0.2761 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 58 | lr_9e-6 | 0.3812 | 0.5284 | 0.2751 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 59 | lr_14e-6 | 0.3793 | 0.526 | 0.2735 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 60 | lr_3e-6 | 0.3799 | 0.5269 | 0.2739 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 61 | lr_9e-6 | 0.381 | 0.5256 | 0.2761 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 62 | lr_14e-6 | 0.3791 | 0.5271 | 0.2726 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 63 | lr_14e-6 | 0.3804 | 0.5272 | 0.2744 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 64 | lr_3e-6 | 0.3798 | 0.5276 | 0.2735 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 65 | lr_6e-6 | 0.3798 | 0.5251 | 0.2746 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 66 | lr_9e-6 | 0.3816 | 0.524 | 0.2778 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 67 | lr_3e-6 | 0.3794 | 0.5294 | 0.2719 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 68 | lr_3e-6 | 0.3798 | 0.5263 | 0.2741 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 69 | lr_3e-6 | 0.3807 | 0.5289 | 0.274 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 70 | lr_14e-6 | 0.3791 | 0.5221 | 0.2752 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 71 | lr_9e-6 | 0.3808 | 0.5245 | 0.2765 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 72 | lr_6e-6 | 0.3803 | 0.5282 | 0.2737 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 73 | lr_9e-6 | 0.3799 | 0.525 | 0.2749 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 74 | lr_9e-6 | 0.3805 | 0.5262 | 0.2751 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 75 | lr_14e-6 | 0.3803 | 0.5263 | 0.2748 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 76 | lr_6e-6 | 0.379 | 0.5263 | 0.273 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 77 | lr_3e-6 | 0.38 | 0.5262 | 0.2743 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 78 | lr_14e-6 | 0.3795 | 0.5252 | 0.2742 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 79 | lr_9e-6 | 0.3817 | 0.5289 | 0.2754 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 80 | lr_14e-6 | 0.3784 | 0.5238 | 0.2734 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 81 | lr_6e-6 | 0.3794 | 0.5278 | 0.2728 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 82 | lr_3e-6 | 0.3797 | 0.5246 | 0.2748 | 9.072e-06 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 83 | lr_6e-6 | 0.3799 | 0.5272 | 0.2738 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 84 | lr_14e-6 | 0.3794 | 0.5264 | 0.2734 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 85 | lr_9e-6 | 0.3819 | 0.527 | 0.2767 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 86 | lr_9e-6 | 0.3793 | 0.5255 | 0.2738 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 87 | lr_14e-6 | 0.3792 | 0.5275 | 0.2725 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 88 | lr_6e-6 | 0.379 | 0.5253 | 0.2734 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 89 | lr_9e-6 | 0.3796 | 0.5253 | 0.2743 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 90 | lr_6e-6 | 0.3818 | 0.5298 | 0.2752 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 91 | lr_14e-6 | 0.3796 | 0.5264 | 0.2737 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 92 | lr_14e-6 | 0.3797 | 0.5274 | 0.2733 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 93 | lr_9e-6 | 0.3799 | 0.5243 | 0.2753 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 94 | lr_6e-6 | 0.3786 | 0.5272 | 0.2719 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 95 | lr_9e-6 | 0.3794 | 0.5225 | 0.2755 | 1.134e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 96 | lr_14e-6 | 0.3804 | 0.5299 | 0.273 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 97 | lr_6e-6 | 0.3814 | 0.5272 | 0.2759 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 98 | lr_6e-6 | 0.3809 | 0.5292 | 0.2741 | 1.021e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |
| 99 | lr_14e-6 | 0.3799 | 0.5262 | 0.2742 | 1.361e-05 | 1.134e-05 | 1.134e-05 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_14e-6` -> `lr_3e-6`, donor metric 0.427787, recipient metric 0.433589, LR 3e-06 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_6e-6`, donor metric 0.427787, recipient metric 0.428764, LR 6e-06 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_9e-6`, donor metric 0.427787, recipient metric 0.429255, LR 9e-06 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_14e-6`, donor metric 0.427787, recipient metric 0.427787, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418322, recipient metric 0.423614, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418322, recipient metric 0.418322, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418322, recipient metric 0.420252, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418322, recipient metric 0.420252, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_3e-6`, donor metric 0.414417, recipient metric 0.417206, LR 1.01e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_6e-6`, donor metric 0.414417, recipient metric 0.415864, LR 1.13e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_9e-6`, donor metric 0.414417, recipient metric 0.41615, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_14e-6`, donor metric 0.414417, recipient metric 0.414417, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_3e-6`, donor metric 0.406391, recipient metric 0.407453, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_6e-6`, donor metric 0.406391, recipient metric 0.407075, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_9e-6`, donor metric 0.406391, recipient metric 0.406391, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_9e-6` -> `lr_14e-6`, donor metric 0.406391, recipient metric 0.408084, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_3e-6`, donor metric 0.401117, recipient metric 0.403932, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_6e-6`, donor metric 0.401117, recipient metric 0.403821, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_9e-6`, donor metric 0.401117, recipient metric 0.403464, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_14e-6` -> `lr_14e-6`, donor metric 0.401117, recipient metric 0.401117, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_3e-6`, donor metric 0.398526, recipient metric 0.398976, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_6e-6`, donor metric 0.398526, recipient metric 0.401073, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_9e-6`, donor metric 0.398526, recipient metric 0.399351, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_14e-6` -> `lr_14e-6`, donor metric 0.398526, recipient metric 0.398526, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_3e-6`, donor metric 0.395351, recipient metric 0.399019, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_6e-6`, donor metric 0.395351, recipient metric 0.395444, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_9e-6`, donor metric 0.395351, recipient metric 0.395676, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_14e-6` -> `lr_14e-6`, donor metric 0.395351, recipient metric 0.395351, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_3e-6`, donor metric 0.394402, recipient metric 0.395539, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_6e-6`, donor metric 0.394402, recipient metric 0.394402, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_9e-6`, donor metric 0.394402, recipient metric 0.394769, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_14e-6`, donor metric 0.394402, recipient metric 0.394769, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_3e-6`, donor metric 0.390501, recipient metric 0.393495, LR 1.01e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_6e-6`, donor metric 0.390501, recipient metric 0.39201, LR 1.13e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_9e-6`, donor metric 0.390501, recipient metric 0.393329, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_14e-6` -> `lr_14e-6`, donor metric 0.390501, recipient metric 0.390501, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_3e-6`, donor metric 0.388492, recipient metric 0.389612, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_6e-6`, donor metric 0.388492, recipient metric 0.389754, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_9e-6`, donor metric 0.388492, recipient metric 0.393084, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_14e-6` -> `lr_14e-6`, donor metric 0.388492, recipient metric 0.388492, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_3e-6`, donor metric 0.391872, recipient metric 0.391399, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_6e-6`, donor metric 0.391872, recipient metric 0.390735, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_9e-6`, donor metric 0.391872, recipient metric 0.39088, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_14e-6` -> `lr_14e-6`, donor metric 0.391872, recipient metric 0.391872, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_3e-6`, donor metric 0.390574, recipient metric 0.388823, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_6e-6`, donor metric 0.390574, recipient metric 0.390643, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_9e-6`, donor metric 0.390574, recipient metric 0.38989, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_14e-6` -> `lr_14e-6`, donor metric 0.390574, recipient metric 0.390574, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_3e-6`, donor metric 0.392837, recipient metric 0.390289, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_6e-6`, donor metric 0.392837, recipient metric 0.391659, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_9e-6`, donor metric 0.392837, recipient metric 0.393282, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_14e-6` -> `lr_14e-6`, donor metric 0.392837, recipient metric 0.392837, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_3e-6`, donor metric 0.390292, recipient metric 0.3915, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_6e-6`, donor metric 0.390292, recipient metric 0.391773, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_9e-6`, donor metric 0.390292, recipient metric 0.391708, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_14e-6` -> `lr_14e-6`, donor metric 0.390292, recipient metric 0.390292, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_3e-6`, donor metric 0.390371, recipient metric 0.389479, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_6e-6`, donor metric 0.390371, recipient metric 0.393275, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_9e-6`, donor metric 0.390371, recipient metric 0.39116, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_14e-6` -> `lr_14e-6`, donor metric 0.390371, recipient metric 0.390371, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_3e-6`, donor metric 0.388313, recipient metric 0.388941, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_6e-6`, donor metric 0.388313, recipient metric 0.388501, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_9e-6`, donor metric 0.388313, recipient metric 0.390356, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_14e-6` -> `lr_14e-6`, donor metric 0.388313, recipient metric 0.388313, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_3e-6`, donor metric 0.387868, recipient metric 0.38803, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_6e-6`, donor metric 0.387868, recipient metric 0.387868, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_9e-6`, donor metric 0.387868, recipient metric 0.389281, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_14e-6`, donor metric 0.387868, recipient metric 0.388488, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_3e-6`, donor metric 0.387234, recipient metric 0.390289, LR 1.01e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_6e-6`, donor metric 0.387234, recipient metric 0.387839, LR 1.13e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_9e-6`, donor metric 0.387234, recipient metric 0.389813, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_14e-6` -> `lr_14e-6`, donor metric 0.387234, recipient metric 0.387234, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_3e-6`, donor metric 0.386716, recipient metric 0.387416, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_6e-6`, donor metric 0.386716, recipient metric 0.386942, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_9e-6`, donor metric 0.386716, recipient metric 0.38733, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_14e-6` -> `lr_14e-6`, donor metric 0.386716, recipient metric 0.386716, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_3e-6`, donor metric 0.383328, recipient metric 0.383664, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_6e-6`, donor metric 0.383328, recipient metric 0.384371, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_9e-6`, donor metric 0.383328, recipient metric 0.383328, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_9e-6` -> `lr_14e-6`, donor metric 0.383328, recipient metric 0.387433, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_3e-6`, donor metric 0.385188, recipient metric 0.385179, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_6e-6`, donor metric 0.385188, recipient metric 0.38519, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_9e-6`, donor metric 0.385188, recipient metric 0.385188, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_9e-6` -> `lr_14e-6`, donor metric 0.385188, recipient metric 0.384198, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_3e-6`, donor metric 0.383871, recipient metric 0.383621, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_6e-6`, donor metric 0.383871, recipient metric 0.383692, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_9e-6`, donor metric 0.383871, recipient metric 0.383871, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_9e-6` -> `lr_14e-6`, donor metric 0.383871, recipient metric 0.385198, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_3e-6`, donor metric 0.38214, recipient metric 0.384421, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_6e-6`, donor metric 0.38214, recipient metric 0.382885, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_9e-6`, donor metric 0.38214, recipient metric 0.38214, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_9e-6` -> `lr_14e-6`, donor metric 0.38214, recipient metric 0.384402, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_3e-6`, donor metric 0.382936, recipient metric 0.383279, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_6e-6`, donor metric 0.382936, recipient metric 0.383269, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_9e-6`, donor metric 0.382936, recipient metric 0.382936, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_9e-6` -> `lr_14e-6`, donor metric 0.382936, recipient metric 0.385237, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38141, recipient metric 0.38153, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38141, recipient metric 0.38141, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38141, recipient metric 0.382922, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38141, recipient metric 0.384465, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_3e-6`, donor metric 0.381107, recipient metric 0.382261, LR 1.01e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_6e-6`, donor metric 0.381107, recipient metric 0.382635, LR 1.13e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_9e-6`, donor metric 0.381107, recipient metric 0.382142, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_14e-6` -> `lr_14e-6`, donor metric 0.381107, recipient metric 0.381107, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_3e-6`, donor metric 0.378617, recipient metric 0.382824, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_6e-6`, donor metric 0.378617, recipient metric 0.378617, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_9e-6`, donor metric 0.378617, recipient metric 0.382268, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_6e-6` -> `lr_14e-6`, donor metric 0.378617, recipient metric 0.381145, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38056, recipient metric 0.381922, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38056, recipient metric 0.38056, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38056, recipient metric 0.381341, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38056, recipient metric 0.380306, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381508, recipient metric 0.382508, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381508, recipient metric 0.381508, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381508, recipient metric 0.382291, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381508, recipient metric 0.382173, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382339, recipient metric 0.381034, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382339, recipient metric 0.382339, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382339, recipient metric 0.381672, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382339, recipient metric 0.38201, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_3e-6`, donor metric 0.37901, recipient metric 0.3789, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_6e-6`, donor metric 0.37901, recipient metric 0.37901, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_9e-6`, donor metric 0.37901, recipient metric 0.382242, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_6e-6` -> `lr_14e-6`, donor metric 0.37901, recipient metric 0.38336, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_3e-6`, donor metric 0.378275, recipient metric 0.382286, LR 1.01e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_6e-6`, donor metric 0.378275, recipient metric 0.378275, LR 1.13e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_9e-6`, donor metric 0.378275, recipient metric 0.382263, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_6e-6` -> `lr_14e-6`, donor metric 0.378275, recipient metric 0.384022, LR 1.4e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381555, recipient metric 0.381817, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381555, recipient metric 0.381555, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381555, recipient metric 0.381713, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381555, recipient metric 0.381426, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38122, recipient metric 0.380291, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38122, recipient metric 0.38122, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38122, recipient metric 0.381264, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38122, recipient metric 0.380316, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380778, recipient metric 0.380689, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380778, recipient metric 0.380778, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380778, recipient metric 0.378897, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380778, recipient metric 0.383765, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381845, recipient metric 0.378289, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381845, recipient metric 0.381845, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381845, recipient metric 0.378353, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381845, recipient metric 0.383061, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382122, recipient metric 0.379561, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382122, recipient metric 0.382122, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382122, recipient metric 0.379702, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382122, recipient metric 0.384563, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382291, recipient metric 0.383647, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382291, recipient metric 0.382291, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382291, recipient metric 0.383355, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382291, recipient metric 0.382905, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38511, recipient metric 0.384778, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38511, recipient metric 0.38511, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38511, recipient metric 0.385086, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38511, recipient metric 0.383457, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_3e-6`, donor metric 0.383533, recipient metric 0.383851, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_6e-6`, donor metric 0.383533, recipient metric 0.383533, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_9e-6`, donor metric 0.383533, recipient metric 0.38134, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_6e-6` -> `lr_14e-6`, donor metric 0.383533, recipient metric 0.381695, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381939, recipient metric 0.379985, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381939, recipient metric 0.381939, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381939, recipient metric 0.383088, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381939, recipient metric 0.383494, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379574, recipient metric 0.379297, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379574, recipient metric 0.379574, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379574, recipient metric 0.379453, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379574, recipient metric 0.382609, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380925, recipient metric 0.380858, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380925, recipient metric 0.380925, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380925, recipient metric 0.380838, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380925, recipient metric 0.381436, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380685, recipient metric 0.38073, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380685, recipient metric 0.380685, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380685, recipient metric 0.381628, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380685, recipient metric 0.381479, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381337, recipient metric 0.379455, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381337, recipient metric 0.381337, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381337, recipient metric 0.383506, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381337, recipient metric 0.383324, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380518, recipient metric 0.381935, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380518, recipient metric 0.380518, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380518, recipient metric 0.379252, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380518, recipient metric 0.381116, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382404, recipient metric 0.381363, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382404, recipient metric 0.382404, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382404, recipient metric 0.383109, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382404, recipient metric 0.381388, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379965, recipient metric 0.381224, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379965, recipient metric 0.379965, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379965, recipient metric 0.381629, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379965, recipient metric 0.380501, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382211, recipient metric 0.381283, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382211, recipient metric 0.382211, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382211, recipient metric 0.383102, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 48: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382211, recipient metric 0.380953, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379086, recipient metric 0.381615, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379086, recipient metric 0.379086, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379086, recipient metric 0.381835, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 49: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379086, recipient metric 0.383008, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_3e-6`, donor metric 0.383426, recipient metric 0.382539, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_6e-6`, donor metric 0.383426, recipient metric 0.383426, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_9e-6`, donor metric 0.383426, recipient metric 0.380384, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 50: `lr_6e-6` -> `lr_14e-6`, donor metric 0.383426, recipient metric 0.382957, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38197, recipient metric 0.381506, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38197, recipient metric 0.38197, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38197, recipient metric 0.381378, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 51: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38197, recipient metric 0.381409, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382554, recipient metric 0.380546, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382554, recipient metric 0.382554, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382554, recipient metric 0.381995, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 52: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382554, recipient metric 0.382612, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38117, recipient metric 0.381366, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38117, recipient metric 0.38117, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38117, recipient metric 0.37881, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 53: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38117, recipient metric 0.381427, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381415, recipient metric 0.381029, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381415, recipient metric 0.381415, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381415, recipient metric 0.381273, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 54: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381415, recipient metric 0.380748, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382837, recipient metric 0.383048, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382837, recipient metric 0.382837, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382837, recipient metric 0.380042, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 55: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382837, recipient metric 0.380746, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380835, recipient metric 0.380623, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380835, recipient metric 0.380835, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380835, recipient metric 0.380917, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 56: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380835, recipient metric 0.378519, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382268, recipient metric 0.382833, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382268, recipient metric 0.382268, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382268, recipient metric 0.381129, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 57: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382268, recipient metric 0.382433, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381892, recipient metric 0.381299, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381892, recipient metric 0.381892, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381892, recipient metric 0.381232, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 58: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381892, recipient metric 0.381433, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380783, recipient metric 0.381753, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380783, recipient metric 0.380783, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380783, recipient metric 0.379513, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 59: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380783, recipient metric 0.37926, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382263, recipient metric 0.379886, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382263, recipient metric 0.382263, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382263, recipient metric 0.381508, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 60: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382263, recipient metric 0.379999, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_3e-6`, donor metric 0.384569, recipient metric 0.382262, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_6e-6`, donor metric 0.384569, recipient metric 0.384569, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_9e-6`, donor metric 0.384569, recipient metric 0.38096, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 61: `lr_6e-6` -> `lr_14e-6`, donor metric 0.384569, recipient metric 0.382377, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382186, recipient metric 0.383219, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382186, recipient metric 0.382186, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382186, recipient metric 0.379439, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 62: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382186, recipient metric 0.379054, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382176, recipient metric 0.38051, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382176, recipient metric 0.382176, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382176, recipient metric 0.383345, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 63: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382176, recipient metric 0.380369, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380054, recipient metric 0.379835, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380054, recipient metric 0.380054, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380054, recipient metric 0.381674, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 64: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380054, recipient metric 0.381225, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379768, recipient metric 0.381738, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379768, recipient metric 0.379768, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379768, recipient metric 0.38129, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 65: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379768, recipient metric 0.380921, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381596, recipient metric 0.381675, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381596, recipient metric 0.381596, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381596, recipient metric 0.381573, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 66: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381596, recipient metric 0.381812, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379483, recipient metric 0.379393, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379483, recipient metric 0.379483, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379483, recipient metric 0.38216, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 67: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379483, recipient metric 0.379458, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380489, recipient metric 0.379796, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380489, recipient metric 0.380489, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380489, recipient metric 0.380229, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 68: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380489, recipient metric 0.38342, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38079, recipient metric 0.380682, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38079, recipient metric 0.38079, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38079, recipient metric 0.383145, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 69: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38079, recipient metric 0.381394, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381455, recipient metric 0.379723, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381455, recipient metric 0.381455, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381455, recipient metric 0.381676, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 70: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381455, recipient metric 0.379057, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382133, recipient metric 0.381355, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382133, recipient metric 0.382133, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382133, recipient metric 0.380828, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 71: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382133, recipient metric 0.382208, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380258, recipient metric 0.380973, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380258, recipient metric 0.380258, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380258, recipient metric 0.380752, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 72: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380258, recipient metric 0.380675, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380151, recipient metric 0.3805, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380151, recipient metric 0.380151, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380151, recipient metric 0.379901, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 73: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380151, recipient metric 0.380395, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_3e-6`, donor metric 0.38363, recipient metric 0.380985, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_6e-6`, donor metric 0.38363, recipient metric 0.38363, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_9e-6`, donor metric 0.38363, recipient metric 0.380481, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 74: `lr_6e-6` -> `lr_14e-6`, donor metric 0.38363, recipient metric 0.38271, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380738, recipient metric 0.380618, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380738, recipient metric 0.380738, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380738, recipient metric 0.382483, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 75: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380738, recipient metric 0.380298, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_3e-6`, donor metric 0.37901, recipient metric 0.382321, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_6e-6`, donor metric 0.37901, recipient metric 0.37901, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_9e-6`, donor metric 0.37901, recipient metric 0.379631, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 76: `lr_6e-6` -> `lr_14e-6`, donor metric 0.37901, recipient metric 0.382082, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380453, recipient metric 0.379954, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380453, recipient metric 0.380453, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380453, recipient metric 0.382057, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 77: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380453, recipient metric 0.380853, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_3e-6`, donor metric 0.383911, recipient metric 0.381816, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_6e-6`, donor metric 0.383911, recipient metric 0.383911, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_9e-6`, donor metric 0.383911, recipient metric 0.382182, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 78: `lr_6e-6` -> `lr_14e-6`, donor metric 0.383911, recipient metric 0.379496, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382435, recipient metric 0.382661, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382435, recipient metric 0.382435, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382435, recipient metric 0.381667, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 79: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382435, recipient metric 0.382861, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_3e-6`, donor metric 0.378569, recipient metric 0.379862, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_6e-6`, donor metric 0.378569, recipient metric 0.378569, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_9e-6`, donor metric 0.378569, recipient metric 0.380309, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 80: `lr_6e-6` -> `lr_14e-6`, donor metric 0.378569, recipient metric 0.378429, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379446, recipient metric 0.382278, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379446, recipient metric 0.379446, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379446, recipient metric 0.379595, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 81: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379446, recipient metric 0.379793, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382352, recipient metric 0.379696, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382352, recipient metric 0.382352, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382352, recipient metric 0.38138, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 82: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382352, recipient metric 0.380807, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379932, recipient metric 0.380274, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379932, recipient metric 0.379932, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379932, recipient metric 0.382315, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 83: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379932, recipient metric 0.379965, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381892, recipient metric 0.381331, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381892, recipient metric 0.381892, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381892, recipient metric 0.381699, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 84: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381892, recipient metric 0.379363, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382026, recipient metric 0.382137, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382026, recipient metric 0.382026, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382026, recipient metric 0.381868, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 85: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382026, recipient metric 0.381976, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381706, recipient metric 0.383554, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381706, recipient metric 0.381706, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381706, recipient metric 0.379327, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 86: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381706, recipient metric 0.380161, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379304, recipient metric 0.380538, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379304, recipient metric 0.379304, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379304, recipient metric 0.38067, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 87: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379304, recipient metric 0.379181, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379009, recipient metric 0.381233, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379009, recipient metric 0.379009, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379009, recipient metric 0.381367, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 88: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379009, recipient metric 0.380421, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_3e-6`, donor metric 0.383074, recipient metric 0.379623, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_6e-6`, donor metric 0.383074, recipient metric 0.383074, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_9e-6`, donor metric 0.383074, recipient metric 0.379611, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 89: `lr_6e-6` -> `lr_14e-6`, donor metric 0.383074, recipient metric 0.382994, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381841, recipient metric 0.381896, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381841, recipient metric 0.381841, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381841, recipient metric 0.381896, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 90: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381841, recipient metric 0.382048, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380663, recipient metric 0.380746, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380663, recipient metric 0.380663, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380663, recipient metric 0.381552, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 91: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380663, recipient metric 0.379576, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_3e-6`, donor metric 0.383368, recipient metric 0.380715, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_6e-6`, donor metric 0.383368, recipient metric 0.383368, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_9e-6`, donor metric 0.383368, recipient metric 0.381065, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 92: `lr_6e-6` -> `lr_14e-6`, donor metric 0.383368, recipient metric 0.379698, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_3e-6`, donor metric 0.379987, recipient metric 0.38183, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_6e-6`, donor metric 0.379987, recipient metric 0.379987, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_9e-6`, donor metric 0.379987, recipient metric 0.379926, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 93: `lr_6e-6` -> `lr_14e-6`, donor metric 0.379987, recipient metric 0.379974, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_3e-6`, donor metric 0.378609, recipient metric 0.379586, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_6e-6`, donor metric 0.378609, recipient metric 0.378609, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_9e-6`, donor metric 0.378609, recipient metric 0.382126, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 94: `lr_6e-6` -> `lr_14e-6`, donor metric 0.378609, recipient metric 0.380972, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380156, recipient metric 0.380006, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380156, recipient metric 0.380156, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380156, recipient metric 0.379389, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 95: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380156, recipient metric 0.381432, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381447, recipient metric 0.380604, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381447, recipient metric 0.381447, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381447, recipient metric 0.381486, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 96: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381447, recipient metric 0.380367, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_3e-6`, donor metric 0.381412, recipient metric 0.382805, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_6e-6`, donor metric 0.381412, recipient metric 0.381412, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_9e-6`, donor metric 0.381412, recipient metric 0.383626, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 97: `lr_6e-6` -> `lr_14e-6`, donor metric 0.381412, recipient metric 0.382274, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_3e-6`, donor metric 0.380868, recipient metric 0.381912, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_6e-6`, donor metric 0.380868, recipient metric 0.380868, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_9e-6`, donor metric 0.380868, recipient metric 0.381827, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 98: `lr_6e-6` -> `lr_14e-6`, donor metric 0.380868, recipient metric 0.383738, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_3e-6`, donor metric 0.382754, recipient metric 0.383057, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_6e-6`, donor metric 0.382754, recipient metric 0.382754, LR 1.02e-05 -> 1.02e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_9e-6`, donor metric 0.382754, recipient metric 0.380412, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 99: `lr_6e-6` -> `lr_14e-6`, donor metric 0.382754, recipient metric 0.379872, LR 1.36e-05 -> 1.36e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- [Skipped exploits (significance gating)](plots/report/skipped_exploits.csv) -- 0 donor->recipient replacement(s) declined for insufficient significance

## Method
- Method: `anchor_copy_lr_recenter`
- Population: 4 trials
- Training interval: 120000 samples/trial chunk (1x samples_per_epoch)
- Evaluation interval: every 1 training chunk(s), 3000000 validation samples
- Exploit interval: every 1 training chunk(s)
- Exploit significance gating: disabled (nominal rank order only)
- Burn-in: 0 generation(s) (observe-only, no exploit/controller LR action applied)
- Monitor-tier cadence: disabled generation(s), all population members, read-only
- Full-tier cadence: 20 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `d87fbe39f71b00f616fbe550e95f97942636b66f`
- Git dirty: `False`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_100gen_1mval.yaml', '--slots', 'iutgpu01:4,iutgpu01:5,iutgpu01:6,iutgpu01:7', '--experiment-name', 'anchor_copy_lr_recenter_100gen_1mval_20260816_105629']`
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
