# anchor_copy_lr_recenter_8h_20260807_015714

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
- Final checkpoint controller objective: 1.01644 by `lr_6e-6`
- Global best configured metric: 0.408278 by `lr_14e-6`
- Delta vs measured baseline: n/a%
- Best checkpoint: `/data/suehara/part/march/runs/pbt/anchor_copy_lr_recenter_8h_20260807_015714/checkpoints/global_best_state.pt`

## Final Physics Performance
- [Background efficiency curves](plots/diagnostics/background_efficiency_curves.png)
- Checkpoint: **global best (PBT selection)** (`lr_14e-6`, generation 26), selection metric: `validation_total_reference_mistag_geomean_percent` (min)
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
- Population-wide, generation-controlled correlation (log10 LR vs. total_mistag_score, detrended by each generation's median): n=192, Pearson r=-0.162, Spearman rho=-0.181
- Detrending removes the ordinary training-progress trend (score improves over generations regardless of LR) so this number isolates an LR effect, not a training-progress effect mistaken for one. Sign convention: positive means higher LR associates with a worse-than-typical (for that generation) score; negative means better-than-typical. Not a causal claim.

## Proxy Validation
- [Proxy validation](plots/proxy_validation.png)
- control vs. monitor correlation: n=0 paired observations -- too few for a meaningful correlation
- control vs. full_holdout (independent, excludes control+monitor) correlation: n=12, Pearson r=0.170, Spearman rho=0.140
- Best checkpoint by tier: control: `lr_9e-6` gen 31 (0.412861), full_holdout: `lr_9e-6` gen 31 (0.40053)
- Best-checkpoint agreement across tiers: AGREE
- Control-selected global best has not been evaluated on monitor/full yet.
- Corroboration status: **provisional**
  - monitor: not available (baseline or selected checkpoint not evaluated on this tier)
  - full: not available (baseline or selected checkpoint not evaluated on this tier)
- No proxy-overfitting cases detected (control improved while monitor did not) in the paired generations evaluated so far.

## Model Selection Scores
- Final generation: 47
- All mistag/score values in percent (lower is better); status marks the generation's winner and/or the persisted anchor member.

| member | bc@0.8 | bd@0.8 | bc@0.9 | bd@0.9 | cb@0.5 | cd@0.5 | cb@0.8 | cd@0.8 | ctag_score | btag_score | total_mistag_score | LR | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr_14e-6 | 0.1846 | 0.07616 | 3.249 | 0.2044 | 0.5324 | 0.06614 | 2.747 | 1.168 | 0.5798 | 0.3108 | 0.4245 | 1.21e-05 | anchor |
| lr_3e-6 | 0.1785 | 0.08021 | 3.182 | 0.2045 | 0.5345 | 0.06618 | 2.725 | 1.139 | 0.5756 | 0.3107 | 0.4229 | 8.06e-06 | - |
| lr_6e-6 | 0.1723 | 0.07832 | 3.16 | 0.2048 | 0.5341 | 0.06828 | 2.757 | 1.157 | 0.5839 | 0.3057 | 0.4225 | 9.07e-06 | winner |
| lr_9e-6 | 0.1846 | 0.07414 | 3.234 | 0.2044 | 0.5326 | 0.06813 | 2.745 | 1.192 | 0.587 | 0.3084 | 0.4255 | 1.01e-05 | - |

## PBT Decision Summary (anchor_copy_lr_recenter)
- `total_mistag_score` is this strategy's ranking metric; ctag_score/btag_score are shown for diagnosis only.

| generation | winner | winner total_mistag_score | winner ctag_score | winner btag_score | winner LR | previous LR center | new LR center | decision | spread_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lr_14e-6 | 0.4461 | 0.613 | 0.3247 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 1 | lr_9e-6 | 0.4412 | 0.6126 | 0.3177 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 2 | lr_14e-6 | 0.4213 | 0.5817 | 0.3052 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 3 | lr_6e-6 | 0.4164 | 0.5716 | 0.3033 | 1.26e-05 | 1.4e-05 | 1.26e-05 | accepted_new_anchor | no |
| 4 | lr_3e-6 | 0.4229 | 0.5876 | 0.3043 | 1.008e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 5 | lr_9e-6 | 0.4205 | 0.5844 | 0.3027 | 1.26e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 6 | lr_14e-6 | 0.414 | 0.5799 | 0.2956 | 1.4e-05 | 1.26e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 7 | lr_14e-6 | 0.421 | 0.5788 | 0.3063 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 8 | lr_14e-6 | 0.4252 | 0.5863 | 0.3084 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 9 | lr_14e-6 | 0.4162 | 0.5813 | 0.2979 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 10 | lr_9e-6 | 0.4241 | 0.5745 | 0.3131 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 11 | lr_9e-6 | 0.423 | 0.58 | 0.3086 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 12 | lr_9e-6 | 0.4148 | 0.5918 | 0.2908 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 13 | lr_9e-6 | 0.4172 | 0.5819 | 0.2991 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 14 | lr_9e-6 | 0.4168 | 0.5796 | 0.2997 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 15 | lr_3e-6 | 0.4189 | 0.5829 | 0.301 | 1.12e-05 | 1.4e-05 | 1.12e-05 | reused_previous_anchor | no |
| 16 | lr_6e-6 | 0.4233 | 0.5771 | 0.3105 | 1.008e-05 | 1.12e-05 | 1.12e-05 | rewound_to_previous_anchor | no |
| 17 | lr_9e-6 | 0.4182 | 0.5722 | 0.3057 | 1.12e-05 | 1.12e-05 | 1.12e-05 | reused_previous_anchor | no |
| 18 | lr_3e-6 | 0.4227 | 0.5807 | 0.3077 | 8.96e-06 | 1.12e-05 | 1.12e-05 | rewound_to_previous_anchor | no |
| 19 | lr_14e-6 | 0.413 | 0.5718 | 0.2983 | 1.344e-05 | 1.12e-05 | 1.344e-05 | reused_previous_anchor | no |
| 20 | lr_6e-6 | 0.4218 | 0.5909 | 0.3011 | 1.21e-05 | 1.344e-05 | 1.344e-05 | rewound_to_previous_anchor | no |
| 21 | lr_3e-6 | 0.4244 | 0.5677 | 0.3173 | 1.075e-05 | 1.344e-05 | 1.344e-05 | rewound_to_previous_anchor | no |
| 22 | lr_14e-6 | 0.4159 | 0.5855 | 0.2955 | 1.4e-05 | 1.344e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 23 | lr_9e-6 | 0.4155 | 0.5782 | 0.2985 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 24 | lr_14e-6 | 0.4174 | 0.5791 | 0.3009 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 25 | lr_14e-6 | 0.4197 | 0.5844 | 0.3014 | 1.4e-05 | 1.4e-05 | 1.4e-05 | reused_previous_anchor | yes |
| 26 | lr_14e-6 | 0.4083 | 0.5731 | 0.2909 | 1.4e-05 | 1.4e-05 | 1.4e-05 | accepted_new_anchor | yes |
| 27 | lr_6e-6 | 0.4147 | 0.5839 | 0.2945 | 1.26e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 28 | lr_3e-6 | 0.4157 | 0.5627 | 0.307 | 1.12e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 29 | lr_14e-6 | 0.4131 | 0.5756 | 0.2964 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 30 | lr_9e-6 | 0.4163 | 0.5729 | 0.3025 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 31 | lr_9e-6 | 0.4129 | 0.5538 | 0.3078 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 32 | lr_14e-6 | 0.4159 | 0.5766 | 0.2999 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 33 | lr_9e-6 | 0.418 | 0.5851 | 0.2986 | 1.4e-05 | 1.4e-05 | 1.4e-05 | rewound_to_previous_anchor | yes |
| 34 | lr_6e-6 | 0.4121 | 0.5695 | 0.2982 | 1.26e-05 | 1.4e-05 | 1.26e-05 | reused_previous_anchor | no |
| 35 | lr_14e-6 | 0.4227 | 0.5841 | 0.3058 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 36 | lr_14e-6 | 0.4188 | 0.587 | 0.2988 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 37 | lr_6e-6 | 0.4191 | 0.5766 | 0.3046 | 1.134e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 38 | lr_14e-6 | 0.4143 | 0.5686 | 0.3019 | 1.4e-05 | 1.26e-05 | 1.26e-05 | rewound_to_previous_anchor | no |
| 39 | lr_3e-6 | 0.4103 | 0.5879 | 0.2864 | 1.008e-05 | 1.26e-05 | 1.008e-05 | reused_previous_anchor | no |
| 40 | lr_9e-6 | 0.4281 | 0.5939 | 0.3086 | 1.008e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 41 | lr_14e-6 | 0.4218 | 0.5953 | 0.2989 | 1.21e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 42 | lr_3e-6 | 0.4216 | 0.5802 | 0.3064 | 8.064e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 43 | lr_6e-6 | 0.419 | 0.5758 | 0.3049 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 44 | lr_14e-6 | 0.4151 | 0.5682 | 0.3033 | 1.21e-05 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 45 | lr_6e-6 | 0.4207 | 0.5852 | 0.3025 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 46 | lr_3e-6 | 0.4152 | 0.5698 | 0.3025 | 8.064e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |
| 47 | lr_6e-6 | 0.4225 | 0.5839 | 0.3057 | 9.072e-06 | 1.008e-05 | 1.008e-05 | rewound_to_previous_anchor | no |

## Exploit History
- [Exploit table](plots/report/exploit_table.csv)
- generation 0: `lr_14e-6` -> `lr_3e-6`, donor metric 0.446135, recipient metric 0.459279, LR 3e-06 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_6e-6`, donor metric 0.446135, recipient metric 0.457009, LR 6e-06 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_9e-6`, donor metric 0.446135, recipient metric 0.451044, LR 9e-06 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 0: `lr_14e-6` -> `lr_14e-6`, donor metric 0.446135, recipient metric 0.446135, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_3e-6`, donor metric 0.441189, recipient metric 0.450815, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_6e-6`, donor metric 0.441189, recipient metric 0.449268, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_9e-6`, donor metric 0.441189, recipient metric 0.441189, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 1: `lr_9e-6` -> `lr_14e-6`, donor metric 0.441189, recipient metric 0.450377, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_3e-6`, donor metric 0.421314, recipient metric 0.421826, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_6e-6`, donor metric 0.421314, recipient metric 0.445467, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_9e-6`, donor metric 0.421314, recipient metric 0.43752, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 2: `lr_14e-6` -> `lr_14e-6`, donor metric 0.421314, recipient metric 0.421314, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_3e-6`, donor metric 0.416359, recipient metric 0.442387, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_6e-6`, donor metric 0.416359, recipient metric 0.416359, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_9e-6`, donor metric 0.416359, recipient metric 0.440283, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 3: `lr_6e-6` -> `lr_14e-6`, donor metric 0.416359, recipient metric 0.417413, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_3e-6`, donor metric 0.427923, recipient metric 0.422852, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_6e-6`, donor metric 0.427923, recipient metric 0.427923, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_9e-6`, donor metric 0.427923, recipient metric 0.42875, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 4: `lr_6e-6` -> `lr_14e-6`, donor metric 0.427923, recipient metric 0.426483, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_3e-6`, donor metric 0.424349, recipient metric 0.430526, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_6e-6`, donor metric 0.424349, recipient metric 0.424349, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_9e-6`, donor metric 0.424349, recipient metric 0.420549, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 5: `lr_6e-6` -> `lr_14e-6`, donor metric 0.424349, recipient metric 0.424514, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_3e-6`, donor metric 0.424455, recipient metric 0.425848, LR 1.01e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_6e-6`, donor metric 0.424455, recipient metric 0.424455, LR 1.13e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_9e-6`, donor metric 0.424455, recipient metric 0.425291, LR 1.26e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 6: `lr_6e-6` -> `lr_14e-6`, donor metric 0.424455, recipient metric 0.413983, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_3e-6`, donor metric 0.429784, recipient metric 0.43145, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_6e-6`, donor metric 0.429784, recipient metric 0.429784, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_9e-6`, donor metric 0.429784, recipient metric 0.429988, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 7: `lr_6e-6` -> `lr_14e-6`, donor metric 0.429784, recipient metric 0.421033, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_3e-6`, donor metric 0.428128, recipient metric 0.431511, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_6e-6`, donor metric 0.428128, recipient metric 0.428128, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_9e-6`, donor metric 0.428128, recipient metric 0.431318, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 8: `lr_6e-6` -> `lr_14e-6`, donor metric 0.428128, recipient metric 0.425193, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_3e-6`, donor metric 0.42011, recipient metric 0.422083, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_6e-6`, donor metric 0.42011, recipient metric 0.42011, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_9e-6`, donor metric 0.42011, recipient metric 0.428838, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 9: `lr_6e-6` -> `lr_14e-6`, donor metric 0.42011, recipient metric 0.416154, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_3e-6`, donor metric 0.424964, recipient metric 0.426688, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_6e-6`, donor metric 0.424964, recipient metric 0.424964, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_9e-6`, donor metric 0.424964, recipient metric 0.424142, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 10: `lr_6e-6` -> `lr_14e-6`, donor metric 0.424964, recipient metric 0.427041, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_3e-6`, donor metric 0.424181, recipient metric 0.43543, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_6e-6`, donor metric 0.424181, recipient metric 0.424181, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_9e-6`, donor metric 0.424181, recipient metric 0.423038, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 11: `lr_6e-6` -> `lr_14e-6`, donor metric 0.424181, recipient metric 0.425552, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418271, recipient metric 0.417829, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418271, recipient metric 0.418271, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418271, recipient metric 0.414845, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 12: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418271, recipient metric 0.414845, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_3e-6`, donor metric 0.42896, recipient metric 0.417987, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_6e-6`, donor metric 0.42896, recipient metric 0.42896, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_9e-6`, donor metric 0.42896, recipient metric 0.417175, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 13: `lr_6e-6` -> `lr_14e-6`, donor metric 0.42896, recipient metric 0.417175, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418685, recipient metric 0.420018, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418685, recipient metric 0.418685, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418685, recipient metric 0.41676, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 14: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418685, recipient metric 0.41676, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_6e-6` -> `lr_3e-6`, donor metric 0.419092, recipient metric 0.418862, LR 1.12e-05 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_6e-6` -> `lr_6e-6`, donor metric 0.419092, recipient metric 0.419092, LR 1.26e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_6e-6` -> `lr_9e-6`, donor metric 0.419092, recipient metric 0.422478, LR 1.4e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 15: `lr_6e-6` -> `lr_14e-6`, donor metric 0.419092, recipient metric 0.428814, LR 1.4e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_3e-6`, donor metric 0.423323, recipient metric 0.424736, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_6e-6`, donor metric 0.423323, recipient metric 0.423323, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_9e-6`, donor metric 0.423323, recipient metric 0.42467, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 16: `lr_6e-6` -> `lr_14e-6`, donor metric 0.423323, recipient metric 0.424262, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_6e-6` -> `lr_3e-6`, donor metric 0.423445, recipient metric 0.422389, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_6e-6` -> `lr_6e-6`, donor metric 0.423445, recipient metric 0.423445, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_6e-6` -> `lr_9e-6`, donor metric 0.423445, recipient metric 0.418215, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 17: `lr_6e-6` -> `lr_14e-6`, donor metric 0.423445, recipient metric 0.419749, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_6e-6` -> `lr_3e-6`, donor metric 0.424823, recipient metric 0.42268, LR 8.96e-06 -> 8.96e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_6e-6` -> `lr_6e-6`, donor metric 0.424823, recipient metric 0.424823, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_6e-6` -> `lr_9e-6`, donor metric 0.424823, recipient metric 0.43007, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 18: `lr_6e-6` -> `lr_14e-6`, donor metric 0.424823, recipient metric 0.425946, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_3e-6`, donor metric 0.42224, recipient metric 0.424984, LR 8.96e-06 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_6e-6`, donor metric 0.42224, recipient metric 0.42224, LR 1.01e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_9e-6`, donor metric 0.42224, recipient metric 0.422474, LR 1.12e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 19: `lr_6e-6` -> `lr_14e-6`, donor metric 0.42224, recipient metric 0.413012, LR 1.34e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_3e-6`, donor metric 0.421774, recipient metric 0.43714, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_6e-6`, donor metric 0.421774, recipient metric 0.421774, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_9e-6`, donor metric 0.421774, recipient metric 0.43422, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 20: `lr_6e-6` -> `lr_14e-6`, donor metric 0.421774, recipient metric 0.425614, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_3e-6`, donor metric 0.436484, recipient metric 0.424395, LR 1.08e-05 -> 1.08e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_6e-6`, donor metric 0.436484, recipient metric 0.436484, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_9e-6`, donor metric 0.436484, recipient metric 0.425717, LR 1.34e-05 -> 1.34e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 21: `lr_6e-6` -> `lr_14e-6`, donor metric 0.436484, recipient metric 0.42594, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_3e-6`, donor metric 0.417269, recipient metric 0.417718, LR 1.08e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_6e-6`, donor metric 0.417269, recipient metric 0.417269, LR 1.21e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_9e-6`, donor metric 0.417269, recipient metric 0.416661, LR 1.34e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 22: `lr_6e-6` -> `lr_14e-6`, donor metric 0.417269, recipient metric 0.415946, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_3e-6`, donor metric 0.418253, recipient metric 0.42539, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_6e-6`, donor metric 0.418253, recipient metric 0.418253, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_9e-6`, donor metric 0.418253, recipient metric 0.415462, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 23: `lr_6e-6` -> `lr_14e-6`, donor metric 0.418253, recipient metric 0.435312, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_3e-6`, donor metric 0.433632, recipient metric 0.432029, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_6e-6`, donor metric 0.433632, recipient metric 0.433632, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_9e-6`, donor metric 0.433632, recipient metric 0.419795, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 24: `lr_6e-6` -> `lr_14e-6`, donor metric 0.433632, recipient metric 0.417427, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_3e-6`, donor metric 0.425235, recipient metric 0.421445, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_6e-6`, donor metric 0.425235, recipient metric 0.425235, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_9e-6`, donor metric 0.425235, recipient metric 0.424011, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 25: `lr_6e-6` -> `lr_14e-6`, donor metric 0.425235, recipient metric 0.419658, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_3e-6`, donor metric 0.408278, recipient metric 0.413732, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_6e-6`, donor metric 0.408278, recipient metric 0.428305, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_9e-6`, donor metric 0.408278, recipient metric 0.423503, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 26: `lr_14e-6` -> `lr_14e-6`, donor metric 0.408278, recipient metric 0.408278, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_3e-6`, donor metric 0.4286, recipient metric 0.422585, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_6e-6`, donor metric 0.4286, recipient metric 0.414687, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_9e-6`, donor metric 0.4286, recipient metric 0.42776, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 27: `lr_14e-6` -> `lr_14e-6`, donor metric 0.4286, recipient metric 0.4286, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_3e-6`, donor metric 0.423665, recipient metric 0.415676, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_6e-6`, donor metric 0.423665, recipient metric 0.426193, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_9e-6`, donor metric 0.423665, recipient metric 0.415743, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 28: `lr_14e-6` -> `lr_14e-6`, donor metric 0.423665, recipient metric 0.423665, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_3e-6`, donor metric 0.413068, recipient metric 0.415455, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_6e-6`, donor metric 0.413068, recipient metric 0.419362, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_9e-6`, donor metric 0.413068, recipient metric 0.424049, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 29: `lr_14e-6` -> `lr_14e-6`, donor metric 0.413068, recipient metric 0.413068, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_3e-6`, donor metric 0.421797, recipient metric 0.427304, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_6e-6`, donor metric 0.421797, recipient metric 0.422409, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_9e-6`, donor metric 0.421797, recipient metric 0.416269, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 30: `lr_14e-6` -> `lr_14e-6`, donor metric 0.421797, recipient metric 0.421797, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_3e-6`, donor metric 0.427866, recipient metric 0.413701, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_6e-6`, donor metric 0.427866, recipient metric 0.414008, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_9e-6`, donor metric 0.427866, recipient metric 0.412861, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 31: `lr_14e-6` -> `lr_14e-6`, donor metric 0.427866, recipient metric 0.427866, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415889, recipient metric 0.425241, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415889, recipient metric 0.415993, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415889, recipient metric 0.420534, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 32: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415889, recipient metric 0.415889, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_3e-6`, donor metric 0.430266, recipient metric 0.429806, LR 1.12e-05 -> 1.12e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_6e-6`, donor metric 0.430266, recipient metric 0.429845, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_9e-6`, donor metric 0.430266, recipient metric 0.41796, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 33: `lr_14e-6` -> `lr_14e-6`, donor metric 0.430266, recipient metric 0.430266, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_3e-6`, donor metric 0.423225, recipient metric 0.423125, LR 1.12e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_6e-6`, donor metric 0.423225, recipient metric 0.412124, LR 1.26e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_9e-6`, donor metric 0.423225, recipient metric 0.414662, LR 1.4e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 34: `lr_14e-6` -> `lr_14e-6`, donor metric 0.423225, recipient metric 0.423225, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_3e-6`, donor metric 0.422665, recipient metric 0.429026, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_6e-6`, donor metric 0.422665, recipient metric 0.429452, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_9e-6`, donor metric 0.422665, recipient metric 0.429386, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 35: `lr_14e-6` -> `lr_14e-6`, donor metric 0.422665, recipient metric 0.422665, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_3e-6`, donor metric 0.418755, recipient metric 0.429758, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_6e-6`, donor metric 0.418755, recipient metric 0.419591, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_9e-6`, donor metric 0.418755, recipient metric 0.425888, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 36: `lr_14e-6` -> `lr_14e-6`, donor metric 0.418755, recipient metric 0.418755, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_3e-6`, donor metric 0.4203, recipient metric 0.426243, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_6e-6`, donor metric 0.4203, recipient metric 0.419054, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_9e-6`, donor metric 0.4203, recipient metric 0.419556, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 37: `lr_14e-6` -> `lr_14e-6`, donor metric 0.4203, recipient metric 0.4203, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_3e-6`, donor metric 0.414336, recipient metric 0.416592, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_6e-6`, donor metric 0.414336, recipient metric 0.424497, LR 1.13e-05 -> 1.13e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_9e-6`, donor metric 0.414336, recipient metric 0.424297, LR 1.26e-05 -> 1.26e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 38: `lr_14e-6` -> `lr_14e-6`, donor metric 0.414336, recipient metric 0.414336, LR 1.4e-05 -> 1.4e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_3e-6`, donor metric 0.411175, recipient metric 0.410299, LR 1.01e-05 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_6e-6`, donor metric 0.411175, recipient metric 0.413061, LR 1.13e-05 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_9e-6`, donor metric 0.411175, recipient metric 0.422126, LR 1.26e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 39: `lr_14e-6` -> `lr_14e-6`, donor metric 0.411175, recipient metric 0.411175, LR 1.4e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_3e-6`, donor metric 0.430509, recipient metric 0.429949, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_6e-6`, donor metric 0.430509, recipient metric 0.428635, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_9e-6`, donor metric 0.430509, recipient metric 0.428129, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 40: `lr_14e-6` -> `lr_14e-6`, donor metric 0.430509, recipient metric 0.430509, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_3e-6`, donor metric 0.421815, recipient metric 0.423886, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_6e-6`, donor metric 0.421815, recipient metric 0.426584, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_9e-6`, donor metric 0.421815, recipient metric 0.425539, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 41: `lr_14e-6` -> `lr_14e-6`, donor metric 0.421815, recipient metric 0.421815, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_3e-6`, donor metric 0.423468, recipient metric 0.421614, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_6e-6`, donor metric 0.423468, recipient metric 0.422882, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_9e-6`, donor metric 0.423468, recipient metric 0.431314, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 42: `lr_14e-6` -> `lr_14e-6`, donor metric 0.423468, recipient metric 0.423468, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_3e-6`, donor metric 0.41903, recipient metric 0.422393, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_6e-6`, donor metric 0.41903, recipient metric 0.418996, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_9e-6`, donor metric 0.41903, recipient metric 0.43244, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 43: `lr_14e-6` -> `lr_14e-6`, donor metric 0.41903, recipient metric 0.41903, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_3e-6`, donor metric 0.415136, recipient metric 0.421839, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_6e-6`, donor metric 0.415136, recipient metric 0.415293, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_9e-6`, donor metric 0.415136, recipient metric 0.429333, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 44: `lr_14e-6` -> `lr_14e-6`, donor metric 0.415136, recipient metric 0.415136, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_3e-6`, donor metric 0.424824, recipient metric 0.43316, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_6e-6`, donor metric 0.424824, recipient metric 0.420717, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_9e-6`, donor metric 0.424824, recipient metric 0.420723, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 45: `lr_14e-6` -> `lr_14e-6`, donor metric 0.424824, recipient metric 0.424824, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_3e-6`, donor metric 0.419901, recipient metric 0.415201, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_6e-6`, donor metric 0.419901, recipient metric 0.415234, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_9e-6`, donor metric 0.419901, recipient metric 0.419954, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 46: `lr_14e-6` -> `lr_14e-6`, donor metric 0.419901, recipient metric 0.419901, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_3e-6`, donor metric 0.424534, recipient metric 0.42289, LR 8.06e-06 -> 8.06e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_6e-6`, donor metric 0.424534, recipient metric 0.422527, LR 9.07e-06 -> 9.07e-06, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_9e-6`, donor metric 0.424534, recipient metric 0.425491, LR 1.01e-05 -> 1.01e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
- generation 47: `lr_14e-6` -> `lr_14e-6`, donor metric 0.424534, recipient metric 0.424534, LR 1.21e-05 -> 1.21e-05, mutation `None`, weight `anchor_copy_lr_recenter`, optimizer `anchor_copy_lr_recenter`
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
- Full-tier cadence: 16 generation(s), all population members, read-only

## Provenance
- Starting checkpoint: `/data/suehara/part/march-worktrees/train_1eb2c79/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_epoch-17_state.pt`
- Git commit: `6b3ca7f3234ece7b12580231b64102ce712c9611`
- Git dirty: `True`
- Launch command: `['/data/suehara/part/march/.venv/bin/python', 'scripts/training/run_pbt.py', '--config', 'configs/experiments/anchor_copy_lr_recenter_8h.yaml', '--slots', 'iutgpu01:0,iutgpu01:1,iutgpu01:2,iutgpu01:3', '--experiment-name', 'anchor_copy_lr_recenter_8h_20260807_015714']`
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
- No data-loader shutdown-race warnings observed across 216 evaluation(s).
