# PBT v2 — 100 epochs

Verified continuation of the frozen windowed PBT v2 population through 100 full epochs.

The horizon is 20 PBT generations × 5 training epochs = 100 epochs. Selection
and any copy/LR mutation happen after each five-epoch generation; only actual
copy/mutation events are marked in the presentation figures.

The two primary figures use the same epoch axis, member colors, and recorded
copy evidence. Both use a lineage grammar: a recipient's old path ends, and its
new path branches from the donor checkpoint. `01_performance_progression.png`
shows validation performance and the final-10 comparison;
`02_learning_rate_evolution.png` shows donor→recipient LR mutations.

Only two supplementary physics outputs are retained:
`physics_performance.png` and `background_efficiency_curves.png`. They live
directly in `plots/`; redundant population, score-evolution, LR-correlation,
and exploit-timeline candidates were removed.

- Status: `completed`
- Method: `windowed_pbt_v2`
- Canonical server path: `runs/pbt/windowed_pbt_v2_100epochs`
- Best `validation_total_reference_mistag_geomean_percent`: `0.32138124685977204`
- Best member / generation: `lr_3e-6` / `88`

## Contents

The JSON and CSV files are generated from recorded evidence. `plots/` contains selected run figures; `models/` contains only explicitly selected release assets.

## Provenance

Source manifest SHA256: `6baf6613787787635b054043a453b0334726fc2bb446178d48e9d790ecb64bb8`
