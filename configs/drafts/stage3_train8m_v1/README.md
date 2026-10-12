# Stage 3 train8m drafts — INACTIVE, not approved

Two versioned drafts for a future matched comparison on the Stage 2B `train8m`
input. Nothing launches them automatically; Stage 3 requires explicit approval.

| | Arm A `fixed_lr.yaml` | Arm B `windowed_pbt_v2.yaml` |
|---|---|---|
| Inherits | `configs/experiments/foundation_fixed_lr_50epochs.yaml` | `configs/experiments/windowed_pbt_v2.yaml` (frozen policy) |
| Strategy | `fixed_lr_grid`, never exploits | `windowed_pbt_v2`: 5-epoch windows, 3-epoch score, margin 0.002, 2 losing windows, ≤2 recipients, ×0.8/×1.2 |
| Data | train8m (full natural epochs) / `val50k_tail` | same |
| Start | epoch-17 checkpoint, raw Ranger state | same |
| Population | 3e-6, 5.75e-6, 8.5e-6, 11.25e-6, 1.4e-5 | same |
| Horizon | 20 epochs (100 member-epochs) | same |

Both keep batch 1024, FP16 AMP, frozen BN, deterministic audited data, seed
12345 (+generation), the epoch-restarted Ranger contract and the same final
evaluation. `bash scripts/experiments/run_stage2b_smoke.sh --parity` (read-only)
checks parity programmatically and prints the cadence analysis; the GPU smoke
additionally runs a real `--dry-run` of both drafts against the prepared data.

## Horizon and cadence (decision needed)

One train8m epoch is ~21.5 M accepted samples, ten times a train800k epoch.
With the frozen cadence the windowed boundaries fall at epochs 5, 10, 15, 20;
the first copy is possible at 10 (two losing windows) and the terminal boundary
never copies, so a 20-epoch run has two copy opportunities (≤4 copies). A
10-epoch horizon (the sample budget of the Stage 2A 100-epoch runs) would never
copy and would collapse arm B into fixed LR. The 50-epoch train800k protocol
had eight copy boundaries.

Keeping the five-epoch cadence is correct and executable, but each decision now
covers ~108 M samples instead of ~11 M and decisions are ten times rarer per
sample. Matching the Stage 2A decision density would need sub-epoch
generations (outside the frozen full-epoch contract) or a much longer horizon
(about 50 epochs, roughly 120–200 GPU-hours per arm). The 20-epoch value is the
smallest horizon that lets the frozen policy act twice; it is a placeholder for
approval, not a tuned choice. No Stage 2A holdout result was used to set it.
