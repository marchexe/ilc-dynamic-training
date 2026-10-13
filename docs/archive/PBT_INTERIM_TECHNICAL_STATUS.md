# PBT Interim Technical Status

Prepared from completed artifacts on 2026-10-01. No training, validation, PBT,
or GPU job was run to produce this milestone. The primary metric is the
final-10 mean of the current population's best full-reference mistag score;
lower is better.

## Supervisor proposal

The supervisor proposal has two distinct policies:

| Policy | Generation | Weight-copy opportunity | LR-mutation opportunity | Status |
|---|---:|---:|---:|---|
| Supervisor Variant A (`cadenced_pbt_v1`) | 1 epoch | Every generation | Every generation | **DONE — implemented and validated** |
| Supervisor Variant B | 0.2 epoch | Every generation | Every 5 generations | **BLOCKED — not implemented** |

Two other experiments must not be confused with those variants:

- `windowed_pbt_v2` is a useful historical reference, but not a clean
  cadence-only comparator. Its evidence windows, losing-history rule,
  recipient count, and replacement policy differ from Variant A.
- The cadence5 control keeps one-epoch generations and changes only
  `exploit_interval_generations: 1 -> 5`. It is a cadence-control experiment,
  not supervisor Variant B.

## What has been implemented

- Variant A: pretrained epoch-17 initialization, five members, one full epoch
  per generation, full-reference validation every epoch, and copy/LR mutation
  opportunities every epoch after the two-epoch warm-up.
- Cadence5 control: the same production contract, with copy/LR opportunities
  every five globally completed epochs.
- Deterministic reporting and recovery evidence, including pre-copy and
  post-copy identities and model/RAdam/AMP-scaler transfer checks.
- Scratch initialization semantics and CPU tests. A real three-generation
  full-reference scratch smoke completed successfully.

No full 50-epoch scratch production comparison has been run. Supervisor
Variant B has not been implemented.

## Main results

The cadence comparison is paired by deterministic training seed. Both arms use
one-epoch generations, the same pretrained checkpoint, population, initial
learning rates, data, validation, ranking, margin, mutation rule, and final
selection rule. The operational decision margin is `0.002`; it is not a
statistical-significance threshold.

| Seed | Variant A final-10 | Cadence5 final-10 | Cadence5 − A (pp) | Global best A / C5 | Lower trajectory epochs A / C5 / tie | Exploits A / C5 | Lineage collapse epoch A / C5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12345 | **0.332391329420** | 0.333757664732 | +0.001366335312 | **0.330930651037** / 0.331368916743 | 47 / 1 / 2 | 32 / 9 | 5 / 20 |
| 22345 | **0.332398320401** | 0.334039837954 | +0.001641517554 | **0.330665053977** / 0.331416754989 | 42 / 5 / 3 | 25 / 8 | 5 / 20 |

![Matched-seed final-10 comparison](../_static/results/pbt_interim_final10_matched_seeds.png)

![Current-best full-reference trajectories](../_static/results/pbt_interim_current_best_trajectories.png)

![PBT exploit and lineage summary](../_static/results/pbt_interim_behavior_summary.png)

The direction replicated on two deterministic training seeds: Variant A had
the lower primary endpoint and global best in both pairs. Across 100 paired
epochs, Variant A had the lower current-best value in 89, cadence5 in 6, with 5
ties. This is consistent evidence for more frequent intervention under this
specific pretrained setup. Two seeds do not establish statistical significance
or general cadence superiority.

## Why we added the cadence5 control

The historical windowed reference changes several policy dimensions at once,
so comparing it with Variant A cannot isolate cadence. The cadence5 control was
introduced to change intervention frequency without also changing generation
length, worker restart frequency, epoch-level data traversal, or the historical
optimizer-continuation behavior. It therefore answers a narrow cadence question
within one-epoch generations.

## Variant B blocker

Ranger in this project is RAdam wrapped by Lookahead. The historical
checkpoint contract serializes the inner RAdam state but not Lookahead's slow
weights or step counter. Every supervisor generation starts a new Weaver
process, so a naive 0.2-epoch implementation would reset Lookahead five times
per full epoch rather than once per epoch as in historical Variant A.

The deterministic design-gate test compared one uninterrupted epoch with five
consecutive, non-overlapping 0.2-epoch chunks from the same sample order. Sample
IDs/order, optimizer-step count, LR, AMP scaler, and RNG continuation matched;
model parameters and optimizer state did not. The observed maximum absolute
model-parameter difference was `7.5928867e-4`. This was a CPU design-gate
measurement, not a production performance result.

Patching Lookahead serialization only for Variant B would make Variant B use a
different optimizer-continuation contract from historical Variant A. Any
difference could then be caused by cadence, generation length, or optimizer
state preservation. That is scientifically invalid as a clean A/B comparison,
so implementation stopped before creating a Variant B strategy or config.

## Scratch status

- Scratch semantics: **implemented and CPU-tested**.
- Real scratch smoke: **completed successfully**, three generations and five
  members using the full-reference metric.
- Full scratch production comparison: **not run**.

Scratch is a separate robustness question about initialization regime. It does
not resolve the Variant B optimizer-continuation blocker.

## What we can conclude now

Under the current pretrained setup, the 1-epoch/adapt-every-epoch policy
outperformed the otherwise matched 1-epoch/adapt-every-5-epochs control on both
tested training seeds. This supports more frequent adaptation under this setup,
but does not test the supervisor's 0.2-epoch Variant B. Variant B is blocked by
non-serialized Lookahead state in the historical Ranger continuation contract.

## What we cannot conclude yet

- We cannot claim statistical significance or general superiority of
  adapt-every-epoch PBT from two deterministic seeds.
- We cannot attribute the historical `windowed_pbt_v2` comparison to cadence
  alone.
- We have not tested 0.2-epoch generations without an optimizer confound.
- We have not established whether the pretrained result transfers to scratch
  initialization.
- The seed-22345 manifests recorded a dirty worktree. Both paired arms used the
  same commit and matching recorded runtime source hashes apart from their entry
  configs, but clean provenance is preferable for future production studies.

## Recommended next steps

Preferred rigorous Variant B route:

1. Introduce a **new**, versioned Ranger/Lookahead checkpoint contract that
   preserves inner RAdam state, slow weights, and the Lookahead counter.
2. Prove deterministic restarted-continuation equivalence, including model,
   full optimizer, AMP scaler, RNG, and data-stream cursor.
3. Create two new matched strategies: corrected Variant A with one-epoch
   generations and corrected Variant B with 0.2-epoch generations.
4. Smoke both corrected strategies.
5. Run a matched-seed production comparison.
6. Do not reuse historical `cadenced_pbt_v1` as the corrected comparator.

Separate robustness track:

- Run the already prepared full scratch production arms when resources and
  priority permit, treating them as an initialization-regime study.

Optional later work:

- Add more cadence1/cadence5 matched seeds.
- Qualify a lower-cost validation proxy before using it for selection.
- Optimize the fivefold validation/process overhead implied by 0.2-epoch
  generations.

## Reproducibility and immutable evidence

All four production manifests passed the existing deep read-only audit: 50/50
generations, 250 completed worker intervals, six successful final evaluations,
50 replayed decisions, valid terminal suppression, no duplicate/truncated event
records, checkpoint SHA verification, and semantic optimizer-transfer checks.

| Evidence | Path | Git commit | Dirty | Resolved-config SHA256 |
|---|---|---|---:|---|
| Variant A, seed 12345 | [`runs/pbt/cadenced_pbt_v1_50epochs`](../../runs/pbt/cadenced_pbt_v1_50epochs/report.md) | `90fa89d7eeb351fa2c787489f77faea286bbb31a` | no | `91f1d861df090b9ab34cff67b64c0d1e6742072aa10bdb47e4b31192ee7d434c` |
| Cadence5, seed 12345 | [`runs/pbt/cadenced_pbt_v1_cadence5_50epochs`](../../runs/pbt/cadenced_pbt_v1_cadence5_50epochs/report.md) | `78dccacdd9dc913fba7df500a0d164bd93fb45ce` | no | `f1e8a11209340efc51d5b9de84029c4b7bfff6936a9a85665987e485a84e5c9d` |
| Variant A, seed 22345 | [`runs/pbt/cadenced_pbt_v1_seed22345_50epochs`](../../runs/pbt/cadenced_pbt_v1_seed22345_50epochs/report.md) | `f511919d53339d20efb9aefd5fd16538acd8950b` | yes | `7dfd2a45e77525e4d2c2f8c6415b7cb4c4a893567450172f7eaceee6db697a18` |
| Cadence5, seed 22345 | [`runs/pbt/cadenced_pbt_v1_cadence5_seed22345_50epochs`](../../runs/pbt/cadenced_pbt_v1_cadence5_seed22345_50epochs/report.md) | `f511919d53339d20efb9aefd5fd16538acd8950b` | yes | `3d411fe15a62966909a7da7b2f556f07ad8fdbea79e7b85a3c33cd9f87c0d46c` |
| Historical windowed reference | [`runs/pbt/windowed_pbt_v2`](../../runs/pbt/windowed_pbt_v2/report.md) | `a4f750793911508812b5b09296afe5306306c930` | no | `872bd2d5c8918a974064270b2b7dd86596ca022d728728afdc90a19a2861a905` |
| Scratch smoke | [`runs/pbt/cadenced_pbt_v1_scratch_full_reference_smoke`](../../runs/pbt/cadenced_pbt_v1_scratch_full_reference_smoke/report.md) | `9755026381062cd827cdf99531b768fe7b7ed193` | yes | `b29c4e4ca528a85e15cb7f2168df6f470433581b5039d360013f85db8e5c7250` |

The pretrained arms share initial model checkpoint SHA256
`ae4928aa088b73538597f23b78b51678298e59d23552c9cd2c2e849fb3ced501`,
training dataset fingerprint
`83cd373df93aba62ef1faafc0d25c37af2c20bb4bbe567c972c76d57a439b0a3`,
and validation fingerprint
`831619c8ec495885fe0a98fa83382a48f341508b2beb1b250a60206ba24090c9`.

The machine-readable [interim metrics and provenance](../_static/results/pbt_interim_metrics.json)
record contains the manifest hashes, trajectories, complete source hashes,
dataset fingerprints, and paired calculations. Figures are reproducibly
generated by
[`scripts/reports/plot_pbt_interim_milestone.py`](../../scripts/reports/plot_pbt_interim_milestone.py).
Completed run directories were read only and remain immutable.
