# Stage 3 preregistration: fixed LR vs frozen windowed PBT v2 on train8m

Machine-readable and binding: `protocol_v2.json` (SHA256 in
`protocol_v2.sha256`). This page restates it. If they disagree, the JSON wins.
The scientific content was frozen in `protocol.json` (v1) before any Stage 3
training; v1 records its identity paths relative to the checkout. Every launch
and resume recomputes the identity block and refuses to start on drift:
configs, train8m manifest, test specification, epoch-17 checkpoint,
data/network configs, policy, and 29 execution files (runner, Weaver, Ranger,
frozen windowed policy, Stage 1 endpoint, verifier).

## Administrative revision v2

v2 was made before any Stage 3 training. Its scientific sections (design,
horizon, endpoints, validity, gate, references) are copied verbatim from v1.
Every load verifies that they still match v1, and that v2 cites v1's
unchanged SHA256.

Only the identity block was recomputed, for portability:
- Cluster node names moved from `policy.json` to the untracked local site
  configuration (`configs/site.example.json`).
- Identity paths are recorded relative to the checkout.
- The sealed-test source paths and the train8m manifest no longer contain
  machine-specific locations.
- One execution file changed: a node name was removed from the runner's
  `--slots` help text.

All arm and gate config fingerprints, the checkpoint hashes, the train8m
specification, and the data and network configs are identical to v1.

## Question and design

Does the frozen `windowed_pbt_v2` policy improve the Particle Transformer
b/c/d tagger relative to the matched fixed-LR population when both continue
from the epoch-17 checkpoint on the ten-times larger train8m input?

| | Arm A `stage3_train8m_fixed_lr.yaml` | Arm B `stage3_train8m_windowed_pbt_v2.yaml` |
|---|---|---|
| Policy | `fixed_lr_grid`, never copies | frozen `windowed_pbt_v2` (5-epoch windows, 3-epoch score, margin 0.002, 2 losing windows, ≤2 recipients, ×0.8/×1.2) |
| Inherits | `foundation_fixed_lr_50epochs.yaml` | `windowed_pbt_v2.yaml` |

These are shared by both arms and verified by `--parity` and the frozen
fingerprints:
- train8m: 23,921,449 rows, full natural epochs, no caps.
- Selection split: `val50k_tail`, 150,000 rows.
- Start: epoch-17 checkpoint with its raw Ranger state.
- Population: 3e-6, 5.75e-6, 8.5e-6, 1.125e-5, 1.4e-5.
- Seeds: Weaver seed = 12345 + generation. Accepted rows per generation are
  therefore identical for all ten workers.
- Training contract: batch 1024, FP16 AMP, frozen BN, deterministic audited
  data, one Weaver process per epoch (the epoch-restarted contract).
- Horizon: 50 epochs per arm (250 member-epochs each).

The only intended difference is B's selection/copy/mutation behaviour. There
is one matched seed schedule and no seed replication.

## Horizon decision: fixed 50 epochs

A 30-epoch campaign with a later 30→50 continuation was rejected because it
is not the same treatment as a native 50-epoch run.

- `windowed_pbt_v2_plan` treats the boundary at `index + 1 == generations` as
  terminal: no copies are made, and eligible members keep their loss counters.
- `state/continuation.py` replays that terminal decision as recorded.
- Replaying the frozen planner on identical synthetic outcomes
  (`protocol.horizon.evidence`, reproduced by `tests/test_stage3_campaign.py`):
  - Native 50-epoch run: copies at epochs 10, 20, 30 and 40.
  - 30→50 continuation: copies at epochs 10, 20, then 35 (the terminal
    epoch-30 boundary does not copy).
  - First divergence at epoch 30; the policies also differ at 35 and 40.
  - Positive control: a continuation horizon of 50 is identical.
- The fixed-LR arm alone would be continuation-safe, but the comparison needs
  one horizon contract for both arms.

Resuming an interrupted run keeps the configured 50-epoch horizon and is
equivalent. `--interim-report` gives a descriptive snapshot at any completed
epoch (for example 30). It never stops, extends or changes either arm, and is
not an endpoint. The horizon was not chosen from any observed performance.

## Endpoints (defined before training)

**Primary (validation, selection-biased):** final-10 current-best population
mean of `validation_total_reference_mistag_geomean_percent` on `val50k_tail`
over full epochs 41–50, per arm.
- Implementation: the Stage 1 `endpoints.current_best_trajectory` and
  `final_window_mean`.
- Report Δ = B − A in percentage points, and Δ/A in percent. Lower is better.
- |Δ| < 0.002 pp (the frozen windowed decision margin on the same split) is
  reported as within the selection-noise margin.
- The same split drives B's copy decisions, so B's validation values are
  optimistic.

**Independent (sealed test3000k):** `Entry$` 17M–20M; 8,970,477 rows.
- Isolation: zero identical scalar records with train8m, val1000k (hence
  val50k_tail and the Stage 2A holdout) and the 20250218 pretraining inputs
  (`TEST3000K_PROVENANCE.md`). No file in this repository referenced it before
  Stage 3.
- Converted and opened only after both arms are complete and valid.
- Checkpoints are fixed by rule and frozen with hashes in
  `selection_freeze.json` before inference. Per arm: the validation global-best
  bundle and all five final members.
- Comparisons (B − A):
  - selected-best;
  - final current-best member;
  - each final member, paired by original identity;
  - final population mean.
- Reported for the primary metric and all eight working points, with paired
  background pass/fail counts (jet level).

**Descriptive holdout:** the Stage 2A `val_holdout`, for the two selected-best
checkpoints only. It was previously inspected, so it is never used for any
choice.

**Secondary:**
- Global-best validation value and epoch.
- Final-member metrics and population mean.
- Training and validation losses, AMP skips.
- Exploit/explore events and LR trajectories.
- Diversity (member spread, LR spread).
- Optimizer steps and accepted rows.
- GPU-hours and wall time.
- Resume/ledger reliability.
- Determinism checks.

## Validity

A comparison is stated only if all of the following hold:
- Both arms completed 50 generations.
- `scripts/validation/verify_fixed_lr.py` passes for both: full traversal of
  every epoch, raw optimizer continuation, replayed windowed decisions, and
  final-evaluation prediction parity.
- All ten workers consumed identical accepted rows in every generation.
- Initial evaluations are identical.
- Final evaluations are complete and finite.
- No committed checkpoint changed against the ledger.

Otherwise the report is INVALID or INCOMPLETE and the sealed test stays
closed.

Determinism checks are preregistered separately:
- Same-named members are bit-identical across arms until that member's first
  windowed copy, including the recorded pre-copy state.
- The gate's epoch-18 state is reproduced by both arms.

A failure is shown as "determinism not confirmed". It qualifies attributing
differences to the policy, but does not invalidate the comparison.

## Limits stated in advance

- **Single seed:** one matched seed schedule describes this realization, not a
  distribution over seeds. No significance is claimed.
- **No event-level intervals:** no physical event IDs exist, so there are no
  event-level confidence intervals; counts are at jet level.
- **Interruption cost:** interrupting inside an epoch loses that epoch's
  uncommitted work; it is repeated from the last checkpoint.
- **Not comparable across datasets:** train8m epochs are not comparable units
  to train800k epochs (restart artefacts are ten times rarer per sample).
