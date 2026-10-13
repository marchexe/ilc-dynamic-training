# Stage 2A: frozen independent holdout campaign

This package evaluates nine **previously validation-selected** checkpoints on
the existing `val_holdout`. It runs inference only. It never trains, mutates
learning rates, changes physics working points, or selects a checkpoint using
holdout performance. Preparation does not run inference.

## One-command overnight launch

From the repository root:

```bash
bash scripts/experiments/run_stage2a_holdout.sh
```

The launcher verifies the frozen protocol and input hashes, discovers current
GPU availability, and submits the whole campaign to detached remote **tmux**.
SSH disconnection does not terminate the campaign. Postprocessing is automatic.
No GPU IDs, configuration edits, checkpoint choices, or postprocessing commands
are needed. The same command resumes an interrupted campaign; it skips only
results whose checkpoint, log, predictions, traversal, protocol, and hashes
still validate. A fully verified completed campaign returns after preflight,
without requiring free GPUs. Failed attempts remain in separate directories.

`protocol.json` and `protocol.sha256` are the frozen scientific contract.
`dry_run.json` is a preparation-time observation, **not** a GPU reservation.
Runtime observations are recorded separately and checked again before each job.

## Allocation policy and resources

The user explicitly confirmed on 2026-10-12 that direct SSH, tmux, and use of
idle GPUs on the authorized cluster nodes are permitted; Slurm/PBS is not required.
Permission is recorded separately from `scripts/cluster/check.sh` availability.

Default concurrency is **three**, with one process per GPU. Prefer a homogeneous
cohort on the preferred node (A100 80GB), otherwise select a homogeneous authorized
cohort dynamically. A GPU must have no compute processes, less than 500 MiB
used memory, utilization below 5%, and at least 40,000 MiB total memory.
Recorded production training peak is 21,539.7 MiB at the same model/batch size;
40GB capacity provides margin, while test mode removes gradients/optimizer
state. GPU OOM still fails the checkpoint explicitly.

`STAGE2A_MAX_PARALLEL` optionally limits concurrency to 1–9; ordinary launch
needs no override. Existing `ML_PFA_CLUSTER_NODES` can restrict discovery to a
subset of authorized nodes. Existing `ML_PFA_SSH_CONFIG` is respected.

Availability is checked after acquiring a cooperative GPU lease and immediately
before launching. GPU UUID visibility isolates the worker. During inference,
foreign compute PIDs cause termination of **only this campaign's worker**.
GPU/checkpoint leases are inherited by the child, preventing duplicate work
after abrupt supervisor death. The leases are advisory: unrelated launchers
that do not follow this convention cannot be globally excluded without a
scheduler. Foreign-process monitoring limits that direct-SSH race; this is not
a claim of scheduler-enforced exclusive allocation.

If resources become busy, acquisition waits at most 120 seconds per attempt.
Each inference has a four-hour timeout. There are no automatic inference
retries and no CPU inference fallback. The next one-command invocation retries
failed/incomplete checkpoints. Submission exits 0; unavailable resources exit
69, invalid protocol/configuration exits 2, and an active campaign lock exits
75. The detached supervisor writes its final exit code to
`supervisor_exit.json`; checkpoint failures remain visible in `status.json`.

## Frozen checkpoint inventory

The selected set is original epoch 17, fixed-LR 50, windowed PBT 50/100,
fixed-LR 14e-6 100, and cadence1/cadence5 at training seeds 12345 and 22345.
Exact model paths, hashes, selection generations, epochs, source manifests,
resolved configurations, historical commits, and seed provenance are in the
protocol. All nine model state dictionaries were loaded strictly on CPU.
Python and numerical-package versions are pinned and checked at launch;
per-checkpoint records also capture the execution Git state and GPU hardware.

All trained selections use the recorded primary-metric global best, not the
last checkpoint or the final-10 winning branch. Internal Weaver epochs include
the original epoch-17 resume offset. Fixed-LR 100 is a 50-epoch continuation:
its configured seed is 12395, with effective original seed base 12345 and a
100-epoch combined horizon. Its best local generation 48 is combined training
epoch 99 (Weaver epoch 116).

## Data and statistical contract

Holdout contains **2,825,125** Parquet rows: bb 943,645; cc 941,995; dd 939,485.
For each flavor, it is the prefix of `val1000k`, followed by the disjoint 50k
monitor and 5k control windows. Preparation verifies the source ranges, exact
scalar-record content/order, file hashes, and absence of identical scalar
records in both current `20250711` and pretrained-source `20250218` training
inputs. Those checks do not establish physical-event lineage between different
jets when no event ID is stored.

Evaluation retains the b/c/d class order and all eight working points:
bc/bd at b efficiency 80/90%, cb/cd at c efficiency 50/80%.
The primary metric remains
`validation_total_reference_mistag_geomean_percent`, using the existing
geometric mean of the b- and c-tag four-point scores.

Every successful result requires a finite primary metric, all eight finite
working-point metrics, full natural dataset exhaustion, exactly zero optimizer
steps, a verified prediction-score digest, and the canonical ordered row-ID
sequence. Stage 2A opts into saving the actual `__weaver_row_id__` observer.
File chunks can interleave flavors, so ordering is checked against the recorded
audit sequence digest rather than assumed from file order. IDs must cover
exactly `arange(N)` without repetition; labels must match frozen per-file
class offsets. Analysis pairs checkpoints by the actual IDs, accepting valid
differences in chunk delivery order while requiring identical row coverage.
Analysis recomputes the canonical metrics from predictions and checks the log.

Paired comparisons use the same holdout rows and report matched differences
plus per-working-point concordant/discordant background counts. A tested
physical-event cluster bootstrap uses common event draws when actual physical
event IDs are provided. **This dataset has none**, so physical-event confidence
intervals are explicitly unavailable. Row IDs are not substituted for event
IDs. No significance claim is made from two training seeds. Holdout ranking is
diagnostic and never a new checkpoint-selection rule. The holdout has historical
diagnostic uses; it is not described as a pristine, never-inspected test set.

Reports distinguish validation-selected checkpoint performance, global-best
validation, final-10 current-best population performance, and independent
holdout. Final-10 values come from recorded trajectories and are never
computed from the newly evaluated checkpoint.

## Outputs

All campaign outputs are under `runs/eval/stage2a_holdout_v1/`:

- `launch.json`, `runtime_preflight.json`, `cluster_snapshot.txt`: allocation
  and launch provenance;
- `supervisor.log`, `supervisor_exit.json`, `campaign.log`, `status.json`:
  execution state and timestamped logs;
- `evaluations/<checkpoint>/result.json`: atomic verified per-checkpoint state;
- `evaluations/<checkpoint>/attempt-NNN/`: complete inference logs, console
  output, prediction Parquet, and audit reports;
- `analysis.json`, `comparison.csv`: machine-readable comparisons, all working
  points, ranking consistency, paired counts and uncertainty availability;
- `report.md`: readable comparative scientific report;
- `figures/holdout_comparison.png`, `figures/paired_differences.png`;
- `report_integrity.json`: final report/figure hashes.

All historical runs, manifests, metrics, events, checkpoints, publications, data
manifests and splits remain unchanged. No output targets a historical run.

## Planning estimates, not new benchmarks

Preparation verification is recorded in `preparation_checks.json`. The final
suite ran 526 tests: 525 passed and one expected Ray skip; all 33 Stage 2A tests
passed. Static compilation, shell syntax and whitespace checks passed. The real
read-only cluster dry run resolved all nine jobs onto an observed three-GPU
A100 80GB cohort on the preferred node without reserving GPUs or starting inference.
A harmless CPU-only tmux smoke job survived the submitting SSH connection.
Synthetic nine-checkpoint execution, automatic reporting and verified restart
also passed; these fixtures are not holdout results.

Historical 150k final evaluations took approximately 41–78 seconds. Linear
row-count extrapolation gives roughly 13–25 minutes per holdout checkpoint,
or 40–75 minutes for nine checkpoints on three comparable GPUs, **plus**
preflight hashing, shared-filesystem I/O, restart costs and CPU analysis.
Hardware and I/O differ, so this is not an overnight completion guarantee.
No real holdout benchmark was run during preparation.

The holdout files occupy 6,809,163,943 bytes and are read in place. Nine raw
score/label/row-ID arrays require about 0.71GB before Parquet overhead/compression;
logs, audit records and figures add space. Require at least **10 GiB free**
for output and failed attempts. Per-user quota is not inferred from filesystem
free space; a quota/write failure is explicit. Failed attempts are preserved,
so storage grows across repeated interrupted campaigns.

## Future train8m planning — no conversion/training performed

Current `train800k` has 2,392,232 post-cut rows and processes about 2.15M after
the existing stochastic reweighting. The existing three `train8m` ROOT files
contain **23,921,449** post-cut rows; expected accepted rows are approximately
**21,512,821**, about ten times larger. These are jet/ROOT-row counts, not a
verified count of unique physical events.

Prepare a new versioned Parquet conversion/manifest for the existing train8m
ROOT inputs, preserving branches, three-class labels and cuts. Existing row-ID
traversal auditing requires Parquet. Point a new configuration at the expanded
input/suffix; retain natural full-pass traversal and remove any shadow sample
cap when a full epoch is intended. No training algorithm redesign is required.
Check source integrity, labels/features and complete traversal before launch.

Verify raw source Entry ranges and event/record isolation. `train8m` uses
`Entry$ < 8,000,000`; the current `val1000k` source uses 16M–17M. Early
`val5k`/`val50k` from the smaller size ladder can lie inside a larger train
window and must not be reused blindly. Retain the isolated tail/holdout
contract or prepare separately reviewed validation/test splits. Repeatedly
inspected holdout should not become an unseen final test for future policy
tuning. Including gg/ss/uu would require a separate class/physics contract.

At batch 1024, expected training batches rise from about 2,102 to about 21,009.
Recorded pure training throughput was about 8,188–13,032 accepted rows/s;
linear extrapolation gives about **28–44 minutes of training per member per
epoch**, plus I/O, validation and restart overhead. The current observed worker
wall-time medians were about 229–351 seconds. Input conversion/storage and
audit hashing also grow approximately tenfold.

Existing PBT remains executable with its frozen epoch cadence: cadenced1
decides each eligible epoch; cadence5/windowed decide at their existing
five-epoch boundaries. Those decisions then cover roughly ten times more
samples and occur much less often in wall-clock time. Cadence normalization
is **not needed for correctness**, but is needed if a future comparison is
intended to preserve sample/step/compute decision budgets. Such normalization
changes the experimental protocol/policy interpretation and must be reviewed
and versioned separately. Stage 2A makes no cadence change.

## Administrative revision after completion

The campaign is complete. Its protocol, inputs, inference and results are
unchanged. Afterwards one pinned execution file, `scripts/cluster/check.sh`
(an informational GPU snapshot), was changed for security: it now verifies
SSH host keys strictly and reads node names from the local site
configuration. `administrative_revision_1.json` (SHA256 in
`administrative_revision_1.sha256`) records the change. The frozen version
remains byte-identical in commit `953a170`.

The launcher's own input check therefore stops at this file, which is the
intended fail-closed behaviour. To verify the completed campaign, run:

```bash
.venv/bin/python scripts/validation/verify_stage2a_provenance.py
```

It performs the same checks and accepts only the recorded change, and only
if the frozen bytes are recoverable from Git with the pinned SHA256.
