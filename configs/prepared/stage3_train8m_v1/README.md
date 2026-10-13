# Stage 3 campaign: operations

The scientific contract is in `PROTOCOL.md` and `protocol_v2.json` (administrative
revision of the original `protocol.json`). Requires the local site configuration
(`configs/site.example.json` → `configs/local/site.json`). This page
covers running the campaign, recovery, resources and outputs.

## One command

```bash
bash scripts/experiments/run_stage3_train8m.sh
```

| Option | Effect |
|---|---|
| `--dry-run` | Read-only: preflight and GPU discovery; nothing converted, leased or launched |
| `--status` | Phase, gate verdict, per-arm generation/worker progress, GPU-hours, tmux liveness |
| `--parity` | Arm parity check (read-only) |
| `--interim-report` | Descriptive snapshot at the last epoch both arms completed; never touches the sealed test |
| `--retry-gate` | Retry a gate that failed its criteria, only after the cause is reviewed |

On the submitting host, the launcher checks the following and stops if any
fails:
- A clean tracked tree, with all Stage 3 files committed.
- The frozen protocol hash, with no identity drift.
- Arm parity and the 50-epoch horizon, with the sealed split unused.
- A quick train8m record check.
- At least 100 GiB of free disk.

It then probes the authorized nodes of the local site configuration over SSH.
It prefers the configured preferred node and a node with five or more idle
GPUs of one model (idle means no process, <500 MiB and <5% utilization,
≥40 GB). It submits one detached tmux supervisor; SSH disconnection does not
stop it.

The supervisor runs the phases in order (`--status` shows the current one):

1. **PREP**:
   - Full SHA256 re-hash of train8m against the committed manifest.
   - Conversion and verification of the sealed `test3000k` with the Stage 2B
     converter: three CPU workers at `nice 10`, about 11 min.
   - Converted rows are checked to be identical to the ROOT source.
2. **GATE**: one real, uncapped full train8m epoch of the arm-A contract for
   member `lr_14e-6` on one GPU.
   - Checks:
     - `verify_fixed_lr` (full traversal, raw continuation, final-evaluation
       prediction parity).
     - Every one of 23,921,449 rows scanned.
     - Accepted rows within 6σ of the reweighting expectation
       (21,512,821 ± 8,831).
     - Optimizer steps = batches − AMP skips; AMP skips ≤1%.
     - No sample caps.
     - Finite training and validation loss, primary metric and all eight
       working points. A failure lists each failing key with its observed
       value.
     - Maximum gradient norm, read from the worker log. Weaver's epoch
       maximum includes AMP-skipped steps, whose gradients overflowed, so it
       may be `inf`. A finite value passes. `inf` passes only when all of the
       following hold:
       - the log, the manifest and the data-audit accounting
         (batches − optimizer steps) agree on at least one skipped step;
       - the manifest records no gradient norm;
       - the step-accounting and AMP-fraction checks pass.

       `nan`, `-inf`, missing, malformed or duplicated values, missing AMP
       counters and inconsistent counters fail.
     - Complete 150,000-row validation.
     - A consistent model/Ranger/scaler bundle.
     - CUDA allocation within ±10% of the Stage 2B 21,540 MB; nvidia-smi
       ≤36,000 MiB; host RSS ≤32 GiB.
     - Throughput 3k–20k rows/s; startup ≤15 min; run disk ≤2 GiB.
     - Clean termination.
   - The measured epoch also projects the campaign. The projection must fit
     800 GPU-hours and 240 h of wall time, assuming the sequential worst case
     and a 1.3 safety factor.
   - **A failed gate blocks the campaign.** Rerunning only reruns the gate if
     the failure was operational (no GPU, runner crash). After a criteria
     failure, reruns stay blocked until `--retry-gate`.
   - Gate outputs are never reused by either arm. Its epoch-18 state is only
     a determinism reference.
   - **Gate history.**
     - Attempt 1 failed only the earlier combined finiteness criterion. The
       log reported `Max Grad Norm: inf` with 7 AMP-skipped steps out of
       21,012, and the shared parser does not record `inf`.
     - The criterion was then split into the two checks above.
       `gate_revision_1.json` (+ `.sha256`) records the failed verdict's hash,
       the defect, the exact rule, and that the correction was made after
       inspecting attempt 1. The protocol, thresholds and pinned files are
       unchanged.
     - Attempt 2 passed all 19 checks and reproduced attempt 1's epoch-18
       state bit for bit.
     - Both attempt records are kept in `gate/`.
3. **ARMS**: the production runner per arm, each on five leased GPUs of the
   campaign model.
   - Each arm starts as soon as a full five-GPU set is idle. Both run
     concurrently when ten GPUs are idle, otherwise sequentially. Partial sets
     are never held.
   - A GPU used by another person stops only this campaign's processes.
4. **REPORT**:
   - Full ledger re-verification.
   - Validation analysis; it must be COMPLETE before the sealed test opens.
   - Selection freeze.
   - Sealed test and descriptive holdout inference, through the Stage 2A
     inference campaign (leases, full-pass audit, row-ID coverage, finite
     eight working points).
   - Final report.

## Recovery and safety

- **Rerunning is the resume path.** A completed and verified campaign returns
  immediately. A live tmux session is reported, never duplicated. A stale
  session (dead tmux) is replaced. Concurrent launches or supervisors exit 75
  (flock).
- **Runs never restart from epoch 17 once they exist.**
  - An existing unfinished arm is resumed with `--resume`, which repeats only
    incomplete generations.
  - A run directory without a manifest, or with a manifest from a different
    configuration, is refused.
  - Seeds, epochs, Ranger state, AMP scaler and windowed history continue
    from the manifest and the per-epoch bundles.
- **Checkpoint ledger.** After every committed generation (post-copy), the
  supervisor records SHA256 of each member's model/optimizer/scaler in
  `ledger/<arm>.json`.
  - Before any resume, the bundle being resumed from must match the ledger.
  - The full ledger is re-verified before reporting.
  - A mismatch, a rolled-back manifest, or duplicate generations fail
    explicitly.
- **Stopping.** Killing the tmux session or sending SIGTERM/SIGHUP sends
  SIGINT to the runners. They terminate their workers and record
  `interrupted`; the current epoch's uncommitted work (up to ~1 h) is lost.
  GPU and campaign leases are inherited by the runners.
- **Failures.**
  - A failed worker fails only its arm; the other arm continues, and the
    failure is recorded with no automatic retry (`max_auto_resumes_per_arm: 0`).
    The same command resumes it.
  - Budget overrun, which also counts gate attempts, stops both arms
    resumably.
  - An arm waiting more than 72 h for GPUs ends that launch with exit 69.
- **Exit codes:** 0 submitted/completed; 1 gate/arm/validity failure; 2 invalid
  inputs or drift; 69 resources; 75 locked; 130 interrupted.

## Resources

Measured in Stage 2B with two concurrent workers on A100 80GB:
- CUDA 21,540 MB allocated; nvidia-smi 24,894 MiB.
- Process-tree RSS 37.8 GB, about 19 GB per worker.
- Worker startup 60–67 s.
- Validation 61 s.
- Bounded-sample training 5.3–6.4k rows/s (includes first-fetch latency).
- Checkpoint bundle 10.7 MB per member-epoch.

The production train800k throughput was 8.2–13k rows/s.

Estimates; the gate replaces these with a measured projection before the arms
start:

| Item | Estimate |
|---|---|
| Member-epoch | 30–70 min (21.5 M rows at 5.3–13k rows/s + ~2 min startup and validation) |
| Gate | 1 GPU, 0.5–1.2 h |
| Arms | 500 member-epochs: 250–580 GPU-hours |
| Wall time, arms | 25–58 h concurrent (10 GPUs); 50–117 h sequential (5 GPUs) |
| Sealed test + holdout | 12 × 8.97 M + 2 × 2.83 M rows at ~2.1–2.5k rows/s: 13–15 GPU-hours, ~2.5 h on 10 GPUs |
| Total | ~265–600 GPU-hours |
| Host RAM | ~190 GB for ten workers; supervisor requires MemAvailable ≥ workers × max(20 GiB, gate peak) × 1.25 |
| Disk | ~5.4 GB checkpoints, <1 GB archives/logs, 21.7 GB test Parquet, ~2.6 GB predictions |

All ten workers share one node, so train8m (57.8 GB) fits in the page cache.
Each worker start hashes all training files (`--data-audit`), so start-up I/O
is repeated every epoch.

## Outputs (`runs/eval/stage3_train8m_v1/`, arms in `runs/pbt/stage3_train8m_*_v1/`)

**Campaign records:**
- `status.json`, `campaign.log` (`[PREP]`, `[GATE]`, `[ARM]`, `[REPORT]`),
  `supervisor.log`, `supervisor_exit.json`.
- `launch.json`, `runtime_preflight.json`, `cluster_snapshot.txt`,
  `allocations.json`.
- `ledger/`.

**Gate:** `gate/verdict.json`, `gate/attempt-NNN.json`, and the gate runs.

**Validation analysis:** `validation_analysis/`.

**Sealed test and holdout:** `sealed_test/selection_freeze.json`,
`sealed_test/evaluation_protocol.json`, `sealed_test/evaluations/<checkpoint>/`,
and `holdout_descriptive/`.

**Final report** (`report/`):
- `report.md`, `analysis.json`, `comparison.csv`, `report_integrity.json`.
- Figures:
  - `figures/01_current_best_primary.png`
  - `02_member_trajectories.png`
  - `03_windowed_learning_rates.png`
  - `04_training_loss.png`
  - `05_test_working_points.png`

Figures use the validated reference categorical pair (`#2a78d6`, `#eb6834`).
The palette validator (`node`) is not installed on the cluster and was not
re-run.

**Interim snapshots:** `interim/epoch-NNN/`.
