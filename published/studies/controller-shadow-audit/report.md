# Representative-60k shadow-controller audit

Reproduce the machine-readable audit and dashboard from the recorded run:

```bash
.venv/bin/python scripts/research/audit_shadow_controller.py \
  runs/pbt/pretrained_fixed_lr_8gpu_representative60k_shadow \
  --output-dir analysis_outputs/representative60k_shadow \
  --verify-crc
```

This command reads completed artifacts only; it does not train or evaluate a
model.

## Verdict

**C — shadow evidence is insufficient; another shadow run is needed before a live adaptive-LR run.**

The execution is intact and the controller is stable, but it is not yet shown to be useful: only two active proposals occurred, both were UP, both were clamped back to the unchanged LR, six of eight members were permanently inactive, and DOWN was never exercised by either the dense shadow controller or the cadence-normalized reference replay.

## Integrity

- All 8 members completed generations 0–95; all 768 worker records are `completed`.
- Every recorded worker LR, controller LR, and optimizer LR matches the member's configured fixed LR.
- All 768 action records have `applied=false`; no exploit event occurred.
- Counts are complete: 768 generation proxy rows plus one initial proxy evaluation; 19 reference rounds × 8 members plus one initial reference row (153); 152 mirrored control-tier rows.
- All 2,304 expected generation checkpoint components (state, optimizer, scaler for epochs 18–113 across eight members) are present. ZIP CRC validation read 8,243,259,776 bytes and found no corruption.
- Minor metadata note: terminal generation 95 omits the redundant `dynamic_controller.applied_action_count` field. Its controller record still has `applied=false`, and all eight terminal action records have `applied=false`.

## Controller behavior

| member | UP | KEEP | DOWN | patience-suppressed | cooldown-suppressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_lr_3e-6 | 0 | 96 | 0 | 0 | 0 |
| fixed_lr_4p5e-6 | 0 | 96 | 0 | 1 | 0 |
| fixed_lr_6e-6 | 0 | 96 | 0 | 2 | 0 |
| fixed_lr_7p5e-6 | 0 | 96 | 0 | 3 | 0 |
| fixed_lr_9e-6 | 0 | 96 | 0 | 3 | 0 |
| fixed_lr_10p5e-6 | 0 | 96 | 0 | 3 | 0 |
| fixed_lr_12e-6 | 1 | 95 | 0 | 1 | 2 |
| fixed_lr_14e-6 | 1 | 95 | 0 | 2 | 2 |
| **total** | **2** | **766** | **0** | **15** | **4** |

The UPs were `fixed_lr_14e-6` at generation 4 and `fixed_lr_12e-6` at generation 5. There were no active-action reversals or chatter. The raw change crossed the threshold 38/760 times; the beta=0.5 EMA crossed it 18/760 times. No negative EMA change crossed the threshold. After generation 5 there were no more active proposals.

Both UPs were clamped to no change. The 14e-6 member proposed 14.7e-6 but was bounded to 14e-6; the 12e-6 member proposed 12.6e-6 but was bounded to 12e-6. `max_cumulative_lr_factor_per_epoch: 1.0` makes every non-KEEP action a no-op even in active mode; the 14e-6 member also sits at `max_lr`.

## Proxy/reference agreement

Comparison uses matched 60k control and 150k monitor values at generations 4, 9, ..., 94, seeded by the measured initial proxy/reference checkpoint. This gives 152 member-round changes. The cadence-normalized action comparison replays the same beta, threshold, patience, and two-observation cooldown on both sparse series; it is separate from the denser actual shadow action stream.

- Raw sign disagreements: 45/152 (29.6%). Of these, 43 are below 0.00457953 pp on both sides.
- The two remaining cases are `fixed_lr_6e-6` at generation 24 (proxy −0.005304 pp, reference +0.001373 pp) and generation 39 (proxy −0.008130 pp, reference +0.001825 pp). The opposing reference movements are below threshold, and the actual dense controller kept LR unchanged in both cases.
- EMA/threshold direction disagreements: 13/152 (8.6%), all proxy UP versus reference FLAT. There are zero UP-versus-DOWN conflicts.
- Cadence-normalized action disagreements: 5/152 (3.3%), all at generation 29 and all proxy UP versus reference KEEP (`4p5e-6`, `9e-6`, `10p5e-6`, `12e-6`, `14e-6`). These are counterfactual sparse-cadence actions; the actual dense controller proposed KEEP in all five cases.
- Proxy/reference change residual MAE is 0.00224 pp and robust sigma is 0.00251 pp, below the configured 0.00457953 pp gate.

The two actual UP proposals were directionally supported: the 14e-6 reference improved from 0.456705 initially to 0.418101 at generation 4 and 0.408377 at generation 9; the 12e-6 reference improved from 0.423999 at generation 4 to 0.409775 at generation 9.

## DOWN and responsiveness

There is no DOWN case to inspect. There were zero DOWN proposals, zero `degraded`/`unsafe` labels, zero negative EMA threshold crossings, and zero cadence-normalized reference DOWN actions. Therefore DOWN timing, reference support, and stability remain untested.

Patience and cooldown do not look like the main bottleneck: only 15 and 4 observations, respectively, were suppressed. The meaningful-change gate plus smoothing dominate activity. Beta=0.5 usefully reduces 38 raw crossings to 18 EMA crossings, and the reference residual scale supports retaining the current threshold for safety. Those facts do not establish utility: six members never act and the controller is silent after generation 5. A ±5% action remains a conservative candidate magnitude, but its training effect was not tested and the current cumulative-factor bound reduces it to 0%.

## Recommendation

Do not specify or launch a live run yet. Run a pre-registered plateau/continuation shadow with the current beta, threshold, patience, cooldown, and ±5% proposals unchanged, using the same 60k/150k measurement cadence. Make the evidence-driven mechanical correction `max_cumulative_lr_factor_per_epoch: 1.10` in shadow so proposed bounds are non-zero; keep `mode: shadow`. Require real DOWN coverage across more than one member, no meaningful opposite reference direction, and non-clamped UP/DOWN proposals before live review.

An optional sensitivity sidecar may compare a pre-registered 0.75× threshold, but it must remain observational and must not replace the primary policy. On this history it creates more UPs but still no DOWN evidence, so it is not an evidence-driven parameter change.

No live-run config is proposed under verdict C.
