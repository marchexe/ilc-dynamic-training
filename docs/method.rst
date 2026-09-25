Method
======

Particle Transformer training
-----------------------------

The project continues a pretrained three-class Particle Transformer with Weaver.
Reference experiments resume epoch 17 with the recorded model and raw optimizer
state, use deterministic full-epoch traversal, mixed precision with scaler state,
and frozen BatchNorm statistics. Every population member sees the same training
and validation definitions; only its trajectory and current LR may differ.

Historical Ranger restart contract
----------------------------------

The production ``ranger`` optimizer is Weaver's historical RAdam optimizer
wrapped by Lookahead.  Its checkpoint ``state_dict`` intentionally serializes
only the inner RAdam state.  On every worker restart, Weaver restores the RAdam
per-parameter state (including steps, first moments and second moments), then
Lookahead rebuilds its slow weights from the restored model parameters.  A new
worker constructs a fresh Lookahead wrapper with a zero step counter; because
the counter is absent from the checkpoint, no previous counter value is
restored.  The slow weights and counter therefore do not continue across a
one-epoch worker boundary.

This is the existing behavior for both ``windowed_pbt_v2`` and
``cadenced_pbt_v1``.  Their copy paths transfer the same model, RAdam-state and
AMP-scaler bundle, and only rewrite the planned learning rate.  The comparison
must retain this shared restart behavior; this documentation and its regression
test do not change optimizer serialization or resume semantics.

Validation metric
-----------------

Physics performance is evaluated through background mistag percentages at fixed
signal efficiencies:

* b-tag at 80% and 90% efficiency, against c and d backgrounds;
* c-tag at 50% and 80% efficiency, against b and d backgrounds.

Four b-tag mistags form a geometric-mean b-tag score; four c-tag mistags form a
geometric-mean c-tag score. The PBT selection metric is the geometric mean of
those two scores. This balances the two taggers and remains finite only when all
required working-point measurements exist.

Population Based Training
-------------------------

``windowed_pbt_v2`` maintains five persistent member identities. Every member
trains and validates for one full epoch. Every five epochs, the strategy ranks
the mean score over the last three epochs. A branch becomes eligible for
replacement only after two losing windows outside a 0.002 decision margin.

At most two recipients copy the donor's model, optimizer and AMP scaler bundle,
then receive deterministic ×0.8 or ×1.2 LR mutations bounded to
2 × 10⁻⁶ through 3 × 10⁻⁵. Decisions, hashes and pre/post-copy evidence are
recorded for replay. The algorithm's scoring, tie handling, exploration and
terminal behavior are frozen for reproducibility.

``cadenced_pbt_v1`` also trains and fully validates every member for one full
epoch, but has two warm-up epochs and then offers one best-to-worst replacement
at every eligible post-validation boundary.  A replacement copies the complete
model/RAdam/AMP-scaler bundle.  Exploration deterministically chooses a bounded
×0.8 or ×1.2 LR mutation when one is collision-free; if no mutation is valid,
the weight/optimizer/scaler copy still occurs at the recipient's prior LR.

Its generic ``exploit_interval_generations`` setting uses global completed-epoch
cadence: interval *N* is due when ``completed_epoch % N == 0``.  The two-epoch
warm-up is an independent eligibility gate and never shifts that anchor.  Thus
the cadence-5 control has opportunities after epochs 5, 10, ..., 45; the due
epoch-50 boundary is terminal-suppressed.  Apart from that interval, the
cadence-1 production arm and cadence-5 control share the same resolved policy,
initialization, data, execution, checkpoint, recovery and reporting contracts.

Scientific comparison contract
------------------------------

The production study compares two complete PBT policies,
``windowed_pbt_v2`` and ``cadenced_pbt_v1``.  It is not an experiment that
isolates cadence alone: the evidence window, losing-history rule, recipient
count and replacement policy differ as well as the decision cadence.

The primary endpoint is the mean over the final ten epochs of the current
population's best full-reference metric at each epoch.  The metric is
``validation_total_reference_mistag_geomean_percent`` and lower is better.  A
single lucky checkpoint or all-time global best is not the primary endpoint.
The configured ``0.002`` decision margin is an operational threshold used by
the policies; it is not a p-value, confidence interval, or claim of statistical
significance.

Adaptive-LR direction
---------------------

The offline :doc:`proxy-validation qualification <proxy_validation>` selected
the deterministic 60k representative subset as a sufficiently reliable signal
for shadow testing. This qualifies the measurement, not the controller:
temporal direction agreement is 0.720, so a future policy must use smoothed
short-window evidence, patience and cooldown instead of reacting to one
measurement.

The prepared shadow policy uses an EMA with beta 0.50 and a 0.00457953
percentage-point threshold estimated as one robust sigma of paired adjacent
proxy-minus-reference changes. Two consistent directional observations are
required, followed by two complete cooldown observations. Proposed ×1.05,
KEEP and ×0.95 actions are logged only. Training remains an eight-member
fixed-LR grid, and the deterministic 150k reference runs every five
generations as an observational tier that is not read by the controller.

That controller is not yet part of the reference method. Shadow mode tests its
behavior, not whether adaptive LR improves training. Any live policy must be
introduced as a separately reviewed version and evaluated against both
``windowed_pbt_v2`` and fixed-LR controls.
