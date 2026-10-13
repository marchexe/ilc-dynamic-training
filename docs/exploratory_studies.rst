Exploratory Studies: Adaptive Control and Finer Adaptation
==========================================================

These studies explored learning-rate adaptation beyond epoch-level PBT. None of them
produced an evaluable improvement. They are reported because their negative or
blocked outcomes constrain the next experiments.

Adaptive learning-rate controller in shadow mode
------------------------------------------------

**Question.** Can a rule-based controller, fed by the qualified 60,000-jet validation
proxy (:doc:`proxy_validation`), propose stable learning-rate changes?

**Design.** Eight fixed-learning-rate members (3.0 to 14 × 10\ :sup:`-6`) were trained
for 96 epochs from the common pretrained checkpoint. After every epoch the controller
smoothed the proxy measurement (exponential moving average, β = 0.5) and proposed an
increase (×1.05), no change, or a decrease (×0.95) only when the smoothed change
exceeded 0.00458 pp for two consecutive observations, followed by a two-observation
cooldown. Proposals were logged and never applied (*shadow mode*). The full
validation set was evaluated every five epochs as an independent reference that the
controller did not read. An offline replay of the same rule on earlier trajectories
had proposed 1 increase, 37 no-change and 0 decrease actions.

**Outcome.** The run completed intact (768 member-epochs, all checkpoints verified).

* The controller proposed 2 increases, 766 no-change and 0 decrease actions. Six of
  the eight members never proposed a change, and no proposal occurred after epoch 6.
* Both increases were cancelled by a configured bound on the cumulative change per
  epoch (factor 1.0), so even in active mode they would have had no effect.
* Proxy and reference changes agreed well: the smoothed directions disagreed in 13
  of 152 comparisons, all of them "increase" versus "flat", with no opposite
  directions.

**Conclusion.** The measurement is adequate, but the controller's usefulness could not
be assessed: decreases were never exercised and the only proposals were inactive. No
live adaptive run was started. A further shadow study would need a non-zero
cumulative bound and evidence of decreases across several members. Audit:
``analysis_outputs/representative60k_shadow/report.md``.

Adaptation more often than once per epoch
-----------------------------------------

**Question.** Does adapting within an epoch (for example every 0.2 epochs) help?

**Obstacle.** Each adaptation step requires a process restart from a checkpoint
(:doc:`method`). The Ranger checkpoint stores the RAdam state but not the Lookahead
slow weights or step counter, so each restart re-initialises Lookahead. A
deterministic CPU test compared one uninterrupted epoch with the same epoch run as five
restarted 0.2-epoch chunks. Sample order, optimizer step count, learning rate,
loss-scaler state and random-number continuation matched, but the model and optimizer
states diverged (maximum parameter difference 7.59 × 10\ :sup:`-4`).

**Consequence.** Implementing sub-epoch adaptation with the current checkpoint format
would confound adaptation frequency with optimizer restarts. A valid study needs a new,
versioned checkpoint format that preserves the full Lookahead state, a proof that
restarted and uninterrupted training are equivalent, and newly matched comparators at
both granularities. This has not been implemented.

Training from random initialization
-----------------------------------

All reported runs fine-tune the pretrained checkpoint. Support for starting from
random initialization was implemented and tested, and a three-epoch, five-member
smoke run completed. No full comparison from random initialization has been run, so
none of the results transfer to that regime.
