Method
======

Particle Transformer training
-----------------------------

The project continues a pretrained three-class Particle Transformer with Weaver.
Reference experiments resume epoch 17 with the recorded model and raw optimizer
state, use deterministic full-epoch traversal, mixed precision with scaler state,
and frozen BatchNorm statistics. Every population member sees the same training
and validation definitions; only its trajectory and current LR may differ.

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

Adaptive-LR direction
---------------------

The offline :doc:`proxy-validation qualification <proxy_validation>` selected
the deterministic 60k representative subset as a sufficiently reliable signal
for shadow testing. This qualifies the measurement, not the controller:
temporal direction agreement is 0.720, so a future policy must use smoothed
short-window evidence, patience and cooldown instead of reacting to one
measurement.

That controller is not yet part of the reference method. It must be introduced
as a new version and evaluated against both ``windowed_pbt_v2`` and fixed-LR
controls, with an independent validation tier that never drives the decisions
being assessed.
