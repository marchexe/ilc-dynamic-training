Methodology
===========

Physics task and data
---------------------

The tagger classifies hadronic jets from simulated e⁺e⁻ → νν̄qq̄
events at the ILC into three classes: b, c and light (d) quark jets. Events were
generated and passed through the SGV fast detector simulation; each jet is
represented by up to 75 charged and up to 75 neutral particle-flow candidates
with 27 and 28 per-particle features respectively, plus their four-momenta.

Training uses the 2025-07 SGV sample restricted to the three classes. Most
completed experiments train on a 2.39-million-jet subset (``train800k``); the
ongoing study uses a ten-times larger training set (``train8m``, 23.9 million
jets) of which ``train800k`` is an exact prefix. During training, fixed
preselection cuts and Weaver's class-balancing reweighting (flat method, equal class
weights) are applied, so about 90% of the jets in a training file are used in each
epoch. Evaluation never reweights. The splits and their isolation are defined
in :doc:`evaluation`.

Model
-----

The network is a Particle Transformer (ParT) with particle-pair interaction
features: three particle-attention blocks with eight heads and 128-dimensional
embeddings, pair embeddings of width 64, and a three-class output
(``networks/pretrained_sgv_particle_transformer.py``). Training uses the Weaver
framework (vendored and patched in ``weaver-core/``).

All experiments start from the same pretrained checkpoint, epoch 17 of an earlier
training on an independent 2025-02 SGV sample, together with its recorded
optimizer state. Training epochs in this documentation are counted from that
start: training epoch *t* corresponds to Weaver checkpoint epoch 17 + *t*.

Training protocol
-----------------

Every population member is trained with the same contract:

* Ranger optimizer (RAdam wrapped by Lookahead), batch size 1024, no learning-rate
  schedule within an epoch;
* mixed precision (FP16) with dynamic loss scaling; steps whose scaled gradients
  overflow are skipped by the gradient scaler;
* frozen BatchNorm statistics;
* deterministic, audited data loading: every epoch traverses the full training set
  in a recorded order, and the consumed rows are verified;
* one complete epoch per training process. After each epoch the model, optimizer
  and loss-scaler state are saved and the population policy may act before the next
  epoch starts.

Optimizer restart contract
~~~~~~~~~~~~~~~~~~~~~~~~~~

Weaver's checkpoint of the Ranger optimizer stores the inner RAdam state (step
counts and moment estimates) but not the Lookahead slow weights or its step counter.
Because every epoch runs in a new process, Lookahead is re-initialised from the
current model weights at every epoch boundary. This *epoch-restarted* behaviour is
identical for all compared policies, including fixed-learning-rate training, so it
does not bias the comparisons. It does mean that results describe this training
contract rather than uninterrupted Ranger training, and it prevents adaptation at
finer than one-epoch granularity without a new checkpoint format (see
:doc:`exploratory_studies`).

Learning-rate policies
----------------------

All policies use a population of five members with initial learning rates
3.0, 5.75, 8.5, 11.25 and 14.0 × 10\ :sup:`-6`. Member names (for example
``lr_3e-6``) are stable identities and do not indicate the current learning rate
after PBT has changed it.

**Fixed learning rate** (``fixed_lr_grid``)
   Each member keeps its initial learning rate; members never interact. This is the
   comparator for PBT.

**Windowed PBT** (``windowed_pbt_v2``)
   Every member is validated after every epoch. Every five epochs the policy ranks
   members by their mean validation score over the last three epochs. A member is
   replaced only after losing in two decision windows by more than an operational
   margin of 0.002 percentage points; at most two members are replaced per
   decision. A replaced member copies the donor's model, optimizer and loss-scaler
   state and continues with the donor's learning rate multiplied by 0.8 or 1.2,
   bounded to 2 × 10\ :sup:`-6` – 3 × 10\ :sup:`-5`. No copy is made at the final
   boundary of a run. The decision logic is frozen
   (``scripts/training/pbt/reference/windowed_v2.py``) and every decision is
   recorded for replay.

**Cadenced PBT** (``cadenced_pbt_v1``)
   Every member is validated after every epoch. After a two-epoch warm-up, at every
   eligible boundary the best member may replace the worst one (at most one
   replacement), copying the full model, optimizer and loss-scaler state. A ×0.8 or
   ×1.2 learning-rate mutation within the same bounds is applied when it does not
   duplicate another member's learning rate; otherwise only the state is copied. With an
   adaptation interval of one epoch this is called *every-epoch PBT*. An otherwise
   identical control with an interval of five epochs (*every-five-epochs PBT*)
   isolates the effect of adaptation frequency.

Windowed and cadenced PBT differ in their evidence window, losing-history rule,
number of replacements and replacement rule, not only in cadence. A comparison
between them is therefore a comparison of two complete policies.

The decision margin of 0.002 percentage points is an operational threshold used
by the policies. It is not a statistical test.

Scope
-----

No learning-rate schedule, adaptive controller, rollback or early stopping is part
of any compared method. An adaptive learning-rate controller was studied separately
in shadow mode, where its proposals were logged but never applied
(:doc:`exploratory_studies`).
