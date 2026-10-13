Methodology
===========

Data and model
--------------

Jets come from simulated e⁺e⁻ → νν̄qq̄ events at the ILC (SGV fast detector simulation,
2025-07 sample), restricted to b, c and d jets. Each jet is described by up to 75
charged and 75 neutral particle-flow candidates. The splits are defined in
:doc:`evaluation`.

The tagger is a Particle Transformer with pair-interaction features (three attention
blocks, eight heads, 128-dimensional embeddings), trained with Weaver
(:repo:`networks/pretrained_sgv_particle_transformer.py`, :repo:`weaver-core`). Every
experiment starts from the same pretrained checkpoint (epoch 17 of an earlier training
on a 2025-02 sample) and its optimizer state. Training epoch *t* is checkpoint epoch
17 + *t*.

Training contract
-----------------

Shared by every population member and policy:

* Ranger (RAdam with Lookahead), batch 1024, constant learning rate within an epoch;
* FP16 mixed precision with dynamic loss scaling (overflowing steps are skipped);
* frozen BatchNorm; class-balancing reweighting during training, none at evaluation;
* full, deterministic, audited passes over the training set;
* one process per epoch; model, optimizer and loss-scaler state are saved after each
  epoch, then the policy acts.

Weaver checkpoints store the RAdam state but not Lookahead's slow weights or step
counter, so Lookahead restarts at every epoch boundary. This is identical for all
policies and does not bias comparisons, but results describe this *epoch-restarted*
contract, and it prevents adaptation more often than once per epoch.

Learning-rate policies
----------------------

Five members start at 3.0, 5.75, 8.5, 11.25 and 14.0 × 10⁻⁶. Member names (such as
``lr_3e-6``) are identities, not current learning rates.

**Fixed learning rate** (``fixed_lr_grid``)
   Each member keeps its learning rate. The comparator for PBT.

**Windowed PBT** (``windowed_pbt_v2``)
   Every five epochs, members are ranked by their mean validation score over the last
   three epochs. A member losing in two decision windows by more than 0.002 percentage
   points is replaced (at most two per decision): it copies the donor's model, optimizer and
   loss-scaler state and continues at the donor's learning rate × 0.8 or × 1.2, within
   2 × 10⁻⁶ – 3 × 10⁻⁵. No copy at a run's last boundary. Frozen in
   :repo:`scripts/training/pbt/reference/windowed_v2.py`.

**Cadenced PBT** (``cadenced_pbt_v1``)
   After a two-epoch warm-up, at every *n*-th epoch boundary the best member replaces
   the worst (at most one), with the same copy and a × 0.8 or × 1.2 mutation unless it
   would duplicate a learning rate. *n* = 1 is **every-epoch PBT**; *n* = 5,
   otherwise identical, is **every-five-epochs PBT**.

Windowed and cadenced PBT differ in several rules, so comparing them is a comparison
of policies, not of cadence. The 0.002 margin is a policy parameter, not a test.
