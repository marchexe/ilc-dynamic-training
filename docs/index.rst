ILC jet-flavour tagging with adaptive training
==============================================

This project fine-tunes a Particle Transformer to distinguish ``bb``, ``cc``
and light-quark ``dd`` jets in simulated ILC events. The research question is
whether Population Based Training (PBT) can choose useful learning-rate
trajectories while preserving reproducible model, optimizer and AMP-scaler
state.

The current reference is ``windowed_pbt_v2``: five persistent population
members are evaluated every full epoch, while copy/mutation decisions use
multi-epoch evidence. Fixed-learning-rate runs are matched controls rather than
the primary method.

Current stage
-------------

The deterministic training foundation and reference PBT algorithm are complete,
with verified 50- and 100-full-epoch runs. A completed offline
:doc:`proxy-validation qualification <proxy_validation>` study evaluated 38
existing checkpoints on nine frozen candidates and selected
``representative_60k`` as the control signal for shadow adaptive-LR testing.

The next stage is offline controller replay and shadow measurement using
smoothing, short-window trends, patience and cooldown. No adaptive-LR policy is
part of the frozen reference algorithm, and the proxy study establishes
measurement quality rather than a training improvement claim.

.. toctree::
   :maxdepth: 1

   results
   proxy_validation
   method
   experiments
   status
   changelog
   development
