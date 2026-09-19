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

The deterministic training foundation and the reference PBT algorithm are
complete. Verified runs cover 50 and 100 full epochs. In the 100-epoch study,
the recorded final-10-epoch mean selection score is 0.327181% for PBT and
0.332571% for the matched fixed-``14e-6`` control, a 1.62% relative reduction.
This is control-proxy evidence; independent validation corroboration is still a
next step.

The immediate research direction is proxy-validation design followed by a
separately versioned, rule-based adaptive learning-rate controller. Neither is
part of the frozen reference algorithm.

.. toctree::
   :maxdepth: 1

   results
   method
   experiments
   status
   changelog
   development
