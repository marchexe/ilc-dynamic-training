Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging
===========================================================

This project studies whether adapting the learning rate during training improves a
Particle Transformer that separates b-, c- and light-quark (d) jets in simulated
International Linear Collider (ILC) events. A pretrained three-class tagger is
fine-tuned under matched conditions with either a constant learning rate per
population member or Population Based Training (PBT), which periodically copies
well-performing models and perturbs their learning rates. The outcome is measured as
background mistag rates at fixed signal efficiencies.

Research question
-----------------

Under a fixed training budget, does PBT-style learning-rate adaptation produce a
better b/c/light tagger than a matched population trained with constant learning
rates, and does any advantage persist on data that was never used for model
selection?

Summary of findings
-------------------

* **Windowed PBT versus a matched fixed-learning-rate population (50 epochs).**
  Windowed PBT reached a 1.26% lower population validation endpoint and a 1.36%
  lower composite mistag on an independent holdout of 2.8 million jets. The
  direction agrees on both evaluations, but this is a single training seed and no
  confidence interval at the level of physical events is available.
* **100 epochs.** Windowed PBT was again lower on validation (1.69%) and on the
  holdout (1.44%). The comparator is a single fixed-learning-rate member that was
  continued after it had been identified at epoch 50, not a matched population, so
  this comparison is asymmetric.
* **Adaptation frequency is inconclusive.** Adapting every epoch gave a lower
  validation endpoint than adapting every five epochs for both tested seeds, but the
  ordering reversed on the independent holdout for both seeds.
* **Negative and blocked results.** An adaptive learning-rate controller audited in
  shadow mode produced too few actionable decisions to be evaluated, and adaptation
  more often than once per epoch is blocked by the optimizer restart contract.
* **Ongoing.** A preregistered comparison of fixed-learning-rate training and
  windowed PBT on a training set ten times larger is in progress. No results from it
  are reported yet.

All differences are small, of order 0.004–0.006 percentage points of composite
mistag (about 1–2% relative), and must be read together with the
:doc:`limitations <results>`.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   method
   evaluation
   results
   ongoing
   reproducibility

* :doc:`method` — data, model, training protocol and the learning-rate policies.
* :doc:`evaluation` — data splits, metrics, endpoints, comparisons and the
  statistical policy. These definitions are used throughout.
* :doc:`results` — completed comparisons, the independent holdout evaluation,
  supporting studies, cost and limitations.
* :doc:`ongoing` — design and verified status of the large-dataset comparison.
* :doc:`reproducibility` — environment, verification tools, experiment records,
  research history and archived reports.

Source code: `github.com/marchexe/ilc-dynamic-training <https://github.com/marchexe/ilc-dynamic-training>`_.
