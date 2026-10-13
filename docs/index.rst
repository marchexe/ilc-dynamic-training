Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging
===========================================================

A pretrained Particle Transformer separates b, c and light-quark jets in simulated
ILC events. This project asks whether adapting the learning rate during fine-tuning,
with Population Based Training (PBT), gives a better tagger than a matched population
trained at constant learning rates, and whether any gain holds on data never used
for model selection.

**Status.**

* Completed: windowed PBT improved on fixed learning rates by about 1–2% in composite
  mistag, both on validation and on an independent holdout. The evidence is one
  training seed and has no event-level uncertainty.
* Inconclusive: whether adapting more often helps; the ranking reversed on the
  holdout.
* Running: a preregistered comparison on ten times more training data with a sealed
  test set.

.. toctree::
   :maxdepth: 1

   method
   evaluation
   results
   ongoing
   reproducibility

* :doc:`method`: data, model, training and the learning-rate policies.
* :doc:`evaluation`: splits, metrics, endpoints and statistical rules used everywhere.
* :doc:`results`: completed comparisons, supporting studies and limitations.
* :doc:`ongoing`: the large-dataset comparison.
* :doc:`reproducibility`: repository map, commands, experiment records and history.
