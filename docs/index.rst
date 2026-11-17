Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging
===========================================================

A pretrained Particle Transformer separates b, c and light-quark jets in simulated
ILC events. This project asks whether adapting the learning rate during fine-tuning,
with Population Based Training (PBT), gives a better tagger than a matched population
trained at constant learning rates, and whether any gain holds on data never used
for model selection.

**Status.**

* Main result: with three training seeds, PBT gives a lower composite mistag on the
  holdout than standard ParT training (Weaver's flat+decay learning-rate schedule) on every
  seed, by 0.67%, 0.85% and 0.91%; 0.81% on average, 95% interval 0.34–1.34% for the
  holdout sample. Both use the same compute (:ref:`pbt-vs-standard-part`).
* On ten times more training data (8M jets, one seed) the result reverses: PBT is 10.8%
  worse than standard ParT on the holdout. PBT lowers the learning rate too early
  (:ref:`pbt-vs-standard-part-8m`).
* Corrected: the earlier gains of 1–2% and 7.7% were measured against fixed learning
  rates that were too low, and do not measure the gain over a tuned baseline
  (:ref:`correction-of-earlier-comparisons`).
* Inconclusive: whether adapting more often helps; the ranking reversed on the
  holdout (:ref:`adaptation-frequency`).

.. toctree::
   :maxdepth: 1

   method
   evaluation
   results
   ongoing
   reproducibility

* :doc:`method`: data, model, training and the learning-rate policies.
* :doc:`evaluation`: splits, metrics, endpoints and statistical rules used everywhere.
* :doc:`results`: completed comparisons, including the large-dataset study, supporting
  studies and limitations.
* :doc:`ongoing`: sub-epoch adaptation.
* :doc:`reproducibility`: repository map, commands, experiment records and history.
