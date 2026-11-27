Adaptive Learning-Rate Training for ILC Jet-Flavour Tagging
===========================================================

A pretrained Particle Transformer (ParT) separates b, c and light-quark jets in simulated
ILC events. This project asks whether adapting the learning rate during fine-tuning, with
Population Based Training (PBT), gives a better tagger than standard ParT training (Weaver's
flat+decay learning-rate schedule), and whether any gain holds on data never used for model
selection. All results are on Test (2.8M), 2.8M jets that are used only after the checkpoints are
fixed.

Findings
--------

1. **On 800k training jets PBT wins.** With three training seeds and the same compute, PBT
   gives a lower composite mistag than standard ParT on every seed: 0.67%, 0.85% and 0.91%;
   0.81% on average, 95% interval 0.34–1.34% for the Test (2.8M) sample
   (:ref:`pbt-vs-standard-part`).
2. **On 8M training jets PBT loses.** It is 12.4% worse than standard ParT (one seed), because
   it lowers the learning rate too early (:ref:`pbt-vs-standard-part-8m`). Cool-down probes and
   a final decay close 78% of the gap, to 2.7% (:ref:`probe-scored-pbt`).
3. **The reason is overfitting.** At 800k the model starts to memorise its training data, and
   lowering the learning rate early helps. At 8M it does not overfit, and a long phase at a high
   learning rate followed by a decay is better (:ref:`no-overfitting-at-8m`).
4. **An ensemble of two standard ParT models is 5.0% better** on Test (2.8M) with 8M training jets than the better
   single model, on each of three training seeds (5.05%, 4.86%, 5.14%; mean interval
   4.17–5.84%). The two models are trained separately, for example two members of the
   learning-rate scan that standard ParT training runs anyway (:ref:`prediction-ensemble`). At
   800k ensembles give no gain (:ref:`ensemble-800k`).

In short: with little training data PBT helps a little and ensembles do not; with much data an
ensemble of standard ParT models helps and PBT does not (:ref:`two-regimes`).

.. figure:: ../published/results/sel1m/figures/train8m_holdout_overview.png
   :alt: Test (2.8M) composite mistag of every 8M run and of the ensemble
   :width: 100%

   Test (2.8M) composite mistag on 8M training jets (lower is better).

Corrected: earlier gains of 1–2% and 7.7% were measured against fixed learning rates that were
too low (:ref:`correction-of-earlier-comparisons`). Inconclusive: whether adapting more often
helps (:ref:`adaptation-frequency`).

Contents
--------

.. toctree::
   :maxdepth: 1

   report
   results
   ongoing
   method
   evaluation
   reproducibility

* `Research overview <overview.html>`_: the study as a short paper with figures; `experiment log <log.html>`_ with every trial and run.
* :doc:`report`: every trial in time order, with dates, configuration, short results and an
  interpretation.
* :doc:`results`: all completed comparisons, the 8M study, supporting studies and limitations.
* :doc:`ongoing`: tests in progress.
* :doc:`method`: data, model, training and the learning-rate policies.
* :doc:`evaluation`: splits, metrics, endpoints and statistical rules used everywhere.
* :doc:`reproducibility`: repository map, commands, experiment records and history.
