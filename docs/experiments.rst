Published experiments
=====================

This page presents the small set of experiments used for the current research
claims. Each entry is generated from recorded run evidence with
``scripts/publish/export_run.py``. Display names are independent of stable
server paths and historical member identifiers.

.. list-table:: Public experiment set
   :header-rows: 1
   :widths: 32 28 40

   * - Experiment
     - Role
     - Evidence
   * - :doc:`PBT v2 — 50 epochs <experiments/pbt-v2-50>`
     - Reference milestone
     - Complete 50-epoch population and continuation ancestor.
   * - :doc:`PBT v2 — 100 epochs <experiments/pbt-v2-100>`
     - Current primary result
     - Verified continuation through 100 full epochs.
   * - :doc:`Fixed-LR baseline — 50 epochs <experiments/fixed-lr-50>`
     - Matched control
     - Five-member fixed-learning-rate grid.
   * - :doc:`Fixed-LR baseline — 100 epochs <experiments/fixed-lr-100>`
     - Matched controls
     - The 14e-6 and 8.5e-6 branches continued to the same horizon.
   * - :doc:`Original pretrained checkpoints <experiments/pretrained-0-19>`
     - Starting-point evaluation
     - Recorded evaluation of pretrained epochs 0–19.

.. toctree::
   :hidden:

   experiments/pbt-v2-50
   experiments/pbt-v2-100
   experiments/fixed-lr-50
   experiments/fixed-lr-100
   experiments/pretrained-0-19

Publishing another run
----------------------

After a run is complete and verified, export its evidence and selected model
assets with one command:

.. code-block:: bash

   python scripts/publish/export_run.py runs/pbt/<run> \
     --slug <public-slug> --title "Public title"

The command reads the source run, writes ``published/experiments/<slug>`` and
updates its Sphinx page. It checks source manifest and summary hashes before and
after export. Model binaries remain local release assets; publication metadata,
plots and pages are suitable for Git and GitHub Pages.
