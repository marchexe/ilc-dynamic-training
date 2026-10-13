Reproducibility and Experiment Records
======================================

.. toctree::
   :maxdepth: 1

   experiments
   history

Repository map
--------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Path
     - Content
   * - :repo:`networks`, :repo:`weaver-core`
     - model; vendored and patched Weaver
   * - :repo:`scripts/training/pbt`
     - population training engine: ``planning/`` policies, ``state/`` checkpoints and
       continuation, ``execution/`` Weaver processes, ``reporting/`` endpoints and
       reports, ``runner.py``
   * - :repo:`scripts/launch/experiment.py`
     - start and resume a run from a configuration
   * - :repo:`scripts/validation`
     - run verifiers, checkpoint evaluation, proxy construction and qualification
   * - :repo:`scripts/experiments`
     - campaign launchers: ``stage2a`` holdout evaluation, ``stage2b`` ``train8m``
       preparation, ``stage3`` large-dataset study (names: :doc:`history`)
   * - :repo:`scripts/publish`, :repo:`scripts/reports`
     - exporters for ``published/``; plotting library
   * - :repo:`scripts/research`, :repo:`scripts/data`, :repo:`scripts/cluster`
     - audit tools; dataset conversion; GPU status
   * - :repo:`configs/experiments`
     - configurations of the reported experiments, smoke tests and the ongoing study
   * - :repo:`configs/prepared`
     - frozen protocols and preparation records, one directory per study
   * - :repo:`configs/site.example.json`
     - template for machine-specific settings, copied to the untracked
       ``configs/local/site.json``
   * - :repo:`datasets/manifests`
     - row counts and hashes of the local datasets
   * - :repo:`published`
     - ``results/`` tables and figures, ``experiments/`` run records, ``studies/`` audits
   * - :repo:`tests`
     - unit, integration and provenance tests

Data, checkpoints and run directories are not tracked; manifests and protocols record
their hashes.

Commands
--------

.. code-block:: sh

   # Environment (Python 3.10, GPU node)
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt
   cp configs/site.example.json configs/local/site.json

   # Inspect, start and resume a run
   .venv/bin/python scripts/training/run_pbt.py --config <config.yaml> --dry-run
   .venv/bin/python scripts/launch/experiment.py start  --config <config.yaml>
   .venv/bin/python scripts/launch/experiment.py resume --config <config.yaml>

   # Large-dataset study (read-only modes)
   bash scripts/experiments/run_stage3_train8m.sh --dry-run
   bash scripts/experiments/run_stage3_train8m.sh --status

   # Verify
   .venv/bin/python -m unittest discover -s tests -t .
   .venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/<run>   # any completed run
   .venv/bin/python scripts/validation/verify_stage2a_provenance.py        # holdout evaluation

   # Publish
   .venv/bin/python scripts/publish/export_results_summary.py
   .venv/bin/python scripts/publish/export_run.py runs/pbt/<run> --slug <slug> --title "<title>"
   .venv/bin/python scripts/publish/public_copies.py

   # Documentation
   .venv/bin/python -m sphinx -n -W -b html docs docs/_build/html

Completed runs are never modified. A continuation is a new run that pins the source
manifest hash. Frozen protocols are never edited; changes are recorded as revisions citing
the original hash (for example :repo:`configs/prepared/stage3_train8m_v1/gate_revision_1.json`).
