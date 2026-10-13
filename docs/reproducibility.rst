Reproducibility and Experiment Records
======================================

.. toctree::
   :maxdepth: 1

   experiments
   history
   archive/index

Repository layout
-----------------

.. code-block:: text

   configs/experiments/   experiment configurations (presets in configs/presets/)
   configs/prepared/      frozen protocols and preparation records of completed and ongoing studies
   networks/              the Particle Transformer used by every experiment
   scripts/training/      population training engine (PBT policies, checkpointing, reporting)
   scripts/validation/    run verifiers and standalone checkpoint evaluation
   scripts/experiments/   campaign launchers for the holdout evaluation and the large-dataset study
   scripts/publish/       exporters for published results and experiment records
   published/             curated public results, experiment bundles and figures
   datasets/manifests/    dataset manifests with row counts and SHA256 (data are not committed)
   weaver-core/           vendored and patched Weaver framework
   tests/                 unit, integration and provenance tests

Large artifacts (datasets, checkpoints, run directories, predictions) are not tracked.
Their identities are recorded by hash in the manifests and protocols.

Environment
-----------

On a GPU node with Python 3.10:

.. code-block:: sh

   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt

Machine-specific settings (GPU nodes and private data locations) are kept out of the
repository. Copy ``configs/site.example.json`` to ``configs/local/site.json``, which is
ignored by Git, or point ``ILC_SITE_CONFIG`` to another file. Tools that need it stop
with an explicit message when it is missing; tests use the template.

Data and starting checkpoint
----------------------------

Datasets are local artifacts. ``datasets/manifests/`` records the expected Parquet
files, row counts and hashes; the conversion scripts are in ``scripts/data/``. The
pretrained bundle records its provenance in
``checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/``.

Running experiments
-------------------

Inspect any configuration without training:

.. code-block:: sh

   .venv/bin/python scripts/training/run_pbt.py --config <config.yaml> --dry-run

Start and resume runs through the launcher, which records ownership and resumes only
incomplete epochs:

.. code-block:: sh

   .venv/bin/python scripts/launch/experiment.py start  --config <config.yaml>
   .venv/bin/python scripts/launch/experiment.py resume --config <config.yaml>

A completed windowed-PBT run can be continued into a **new** run directory by adding a
``continuation`` block that pins the source run's manifest hash and checkpoint
digest; the runner's ``--dry-run`` validates the pins. Never extend a completed run in
place.

The independent holdout evaluation and the large-dataset study each have a
one-command launcher with read-only modes:

.. code-block:: sh

   bash scripts/experiments/run_stage2a_holdout.sh            # holdout evaluation (completed)
   bash scripts/experiments/run_stage3_train8m.sh --dry-run   # large-dataset study: preflight only
   bash scripts/experiments/run_stage3_train8m.sh --status    # large-dataset study: progress

Their operating notes are ``configs/prepared/stage2a_holdout_v1/README.md`` and
``configs/prepared/stage3_train8m_v1/README.md``.

Verification
------------

.. code-block:: sh

   # Unit and integration tests (keep -t . so tests/research does not shadow scripts/research)
   .venv/bin/python -m unittest discover -s tests -t .

   # Replay and audit a completed run: full traversal, optimizer continuation, PBT decisions
   .venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/<run>

   # Re-verify the frozen holdout evaluation, its inputs and its results
   .venv/bin/python scripts/validation/verify_stage2a_provenance.py

Despite its historical name, ``verify_fixed_lr.py`` also verifies PBT runs and
continuations. None of these commands trains or evaluates a model.

Publishing results
------------------

.. code-block:: sh

   # Results tables, holdout summary and figures -> published/results/
   .venv/bin/python scripts/publish/export_results_summary.py

   # Experiment record bundle and its documentation page -> published/experiments/<slug>/
   .venv/bin/python scripts/publish/export_run.py runs/pbt/<run> --slug <slug> --title "Title"

   # Path-free copies of bundle files for the documentation site -> published/experiments/<slug>/public/
   .venv/bin/python scripts/publish/public_copies.py

All three read completed runs and never modify them. Model binaries are distributed
as release assets, not through Git.

Documentation
-------------

.. code-block:: sh

   python3 -m venv .venv-docs
   .venv-docs/bin/python -m pip install -r docs/requirements.txt
   .venv-docs/bin/python -m sphinx -n -W --keep-going -b html docs docs/_build/html

The documentation build uses only Sphinx and never imports training code.

Reproducibility boundaries
--------------------------

* Model, optimizer and loss-scaler state form one checkpoint bundle; copies and
  continuations transfer and verify the whole bundle.
* Member names are stable identities; a name such as ``lr_3e-6`` does not give the
  member's current learning rate after PBT.
* The windowed-PBT decision rules are frozen in
  ``scripts/training/pbt/reference/windowed_v2.py``; a new policy requires a new
  version.
* Run manifests, resolved configurations, event logs and checkpoint hashes are the
  primary evidence. Reports, figures and published tables are derived from them.
* Frozen protocols are never edited. Changes are recorded as administrative revisions
  that cite the original hash, for example
  ``configs/prepared/stage2a_holdout_v1/administrative_revision_1.json`` and
  ``configs/prepared/stage3_train8m_v1/gate_revision_1.json``.
