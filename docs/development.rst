Development
===========

Setup
-----

Create the project environment on a GPU node and install the pinned runtime:

.. code-block:: sh

   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt -c requirements-lock.txt

Datasets are local artifacts. Their expected layouts and hashes are recorded in
``datasets/manifests/``; the pretrained bundle records its provenance under
``checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/``.

Documentation
-------------

Build documentation from the repository root in a separate environment:

.. code-block:: sh

   python3 -m venv /tmp/march-docs
   /tmp/march-docs/bin/python -m pip install -r docs/requirements.txt
   /tmp/march-docs/bin/python -m sphinx -n -W --keep-going -b html docs /tmp/march-docs-html

The documentation build uses only Sphinx and does not import training code or
launch workers.

Publishing experiments
----------------------

After read-only verification, create a curated bundle and its Sphinx page:

.. code-block:: sh

   .venv/bin/python scripts/publish/export_run.py runs/pbt/<run> \
     --slug <slug> --title "Public title" \
     --resume-bundle protected_best --release-tag <release-tag>

The publisher copies recorded metrics, configuration, plots and selected model
assets into ``published/experiments/<slug>``. It verifies that source manifest
and summary hashes do not change. Pass ``--upload-release-assets`` only when the
generated bundle has been reviewed and ``gh`` is authenticated; otherwise the
bundle contains the exact create/upload commands in ``release-command.txt``.

Tests and verification
----------------------

Run repository tests from the root:

.. code-block:: sh

   .venv/bin/python -m unittest discover -s tests -t .

Keep ``-t .`` so ``tests/research`` cannot shadow ``scripts/research``.

Read-only verification of the completed PBT references:

.. code-block:: sh

   .venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/windowed_pbt_v2
   .venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/windowed_pbt_v2_100epochs

Despite its historical filename, the verifier also supports windowed PBT and
completed-run continuation. These commands do not launch training or inference.

Use ``scripts/training/pbt/rebuild_artifacts.py <run>`` to regenerate reports
without training. Reporting is read-only with respect to ``manifest.json``.

Reproducibility boundaries
--------------------------

* The model, optimizer and optional AMP scaler form one checkpoint bundle.
  Exploit and continuation operations preserve the bundle and verify identities.
* Historical member names are stable identities. A name such as ``lr_3e-6``
  does not imply the member's current LR after PBT mutation.
* ``windowed_pbt_v2`` decision semantics are frozen in
  ``scripts/training/pbt/reference/windowed_v2.py``. New policies require new
  strategy versions.
* Completed-run continuation pins the source manifest hash and checkpoint-state
  digest. The 100-epoch reference depends on the 50-epoch source path and
  evidence; do not rename or rewrite either run.
* Manifests, resolved configs, event logs and checkpoint hashes are primary
  evidence. Reports and plots are derived artifacts.

Inspect a launch without executing it with the runner's ``--dry-run`` option.
Actual training should use a reviewed experiment config and a new output path;
never extend a completed run in place.
