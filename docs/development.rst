Build and checks
================

Build documentation from the repository root in a separate environment:

.. code-block:: sh

   python3 -m venv /tmp/march-docs
   /tmp/march-docs/bin/python -m pip install -r docs/requirements.txt
   /tmp/march-docs/bin/python -m sphinx -n -W --keep-going -b html docs /tmp/march-docs-html

The documentation build uses only Sphinx and does not import training modules,
scan run directories, or launch workers. Training requirements are unchanged.

Run repository tests from the root:

.. code-block:: sh

   .venv/bin/python -m unittest discover -s tests -t .

Keep ``-t .``: otherwise ``tests/research`` can shadow ``scripts/research``.
This reusable fix comes from ``docs/nightly_known_issues.md``; historical test
counts and nightly execution plans are intentionally omitted.

Read-only verification of the completed PBT references:

.. code-block:: sh

   .venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/windowed_pbt_v2
   .venv/bin/python scripts/validation/verify_fixed_lr.py runs/pbt/windowed_pbt_v2_100epochs

Despite its historical filename, the verifier also supports windowed PBT and
completed-run continuation. These commands do not launch training or inference.
