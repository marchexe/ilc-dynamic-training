# Nightly known issues

Full existing test suite run before tonight's changes
(`.venv/bin/python3 -m unittest discover -s tests -p "test_*.py"`):
166 tests, **all pass**, 1 skip (`test_ray_backend_dependency_error_is_clear`
— skipped because Ray is installed in this environment, not a failure).

No unrelated pre-existing failures were found. Nothing to record here beyond
this baseline confirmation.

## Test-discovery gotcha introduced tonight (not a code bug -- a naming collision)

`tests/research/` (this audit's new tests) and `scripts/research/` (this
audit's new source package) share the name `research`. `python -m unittest
discover -s tests -p "test_*.py"` (the command used for the baseline above)
implicitly sets `top_level_dir` to `tests/`, so it imports
`tests/research/test_*.py` as top-level module `research.test_*` --
colliding with the *other* `research` package already on `sys.path` via
`scripts/` (inserted by `tests/helpers.py`), and failing with
`ModuleNotFoundError: No module named 'research.run_proxy_audit'`.

**Fix: always pass `-t .` (the project root) when running the suite from
now on:**

```
.venv/bin/python3 -m unittest discover -s tests -t . -p "test_*.py"
```

This resolves the new tests as `tests.research.test_*`, distinct from the
`research` package under `scripts/`. Verified: 214 tests (166 existing + 48
new), all pass, 1 skip, with `-t .`. Neither package was renamed to avoid
the collision -- both `scripts/research/` and `tests/research/` are the
exact paths the task suggested, and adding `-t .` is a one-flag fix with no
source changes needed.
