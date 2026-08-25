# Test policy

The default suite covers `tests/` and `deribit_paper/tests/` while excluding
tests marked `integration`.

- Unit and regression tests must run without private market-data artifacts.
- Data-, cache-, model-, or exchange-dependent checks belong in
  `tests/integration/` and carry both `integration` and `slow` markers.
- Missing integration artifacts must produce an explicit skip naming the
  required path or environment variable.
- A bug regression must import and exercise the canonical project symbol.
  Reimplementing “before” and “after” behavior inside a test file does not
  verify the repository and must not be committed as a passing test.
- Exact performance baselines require pinned input hashes. Otherwise assert a
  mechanism-specific invariant and keep the performance comparison in a
  versioned integration artifact.

Run the fresh-checkout suite with:

```bash
python -m pytest -q
```

Run local integration checks explicitly with:

```bash
python -m pytest -q -m integration
```
