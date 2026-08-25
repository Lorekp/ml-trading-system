# Experiments

This directory is the tracked provenance record for research programs. Python
source belongs in version control even when an experiment was inconclusive or
invalidated; its outcome and status should be recorded in the file docstring or
the canonical research documentation.

Many legacy filenames, docstrings, and stdout messages predate the current
validity audit. Words such as `production`, `validated`, `ready`, or
`honest` inside an experiment are untrusted historical language, not current
status. Only `docs/VALIDATION.md` may promote a result.

Preserved experiments may also retain the machine-specific data roots used for
their original run. Treat those paths as provenance, not supported defaults;
new or promoted workflows must accept an artifact root explicitly.

Generated state does not belong in version control. Keep caches, datasets,
models, logs, plots, reports, and serialized results under the ignored artifact
paths already defined in `.gitignore`, including `experiments/cache/` and
`experiments/.cache/`.

Code reused by tests or more than one workflow belongs under `src/`. Experiment
programs may import reusable code from there, but production and test code must
not import from `experiments/`.

`experiments/invalidated/` preserves programs whose evaluation logic was
shown to be non-causal, self-validating, or otherwise unsuitable for result
promotion. Keep the warning in each quarantined module; maintained workflows
must not import that namespace.

`experiments/legacy_directional/` contains early standalone audits and a
hard-coded milestone report moved out of the reusable package tree. Several
embed obsolete data layouts or historical recommendations and are preserved
only as provenance.
