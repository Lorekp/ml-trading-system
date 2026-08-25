# Documentation

This directory contains the repository's canonical human-authored
documentation.

- [STATUS.md](STATUS.md) — current research and deployment status.
- [ARCHITECTURE.md](ARCHITECTURE.md) — code and data-flow map.
- [VALIDATION.md](VALIDATION.md) — result vocabulary, gates, and evidence
  policy.
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — unresolved correctness and
  reproducibility work.
- [postmortems/](postmortems/) — scoped incident records.

Subsystem READMEs explain how a module works. Files under `*/output/`,
`results/`, and ignored experiment artifact trees are generated evidence,
not documentation and not a current conclusion.

When a report changes the project's conclusion, summarize the result in
`STATUS.md` or `VALIDATION.md` and add a post-mortem when an earlier claim
was invalidated. Do not add another session handoff or root-level result log.
