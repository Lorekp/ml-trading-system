# Scripts

This directory contains command-line orchestration and one-off research
drivers. Reusable logic belongs in an importable package, not here.

## Groups

- `aws/` — volatility-forecast worker launch, monitoring, and result-shard
  utilities. Account IDs, buckets, credentials, and machine paths come from
  arguments or environment variables.
- `forensic/` — incident reproduction and diagnostic programs.
- `s5_drawdown_reduction/` — S5 research drivers; outputs are not promoted
  results.
- Top-level scripts — data preparation, focused evaluations, sweeps, and
  historical artifact regeneration.

Use `regenerate_historical_v2.py` only for forensic V2 reproduction; it must
pass the strict tier-1 PnL-realism gate before writing its artifact. Use
`evaluate_s5_historical.py` only as a historical evaluator: it explicitly
reports independent PnL realism as not evaluated and cannot accept a result.

Script names inherited from earlier work may contain words such as `honest`,
`production`, `gate`, or `validated`. Those names do not establish status.
Only `docs/VALIDATION.md` can promote a result.

Generated Markdown, JSON, parquet, plots, caches, and logs belong under the
ignored output paths. If a run changes the project conclusion, summarize it in
`docs/STATUS.md` or `docs/VALIDATION.md`; do not add another session report.
