# Validation

This is the maintained policy and result ledger. It records what the current
repository supports, not what an earlier experiment hoped to show.

## Status vocabulary

- **Validated** — passed named, precommitted gates on pinned code, data, and a
  reproducible artifact. Validation is scoped to the exact claim.
- **Negative** — a sound evaluation failed its edge or deployment criterion.
- **Invalidated** — a prior result depended on a bug, leak, non-causal
  transformation, or invalid measurement.
- **Uncertain** — evidence is incomplete, stale, uncommitted, or not
  independently reproducible. Uncertain does not mean negative.

## Current result ledger

| Claim | Status | Current conclusion | Record |
| --- | --- | --- | --- |
| Deribit V2 15-minute straddle PnL/Sharpe | **Invalidated** | Per-bar SVI calibration jitter was treated as executable vega PnL. | [V2 post-mortem](postmortems/2026-06-v2-fake-pnl.md), commit bb509ad |
| V2 under honest option marking | **Negative** | DVOL and real-fill checks are losses; no deployable 15-minute edge is established. | [V2 post-mortem](postmortems/2026-06-v2-fake-pnl.md) |
| S3 short-vol-to-expiry baseline | **Negative** | The fill-covered subset was positive, but the frozen DSR gate failed and selection bias was material. | [S3/S5 post-mortem](postmortems/2026-06-s3-s5-bug-hunt.md) |
| S3 E2/H2 rescue | **Invalidated** | Terminal PnL was clipped and gains were added without a causal executable path. | [S3/S5 post-mortem](postmortems/2026-06-s3-s5-bug-hunt.md) |
| S5 Binance directional Sharpe 3.40 | **Uncertain** | Promising historical evidence, but no clean full-universe run on the final sizing fix and no independent PnL-realism check. | [S3/S5 post-mortem](postmortems/2026-06-s3-s5-bug-hunt.md) |
| TSRV / realized-volatility forecasting | **Uncertain** | The V2 incident does not invalidate forecast loss, but current QLIKE/R2 provenance and TransVol comparisons need regeneration. | [Project status](STATUS.md) |
| Current local SVI spot-feature sidecar | **Invalidated artifact** | It fails declared coverage and timestamp-stamping integration gates; do not use it as forecast or PnL evidence. | [Known issues](KNOWN_ISSUES.md) |
| ML close-model Sharpe 7.64 | **Invalidated** | Hourly regime features included future prices at 15-minute decisions. | [Close-model post-mortem](postmortems/2026-03-close-model-lookahead.md) |

No strategy in the ledger is currently **Validated** for live capital. That is
an evidence statement, not a claim that the project lacks future value.

## Safeguards now in source

- BacktestConfig defaults the V2 vega mark to honest DVOL movement rather than
  per-bar SVI fit movement.
- vol_arb/honest_pnl.py marks the same option contracts to causal Deribit
  premiums and separates tier-1 fills from tier-2 mark fallbacks.
- strict_pnl_realism_gate admits only finite tier-1 pairs and requires declared
  minimum trade count and coverage, sign agreement, bounded aggregate error,
  and per-trade correlation.
- scripts/regenerate_historical_v2.py runs that strict gate before writing the
  historical V2 trade-log artifact. The former override that could write a
  failed result has been removed.
- gate_no_fake_profit remains a sign-oriented research smoke check. Passing it
  is not result certification.
- The invalidated V2 preset is no longer exposed through the paper-engine CLI;
  retained parity configuration is explicitly historical.
- `TSRV/windowing.py` normalizes timestamps to nanoseconds before integer
  windowing. This prevents pandas 3 microsecond-backed indexes from collapsing
  an entire series into one 1970 bucket; regressions cover both TSRV and
  realized-quarticity callers.

These safeguards block the known false-PnL route. They do not certify V2, S3,
S5, or the forecasting stack.

## Evidence required for a promoted result

A result may be promoted to **Validated** only when all of the following hold:

1. A clean pinned commit for every result-affecting file.
2. A reproducible command plus input artifact names, hashes, and data window.
3. Explicit timestamp, feature, label, sizing, and execution semantics.
4. Train-only fitting for preprocessing, clipping, selection, and thresholds.
5. Independent fills or execution evidence for PnL; a series compared with
   itself is not evidence.
6. Declared fill coverage and an assessment of covered-versus-universe
   selection bias.
7. Precommitted statistical gates, including the number of trials or variants.
8. Per-period results, costs, drawdown definition, and non-compound/compound
   convention.
9. A mechanism-specific regression test for each prior incident class.
10. Failed gates retained beside passes.

## Maintained conclusions versus raw evidence

Maintained conclusions live in README.md and docs/. Subsystem READMEs describe
technical contracts.

Python source under experiments/ is versioned for provenance. Its caches,
plots, logs, data, and generated Markdown/text reports are ignored, as are
files under vol_arb/output/, vol_model/output/, results/, and similar run
directories. Those local files may support an audit, but they are not a
portable citation and may contradict later work.

Local raw reports used during this consolidation included:

- vol_arb/output/PHASE6_HONEST_VERDICT.md
- vol_arb/output/S3_REVALIDATION.md
- vol_arb/output/S3_GO_LIVE_SPEC.md
- vol_arb/output/S3_S5_BUG_FIX_REPORT.md
- vol_arb/output/S5_REGRESSION_DIAGNOSIS.md
- vol_arb/output/TRANSVOL_LEAKAGE_AUDIT.md
- experiments/REPORT_leakage_audit.md

They remain local, ignored evidence. Their scoped conclusions are preserved in
the post-mortems.

## Known gaps

The strict gate is now wired, but the broader backlog remains open: the
historical replay preset contains invalidated tuning, honest V2 input artifacts
are not immutably identified, S5 lacks independent PnL-realism evidence, and multiple June
defect proposals were never integrated into active callers. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).
