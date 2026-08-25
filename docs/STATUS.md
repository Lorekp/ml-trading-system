# Project status

**As of:** 2026-08-25

The project is active research infrastructure. It does not currently contain a
strategy validated for live capital.

## Current ledger

| Area | Status | Decision |
| --- | --- | --- |
| Realized-volatility data and features | **Uncertain but reusable** | TSRV, jump, quarticity, replay, and alignment code remain useful. Recompute pinned headline metrics before publishing them. |
| Volatility forecasting | **Uncertain** | QLIKE/R² research is not invalidated by the options-PnL incident, but metric provenance and final label semantics need a clean rerun. TransVol keep/retire is unresolved. |
| Deribit V2 15-minute straddle | **Invalidated / negative** | SVI calibration jitter was booked as tradable vega PnL. Honest DVOL and real-fill marking lose money. Do not deploy or quote the old positive results. |
| S3 short-vol-to-expiry | **Negative** | A fill-covered subset was positive, but the frozen DSR gate failed and selection bias was material. Later E2/H2 rescue claims were non-causal. |
| S5 Binance directional | **Uncertain / promising** | The historical full-universe result needs a fresh run on the current sizing fix. Current evaluators correctly report that independent PnL realism is not evaluated and block acceptance. Paper research only. |
| Execution engines | **Disabled as evidence** | They are replay and simulated-execution infrastructure. Legacy real connectors exist but no current strategy/configuration is authorized for capital, and the Deribit CLI has no real-order route. |

The definitions of **Validated**, **Negative**, **Invalidated**, and
**Uncertain** are in [VALIDATION.md](VALIDATION.md).

## What the V2 incident did and did not invalidate

It invalidated results whose PnL used per-bar SVI fit changes as executable
option-price movement, plus downstream sizing, filters, and labels chosen from
that trade log.

It did not automatically invalidate:

- raw trade ingestion and replay;
- TSRV and realized-volatility estimators;
- SVI calibration as a cross-sectional surface model;
- forecast losses such as QLIKE;
- the separate directional stack.

Those areas still need their own causal, reproducible validation. “Not affected
by this bug” is not the same as “validated.”

## Immediate priorities

1. Rotate the historical exchange credentials removed during this cleanup.
2. Run a clean, full-universe S5 evaluation on pinned code and persist its
   command, input hashes, annual breakdown, costs, and independent execution
   check.
3. Regenerate volatility-forecast metrics from pinned folds and corrected
   label stamping.
4. Integrate verified bug fixes into canonical modules; unimported wrappers do
   not count as fixes.
5. Pin and hash the option-price artifacts used by strict PnL-realism runs.

The concrete backlog is maintained in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Repository state after this cleanup

- Canonical documentation is under `docs/`; old session notes and generated
  reports have been removed from the active Git tree.
- Python experiment source is versioned. Its caches, logs, plots, datasets, and
  generated reports remain local.
- Large market data and model artifacts have not been moved or deleted.
- The static dashboard is dependency-free and mirrors this ledger without
  generated performance fixtures.
- Reusable packages under `src/` install as top-level packages; root workflow
  packages still run from a checkout pending a dedicated namespace refactor.

## Promotion rule

No future README or deployment document may quote a performance number unless
the same commit contains, or immutably identifies, the reproduction command,
data window/hash, cost model, fill coverage, per-period breakdown, and named
validation gates. Failed gates stay in the record.
