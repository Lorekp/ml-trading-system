# Archived orphan tests (2026-05-01)

Two unrelated test trees archived together because they were both
blocking `pytest --collect-only`:

1. **`tests/test_live/`** (10 files, 1634 LOC) — Bybit-demo
   paper-trading harness; superseded by `deribit_paper/`.
2. **`test_microstructure.py`** (211 LOC) — orphan tests for early
   versions of `vol_model/features/microstructure.py` whose API has
   since been substantially redesigned (`_vpin_ring_buffer` →
   `_vpin_rolling` with different signature; `align_lob_to_bars`
   removed entirely). The current production code uses
   `MicrostructureFeatures` (the class) via `vol_model.features.engineer`,
   not the lower-level helper functions these tests covered.

The first archive (test_live) is documented in detail below; the
second is a single file that was orphaned the same way (test
references functions that no longer exist in the module under test).

---

## What `tests/test_live/` tested

## What this tested (test_live/)

A Bybit-demo paper-trading harness whose source modules lived under
`live/` (a top-level `live` package). The harness had:

- `live.config` exposing `CircuitBreakerConfig`, `LiveModeConfig`,
  `PaperConfig`, `RiskConfig`, `LiveConfig`, `TelegramConfig`,
  `SchedulerConfig`, `ExchangeConfig` (Pydantic models).
- `live.db.LiveDB` — SQLite state store.
- `live.exchange.MockExchangeAdapter` — synthetic exchange for tests.
- `live.models.OHLCVBar`, `Side`, `Signal` — domain types.

The tests covered the engine, scheduler, position tracker, risk
manager, executor (paper + live), and DB layer of that older system.

## Why archived

The Bybit-demo paper-trading harness was **superseded by `deribit_paper/`**
during the Stage 17-23 cycle (2026-04-10..2026-04-13 in the
documentation timeline). The `live/` package itself was either
deleted or substantially refactored — `src/live/config.py` survives
but exposes a totally different class set (`BreakerConfig`,
`PortfolioConfig`, `S8Config`, `MonitorConfig`, etc.).

The result was that `tests/test_live/conftest.py`'s imports
(`CircuitBreakerConfig` etc.) became unresolvable and `pytest --collect-only`
errored out across the entire `tests/test_live/` subtree, blocking
clean test discovery for the rest of the repo.

The tests have not been updated to the current architecture and have
not been run successfully since at least 2026-04 (they may pre-date
the Stage 17 audit cycle entirely — git log shows they all date to
the initial repo commit `54b44b6`).

## Where the equivalent current tests live

The current production live engine is `deribit_paper/`, with its
own test suite at:

- `deribit_paper/tests/` — 555 tests covering engine wiring,
  predictor, straddle flow, order manager, risk limits,
  kill-switch, harness writer, etc.
- `tests/test_sidecars.py` — sidecar loaders shared with backtester.
- `tests/test_h15_alignment_harness.py` — Stage A live-vs-V2 diff.
- `tests/test_refresh_live_sidecars.py` — monthly bootstrap refresh.
- `tests/test_refresh_aligner_monthly.py` — full aligner orchestrator.

## How to revive

If anyone needs to reconstruct the older Bybit-demo system:

```sh
# View the archive contents
ls backups/2026-05-01_orphan_tests_test_live/test_live/

# Restore to original location
git mv backups/2026-05-01_orphan_tests_test_live/test_live tests/test_live

# Find the matching live/ package code in git history
git log --all --diff-filter=D -- 'live/*.py'
```

Note that to actually run them you'd also need the original `live/`
package source code, which has been replaced.
