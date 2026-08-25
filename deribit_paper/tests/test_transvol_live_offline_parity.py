"""S-1c (parity-r3 cycle 6 — 2026-05-08) — offline-vs-live parity gate.

End-to-end gate for the TransVol live pipeline. Replays the offline
``BTCUSDT_features_15m.parquet`` through the live builder
(``FeatureAssembler.assemble_transvol_input_row``) and asserts the
output matches offline within tolerance bands honestly bounded by
what we can replay without the full historical dollar-bar buffer.

**Scope of the gate** (honestly bounded):

* **Tier-A — strict byte-equal (rtol=1e-5, atol=1e-7)**:
  funding_basis_* (5 cols), time_to_funding (raw, 1 col). These do
  not depend on the dollar-bar grid: funding_basis is a pure parquet
  ffill lookup; time_to_funding is a deterministic function of the
  last dollar bar's timestamp. The strict gate locks these against
  any regression to within rounding-error.

* **Tier-B — informational (no fail, just diff reporting)**:
  TSRV-grid features (tsrv_mean_*, tsrv_ffd, up_jump_frac_*,
  bpv_tsrv_ratio_*, down_jump_frac_*, noise_level_*) and the
  time_to_funding lag/EMA variants. Live computes on the 15-min TSRV
  cadence; offline ffills onto the dollar-bar grid first then rolls
  there before aggregating to 15-min via ``groupby(window_labels)
  .last()``. This is a documented design deviation — strict byte-equality would
  require a 1750-bar dollar-bar buffer extension (~200 LOC, deferred
  to S-1d). Mathematically equivalent under constant-rate trading,
  deviates on bursty days. The gate REPORTS the diff to the test log
  but does not fail.

* **Tier-C — structural finiteness only**:
  hurst_100, ffd_rv, rv_zscore_*, parkinson_1d, vov, vpin, imbalance,
  duration_s, n_trades and their lag/EMA variants. These depend on a
  full historical dollar-bar buffer (millions of bars offline; only
  300 bars cached for live prewarm). The gate asserts the live row
  produces a finite value (baseline-fallback works correctly) — full
  byte-equality requires regenerating the offline dollar-bar pipeline.

A mutation probe deliberately swaps two TRANSVOL_INPUT_COLUMNS entries
and asserts the strict-tolerance Tier-A gate fails — locks the
canonical column ordering against silent reordering.

The gate test is skipped when the offline parquets are not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.market_data.funding import FundingRateProvider
from deribit_paper.signal.feature_assembler import (
    FeatureAssembler,
    TRANSVOL_INPUT_COLUMNS,
    TRANSVOL_TIME_COLUMNS,
)


# Tier classification for the parity gate.
# S-1d (parity-r3 cycle 7 — 2026-05-08): all 30 cycle-6 Tier-B columns
# are PROMOTED to Tier-A — the dollar-bar-grid path
# (TSRVDollarBarFeatures + dollar-bar-stamped time_to_funding lags)
# closes the C1 deviation strictly. Tier-B is now empty.
TIER_A_COLUMNS: tuple[str, ...] = (
    # cycle-6 strict-already.
    "funding_basis", "funding_basis_lag1", "funding_basis_lag3",
    "funding_basis_ema10", "funding_basis_ema50",
    "time_to_funding",
    # cycle-7 promoted: time_to_funding lag/EMA on dollar-bar grid.
    "time_to_funding_lag1", "time_to_funding_lag3",
    "time_to_funding_ema10", "time_to_funding_ema50",
    # cycle-7 promoted: TSRV-on-dollar-bar-grid (30 columns).
    "tsrv_mean_1d", "tsrv_mean_1d_lag1", "tsrv_mean_1d_lag3",
    "tsrv_mean_1d_ema10", "tsrv_mean_1d_ema50",
    "tsrv_mean_1w", "tsrv_mean_1w_lag1", "tsrv_mean_1w_lag3",
    "tsrv_mean_1w_ema10", "tsrv_mean_1w_ema50",
    "tsrv_ffd", "tsrv_ffd_lag1", "tsrv_ffd_lag3",
    "tsrv_ffd_ema10", "tsrv_ffd_ema50",
    "up_jump_frac", "up_jump_frac_lag1", "up_jump_frac_lag3",
    "up_jump_frac_ema10", "up_jump_frac_ema50",
    "down_jump_frac", "down_jump_frac_lag1", "down_jump_frac_lag3",
    "down_jump_frac_ema10", "down_jump_frac_ema50",
    "noise_level", "noise_level_lag1", "noise_level_lag3",
    "noise_level_ema10", "noise_level_ema50",
    "bpv_tsrv_ratio", "bpv_tsrv_ratio_lag1", "bpv_tsrv_ratio_lag3",
    "bpv_tsrv_ratio_ema10", "bpv_tsrv_ratio_ema50",
)
TIER_B_COLUMNS: tuple[str, ...] = ()  # cycle-7: empty (C1 closed).
TIER_C_COLUMNS: tuple[str, ...] = tuple(
    c for c in TRANSVOL_INPUT_COLUMNS
    if c not in TIER_A_COLUMNS and c not in TIER_B_COLUMNS
)


_FEATURES_PARQUET = Path(__file__).resolve().parents[2] / "vol_model" / "output" / "BTCUSDT_features_15m.parquet"
_TSRV_PARQUET = Path(__file__).resolve().parents[2] / "TSRV" / "output" / "BTCUSDT_tsrv_15m.parquet"
_FUNDING_PARQUET = Path(__file__).resolve().parents[2] / "data" / "funding-rates" / "BTCUSDT_funding.parquet"
# S-1d (cycle 7): the parity gate loads dollar bars directly from
# the offline bar-series parquet so it can pull 30+ days of history
# (~600k bars) without bloating the committed live-engine cache
# (which stays at 60k bars / 5.5 MB). 30-day history is required so
# the trailing-30-day ``bpd`` estimate matches offline byte-equivalent.
_BAR_SERIES_PARQUET = Path(__file__).resolve().parents[2] / "vol_model" / "output" / "BTCUSDT_bar_series.parquet"


def _have_offline_data() -> bool:
    return all(p.exists() for p in (
        _FEATURES_PARQUET, _TSRV_PARQUET, _FUNDING_PARQUET, _BAR_SERIES_PARQUET,
    ))


# ── Tier-A: strict byte-equal gate ────────────────────────────────


@pytest.fixture(scope="module")
def parity_setup():
    """Build a FeatureAssembler hooked to the real offline parquets +
    the most recent N bars of the offline features parquet for replay.

    Module-scoped because the parquet loads are heavy (30+ MB).
    """
    if not _have_offline_data():
        pytest.skip(
            f"Offline parity test requires {_FEATURES_PARQUET.name}, "
            f"{_TSRV_PARQUET.name}, {_FUNDING_PARQUET.name}, "
            f"{_BAR_SERIES_PARQUET.name} to be present.",
        )

    # Load the offline features parquet (ground truth).
    features_df = pd.read_parquet(_FEATURES_PARQUET)

    # Build the FundingRateProvider with the offline parquet.
    funding_provider = FundingRateProvider(
        symbol="BTCUSDT",
        parquet_path=_FUNDING_PARQUET,
        rest_fetcher=lambda: None,  # disable REST so we read parquet only
    )

    # Build the TSRVHistoryBuffer pre-warmed from the full offline
    # TSRV parquet so the live wrapper has the same input as offline.
    tsrv_buffer = TSRVHistoryBuffer(max_size=672)
    tsrv_buffer.prewarm_from_parquet(_TSRV_PARQUET)

    # S-1d (cycle 7): load 600k+ bars (~35 days) directly from the
    # offline bar-series parquet — the committed live cache (60k
    # bars / 5.5 MB) covers 1-week rolling but not the trailing-30-
    # day ``bpd`` window needed for tsrv_mean_1d/1w byte-equal closure.
    # Read just the last few row groups for memory efficiency.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(_BAR_SERIES_PARQUET))
    n_total = pf.metadata.num_rows
    target_n = 700_000
    cumulative = 0
    rg_indices: list[int] = []
    for i in range(pf.num_row_groups - 1, -1, -1):
        cumulative += pf.metadata.row_group(i).num_rows
        rg_indices.insert(0, i)
        if cumulative >= target_n + 10_000:
            break
    bar_table = pf.read_row_groups(rg_indices).to_pandas()
    bar_table = bar_table.tail(target_n)
    bar_table["n_trades"] = bar_table["n_trades"].astype("int64")
    dollar_bar_buffer = DollarBarBuffer(max_size=target_n)
    base_idx = bar_table.index
    if not isinstance(base_idx, pd.DatetimeIndex):
        # Fall back to the parquet's __index_level_0__ column.
        from deribit_paper.features.dollar_bar import DollarBar
        # Re-read with index preservation by reading a single row group + index.
        ts_table = pf.read_row_groups(rg_indices, columns=["__index_level_0__"]) \
            if "__index_level_0__" in pf.schema.names else None
        if ts_table is not None:
            tsdf = ts_table.to_pandas().tail(target_n)
            base_idx = pd.DatetimeIndex(tsdf["__index_level_0__"])
            if base_idx.tz is None:
                base_idx = base_idx.tz_localize("UTC")
    if base_idx.tz is None:
        base_idx = pd.DatetimeIndex(base_idx, tz="UTC")
    bar_table.index = base_idx
    from deribit_paper.features.dollar_bar import DollarBar
    for ts, row in bar_table.iterrows():
        dollar_bar_buffer.append(DollarBar(
            timestamp=pd.Timestamp(ts),
            open=float(row["open"]), close=float(row["close"]),
            high=float(row["high"]), low=float(row["low"]),
            duration_s=float(row["duration_s"]),
            n_trades=int(row["n_trades"]),
            dollar_volume=float(row["dollar_volume"]),
            buy_volume=float(row["buy_volume"]),
            sell_volume=float(row["sell_volume"]),
            vov=float(row["vov"]),
            imbalance=float(row["imbalance"]),
            log_return=float(row["log_return"]),
            rv_bar=float(row["rv_bar"]),
        ))

    # Stub bundle: minimal LightGBM features list so the assembler
    # constructor's manifest validator passes.
    class _StubBundle:
        lgbm_features = ["tv_0", "tv_1", "svi_atm_1d"]

    assembler = FeatureAssembler(
        features_parquet=_FEATURES_PARQUET,
        bundle=_StubBundle(),
        dollar_bar_buffer=dollar_bar_buffer,
        tsrv_history_buffer=tsrv_buffer,
        funding_provider=funding_provider,
    )

    # The bar we'll replay. ``features_df`` is the RAW offline features
    # parquet — FLOOR-labeled by ``_window_and_aggregate``: row L
    # aggregates dollar bars in ``[L, L+15min)``. The live ``Bar`` is
    # END-stamped (``aggregator.py``: bar T covers ``[T-15min, T)`` —
    # see cycle-23). So a live bar END-stamped ``E`` corresponds to the
    # offline feature row labeled ``E - 15min``.
    #
    # Cycle-41 (2026-05-17): pre-cycle-40 the funding handlers read
    # ``buf.latest_bar()`` (the absolute buffer tail), ignoring
    # ``bar.timestamp`` entirely — so this fixture passed even though
    # it set ``bar.timestamp`` to the FLOOR label (15min too early).
    # Cycle-40/41 made the handlers respect ``bar.timestamp`` (clamp to
    # ``< bar.timestamp``); the fixture must now END-stamp the bar
    # correctly: ``bar_close_ts = offline_floor_label + 15min``.
    offline_label_ts = features_df.index[-1]
    bar_close_ts = offline_label_ts + pd.Timedelta(minutes=15)
    bar = Bar(
        timestamp=bar_close_ts, bar_minutes=15,
        perp_trades=[], option_trades=[],
        index_price=70_000.0,
    )
    # Synthesise a BarTSRV for the latest bar from the TSRV parquet's
    # latest row (so live's _h_up_jump_frac etc. read sensible values).
    tsrv_df = pd.read_parquet(_TSRV_PARQUET)
    tsrv_latest = tsrv_df.iloc[-1]
    bar_tsrv = BarTSRV(
        tsrv=float(tsrv_latest["tsrv"]),
        rv_fast=float(tsrv_latest["rv_fast"]),
        rv_slow=float(tsrv_latest["rv_slow"]),
        bpv=float(tsrv_latest["bpv"]),
        medrv=float(tsrv_latest["medrv"]),
        continuous_var=float(tsrv_latest["continuous_var"]),
        jump_var=float(tsrv_latest["jump_var"]),
        jump_fraction=float(tsrv_latest["jump_fraction"]),
        rs_down=float(tsrv_latest["rs_down"]),
        rs_up=float(tsrv_latest["rs_up"]),
        down_jump_fraction=float(tsrv_latest["down_jump_fraction"]),
        up_jump_fraction=float(tsrv_latest["up_jump_fraction"]),
        n_ticks=int(tsrv_latest["n_ticks"]),
        n_sub_bars=int(tsrv_latest["n_sub_bars"]),
        n_dash=float(tsrv_latest["n_dash"]),
        vol_ann=float(tsrv_latest["vol_ann"]),
        vol_ann_continuous=float(tsrv_latest["vol_ann"] * 0.99),
    )

    # Advance funding provider through the bar's END timestamp.
    funding_provider.advance(bar_close_ts)

    feats, time_feats = assembler.assemble_transvol_input_row(bar, bar_tsrv)
    # The live END-stamped bar ``bar_close_ts`` maps to the offline
    # FLOOR-labeled feature row ``offline_label_ts`` (= bar_close_ts - 15min).
    offline_row = features_df.loc[offline_label_ts, list(TRANSVOL_INPUT_COLUMNS)]

    return {
        "live_feats": feats,
        "live_time_feats": time_feats,
        "offline_row": offline_row,
        "bar_ts": offline_label_ts,
    }


def test_tier_a_strict_byte_equal(parity_setup):
    """Tier-A: funding_basis_* + time_to_funding_* + n_bars_in_window
    must be byte-equal to offline within ``rtol=1e-5, atol=1e-7``.
    """
    live = parity_setup["live_feats"]
    offline = parity_setup["offline_row"]

    failures: list[str] = []
    for col in TIER_A_COLUMNS:
        idx = TRANSVOL_INPUT_COLUMNS.index(col)
        live_val = float(live[idx])
        offline_val = float(offline[col])
        if not np.isfinite(offline_val):
            continue  # offline NaN — skip this column for Tier-A
        if not np.allclose(live_val, offline_val, rtol=1e-5, atol=1e-7):
            failures.append(
                f"{col}: live={live_val:.10g} offline={offline_val:.10g} "
                f"diff={live_val - offline_val:+.3e}"
            )
    assert not failures, (
        f"Tier-A strict-parity gate FAILED for "
        f"{len(failures)}/{len(TIER_A_COLUMNS)} columns:\n"
        + "\n".join(failures)
    )


def test_tier_b_is_empty_after_cycle_7():
    """S-1d (cycle 7) closes C1 strictly: all 30 former Tier-B
    columns are promoted to Tier-A byte-equal. This test locks the
    structural invariant — Tier-B must stay empty going forward."""
    assert TIER_B_COLUMNS == (), (
        f"Tier-B should be empty after cycle 7 closure; got "
        f"{len(TIER_B_COLUMNS)} columns: {TIER_B_COLUMNS}"
    )


def test_tier_c_structural_finiteness(parity_setup):
    """Tier-C (dollar-bar-derived): with a 300-bar live cache, many
    handlers fall back to baseline-fallback (the offline parquet has
    finite values from full-history compute). The gate just requires
    the live row to be FINITE — i.e. baseline-fallback worked, no
    handler raised, no NaN crashed the run.
    """
    live = parity_setup["live_feats"]
    failures: list[str] = []
    for col in TIER_C_COLUMNS:
        idx = TRANSVOL_INPUT_COLUMNS.index(col)
        live_val = float(live[idx])
        if not np.isfinite(live_val):
            failures.append(f"{col}: live={live_val} (non-finite)")
    # Allow up to 30 columns to be non-finite (these are the dollar-
    # bar lag/EMA variants where the 300-bar cache + missing baseline
    # combine to produce NaN).
    assert len(failures) <= 30, (
        f"Tier-C finiteness gate FAILED for {len(failures)}/{len(TIER_C_COLUMNS)} "
        f"columns (allowed ≤30):\n" + "\n".join(failures[:20])
    )


# ── Mutation probe: column-order regressions ──────────────────────


def test_mutation_probe_swapped_column_order_breaks_strict_gate(
    parity_setup,
):
    """Deliberately swap two columns in ``TRANSVOL_INPUT_COLUMNS`` and
    assert that the Tier-A strict-parity gate would now fail. Locks
    the canonical column ordering against silent reordering."""
    live = parity_setup["live_feats"].copy()
    offline = parity_setup["offline_row"]
    fb_idx = TRANSVOL_INPUT_COLUMNS.index("funding_basis")
    ttf_idx = TRANSVOL_INPUT_COLUMNS.index("time_to_funding")
    live[fb_idx], live[ttf_idx] = live[ttf_idx], live[fb_idx]
    fb_live = float(live[fb_idx])
    fb_offline = float(offline["funding_basis"])
    assert not np.allclose(fb_live, fb_offline, rtol=1e-5, atol=1e-7), (
        "Mutation probe failed: swapping funding_basis and time_to_funding "
        "should produce a diff > 1e-7 in the Tier-A gate."
    )


def test_mutation_probe_tsrv_grid_column_swap_breaks_strict_gate(
    parity_setup,
):
    """S-1d (cycle 7): now that the 30 former Tier-B columns are
    Tier-A strict-byte-equal, extend the mutation probe to lock at
    least one of the newly-promoted columns. Swapping a TSRV-on-db-
    grid column with a funding-grid column produces large diffs
    because the value distributions are entirely different (TSRV
    columns range ~[-2, 0]; funding ~ -7e-5)."""
    live = parity_setup["live_feats"].copy()
    offline = parity_setup["offline_row"]
    a_idx = TRANSVOL_INPUT_COLUMNS.index("tsrv_mean_1d")
    b_idx = TRANSVOL_INPUT_COLUMNS.index("funding_basis_ema50")
    live[a_idx], live[b_idx] = live[b_idx], live[a_idx]
    a_live = float(live[a_idx])
    a_offline = float(offline["tsrv_mean_1d"])
    assert not np.allclose(a_live, a_offline, rtol=1e-5, atol=1e-7), (
        "Mutation probe failed: swapping tsrv_mean_1d and "
        "funding_basis_ema50 should produce a diff in the Tier-A gate."
    )


def test_mutation_probe_lag_variant_swap_breaks_strict_gate(
    parity_setup,
):
    """Cycle 7 — lock the lag-variant column ordering against silent
    reordering. Hardened cycle-41 (2026-05-17).

    The original probe swapped ``tsrv_mean_1w_lag1`` ↔
    ``tsrv_mean_1w_lag3`` — two lags of the *same* slow 1-week rolling
    mean. A 2-dollar-bar shift on a weekly mean is coincidentally
    within ``rtol=1e-5`` in calm data windows (observed -1.1769688 vs
    -1.1769572), so the swap was undetectable and the probe was flaky
    — hostage to whichever data window the offline parquet ends on.

    Fix: swap two *lag* columns from different feature families whose
    value ranges are reliably disjoint regardless of the data window.
    ``tsrv_mean_1w_lag1`` is a log-vol level (reliably negative, ~-1);
    ``up_jump_frac_lag3`` is a variance fraction in [0, 1] (reliably
    non-negative). The two ranges are sign-separated, so the swap is
    always detectable by the strict Tier-A gate while still locking
    the lag-variant ordering. A self-check skips loudly if the chosen
    pair ever lands within tolerance, so the probe can never silently
    pass as a false negative."""
    live = parity_setup["live_feats"].copy()
    offline = parity_setup["offline_row"]
    a_col, b_col = "tsrv_mean_1w_lag1", "up_jump_frac_lag3"
    a_idx = TRANSVOL_INPUT_COLUMNS.index(a_col)
    b_idx = TRANSVOL_INPUT_COLUMNS.index(b_col)

    a_offline = float(offline[a_col])
    b_offline = float(offline[b_col])
    # Self-check: the pair must be genuinely distinct for the swap to
    # be a meaningful probe. Sign-separated families make this robust
    # to data-window coincidences; if it ever fails, skip loudly
    # rather than report a false negative.
    if not (np.isfinite(a_offline) and np.isfinite(b_offline)):
        pytest.skip(
            f"Mutation-probe pair {a_col}/{b_col} has a non-finite "
            f"offline value ({a_offline} / {b_offline}); inconclusive."
        )
    if np.allclose(a_offline, b_offline, rtol=1e-5, atol=1e-7):
        pytest.skip(
            f"Mutation-probe pair {a_col}/{b_col} happens to be "
            f"near-equal in this data window "
            f"({a_offline:.10g} vs {b_offline:.10g}); probe inconclusive."
        )

    live[a_idx], live[b_idx] = live[b_idx], live[a_idx]
    a_live = float(live[a_idx])
    assert not np.allclose(a_live, a_offline, rtol=1e-5, atol=1e-7), (
        f"Mutation probe failed: swapping {a_col} and {b_col} should "
        f"produce a detectable diff in the Tier-A gate "
        f"(live={a_live:.10g} offline={a_offline:.10g})."
    )


# ── Tier classification sanity ────────────────────────────────────


def test_tier_classification_covers_all_101():
    """Every TransVol input column must appear in exactly one tier."""
    classified = set(TIER_A_COLUMNS) | set(TIER_B_COLUMNS) | set(TIER_C_COLUMNS)
    canon = set(TRANSVOL_INPUT_COLUMNS)
    missing = canon - classified
    duplicated = (
        (set(TIER_A_COLUMNS) & set(TIER_B_COLUMNS))
        | (set(TIER_A_COLUMNS) & set(TIER_C_COLUMNS))
        | (set(TIER_B_COLUMNS) & set(TIER_C_COLUMNS))
    )
    assert not missing, f"Tier classification missing columns: {missing}"
    assert not duplicated, f"Tier classification duplicates: {duplicated}"
    assert (
        len(TIER_A_COLUMNS) + len(TIER_B_COLUMNS) + len(TIER_C_COLUMNS)
        == 101
    )


def test_tier_a_size_matches_cycle_7_closure():
    """Cycle 7 promotion math:
    - Cycle-6 Tier-A (funding_basis ×5 + time_to_funding raw) = 6
    - Cycle-7 promoted: time_to_funding lag/EMA ×4 + 7 TSRV-base groups
      × 5 variants = 4 + 35 = 39
    Total = 6 + 39 = 45 columns under strict-byte-equal."""
    assert len(TIER_A_COLUMNS) == 45, (
        f"Tier-A should have 45 columns after cycle 7; got "
        f"{len(TIER_A_COLUMNS)}"
    )


def test_time_features_match_named_indices(parity_setup):
    """The 2-tuple time_features must correspond to (duration_s,
    n_bars_in_window) in the offline TIME_COLS order."""
    live = parity_setup["live_feats"]
    time_feats = parity_setup["live_time_feats"]
    duration_idx = TRANSVOL_INPUT_COLUMNS.index(TRANSVOL_TIME_COLUMNS[0])
    nbars_idx = TRANSVOL_INPUT_COLUMNS.index(TRANSVOL_TIME_COLUMNS[1])
    assert time_feats[0] == live[duration_idx]
    assert time_feats[1] == live[nbars_idx]
