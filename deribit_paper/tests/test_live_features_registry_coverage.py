"""S-1c (parity-r3 cycle 6 — 2026-05-08) — registry-coverage gate.

This test fixes the "what fraction of the 101 TransVol input columns
have a live handler" tally as a structural assertion. Pre-cycle-6 the
ratio was 45/101 (cycle-5 partials + 6 fully-handled bases). After
cycle 6 Phase 1d the ratio is 96/101 (only the 5 ``funding_basis_*``
columns remain — they ship in Phase 2 with the live REST/parquet
provider).

If a future cycle drops a handler or breaks an alias, this test
fails immediately rather than silently regressing live to baseline-
fallback for that column.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.signal.feature_assembler import _LIVE_HANDLERS


# Canonical 101-feature TransVol input column list, in the order the
# encoder was trained on. Matches `pd.read_parquet('vol_model/output/
# BTCUSDT_features_15m.parquet').columns` minus the 25 dropped
# OFI/depth/OI columns (see vol_model/transvol/dataset.py:142-144).
TRANSVOL_INPUT_COLUMNS_FOR_GATE: tuple[str, ...] = (
    "n_bars_in_window",
    # hurst_100 ×5
    "hurst_100", "hurst_100_lag1", "hurst_100_lag3",
    "hurst_100_ema10", "hurst_100_ema50",
    # ffd_rv ×5
    "ffd_rv", "ffd_rv_lag1", "ffd_rv_lag3",
    "ffd_rv_ema10", "ffd_rv_ema50",
    # rv_zscore_1d ×5
    "rv_zscore_1d", "rv_zscore_1d_lag1", "rv_zscore_1d_lag3",
    "rv_zscore_1d_ema10", "rv_zscore_1d_ema50",
    # rv_zscore_1w ×5
    "rv_zscore_1w", "rv_zscore_1w_lag1", "rv_zscore_1w_lag3",
    "rv_zscore_1w_ema10", "rv_zscore_1w_ema50",
    # rv_zscore_1m ×5
    "rv_zscore_1m", "rv_zscore_1m_lag1", "rv_zscore_1m_lag3",
    "rv_zscore_1m_ema10", "rv_zscore_1m_ema50",
    # tsrv_ffd ×5
    "tsrv_ffd", "tsrv_ffd_lag1", "tsrv_ffd_lag3",
    "tsrv_ffd_ema10", "tsrv_ffd_ema50",
    # tsrv_mean_1d ×5
    "tsrv_mean_1d", "tsrv_mean_1d_lag1", "tsrv_mean_1d_lag3",
    "tsrv_mean_1d_ema10", "tsrv_mean_1d_ema50",
    # tsrv_mean_1w ×5
    "tsrv_mean_1w", "tsrv_mean_1w_lag1", "tsrv_mean_1w_lag3",
    "tsrv_mean_1w_ema10", "tsrv_mean_1w_ema50",
    # down_jump_frac ×5
    "down_jump_frac", "down_jump_frac_lag1", "down_jump_frac_lag3",
    "down_jump_frac_ema10", "down_jump_frac_ema50",
    # up_jump_frac ×5
    "up_jump_frac", "up_jump_frac_lag1", "up_jump_frac_lag3",
    "up_jump_frac_ema10", "up_jump_frac_ema50",
    # noise_level ×5
    "noise_level", "noise_level_lag1", "noise_level_lag3",
    "noise_level_ema10", "noise_level_ema50",
    # bpv_tsrv_ratio ×5
    "bpv_tsrv_ratio", "bpv_tsrv_ratio_lag1", "bpv_tsrv_ratio_lag3",
    "bpv_tsrv_ratio_ema10", "bpv_tsrv_ratio_ema50",
    # parkinson_1d ×5
    "parkinson_1d", "parkinson_1d_lag1", "parkinson_1d_lag3",
    "parkinson_1d_ema10", "parkinson_1d_ema50",
    # duration_s ×5
    "duration_s", "duration_s_lag1", "duration_s_lag3",
    "duration_s_ema10", "duration_s_ema50",
    # n_trades ×5
    "n_trades", "n_trades_lag1", "n_trades_lag3",
    "n_trades_ema10", "n_trades_ema50",
    # imbalance ×5
    "imbalance", "imbalance_lag1", "imbalance_lag3",
    "imbalance_ema10", "imbalance_ema50",
    # vov ×5
    "vov", "vov_lag1", "vov_lag3", "vov_ema10", "vov_ema50",
    # vpin ×5
    "vpin", "vpin_lag1", "vpin_lag3", "vpin_ema10", "vpin_ema50",
    # funding_basis ×5 (Phase 2 — handlers wired with the live provider)
    "funding_basis", "funding_basis_lag1", "funding_basis_lag3",
    "funding_basis_ema10", "funding_basis_ema50",
    # time_to_funding ×5
    "time_to_funding", "time_to_funding_lag1", "time_to_funding_lag3",
    "time_to_funding_ema10", "time_to_funding_ema50",
)


def test_total_transvol_input_columns_is_101():
    """Locks the count: 21 bases × 5 variants − 4 absent for n_bars_in_window."""
    assert len(TRANSVOL_INPUT_COLUMNS_FOR_GATE) == 101


def test_post_cycle_6_phase_2_coverage_is_101_of_101():
    """All 101 TransVol input columns must have a live handler after
    Phase 2 (funding_basis live provider) ships."""
    missing = [
        c for c in TRANSVOL_INPUT_COLUMNS_FOR_GATE
        if c not in _LIVE_HANDLERS
    ]
    assert missing == [], (
        f"{len(missing)} TransVol input columns still resolve to "
        f"baseline-fallback: {missing}"
    )


def test_no_duplicate_column_names_in_canonical_list():
    """Sanity: the canonical list must be a set (unique column names)."""
    assert len(TRANSVOL_INPUT_COLUMNS_FOR_GATE) == len(set(TRANSVOL_INPUT_COLUMNS_FOR_GATE))
