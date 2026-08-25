from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.research.close_walkforward import (
    apply_train_window_winsorization,
    canonical_prediction_de_risk_mode,
    close_strategy_overrides,
    combined_close_example_cache_path,
    compute_cscv_pbo,
    filter_rows_by_period,
    generate_asset_signals,
    get_pcfg,
    prediction_cache_path,
    historical_holdout_gate,
    selected_trade_cache_path,
)
from signal_engine.config import CloseModelConfig, ConfidenceFilterConfig, PortfolioConfig


def test_apply_train_window_winsorization_uses_train_window_only() -> None:
    train = pd.DataFrame(
        {
            "ret_1": [0.0, 1.0, 2.0, 3.0],
            "ret_3": [10.0, 11.0, 12.0, 13.0],
        }
    )
    test = pd.DataFrame(
        {
            "ret_1": [100.0, -100.0],
            "ret_3": [1000.0, -1000.0],
        }
    )

    win_train, win_test = apply_train_window_winsorization(
        train,
        test,
        ["ret_1", "ret_3"],
        winsor_lo=0.25,
        winsor_hi=0.75,
    )

    assert win_train["ret_1"].min() == train["ret_1"].quantile(0.25)
    assert win_train["ret_1"].max() == train["ret_1"].quantile(0.75)
    assert win_test["ret_1"].max() == train["ret_1"].quantile(0.75)
    assert win_test["ret_1"].min() == train["ret_1"].quantile(0.25)


def test_close_strategy_overrides_disable_calibration_for_report() -> None:
    cfg = CloseModelConfig(calibration="isotonic")

    xgb = close_strategy_overrides("xgboost", "residual_triple_barrier", "15min", cfg)
    logreg = close_strategy_overrides("logistic", "best_remaining", "15min", cfg)

    assert xgb.calibration == "none"
    assert logreg.calibration == "none"
    assert logreg.threshold == 0.5


def test_prediction_cache_is_shared_for_xgb_binary_and_partial() -> None:
    pred_binary = prediction_cache_path(
        10,
        "15min",
        "2026-02",
        "feat",
        "scope",
        "xgboost",
        "residual_triple_barrier",
    )
    pred_partial = prediction_cache_path(
        10,
        "15min",
        "2026-02",
        "feat",
        "scope",
        "xgboost",
        "residual_triple_barrier",
    )
    selected_binary = selected_trade_cache_path(
        10,
        "15min",
        "2026-02",
        "feat",
        "scope",
        "xgb_residtb_binary",
    )
    selected_partial = selected_trade_cache_path(
        10,
        "15min",
        "2026-02",
        "feat",
        "scope",
        "xgb_residtb_partial",
    )

    assert pred_binary == pred_partial
    assert selected_binary != selected_partial


def test_combined_example_cache_is_label_mode_scoped() -> None:
    path = combined_close_example_cache_path(
        12,
        "10min",
        "2026-02",
        "feat",
        "scope",
        "residual_triple_barrier",
    )

    assert "residtb" in str(path)
    assert "xgb" not in str(path)
    assert "partial" not in str(path)


def test_canonical_prediction_mode_shares_xgb_residual_training() -> None:
    assert (
        canonical_prediction_de_risk_mode(
            "xgboost",
            "residual_triple_barrier",
            "binary_exit",
        )
        == "partial_ladder"
    )
    assert (
        canonical_prediction_de_risk_mode(
            "logistic",
            "best_remaining",
            "binary_exit",
        )
        == "binary_exit"
    )


def test_get_pcfg_uses_loaded_portfolio_config() -> None:
    class DummyCfg:
        portfolio = PortfolioConfig(
            starting_capital=12345.0,
            max_exposure=0.33,
            max_per_asset=0.07,
        )

    pcfg = get_pcfg(DummyCfg())

    assert pcfg.starting_capital == 12345.0
    assert pcfg.max_exposure == 0.33
    assert pcfg.max_per_asset == 0.07


def test_filter_rows_by_period_splits_search_and_holdout() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-12-15", "2025-01-02", "2025-02-03"],
                utc=True,
            ),
            "value": [1, 2, 3],
        }
    )

    search = filter_rows_by_period(df, "2022-01", "2024-12")
    holdout = filter_rows_by_period(df, "2025-01", None)

    assert search["value"].tolist() == [1]
    assert holdout["value"].tolist() == [2, 3]


def test_historical_holdout_gate_requires_holdout_and_dsr() -> None:
    baseline = {"status": "ok", "sharpe": 2.0, "max_dd_pct": 5.0}
    candidate = {"status": "ok", "sharpe": 2.1, "max_dd_pct": 6.0}

    enabled, holdout_pass, dsr_pass = historical_holdout_gate(candidate, baseline, 0.97)
    assert enabled is True
    assert holdout_pass is True
    assert dsr_pass is True

    enabled, holdout_pass, dsr_pass = historical_holdout_gate(candidate, baseline, 0.90)
    assert enabled is False
    assert holdout_pass is True
    assert dsr_pass is False


def test_generate_asset_signals_uses_configured_confidence_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cover January through mid-March so the configured two-month lookback can
    # train on January-February and exercise the March confidence filter.
    idx = pd.date_range("2023-01-01", periods=7000, freq="15min", tz="UTC")
    close = 100 + np.sin(np.linspace(0, 50, len(idx))).cumsum()
    ohlcv = pd.DataFrame(
        {
            "close": close,
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "ret_1": 0.001,
            "ret_3": 0.002,
            "ret_7": 0.003,
            "ret_14": 0.004,
            "bar_range": 0.01,
            "vol_zscore": 0.5,
            "rsi_14": 55.0,
            "stoch_rsi": 0.6,
            "macd_hist": 0.1,
            "bb_pct_b": 0.4,
            "atr_14": 0.02,
            "dist_swing_high": 0.03,
            "dist_swing_low": 0.02,
        },
        index=idx,
    )

    captured = {"cfg": None}

    def fake_filter(signals, cfg):
        captured["cfg"] = cfg
        return signals

    monkeypatch.setattr(
        "signal_engine.research.close_walkforward.apply_confidence_filter",
        fake_filter,
    )

    class DummyCfg:
        class Features:
            feature_list = list(features.columns)
            vol_filter_quantile = 0.5
            winsor_lo = 0.01
            winsor_hi = 0.99

        class Model:
            fwd_bars = 4
            signal_model_type = "asset_logistic"
            seed = 42

        features = Features()
        model = Model()
        confidence_filter = ConfidenceFilterConfig(mode="percentile", threshold=70.0)

        class Signal:
            step_size = 0.1

        signal = Signal()

    generate_asset_signals("BTC", ohlcv, features, None, DummyCfg())

    assert captured["cfg"] is not None
    assert captured["cfg"].mode == "percentile"
    assert captured["cfg"].threshold == 70.0


def test_compute_cscv_pbo_returns_summary_for_valid_rows() -> None:
    rows = []
    for idx, sharpe in enumerate([1.0, 1.5, 2.0]):
        monthly = {
            f"2022-{month:02d}": float((idx + 1) * month)
            for month in range(1, 13)
        }
        rows.append(
            {
                "lookback": idx + 1,
                "timeframe": "15min",
                "status": "ok",
                "sharpe": sharpe,
                "search_full_stats": {"monthly_pnl": monthly},
            }
        )

    summary = compute_cscv_pbo(rows, n_partitions=4)

    assert summary
    assert 0.0 <= summary["pbo"] <= 1.0
    assert summary["num_strategies"] == 3
