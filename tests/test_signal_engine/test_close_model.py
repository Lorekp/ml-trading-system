"""Tests for signal_engine.close_model and precomputed exit simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import signal_engine.close_model as close_model_mod
from signal_engine.close_model import (
    _choose_threshold_from_validation,
    build_close_decision_examples,
    close_feature_columns,
    close_model_thread_budget,
    compute_afml_close_sample_weights,
    select_close_predictions,
    walk_forward_close_prediction_cache,
    walk_forward_close_predictions,
)
from signal_engine.config import CloseModelConfig, PortfolioConfig
from signal_engine.portfolio_path import simulate_portfolio_precomputed_exits


def test_build_close_decision_examples_labels_best_remaining() -> None:
    entry_index = pd.date_range("2024-01-01 00:00", periods=2, freq="15min", tz="UTC")
    entry_ohlcv = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1.0, 1.0],
        },
        index=entry_index,
    )
    decision_index = pd.date_range("2024-01-01 00:05", periods=4, freq="5min", tz="UTC")
    decision_ohlcv = pd.DataFrame(
        {
            "open": [101.0, 103.0, 102.0, 101.0],
            "high": [101.0, 103.0, 102.0, 101.0],
            "low": [101.0, 103.0, 102.0, 101.0],
            "close": [101.0, 103.0, 102.0, 101.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=decision_index,
    )
    decision_features = pd.DataFrame(
        {
            "ret_1": [0.1, 0.2, 0.3, 0.4],
            "ret_3": [0.0, 0.1, 0.2, 0.3],
        },
        index=decision_index,
    )
    signals = pd.DataFrame(
        {
            "timestamp": [entry_index[0]],
            "direction": [1.0],
            "probability": [0.7],
            "confidence": [0.2],
            "signal_strength": [0.8],
            "asset": ["BTC"],
        }
    )

    examples = build_close_decision_examples(
        signals,
        entry_ohlcv,
        decision_ohlcv,
        decision_features,
        ["ret_1", "ret_3"],
        decision_timeframe="5min",
        max_hold_minutes=15,
        slippage=0.0,
    )

    assert len(examples) == 3
    assert examples["close_label"].tolist() == [0, 1, 1]
    assert np.allclose(
        examples["current_unrealized_return"].to_numpy(),
        [0.01, 0.03, 0.02],
    )


def test_build_close_decision_examples_preserves_regime_exposure_mult() -> None:
    entry_index = pd.date_range("2024-01-01 00:00", periods=1, freq="15min", tz="UTC")
    entry_ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1.0],
        },
        index=entry_index,
    )
    decision_index = pd.date_range("2024-01-01 00:15", periods=4, freq="15min", tz="UTC")
    decision_ohlcv = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0, 101.5],
            "high": [100.0, 100.5, 101.0, 101.5],
            "low": [100.0, 100.5, 101.0, 101.5],
            "close": [100.0, 100.5, 101.0, 101.5],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=decision_index,
    )
    decision_features = pd.DataFrame({"ret_1": [0.0, 0.1, 0.1, 0.1]}, index=decision_index)
    signals = pd.DataFrame(
        {
            "timestamp": [entry_index[0]],
            "direction": [1.0],
            "probability": [0.7],
            "confidence": [0.2],
            "signal_strength": [0.8],
            "regime_exposure_mult": [0.65],
            "asset": ["BTC"],
        }
    )

    examples = build_close_decision_examples(
        signals,
        entry_ohlcv,
        decision_ohlcv,
        decision_features,
        ["ret_1"],
        decision_timeframe="15min",
        max_hold_minutes=60,
        slippage=0.0,
    )

    assert "regime_exposure_mult" in examples.columns
    assert np.allclose(examples["regime_exposure_mult"].to_numpy(), 0.65)


def test_simulate_portfolio_precomputed_exits_uses_exit_timestamps() -> None:
    timestamps = pd.date_range("2024-01-01 00:00", periods=20, freq="10min", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "exit_timestamp": timestamps + pd.Timedelta(minutes=5),
            "direction": np.ones(len(timestamps)),
            "fwd_return": np.full(len(timestamps), 0.001),
            "probability": np.full(len(timestamps), 0.6),
            "confidence": np.full(len(timestamps), 0.1),
            "signal_strength": np.full(len(timestamps), 1.0),
            "asset": ["BTC"] * len(timestamps),
            "bars_held": np.full(len(timestamps), 2),
        }
    )
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_exposure=1.0,
        max_per_asset=1.0,
        base_sizing_pct=0.5,
        max_signal_mult=2.0,
    )

    sim = simulate_portfolio_precomputed_exits(signals, cfg)

    assert sim is not None
    assert sim["n_trades"] == len(signals)


def test_simulate_portfolio_precomputed_exits_supports_signed_returns_and_partial_derisk() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:10",
            "2024-01-01 00:20",
            "2024-01-01 00:30",
            "2024-01-01 00:40",
            "2024-01-01 00:50",
            "2024-01-01 01:00",
            "2024-01-01 01:10",
            "2024-01-01 01:20",
            "2024-01-01 01:30",
        ],
        utc=True,
    )
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "exit_timestamp": timestamps + pd.Timedelta(minutes=30),
            "de_risk_timestamp": [
                timestamps[1],
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
            "direction": np.ones(len(timestamps)),
            "fwd_return": np.full(len(timestamps), 0.002),
            "return_already_signed": np.ones(len(timestamps), dtype=bool),
            "remaining_fraction": [0.25] + [0.0] * (len(timestamps) - 1),
            "signal_strength": np.ones(len(timestamps)),
            "asset": ["BTC"] * len(timestamps),
            "bars_held": np.ones(len(timestamps), dtype=int),
        }
    )
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_exposure=0.25,
        max_per_asset=0.25,
        base_sizing_pct=0.25,
        max_signal_mult=1.0,
        slippage=0.0,
    )

    sim = simulate_portfolio_precomputed_exits(signals, cfg)

    assert sim is not None
    assert sim["n_trades"] >= 2


def test_compute_afml_close_sample_weights_positive_and_time_decayed() -> None:
    idx = pd.date_range("2024-01-01 00:00", periods=4, freq="5min", tz="UTC")
    examples = pd.DataFrame(
        {
            "decision_timestamp": idx,
            "label_end_timestamp": idx + pd.Timedelta(minutes=10),
            "label_return_abs": [0.01, 0.02, 0.03, 0.04],
        }
    )

    weights = compute_afml_close_sample_weights(examples, clf_last_w=0.5)

    assert len(weights) == len(examples)
    assert np.all(weights.to_numpy() > 0.0)
    assert weights.iloc[-1] > weights.iloc[0]


def test_select_close_predictions_can_share_probabilities_across_policies() -> None:
    prediction_rows = pd.DataFrame(
        {
            "target_month": ["2025-01", "2025-01", "2025-01", "2025-01"],
            "trade_id": ["t1", "t1", "t2", "t2"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:00",
                    "2025-01-01 01:00",
                    "2025-01-01 01:00",
                ],
                utc=True,
            ),
            "decision_timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:15",
                    "2025-01-01 00:30",
                    "2025-01-01 01:15",
                    "2025-01-01 01:30",
                ],
                utc=True,
            ),
            "expiry_timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:30",
                    "2025-01-01 00:30",
                    "2025-01-01 01:30",
                    "2025-01-01 01:30",
                ],
                utc=True,
            ),
            "direction": [1.0, 1.0, 1.0, 1.0],
            "entry_probability": [0.7, 0.7, 0.7, 0.7],
            "entry_confidence": [0.2, 0.2, 0.2, 0.2],
            "entry_signal_strength": [0.8, 0.8, 0.8, 0.8],
            "regime_exposure_mult": [1.0, 1.0, 1.0, 1.0],
            "asset": ["BTC", "BTC", "ETH", "ETH"],
            "exit_return": [0.02, 0.01, -0.01, -0.02],
            "gross_exit_return": [0.02, 0.01, -0.01, -0.02],
            "expiry_return": [0.01, 0.01, -0.02, -0.02],
            "minutes_since_entry": [15, 30, 15, 30],
            "time_to_expiry_minutes": [15, 0, 15, 0],
            "current_unrealized_return": [0.02, 0.01, -0.01, -0.02],
            "realized_vol_since_entry": [0.1, 0.1, 0.1, 0.1],
            "mfe_minus_mae": [0.01, 0.0, -0.01, -0.02],
            "distance_from_entry_mfe": [0.0, 0.01, 0.0, 0.01],
            "close_label": [1, 0, 1, 1],
            "close_probability": [0.8, 0.6, 0.95, 0.4],
        }
    )
    validation_rows = prediction_rows.copy()
    validation_rows["target_month"] = "2025-01"
    prediction_diag = {"months": [{"month": "2025-01", "n_train": 200, "n_validation": 4}], "summary": {}}

    binary_cfg = CloseModelConfig(de_risk_mode="binary_exit", threshold=0.5)
    partial_cfg = CloseModelConfig(de_risk_mode="partial_ladder", threshold=0.5)

    selected_binary, _ = select_close_predictions(
        prediction_rows,
        validation_rows,
        binary_cfg,
        prediction_diagnostics=prediction_diag,
    )
    selected_partial, _ = select_close_predictions(
        prediction_rows,
        validation_rows,
        partial_cfg,
        prediction_diagnostics=prediction_diag,
    )

    assert len(selected_binary) == len(selected_partial) == 2
    assert not np.allclose(
        selected_binary["remaining_fraction"].to_numpy(dtype=float),
        selected_partial["remaining_fraction"].to_numpy(dtype=float),
    )


def test_close_model_thread_budget_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ML_CLOSE_MODEL_THREADS", "3")
    assert close_model_thread_budget() == 3


def test_close_model_thread_budget_clamps_multicore_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ML_CLOSE_MODEL_THREADS", raising=False)
    monkeypatch.setenv("ML_CLOSE_EVAL_WORKERS", "8")
    monkeypatch.setattr(close_model_mod.os, "cpu_count", lambda: 16)
    assert close_model_thread_budget() == 2


def test_build_close_decision_examples_residual_triple_barrier() -> None:
    entry_index = pd.date_range("2024-01-01 00:00", periods=1, freq="15min", tz="UTC")
    entry_ohlcv = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1.0],
        },
        index=entry_index,
    )
    decision_index = pd.date_range("2024-01-01 00:05", periods=4, freq="5min", tz="UTC")
    decision_ohlcv = pd.DataFrame(
        {
            "open": [100.0, 99.5, 99.0, 98.0],
            "high": [100.0, 99.5, 99.0, 98.0],
            "low": [100.0, 99.5, 99.0, 98.0],
            "close": [100.0, 99.5, 99.0, 98.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=decision_index,
    )
    decision_features = pd.DataFrame(
        {
            "ret_1": [0.0, -0.005, -0.005, -0.01],
            "ret_3": [0.0, 0.0, -0.01, -0.02],
        },
        index=decision_index,
    )
    regime_features = pd.DataFrame(
        {
            "ret_1": [0.0],
            "ret_3": [0.0],
        },
        index=pd.date_range("2024-01-01 00:00", periods=1, freq="1h", tz="UTC"),
    )
    signals = pd.DataFrame(
        {
            "timestamp": [entry_index[0]],
            "direction": [1.0],
            "probability": [0.7],
            "confidence": [0.2],
            "signal_strength": [0.8],
            "asset": ["BTC"],
        }
    )

    examples = build_close_decision_examples(
        signals,
        entry_ohlcv,
        decision_ohlcv,
        decision_features,
        ["ret_1", "ret_3"],
        decision_timeframe="5min",
        max_hold_minutes=20,
        slippage=0.0,
        label_mode="residual_triple_barrier",
        min_hold_minutes=5,
        pt_mult=1.0,
        sl_mult=1.0,
        min_improvement_bps=2.0,
        regime_features=regime_features,
    )

    assert len(examples) == 4
    assert examples["close_label"].iloc[0] == 1
    assert "time_to_expiry_minutes" in examples.columns
    assert "regime_1h_ret_1" in examples.columns


def test_close_feature_columns_excludes_placeholder_signal_change_features() -> None:
    cols = close_feature_columns(["ret_1", "ret_3"])

    assert "signal_prob_change_since_entry" not in cols
    assert "signal_strength_change_since_entry" not in cols
    assert "time_to_expiry_minutes" in cols


def test_choose_threshold_from_validation_can_differ_for_portfolio_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    base_ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    for trade_id in range(12):
        ts = base_ts + pd.Timedelta(minutes=30 * trade_id)
        rows.append(
            {
                "trade_id": trade_id,
                "decision_timestamp": ts + pd.Timedelta(minutes=15),
                "close_probability": 0.55,
                "close_label": 1,
                "exit_return": -0.02,
                "entry_probability": 0.6,
                "entry_confidence": 0.2,
                "entry_signal_strength": 0.7,
                "direction": 1.0,
                "asset": "BTC",
                "minutes_since_entry": 15,
            }
        )
        rows.append(
            {
                "trade_id": trade_id,
                "decision_timestamp": ts + pd.Timedelta(minutes=60),
                "close_probability": 0.70,
                "close_label": 0,
                "exit_return": 0.01,
                "entry_probability": 0.6,
                "entry_confidence": 0.2,
                "entry_signal_strength": 0.7,
                "direction": 1.0,
                "asset": "BTC",
                "minutes_since_entry": 60,
            }
        )
    validation_rows = pd.DataFrame(rows)
    close_cfg = CloseModelConfig(calibration="none", threshold=0.5)

    def fake_score(selected: pd.DataFrame, portfolio_cfg=None) -> tuple[float, float]:
        early_frac = float((selected["minutes_since_entry"] <= 15).mean())
        if portfolio_cfg is None:
            return early_frac, 50.0
        return 1.0 - early_frac, 1.0

    monkeypatch.setattr("signal_engine.close_model._score_selected_rows", fake_score)

    proxy = _choose_threshold_from_validation(
        validation_rows,
        close_cfg,
        signal_bar_minutes=15,
        portfolio_cfg=None,
    )
    portfolio = _choose_threshold_from_validation(
        validation_rows,
        close_cfg,
        signal_bar_minutes=15,
        portfolio_cfg=PortfolioConfig(),
    )

    assert proxy["threshold"] != portfolio["threshold"]


def test_walk_forward_close_predictions_passes_portfolio_cfg_to_threshold_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_cols = ["ret_1", "ret_3"]
    all_rows = []
    months = ["2021-10", "2021-11", "2021-12", "2022-01"]
    for month_idx, month in enumerate(months):
        base = pd.Timestamp(f"{month}-01 00:00", tz="UTC")
        for trade_id in range(30):
            entry_ts = base + pd.Timedelta(days=trade_id)
            for step, prob in enumerate([0.35, 0.45, 0.55, 0.75], start=1):
                all_rows.append(
                    {
                        "trade_id": month_idx * 100 + trade_id,
                        "timestamp": entry_ts,
                        "entry_month": month,
                        "decision_timestamp": entry_ts + pd.Timedelta(minutes=15 * step),
                        "decision_month": month,
                        "expiry_timestamp": entry_ts + pd.Timedelta(minutes=60),
                        "label_end_timestamp": entry_ts + pd.Timedelta(minutes=60),
                        "direction": 1.0,
                        "entry_probability": 0.6,
                        "entry_confidence": 0.2,
                        "entry_signal_strength": 0.8,
                        "asset": "BTC",
                        "exit_return": 0.01 * step,
                        "gross_exit_return": 0.01 * step,
                        "expiry_return": 0.04,
                        "minutes_since_entry": 15 * step,
                        "time_to_expiry_minutes": 60 - 15 * step,
                        "current_unrealized_return": 0.01 * step,
                        "realized_vol_since_entry": 0.001,
                        "mfe_minus_mae": 0.002,
                        "distance_from_entry_mfe": -0.001,
                        "close_label": int(
                            (step == 1 and trade_id % 2 == 0)
                            or (step == 2 and trade_id % 2 == 1)
                        ),
                        "label_return_abs": 0.01,
                        "ret_1": 0.1 * step,
                        "ret_3": 0.05 * step,
                        "regime_1h_ret_1": 0.01,
                        "regime_1h_ret_3": 0.02,
                        "entry_direction": 1.0,
                        "bars_since_entry": step,
                        "remaining_decision_steps": 4 - step,
                        "max_favorable_return_since_entry": 0.01 * step,
                        "max_adverse_return_since_entry": -0.005,
                        "return_since_last_decision": 0.002,
                        "rolling_2bar": 0.003,
                        "rolling_3bar": 0.004,
                        "distance_from_entry_mae": 0.001,
                        "entry_to_now_trend_slope": 0.002,
                    }
                )
    examples = pd.DataFrame(all_rows)

    captured = {"portfolio_cfg": None}

    class DummyBooster:
        def get_score(self, importance_type: str = "gain") -> dict[str, float]:
            return {"f0": 1.0}

    class DummyModel:
        def fit(self, X, y, sample_weight=None, **kwargs):
            return self

        def predict_proba(self, X):
            out = np.zeros((len(X), 2), dtype=float)
            out[:, 1] = np.clip(X[:, 0], 0.0, 1.0)
            out[:, 0] = 1.0 - out[:, 1]
            return out

        def get_booster(self):
            return DummyBooster()

    def fake_make_xgb(scale_pos_weight: float = 1.0):
        return DummyModel()

    def fake_choose_threshold(validation_rows, close_cfg, signal_bar_minutes, portfolio_cfg=None):
        captured["portfolio_cfg"] = portfolio_cfg
        return {
            "threshold": 0.5,
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
            "positive_rate": 0.3,
            "validation_sharpe": 1.0,
            "validation_max_dd_pct": 1.0,
        }

    monkeypatch.setattr(close_model_mod, "make_xgb_close_model", fake_make_xgb)
    monkeypatch.setattr(close_model_mod, "_choose_threshold_from_validation", fake_choose_threshold)

    close_cfg = CloseModelConfig(
        model_type="xgboost",
        calibration="none",
        threshold=0.5,
        threshold_min_precision=0.0,
    )
    selected, diagnostics = walk_forward_close_predictions(
        examples,
        feature_cols,
        lookback_months=3,
        close_cfg=close_cfg,
        test_start="2022-01",
        signal_bar_minutes=15,
        min_train_rows=2,
        portfolio_cfg=PortfolioConfig(),
    )

    assert len(selected) > 0
    assert diagnostics["summary"]["calibration"] == "none"
    assert captured["portfolio_cfg"] is not None
