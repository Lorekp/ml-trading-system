"""
Integration test — runs the full AFML pipeline end-to-end.

This is the most important test: it verifies that all modules
(loader → features → events → labels → weights) work together
correctly on realistic data.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from data.pipeline import generate_features, generate_labels, generate_sample_weights, run_pipeline
from data.loader import BINANCE_COLUMNS


def make_realistic_ohlcv(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic crypto-like OHLCV data."""
    np.random.seed(seed)

    # Regime-switching random walk (bull/bear/volatile)
    returns = np.zeros(n)
    regime = 0  # 0=normal, 1=bull, 2=bear, 3=volatile
    for i in range(n):
        if np.random.random() < 0.01:
            regime = np.random.choice([0, 1, 2, 3])
        if regime == 0:
            returns[i] = np.random.normal(0, 0.008)
        elif regime == 1:
            returns[i] = np.random.normal(0.002, 0.01)
        elif regime == 2:
            returns[i] = np.random.normal(-0.002, 0.01)
        else:
            returns[i] = np.random.normal(0, 0.025)

    close = 70000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    opn = np.roll(close, 1) * (1 + np.random.normal(0, 0.001, n))
    opn[0] = close[0]
    volume = np.random.exponential(5000, n) * (1 + np.abs(returns) * 50)

    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


class TestGenerateFeatures:
    def test_produces_expected_columns(self):
        ohlcv = make_realistic_ohlcv(1000)
        features = generate_features(ohlcv, ffd_threshold=1e-2)

        expected = ["ret_1", "ret_3", "ret_7", "ret_14", "ret_42",
                    "ffd", "parkinson_vol", "cs_spread", "daily_vol",
                    "vol_zscore", "entropy", "hour_sin", "dow_sin"]
        for col in expected:
            assert col in features.columns, f"Missing feature: {col}"

    def test_no_all_nan_columns(self):
        ohlcv = make_realistic_ohlcv(1000)
        features = generate_features(ohlcv, ffd_threshold=1e-2)
        for col in features.columns:
            non_nan = features[col].dropna()
            assert len(non_nan) > 100, f"Feature {col} has too few non-NaN values: {len(non_nan)}"

    def test_ffd_d_stored_in_attrs(self):
        ohlcv = make_realistic_ohlcv(1000)
        features = generate_features(ohlcv, ffd_threshold=1e-2)
        assert "ffd_d" in features.attrs
        assert 0 < features.attrs["ffd_d"] <= 1


class TestGenerateLabels:
    def test_produces_events_and_bins(self):
        ohlcv = make_realistic_ohlcv(2000)
        from afml.filters import get_daily_vol
        vol = get_daily_vol(ohlcv["close"], span=50)
        events, bins = generate_labels(ohlcv["close"], vol, num_bars_vertical=20)

        assert len(events) > 0, "Should generate some events"
        assert len(bins) > 0, "Should generate some labels"
        assert "t1" in events.columns
        assert "bin" in bins.columns

    def test_label_distribution_reasonable(self):
        ohlcv = make_realistic_ohlcv(2000)
        from afml.filters import get_daily_vol
        vol = get_daily_vol(ohlcv["close"], span=50)
        events, bins = generate_labels(ohlcv["close"], vol, num_bars_vertical=20)

        if len(bins) > 50:
            dist = bins["bin"].value_counts(normalize=True)
            # No single class should dominate completely
            assert dist.max() < 0.9, f"Label distribution too skewed: {dist.to_dict()}"

    def test_event_rate_reasonable(self):
        ohlcv = make_realistic_ohlcv(2000)
        from afml.filters import get_daily_vol
        vol = get_daily_vol(ohlcv["close"], span=50)
        events, _ = generate_labels(ohlcv["close"], vol, num_bars_vertical=20)

        rate = len(events) / len(ohlcv)
        # Expect 5-50% event rate with CUSUM
        assert 0.01 < rate < 0.8, f"Event rate {rate:.2%} is extreme"


class TestGenerateSampleWeights:
    def test_weights_positive(self):
        ohlcv = make_realistic_ohlcv(2000)
        from afml.filters import get_daily_vol
        vol = get_daily_vol(ohlcv["close"], span=50)
        events, _ = generate_labels(ohlcv["close"], vol, num_bars_vertical=20)

        if len(events) < 10:
            pytest.skip("Too few events")

        weights = generate_sample_weights(events, ohlcv["close"])
        assert (weights.dropna() >= 0).all(), "Weights should be non-negative"

    def test_weights_normalized(self):
        ohlcv = make_realistic_ohlcv(2000)
        from afml.filters import get_daily_vol
        vol = get_daily_vol(ohlcv["close"], span=50)
        events, _ = generate_labels(ohlcv["close"], vol, num_bars_vertical=20)

        if len(events) < 10:
            pytest.skip("Too few events")

        weights = generate_sample_weights(events, ohlcv["close"])
        # Mean should be ~1.0 after normalization
        assert weights.dropna().mean() == pytest.approx(1.0, abs=0.5)


class TestRunPipeline:
    def test_full_pipeline_on_synthetic(self, tmp_path):
        """Full end-to-end pipeline test with synthetic Binance-format data."""
        # Create synthetic Binance CSVs
        np.random.seed(42)
        n = 2000  # ~1.4 days of 1m data → gives enough for 4H resampling
        ohlcv = make_realistic_ohlcv(n)

        # Write as Binance CSV format
        sym_dir = tmp_path / "data" / "TESTUSDT"
        sym_dir.mkdir(parents=True)

        timestamps_ms = (ohlcv.index.astype(np.int64) // 10**6).values
        rows = []
        for i in range(n):
            rows.append([
                timestamps_ms[i],
                ohlcv.iloc[i]["open"], ohlcv.iloc[i]["high"],
                ohlcv.iloc[i]["low"], ohlcv.iloc[i]["close"],
                ohlcv.iloc[i]["volume"],
                timestamps_ms[i] + 14399999,
                ohlcv.iloc[i]["volume"] * ohlcv.iloc[i]["close"],
                100, ohlcv.iloc[i]["volume"] * 0.6,
                ohlcv.iloc[i]["volume"] * ohlcv.iloc[i]["close"] * 0.6, 0,
            ])

        csv_df = pd.DataFrame(rows)
        csv_df.to_csv(sym_dir / "TESTUSDT-1m-2024-01.csv", header=False, index=False)

        output_dir = tmp_path / "processed"
        cache_dir = tmp_path / "cache"

        # Since data is already at 4H freq but saved as "1m" CSV,
        # we'll load at native freq (resample will produce 1 bar per input)
        summary = run_pipeline(
            symbol="TESTUSDT",
            data_dir=tmp_path / "data",
            timeframe="1m",  # Load as-is
            output_dir=output_dir,
            cache_dir=cache_dir,
            num_bars_vertical=20,
        )

        assert summary["status"] == "ok"
        assert summary["events"] > 0
        assert (output_dir / "TESTUSDT_1m_features.parquet").exists()
        assert (output_dir / "TESTUSDT_1m_events.parquet").exists()
        assert (output_dir / "TESTUSDT_1m_bins.parquet").exists()
        assert (output_dir / "TESTUSDT_1m_weights.parquet").exists()
        assert (output_dir / "TESTUSDT_1m_meta.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
