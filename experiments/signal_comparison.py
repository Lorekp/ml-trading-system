"""Compare live paper trader signals against backtest reproduction."""
import sys
sys.path.insert(0, "src")

import asyncio, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

from pathlib import Path
from signal_engine.config import load_config
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.data import load_binance_1m, resample_ohlcv
from afml.bet_sizing import get_signal, discretize_signal
from live.exchange import MexcAdapter
from live.config import ExchangeConfig

cfg = load_config(Path("configs/default.yaml"))
cfg.regime_filter.enabled = False
cfg.confidence_filter.mode = "absolute"
cfg.confidence_filter.threshold = 0.08

# Train model (same as live bot startup)
print("Training model on BTC walk-forward...", flush=True)
wf = WalkForwardModel(cfg)
ohlcv_1m = load_binance_1m("BTCUSDT", [Path(d) for d in cfg.data.data_dirs])
ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
features_hist = compute_features(ohlcv_15m, cfg.features)
wf.generate_signals_batch(ohlcv_15m, features_hist, "BTC")
model = wf.current_model
scaler = wf.current_scaler

# Fetch current bars from MEXC
async def fetch():
    exc = MexcAdapter(ExchangeConfig(name="mexc", base_url="https://futures.mexc.com",
                                      api_key="x", api_secret="x"))
    bars = {}
    for sym in ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SUIUSDT", "LINKUSDT"]:
        bars[sym] = await exc.fetch_klines(sym, "15m", 200)
    await exc.close()
    return bars

print("Fetching MEXC bars...", flush=True)
all_bars = asyncio.run(fetch())

# Live signals from the paper trader DB
live_signals = [
    ("2026-03-18T19:00", "BTC", "BTCUSDT", 0.583705),
    ("2026-03-18T19:00", "ETH", "ETHUSDT", 0.593972),
    ("2026-03-18T19:00", "SUI", "SUIUSDT", 0.582376),
    ("2026-03-18T19:00", "LINK", "LINKUSDT", 0.590157),
    ("2026-03-18T20:15", "BTC", "BTCUSDT", 0.583071),
    ("2026-03-18T20:15", "ETH", "ETHUSDT", 0.596589),
    ("2026-03-18T20:15", "SUI", "SUIUSDT", 0.584888),
    ("2026-03-18T20:15", "LINK", "LINKUSDT", 0.591600),
]

feature_cols = cfg.features.feature_list

print()
print("=" * 70)
print("SIGNAL COMPARISON: Live Paper Trader vs Backtest Reproduction")
print("=" * 70)
print()
print(f"  {'Time':<18s} {'Asset':<6s} {'Live':>10s} {'Repro':>10s} {'Diff':>10s} {'Match':>6s}")
print("  " + "-" * 62)

for ts_str, asset, symbol, live_prob in live_signals:
    bars = all_bars[symbol]
    data = [{"open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume} for b in bars]
    idx = pd.DatetimeIndex([b.timestamp for b in bars], tz="UTC")
    df = pd.DataFrame(data, index=idx)

    target = pd.Timestamp(ts_str, tz="UTC")
    window = df[df.index <= target]

    if len(window) < 50:
        print(f"  {ts_str:<18s} {asset:<6s} {live_prob:>10.6f} {'N/A':>10s}")
        continue

    feats = compute_features(window, cfg.features)
    last_feats = feats[feature_cols].dropna()
    if len(last_feats) == 0:
        print(f"  {ts_str:<18s} {asset:<6s} {live_prob:>10.6f} {'NO_FEAT':>10s}")
        continue

    X = scaler.transform(last_feats.iloc[[-1]].values)
    repro_prob = float(model.predict_proba(X)[0, 1])
    diff = abs(repro_prob - live_prob)
    match = "YES" if diff < 0.001 else ("CLOSE" if diff < 0.01 else "NO")

    print(f"  {ts_str:<18s} {asset:<6s} {live_prob:>10.6f} {repro_prob:>10.6f} {diff:>10.6f} {match:>6s}")

# Check assets that had NO signals (should be below threshold)
print()
print("Assets with NO signal at 19:00 (should fail confidence >= 0.08):")
for symbol, asset in [("XRPUSDT", "XRP"), ("BNBUSDT", "BNB")]:
    bars = all_bars[symbol]
    data = [{"open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume} for b in bars]
    idx = pd.DatetimeIndex([b.timestamp for b in bars], tz="UTC")
    df = pd.DataFrame(data, index=idx)

    target = pd.Timestamp("2026-03-18T19:00", tz="UTC")
    window = df[df.index <= target]
    feats = compute_features(window, cfg.features)
    last_feats = feats[feature_cols].dropna()
    if len(last_feats) > 0:
        X = scaler.transform(last_feats.iloc[[-1]].values)
        prob = float(model.predict_proba(X)[0, 1])
        conf = abs(prob - 0.5)
        passed = "PASS" if conf >= 0.08 else "FILTERED"
        print(f"  {asset}: prob={prob:.6f} conf={conf:.6f} -> {passed}")
