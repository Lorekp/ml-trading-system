"""EXP-E: Cross-Exchange Price Divergence (Allouche 2025).

Fetch 15m klines from Binance + OKX + Bybit for top assets.
Compute RWM (Robust Weighted Median) price.
Test if exchange deviation is a tradeable signal.
"""
import sys
sys.path.insert(0, "src")

import warnings, os, time
import numpy as np, pandas as pd
import requests
from pathlib import Path

warnings.filterwarnings("ignore")


def fetch_binance_klines(symbol, interval="15m", limit=1000):
    """Binance USDT-M Futures klines."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    data = resp.json()
    if not isinstance(data, list):
        return pd.DataFrame()
    rows = []
    for r in data:
        rows.append({"timestamp": pd.Timestamp(r[0], unit="ms", tz="UTC"),
                      "close": float(r[4]), "volume": float(r[5])})
    return pd.DataFrame(rows).set_index("timestamp")


def fetch_okx_klines(symbol, interval="15m", limit=300):
    """OKX perpetual swap klines."""
    # OKX uses different symbol format: BTC-USDT-SWAP
    okx_sym = symbol.replace("USDT", "-USDT-SWAP")
    # OKX bar: 15m = "15m"
    url = "https://www.okx.com/api/v5/market/candles"
    resp = requests.get(url, params={"instId": okx_sym, "bar": interval, "limit": str(limit)}, timeout=15)
    data = resp.json()
    if data.get("code") != "0" or not data.get("data"):
        return pd.DataFrame()
    rows = []
    for r in data["data"]:
        rows.append({"timestamp": pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
                      "close": float(r[4]), "volume": float(r[5])})
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def fetch_bybit_klines(symbol, interval="15", limit=1000):
    """Bybit linear USDT perpetual klines."""
    url = "https://api.bybit.com/v5/market/kline"
    resp = requests.get(url, params={
        "category": "linear", "symbol": symbol, "interval": interval, "limit": limit
    }, timeout=15)
    data = resp.json()
    if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
        return pd.DataFrame()
    rows = []
    for r in data["result"]["list"]:
        rows.append({"timestamp": pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
                      "close": float(r[4]), "volume": float(r[5])})
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df


def rwm(prices, volumes):
    """Robust Weighted Median (Allouche 2025)."""
    weights = np.log1p(volumes)
    sorted_idx = np.argsort(prices)
    sorted_p = prices[sorted_idx]
    sorted_w = weights[sorted_idx]
    cum_w = np.cumsum(sorted_w)
    mid = cum_w[-1] / 2
    idx = np.searchsorted(cum_w, mid)
    return sorted_p[min(idx, len(sorted_p) - 1)]


def main():
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
               "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]

    print("=" * 80)
    print("EXP-E: CROSS-EXCHANGE PRICE DIVERGENCE")
    print("Fetching 15m klines from Binance, OKX, Bybit")
    print("=" * 80, flush=True)

    for symbol in SYMBOLS:
        print(f"\n--- {symbol} ---", flush=True)

        # Fetch from all 3 exchanges
        t0 = time.time()
        binance = fetch_binance_klines(symbol, limit=500)
        time.sleep(0.2)
        okx = fetch_okx_klines(symbol, limit=300)
        time.sleep(0.2)
        bybit = fetch_bybit_klines(symbol, limit=500)
        print(f"  Fetched: Binance={len(binance)}, OKX={len(okx)}, Bybit={len(bybit)} bars ({time.time()-t0:.1f}s)")

        if len(binance) < 50 or len(okx) < 50 or len(bybit) < 50:
            print("  Insufficient data from one exchange, skipping")
            continue

        # Align timestamps
        common = binance.index.intersection(okx.index).intersection(bybit.index)
        if len(common) < 50:
            print(f"  Only {len(common)} common bars, skipping")
            continue

        b_close = binance.loc[common, "close"].values
        o_close = okx.loc[common, "close"].values
        by_close = bybit.loc[common, "close"].values
        b_vol = binance.loc[common, "volume"].values
        o_vol = okx.loc[common, "volume"].values
        by_vol = bybit.loc[common, "volume"].values

        # Compute RWM and deviations
        deviations = {"binance": [], "okx": [], "bybit": []}
        for i in range(len(common)):
            prices = np.array([b_close[i], o_close[i], by_close[i]])
            volumes = np.array([b_vol[i], o_vol[i], by_vol[i]])
            median = rwm(prices, volumes)
            if median > 0:
                deviations["binance"].append((b_close[i] - median) / median * 10000)  # bps
                deviations["okx"].append((o_close[i] - median) / median * 10000)
                deviations["bybit"].append((by_close[i] - median) / median * 10000)

        if not deviations["binance"]:
            continue

        b_dev = np.array(deviations["binance"])
        o_dev = np.array(deviations["okx"])
        by_dev = np.array(deviations["bybit"])

        print(f"  Binance deviation:  mean={b_dev.mean():>6.2f} bps  std={b_dev.std():>6.2f} bps  max={np.abs(b_dev).max():>6.2f} bps")
        print(f"  OKX deviation:      mean={o_dev.mean():>6.2f} bps  std={o_dev.std():>6.2f} bps  max={np.abs(o_dev).max():>6.2f} bps")
        print(f"  Bybit deviation:    mean={by_dev.mean():>6.2f} bps  std={by_dev.std():>6.2f} bps  max={np.abs(by_dev).max():>6.2f} bps")

        # Is the deviation tradeable at 15m?
        # Need deviation > 2 * slippage (5 bps) = 10 bps to be profitable after costs
        tradeable_bars = np.sum(np.abs(b_dev) > 10)
        print(f"  Bars with |deviation| > 10 bps: {tradeable_bars}/{len(b_dev)} ({100*tradeable_bars/len(b_dev):.1f}%)")

        # Autocorrelation of deviation (does it mean-revert?)
        dev_series = pd.Series(b_dev)
        autocorr_1 = dev_series.autocorr(lag=1) if len(dev_series) > 10 else 0
        print(f"  Deviation autocorrelation(1): {autocorr_1:.3f} (negative = mean-reverting)")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("If deviations are < 5 bps: NOT tradeable at 15m (too small vs slippage)")
    print("If deviations are > 10 bps AND autocorrelation < 0: POTENTIALLY tradeable")
    print("=" * 80)


if __name__ == "__main__":
    main()
