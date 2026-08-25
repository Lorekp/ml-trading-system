"""Tier 1 Improvement Experiments — fully parallelised.

Uses ProcessPoolExecutor for per-asset processing within each experiment.
"""
import sys
sys.path.insert(0, "src")

import copy, warnings, os, json, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

from pathlib import Path

W = max(1, os.cpu_count() - 1)

PROD_ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}
DATA_DIR = "data/binance-1m"


def _process_asset_baseline(args):
    """Process one asset with production config. Runs in subprocess."""
    asset, symbol = args
    import warnings; warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from pathlib import Path
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.filters import apply_confidence_filter

    cfg = load_config(Path("configs/default.yaml"))
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08

    try:
        ohlcv_1m = load_binance_1m(symbol, [Path(DATA_DIR)])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)
        wf = WalkForwardModel(cfg)
        raw = wf.generate_signals_batch(ohlcv_15m, features, asset)
        if len(raw) == 0:
            return asset, None, None
        filtered = apply_confidence_filter(raw, cfg.confidence_filter)
        # Also return close + entropy for EXP-B
        from afml.features import rolling_entropy
        entropy = rolling_entropy(ohlcv_15m["close"], window=50)
        # Return close series for CUSUM
        close = ohlcv_15m["close"]
        return asset, filtered.to_dict("list"), {
            "entropy": entropy.to_dict(),
            "close": close.to_dict(),
        }
    except Exception as e:
        return asset, None, None


def _combine_and_sim(all_signals_dicts, portfolio_cfg):
    """Combine signal dicts and run portfolio sim."""
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    from signal_engine.stats import compute_full_stats
    from signal_engine.config import PortfolioConfig

    frames = []
    for d in all_signals_dicts:
        frames.append(pd.DataFrame(d))

    if not frames:
        return None

    combined = pd.concat(frames).sort_values("timestamp")
    sim = simulate_portfolio(combined, portfolio_cfg)
    if sim is None or sim["n_trades"] < 10:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None

    stats = compute_full_stats(dpnl, portfolio_cfg.starting_capital,
                                sim["n_trades"], sim["n_trades"] / len(dpnl))
    stats["_pnl_day"] = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return stats


def print_result(label, stats):
    if stats is None:
        print(f"  {label:<40s} FAILED", flush=True)
        return
    print(
        f"  {label:<40s} Sharpe={stats['sharpe']:>6.3f}  Ret={stats['total_return_pct']:>7.2f}%  "
        f"DD={stats['max_dd_pct']:>5.2f}%  Trades={stats['n_trades']:>6d}  "
        f"PnL/day=${stats['_pnl_day']:>6.2f}",
        flush=True,
    )


def main():
    from signal_engine.config import load_config, PortfolioConfig

    cfg = load_config(Path("configs/default.yaml"))
    portfolio_cfg = copy.deepcopy(cfg.portfolio)
    portfolio_cfg.base_sizing_pct = 0.04
    portfolio_cfg.slippage = 0.0005
    portfolio_cfg.max_exposure = 0.90
    portfolio_cfg.max_per_asset = 0.03

    print("=" * 80)
    print(f"TIER 1 EXPERIMENTS ({W} CPU workers)")
    print("=" * 80, flush=True)

    # Phase 1: Generate all signals in parallel (shared across experiments)
    print("\nPhase 1: Generating signals for 30 assets in parallel...", flush=True)
    t0 = time.time()

    asset_signals = {}  # {asset: signals_dict}
    asset_extra = {}    # {asset: {entropy, close}}

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {
            pool.submit(_process_asset_baseline, (a, s)): a
            for a, s in PROD_ASSETS.items()
        }
        done = 0
        for future in as_completed(futures):
            asset, sig_dict, extra = future.result()
            done += 1
            if sig_dict is not None:
                asset_signals[asset] = sig_dict
                asset_extra[asset] = extra
            print(f"  [{done:>2d}/30] {asset}: {'OK' if sig_dict else 'skip'}", flush=True)

    gen_time = time.time() - t0
    print(f"\nSignals generated in {gen_time:.0f}s ({len(asset_signals)} assets with signals)")

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n--- BASELINE (production config) ---", flush=True)
    all_baseline = [v for v in asset_signals.values()]
    baseline = _combine_and_sim(all_baseline, portfolio_cfg)
    print_result("Production baseline", baseline)

    # =========================================================================
    # EXP-A: CUSUM EVENT FILTER
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXP-A: CUSUM EVENT FILTER")
    print("Keep signals only at bars where CUSUM detects meaningful price move")
    print("=" * 80, flush=True)

    from afml.filters import cusum_filter, get_daily_vol

    for h_mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        cusum_signals = []
        for asset, sig_dict in asset_signals.items():
            extra = asset_extra.get(asset, {})
            close_dict = extra.get("close", {})
            if not close_dict:
                cusum_signals.append(sig_dict)
                continue

            close = pd.Series(close_dict)
            close.index = pd.to_datetime(close.index)
            vol = get_daily_vol(close, span=50)
            threshold = vol * close * h_mult
            threshold = threshold.dropna()
            if len(threshold) < 100:
                cusum_signals.append(sig_dict)
                continue

            events = cusum_filter(close, threshold=threshold)
            if len(events) == 0:
                continue

            df = pd.DataFrame(sig_dict)
            sig_times = pd.to_datetime(df["timestamp"])
            if sig_times.dt.tz is not None:
                sig_times = sig_times.dt.tz_localize(None)

            cusum_set = set(events)
            mask = sig_times.isin(cusum_set)
            kept = df[mask.values]
            if len(kept) > 0:
                cusum_signals.append(kept.to_dict("list"))

        stats = _combine_and_sim(cusum_signals, portfolio_cfg)
        print_result(f"CUSUM h={h_mult:.1f}", stats)

    # =========================================================================
    # EXP-B: ENTROPY AS FILTER
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXP-B: ENTROPY FILTER (trade only in orderly markets)")
    print("Keep signals where entropy <= Pth percentile")
    print("=" * 80, flush=True)

    for pct_keep in [100, 80, 60, 40, 20]:
        entropy_signals = []
        for asset, sig_dict in asset_signals.items():
            if pct_keep >= 100:
                entropy_signals.append(sig_dict)
                continue

            extra = asset_extra.get(asset, {})
            ent_dict = extra.get("entropy", {})
            if not ent_dict:
                entropy_signals.append(sig_dict)
                continue

            entropy = pd.Series(ent_dict)
            entropy.index = pd.to_datetime(entropy.index)

            df = pd.DataFrame(sig_dict)
            sig_times = pd.to_datetime(df["timestamp"])
            if sig_times.dt.tz is not None:
                sig_times = sig_times.dt.tz_localize(None)

            # Ensure timezone alignment
            if entropy.index.tz is not None and sig_times.dt.tz is None:
                sig_times_aligned = sig_times.dt.tz_localize("UTC")
            elif entropy.index.tz is None and sig_times.dt.tz is not None:
                sig_times_aligned = sig_times.dt.tz_localize(None)
            else:
                sig_times_aligned = sig_times
            ent_aligned = entropy.reindex(sig_times_aligned, method="ffill")
            thresh = np.nanpercentile(entropy.dropna().values, pct_keep)
            mask = ent_aligned.values <= thresh
            kept = df[mask]
            if len(kept) > 0:
                entropy_signals.append(kept.to_dict("list"))

        stats = _combine_and_sim(entropy_signals, portfolio_cfg)
        print_result(f"Entropy keep<={pct_keep}%", stats)

    # =========================================================================
    print("\n" + "=" * 80)
    print("TIER 1 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
