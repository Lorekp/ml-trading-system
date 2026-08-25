#!/usr/bin/env python
"""Side-by-side BT vs paper position sizing for every trade after --since.

Replays the BT sizing pipeline (equity, peak, breaker, caps) sequentially for
each paper trade and compares to what the paper trader actually executed.
"""
import sys, os, sqlite3, json, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import requests
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig
from live.config import load_live_config, BreakerConfig
from live.signal_generator import SklearnBackend
from live.breaker import F1SymmetricBreaker


def _fetch_vol_series(asset, end_time):
    end_ms = int(end_time.timestamp() * 1000)
    params = {"symbol": f"{asset}USDT", "interval": "1d", "endTime": end_ms, "limit": 200}
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=params, timeout=15)
        if r.status_code != 200: return None
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, list) or len(data) < 30:
        return None
    rows = [{"ts": pd.Timestamp(int(k[0]), unit="ms"), "close": float(k[4])} for k in data]
    df = pd.DataFrame(rows).set_index("ts")
    return df["close"].pct_change().rolling(20).std()

try:
    from afml.bet_sizing import get_signal, discretize_signal
except ImportError:
    def get_signal(prob, num_classes=2): return 2.0 * prob - 1.0
    def discretize_signal(sig, step_size=0.1): return round(sig / step_size) * step_size


def _edge(prob, sig):
    if np.isfinite(prob):
        return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0))
    return float(np.clip(abs(sig), 0.0, 1.0))


TF_MIN = {"15m": 15, "1h": 60, "3h": 180, "4h": 240, "24h": 1440}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/live/paper_binance_snapshot.db")
    parser.add_argument("--since", default="2026-04-09T18:15")
    parser.add_argument("--config", default="configs/paper_binance.yaml")
    args = parser.parse_args()

    cfg = load_live_config(Path(args.config))
    strat_cfg = cfg.strategies.get("s8_sym15", {})
    port_raw = strat_cfg.get("portfolio") or {}
    port_cfg = cfg.portfolio.model_copy(update=port_raw) if port_raw else cfg.portfolio
    breaker_cfg = BreakerConfig(**strat_cfg.get("breaker"))
    initial = cfg.capital

    # Load models + feature config
    backend = SklearnBackend()
    n_models = backend.load_models(Path("data/live/models"))
    print(f"  Models: {n_models}")

    # Vol overlay params
    vo = strat_cfg.get("vol_overlay") or {}
    vol_target = vo.get("target_vol", 0.025)
    vol_floor = vo.get("floor", 0.3)
    vol_cap = vo.get("cap", 1.5)
    fc = FeatureConfig(
        feature_list=[
            "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
            "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
            "dist_swing_high", "dist_swing_low",
        ],
        winsor_lo=0.01, winsor_hi=0.99, vol_filter_quantile=0.50,
        vol_zscore_window=30, swing_lookback=20,
    )

    # Load positions + close events
    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    c.execute(
        """SELECT asset, timeframe, side, entry_time, exit_time, entry_price,
                  exit_price, size_usd, pnl, hold_bars, status
           FROM positions WHERE strategy_id='s8_sym15'
             AND entry_time >= ?
           ORDER BY entry_time""",
        (args.since,),
    )
    rows = c.fetchall()
    conn.close()

    # Build event stream
    events = []
    for r in rows:
        events.append((r[3], "open", r))  # entry_time
        if r[10] == "closed" and r[4] is not None:
            events.append((r[4], "close", r))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))

    # Preload OHLCV + vol_series for unique assets
    assets = sorted(set(r[0] for r in rows))
    print(f"  Loading OHLCV for {len(assets)} assets...")
    ohlcv = {}
    vol_series = {}
    # Cap vol fetch at --since time to mirror when the paper trader cached it
    since_ts = pd.Timestamp(args.since.replace("T", " "))
    for a in assets:
        try:
            raw = load_binance_1m(f"{a}USDT", [Path("data/binance-1m")])
            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
        except Exception:
            continue
        for tf, rs in [("15m", "15min"), ("1h", "1h"), ("3h", "3h"),
                        ("4h", "4h"), ("24h", "1D")]:
            try:
                ohlcv[(a, tf)] = resample_ohlcv(raw, rs)
            except Exception:
                pass
        vs = _fetch_vol_series(a, since_ts)
        if vs is not None:
            vol_series[a] = vs

    def vol_mult_at(asset, ts):
        vs = vol_series.get(asset)
        if vs is None or len(vs) == 0:
            return 1.0
        v = vs.iloc[-1]  # MATCH LIVE: last value, not searchsorted
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return float(np.clip(vol_target / v, vol_floor, vol_cap))

    # Replay
    breaker = F1SymmetricBreaker.from_config(breaker_cfg)
    equity = initial
    peak = initial
    open_pos = {}  # (entry_ts, asset, tf) -> size

    def total_exposure():
        return sum(open_pos.values())

    def asset_exposure(a):
        return sum(s for k, s in open_pos.items() if k[1] == a)

    print()
    print(f"{'Time':<17s} {'Asset':>5s} {'TF':<4s} {'Side':<5s}  "
          f"{'Prob':>6s} {'Edge':>5s} {'Brk':>5s} {'VMlt':>5s}  "
          f"{'Paper':>9s}  {'BT':>9s}  {'Diff':>10s} {'Diff%':>7s}")
    print("-" * 120)

    total_paper = total_bt = total_diff = n = 0

    for ev_time, ev_type, r in events:
        asset, tf, side, entry_time, exit_time, ep, xp, size, pnl, hold, status = r

        if ev_type == "close":
            key = (entry_time[:19], asset, tf)
            if key in open_pos:
                sz = open_pos.pop(key)
                equity += pnl or 0
                if equity > peak:
                    peak = equity
                breaker.record_trade(asset, pnl or 0, sz)
            continue

        # OPEN
        entry_ts = pd.Timestamp(entry_time[:19])
        tf_m = TF_MIN[tf]
        total_min = entry_ts.hour * 60 + entry_ts.minute
        cutoff_min = (total_min // tf_m) * tf_m
        bar_cutoff = entry_ts.replace(
            hour=cutoff_min // 60, minute=cutoff_min % 60,
            second=0, microsecond=0)

        df = ohlcv.get((asset, tf))
        if df is None:
            continue
        df_sl = df[df.index < bar_cutoff]
        if len(df_sl) < 30:
            continue
        feat = compute_features(df_sl, fc)
        if len(feat) == 0:
            continue
        last = feat.iloc[[-1]]
        cols = [c for c in fc.feature_list if c in last.columns]
        if last[cols].isna().any().any():
            continue
        prob = backend.predict(asset, tf, last[cols].values)
        if prob is None:
            continue

        sig_v = get_signal(prob, num_classes=2)
        sig_disc = discretize_signal(sig_v, step_size=0.1)
        strength = abs(float(sig_disc))
        edge = _edge(prob, strength)

        # BT sizing (same as BTPortfolio.compute_size, vol_mult=1.0)
        dd = max(0, (peak - equity) / peak) if peak > 0 else 0
        if dd > port_cfg.delever_start:
            size_mult = max(
                1.0 - (dd - port_cfg.delever_start)
                / (port_cfg.max_dd - port_cfg.delever_start), 0.0)
        else:
            size_mult = 1.0

        vmult = vol_mult_at(asset, entry_ts)
        raw = equity * port_cfg.base_sizing_pct * edge * size_mult * vmult
        raw = min(raw, equity * port_cfg.base_sizing_pct
                  * port_cfg.max_signal_mult * size_mult)
        brk_mult = breaker.get_multiplier(asset)
        raw *= brk_mult

        a_exp = asset_exposure(asset)
        a_cap = equity * port_cfg.max_per_asset
        if a_exp + raw > a_cap:
            raw = max(a_cap - a_exp, 0)
        t_exp = total_exposure()
        t_cap = equity * port_cfg.max_exposure
        if t_exp + raw > t_cap:
            raw = max(t_cap - t_exp, 0)
        # HyroTrader margin
        margin_limit = initial * port_cfg.max_margin_pct
        new_margin = raw / cfg.leverage
        cur_margin = t_exp / cfg.leverage
        if cur_margin + new_margin > margin_limit:
            raw = max((margin_limit - cur_margin) * cfg.leverage, 0)
        notional_limit = initial * port_cfg.max_notional_mult
        if t_exp + raw > notional_limit:
            raw = max(notional_limit - t_exp, 0)

        bt_size = raw if raw >= 1.0 else 0.0
        key = (entry_time[:19], asset, tf)
        # Track EXPOSURE via paper_size (lockstep with paper), but show bt_size
        open_pos[key] = size

        diff = size - bt_size
        diff_pct = (diff / bt_size * 100) if bt_size > 0 else 0
        total_paper += size
        total_bt += bt_size
        total_diff += diff
        n += 1

        print(f"{entry_time[:17]:<17s} {asset:>5s} {tf:<4s} {side:<5s}  "
              f"{prob:>6.4f} {edge:>5.3f} {brk_mult:>5.2f} {vmult:>5.2f}  "
              f"${size:>7,.0f}  ${bt_size:>7,.0f}  ${diff:>+8,.0f} {diff_pct:>+6.1f}%")

    print("-" * 112)
    print(f"TOTALS ({n} positions)                                 "
          f"${total_paper:>7,.0f}  ${total_bt:>7,.0f}  ${total_diff:>+8,.0f} "
          f"{total_diff/total_bt*100 if total_bt else 0:>+6.1f}%")
    print()
    print(f"Paper total notional:  ${total_paper:>12,.0f}")
    print(f"BT total notional:     ${total_bt:>12,.0f}")
    print(f"Paper/BT ratio:        {total_paper/total_bt:.4f}" if total_bt else "N/A")


if __name__ == "__main__":
    main()
