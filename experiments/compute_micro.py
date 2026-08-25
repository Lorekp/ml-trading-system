"""Compute all 12 microstructure features from trades + bookDepth + bookTicker.
Run on homeserver: PYTHONPATH=src python experiments/compute_micro.py
"""
import time, zipfile, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

L2 = Path("/hdd/binance-l2")
CACHE = L2 / "features_15m"
CACHE.mkdir(exist_ok=True)
PQ_BT = L2 / "bookTicker_parquet"

SYMBOLS = [s for s in [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","ADAUSDT",
    "AVAXUSDT","DOTUSDT","LINKUSDT","NEARUSDT","UNIUSDT","AAVEUSDT",
    "LTCUSDT","BCHUSDT","HBARUSDT","TRXUSDT","SUIUSDT","APTUSDT",
    "ENJUSDT","SXPUSDT","ZECUSDT","XLMUSDT","FETUSDT","WLDUSDT","ANKRUSDT",
] if (L2 / "trades" / s).exists()]

TRADE_COLS = ["id","price","qty","quoteQty","time","isBuyerMaker"]


def compute_day(sym, date_str):
    """Compute all features for one symbol-day → 15m DataFrame."""
    # Load trades
    trade_zp = L2 / "trades" / sym / f"{sym}-trades-{date_str}.zip"
    if not trade_zp.exists():
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(trade_zp) as zf:
            with zf.open(zf.namelist()[0]) as f:
                # Peek first line to detect header
                first_line = f.readline().decode("utf-8", errors="ignore").strip()
                f.seek(0)
                has_header = "price" in first_line.lower() or "id" == first_line.split(",")[0].strip().lower()
                trades = pd.read_csv(f,
                    header=0 if has_header else None,
                    names=None if has_header else TRADE_COLS,
                    dtype={"price":"float64","qty":"float64","quoteQty":"float64",
                           "time":"int64","isBuyerMaker":"bool"})
                # Normalize column names
                trades.columns = [c.strip().lower() for c in trades.columns]
                col_map = {"id":"id","price":"price","qty":"qty","quote_qty":"quoteQty",
                           "quotequty":"quoteQty","quoteqty":"quoteQty",
                           "time":"time","is_buyer_maker":"isBuyerMaker",
                           "isbuyermaker":"isBuyerMaker"}
                trades = trades.rename(columns={c: col_map.get(c, c) for c in trades.columns})
    except Exception:
        return pd.DataFrame()
    if len(trades) < 100:
        return pd.DataFrame()

    ts_ms = trades["time"].values
    price = trades["price"].values
    qty = trades["qty"].values
    dollar_vol = price * qty
    is_buy = ~trades["isBuyerMaker"].values

    bin_idx = (ts_ms // 900_000).astype(np.int64)
    unique_bins = np.unique(bin_idx)

    # Load bookDepth
    depth_zp = L2 / "bookDepth" / sym / f"{sym}-bookDepth-{date_str}.zip"
    depth_df = None
    if depth_zp.exists():
        try:
            with zipfile.ZipFile(depth_zp) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    depth_df = pd.read_csv(f)
            depth_df["timestamp"] = pd.to_datetime(depth_df["timestamp"], utc=True)
        except Exception:
            depth_df = None

    # Load bookTicker tick count per bin
    bt_pq = PQ_BT / sym / f"{date_str}.parquet"
    bt_count = {}
    if bt_pq.exists():
        try:
            bt = pq.read_table(str(bt_pq)).to_pandas()
            bt_bins = (bt["transaction_time"].values // 900_000).astype(np.int64)
            for b in unique_bins:
                bt_count[b] = int((bt_bins == b).sum())
        except Exception:
            pass

    large_threshold = np.percentile(dollar_vol, 95) if len(dollar_vol) > 100 else 1e12

    rows = []
    for b in unique_bins:
        m = bin_idx == b
        cnt = m.sum()
        if cnt < 10:
            continue

        p = price[m]
        dv = dollar_vol[m]
        ib = is_buy[m]
        buy_vol = dv[ib].sum()
        sell_vol = dv[~ib].sum()
        total_vol = buy_vol + sell_vol

        trade_imb = (buy_vol - sell_vol) / (total_vol + 1e-10)
        vpin = abs(buy_vol - sell_vol) / (total_vol + 1e-10)
        ret = (p[-1] - p[0]) / (p[0] + 1e-10) if p[0] > 0 else 0
        amihud = abs(ret) / (total_vol / 1e6 + 1e-10)
        large_frac = dv[dv > large_threshold].sum() / (total_vol + 1e-10)

        # Depth features
        d_imb1 = d_imb2 = d_total2 = d_slope = 0.0
        if depth_df is not None:
            bar_start = pd.Timestamp(b * 900, unit="s", tz="UTC")
            bar_end = bar_start + pd.Timedelta(minutes=15)
            bd = depth_df[(depth_df["timestamp"] >= bar_start) & (depth_df["timestamp"] < bar_end)]
            if len(bd) > 0:
                bids = bd[bd["percentage"] < 0]
                asks = bd[bd["percentage"] > 0]
                b1 = bids[bids["percentage"] == -1]["notional"].mean()
                a1 = asks[asks["percentage"] == 1]["notional"].mean()
                b2 = bids[bids["percentage"].isin([-1,-2])]["notional"].mean()
                a2 = asks[asks["percentage"].isin([1,2])]["notional"].mean()
                if np.isfinite(b1) and np.isfinite(a1):
                    d_imb1 = (b1 - a1) / (b1 + a1 + 1e-10)
                if np.isfinite(b2) and np.isfinite(a2):
                    d_imb2 = (b2 - a2) / (b2 + a2 + 1e-10)
                    d_total2 = b2 + a2
                if len(bids) > 2:
                    x = np.abs(bids["percentage"].values).astype(float)
                    y = bids["notional"].values.astype(float)
                    if x.std() > 0 and np.all(np.isfinite(y)):
                        d_slope = float(np.polyfit(x, y, 1)[0])

        qt_ratio = bt_count.get(b, 0) / (cnt + 1e-10)

        rows.append({
            "ts_bin": b,
            "ms_trade_imbalance": trade_imb,
            "ms_vpin": vpin,
            "ms_kyle_lambda": 0.0,  # filled after concat
            "ms_amihud": amihud,
            "ms_trade_count": float(cnt),
            "ms_large_trade_frac": large_frac,
            "ms_depth_imb_1pct": d_imb1,
            "ms_depth_imb_2pct": d_imb2,
            "ms_depth_total_2pct": d_total2,
            "ms_depth_slope": d_slope,
            "ms_effective_spread": 0.0,  # needs tick-level matching, skip for now
            "ms_quote_trade_ratio": qt_ratio,
        })

    if not rows:
        return pd.DataFrame()
    bars = pd.DataFrame(rows)
    bars.index = pd.to_datetime(bars["ts_bin"] * 900_000, unit="ms", utc=True)
    return bars.drop(columns=["ts_bin"]).astype(np.float32)


def process_symbol(sym):
    cache_path = CACHE / f"{sym}_micro_15m.parquet"
    if cache_path.exists():
        return f"{sym}: cached"

    trade_dir = L2 / "trades" / sym
    zips = sorted(trade_dir.glob("*.zip"))
    dates = []
    for zp in zips:
        parts = zp.stem.split("-trades-")
        if len(parts) == 2:
            d = parts[1]
            if "2023-05-16" <= d <= "2024-03-30":
                dates.append(d)

    if not dates:
        return f"{sym}: no dates in range"

    all_bars = []
    for d in dates:
        try:
            bars = compute_day(sym, d)
            if len(bars) > 0:
                all_bars.append(bars)
        except Exception:
            pass

    if not all_bars:
        return f"{sym}: all failed"

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined["ms_kyle_lambda"] = combined["ms_amihud"].rolling(20, min_periods=5).mean()
    combined.to_parquet(str(cache_path))
    return f"{sym}: {len(combined)} bars ({combined.index[0].date()} to {combined.index[-1].date()})"


if __name__ == "__main__":
    t0 = time.time()
    print(f"Computing microstructure features for {len(SYMBOLS)} symbols...")

    for old in CACHE.glob("*_micro_15m.parquet"):
        old.unlink()

    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(process_symbol, s): s for s in SYMBOLS}
        for f in as_completed(futs):
            print(f"  {f.result()}")
    print(f"Done in {time.time()-t0:.0f}s")
