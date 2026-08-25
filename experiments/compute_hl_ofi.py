"""Compute multi-level OFI from Hyperliquid 20-level L2 snapshots (Kolm 2023).

Vectorized: no Python loops over snapshots. ~5s per day per asset.
Run on homeserver: python compute_hl_ofi.py

Output: /hdd/hl-l2/features_15m/{ASSET}_ofi_15m.parquet
"""
import json, time, warnings, os
import lz4.frame
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import skew as scipy_skew
warnings.filterwarnings("ignore")

RAW_DIR = Path("/hdd/hl-l2/raw")
CACHE_DIR = Path("/hdd/hl-l2/features_15m")
CACHE_DIR.mkdir(exist_ok=True)
N_LEVELS = 20


def parse_hour(lz4_path):
    """Parse one hour of L2 snapshots into numpy arrays. Returns None if file missing."""
    if not lz4_path.exists():
        return None
    data = lz4.frame.decompress(open(lz4_path, "rb").read())
    lines = data.decode("utf-8").strip().split("\n")
    n = len(lines)
    if n < 10:
        return None

    ts = np.empty(n, dtype=np.int64)
    bp = np.zeros((n, N_LEVELS), dtype=np.float64)
    bs = np.zeros((n, N_LEVELS), dtype=np.float64)
    ap = np.zeros((n, N_LEVELS), dtype=np.float64)
    az = np.zeros((n, N_LEVELS), dtype=np.float64)
    bn = np.zeros((n, N_LEVELS), dtype=np.int32)
    an = np.zeros((n, N_LEVELS), dtype=np.int32)

    for i, line in enumerate(lines):
        try:
            snap = json.loads(line)
            levels = snap["raw"]["data"]["levels"]
            ts[i] = snap["raw"]["data"]["time"]
            bids = levels[0]
            asks = levels[1]
            for j in range(min(N_LEVELS, len(bids))):
                bp[i, j] = float(bids[j]["px"])
                bs[i, j] = float(bids[j]["sz"])
                bn[i, j] = int(bids[j]["n"])
            for j in range(min(N_LEVELS, len(asks))):
                ap[i, j] = float(asks[j]["px"])
                az[i, j] = float(asks[j]["sz"])
                an[i, j] = int(asks[j]["n"])
        except Exception:
            pass

    return ts, bp, bs, ap, az, bn, an


def compute_ofi_vectorized(bp, bs, ap, az):
    """Kolm (2023) Eq 2-5: vectorized OFI at all 20 levels. Shape: (N, 20)."""
    prev_bp = np.roll(bp, 1, axis=0)
    prev_bs = np.roll(bs, 1, axis=0)
    prev_ap = np.roll(ap, 1, axis=0)
    prev_az = np.roll(az, 1, axis=0)

    # Bid OF (Eq 2)
    bOF = np.where(bp > prev_bp, bs,
           np.where(bp == prev_bp, bs - prev_bs,
           -prev_bs))
    # Ask OF (Eq 3)
    aOF = np.where(ap > prev_ap, -prev_az,
           np.where(ap == prev_ap, az - prev_az,
           az))
    # OFI (Eq 5)
    OFI = bOF - aOF
    OFI[0, :] = 0
    return OFI


def aggregate_to_15m(ts_ms, OFI, bs, az, bp, ap, bn, an):
    """Aggregate per-snapshot data to 15m bar features."""
    bin_idx = ts_ms // 900_000
    unique_bins = np.unique(bin_idx)

    rows = []
    for b in unique_bins:
        m = bin_idx == b
        cnt = m.sum()
        if cnt < 20:
            continue

        ofi = OFI[m]  # (cnt, 20)
        bid_s = bs[m]
        ask_s = az[m]
        bid_n = bn[m]
        ask_n = an[m]
        bid_p = bp[m]
        ask_p = ap[m]

        # Multi-level OFI aggregates
        ofi_per_snap_L1 = ofi[:, 0]
        ofi_per_snap_L5 = ofi[:, :5].sum(axis=1)
        ofi_per_snap_L10 = ofi[:, :10].sum(axis=1)
        ofi_per_snap_L20 = ofi.sum(axis=1)

        # Distance-weighted: OFI[i] / (i+1)
        weights = 1.0 / np.arange(1, N_LEVELS + 1)
        ofi_weighted_per_snap = (ofi * weights).sum(axis=1)

        # Inner vs outer
        ofi_inner = ofi[:, :5].sum(axis=1)
        ofi_outer = ofi[:, 5:10].sum(axis=1)

        # Depth imbalance
        bid_vol_L5 = bid_s[:, :5].sum(axis=1)
        ask_vol_L5 = ask_s[:, :5].sum(axis=1)
        bid_vol_L10 = bid_s[:, :10].sum(axis=1)
        ask_vol_L10 = ask_s[:, :10].sum(axis=1)
        depth_imb_L5 = (bid_vol_L5 - ask_vol_L5) / (bid_vol_L5 + ask_vol_L5 + 1e-10)
        depth_imb_L10 = (bid_vol_L10 - ask_vol_L10) / (bid_vol_L10 + ask_vol_L10 + 1e-10)

        # Depth slope: regress bid volume on level index
        if cnt > 5:
            x = np.arange(min(10, bid_s.shape[1]), dtype=float)
            y = bid_s[:, :len(x)].mean(axis=0)
            if x.std() > 0 and np.all(np.isfinite(y)):
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = 0.0
        else:
            slope = 0.0

        # Spread
        spread_L1 = (ask_p[:, 0] - bid_p[:, 0]).mean()
        spread_L5 = (ask_p[:, 4] - bid_p[:, 4]).mean() if bid_p.shape[1] >= 5 else spread_L1

        # Order count imbalance
        total_bn = bid_n.sum()
        total_an = ask_n.sum()
        n_orders_imb = (total_bn - total_an) / (total_bn + total_an + 1e-10)

        sk = float(scipy_skew(ofi_per_snap_L10)) if cnt > 3 else 0.0

        rows.append({
            "ts_bin": b,
            "ofi_L1": ofi_per_snap_L1.sum(),
            "ofi_L5": ofi_per_snap_L5.sum(),
            "ofi_L10": ofi_per_snap_L10.sum(),
            "ofi_L20": ofi_per_snap_L20.sum(),
            "ofi_L10_std": ofi_per_snap_L10.std(),
            "ofi_L10_skew": sk,
            "ofi_inner_outer": (ofi_inner - ofi_outer).sum(),
            "ofi_weighted": ofi_weighted_per_snap.sum(),
            "depth_imb_L5": depth_imb_L5.mean(),
            "depth_imb_L10": depth_imb_L10.mean(),
            "depth_slope": slope,
            "depth_resilience": depth_imb_L10.std(),
            "spread_L1": spread_L1,
            "spread_L5": spread_L5,
            "n_orders_imb": n_orders_imb,
            "snap_count": float(cnt),
        })

    if not rows:
        return pd.DataFrame()
    bars = pd.DataFrame(rows)
    bars.index = pd.to_datetime(bars["ts_bin"] * 900_000, unit="ms", utc=True)
    return bars.drop(columns=["ts_bin"]).astype(np.float32)


def process_day(asset, date_str):
    """Process one day: load 24 hours, compute OFI, aggregate to 15m."""
    all_ts, all_bp, all_bs, all_ap, all_az, all_bn, all_an = [], [], [], [], [], [], []

    for hour in range(24):
        lz4_path = RAW_DIR / date_str / str(hour) / "l2Book" / f"{asset}.lz4"
        result = parse_hour(lz4_path)
        if result is None:
            continue
        ts, bp, bs, ap, az, bn, an = result
        all_ts.append(ts)
        all_bp.append(bp)
        all_bs.append(bs)
        all_ap.append(ap)
        all_az.append(az)
        all_bn.append(bn)
        all_an.append(an)

    if not all_ts:
        return pd.DataFrame()

    ts = np.concatenate(all_ts)
    bp = np.vstack(all_bp)
    bs = np.vstack(all_bs)
    ap = np.vstack(all_ap)
    az = np.vstack(all_az)
    bn = np.vstack(all_bn)
    an = np.vstack(all_an)

    OFI = compute_ofi_vectorized(bp, bs, ap, az)
    return aggregate_to_15m(ts, OFI, bs, az, bp, ap, bn, an)


def process_asset(asset):
    """Process all available days for one asset."""
    cache_path = CACHE_DIR / f"{asset}_ofi_15m.parquet"
    if cache_path.exists():
        return f"{asset}: cached"

    # Find all dates with data for this asset
    dates = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    all_bars = []
    for d in dates:
        try:
            bars = process_day(asset, d)
            if len(bars) > 0:
                all_bars.append(bars)
        except Exception:
            pass

    if not all_bars:
        return f"{asset}: no data"

    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.to_parquet(str(cache_path))
    return f"{asset}: {len(combined)} bars ({combined.index[0].date()} to {combined.index[-1].date()})"


if __name__ == "__main__":
    t0 = time.time()

    # Find available assets
    sample_dir = RAW_DIR / sorted(os.listdir(RAW_DIR))[-1]
    sample_hour = sorted(os.listdir(sample_dir))[0]
    sample_l2 = sample_dir / sample_hour / "l2Book"
    available = [f.stem for f in sample_l2.glob("*.lz4")] if sample_l2.exists() else []

    # Our trading assets (HL uses short names)
    targets = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "AVAX", "DOT",
               "LINK", "NEAR", "UNI", "AAVE", "LTC", "BCH", "HBAR", "TRX",
               "SUI", "APT", "ENJ", "SXP", "ZEC", "XLM", "FET", "WLD", "ANKR"]
    assets = [a for a in targets if a in available]

    print(f"Computing multi-level OFI for {len(assets)} assets...")
    print(f"Available dates: {len(sorted(os.listdir(RAW_DIR)))}")

    # Benchmark one day
    bt = time.time()
    test = process_day("BTC", sorted(os.listdir(RAW_DIR))[500])
    print(f"  Benchmark: 1 day = {time.time()-bt:.1f}s, {len(test)} bars")

    # Process all assets (sequential — JSON parsing is CPU-bound, parallel helps less)
    for asset in assets:
        at = time.time()
        result = process_asset(asset)
        print(f"  {result} ({time.time()-at:.0f}s)")

    print(f"Done in {time.time()-t0:.0f}s")
