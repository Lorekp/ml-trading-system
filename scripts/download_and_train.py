#!/usr/bin/env python
"""Download missing Binance Vision data + train models for specified assets.

End-to-end pipeline for new asset onboarding:
  Phase 1: Download 1m OHLCV from Binance Vision (no auth, bulk zips)
  Phase 2: Train per-asset per-TF walk-forward models

Usage:
    PYTHONPATH=src python scripts/download_and_train.py --assets SOL,TRUMP,AVAX
    PYTHONPATH=src python scripts/download_and_train.py --assets SOL --dry-run
"""

import io
import os
import sys
import time
import json
import zipfile
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import requests

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance-1m")
MODEL_DIR = Path("data/live/models")
VISION_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
MIN_MONTHS = 6  # walk-forward model needs 180 days (6 months) minimum
DRY_RUN = "--dry-run" in sys.argv
RETRAIN = "--retrain" in sys.argv  # monthly retrain: download new data for all assets

# ── Reuse training config from train_live_models.py ──

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]
TFS = [("15m", "15min"), ("1h", "1h"), ("3h", "3h"), ("4h", "4h"), ("24h", "1D")]


def parse_assets() -> list[str]:
    """Parse --assets flag."""
    for i, arg in enumerate(sys.argv):
        if arg == "--assets" and i + 1 < len(sys.argv):
            return [a.strip() for a in sys.argv[i + 1].split(",") if a.strip()]
    print("Usage: --assets SOL,TRUMP,AVAX [--dry-run]")
    sys.exit(1)


# ════════════════════════════════════════════
#  Phase 1: Download from Binance Vision
# ════════════════════════════════════════════

def download_month(symbol: str, year: int, month: int) -> tuple[int, str | None]:
    """Download one month of 1m data. Returns (n_rows, error_or_None)."""
    ym = f"{year}-{month:02d}"
    filename = f"{symbol}-1m-{ym}"
    out_dir = DATA_DIR / symbol
    csv_path = out_dir / f"{filename}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 100:
        return -1, None  # already exists

    url = f"{VISION_URL}/{symbol}/1m/{filename}.zip"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return 0, None  # not available
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return 0, "no CSV in zip"
            csv_data = zf.read(csv_names[0]).decode("utf-8")

        lines = csv_data.strip().split("\n")
        if lines and not lines[0][0].isdigit():
            lines = lines[1:]
        if not lines:
            return 0, "empty CSV"

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 6:
                    f.write(",".join(parts[:6]) + "\n")

        return len(lines), None

    except Exception as e:
        return 0, str(e)[:80]


def download_asset(asset: str) -> tuple[str, int, int, list[str]]:
    """Download all available months for one asset.

    Returns (asset, new_months, existing_months, errors).
    """
    symbol = f"{asset}USDT"
    now = datetime.now()

    # Generate months: 24 months back to previous month (current month not on Vision)
    months = []
    for offset in range(24, 0, -1):  # stop at 1, skip current month (offset=0)
        y = now.year - (offset // 12)
        m = now.month - (offset % 12)
        if m <= 0:
            y -= 1
            m += 12
        months.append((y, m))

    new, existing = 0, 0
    errors = []
    consecutive_404 = 0

    for year, month in months:
        n_rows, err = download_month(symbol, year, month)
        if n_rows == -1:
            existing += 1
            consecutive_404 = 0
        elif n_rows > 0:
            new += 1
            consecutive_404 = 0
        elif err:
            errors.append(f"{year}-{month:02d}: {err}")
            consecutive_404 = 0
        else:
            consecutive_404 += 1
            if consecutive_404 >= 3 and (new + existing) > 0:
                break

    return asset, new, existing, errors


def count_existing_months(asset: str) -> int:
    """Count monthly CSV files on disk (exclude daily files).

    Monthly files: {SYM}-1m-YYYY-MM.csv  (17 chars after -1m-)
    Daily files:   {SYM}-1m-YYYY-MM-DD.csv (20 chars after -1m-)
    """
    import re
    sym_dir = DATA_DIR / f"{asset}USDT"
    if not sym_dir.exists():
        return 0
    monthly_pat = re.compile(rf"^{asset}USDT-1m-\d{{4}}-\d{{2}}\.csv$")
    return sum(1 for f in sym_dir.iterdir() if monthly_pat.match(f.name))


# ════════════════════════════════════════════
#  Phase 2: Train models
# ════════════════════════════════════════════

def train_one(args):
    """Train one (asset, TF) pair. Returns (asset, tf, n_signals, error)."""
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.config import (
        SystemConfig, ModelConfig, FeatureConfig, RegimeFilterConfig,
        ConfidenceFilterConfig, SignalConfig, DataConfig,
    )
    import joblib

    name, sym, tf_label, tf_resample = args
    warnings.filterwarnings("ignore")

    out_dir = MODEL_DIR / tf_label
    model_path = out_dir / f"{name}_model.joblib"
    scaler_path = out_dir / f"{name}_scaler.joblib"

    try:
        config = SystemConfig(
            model=ModelConfig(
                C=0.10, penalty="elasticnet", solver="saga", max_iter=1000,
                seed=42, fwd_bars=4, signal_model_type="asset_logistic",
            ),
            features=FeatureConfig(
                feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
            ),
            regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
            confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
            signal=SignalConfig(step_size=0.1),
            data=DataConfig(
                data_dirs=[str(DATA_DIR)],
                timeframe=tf_resample, regime_timeframe="4h",
            ),
        )

        ohlcv_1m = load_binance_1m(sym, [DATA_DIR])
        ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
        feat = compute_features(ohlcv_tf, config.features)

        try:
            wf = WalkForwardModel(config, force_monthly_retrain=True)
        except TypeError:
            wf = WalkForwardModel(config)
        sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)

        if len(sigs) == 0:
            return name, tf_label, 0, "no signals"

        model = wf._cached_model
        scaler = wf._cached_scaler
        if model is None or scaler is None:
            return name, tf_label, 0, "no model after training"

        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Bake per-(asset, tf) sidecar metadata: the last-retrain thresholds
        # that the backtest uses. Live inference reads these from the JSON so
        # the two pipelines apply the same filter thresholds (fixes D1 + D2).
        meta_path = out_dir / f"{name}_meta.json"
        meta = {
            "asset": name,
            "tf": tf_label,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "features": list(OHLCV_FEATURES),
            "vol_filter_quantile": 0.50,
            "conf_threshold_pct": 70.0,
            "conf_threshold": wf.last_conf_threshold,
            "vol_threshold": wf.last_vol_threshold,
            "n_signals": int(len(sigs)),
        }
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            # Metadata write failure is non-fatal — live still works with
            # fallback thresholds. Log and continue.
            print(f"    WARN: failed to write {meta_path.name}: {e}")

        return name, tf_label, len(sigs), None

    except FileNotFoundError:
        return name, tf_label, 0, "no data"
    except Exception as e:
        return name, tf_label, 0, f"{type(e).__name__}: {str(e)[:80]}"


# ════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════

def main():
    T0 = time.time()
    assets = parse_assets()

    print("=" * 70)
    print(f"  DOWNLOAD & TRAIN — {'Monthly Retrain' if RETRAIN else 'New Asset Onboarding'}")
    print("=" * 70)
    print(f"  Assets: {assets}")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'RETRAIN' if RETRAIN else 'LIVE'}")
    print()

    # ── Phase 1: Download ──
    print("  Phase 1: Downloading from Binance Vision...")
    need_download = []
    sufficient = []

    if RETRAIN:
        # Monthly retrain: download ALL assets to pick up the newest month's data.
        # download_month() skips files already on disk, so only new months are fetched.
        need_download = list(assets)
    else:
        for asset in assets:
            n_existing = count_existing_months(asset)
            if n_existing >= MIN_MONTHS:
                print(f"    {asset}: {n_existing} months on disk (sufficient)")
                sufficient.append(asset)
            else:
                print(f"    {asset}: {n_existing} months on disk (need {MIN_MONTHS}+, downloading...)")
                need_download.append(asset)

    if need_download and not DRY_RUN:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = {pool.submit(download_asset, a): a for a in need_download}
            for fut in as_completed(futs):
                asset, new, existing, errors = fut.result()
                total = new + existing
                status = "OK" if total >= MIN_MONTHS else "INSUFFICIENT"
                print(f"    {asset}: +{new} new, {existing} cached = {total} months [{status}]")
                if errors:
                    for e in errors[:3]:
                        print(f"      ERROR: {e}")
                if total >= MIN_MONTHS:
                    sufficient.append(asset)
                else:
                    print(f"    WARNING: {asset} has only {total} months "
                          f"(need {MIN_MONTHS}+), skipping training")
    elif DRY_RUN:
        print("    (dry run — no downloads)")
        sufficient.extend(need_download)  # pretend for dry-run reporting

    print(f"\n  Download complete: {len(sufficient)}/{len(assets)} assets ready for training")

    if not sufficient:
        print("  No assets with sufficient data. Exiting.")
        sys.exit(1)

    if DRY_RUN:
        print("  (dry run — no training)")
        return

    # ── Phase 2: Train ──
    print(f"\n  Phase 2: Training models for {len(sufficient)} assets x {len(TFS)} TFs...")

    work = []
    for asset in sufficient:
        sym = f"{asset}USDT"
        for tf_label, tf_resample in TFS:
            work.append((asset, sym, tf_label, tf_resample))

    n_ok, n_fail = 0, 0
    results_by_asset = {}

    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(train_one, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            done += 1
            name, tf_label, n_sigs, err = fut.result()
            results_by_asset.setdefault(name, [])

            if err:
                results_by_asset[name].append(f"{tf_label}: FAIL ({err})")
                n_fail += 1
            elif n_sigs > 0:
                results_by_asset[name].append(f"{tf_label}: OK ({n_sigs} sigs)")
                n_ok += 1
            else:
                results_by_asset[name].append(f"{tf_label}: no signals")
                n_fail += 1

            if done % 10 == 0 or done == len(work):
                print(f"    [{done}/{len(work)}] {n_ok} OK, {n_fail} fail ({time.time()-T0:.0f}s)")

    # ── Report ──
    print(f"\n  Training complete: {n_ok} models OK, {n_fail} failed")
    for asset in sorted(results_by_asset):
        tf_results = results_by_asset[asset]
        ok_count = sum(1 for r in tf_results if "OK" in r)
        print(f"    {asset}: {ok_count}/{len(TFS)} TFs trained")
        for r in tf_results:
            print(f"      {r}")

    elapsed = time.time() - T0
    print(f"\n  Total runtime: {elapsed:.0f}s")
    sys.exit(0 if n_ok > 0 else 1)


if __name__ == "__main__":
    main()
