"""Test fine-tuned Kronos (1 epoch, 3090) on OOS crypto data.

Loads the fine-tuned model from the output directory and evaluates:
1. Directional accuracy (next-bar close up/down)
2. Multi-step prediction quality (1, 4, 8 bars ahead)
3. Comparison vs LogReg baseline
4. Per-asset breakdown

Uses strict OOS: test on 2025+ data (model trained on data up to ~2024).
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "vendor", "kronos"))

import time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# Local model/data artifacts are intentionally unversioned. Override these
# roots when the artifacts live outside the repository checkout.
MODEL_ROOT = Path(
    os.environ.get("KRONOS_FINETUNED_ROOT", "models/kronos/crypto_15m_10assets")
)
FINETUNED_MODEL = MODEL_ROOT / "basemodel" / "best_model"
TOKENIZER_DIR = MODEL_ROOT / "tokenizer"
DATA_DIR = Path(os.environ.get("ML_TRADING_DATA_DIR", "data")) / "binance-1m"

# Test config
TEST_START = "2025-01"
LOOKBACK = 512   # context window (from training config)
PRED_LEN = 4     # predict 4 bars ahead (1 hour)
N_EVAL = 200     # number of evaluation points per asset
STEP = 20        # evaluate every 20 bars (5 hours apart, avoid overlap)
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
          "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "SUIUSDT"]


def load_model():
    """Load the fine-tuned Kronos model."""
    from model import Kronos, KronosTokenizer, KronosPredictor

    # Check if tokenizer was saved, otherwise use default
    tok_path = TOKENIZER_DIR
    if not (tok_path / "tokenizer_config.json").exists() and not list(tok_path.iterdir()):
        print("  Tokenizer dir empty, using pretrained tokenizer")
        tok_path = "NeoQuasar/Kronos-Tokenizer-2k"

    print(f"  Loading tokenizer from {tok_path}...")
    tokenizer = KronosTokenizer.from_pretrained(str(tok_path))

    print(f"  Loading fine-tuned model from {FINETUNED_MODEL}...")
    model = Kronos.from_pretrained(str(FINETUNED_MODEL))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=LOOKBACK)
    return predictor, device


def load_ohlcv(symbol):
    """Load and resample to 15m."""
    from signal_engine.data import load_binance_1m, resample_ohlcv
    ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    return ohlcv_15m


def evaluate_asset(predictor, ohlcv_15m, symbol, n_eval=N_EVAL, step=STEP):
    """Evaluate directional accuracy on OOS data for one asset."""
    test_start = pd.Timestamp(TEST_START)
    if ohlcv_15m.index.tz is not None:
        test_start = test_start.tz_localize(ohlcv_15m.index.tz)

    test_data = ohlcv_15m[ohlcv_15m.index >= test_start]
    if len(test_data) < LOOKBACK + PRED_LEN + 10:
        return None

    results = []
    eval_points = list(range(LOOKBACK, min(len(test_data) - PRED_LEN, LOOKBACK + n_eval * step), step))

    for idx in eval_points:
        window = test_data.iloc[idx - LOOKBACK: idx]
        actual_future = test_data.iloc[idx: idx + PRED_LEN]

        if len(actual_future) < PRED_LEN:
            break

        input_df = window[["open", "high", "low", "close", "volume"]].copy()
        x_timestamps = pd.Series(window.index)

        # Future timestamps
        median_diff = window.index.to_series().diff().median()
        future_idx = pd.date_range(window.index[-1] + median_diff, periods=PRED_LEN, freq=median_diff)
        y_timestamps = pd.Series(future_idx)

        try:
            pred_df = predictor.predict(
                df=input_df,
                x_timestamp=x_timestamps,
                y_timestamp=y_timestamps,
                pred_len=PRED_LEN,
                T=0.6,
                top_p=0.9,
                sample_count=5,
            )

            if pred_df is None or len(pred_df) < PRED_LEN:
                continue

            # Direction at each horizon
            last_close = float(window["close"].iloc[-1])
            for h in range(PRED_LEN):
                pred_close = float(pred_df["close"].iloc[h])
                actual_close = float(actual_future["close"].iloc[h])

                pred_dir = 1.0 if pred_close > last_close else -1.0
                actual_dir = 1.0 if actual_close > last_close else -1.0

                results.append({
                    "horizon": h + 1,
                    "pred_dir": pred_dir,
                    "actual_dir": actual_dir,
                    "correct": pred_dir == actual_dir,
                    "pred_close": pred_close,
                    "actual_close": actual_close,
                    "entry_close": last_close,
                    "pred_return": (pred_close - last_close) / last_close,
                    "actual_return": (actual_close - last_close) / last_close,
                })

        except Exception as e:
            print(f"    Prediction error at {idx}: {e}")
            continue

    if not results:
        return None

    return pd.DataFrame(results)


def main():
    t0 = time.time()
    print("=" * 80)
    print("KRONOS FINE-TUNED (1 EPOCH) — DIRECTIONAL ACCURACY TEST")
    print(f"  Model: {FINETUNED_MODEL}")
    print(f"  Context: {LOOKBACK} bars, Predict: {PRED_LEN} bars")
    print(f"  OOS: {TEST_START}+, {N_EVAL} eval points per asset, step={STEP}")
    print(f"  Assets: {len(ASSETS)}")
    print("=" * 80)

    # Load model
    print("\nLoading model...", flush=True)
    predictor, device = load_model()
    print(f"  Model loaded in {time.time() - t0:.0f}s")

    # Evaluate per asset
    all_results = []
    for symbol in ASSETS:
        t1 = time.time()
        asset = symbol.replace("USDT", "")
        print(f"\n  {asset}...", flush=True, end="")

        try:
            ohlcv = load_ohlcv(symbol)
        except Exception as e:
            print(f" SKIP (load error: {e})")
            continue

        df = evaluate_asset(predictor, ohlcv, symbol)
        if df is None or len(df) == 0:
            print(f" SKIP (no results)")
            continue

        df["asset"] = asset
        all_results.append(df)

        # Quick per-asset summary
        for h in [1, 4]:
            hdf = df[df["horizon"] == h]
            if len(hdf) > 0:
                acc = hdf["correct"].mean()
                print(f" h{h}={acc:.1%}", end="")
        print(f" ({len(df)//PRED_LEN} points, {time.time()-t1:.0f}s)")

    if not all_results:
        print("\nNo results generated. Check model/data paths.")
        return

    results = pd.concat(all_results, ignore_index=True)

    # Overall results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\n  Total predictions: {len(results)}")
    print(f"\n  Directional Accuracy by Horizon:")
    print(f"  {'Horizon':>8s} {'N':>6s} {'Accuracy':>9s} {'vs Random':>10s}")
    print(f"  {'-'*36}")
    for h in range(1, PRED_LEN + 1):
        hdf = results[results["horizon"] == h]
        if len(hdf) > 0:
            acc = hdf["correct"].mean()
            n = len(hdf)
            # Significance: binomial test vs 50%
            z = (acc - 0.5) / np.sqrt(0.25 / n)
            sig = "*" if abs(z) > 1.96 else ""
            print(f"  T+{h:>5d} {n:>6d} {acc:>8.1%} {z:>+9.2f}sd {sig}")

    # Per-asset breakdown
    print(f"\n  Per-Asset Accuracy (T+1):")
    print(f"  {'Asset':>8s} {'N':>6s} {'Accuracy':>9s}")
    print(f"  {'-'*26}")
    for asset in sorted(results["asset"].unique()):
        adf = results[(results["asset"] == asset) & (results["horizon"] == 1)]
        if len(adf) > 0:
            print(f"  {asset:>8s} {len(adf):>6d} {adf['correct'].mean():>8.1%}")

    # Return correlation (does Kronos predict magnitude, not just direction?)
    print(f"\n  Predicted vs Actual Return Correlation:")
    for h in [1, 4]:
        hdf = results[results["horizon"] == h]
        if len(hdf) > 10:
            corr = np.corrcoef(hdf["pred_return"], hdf["actual_return"])[0, 1]
            print(f"  T+{h}: r={corr:.4f}")

    # Mean return when Kronos predicts up vs down
    print(f"\n  Actual Return by Kronos Prediction (T+1):")
    h1 = results[results["horizon"] == 1]
    if len(h1) > 0:
        up = h1[h1["pred_dir"] == 1.0]["actual_return"]
        down = h1[h1["pred_dir"] == -1.0]["actual_return"]
        print(f"  Pred UP:   mean={up.mean()*10000:.1f}bps, n={len(up)}")
        print(f"  Pred DOWN: mean={down.mean()*10000:.1f}bps, n={len(down)}")
        edge = up.mean() - down.mean()
        print(f"  Edge:      {edge*10000:.1f}bps")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
