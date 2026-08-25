"""Rigorous cointegration search with cache-first screening and walk-forward tests.

Full-sample screening uses 15m closes over the latest 12 months, with:
- vectorized overlap / return-correlation prescreen
- Engle-Granger, KPSS, Johansen, Hurst, half-life, beta-stability confirmation

Walk-forward backtests reuse cached monthly segments so parameter sweeps do not
re-fit spreads repeatedly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

sys.path.insert(0, "src")

from signal_engine.data import load_binance_1m, resample_ohlcv

warnings.filterwarnings("ignore")

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT",
    "BNX": "BNXUSDT",
    "BNB": "BNBUSDT",
    "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT",
    "ADA": "ADAUSDT",
    "ANKR": "ANKRUSDT",
    "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT",
    "FIL": "FILUSDT",
    "DOT": "DOTUSDT",
    "SXP": "SXPUSDT",
    "ENA": "ENAUSDT",
    "BCH": "BCHUSDT",
    "LTC": "LTCUSDT",
    "UNI": "UNIUSDT",
    "WLD": "WLDUSDT",
    "TRX": "TRXUSDT",
    "FTM": "FTMUSDT",
    "AAVE": "AAVEUSDT",
    "FET": "FETUSDT",
    "APT": "APTUSDT",
    "HBAR": "HBARUSDT",
    "XLM": "XLMUSDT",
}

DATA_DIR = Path("data/binance-1m")
OUT_DIR = Path("results/cointegration_rigorous")
CACHE_DIR = OUT_DIR / "cache"
ASSET_CACHE_DIR = CACHE_DIR / "asset_15m"
SCREEN_CACHE_DIR = CACHE_DIR / "screen_15m_1y"
SEGMENT_CACHE_DIR = CACHE_DIR / "segments"

TIMEFRAME = "15min"
SCREEN_MONTHS = 12
SCREEN_MIN_BARS = 5_000
ROLLING_MIN_BARS = 96 * 30
MIN_RETURN_CORR = 0.45
MIN_LOG_CORR = 0.85
LOOKBACKS_MONTHS = [1, 2, 3, 4, 6, 8, 10, 12]
ENTRY_Z_GRID = [1.5, 2.0, 2.5]
COST_BPS_GRID = [2, 5, 10]
MAX_BACKTEST_PAIRS = 20
SCREEN_PROGRESS_EVERY = 25
TOP_SLOWEST = 10

LOGGER = logging.getLogger("cointegration_rigorous")
_SCREEN_CACHE: dict[str, object] | None = None
_FULL_CLOSE_DF: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "screen", "backtest", "report"], default="all")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--max-pairs", type=int, default=MAX_BACKTEST_PAIRS)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [OUT_DIR, CACHE_DIR, ASSET_CACHE_DIR, SCREEN_CACHE_DIR, SEGMENT_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_dirs()
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(OUT_DIR / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 2:
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def asset_cache_path(asset: str) -> Path:
    return ASSET_CACHE_DIR / f"{asset}_{TIMEFRAME}.parquet"


def screen_cache_paths() -> dict[str, Path]:
    return {
        "close": SCREEN_CACHE_DIR / "close_15m_1y.parquet",
        "log": SCREEN_CACHE_DIR / "log_close_15m_1y.parquet",
        "ret": SCREEN_CACHE_DIR / "ret_15m_1y.parquet",
        "valid_obs": SCREEN_CACHE_DIR / "valid_obs_count.npy",
        "corr_ret": SCREEN_CACHE_DIR / "corr_ret.npy",
        "corr_log": SCREEN_CACHE_DIR / "corr_log.npy",
        "meta": SCREEN_CACHE_DIR / "meta.json",
        "prescreen": OUT_DIR / "screen_prescreen.parquet",
        "progress": OUT_DIR / "screen_progress.parquet",
    }


def segment_cache_path(pair_key: str, lookback_months: int) -> Path:
    safe_pair = pair_key.replace("/", "__")
    return SEGMENT_CACHE_DIR / f"{safe_pair}_{lookback_months}mo.pkl"


def load_close_series(asset: str, symbol: str) -> pd.Series | None:
    cache_path = asset_cache_path(asset)
    if cache_path.exists():
        series = pd.read_parquet(cache_path)["close"].astype(float)
        series.index = pd.to_datetime(series.index, utc=True)
        return series.rename(asset)

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, TIMEFRAME)
    except Exception as exc:
        LOGGER.warning("asset load failed | asset=%s | err=%s", asset, exc)
        return None

    close = ohlcv_15m[["close"]].astype(float)
    close.to_parquet(cache_path)
    return close["close"].rename(asset)


def build_universe() -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for asset, symbol in ASSETS.items():
        series = load_close_series(asset, symbol)
        if series is not None and not series.empty:
            series_map[asset] = series
    if not series_map:
        raise RuntimeError("No assets loaded for cointegration experiment")
    close_df = pd.DataFrame(series_map).sort_index().astype(float)
    return close_df


def fast_ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_dev = x - x_mean
    denom = float(np.dot(x_dev, x_dev))
    if denom <= 0:
        return float("nan"), float("nan")
    beta = float(np.dot(x_dev, y - y_mean) / denom)
    intercept = y_mean - beta * x_mean
    return beta, intercept


def residual_from_fit(y: np.ndarray, x: np.ndarray, beta: float, intercept: float) -> np.ndarray:
    return y - beta * x - intercept


def half_life_days(residual: np.ndarray) -> float:
    if residual.size < 50:
        return float("inf")
    lag = residual[:-1]
    delta = residual[1:] - residual[:-1]
    denom = float(np.dot(lag - lag.mean(), lag - lag.mean()))
    if denom <= 0:
        return float("inf")
    phi = float(np.dot(lag - lag.mean(), delta - delta.mean()) / denom)
    if not np.isfinite(phi) or phi >= 0:
        return float("inf")
    return float(-math.log(2.0) / phi / 96.0)


def hurst_exponent(series: np.ndarray, max_lag: int = 64) -> float:
    if series.size < max_lag * 3:
        return float("nan")
    lags = np.arange(2, max_lag + 1)
    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        std = np.std(diff)
        if std > 0 and np.isfinite(std):
            tau.append(math.sqrt(std))
        else:
            tau.append(np.nan)
    tau_arr = np.asarray(tau, dtype=float)
    keep = np.isfinite(tau_arr) & (tau_arr > 0)
    if keep.sum() < 10:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)
    return float(slope)


def beta_stability_ratio(y: np.ndarray, x: np.ndarray, splits: int = 4) -> float:
    if y.size < splits * 250:
        return float("inf")
    betas = []
    for ys, xs in zip(np.array_split(y, splits), np.array_split(x, splits)):
        beta, _ = fast_ols(ys, xs)
        if not np.isfinite(beta) or beta <= 0:
            return float("inf")
        betas.append(abs(beta))
    return float(max(betas) / min(betas)) if min(betas) > 0 else float("inf")


def johansen_rank1(y: np.ndarray, x: np.ndarray) -> bool:
    try:
        joh = coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)
        return bool(joh.lr1[0] > joh.cvt[0, 1])
    except Exception:
        return False


def extract_pair_arrays(values: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_i = values[:, i]
    col_j = values[:, j]
    mask = np.isfinite(col_i) & np.isfinite(col_j)
    return col_i[mask], col_j[mask], mask


def pair_runtime_stats(runtimes: list[float]) -> tuple[float, float]:
    if not runtimes:
        return 0.0, 0.0
    arr = np.asarray(runtimes, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 95))


def build_screen_cache(close_df: pd.DataFrame) -> dict[str, object]:
    t0 = time.time()
    cutoff = close_df.index.max() - pd.DateOffset(months=SCREEN_MONTHS)
    screen_close = close_df.loc[close_df.index >= cutoff].copy()
    screen_close = screen_close.astype(float)
    screen_log = np.log(screen_close)
    screen_ret = screen_close.pct_change(fill_method=None)

    close_values = screen_close.to_numpy(dtype=np.float64, copy=True)
    valid_mask = np.isfinite(close_values)
    valid_obs = valid_mask.astype(np.int32).T @ valid_mask.astype(np.int32)
    corr_ret = screen_ret.corr(min_periods=SCREEN_MIN_BARS).to_numpy(dtype=np.float64)
    corr_log = screen_log.corr(min_periods=SCREEN_MIN_BARS).to_numpy(dtype=np.float64)

    paths = screen_cache_paths()
    screen_close.to_parquet(paths["close"])
    screen_log.to_parquet(paths["log"])
    screen_ret.to_parquet(paths["ret"])
    np.save(paths["valid_obs"], valid_obs)
    np.save(paths["corr_ret"], corr_ret)
    np.save(paths["corr_log"], corr_log)
    paths["meta"].write_text(
        json.dumps(
            {
                "assets": list(screen_close.columns),
                "rows": int(len(screen_close)),
                "last_ts": screen_close.index.max().isoformat(),
                "months": SCREEN_MONTHS,
                "timeframe": TIMEFRAME,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info("[2b] build overlap/correlation matrices | rows=%s | assets=%s | sec=%.1f",
                len(screen_close), len(screen_close.columns), time.time() - t0)
    return {
        "assets": list(screen_close.columns),
        "index": screen_close.index,
        "close_df": screen_close,
        "log_values": screen_log.to_numpy(dtype=np.float64, copy=True),
        "ret_corr": corr_ret,
        "log_corr": corr_log,
        "valid_obs": valid_obs,
    }


def load_screen_cache_from_disk() -> dict[str, object] | None:
    paths = screen_cache_paths()
    required = [
        paths["close"],
        paths["log"],
        paths["ret"],
        paths["valid_obs"],
        paths["corr_ret"],
        paths["corr_log"],
        paths["meta"],
    ]
    if not all(path.exists() for path in required):
        return None

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if meta.get("months") != SCREEN_MONTHS or meta.get("timeframe") != TIMEFRAME:
        return None

    close_df = pd.read_parquet(paths["close"]).astype(float)
    close_df.index = pd.to_datetime(close_df.index, utc=True)
    log_df = pd.read_parquet(paths["log"]).astype(float)
    log_df.index = pd.to_datetime(log_df.index, utc=True)

    return {
        "assets": list(close_df.columns),
        "index": close_df.index,
        "close_df": close_df,
        "log_values": log_df.to_numpy(dtype=np.float64, copy=True),
        "ret_corr": np.load(paths["corr_ret"]),
        "log_corr": np.load(paths["corr_log"]),
        "valid_obs": np.load(paths["valid_obs"]),
    }


def get_or_build_screen_cache(close_df: pd.DataFrame) -> tuple[dict[str, object], bool]:
    global _SCREEN_CACHE
    cached = load_screen_cache_from_disk()
    if cached is not None:
        _SCREEN_CACHE = cached
        LOGGER.info("[2a] load screen caches | source=warm-cache | rows=%s | assets=%s",
                    len(cached["index"]), len(cached["assets"]))
        LOGGER.info("[2b] build overlap/correlation matrices | source=cache")
        return cached, True

    LOGGER.info("[2a] load screen caches | source=rebuild")
    built = build_screen_cache(close_df)
    _SCREEN_CACHE = built
    return built, False


def build_vectorized_prescreen(screen_cache: dict[str, object]) -> pd.DataFrame:
    t0 = time.time()
    assets = list(screen_cache["assets"])
    valid_obs = np.asarray(screen_cache["valid_obs"])
    corr_ret = np.asarray(screen_cache["ret_corr"])
    corr_log = np.asarray(screen_cache["log_corr"])
    rows = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            n_obs = int(valid_obs[i, j])
            corr = float(corr_ret[i, j]) if np.isfinite(corr_ret[i, j]) else float("nan")
            long_corr = float(corr_log[i, j]) if np.isfinite(corr_log[i, j]) else float("nan")
            prescreen_fail = (
                n_obs < SCREEN_MIN_BARS
                or not np.isfinite(corr)
                or corr < MIN_RETURN_CORR
                or not np.isfinite(long_corr)
                or long_corr < MIN_LOG_CORR
            )
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "pair": f"{assets[i]}/{assets[j]}",
                    "asset_i": assets[i],
                    "asset_j": assets[j],
                    "n_obs": n_obs,
                    "corr_ret": corr,
                    "log_corr": long_corr,
                    "prescreen_fail": prescreen_fail,
                }
            )
    prescreen = pd.DataFrame(rows).sort_values(["prescreen_fail", "corr_ret"], ascending=[True, False])
    keep = int((~prescreen["prescreen_fail"]).sum())
    screen_cache_paths()["prescreen"].unlink(missing_ok=True)
    prescreen.to_parquet(screen_cache_paths()["prescreen"], index=False)
    LOGGER.info("[2c] vectorized prescreen | total_pairs=%s | survivors=%s | sec=%.1f",
                len(prescreen), keep, time.time() - t0)
    return prescreen


def fit_orientation(y: np.ndarray, x: np.ndarray, y_name: str, x_name: str) -> dict[str, object]:
    beta, intercept = fast_ols(y, x)
    if not np.isfinite(beta) or not np.isfinite(intercept) or beta <= 0:
        return {
            "y_name": y_name,
            "x_name": x_name,
            "beta": float("nan"),
            "intercept": float("nan"),
            "eg_pvalue": 1.0,
        }

    try:
        _, eg_pvalue, _ = coint(y, x, trend="c", autolag="aic")
    except Exception:
        eg_pvalue = 1.0

    return {
        "y_name": y_name,
        "x_name": x_name,
        "beta": float(beta),
        "intercept": float(intercept),
        "eg_pvalue": float(eg_pvalue) if np.isfinite(eg_pvalue) else 1.0,
    }


def confirm_pair_task(row: dict[str, object]) -> dict[str, object]:
    assert _SCREEN_CACHE is not None
    t0 = time.time()
    i = int(row["i"])
    j = int(row["j"])
    assets = list(_SCREEN_CACHE["assets"])
    log_values = np.asarray(_SCREEN_CACHE["log_values"])

    y_ij, x_ij, mask = extract_pair_arrays(log_values, i, j)
    n_obs = int(mask.sum())
    base = {
        "pair": row["pair"],
        "asset_i": row["asset_i"],
        "asset_j": row["asset_j"],
        "corr_ret": float(row["corr_ret"]),
        "log_corr": float(row["log_corr"]),
        "n_obs": n_obs,
        "prescreen_fail": bool(row["prescreen_fail"]),
    }
    if n_obs < SCREEN_MIN_BARS:
        return {
            **base,
            "y_name": assets[i],
            "x_name": assets[j],
            "eg_pvalue": 1.0,
            "kpss_pvalue": 0.0,
            "johansen_rank1": False,
            "beta": float("nan"),
            "intercept": float("nan"),
            "half_life_days": float("inf"),
            "hurst": float("nan"),
            "beta_ratio": float("inf"),
            "passes_basic": False,
            "runtime_sec": time.time() - t0,
        }

    fit_ij = fit_orientation(y_ij, x_ij, assets[i], assets[j])
    fit_ji = fit_orientation(x_ij, y_ij, assets[j], assets[i])
    fit = fit_ij if fit_ij["eg_pvalue"] <= fit_ji["eg_pvalue"] else fit_ji

    y = y_ij if fit["y_name"] == assets[i] else x_ij
    x = x_ij if fit["x_name"] == assets[j] else y_ij
    if float(fit["eg_pvalue"]) >= 0.10:
        return {
            **base,
            "y_name": fit["y_name"],
            "x_name": fit["x_name"],
            "eg_pvalue": float(fit["eg_pvalue"]),
            "kpss_pvalue": 0.0,
            "johansen_rank1": False,
            "beta": float(fit["beta"]),
            "intercept": float(fit["intercept"]),
            "half_life_days": float("inf"),
            "hurst": float("nan"),
            "beta_ratio": float("inf"),
            "passes_basic": False,
            "runtime_sec": time.time() - t0,
        }

    residual = residual_from_fit(y, x, float(fit["beta"]), float(fit["intercept"]))

    try:
        _, kpss_pvalue, _, _ = kpss(residual, regression="c", nlags="auto")
    except Exception:
        kpss_pvalue = 0.0

    half_life = half_life_days(residual)
    hurst = hurst_exponent(residual)
    beta_ratio = beta_stability_ratio(y, x)
    rank1 = johansen_rank1(y, x)
    passes_basic = bool(
        fit["eg_pvalue"] < 0.05
        and kpss_pvalue > 0.05
        and rank1
        and np.isfinite(half_life)
        and 0.25 <= half_life <= 20.0
        and np.isfinite(hurst)
        and hurst < 0.5
        and np.isfinite(beta_ratio)
        and beta_ratio <= 1.5
    )

    return {
        **base,
        "y_name": fit["y_name"],
        "x_name": fit["x_name"],
        "eg_pvalue": float(fit["eg_pvalue"]),
        "kpss_pvalue": float(kpss_pvalue),
        "johansen_rank1": rank1,
        "beta": float(fit["beta"]),
        "intercept": float(fit["intercept"]),
        "half_life_days": float(half_life),
        "hurst": float(hurst),
        "beta_ratio": float(beta_ratio),
        "passes_basic": passes_basic,
        "runtime_sec": time.time() - t0,
    }


def log_slowest_pairs(screen_df: pd.DataFrame, top_n: int = TOP_SLOWEST) -> None:
    if screen_df.empty:
        return
    slowest = screen_df.nlargest(min(top_n, len(screen_df)), "runtime_sec")
    for _, row in slowest.iterrows():
        LOGGER.info(
            "[2d] slowest pair | pair=%s | sec=%.2f | eg=%.4f | corr=%.3f",
            row["pair"],
            row["runtime_sec"],
            row["eg_pvalue"],
            row["log_corr"],
        )


def full_sample_screen(close_df: pd.DataFrame, workers: int, max_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    screen_cache, _ = get_or_build_screen_cache(close_df)
    prescreen = build_vectorized_prescreen(screen_cache)
    candidates = prescreen.loc[~prescreen["prescreen_fail"]].to_dict("records")

    LOGGER.info("[2d] expensive statistical confirmation | candidates=%s | workers=%s", len(candidates), workers)
    t0 = time.time()
    records: list[dict[str, object]] = []
    runtimes: list[float] = []
    done = 0
    progress_path = screen_cache_paths()["progress"]
    csv_progress = OUT_DIR / "screen_progress.csv"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(confirm_pair_task, row) for row in candidates]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            runtimes.append(float(record["runtime_sec"]))
            done += 1
            if done == 1 or done % SCREEN_PROGRESS_EVERY == 0 or done == len(candidates):
                progress = pd.DataFrame(records).sort_values("eg_pvalue")
                progress.to_parquet(progress_path, index=False)
                progress.to_csv(csv_progress, index=False)
                mean_rt, p95_rt = pair_runtime_stats(runtimes)
                elapsed = time.time() - t0
                throughput = done / elapsed if elapsed > 0 else 0.0
                LOGGER.info(
                    "[2d] confirmation progress | done=%s/%s | sec=%.1f | pairs_per_sec=%.2f | avg_rt=%.2f | p95_rt=%.2f",
                    done,
                    len(candidates),
                    elapsed,
                    throughput,
                    mean_rt,
                    p95_rt,
                )

    screen_df = pd.DataFrame(records).sort_values(["passes_basic", "eg_pvalue"], ascending=[False, True])
    if screen_df.empty:
        screen_df = pd.DataFrame(
            columns=[
                "pair",
                "y_name",
                "x_name",
                "eg_pvalue",
                "kpss_pvalue",
                "johansen_rank1",
                "beta",
                "intercept",
                "half_life_days",
                "hurst",
                "beta_ratio",
                "corr_ret",
                "log_corr",
                "n_obs",
                "passes_basic",
                "prescreen_fail",
                "runtime_sec",
            ]
        )

    if not screen_df.empty:
        reject, qvals, _, _ = multipletests(screen_df["eg_pvalue"].fillna(1.0), alpha=0.05, method="fdr_bh")
        screen_df["fdr_reject"] = reject
        screen_df["eg_qvalue"] = qvals
    else:
        screen_df["fdr_reject"] = []
        screen_df["eg_qvalue"] = []
    screen_df.to_parquet(OUT_DIR / "screen.parquet", index=False)
    screen_df.to_csv(OUT_DIR / "screen.csv", index=False)

    shortlist = screen_df.loc[screen_df["passes_basic"] & screen_df["fdr_reject"]].copy()
    if shortlist.empty:
        shortlist = screen_df.loc[screen_df["passes_basic"]].nsmallest(max_pairs, "eg_pvalue").copy()

    shortlist = shortlist.head(max_pairs).copy()
    shortlist.to_csv(OUT_DIR / "shortlist.csv", index=False)
    log_slowest_pairs(screen_df)
    LOGGER.info("[2e] FDR + shortlist write | tested=%s | basic_pass=%s | fdr_survivors=%s | shortlist=%s | sec=%.1f",
                len(screen_df),
                int(screen_df["passes_basic"].sum()) if not screen_df.empty else 0,
                int((screen_df["passes_basic"] & screen_df["fdr_reject"]).sum()) if not screen_df.empty else 0,
                len(shortlist),
                time.time() - t0)
    return screen_df, shortlist


def load_pair_history(y_name: str, x_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    assert _FULL_CLOSE_DF is not None
    pair_df = _FULL_CLOSE_DF[[y_name, x_name]].dropna()
    if len(pair_df) < ROLLING_MIN_BARS + 500:
        return None
    timestamps = pair_df.index.to_numpy()
    y = pair_df[y_name].to_numpy(dtype=np.float64, copy=True)
    x = pair_df[x_name].to_numpy(dtype=np.float64, copy=True)
    return timestamps, y, x


def prepare_segments(row: dict[str, object], lookback_months: int) -> list[dict[str, object]]:
    pair_key = str(row["pair"])
    cache_path = segment_cache_path(pair_key, lookback_months)
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    pair_history = load_pair_history(str(row["y_name"]), str(row["x_name"]))
    if pair_history is None:
        return []
    timestamps, y_price, x_price = pair_history
    log_y = np.log(y_price)
    log_x = np.log(x_price)
    y_ret = np.zeros_like(y_price)
    x_ret = np.zeros_like(x_price)
    y_ret[1:] = y_price[1:] / y_price[:-1] - 1.0
    x_ret[1:] = x_price[1:] / x_price[:-1] - 1.0

    ts_index = pd.DatetimeIndex(timestamps)
    months = ts_index.to_period("M")
    unique_months = months.unique().sort_values()
    segments: list[dict[str, object]] = []
    for month in unique_months:
        test_mask = months == month
        if not np.any(test_mask):
            continue
        test_start = pd.Timestamp(month.start_time, tz="UTC")
        train_start = test_start - pd.DateOffset(months=lookback_months)
        train_mask = (ts_index >= train_start) & (ts_index < test_start)
        if int(train_mask.sum()) < max(ROLLING_MIN_BARS, lookback_months * 96 * 20):
            continue

        train_y = log_y[train_mask]
        train_x = log_x[train_mask]
        beta, intercept = fast_ols(train_y, train_x)
        if not np.isfinite(beta) or beta <= 0:
            continue
        train_resid = residual_from_fit(train_y, train_x, beta, intercept)
        sigma = float(np.std(train_resid))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            continue
        mu = float(np.mean(train_resid))

        seg_y = log_y[test_mask]
        seg_x = log_x[test_mask]
        zscore = (residual_from_fit(seg_y, seg_x, beta, intercept) - mu) / sigma
        hedge_ret = y_ret[test_mask] - beta * x_ret[test_mask]
        if zscore.size < 20:
            continue
        segments.append(
            {
                "month": str(month),
                "beta": float(beta),
                "zscore": zscore.astype(np.float64, copy=False),
                "hedge_ret": hedge_ret.astype(np.float64, copy=False),
            }
        )

    with cache_path.open("wb") as fh:
        pickle.dump(segments, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return segments


def simulate_segments(segments: list[dict[str, object]], entry_z: float, cost_bps: float) -> dict[str, object] | None:
    if not segments:
        return None
    monthly_pnl = []
    total_trades = 0
    for seg in segments:
        zscore = np.asarray(seg["zscore"], dtype=np.float64)
        hedge_ret = np.asarray(seg["hedge_ret"], dtype=np.float64)
        cost = cost_bps / 10_000.0 * (1.0 + abs(float(seg["beta"])))
        pnl = 0.0
        position = 0
        for i in range(1, len(zscore)):
            if position != 0:
                pnl += position * hedge_ret[i]
            z = zscore[i]
            if position == 0:
                if z <= -entry_z:
                    position = 1
                    pnl -= cost
                    total_trades += 1
                elif z >= entry_z:
                    position = -1
                    pnl -= cost
                    total_trades += 1
            elif position == 1 and z >= 0.0:
                position = 0
                pnl -= cost
            elif position == -1 and z <= 0.0:
                position = 0
                pnl -= cost
        if position != 0:
            pnl -= cost
        monthly_pnl.append(pnl)

    pnl_arr = np.asarray(monthly_pnl, dtype=np.float64)
    if pnl_arr.size < 3:
        return None
    mean_ret = float(np.mean(pnl_arr))
    std_ret = float(np.std(pnl_arr, ddof=1)) if pnl_arr.size > 1 else 0.0
    sharpe = float(np.sqrt(12.0) * mean_ret / std_ret) if std_ret > 0 else 0.0
    equity = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(equity)
    drawdown = float(np.max(peak - equity)) if equity.size else 0.0
    return {
        "sharpe": sharpe,
        "total_return_pct": float(100.0 * pnl_arr.sum()),
        "mean_monthly_ret_pct": float(100.0 * mean_ret),
        "positive_months_pct": float(100.0 * np.mean(pnl_arr > 0)),
        "max_drawdown_pct": float(100.0 * drawdown),
        "n_months": int(pnl_arr.size),
        "n_trades": int(total_trades),
    }


def backtest_pair_task(row: dict[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for lookback_months in LOOKBACKS_MONTHS:
        segments = prepare_segments(row, lookback_months)
        if not segments:
            continue
        for entry_z in ENTRY_Z_GRID:
            for cost_bps in COST_BPS_GRID:
                metrics = simulate_segments(segments, entry_z=entry_z, cost_bps=cost_bps)
                if metrics is None:
                    continue
                results.append(
                    {
                        "pair": row["pair"],
                        "y_name": row["y_name"],
                        "x_name": row["x_name"],
                        "lookback_months": lookback_months,
                        "entry_z": entry_z,
                        "cost_bps": cost_bps,
                        "eg_pvalue": row["eg_pvalue"],
                        "eg_qvalue": row.get("eg_qvalue", np.nan),
                        "kpss_pvalue": row["kpss_pvalue"],
                        "johansen_rank1": row["johansen_rank1"],
                        "half_life_days": row["half_life_days"],
                        "hurst": row["hurst"],
                        "beta_ratio": row["beta_ratio"],
                        **metrics,
                    }
                )
    return results


def robust_backtests(shortlist: pd.DataFrame, workers: int) -> pd.DataFrame:
    if shortlist.empty:
        empty = pd.DataFrame()
        empty.to_csv(OUT_DIR / "backtests.csv", index=False)
        empty.to_csv(OUT_DIR / "summary.csv", index=False)
        empty.to_csv(OUT_DIR / "robust_pairs.csv", index=False)
        return empty

    LOGGER.info("[3] walk-forward backtests | pairs=%s | workers=%s", len(shortlist), workers)
    t0 = time.time()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(backtest_pair_task, rec) for rec in shortlist.to_dict("records")]
        done = 0
        for future in as_completed(futures):
            rows.extend(future.result())
            done += 1
            if done == 1 or done % 5 == 0 or done == len(futures):
                partial = pd.DataFrame(rows)
                partial.to_csv(OUT_DIR / "backtests.csv", index=False)
                LOGGER.info("[3] backtest progress | done=%s/%s | rows=%s | sec=%.1f",
                            done, len(futures), len(partial), time.time() - t0)

    backtests = pd.DataFrame(rows)
    if backtests.empty:
        backtests.to_csv(OUT_DIR / "backtests.csv", index=False)
        return backtests

    summary = (
        backtests.sort_values("sharpe", ascending=False)
        .groupby("pair", as_index=False)
        .first()
        .sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
    )
    robust_pairs = summary.loc[
        (summary["sharpe"] > 0.5)
        & (summary["positive_months_pct"] >= 50.0)
        & (summary["n_months"] >= 6)
    ].copy()

    backtests.to_csv(OUT_DIR / "backtests.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    robust_pairs.to_csv(OUT_DIR / "robust_pairs.csv", index=False)
    LOGGER.info("[3] backtests complete | rows=%s | unique_pairs=%s | robust=%s | sec=%.1f",
                len(backtests), summary["pair"].nunique(), len(robust_pairs), time.time() - t0)
    return backtests


def write_report(screen_df: pd.DataFrame, shortlist: pd.DataFrame, backtests: pd.DataFrame) -> None:
    lines = [
        "# Cointegration Pairs Report",
        "",
        f"- Screening timeframe: `{TIMEFRAME}` over latest `{SCREEN_MONTHS}` months",
        f"- Prescreen: overlap >= `{SCREEN_MIN_BARS}` and return correlation >= `{MIN_RETURN_CORR}`",
        f"- Confirmed pairs tested: `{len(screen_df)}`",
        f"- Basic statistical survivors: `{int(screen_df['passes_basic'].sum()) if not screen_df.empty else 0}`",
        f"- FDR survivors: `{int((screen_df['passes_basic'] & screen_df['fdr_reject']).sum()) if not screen_df.empty else 0}`",
        f"- Backtest shortlist: `{len(shortlist)}`",
        "",
    ]

    if not shortlist.empty:
        lines.extend(["## Shortlist", ""])
        for _, row in shortlist.head(10).iterrows():
            lines.append(
                f"- `{row['pair']}` | EG p={row['eg_pvalue']:.4g} | q={row.get('eg_qvalue', np.nan):.4g} | "
                f"HL={row['half_life_days']:.2f}d | H={row['hurst']:.3f} | corr={row['corr_ret']:.3f}"
            )
        lines.append("")

    if not backtests.empty:
        summary = (
            backtests.sort_values("sharpe", ascending=False)
            .groupby("pair", as_index=False)
            .first()
            .sort_values("sharpe", ascending=False)
        )
        lines.extend(["## Best Walk-Forward Configs", ""])
        for _, row in summary.head(10).iterrows():
            lines.append(
                f"- `{row['pair']}` | Sharpe={row['sharpe']:.2f} | total={row['total_return_pct']:.2f}% | "
                f"lookback={int(row['lookback_months'])}mo | entry_z={row['entry_z']:.1f} | cost={int(row['cost_bps'])}bps"
            )

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global _FULL_CLOSE_DF
    args = parse_args()
    setup_logging()
    LOGGER.info("run started | workers=%s | stage=%s", args.workers, args.stage)

    t0 = time.time()
    _FULL_CLOSE_DF = build_universe()
    LOGGER.info("[1] universe loaded | assets=%s | rows=%s | sec=%.1f",
                len(_FULL_CLOSE_DF.columns), len(_FULL_CLOSE_DF), time.time() - t0)

    screen_df = pd.DataFrame()
    shortlist = pd.DataFrame()
    backtests = pd.DataFrame()

    if args.stage in {"all", "screen"}:
        screen_df, shortlist = full_sample_screen(_FULL_CLOSE_DF, workers=args.workers, max_pairs=args.max_pairs)
    elif args.stage in {"backtest", "report"}:
        screen_df = safe_read_parquet(OUT_DIR / "screen.parquet")
        shortlist = safe_read_csv(OUT_DIR / "shortlist.csv")

    if args.stage in {"all", "backtest"}:
        if shortlist.empty:
            shortlist = safe_read_csv(OUT_DIR / "shortlist.csv")
        backtests = robust_backtests(shortlist, workers=args.workers)
    elif args.stage == "report":
        backtests = safe_read_csv(OUT_DIR / "backtests.csv")

    if args.stage in {"all", "report"}:
        if screen_df.empty:
            screen_df = safe_read_parquet(OUT_DIR / "screen.parquet")
        if shortlist.empty:
            shortlist = safe_read_csv(OUT_DIR / "shortlist.csv")
        if backtests.empty:
            backtests = safe_read_csv(OUT_DIR / "backtests.csv")
        write_report(screen_df, shortlist, backtests)


if __name__ == "__main__":
    main()
