"""Reusable ML early-close walk-forward research pipeline.

This module contains the machinery used by the invalidated historical
close-model study. Its holdout gate is preserved as a research diagnostic and
is not sufficient for promotion under ``docs/VALIDATION.md``.
"""

import copy
import hashlib
import json
import os
import time
import warnings
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from afml.evaluation import deflated_sharpe_ratio
from afml.bet_sizing import discretize_signal, get_signal
from afml.cross_val import PurgedKFold
from afml.importance import (
    mean_decrease_accuracy,
    mean_decrease_impurity,
    single_feature_importance,
)
from signal_engine.allocation import compute_kelly_weights, compute_nco_weights
from signal_engine.close_model import (
    build_close_decision_examples,
    close_feature_columns,
    compute_afml_close_sample_weights,
    normalize_timeframe,
    select_close_predictions,
    walk_forward_close_prediction_cache,
    walk_forward_close_predictions,
)
from signal_engine.config import (
    CloseModelConfig,
    ConfidenceFilterConfig,
    FeatureConfig,
    PortfolioConfig,
    load_config,
)
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import compute_daily_pnl_series, simulate_portfolio
from signal_engine.portfolio_path import simulate_portfolio_precomputed_exits
from signal_engine.stats import compute_full_stats
from signal_engine.weights import compute_sample_weights

import sklearn

warnings.filterwarnings("ignore")

NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")

# Process-local config cache — avoids re-parsing YAML in every subprocess call
_config_cache = None

def _get_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config(Path("configs/default.yaml"))
    return _config_cache
CACHE_DIR = Path("experiments/cache/ml_close_walkforward")
LOOKBACKS = [2, 4, 6, 8, 10, 12]
CLOSE_TIMEFRAMES = ["5min", "10min", "15min"]
SIGNAL_BOOTSTRAP_START = "2021-01"
TEST_START = "2022-01"
SEARCH_START = "2022-01"
SEARCH_END = "2024-12"
HOLDOUT_START = "2025-01"
EVAL_REGIME_TAG = "search2022-2024_holdout2025plus"
CACHE_VERSION = "v10"

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


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(
            C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42
        )
    return LogisticRegression(
        C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42
    )


def make_signal_xgb(
    seed: int = 42,
    early_stopping_rounds: int | None = None,
    params: dict[str, float | int] | None = None,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        early_stopping_rounds=early_stopping_rounds,
    )
    if params:
        model.set_params(**params)
    return model


def timeframe_token(timeframe: str) -> str:
    return normalize_timeframe(timeframe).replace("min", "m")


def feature_tag(cfg) -> str:
    regime_token = "-".join(cfg.close_model.regime_timeframes)
    return (
        f"{CACHE_VERSION}_{EVAL_REGIME_TAG}_{cfg.model.signal_model_type}_"
        f"{len(cfg.features.feature_list)}f_{cfg.model.fwd_bars}b_{regime_token}"
    )


def strategy_tag(model_type: str, label_mode: str) -> str:
    model_token = "xgb" if model_type == "xgboost" else "logreg"
    label_token = "residtb" if label_mode == "residual_triple_barrier" else "bestrem"
    return f"{model_token}_{label_token}"


def strategy_variant_tag(model_type: str, label_mode: str, de_risk_mode: str) -> str:
    mode_token = "partial" if de_risk_mode == "partial_ladder" else "binary"
    return f"{strategy_tag(model_type, label_mode)}_{mode_token}"


def example_spec_tag(label_mode: str) -> str:
    return "residtb" if label_mode == "residual_triple_barrier" else "bestrem"


def canonical_prediction_de_risk_mode(
    model_type: str,
    label_mode: str,
    de_risk_mode: str,
) -> str:
    if model_type == "xgboost" and label_mode == "residual_triple_barrier":
        return "partial_ladder"
    return de_risk_mode


def asset_scope_tag(assets: dict[str, str]) -> str:
    joined = ",".join(sorted(assets))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:8]
    return f"a{len(assets)}_{digest}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def write_json(data: dict, path: Path) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def per_worker_thread_budget() -> int:
    override = os.getenv("ML_CLOSE_THREADS_PER_ASSET")
    if override:
        return max(1, int(override))
    cpu = max(int(os.cpu_count() or 1), 1)
    outer = int(os.getenv("ML_CLOSE_WORKERS", min(W, 8)))
    selected_assets = [
        item.strip()
        for item in os.getenv("ML_CLOSE_ASSETS", "").split(",")
        if item.strip()
    ]
    active_outer = min(outer, len(selected_assets)) if selected_assets else outer
    active_outer = max(1, min(active_outer, cpu))
    return max(1, cpu // active_outer)


def timeframe_sort_key(tf: str) -> int:
    norm = normalize_timeframe(tf)
    if norm.endswith("min"):
        return int(norm[:-3])
    if norm.endswith("h"):
        return int(norm[:-1]) * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def signal_parallelism_plan(signal_model_type: str) -> tuple[int, int]:
    task_override = os.getenv("ML_CLOSE_SIGNAL_TASK_WORKERS")
    model_override = os.getenv("ML_CLOSE_SIGNAL_MODEL_THREADS")
    total_threads = max(1, per_worker_thread_budget())

    if task_override:
        task_workers = max(1, min(len(LOOKBACKS), int(task_override)))
    elif signal_model_type.endswith("xgboost"):
        task_workers = max(1, min(len(LOOKBACKS), int(round(total_threads ** 0.5))))
    else:
        task_workers = max(1, min(len(LOOKBACKS), total_threads))

    if model_override:
        model_threads = max(1, int(model_override))
    elif signal_model_type.endswith("xgboost"):
        model_threads = max(1, total_threads // task_workers)
    else:
        model_threads = 1

    return task_workers, model_threads


def latest_symbol_data_tag(symbol: str, data_dirs: list[Path]) -> str | None:
    latest_tag = None
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        symbol_dir = data_dir / symbol
        pattern = f"{symbol}-1m-*.csv"
        found = sorted(symbol_dir.glob(pattern))
        if not found:
            found = sorted(data_dir.glob(pattern))
        for fp in found:
            stem = fp.stem
            parts = stem.split("-")
            if len(parts) >= 4:
                tag = f"{parts[-2]}-{parts[-1]}"
                if latest_tag is None or tag > latest_tag:
                    latest_tag = tag
    return latest_tag


def prepare_cache_requirements(
    asset: str,
    data_tag: str,
    feat_sig: str,
    cfg,
) -> tuple[set[str], dict[int, Path]]:
    cache_tfs = set(cfg.close_model.research_decision_timeframes)
    cache_tfs.add(cfg.close_model.reference_decision_timeframe)
    cache_tfs.update(["15min", *cfg.close_model.regime_timeframes])
    signal_paths = {
        lb: signal_cache_path(asset, lb, data_tag, feat_sig) for lb in LOOKBACKS
    }
    return cache_tfs, signal_paths


def feature_config_for_timeframe(
    feature_cfg: FeatureConfig,
    timeframe: str,
    regime_timeframes: list[str],
) -> FeatureConfig:
    payload = feature_cfg.model_dump()
    feature_list = list(payload["feature_list"])
    coarsest_regime_tf = (
        max(regime_timeframes, key=timeframe_sort_key) if regime_timeframes else None
    )
    if timeframe == coarsest_regime_tf and "sadf" not in feature_list:
        feature_list.append("sadf")
    payload["feature_list"] = feature_list
    return FeatureConfig(**payload)


def ohlcv_cache_path(asset: str, tf: str, tag: str) -> Path:
    return CACHE_DIR / "ohlcv" / asset / f"{timeframe_token(tf)}_{tag}.parquet"


def raw_ohlcv_cache_path(asset: str, tag: str) -> Path:
    return CACHE_DIR / "ohlcv_raw_1m" / asset / f"raw_1m_{tag}.parquet"


def feature_cache_path(asset: str, tf: str, tag: str, feat_tag: str) -> Path:
    return CACHE_DIR / "features" / asset / f"{timeframe_token(tf)}_{feat_tag}_{tag}.parquet"


def signal_cache_path(asset: str, lookback: int, tag: str, feat_tag: str) -> Path:
    return CACHE_DIR / "signals" / asset / f"lb{lookback}_{feat_tag}_{tag}.parquet"


def adjusted_signal_cache_path(
    asset: str,
    lookback: int,
    tag: str,
    feat_tag: str,
    scope_tag: str,
) -> Path:
    return (
        CACHE_DIR
        / "signals_adjusted"
        / scope_tag
        / asset
        / f"lb{lookback}_{feat_tag}_{tag}.parquet"
    )


def combined_cache_path(lookback: int, tag: str, feat_tag: str, scope_tag: str) -> Path:
    return CACHE_DIR / "combined" / scope_tag / f"lb{lookback}_{feat_tag}_{tag}.parquet"


def close_example_cache_path(
    asset: str,
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    label_mode: str,
) -> Path:
    return (
        CACHE_DIR
        / "close_examples"
        / scope_tag
        / asset
        / f"lb{lookback}_{example_spec_tag(label_mode)}_{timeframe_token(tf)}_{feat_tag}_{tag}.parquet"
    )


def result_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    strategy: str,
) -> Path:
    return (
        CACHE_DIR
        / "results"
        / scope_tag
        / f"lb{lookback}_{strategy}_{timeframe_token(tf)}_{feat_tag}_{tag}.json"
    )


def combined_close_example_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    label_mode: str,
) -> Path:
    return (
        CACHE_DIR
        / "close_examples_combined"
        / scope_tag
        / f"lb{lookback}_{example_spec_tag(label_mode)}_{timeframe_token(tf)}_{feat_tag}_{tag}.parquet"
    )


def prediction_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    model_type: str,
    label_mode: str,
) -> Path:
    return (
        CACHE_DIR
        / "predictions"
        / scope_tag
        / f"lb{lookback}_{strategy_tag(model_type, label_mode)}_{timeframe_token(tf)}_{feat_tag}_{tag}.parquet"
    )


def prediction_validation_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    model_type: str,
    label_mode: str,
) -> Path:
    return (
        CACHE_DIR
        / "predictions_validation"
        / scope_tag
        / f"lb{lookback}_{strategy_tag(model_type, label_mode)}_{timeframe_token(tf)}_{feat_tag}_{tag}.parquet"
    )


def prediction_diagnostics_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    model_type: str,
    label_mode: str,
) -> Path:
    return (
        CACHE_DIR
        / "predictions_diagnostics"
        / scope_tag
        / f"lb{lookback}_{strategy_tag(model_type, label_mode)}_{timeframe_token(tf)}_{feat_tag}_{tag}.json"
    )


def selected_trade_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    strategy: str,
) -> Path:
    return (
        CACHE_DIR
        / "selected_trades"
        / scope_tag
        / f"lb{lookback}_{strategy}_{timeframe_token(tf)}_{feat_tag}_{tag}.parquet"
    )


def selected_trade_diagnostics_cache_path(
    lookback: int,
    tf: str,
    tag: str,
    feat_tag: str,
    scope_tag: str,
    strategy: str,
) -> Path:
    return (
        CACHE_DIR
        / "selected_trade_diagnostics"
        / scope_tag
        / f"lb{lookback}_{strategy}_{timeframe_token(tf)}_{feat_tag}_{tag}.json"
    )


def manifest_cache_path(asset: str, tag: str, feat_tag: str) -> Path:
    return CACHE_DIR / "manifests" / f"{asset}_{feat_tag}_{tag}.json"


def report_json_path(scope_tag: str, data_tag: str) -> Path:
    run_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    return (
        Path("experiments")
        / f"ml_close_walkforward_report_metrics_{run_date}_{scope_tag}_{data_tag}.json"
    )


def load_or_build_frame(path: Path, builder):
    if path.exists():
        return pd.read_parquet(path)
    df = builder()
    write_parquet(df, path)
    return df


def close_example_required_columns(feature_cols: list[str]) -> list[str]:
    cols = [
        "trade_id",
        "asset",
        "timestamp",
        "entry_month",
        "direction",
        "entry_direction",
        "entry_probability",
        "entry_confidence",
        "entry_signal_strength",
        "regime_exposure_mult",
        "decision_timestamp",
        "decision_month",
        "expiry_timestamp",
        "label_end_timestamp",
        "bars_since_entry",
        "minutes_since_entry",
        "current_unrealized_return",
        "max_favorable_return_since_entry",
        "max_adverse_return_since_entry",
        "close_label",
        "exit_return",
        "gross_exit_return",
        "expiry_return",
        "label_return_abs",
    ]
    cols.extend(close_feature_columns(feature_cols))
    return list(dict.fromkeys(cols))


def read_close_example_frame(path: Path, feature_cols: list[str]) -> pd.DataFrame:
    required = close_example_required_columns(feature_cols)
    try:
        return pd.read_parquet(path, columns=required)
    except Exception:
        df = pd.read_parquet(path)
        keep = [col for col in required if col in df.columns]
        return df[keep].copy()


def read_prediction_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def close_strategy_overrides(
    model_type: str,
    label_mode: str,
    timeframe: str,
    cfg: CloseModelConfig,
    de_risk_mode: str | None = None,
) -> CloseModelConfig:
    payload = cfg.model_dump()
    payload["model_type"] = model_type
    payload["label_mode"] = label_mode
    payload["reference_decision_timeframe"] = timeframe
    payload["calibration"] = "none"
    if model_type == "logistic":
        payload["threshold"] = 0.5
        payload["de_risk_mode"] = "binary_exit"
    elif de_risk_mode is not None:
        payload["de_risk_mode"] = de_risk_mode
    return CloseModelConfig(**payload)


def get_pcfg(cfg) -> PortfolioConfig:
    return PortfolioConfig(**cfg.portfolio.model_dump())


def filter_rows_by_period(
    df: pd.DataFrame,
    start_month: str,
    end_month: str | None = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    # Use strftime for month strings — ~3x faster than .to_period("M").astype(str)
    months = ts.dt.strftime("%Y-%m")
    mask = months >= start_month
    if end_month is not None:
        mask &= months <= end_month
    return df.loc[mask].reset_index(drop=True)


def apply_train_window_winsorization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    winsor_lo: float,
    winsor_hi: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Filter to cols that actually exist — avoid per-col membership check in loop
    valid_cols = [col for col in cols if col in train_df.columns]
    if not valid_cols:
        return train_df.copy(), test_df.copy()
    # Batch quantile computation — single call for all columns at once
    train_arr = train_df[valid_cols].to_numpy(dtype=np.float32)
    quantiles = np.nanquantile(train_arr, [winsor_lo, winsor_hi], axis=0)  # shape (2, n_cols)
    train = train_df.copy()
    test = test_df.copy()
    for i, col in enumerate(valid_cols):
        lo, hi = quantiles[0, i], quantiles[1, i]
        train[col] = np.clip(train_arr[:, i], lo, hi)
        if col in test.columns:
            test[col] = np.clip(test[col].to_numpy(dtype=np.float32, copy=False), lo, hi)
    return train, test


def build_signal_regime_frame(
    index: pd.DatetimeIndex,
    regime_feature_sets: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    if not regime_feature_sets:
        return pd.DataFrame(index=index)
    parts = []
    for timeframe, regime_df in sorted(regime_feature_sets.items()):
        if regime_df is None or len(regime_df) == 0:
            continue
        aligned = regime_df.reindex(index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{timeframe}_")
        parts.append(aligned)
    if not parts:
        return pd.DataFrame(index=index)
    return pd.concat(parts, axis=1)


def build_regime_exposure_multiplier(
    index: pd.DatetimeIndex,
    regime_feature_sets: dict[str, pd.DataFrame] | None,
    floor: float,
) -> pd.Series:
    if not regime_feature_sets:
        return pd.Series(1.0, index=index, dtype=np.float32)
    preferred = None
    for key in ("4h", "1h"):
        if key in regime_feature_sets and len(regime_feature_sets[key]) > 0:
            preferred = regime_feature_sets[key]
            break
    if preferred is None:
        preferred = next(
            (df for df in regime_feature_sets.values() if df is not None and len(df) > 0),
            None,
        )
    if preferred is None:
        return pd.Series(1.0, index=index, dtype=np.float32)

    risk_terms = []
    for col in ("parkinson_vol_20", "cs_spread", "entropy"):
        if col not in preferred.columns:
            continue
        series = preferred[col].astype(np.float64)
        expanding_med = series.expanding(min_periods=20).median().shift(1)
        expanding_std = series.expanding(min_periods=20).std().shift(1).replace(0.0, np.nan)
        z = ((series - expanding_med) / expanding_std).clip(lower=0.0)
        risk_terms.append(z.fillna(0.0))
    if "sadf" in preferred.columns:
        sadf = preferred["sadf"].astype(np.float64)
        expanding_med = sadf.expanding(min_periods=20).median().shift(1)
        expanding_std = sadf.expanding(min_periods=20).std().shift(1).replace(0.0, np.nan)
        risk_terms.append(((sadf - expanding_med) / expanding_std).clip(lower=0.0).fillna(0.0))
    if not risk_terms:
        return pd.Series(1.0, index=index, dtype=np.float32)

    risk_score = pd.concat(risk_terms, axis=1).mean(axis=1)
    mult = 1.0 / (1.0 + 0.35 * risk_score)
    mult = mult.clip(lower=floor, upper=1.0)
    return mult.reindex(index, method="ffill").fillna(1.0).astype(np.float32)


def stats_to_row(lookback: int, stats: dict | None) -> dict:
    if stats is None:
        return {"lookback": lookback, "status": "no_result"}
    return {
        "lookback": lookback,
        "sharpe": stats["sharpe"],
        "total_return_pct": stats["total_return_pct"],
        "max_dd_pct": stats["max_dd_pct"],
        "n_trades": stats["n_trades"],
        "n_days": stats["n_days"],
        "pnl_day": stats["pnl_day"],
        "skewness": stats.get("skewness", 0.0),
        "kurtosis": stats.get("kurtosis", 3.0),
        "status": "ok",
    }


_STATS_SKIP_KEYS = frozenset({"dpnl", "sim", "combined"})

def stats_to_jsonable(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    out = {}
    for key, value in stats.items():
        if key in _STATS_SKIP_KEYS:
            continue
        if value is None:
            out[key] = value
        elif type(value) is float or type(value) is int or type(value) is str or type(value) is bool:
            out[key] = value
        elif isinstance(value, np.floating):
            out[key] = float(value)
        elif isinstance(value, np.integer):
            out[key] = int(value)
    dpnl = stats.get("dpnl")
    if isinstance(dpnl, pd.Series) and len(dpnl) > 0:
        monthly = dpnl.resample("ME").sum()
        out["monthly_pnl"] = {
            ts.strftime("%Y-%m"): float(val)
            for ts, val in monthly.items()
        }
    diagnostics = stats.get("diagnostics")
    if diagnostics:
        out["diagnostics"] = diagnostics
    return out


def summarize_diagnostics_by_period(
    diagnostics: dict,
    start_month: str,
    end_month: str | None = None,
    default_threshold: float = 0.5,
) -> dict:
    months = diagnostics.get("months", []) if diagnostics else []
    if not months:
        return {}
    # Vectorized month filtering — avoid per-row string comparisons in Python loop
    all_months = np.array([str(row.get("month", "")) for row in months])
    mask = all_months >= start_month
    if end_month is not None:
        mask &= all_months <= end_month
    indices = np.nonzero(mask)[0]
    if len(indices) == 0:
        return {}
    selected = [months[i] for i in indices]
    # Pre-extract all metric arrays in one pass
    n = len(selected)
    weights = np.empty(n, dtype=np.float64)
    thresholds = np.empty(n, dtype=np.float64)
    precisions = np.empty(n, dtype=np.float64)
    recalls = np.empty(n, dtype=np.float64)
    f1s = np.empty(n, dtype=np.float64)
    pos_rates = np.empty(n, dtype=np.float64)
    for i, row in enumerate(selected):
        weights[i] = max(int(row.get("n_validation", 0)), 1)
        thresholds[i] = row.get("threshold", default_threshold)
        precisions[i] = row.get("precision", 0.0)
        recalls[i] = row.get("recall", 0.0)
        f1s[i] = row.get("f1", 0.0)
        pos_rates[i] = row.get("positive_rate", 0.0)
    summary = diagnostics.get("summary", {})
    return {
        "calibration": summary.get("calibration", "none"),
        "calibration_note": summary.get(
            "calibration_note",
            "probability calibration disabled in this path",
        ),
        "avg_threshold": float(np.average(thresholds, weights=weights)),
        "precision": float(np.average(precisions, weights=weights)),
        "recall": float(np.average(recalls, weights=weights)),
        "f1": float(np.average(f1s, weights=weights)),
        "positive_rate": float(np.average(pos_rates, weights=weights)),
        "top_features": summary.get("top_features", []),
        "theory_sanity": summary.get("theory_sanity", []),
    }


def compute_dsr_for_rows(rows: list[dict]) -> dict:
    valid = [row for row in rows if row and row.get("status") in (None, "ok")]
    if not valid:
        return {"dsr": float("nan"), "num_trials": 0}
    sharpes = np.array([row["sharpe"] / np.sqrt(365.25) for row in valid], dtype=float)
    sr_variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.1
    best = max(valid, key=lambda row: row["sharpe"])
    dsr = deflated_sharpe_ratio(
        observed_sr=float(best["sharpe"] / np.sqrt(365.25)),
        num_trials=max(len(valid), 1),
        num_observations=max(int(best["n_days"]), 2),
        skewness=float(best.get("skewness", 0.0)),
        kurtosis=float(best.get("kurtosis", 3.0)),
        sr_variance=sr_variance if np.isfinite(sr_variance) and sr_variance > 0 else 0.1,
    )
    return {
        "dsr": float(dsr),
        "num_trials": int(len(valid)),
        "best_lookback": int(best["lookback"]),
    }


def _monthly_pnl_series(row: dict, period_key: str = "search_full_stats") -> pd.Series:
    stats = row.get(period_key) or {}
    monthly = stats.get("monthly_pnl") or {}
    if not monthly:
        return pd.Series(dtype=np.float64)
    series = pd.Series(monthly, dtype=np.float64)
    series.index = pd.to_datetime(series.index + "-01", utc=True)
    return series.sort_index()


def _partition_score(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).any():
        return -np.inf
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    if not np.isfinite(std) or std <= 1e-12:
        return mean
    return mean / std * np.sqrt(len(values))


def compute_cscv_pbo(
    rows: list[dict],
    period_key: str = "search_full_stats",
    n_partitions: int = 6,
    top_k_finalists: int = 5,
) -> dict:
    valid = [row for row in rows if row and row.get("status") in (None, "ok")]
    if len(valid) < 2:
        return {}
    monthly_map = {
        f"{row.get('timeframe', 'baseline')}:{row['lookback']}": _monthly_pnl_series(row, period_key)
        for row in valid
    }
    monthly_map = {k: v for k, v in monthly_map.items() if len(v) > 0}
    if len(monthly_map) < 2:
        return {}

    all_months = sorted(set().union(*[set(series.index) for series in monthly_map.values()]))
    if len(all_months) < n_partitions * 2:
        return {}

    month_groups = np.array_split(np.array(all_months, dtype="datetime64[ns]"), n_partitions)
    labels = list(monthly_map.keys())
    perf_matrix = np.full((len(labels), n_partitions), np.nan, dtype=np.float64)
    for i, label in enumerate(labels):
        series = monthly_map[label]
        for j, group in enumerate(month_groups):
            group_idx = pd.to_datetime(group, utc=True)
            perf_matrix[i, j] = float(series.reindex(group_idx, fill_value=0.0).sum())

    split_scores = []
    n_labels = len(labels)
    # Pre-allocate finalist OOS as a list-of-lists indexed by int (avoid dict lookups in hot loop)
    finalist_oos_arr: list[list[float]] = [[] for _ in range(n_labels)]
    half = n_partitions // 2
    all_partitions = range(n_partitions)
    all_partitions_set = set(all_partitions)
    for train_idx in combinations(all_partitions, half):
        test_idx_list = sorted(all_partitions_set - set(train_idx))
        train_idx_list = list(train_idx)
        # Vectorized: slice perf_matrix columns and compute scores for all labels at once
        train_block = perf_matrix[:, train_idx_list]
        test_block = perf_matrix[:, test_idx_list]
        train_perf = np.empty(n_labels, dtype=np.float64)
        test_perf = np.empty(n_labels, dtype=np.float64)
        for i in range(n_labels):
            train_perf[i] = _partition_score(train_block[i])
            test_perf[i] = _partition_score(test_block[i])
        if not np.isfinite(train_perf).any() or not np.isfinite(test_perf).any():
            continue
        winner = int(np.nanargmax(train_perf))
        oos_rank = int(np.sum(test_perf <= test_perf[winner]))
        rank_pct = oos_rank / max(n_labels, 1)
        split_scores.append(rank_pct)
        for i in range(n_labels):
            finalist_oos_arr[i].append(float(test_perf[i]))

    if not split_scores:
        return {}

    split_arr = np.clip(np.asarray(split_scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    # Build label→index map for fast lookup
    label_idx = {label: i for i, label in enumerate(labels)}
    top_labels = sorted(
        labels,
        key=lambda label: _partition_score(np.asarray(finalist_oos_arr[label_idx[label]] or [np.nan], dtype=np.float64)),
        reverse=True,
    )[: max(int(top_k_finalists), 1)]
    finalist_summary = {
        label: {
            "median_oos_score": float(np.nanmedian(finalist_oos_arr[label_idx[label]])),
            "p10_oos_score": float(np.nanpercentile(finalist_oos_arr[label_idx[label]], 10)),
            "p90_oos_score": float(np.nanpercentile(finalist_oos_arr[label_idx[label]], 90)),
        }
        for label in top_labels
        if finalist_oos_arr[label_idx[label]]
    }
    return {
        "num_strategies": int(len(labels)),
        "num_partitions": int(n_partitions),
        "num_splits": int(len(split_arr)),
        "pbo": float(np.mean(split_arr <= 0.5)),
        "median_logit_lambda": float(np.median(np.log(split_arr / (1.0 - split_arr)))),
        "finalists": finalist_summary,
    }


def run_feature_research(
    examples: pd.DataFrame,
    feature_cols: list[str],
    close_cfg: CloseModelConfig,
    max_rows: int = 3000,
) -> dict:
    if examples is None or len(examples) < 500:
        return {}

    search_examples = filter_rows_by_period(
        examples,
        SEARCH_START,
        SEARCH_END,
        timestamp_col="decision_timestamp",
    )
    if len(search_examples) < 500:
        return {}

    if len(search_examples) > max_rows:
        sample_idx = np.linspace(0, len(search_examples) - 1, max_rows, dtype=int)
        search_examples = search_examples.iloc[sample_idx].copy()

    regime_cols = close_feature_columns(
        feature_cols,
        regime_timeframes=close_cfg.regime_timeframes,
    )
    missing = [col for col in regime_cols if col not in search_examples.columns]
    for col in missing:
        search_examples[col] = 0.0

    X = search_examples[regime_cols].copy()
    X.index = pd.to_datetime(search_examples["decision_timestamp"], utc=True)
    y = pd.Series(search_examples["close_label"].astype(int).to_numpy(), index=X.index)
    t1 = pd.Series(
        pd.to_datetime(search_examples["label_end_timestamp"], utc=True).to_numpy(),
        index=X.index,
    )
    weights = compute_afml_close_sample_weights(search_examples)
    weights.index = X.index
    if y.nunique() < 2:
        return {}

    positives = int(y.sum())
    negatives = int(len(y) - positives)
    scale_pos_weight = negatives / max(positives, 1) if positives > 0 else 1.0
    fitted = make_xgb_close_model(scale_pos_weight=scale_pos_weight)
    fitted.fit(
        X.to_numpy(dtype=np.float32),
        y.to_numpy(dtype=np.int8),
        sample_weight=weights.to_numpy(dtype=np.float64),
    )
    mdi = mean_decrease_impurity(fitted, X.columns.tolist())

    cv = PurgedKFold(
        n_splits=3,
        t1=t1,
        pct_embargo=close_cfg.pct_embargo,
    )
    model = make_xgb_close_model(scale_pos_weight=scale_pos_weight)
    mda = mean_decrease_accuracy(
        model,
        X,
        y,
        cv=cv,
        sample_weight=weights,
        scoring="neg_log_loss",
        n_repeats=1,
    )
    sfi = single_feature_importance(
        model,
        X,
        y,
        cv=cv,
        sample_weight=weights,
        scoring="neg_log_loss",
    )
    stable = sorted(
        set(mdi.head(10).index)
        & set(mda.head(10).index)
        & set(sfi.head(10).index)
    )
    return {
        "rows": int(len(search_examples)),
        "mdi_top": mdi.head(10).reset_index().rename(columns={"index": "feature"}).to_dict("records"),
        "mda_top": mda.head(10).reset_index().rename(columns={"index": "feature"}).to_dict("records"),
        "sfi_top": sfi.head(10).reset_index().rename(columns={"index": "feature"}).to_dict("records"),
        "stable_features": stable,
    }


def historical_holdout_gate(
    candidate_holdout: dict | None,
    baseline_holdout: dict | None,
    search_dsr: float,
) -> tuple[bool, bool, bool]:
    if (
        candidate_holdout is None
        or candidate_holdout.get("status") != "ok"
        or baseline_holdout is None
        or baseline_holdout.get("status") != "ok"
    ):
        return False, False, False
    holdout_pass = (
        candidate_holdout["sharpe"] > baseline_holdout["sharpe"]
        or (
            np.isclose(candidate_holdout["sharpe"], baseline_holdout["sharpe"])
            and candidate_holdout["max_dd_pct"] < baseline_holdout["max_dd_pct"]
        )
    )
    dsr_pass = bool(search_dsr > 0.95)
    return bool(holdout_pass and dsr_pass), bool(holdout_pass), dsr_pass


def friction_adjusted_holdout_score(result: dict | None) -> tuple[float, float]:
    if not result:
        return -np.inf, np.inf
    holdout = result.get("holdout_stats") or {}
    if holdout.get("status") != "ok":
        return -np.inf, np.inf
    sharpes = [float(holdout["sharpe"])]
    stress = result.get("holdout_stress_tests", {}) or {}
    for row in stress.values():
        if "sharpe" in row:
            sharpes.append(float(row["sharpe"]))
    return min(sharpes), -float(holdout.get("max_dd_pct", np.inf))


def find_result_by_period(
    results: list[dict],
    timeframe: str,
    lookback: int,
    period_key: str,
) -> dict | None:
    for row in results:
        if row.get("timeframe") == timeframe and row.get("lookback") == lookback:
            period_row = row.get(period_key)
            if period_row is None:
                return None
            merged = dict(period_row)
            merged["timeframe"] = timeframe
            return merged
    return None


def flatten_period_results(
    results_by_timeframe: dict[str, list[dict]],
    period_key: str,
) -> list[dict]:
    out: list[dict] = []
    for timeframe, rows in results_by_timeframe.items():
        for row in rows:
            period_row = row.get(period_key)
            if period_row is None:
                out.append(
                    {
                        "lookback": row.get("lookback", "?"),
                        "timeframe": timeframe,
                        "status": "no_result",
                    }
                )
                continue
            merged = dict(period_row)
            merged["timeframe"] = timeframe
            out.append(merged)
    return out


def print_named_comparison_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 86}")
    print(title)
    print(f"{'=' * 86}")
    print(
        f"{'Strategy':<28s} {'Lookback':<10s} {'Sharpe':>7s} {'Ret%':>8s} "
        f"{'MaxDD%':>8s} {'Days':>6s}"
    )
    print("-" * 78)
    for row in rows:
        if row is None or row.get("status") not in (None, "ok"):
            label = row.get("label", "?") if row else "?"
            print(f"{label:<28s} {'-':<10s} {'NO RESULT':>7s}")
            continue
        lookback_label = f"{row['lookback']}mo"
        print(
            f"{row['label']:<28s} {lookback_label:<10s} {row['sharpe']:>7.3f} "
            f"{row['total_return_pct']:>8.2f} {row['max_dd_pct']:>8.2f} {row['n_days']:>6d}"
        )


def apply_nco_wf(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]
        if len(past_sigs) < 50:
            parts.append(km_sigs)
            continue

        assets_in_month = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for asset in assets_in_month:
            a_past = past_sigs[past_sigs["asset"] == asset]
            if len(a_past) < 10:
                continue
            ts_a = pd.to_datetime(a_past["timestamp"])
            rets = pd.Series(
                a_past["direction"].values * a_past["fwd_return"].values, index=ts_a
            )
            daily_rets[asset] = rets.resample("D").sum()

        if len(daily_rets) < 3:
            kw = compute_kelly_weights(past_sigs)
            kmc = km_sigs.copy()
            kmc["signal_strength"] = kmc.apply(
                lambda row: row["signal_strength"] * kw.get(row["asset"], 1.0), axis=1
            )
            parts.append(kmc)
            continue

        ret_df = pd.DataFrame(daily_rets).fillna(0)
        corr = ret_df.corr().values
        names = list(ret_df.columns)
        exp_r = np.array(
            [
                float(
                    np.average(
                        (
                            past_sigs[past_sigs["asset"] == asset]["direction"]
                            * past_sigs[past_sigs["asset"] == asset]["fwd_return"]
                        ),
                        weights=np.clip(
                            np.abs(
                                2.0
                                * past_sigs[past_sigs["asset"] == asset]
                                .get(
                                    "probability",
                                    pd.Series(0.5, index=past_sigs[past_sigs["asset"] == asset].index),
                                )
                                .to_numpy(dtype=np.float64)
                                - 0.5
                            )
                            * 2.0,
                            1e-6,
                            1.0,
                        ),
                    )
                )
                if asset in past_sigs["asset"].values
                else 0.0
                for asset in names
            ]
        )
        weights = compute_nco_weights(corr, exp_r, names, n_clusters=8)

        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda row: row["signal_strength"] * weights.get(row["asset"], 1.0), axis=1
        )
        parts.append(kmc)

    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def generate_asset_signals(
    asset: str,
    ohlcv_15m: pd.DataFrame,
    features_15m: pd.DataFrame,
    regime_feature_sets: dict[str, pd.DataFrame] | None,
    cfg,
) -> dict[int, pd.DataFrame]:
    feat = features_15m[cfg.features.feature_list].copy()
    signal_model_type = cfg.model.signal_model_type
    if signal_model_type.startswith("regime_"):
        feat = pd.concat(
            [feat, build_signal_regime_frame(feat.index, regime_feature_sets)],
            axis=1,
        )
    regime_exposure_mult = build_regime_exposure_multiplier(
        feat.index,
        regime_feature_sets,
        getattr(getattr(cfg, "portfolio", None), "regime_exposure_floor", 0.5),
    )
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(cfg.model.fwd_bars).shift(-cfg.model.fwd_bars)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return {lb: pd.DataFrame() for lb in LOOKBACKS}

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * (cfg.model.fwd_bars + 1)

    results = {lb: [] for lb in LOOKBACKS}
    all_months = pd.period_range(
        start=max(pd.Period(SIGNAL_BOOTSTRAP_START), start_date.to_period("M")),
        end=end_date.to_period("M"),
        freq="M",
    )
    winsor_cols = [col for col in ["ret_1", "ret_3", "ret_7", "ret_14"] if col in feat.columns]
    y_bin = (y_all > 0).astype(np.int8)
    task_workers, model_threads = signal_parallelism_plan(signal_model_type)

    pool_ctx = ThreadPoolExecutor(max_workers=task_workers) if task_workers > 1 else nullcontext()
    with pool_ctx as pool:
        for month in all_months:
            ms = pd.Timestamp(month.start_time, tz="UTC")
            me = pd.Timestamp(month.end_time, tz="UTC")
            if fc.index.tz is None:
                ms = ms.tz_localize(None)
                me = me.tz_localize(None)

            te = valid[(valid >= ms) & (valid <= me)]
            vol_train_start = ms - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
            if len(vol_train) < 50:
                continue
            vt = vol_train.quantile(cfg.features.vol_filter_quantile)
            test_vol = vol.reindex(te)
            hv = (test_vol > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5:
                continue

            feat_test_base = fc.loc[te]
            test_forward = fwd_ret.reindex(te).to_numpy(dtype=np.float32, copy=False)
            test_entry_px = close.reindex(te).to_numpy(dtype=np.float32, copy=False)
            test_regime = regime_exposure_mult.reindex(te).to_numpy(dtype=np.float32, copy=False)

            def fit_lookback(lb: int) -> tuple[int, pd.DataFrame | None]:
                train_start = ms - pd.DateOffset(months=lb)
                if train_start < start_date:
                    return lb, None
                train_cutoff = ms - embargo_td
                tr = valid[(valid >= train_start) & (valid < train_cutoff)]
                if len(tr) < 200:
                    return lb, None

                try:
                    feat_train, feat_test = apply_train_window_winsorization(
                        fc.loc[tr],
                        feat_test_base,
                        winsor_cols,
                        cfg.features.winsor_lo,
                        cfg.features.winsor_hi,
                    )
                    if signal_model_type.endswith("xgboost"):
                        tr_months = pd.DatetimeIndex(tr).to_period("M")
                        val_months = tr_months.unique()
                        fit_idx = tr
                        val_idx = pd.Index([])
                        if len(val_months) >= 2:
                            val_month = val_months.max()
                            val_mask = tr_months == val_month
                            val_idx = tr[val_mask]
                            fit_idx = tr[~val_mask]
                            if len(fit_idx) < 200 or len(val_idx) < 50:
                                fit_idx = tr
                                val_idx = pd.Index([])
                        fit_X = feat_train.loc[fit_idx].to_numpy(dtype=np.float32, copy=False)
                        fit_y = y_bin.loc[fit_idx].to_numpy(dtype=np.int8, copy=False)
                        fit_sw = compute_sample_weights(
                            fit_idx.values.astype(np.int64),
                            fwd_bars=cfg.model.fwd_bars,
                        )
                        model = make_signal_xgb(
                            seed=cfg.model.seed,
                            early_stopping_rounds=25 if len(val_idx) > 0 else None,
                            params={"n_jobs": model_threads},
                        )
                        fit_kwargs = {
                            "sample_weight": fit_sw,
                            "verbose": False,
                        }
                        if len(val_idx) > 0:
                            fit_kwargs["eval_set"] = [
                                (
                                    feat_train.loc[val_idx].to_numpy(dtype=np.float32, copy=False),
                                    y_bin.loc[val_idx].to_numpy(dtype=np.int8, copy=False),
                                )
                            ]
                        model.fit(fit_X, fit_y, **fit_kwargs)
                        yp = model.predict_proba(
                            feat_test.to_numpy(dtype=np.float32, copy=False)
                        )[:, 1]
                    else:
                        sw = compute_sample_weights(
                            feat_train.index.values.astype(np.int64),
                            fwd_bars=cfg.model.fwd_bars,
                        )
                        scaler = StandardScaler()
                        Xtr = scaler.fit_transform(feat_train)
                        model = make_logreg()
                        model.fit(
                            Xtr,
                            y_bin.loc[tr].to_numpy(dtype=np.int8, copy=False),
                            sample_weight=sw,
                        )
                        Xte = scaler.transform(feat_test)
                        yp = model.predict_proba(Xte)[:, 1]
                except Exception:
                    return lb, None

                conf = np.abs(yp - 0.5)
                dirs = np.where(yp > 0.5, 1.0, -1.0)
                sigs = get_signal(yp, num_classes=2)
                sig_disc = discretize_signal(sigs, step_size=cfg.signal.step_size)
                sig_strength = (
                    np.abs(sig_disc)
                    if np.ndim(sig_disc) > 0
                    else np.full(len(yp), abs(float(sig_disc)))
                )
                finite = np.isfinite(test_forward) & np.isfinite(test_entry_px)
                if not finite.any():
                    return lb, None

                signals = pd.DataFrame(
                    {
                        "timestamp": te[finite],
                        "direction": dirs[finite],
                        "fwd_return": test_forward[finite],
                        "confidence": conf[finite],
                        "probability": yp[finite],
                        "signal_strength": sig_strength[finite],
                        "regime_exposure_mult": test_regime[finite],
                        "entry_price": test_entry_px[finite],
                        "asset": asset,
                        "lookback_months": lb,
                    }
                )
                filtered = apply_confidence_filter(signals, cfg.confidence_filter)
                return lb, filtered if len(filtered) > 0 else None

            if pool is not None:
                month_rows = list(pool.map(fit_lookback, LOOKBACKS))
            else:
                month_rows = [fit_lookback(lb) for lb in LOOKBACKS]

            for lb, filtered in month_rows:
                if filtered is not None and len(filtered) > 0:
                    results[lb].append(filtered)

    return {
        lb: (
            pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            if parts
            else pd.DataFrame()
        )
        for lb, parts in results.items()
    }


def prepare_asset_caches(args):
    asset, symbol = args
    cfg = _get_config()
    feat_sig = feature_tag(cfg)
    data_dirs = [Path(path) for path in cfg.data.data_dirs] or [DATA_DIR]
    manifest_tag = latest_symbol_data_tag(symbol, data_dirs)
    if manifest_tag is None:
        raise FileNotFoundError(f"Could not infer data tag for {symbol}")

    cache_tfs, signal_paths = prepare_cache_requirements(asset, manifest_tag, feat_sig, cfg)
    manifest_path = manifest_cache_path(asset, manifest_tag, feat_sig)
    cached_manifest = None
    if manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text())
            have_tf_caches = all(
                ohlcv_cache_path(asset, tf, manifest_tag).exists()
                and feature_cache_path(asset, tf, manifest_tag, feat_sig).exists()
                for tf in cache_tfs
            )
            if (
                cached_manifest.get("data_tag") == manifest_tag
                and have_tf_caches
                and all(path.exists() for path in signal_paths.values())
            ):
                return cached_manifest
        except Exception:
            pass

    raw_1m_path = raw_ohlcv_cache_path(asset, manifest_tag)
    ohlcv_1m = load_or_build_frame(
        raw_1m_path,
        lambda: load_binance_1m(symbol, data_dirs),
    )
    data_tag = ohlcv_1m.index.max().strftime("%Y-%m")

    cache_tfs, signal_paths = prepare_cache_requirements(asset, data_tag, feat_sig, cfg)

    def build_timeframe_cache(tf: str) -> None:
        ohlcv_path = ohlcv_cache_path(asset, tf, data_tag)
        features_path = feature_cache_path(asset, tf, data_tag, feat_sig)
        ohlcv_tf = load_or_build_frame(
            ohlcv_path,
            lambda tf=tf: resample_ohlcv(ohlcv_1m, normalize_timeframe(tf)),
        )
        load_or_build_frame(
            features_path,
            lambda ohlcv_tf=ohlcv_tf, tf=tf: compute_features(
                ohlcv_tf,
                feature_config_for_timeframe(
                    cfg.features,
                    tf,
                    cfg.close_model.regime_timeframes,
                ),
            ),
        )

    cache_workers = min(len(cache_tfs), max(1, per_worker_thread_budget()))
    cache_tfs_ordered = sorted(cache_tfs, key=timeframe_sort_key, reverse=True)
    if cache_workers > 1:
        with ThreadPoolExecutor(max_workers=cache_workers) as pool:
            list(pool.map(build_timeframe_cache, cache_tfs_ordered))
    else:
        for tf in cache_tfs_ordered:
            build_timeframe_cache(tf)

    counts = {}
    if all(path.exists() for path in signal_paths.values()):
        counts = (
            {int(lb): int(count) for lb, count in cached_manifest.get("signal_counts", {}).items()}
            if cached_manifest
            else {}
        )
        if len(counts) != len(signal_paths):
            for lb, path in signal_paths.items():
                counts[lb] = int(len(pd.read_parquet(path)))
    else:
        signal_ohlcv = pd.read_parquet(ohlcv_cache_path(asset, "15min", data_tag))
        signal_features = pd.read_parquet(
            feature_cache_path(asset, "15min", data_tag, feat_sig)
        )
        signal_regime_features = {
            regime_tf: pd.read_parquet(
                feature_cache_path(asset, regime_tf, data_tag, feat_sig)
            )
            for regime_tf in cfg.close_model.regime_timeframes
            if feature_cache_path(asset, regime_tf, data_tag, feat_sig).exists()
        }
        signal_results = generate_asset_signals(
            asset,
            signal_ohlcv,
            signal_features,
            signal_regime_features,
            cfg,
        )
        for lb, df in signal_results.items():
            path = signal_paths[lb]
            write_parquet(df, path)
            counts[lb] = int(len(df))

    manifest = {
        "asset": asset,
        "symbol": symbol,
        "data_tag": data_tag,
        "feature_tag": feat_sig,
        "start": str(ohlcv_1m.index.min()),
        "end": str(ohlcv_1m.index.max()),
        "signal_counts": counts,
    }
    write_json(manifest, manifest_cache_path(asset, data_tag, feat_sig))
    return manifest


def build_asset_close_examples(args):
    asset, lookback, timeframe, data_tag, feat_sig, scope_tag, label_mode = args
    cache_path = close_example_cache_path(
        asset, lookback, timeframe, data_tag, feat_sig, scope_tag, label_mode
    )
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return asset, lookback, timeframe, len(df)

    signals = pd.read_parquet(
        adjusted_signal_cache_path(asset, lookback, data_tag, feat_sig, scope_tag)
    )
    if len(signals) == 0:
        write_parquet(pd.DataFrame(), cache_path)
        return asset, lookback, timeframe, 0

    signal_ohlcv = pd.read_parquet(ohlcv_cache_path(asset, "15min", data_tag))
    decision_ohlcv = pd.read_parquet(ohlcv_cache_path(asset, timeframe, data_tag))
    decision_features = pd.read_parquet(feature_cache_path(asset, timeframe, data_tag, feat_sig))
    cfg = _get_config()
    regime_feature_sets = {
        regime_tf: pd.read_parquet(feature_cache_path(asset, regime_tf, data_tag, feat_sig))
        for regime_tf in cfg.close_model.regime_timeframes
    }

    examples = build_close_decision_examples(
        signals,
        signal_ohlcv,
        decision_ohlcv,
        decision_features,
        cfg.features.feature_list,
        timeframe,
        cfg.close_model.max_hold_minutes,
        slippage=get_pcfg(cfg).slippage,
        label_mode=label_mode,
        min_hold_minutes=cfg.close_model.min_hold_minutes,
        pt_mult=cfg.close_model.pt_mult,
        sl_mult=cfg.close_model.sl_mult,
        min_improvement_bps=cfg.close_model.min_improvement_bps,
        regime_feature_sets=regime_feature_sets,
    )
    if len(examples) > 0:
        examples["trade_id"] = (
            asset + ":" + examples["trade_id"].astype(str)
        )
    write_parquet(examples, cache_path)
    return asset, lookback, timeframe, len(examples)


def build_combined_close_examples(args):
    lookback, timeframe, data_tag, feat_sig, scope_tag, asset_names, label_mode, _pcfg = args
    cache_path = combined_close_example_cache_path(
        lookback, timeframe, data_tag, feat_sig, scope_tag, label_mode
    )
    if cache_path.exists():
        df = read_close_example_frame(
            cache_path,
            _get_config().features.feature_list,
        )
        return lookback, timeframe, len(df)

    cfg = _get_config()
    frames = []
    for asset in asset_names:
        path = close_example_cache_path(
            asset, lookback, timeframe, data_tag, feat_sig, scope_tag, label_mode
        )
        if path.exists():
            df = read_close_example_frame(path, cfg.features.feature_list)
            if len(df) > 0:
                frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_parquet(combined, cache_path)
    return lookback, timeframe, len(combined)


def evaluate_baseline_trades(trades: pd.DataFrame, pcfg: PortfolioConfig) -> dict | None:
    if len(trades) == 0:
        return None
    sim = simulate_portfolio(trades, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(
        dpnl,
        pcfg.starting_capital,
        sim["n_trades"],
        sim["n_trades"] / len(dpnl),
        final_equity=sim.get("final_equity"),
    )
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)
    return {
        **stats,
        "n_trades": sim["n_trades"],
        "pnl_day": pnl_day,
        "dpnl": dpnl,
        "sim": sim,
        "combined": trades,
    }


def evaluate_selected_trades(trades: pd.DataFrame, pcfg: PortfolioConfig) -> dict | None:
    if len(trades) == 0:
        return None
    sim = simulate_portfolio_precomputed_exits(trades, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(
        dpnl,
        pcfg.starting_capital,
        sim["n_trades"],
        sim["n_trades"] / len(dpnl),
        final_equity=sim.get("final_equity"),
    )
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)
    return {
        **stats,
        "n_trades": sim["n_trades"],
        "pnl_day": pnl_day,
        "dpnl": dpnl,
        "sim": sim,
        "combined": trades,
    }


def evaluate_selected_trades_under_cost_scenarios(
    trades: pd.DataFrame,
    pcfg: PortfolioConfig,
    scenarios: list[float],
) -> dict[str, dict]:
    if not scenarios or len(trades) == 0:
        return {}
    # When return_already_signed is all True, slippage has no effect on results.
    # Run the simulation once and reuse for all cost multipliers.
    all_signed = (
        "return_already_signed" in trades.columns
        and trades["return_already_signed"].all()
    )
    if all_signed:
        stats = evaluate_selected_trades(trades, pcfg)
        if stats is None:
            return {}
        jsonable = stats_to_jsonable(stats) or {}
        return {f"{float(mult):.1f}x": jsonable for mult in scenarios}
    out: dict[str, dict] = {}
    for mult in scenarios:
        stressed_cfg = PortfolioConfig(
            **{
                **pcfg.model_dump(),
                "slippage": float(pcfg.slippage) * float(mult),
            }
        )
        stats = evaluate_selected_trades(trades, stressed_cfg)
        if stats is not None:
            out[f"{float(mult):.1f}x"] = stats_to_jsonable(stats) or {}
    return out


def combine_signal_lookback(args):
    lookback, signal_paths, combined_path, pcfg = args
    if combined_path.exists():
        combined = pd.read_parquet(combined_path)
    else:
        frames = []
        for path in signal_paths:
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if len(df) > 0:
                frames.append(df)
        combined = (
            pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            if frames
            else pd.DataFrame()
        )
        if len(combined) > 0:
            combined = apply_nco_wf(combined)
        write_parquet(combined, combined_path)

    search_stats = evaluate_baseline_trades(
        filter_rows_by_period(combined, SEARCH_START, SEARCH_END),
        pcfg,
    )
    holdout_stats = evaluate_baseline_trades(
        filter_rows_by_period(combined, HOLDOUT_START, None),
        pcfg,
    )
    return lookback, {
        "lookback": lookback,
        "search_stats": stats_to_row(lookback, search_stats),
        "holdout_stats": stats_to_row(lookback, holdout_stats),
        "search_full_stats": stats_to_jsonable(search_stats),
        "holdout_full_stats": stats_to_jsonable(holdout_stats),
    }


def evaluate_close_model(args):
    lookback, timeframe, data_tag, feat_sig, scope_tag, asset_names, model_type, label_mode, de_risk_mode, pcfg = args
    _tag = f"{model_type}:{de_risk_mode} lb={lookback} tf={timeframe}"
    _t_total = time.time()
    strategy = strategy_variant_tag(model_type, label_mode, de_risk_mode)
    result_path = result_cache_path(
        lookback, timeframe, data_tag, feat_sig, scope_tag, strategy
    )
    if result_path.exists():
        return json.loads(result_path.read_text())

    cfg = _get_config()
    close_cfg = close_strategy_overrides(
        model_type,
        label_mode,
        timeframe,
        cfg.close_model,
        de_risk_mode=de_risk_mode,
    )
    selected_path = selected_trade_cache_path(
        lookback, timeframe, data_tag, feat_sig, scope_tag, strategy
    )
    selected_diag_path = selected_trade_diagnostics_cache_path(
        lookback, timeframe, data_tag, feat_sig, scope_tag, strategy
    )
    if selected_path.exists() and selected_diag_path.exists():
        selected = pd.read_parquet(selected_path)
        diagnostics = json.loads(selected_diag_path.read_text())
        examples = None
    else:
        selected = None
        diagnostics = None
        examples = None

    if selected is None:
        prediction_cfg = close_strategy_overrides(
            model_type,
            label_mode,
            timeframe,
            cfg.close_model,
            de_risk_mode=canonical_prediction_de_risk_mode(
                model_type,
                label_mode,
                de_risk_mode,
            ),
        )
        combined_path = combined_close_example_cache_path(
            lookback, timeframe, data_tag, feat_sig, scope_tag, label_mode
        )
        _t0 = time.time()
        if combined_path.exists():
            examples = read_close_example_frame(combined_path, cfg.features.feature_list)
        else:
            frames = []
            for asset in asset_names:
                path = close_example_cache_path(
                    asset, lookback, timeframe, data_tag, feat_sig, scope_tag, label_mode
                )
                if path.exists():
                    df = read_close_example_frame(path, cfg.features.feature_list)
                    if len(df) > 0:
                        frames.append(df)
            examples = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            write_parquet(examples, combined_path)
        print(f"    [{_tag}] load examples: {time.time()-_t0:.1f}s ({len(examples)} rows)", flush=True)

        if len(examples) == 0:
            result = {
                "lookback": lookback,
                "timeframe": timeframe,
                "model_type": model_type,
                "label_mode": label_mode,
                "status": "no_examples",
            }
            write_json(result, result_path)
            return result

        prediction_path = prediction_cache_path(
            lookback,
            timeframe,
            data_tag,
            feat_sig,
            scope_tag,
            model_type,
            label_mode,
        )
        validation_path = prediction_validation_cache_path(
            lookback,
            timeframe,
            data_tag,
            feat_sig,
            scope_tag,
            model_type,
            label_mode,
        )
        prediction_diag_path = prediction_diagnostics_cache_path(
            lookback,
            timeframe,
            data_tag,
            feat_sig,
            scope_tag,
            model_type,
            label_mode,
        )
        if prediction_path.exists() and prediction_diag_path.exists():
            _t0 = time.time()
            prediction_rows = read_prediction_frame(prediction_path)
            validation_rows = (
                read_prediction_frame(validation_path)
                if validation_path.exists()
                else pd.DataFrame()
            )
            prediction_diagnostics = json.loads(prediction_diag_path.read_text())
            print(f"    [{_tag}] load predictions (cached): {time.time()-_t0:.1f}s", flush=True)
        else:
            _t0 = time.time()
            prediction_rows, validation_rows, prediction_diagnostics = walk_forward_close_prediction_cache(
                examples,
                cfg.features.feature_list,
                lookback,
                prediction_cfg,
                test_start=TEST_START,
                signal_bar_minutes=15,
                portfolio_cfg=pcfg,
            )
            _t_wf = time.time() - _t0
            print(f"    [{_tag}] walk_forward: {_t_wf:.1f}s ({len(prediction_rows)} predictions)", flush=True)
            _t0 = time.time()
            write_parquet(prediction_rows, prediction_path)
            write_parquet(validation_rows, validation_path)
            write_json(prediction_diagnostics, prediction_diag_path)
            print(f"    [{_tag}] write predictions: {time.time()-_t0:.1f}s", flush=True)

        _t0 = time.time()
        selected, diagnostics = select_close_predictions(
            prediction_rows,
            validation_rows,
            close_cfg,
            signal_bar_minutes=15,
            portfolio_cfg=pcfg,
            prediction_diagnostics=prediction_diagnostics,
        )
        print(f"    [{_tag}] select_close_predictions: {time.time()-_t0:.1f}s ({len(selected)} selected)", flush=True)
        _t0 = time.time()
        write_parquet(selected, selected_path)
        write_json(diagnostics, selected_diag_path)
        print(f"    [{_tag}] write selected: {time.time()-_t0:.1f}s", flush=True)

    if len(selected) == 0:
        result = {
            "lookback": lookback,
            "timeframe": timeframe,
            "model_type": model_type,
            "label_mode": label_mode,
            "status": "no_predictions",
        }
        write_json(result, result_path)
        return result

    _t0 = time.time()
    search_selected = filter_rows_by_period(selected, SEARCH_START, SEARCH_END)
    holdout_selected = filter_rows_by_period(selected, HOLDOUT_START, None)
    search_stats = evaluate_selected_trades(search_selected, pcfg)
    holdout_stats = evaluate_selected_trades(holdout_selected, pcfg)
    print(f"    [{_tag}] evaluate trades: {time.time()-_t0:.1f}s", flush=True)

    if search_stats is None and holdout_stats is None:
        result = {
            "lookback": lookback,
            "timeframe": timeframe,
            "model_type": model_type,
            "label_mode": label_mode,
            "status": "no_period_stats",
        }
        write_json(result, result_path)
        return result

    _t0 = time.time()
    search_diag = summarize_diagnostics_by_period(
        diagnostics,
        SEARCH_START,
        SEARCH_END,
        default_threshold=float(close_cfg.threshold),
    )
    holdout_diag = summarize_diagnostics_by_period(
        diagnostics,
        HOLDOUT_START,
        None,
        default_threshold=float(close_cfg.threshold),
    )
    stress = (
        evaluate_selected_trades_under_cost_scenarios(
            holdout_selected,
            pcfg,
            list(close_cfg.execution_cost_scenarios),
        ) if timeframe in {"5min", "10min"} else {}
    )
    print(f"    [{_tag}] diagnostics+stress: {time.time()-_t0:.1f}s", flush=True)

    result = {
        "lookback": lookback,
        "timeframe": timeframe,
        "model_type": model_type,
        "label_mode": label_mode,
        "de_risk_mode": close_cfg.de_risk_mode,
        "strategy": strategy,
        "status": "ok",
        "search_stats": stats_to_row(lookback, search_stats),
        "holdout_stats": stats_to_row(lookback, holdout_stats),
        "search_full_stats": stats_to_jsonable(search_stats),
        "holdout_full_stats": stats_to_jsonable(holdout_stats),
        "search_diagnostics": search_diag,
        "holdout_diagnostics": holdout_diag,
        "diagnostics": diagnostics.get("summary", {}),
        "holdout_stress_tests": stress,
    }
    write_json(result, result_path)
    print(f"    [{_tag}] TOTAL: {time.time()-_t_total:.1f}s", flush=True)
    return result


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")
    print(
        f"{'Lookback':<10s} {'Sharpe':>7s} {'Ret%':>8s} "
        f"{'MaxDD%':>8s} {'Trades':>8s} {'Days':>6s} {'PnL/day':>9s}"
    )
    print("-" * 68)
    for row in rows:
        if row is None or row.get("status") not in (None, "ok"):
            lookback = row.get("lookback") if row else "?"
            lookback_label = f"{lookback}mo" if isinstance(lookback, int) else str(lookback)
            print(f"  {lookback_label:>6s}    NO RESULT")
            continue
        print(
            f"  {row['lookback']:>2d}mo      {row['sharpe']:>7.3f} {row['total_return_pct']:>8.2f} "
            f"{row['max_dd_pct']:>8.2f} {row['n_trades']:>8d} {row['n_days']:>6d} "
            f"${row['pnl_day']:>8.2f}"
        )


def extract_period_rows(rows: list[dict], period_key: str) -> list[dict]:
    out = []
    for row in rows:
        if not row:
            out.append({"lookback": "?", "status": "no_result"})
            continue
        period_row = row.get(period_key)
        if period_row is None:
            out.append({"lookback": row.get("lookback", "?"), "status": "no_result"})
            continue
        out.append(period_row)
    return out


def full_stats_report(
    label: str,
    result: dict | None,
    training_desc: str,
    pcfg: PortfolioConfig,
) -> None:
    if result is None:
        print(f"\n{label}: NO RESULT")
        return

    dpnl = result["dpnl"]
    sim = result["sim"]
    stats = result
    daily_returns = dpnl / pcfg.starting_capital
    equity = sim["equity"]
    timestamps = sim["timestamps"]

    wins = dpnl[dpnl > 0]
    losses = dpnl[dpnl < 0]
    win_rate = len(wins) / len(dpnl) * 100 if len(dpnl) > 0 else 0
    avg_win_bps = (wins.mean() / pcfg.starting_capital * 10000) if len(wins) > 0 else 0
    avg_loss_bps = (losses.mean() / pcfg.starting_capital * 10000) if len(losses) > 0 else 0
    daily_arr = dpnl.values
    rng = np.random.RandomState(42)
    boot_sharpes = []
    for _ in range(1000):
        sample = rng.choice(daily_arr, size=len(daily_arr), replace=True)
        s_std = sample.std()
        if s_std > 0:
            boot_sharpes.append(sample.mean() / s_std * np.sqrt(365))
    ci_lo = np.percentile(boot_sharpes, 2.5)
    ci_hi = np.percentile(boot_sharpes, 97.5)

    sharpe = stats["sharpe"]
    skew = stats.get("skewness", stats.get("skew", 0.0))
    kurt = stats.get("kurtosis", 3.0)
    sr_std = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / max(len(dpnl), 1))
    psr = norm.cdf(sharpe / sr_std) * 100 if sr_std > 0 else 100
    equity_series = pd.Series(
        equity[~sim["skipped"]],
        index=pd.to_datetime(timestamps[~sim["skipped"]]),
    )
    monthly_pnl = dpnl.resample("ME").sum()

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  Sharpe Ratio                 {stats['sharpe']:.3f}")
    print(f"  Sortino Ratio                {stats.get('sortino', 0):.3f}")
    print(f"  Calmar Ratio                 {stats.get('calmar', 0):.3f}")
    print(f"  PSR                          {psr:.1f}%")
    print(f"  Bootstrap 95% CI             [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  End Equity                   ${stats['final_equity']:,.2f}")
    print(f"  Net Profit                   {stats['total_return_pct']:.2f}%")
    print(f"  Max Drawdown                 {stats['max_dd_pct']:.2f}%")
    print(f"  Total Orders                 {stats['n_trades']:,d}")
    print(f"  Win Rate                     {win_rate:.1f}%")
    print(f"  Average Win                  {avg_win_bps:.1f} bps/day")
    print(f"  Average Loss                 {avg_loss_bps:.1f} bps/day")
    print(f"  PnL per Day                  ${stats['pnl_day']:.2f}")
    print(f"  Days of Data                 {stats['n_days']}")
    print(f"  Training Window              {training_desc}")
    print(f"  Equity Samples               {len(equity_series)}")
    diagnostics = stats.get("diagnostics", {})
    if diagnostics:
        print(f"  Close Precision              {diagnostics.get('precision', 0.0):.3f}")
        print(f"  Close Recall                 {diagnostics.get('recall', 0.0):.3f}")
        print(f"  Close F1                     {diagnostics.get('f1', 0.0):.3f}")
        print(f"  Positive Class Rate          {diagnostics.get('positive_rate', 0.0):.3f}")
        print(f"  Avg Threshold                {diagnostics.get('avg_threshold', 0.0):.3f}")
        print(f"  Calibration                  {diagnostics.get('calibration', 'none')}")
        print(f"  Diagnostics Note             {diagnostics.get('calibration_note', '')}")
    print("  --- MONTHLY PnL ---")
    for ts, pnl in monthly_pnl.items():
        print(f"  {ts.strftime('%Y-%m')}: ${pnl:>10.2f}")
    top_features = diagnostics.get("top_features", []) if diagnostics else []
    if top_features:
        print("  --- TOP FEATURES ---")
        for row in top_features[:5]:
            print(f"  {row['feature']}: {row['importance']:.4f}")
    theory = diagnostics.get("theory_sanity", []) if diagnostics else []
    if theory:
        print("  --- THEORY SANITY ---")
        for row in theory:
            print(f"  {row['feature']}: {row['mean']:.4f}")


def resolve_assets() -> dict[str, str]:
    env_assets = os.getenv("ML_CLOSE_ASSETS", "").strip()
    if not env_assets:
        return ASSETS
    wanted = {token.strip().upper() for token in env_assets.split(",") if token.strip()}
    return {asset: symbol for asset, symbol in ASSETS.items() if asset in wanted}


def main():
    assets = resolve_assets()
    scope_tag = asset_scope_tag(assets)
    workers = int(os.getenv("ML_CLOSE_WORKERS", min(W, 8)))
    eval_workers = int(os.getenv("ML_CLOSE_EVAL_WORKERS", min(workers, 8)))
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = get_pcfg(cfg)
    feat_sig = feature_tag(cfg)
    production_tf = cfg.close_model.reference_decision_timeframe
    research_tfs = list(dict.fromkeys(cfg.close_model.research_decision_timeframes))
    production_candidate_tfs = list(dict.fromkeys(research_tfs))
    if os.getenv("ML_CLOSE_BASELINE_ONLY") == "1":
        strategy_specs = []
    else:
        strategy_specs = [
            {
                "label": f"XGBoost AFML Binary ({production_tf})",
                "model_type": "xgboost",
                "label_mode": "residual_triple_barrier",
                "de_risk_mode": "binary_exit",
                "timeframes": [production_tf],
            },
            {
                "label": "XGBoost AFML Partial",
                "model_type": "xgboost",
                "label_mode": "residual_triple_barrier",
                "de_risk_mode": "partial_ladder",
                "timeframes": research_tfs,
            },
        ]

    print("=" * 86)
    print("ML EARLY-CLOSE WALK-FORWARD REPORT")
    print(f"  Assets: {len(assets)}  Workers: {workers}")
    print(f"  Eval workers: {eval_workers}")
    print(f"  Asset scope tag: {scope_tag}")
    print(f"  Signal lookbacks: {LOOKBACKS}")
    print(f"  Historical reference close timeframe: {production_tf}")
    print(f"  Research candidate close timeframes: {production_candidate_tfs}")
    print(f"  Research close timeframes: {research_tfs}")
    print(f"  Signal bootstrap: {SIGNAL_BOOTSTRAP_START} to 2021-12")
    print(f"  Search window: {SEARCH_START} to {SEARCH_END}")
    print(f"  Final holdout: {HOLDOUT_START} onward")
    print("=" * 86, flush=True)

    t0 = time.time()

    print(f"\n[1] Preparing OHLCV/features/signals with caching ({workers} workers)...", flush=True)
    manifests = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare_asset_caches, (asset, symbol)): asset for asset, symbol in assets.items()}
        done = 0
        for future in as_completed(futures):
            manifest = future.result()
            manifests.append(manifest)
            done += 1
            total_signals = sum(manifest["signal_counts"].values())
            print(f"  [{done:>2d}/{len(assets)}] {manifest['asset']}: cached {total_signals} signals", flush=True)

    data_tag = max(manifest["data_tag"] for manifest in manifests)
    print(f"  Data tag: {data_tag}  Elapsed: {time.time() - t0:.0f}s", flush=True)

    combined_by_lookback = {}
    print(f"\n[2] Combining signals and applying NCO allocation ({min(workers, len(LOOKBACKS))} workers)...", flush=True)
    combine_tasks = []
    for lookback in LOOKBACKS:
        signal_paths = [
            signal_cache_path(manifest["asset"], lookback, manifest["data_tag"], feat_sig)
            for manifest in manifests
        ]
        combine_tasks.append(
            (
                lookback,
                signal_paths,
                combined_cache_path(lookback, data_tag, feat_sig, scope_tag),
                pcfg,
            )
        )

    baseline_by_lookback = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(combine_tasks))) as pool:
        futures = {pool.submit(combine_signal_lookback, task): task[0] for task in combine_tasks}
        for idx, future in enumerate(as_completed(futures), start=1):
            lookback, baseline_row = future.result()
            baseline_by_lookback[lookback] = baseline_row
            print(f"  [{idx:>2}/{len(combine_tasks)}] lb={lookback}: combined", flush=True)

    for lookback in LOOKBACKS:
        combined_path = combined_cache_path(lookback, data_tag, feat_sig, scope_tag)
        combined_by_lookback[lookback] = (
            pd.read_parquet(combined_path) if combined_path.exists() else pd.DataFrame()
        )
    baseline_search_rows = extract_period_rows(
        [baseline_by_lookback.get(lookback, {"lookback": lookback}) for lookback in LOOKBACKS],
        "search_stats",
    )

    print(f"\n[3] Building close-decision examples with caching ({workers} workers)...", flush=True)
    tasks = []
    example_specs = sorted(
        {
            (tf, spec["label_mode"])
            for spec in strategy_specs
            for tf in spec["timeframes"]
        },
        key=lambda item: (timeframe_sort_key(item[0]), item[1]),
    )
    for manifest in manifests:
        for lookback in LOOKBACKS:
            adjusted = combined_by_lookback[lookback]
            if len(adjusted) == 0:
                continue
            adjusted_asset = adjusted[adjusted["asset"] == manifest["asset"]].reset_index(drop=True)
            write_parquet(
                adjusted_asset,
                adjusted_signal_cache_path(
                    manifest["asset"],
                    lookback,
                    manifest["data_tag"],
                    feat_sig,
                    scope_tag,
                ),
            )
            for tf, label_mode in example_specs:
                tasks.append(
                    (
                        manifest["asset"],
                        lookback,
                        tf,
                        manifest["data_tag"],
                        feat_sig,
                        scope_tag,
                        label_mode,
                    )
                )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_asset_close_examples, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            asset, lookback, tf, n_rows = future.result()
            done += 1
            if done % max(1, len(assets)) == 0 or done == len(tasks):
                print(f"  [{done:>3d}/{len(tasks)}] {asset} lb={lookback} tf={tf}: {n_rows} examples", flush=True)

    print(f"\n[4] Combining close-decision examples for evaluation caching...", flush=True)
    combine_tasks = []
    for tf, label_mode in example_specs:
        for lb in LOOKBACKS:
            combine_tasks.append(
                (
                    lb,
                    tf,
                    data_tag,
                    feat_sig,
                    scope_tag,
                    tuple(assets.keys()),
                    label_mode,
                    pcfg,
                )
            )
    if combine_tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(combine_tasks))) as pool:
            futures = {pool.submit(build_combined_close_examples, task): task for task in combine_tasks}
            done = 0
            for future in as_completed(futures):
                lookback, tf, n_rows = future.result()
                done += 1
                print(
                    f"  [{done:>2d}/{len(combine_tasks)}] lb={lookback} tf={tf}: {n_rows} combined examples",
                    flush=True,
            )

    ml_results: dict[tuple[str, str, str], list[dict]] = {}
    if not strategy_specs:
        print("\n[5] Skipped (baseline-only mode)", flush=True)
    else:
        print(f"\n[5] Evaluating ML close model ({eval_workers} workers)...", flush=True)
    eval_tasks_unsorted = [
        (
            lb,
            tf,
            data_tag,
            feat_sig,
            scope_tag,
            tuple(assets.keys()),
            spec["model_type"],
            spec["label_mode"],
            spec["de_risk_mode"],
            pcfg,
        )
        for spec in strategy_specs
        for tf in spec["timeframes"]
        for lb in LOOKBACKS
    ]
    # Pre-compute file sizes once (single stat per unique path) instead of
    # O(n log n) stat calls during sort comparisons.
    _size_cache: dict[str, int] = {}
    def _cached_file_size(path: Path) -> int:
        key = str(path)
        if key not in _size_cache:
            try:
                _size_cache[key] = path.stat().st_size
            except OSError:
                _size_cache[key] = 0
        return _size_cache[key]

    task_sizes = np.array(
        [
            _cached_file_size(
                combined_close_example_cache_path(
                    task[0], task[1], task[2], task[3], task[4], task[7],
                )
            )
            for task in eval_tasks_unsorted
        ],
        dtype=np.int64,
    )
    # Argsort descending — avoids repeated key evaluation during sort
    sort_order = np.argsort(-task_sizes)
    eval_tasks = [eval_tasks_unsorted[i] for i in sort_order]
    # Pre-filter cached results — avoid subprocess overhead for already-computed tasks
    uncached_tasks = []
    done = 0
    total_tasks = len(eval_tasks)
    for task in eval_tasks:
        rpath = result_cache_path(
            task[0], task[1], task[2], task[3], task[4],
            strategy_variant_tag(task[6], task[7], task[8]),
        )
        if rpath.exists():
            try:
                result = json.loads(rpath.read_text())
                key = (
                    result["model_type"],
                    result["timeframe"],
                    result.get("de_risk_mode", "binary_exit"),
                )
                ml_results.setdefault(key, []).append(result)
                done += 1
                print(
                    f"  [{done:>2d}/{total_tasks}] {result['model_type']}:{result.get('de_risk_mode', 'binary_exit')} lb={result['lookback']} tf={result['timeframe']}: {result['status']} (cached)",
                    flush=True,
                )
                continue
            except Exception:
                pass
        uncached_tasks.append(task)

    if uncached_tasks:
        with ProcessPoolExecutor(max_workers=min(eval_workers, len(uncached_tasks))) as pool:
            futures = {pool.submit(evaluate_close_model, task): task for task in uncached_tasks}
            for future in as_completed(futures):
                result = future.result()
                key = (
                    result["model_type"],
                    result["timeframe"],
                    result.get("de_risk_mode", "binary_exit"),
                )
                ml_results.setdefault(key, []).append(result)
                done += 1
                print(
                    f"  [{done:>2d}/{total_tasks}] {result['model_type']}:{result.get('de_risk_mode', 'binary_exit')} lb={result['lookback']} tf={result['timeframe']}: {result['status']}",
                    flush=True,
                )

    xgb_binary_results = sorted(
        ml_results.get(("xgboost", production_tf, "binary_exit"), []),
        key=lambda row: row["lookback"],
    )
    xgb_prod_results = sorted(
        ml_results.get(("xgboost", production_tf, "partial_ladder"), []),
        key=lambda row: row["lookback"],
    )
    xgb_binary_search_rows = extract_period_rows(xgb_binary_results, "search_stats")
    xgb_prod_search_rows = extract_period_rows(xgb_prod_results, "search_stats")

    print(f"\n{'=' * 86}")
    print(f"SEARCH PERIOD TABLES ({SEARCH_START} to {SEARCH_END})")
    print(f"{'=' * 86}")
    print_table("FIXED 1H BASELINE (15m signal, automatic 1h exit) — SEARCH", baseline_search_rows)
    baseline_holdout_rows = extract_period_rows(
        [baseline_by_lookback.get(lookback, {"lookback": lookback}) for lookback in LOOKBACKS],
        "holdout_stats",
    )
    print_table("FIXED 1H BASELINE (15m signal, automatic 1h exit) — HOLDOUT", baseline_holdout_rows)
    if strategy_specs:
        print_table(f"XGBOOST AFML BINARY ABLATION ({production_tf})", xgb_binary_search_rows)
        print_table(f"XGBOOST AFML PARTIAL CLOSE MODEL ({production_tf})", xgb_prod_search_rows)

    xgb_results_by_tf: dict[str, list[dict]] = {production_tf: xgb_prod_results}
    for tf in research_tfs:
        if tf == production_tf:
            continue
        rows = sorted(
            ml_results.get(("xgboost", tf, "partial_ladder"), []),
            key=lambda row: row["lookback"],
        )
        xgb_results_by_tf[tf] = rows
        print_table(f"XGBOOST AFML PARTIAL CLOSE MODEL ({tf})", extract_period_rows(rows, "search_stats"))

    baseline_search_valid = [row for row in baseline_search_rows if row.get("status") == "ok"]
    baseline_dsr = compute_dsr_for_rows(baseline_search_rows)
    baseline_cscv = compute_cscv_pbo(baseline_search_rows)
    best_baseline_search = max(baseline_search_valid, key=lambda row: row["sharpe"]) if baseline_search_valid else None
    best_baseline_holdout = (
        baseline_by_lookback[best_baseline_search["lookback"]]["holdout_stats"]
        if best_baseline_search is not None
        else None
    )

    xgb_candidate_search_rows = flatten_period_results(
        {tf: rows for tf, rows in xgb_results_by_tf.items() if tf in production_candidate_tfs},
        "search_stats",
    )
    xgb_candidate_search_valid = [
        row for row in xgb_candidate_search_rows if row.get("status") == "ok"
    ]
    xgb_prod_dsr = compute_dsr_for_rows(xgb_candidate_search_rows)
    xgb_candidate_cscv = compute_cscv_pbo(xgb_candidate_search_rows)
    xgb_candidate_full = [
        row
        for tf in production_candidate_tfs
        for row in xgb_results_by_tf.get(tf, [])
        if row.get("status") == "ok"
    ]
    best_result_full = (
        max(xgb_candidate_full, key=friction_adjusted_holdout_score)
        if xgb_candidate_full
        else None
    )
    best_result = (
        find_result_by_period(
            xgb_results_by_tf.get(best_result_full["timeframe"], []),
            best_result_full["timeframe"],
            best_result_full["lookback"],
            "search_stats",
        )
        if best_result_full is not None
        else None
    )
    best_holdout = (
        find_result_by_period(
            xgb_results_by_tf.get(best_result_full["timeframe"], []),
            best_result_full["timeframe"],
            best_result_full["lookback"],
            "holdout_stats",
        )
        if best_result_full is not None
        else None
    )

    print_named_comparison_table(
        f"FINAL HOLDOUT COMPARISON ({HOLDOUT_START} onward)",
        [
            {**(best_baseline_holdout or {"lookback": best_baseline_search["lookback"] if best_baseline_search else "-", "status": "no_result"}), "label": "Fixed 1H Baseline"},
            {**(best_holdout or {"lookback": best_result_full["lookback"] if best_result_full else "-", "status": "no_result"}), "label": f"XGBoost Partial {best_result_full['timeframe'] if best_result_full else production_tf}"},
        ],
    )

    print(f"\n{'=' * 86}")
    print("BEST CONFIGURATION")
    print(f"{'=' * 86}")
    report_payload = {
        "metadata": {
            "run_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            "data_tag": data_tag,
            "asset_scope_tag": scope_tag,
            "assets": sorted(assets.keys()),
            "workers": workers,
            "eval_workers": eval_workers,
            "signal_bootstrap_start": SIGNAL_BOOTSTRAP_START,
            "search_start": SEARCH_START,
            "search_end": SEARCH_END,
            "holdout_start": HOLDOUT_START,
            "feature_tag": feat_sig,
            "cache_version": CACHE_VERSION,
        },
        "baseline": [
            baseline_by_lookback[lookback]
            for lookback in LOOKBACKS
            if lookback in baseline_by_lookback
        ],
        "xgboost_binary": {
            production_tf: xgb_binary_results,
        },
        "xgboost": {
            tf: rows
            for tf, rows in sorted(xgb_results_by_tf.items())
        },
        "selection": {
            "baseline_search_dsr": baseline_dsr,
            "baseline_search_cscv": baseline_cscv,
            "xgboost_candidate_search_dsr": xgb_prod_dsr,
            "xgboost_candidate_search_cscv": xgb_candidate_cscv,
            "best_baseline_search": best_baseline_search,
            "best_baseline_holdout": best_baseline_holdout,
            "best_xgboost_search": best_result,
            "best_xgboost_holdout": best_holdout,
        },
    }
    if best_baseline_search is not None:
        print(
            f"  Best fixed 1h baseline (search): {best_baseline_search['lookback']}mo "
            f"Sharpe {best_baseline_search['sharpe']:.3f} Ret {best_baseline_search['total_return_pct']:.2f}% "
            f"| DSR {baseline_dsr['dsr']:.3f} across {baseline_dsr['num_trials']} trials"
        )
        if baseline_cscv:
            print(
                f"    CSCV/PBO: PBO {baseline_cscv['pbo']:.3f} across {baseline_cscv['num_splits']} splits"
            )
    if best_result is None or best_result_full is None:
        report_payload["selection"]["historical_gate_pass"] = False
        report_path = report_json_path(scope_tag, data_tag)
        write_json(report_payload, report_path)
        print(f"  Metrics JSON written to: {report_path}")
        print("  No ML close-model result was produced.")
        print(f"\nDONE in {time.time() - t0:.0f}s")
        return

    print(
        f"  Best XGBoost historical candidate (search): {best_result['lookback']}mo @ {best_result['timeframe']} "
        f"Sharpe {best_result['sharpe']:.3f} Ret {best_result['total_return_pct']:.2f}% "
        f"| DSR {xgb_prod_dsr['dsr']:.3f} across {xgb_prod_dsr['num_trials']} trials"
    )
    if xgb_candidate_cscv:
        print(
            f"    CSCV/PBO: PBO {xgb_candidate_cscv['pbo']:.3f} across {xgb_candidate_cscv['num_splits']} splits"
        )
    if best_holdout is not None and best_holdout.get("status") == "ok":
        print(
            f"  Holdout XGBoost historical candidate: Sharpe {best_holdout['sharpe']:.3f} "
            f"Ret {best_holdout['total_return_pct']:.2f}% MaxDD {best_holdout['max_dd_pct']:.2f}%"
        )
        stress = best_result_full.get("holdout_stress_tests", {}) if best_result_full else {}
        if stress:
            print("  Holdout cost stress:")
            for label, row in sorted(stress.items()):
                print(
                    f"    {label} slippage -> Sharpe {row.get('sharpe', float('nan')):.3f} "
                    f"Ret {row.get('total_return_pct', float('nan')):.2f}% "
                    f"MaxDD {row.get('max_dd_pct', float('nan')):.2f}%"
                )
    historical_gate_pass, holdout_pass, dsr_pass = historical_holdout_gate(
        best_holdout,
        best_baseline_holdout,
        xgb_prod_dsr["dsr"],
    )
    report_payload["selection"].update(
        {
            "holdout_pass": holdout_pass,
            "dsr_pass": dsr_pass,
            "historical_gate_pass": historical_gate_pass,
        }
    )
    print(f"  Holdout-vs-baseline pass: {holdout_pass}")
    print(f"  Search-period DSR pass (>0.95): {dsr_pass}")
    print(f"  Historical research gate pass: {historical_gate_pass}")

    best_lb = best_result["lookback"]
    best_tf = best_result["timeframe"]
    best_strategy = strategy_variant_tag("xgboost", "residual_triple_barrier", "partial_ladder")
    selected_path = selected_trade_cache_path(
        best_lb,
        best_tf,
        data_tag,
        feat_sig,
        scope_tag,
        best_strategy,
    )
    if selected_path.exists():
        selected = pd.read_parquet(selected_path)
    else:
        combined_path = combined_close_example_cache_path(
            best_lb,
            best_tf,
            data_tag,
            feat_sig,
            scope_tag,
            "residual_triple_barrier",
        )
        if combined_path.exists():
            examples = read_close_example_frame(combined_path, cfg.features.feature_list)
        else:
            examples = pd.DataFrame()
        if len(examples) > 0:
            prediction_cfg = close_strategy_overrides(
                "xgboost",
                "residual_triple_barrier",
                best_tf,
                cfg.close_model,
                de_risk_mode="partial_ladder",
            )
            prediction_rows, validation_rows, prediction_diag = walk_forward_close_prediction_cache(
                examples,
                cfg.features.feature_list,
                best_lb,
                prediction_cfg,
                test_start=TEST_START,
                signal_bar_minutes=15,
                portfolio_cfg=get_pcfg(cfg),
            )
            selected, _ = select_close_predictions(
                prediction_rows,
                validation_rows,
                prediction_cfg,
                signal_bar_minutes=15,
                portfolio_cfg=get_pcfg(cfg),
                prediction_diagnostics=prediction_diag,
            )
        else:
            selected = pd.DataFrame()
        write_parquet(selected, selected_path)

    combined_path = combined_close_example_cache_path(
        best_lb,
        best_tf,
        data_tag,
        feat_sig,
        scope_tag,
        "residual_triple_barrier",
    )
    feature_research = {}
    if cfg.features.feature_research_enabled and combined_path.exists():
        try:
            combined_examples = read_close_example_frame(combined_path, cfg.features.feature_list)
            feature_research = run_feature_research(
                combined_examples,
                cfg.features.feature_list,
                close_strategy_overrides(
                    "xgboost",
                    "residual_triple_barrier",
                    best_tf,
                    cfg.close_model,
                    de_risk_mode="partial_ladder",
                ),
            )
        except Exception:
            feature_research = {}
    if feature_research:
        report_payload["selection"]["feature_research"] = feature_research
        print("  Feature research stable set:")
        print(f"    {feature_research.get('stable_features', [])}")
    holdout_selected = filter_rows_by_period(selected, HOLDOUT_START, None)
    if len(holdout_selected) > 0:
        holdout_stats = evaluate_selected_trades(holdout_selected, pcfg)
        if holdout_stats is not None:
                full_stats_report(
                    f"Best ML Close Model ({best_lb}mo, {best_tf})",
                    {
                        **holdout_stats,
                        "diagnostics": best_result_full.get("holdout_diagnostics", {}) if best_result_full else {},
                    },
                    training_desc=(
                        f"{best_lb}-month rolling, retrained monthly, 2021 bootstrap / "
                        f"search {SEARCH_START}-{SEARCH_END}, holdout {HOLDOUT_START}+"
                    ),
                    pcfg=pcfg,
                )
                report_payload["selection"]["best_xgboost_holdout_full_stats"] = stats_to_jsonable(
                    {
                        **holdout_stats,
                        "diagnostics": best_result_full.get("holdout_diagnostics", {}) if best_result_full else {},
                    }
                )

    report_path = report_json_path(scope_tag, data_tag)
    write_json(report_payload, report_path)
    print(f"  Metrics JSON written to: {report_path}")

    print(f"\nDONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
