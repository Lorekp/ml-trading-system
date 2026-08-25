"""Production-style validation for current cointegration research candidates.

Implements a stricter validation layer on top of cached walk-forward segments:
- event extraction from spread/z-score threshold crossings with a CUSUM filter
- triple-barrier-style trade outcomes with a vertical barrier tied to half-life
- purged and embargoed combinatorial CV on overlap-aware event returns
- uniqueness metrics and sequential-bootstrap Sharpe resampling
- simple spread-stability diagnostics from cached segment state
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import math
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
CACHE_DIR = OUT_DIR / "cache" / "validation" / "production_validate"
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"
INPUT_CANDIDATES = OUT_DIR / "research_candidates_validated.csv"
OUTPUT_CSV = OUT_DIR / "production_validation_candidates.csv"
OUTPUT_MD = OUT_DIR / "production_validation_candidates.md"
PROGRESS_CSV = OUT_DIR / "production_validation_progress.csv"
LOG_PATH = OUT_DIR / "production_validation.log"

LOGGER = logging.getLogger("cointegration_production_validate")


def setup_logging() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def load_rigorous_module():
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    src_str = str(ROOT / "src")
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    spec = importlib.util.spec_from_file_location("exp_f_cointegration_rigorous", RIGOROUS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def safe_key(config_key: str) -> str:
    return (
        config_key.replace("/", "__")
        .replace("|", "_")
        .replace("=", "-")
        .replace(":", "-")
        .replace(".", "p")
    )


def estimate_half_life_bars(series: np.ndarray) -> float:
    clean = np.asarray(series, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 50:
        return float("inf")
    lag = clean[:-1]
    delta = clean[1:] - clean[:-1]
    denom = float(np.dot(lag - lag.mean(), lag - lag.mean()))
    if denom <= 0:
        return float("inf")
    phi = float(np.dot(lag - lag.mean(), delta - delta.mean()) / denom)
    if not np.isfinite(phi) or phi >= 0:
        return float("inf")
    return float(-math.log(2.0) / phi)


def ewm_std(arr: np.ndarray, span: int = 96, min_periods: int = 20) -> np.ndarray:
    ser = pd.Series(np.asarray(arr, dtype=np.float64))
    out = ser.ewm(span=span, adjust=False, min_periods=min_periods).std()
    values = out.to_numpy(dtype=np.float64, copy=True)
    finite = values[np.isfinite(values) & (values > 0)]
    fallback = float(np.median(finite)) if finite.size else float(np.nanstd(arr))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1e-6
    values[~np.isfinite(values)] = fallback
    values[values <= 0] = fallback
    return values


def dynamic_cusum_events(diff_series: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    s_pos = 0.0
    s_neg = 0.0
    hits: list[int] = []
    for idx in range(1, len(diff_series)):
        diff = float(diff_series[idx])
        thr = float(thresholds[idx])
        if not np.isfinite(diff) or not np.isfinite(thr) or thr <= 0:
            continue
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        if s_pos > thr:
            s_pos = 0.0
            hits.append(idx)
        elif s_neg < -thr:
            s_neg = 0.0
            hits.append(idx)
    return np.asarray(sorted(set(hits)), dtype=np.int32)


def near_any(sorted_idx: np.ndarray, value: int, window: int) -> bool:
    if sorted_idx.size == 0:
        return False
    pos = int(np.searchsorted(sorted_idx, value))
    if pos < sorted_idx.size and abs(int(sorted_idx[pos]) - value) <= window:
        return True
    if pos > 0 and abs(int(sorted_idx[pos - 1]) - value) <= window:
        return True
    return False


def annualized_event_sharpe(returns: np.ndarray, weights: np.ndarray, events_per_year: float) -> float:
    ret = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(ret) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return float("nan")
    ret = ret[mask]
    w = w[mask]
    w = w / w.sum()
    mean_ret = float(np.sum(w * ret))
    var_ret = float(np.sum(w * (ret - mean_ret) ** 2))
    std_ret = math.sqrt(var_ret) if var_ret > 0 else 0.0
    if std_ret <= 0 or events_per_year <= 0:
        return float("nan")
    return float(math.sqrt(events_per_year) * mean_ret / std_ret)


def build_bar_panel(segments: list[dict[str, object]]) -> dict[str, np.ndarray]:
    z_chunks = []
    ret_chunks = []
    beta_chunks = []
    month_chunks = []
    for seg in segments:
        z = np.asarray(seg["zscore"], dtype=np.float64)
        hedge_ret = np.asarray(seg["hedge_ret"], dtype=np.float64)
        beta = float(seg["beta"])
        if z.size != hedge_ret.size or z.size < 20:
            continue
        z_chunks.append(z)
        ret_chunks.append(hedge_ret)
        beta_chunks.append(np.full(z.size, beta, dtype=np.float64))
        month_chunks.append(np.full(z.size, str(seg["month"]), dtype=object))
    if not z_chunks:
        return {}
    return {
        "zscore": np.concatenate(z_chunks),
        "hedge_ret": np.concatenate(ret_chunks),
        "beta": np.concatenate(beta_chunks),
        "month": np.concatenate(month_chunks),
    }


def extract_trade_events(
    panel: dict[str, np.ndarray],
    entry_z: float,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    zscore = np.asarray(panel["zscore"], dtype=np.float64)
    hedge_ret = np.asarray(panel["hedge_ret"], dtype=np.float64)
    beta = np.asarray(panel["beta"], dtype=np.float64)
    months = np.asarray(panel["month"], dtype=object)
    n_bars = len(zscore)
    if n_bars < 100:
        return pd.DataFrame(), {}

    z_diff = np.diff(zscore, prepend=zscore[0])
    z_thr = np.maximum(ewm_std(z_diff, span=96, min_periods=20), 0.25)
    cusum_hits = dynamic_cusum_events(z_diff, z_thr)

    crosses = np.flatnonzero(
        (np.abs(zscore) >= entry_z) & np.r_[False, np.abs(zscore[:-1]) < entry_z]
    )
    starts = [idx for idx in crosses if near_any(cusum_hits, int(idx), window=2)]
    if len(starts) < max(4, len(crosses) // 3):
        starts = crosses.tolist()
    starts = sorted(set(int(idx) for idx in starts if idx < n_bars - 2))

    half_life_bars = estimate_half_life_bars(zscore)
    if np.isfinite(half_life_bars):
        max_holding = int(np.clip(round(2.0 * half_life_bars), 8, 96 * 10))
    else:
        max_holding = 96 * 3

    hedge_sigma = ewm_std(hedge_ret, span=96, min_periods=20)
    fallback_sigma = float(np.median(hedge_sigma[np.isfinite(hedge_sigma) & (hedge_sigma > 0)]))
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = max(float(np.nanstd(hedge_ret)), 1e-6)

    rows = []
    reason_counts = {"pt": 0, "sl": 0, "mean_revert": 0, "vertical": 0}
    for start in starts:
        side = -1.0 if zscore[start] > 0 else 1.0
        local_sigma = float(hedge_sigma[start]) if np.isfinite(hedge_sigma[start]) else fallback_sigma
        target = max(local_sigma * math.sqrt(max_holding), fallback_sigma * math.sqrt(max_holding))
        round_trip_cost = 2.0 * (cost_bps / 10_000.0) * (1.0 + abs(float(beta[start])))

        gross_return = 0.0
        end = min(start + max_holding, n_bars - 1)
        exit_reason = "vertical"
        for idx in range(start + 1, end + 1):
            gross_return += side * hedge_ret[idx]
            if gross_return >= target:
                end = idx
                exit_reason = "pt"
                break
            if gross_return <= -target:
                end = idx
                exit_reason = "sl"
                break
            if zscore[idx] == 0.0 or zscore[idx] * zscore[start] <= 0.0:
                end = idx
                exit_reason = "mean_revert"
                break

        net_return = gross_return - round_trip_cost
        reason_counts[exit_reason] += 1
        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "duration_bars": int(end - start),
                "entry_zscore": float(zscore[start]),
                "side": float(side),
                "target_return": float(target),
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "label": int(net_return > 0),
                "exit_reason": exit_reason,
                "month": str(months[start]),
            }
        )

    events = pd.DataFrame(rows)
    meta = {
        "n_bars": int(n_bars),
        "half_life_bars": float(half_life_bars) if np.isfinite(half_life_bars) else float("nan"),
        "max_holding_bars": int(max_holding),
        "cusum_hits": int(cusum_hits.size),
        "threshold_crosses": int(len(crosses)),
        "event_starts": int(len(starts)),
        "reason_pt": int(reason_counts["pt"]),
        "reason_sl": int(reason_counts["sl"]),
        "reason_mean_revert": int(reason_counts["mean_revert"]),
        "reason_vertical": int(reason_counts["vertical"]),
    }
    return events, meta


def build_indicator_matrix(events: pd.DataFrame, n_bars: int) -> np.ndarray:
    ind = np.zeros((n_bars, len(events)), dtype=np.uint8)
    for col, row in enumerate(events.itertuples(index=False)):
        ind[int(row.start) : int(row.end) + 1, col] = 1
    return ind


def indicator_active_indices(indicator: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(indicator[:, col]) for col in range(indicator.shape[1])]


def event_uniqueness(indicator: np.ndarray, active_idx: list[np.ndarray] | None = None) -> np.ndarray:
    if indicator.size == 0:
        return np.asarray([], dtype=np.float64)
    if active_idx is None:
        active_idx = indicator_active_indices(indicator)
    concurrency = indicator.sum(axis=1).astype(np.float64)
    concurrency[concurrency <= 0] = np.nan
    uniq = np.zeros(indicator.shape[1], dtype=np.float64)
    for col, active in enumerate(active_idx):
        values = 1.0 / concurrency[active]
        uniq[col] = float(np.nanmean(values)) if values.size else float("nan")
    return uniq


def sequential_bootstrap_sample(
    active_idx: list[np.ndarray],
    n_bars: int,
    sample_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_events = len(active_idx)
    current_conc = np.zeros(n_bars, dtype=np.float64)
    selected = []
    for _ in range(sample_length):
        scores = np.zeros(n_events, dtype=np.float64)
        for col, active in enumerate(active_idx):
            if active.size == 0:
                continue
            conc_if_added = current_conc[active] + 1.0
            scores[col] = float(np.mean(1.0 / conc_if_added))
        if not np.isfinite(scores).any() or float(scores.sum()) <= 0:
            probs = np.full(n_events, 1.0 / n_events, dtype=np.float64)
        else:
            probs = scores / scores.sum()
        choice = int(rng.choice(n_events, p=probs))
        selected.append(choice)
        current_conc[active_idx[choice]] += 1.0
    return np.asarray(selected, dtype=np.int32)


def choose_cv_scheme(n_events: int) -> tuple[int, int, str]:
    if n_events >= 24:
        return 6, 2, "purged_cpcv_6x2"
    if n_events >= 12:
        return 4, 1, "purged_cpcv_4x1"
    if n_events >= 8:
        return 4, 1, "purged_cpcv_small"
    return 0, 0, "insufficient_events"


def overlaps_with_intervals(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    for left, right in intervals:
        if start <= right and end >= left:
            return True
    return False


def purged_event_cpcv(
    events: pd.DataFrame,
    uniq: np.ndarray,
    n_months: int,
) -> dict[str, float | int | str]:
    n_events = len(events)
    n_groups, k_test, scheme = choose_cv_scheme(n_events)
    if n_groups == 0:
        return {
            "purged_cpcv_scheme": scheme,
            "purged_cpcv_paths": 0,
            "purged_cpcv_mean_sharpe": float("nan"),
            "purged_cpcv_median_sharpe": float("nan"),
            "purged_cpcv_p10_sharpe": float("nan"),
            "purged_cpcv_positive_share": float("nan"),
            "purged_cpcv_mean_gap": float("nan"),
        }

    groups = [chunk.tolist() for chunk in np.array_split(np.arange(n_events), n_groups)]
    combos = list(itertools.combinations(range(n_groups), k_test))
    returns = events["net_return"].to_numpy(dtype=np.float64)
    starts = events["start"].to_numpy(dtype=np.int32)
    ends = events["end"].to_numpy(dtype=np.int32)
    events_per_year = float(n_events / max(n_months, 1) * 12.0)
    embargo = max(8, int(np.nanmedian(events["duration_bars"].to_numpy(dtype=np.float64))))

    test_scores = []
    gaps = []
    for combo in combos:
        test_ids = np.sort(np.concatenate([np.asarray(groups[g], dtype=np.int32) for g in combo]))
        if test_ids.size < 2:
            continue
        test_intervals = [(int(starts[i]), int(ends[i])) for i in test_ids]
        embargo_intervals = [(left, right + embargo) for left, right in test_intervals]

        train_ids = []
        for idx in range(n_events):
            if idx in test_ids:
                continue
            start = int(starts[idx])
            end = int(ends[idx])
            if overlaps_with_intervals(start, end, test_intervals):
                continue
            if overlaps_with_intervals(start, start, embargo_intervals):
                continue
            train_ids.append(idx)
        if len(train_ids) < 2:
            continue

        train_ids_arr = np.asarray(train_ids, dtype=np.int32)
        train_score = annualized_event_sharpe(returns[train_ids_arr], uniq[train_ids_arr], events_per_year)
        test_score = annualized_event_sharpe(returns[test_ids], uniq[test_ids], events_per_year)
        if np.isfinite(test_score):
            test_scores.append(float(test_score))
            if np.isfinite(train_score):
                gaps.append(float(test_score - train_score))

    arr = np.asarray(test_scores, dtype=np.float64)
    gaps_arr = np.asarray(gaps, dtype=np.float64)
    return {
        "purged_cpcv_scheme": scheme,
        "purged_cpcv_paths": int(arr.size),
        "purged_cpcv_mean_sharpe": float(np.mean(arr)) if arr.size else float("nan"),
        "purged_cpcv_median_sharpe": float(np.median(arr)) if arr.size else float("nan"),
        "purged_cpcv_p10_sharpe": float(np.percentile(arr, 10)) if arr.size >= 3 else float("nan"),
        "purged_cpcv_positive_share": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
        "purged_cpcv_mean_gap": float(np.mean(gaps_arr)) if gaps_arr.size else float("nan"),
    }


def sequential_bootstrap_stats(
    indicator: np.ndarray,
    returns: np.ndarray,
    n_months: int,
    seed: int = 7,
    n_boot: int = 100,
) -> dict[str, float]:
    if indicator.size == 0 or indicator.shape[1] < 2:
        return {
            "seq_boot_mean_sharpe": float("nan"),
            "seq_boot_p10_sharpe": float("nan"),
            "seq_boot_positive_share": float("nan"),
        }

    rng = np.random.default_rng(seed)
    active_idx = indicator_active_indices(indicator)
    sample_length = indicator.shape[1]
    events_per_year = float(indicator.shape[1] / max(n_months, 1) * 12.0)
    if sample_length >= 150:
        n_boot = min(n_boot, 80)
    sharpe_vals = []
    for _ in range(n_boot):
        sample_idx = sequential_bootstrap_sample(
            active_idx=active_idx,
            n_bars=indicator.shape[0],
            sample_length=sample_length,
            rng=rng,
        )
        sampled_returns = returns[sample_idx]
        sampled_weights = np.full(sample_idx.size, 1.0 / sample_idx.size, dtype=np.float64)
        score = annualized_event_sharpe(sampled_returns, sampled_weights, events_per_year)
        if np.isfinite(score):
            sharpe_vals.append(float(score))
    arr = np.asarray(sharpe_vals, dtype=np.float64)
    return {
        "seq_boot_mean_sharpe": float(np.mean(arr)) if arr.size else float("nan"),
        "seq_boot_p10_sharpe": float(np.percentile(arr, 10)) if arr.size >= 3 else float("nan"),
        "seq_boot_positive_share": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
    }


def stability_metrics(segments: list[dict[str, object]]) -> dict[str, float]:
    betas = np.asarray([float(seg["beta"]) for seg in segments], dtype=np.float64)
    betas = betas[np.isfinite(betas) & (betas > 0)]
    if betas.size:
        beta_ratio = float(np.max(betas) / np.min(betas))
        beta_cv = float(np.std(betas, ddof=1) / np.mean(betas)) if betas.size > 1 else 0.0
    else:
        beta_ratio = float("nan")
        beta_cv = float("nan")

    hl_values = []
    for seg in segments:
        hl = estimate_half_life_bars(np.asarray(seg["zscore"], dtype=np.float64))
        if np.isfinite(hl):
            hl_values.append(float(hl))
    hl_arr = np.asarray(hl_values, dtype=np.float64)
    if hl_arr.size:
        hl_median = float(np.median(hl_arr))
        hl_cv = float(np.std(hl_arr, ddof=1) / np.mean(hl_arr)) if hl_arr.size > 1 else 0.0
    else:
        hl_median = float("nan")
        hl_cv = float("nan")

    return {
        "beta_ratio_monthly": beta_ratio,
        "beta_cv_monthly": beta_cv,
        "half_life_median_bars": hl_median,
        "half_life_cv": hl_cv,
    }


def production_stage(row: dict[str, object]) -> str:
    if (
        row["n_events"] >= 20
        and row["purged_cpcv_mean_sharpe"] > 1.0
        and row["purged_cpcv_p10_sharpe"] > 0.0
        and row["seq_boot_p10_sharpe"] > 0.0
        and row["event_positive_rate"] >= 0.55
        and row["avg_uniqueness"] >= 0.75
        and row["beta_cv_monthly"] <= 0.35
        and row["half_life_cv"] <= 1.0
        and row["dsr_prob"] >= 0.50
        and row["base_max_dd_pct"] <= 20.0
    ):
        return "Paper Trade Candidate"
    if (
        row["purged_cpcv_mean_sharpe"] > 0.0
        and row["seq_boot_positive_share"] >= 0.60
        and row["event_positive_rate"] >= 0.50
    ):
        return "Research Live Watch"
    if row["n_events"] >= 8:
        return "Research Hold"
    return "Reject for now"


def validate_candidate(row: pd.Series, rigorous) -> dict[str, object]:
    config_key = str(row["config_key"])
    cache_path = CACHE_DIR / f"{safe_key(config_key)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["cache_hit"] = True
        return cached

    segments = rigorous.prepare_segments(row.to_dict(), int(row["lookback_months"]))
    panel = build_bar_panel(segments)
    if not panel:
        result = {
            **row.to_dict(),
            "cache_hit": False,
            "n_events": 0,
            "event_positive_rate": float("nan"),
            "avg_uniqueness": float("nan"),
            "effective_n_events": float("nan"),
            "event_mean_return_pct": float("nan"),
            "event_sharpe": float("nan"),
            "purged_cpcv_scheme": "no_data",
            "purged_cpcv_paths": 0,
            "purged_cpcv_mean_sharpe": float("nan"),
            "purged_cpcv_median_sharpe": float("nan"),
            "purged_cpcv_p10_sharpe": float("nan"),
            "purged_cpcv_positive_share": float("nan"),
            "purged_cpcv_mean_gap": float("nan"),
            "seq_boot_mean_sharpe": float("nan"),
            "seq_boot_p10_sharpe": float("nan"),
            "seq_boot_positive_share": float("nan"),
            "beta_ratio_monthly": float("nan"),
            "beta_cv_monthly": float("nan"),
            "half_life_median_bars": float("nan"),
            "half_life_cv": float("nan"),
            "event_half_life_bars": float("nan"),
            "max_holding_bars": float("nan"),
            "cusum_hits": 0,
            "threshold_crosses": 0,
            "event_starts": 0,
            "reason_pt": 0,
            "reason_sl": 0,
            "reason_mean_revert": 0,
            "reason_vertical": 0,
            "production_stage": "Reject for now",
        }
        with cache_path.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    events, meta = extract_trade_events(panel, float(row["entry_z"]), float(row["cost_bps"]))
    if events.empty:
        result = {
            **row.to_dict(),
            "cache_hit": False,
            "n_events": 0,
            "event_positive_rate": float("nan"),
            "avg_uniqueness": float("nan"),
            "effective_n_events": float("nan"),
            "event_mean_return_pct": float("nan"),
            "event_sharpe": float("nan"),
            "purged_cpcv_scheme": "insufficient_events",
            "purged_cpcv_paths": 0,
            "purged_cpcv_mean_sharpe": float("nan"),
            "purged_cpcv_median_sharpe": float("nan"),
            "purged_cpcv_p10_sharpe": float("nan"),
            "purged_cpcv_positive_share": float("nan"),
            "purged_cpcv_mean_gap": float("nan"),
            "seq_boot_mean_sharpe": float("nan"),
            "seq_boot_p10_sharpe": float("nan"),
            "seq_boot_positive_share": float("nan"),
            **stability_metrics(segments),
            "event_half_life_bars": meta.get("half_life_bars", float("nan")),
            "max_holding_bars": meta.get("max_holding_bars", float("nan")),
            "cusum_hits": meta.get("cusum_hits", 0),
            "threshold_crosses": meta.get("threshold_crosses", 0),
            "event_starts": meta.get("event_starts", 0),
            "reason_pt": meta.get("reason_pt", 0),
            "reason_sl": meta.get("reason_sl", 0),
            "reason_mean_revert": meta.get("reason_mean_revert", 0),
            "reason_vertical": meta.get("reason_vertical", 0),
            "production_stage": "Reject for now",
        }
        with cache_path.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    indicator = build_indicator_matrix(events, int(meta["n_bars"]))
    uniq = event_uniqueness(indicator, active_idx=indicator_active_indices(indicator))
    returns = events["net_return"].to_numpy(dtype=np.float64)
    events_per_year = float(len(events) / max(int(row["n_months"]), 1) * 12.0)
    stability = stability_metrics(segments)
    cpcv = purged_event_cpcv(events, uniq, int(row["n_months"]))
    seq_boot = sequential_bootstrap_stats(indicator, returns, int(row["n_months"]))

    result = {
        **row.to_dict(),
        "cache_hit": False,
        "n_events": int(len(events)),
        "event_positive_rate": float(np.mean(events["label"].to_numpy(dtype=np.float64))),
        "avg_uniqueness": float(np.nanmean(uniq)) if uniq.size else float("nan"),
        "effective_n_events": float(np.nansum(uniq)) if uniq.size else float("nan"),
        "event_mean_return_pct": float(100.0 * np.nanmean(returns)),
        "event_sharpe": annualized_event_sharpe(
            returns,
            uniq if uniq.size else np.ones_like(returns),
            events_per_year,
        ),
        **cpcv,
        **seq_boot,
        **stability,
        "event_half_life_bars": meta.get("half_life_bars", float("nan")),
        "max_holding_bars": meta.get("max_holding_bars", float("nan")),
        "cusum_hits": meta.get("cusum_hits", 0),
        "threshold_crosses": meta.get("threshold_crosses", 0),
        "event_starts": meta.get("event_starts", 0),
        "reason_pt": meta.get("reason_pt", 0),
        "reason_sl": meta.get("reason_sl", 0),
        "reason_mean_revert": meta.get("reason_mean_revert", 0),
        "reason_vertical": meta.get("reason_vertical", 0),
    }
    result["production_stage"] = production_stage(result)

    with cache_path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def build_markdown(results: pd.DataFrame) -> str:
    lines = [
        "# Production Validation",
        "",
        "- Event-driven spread validation with CUSUM-filtered threshold crossings",
        "- Triple-barrier-style exits using profit, stop, mean-reversion, and vertical barriers",
        "- Purged and embargoed combinatorial CV on overlap-aware event returns",
        "- Sequential bootstrap Sharpe resampling and spread-stability diagnostics",
        "",
        "## Results",
        "",
    ]
    display = results.sort_values(
        ["production_stage", "purged_cpcv_mean_sharpe", "seq_boot_p10_sharpe"],
        ascending=[True, False, False],
    )
    for row in display.itertuples(index=False):
        lines.append(
            f"- `{row.pair}` | stage={row.production_stage} | events={row.n_events} | "
            f"purged CPCV mean={row.purged_cpcv_mean_sharpe:.2f} | "
            f"CPCV p10={row.purged_cpcv_p10_sharpe:.2f} | "
            f"seq boot p10={row.seq_boot_p10_sharpe:.2f} | "
            f"avg uniqueness={row.avg_uniqueness:.2f} | "
            f"beta cv={row.beta_cv_monthly:.2f} | hl cv={row.half_life_cv:.2f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    LOGGER.info("starting production validation | input=%s", INPUT_CANDIDATES)
    candidates = pd.read_csv(INPUT_CANDIDATES)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        rigorous = load_rigorous_module()
        rigorous.ensure_dirs()
        rigorous._FULL_CLOSE_DF = rigorous.build_universe()

        rows: list[dict[str, object]] = []
        workers = min(4, max(1, (os.cpu_count() or 4) - 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(validate_candidate, row, rigorous): str(row["pair"])
                for _, row in candidates.iterrows()
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                pair = futures[future]
                result = future.result()
                rows.append(result)
                pd.DataFrame(rows).to_csv(PROGRESS_CSV, index=False)
                LOGGER.info(
                    "validated candidate | %s/%s | pair=%s | source=%s | events=%s | cpcv_mean=%.2f | stage=%s",
                    idx,
                    len(futures),
                    pair,
                    "cache" if result.get("cache_hit") else "compute",
                    result["n_events"],
                    result["purged_cpcv_mean_sharpe"] if np.isfinite(result["purged_cpcv_mean_sharpe"]) else float("nan"),
                    result["production_stage"],
                )
    finally:
        os.chdir(original_cwd)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["production_stage", "purged_cpcv_mean_sharpe", "seq_boot_p10_sharpe"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    results.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_MD.write_text(build_markdown(results), encoding="utf-8")
    LOGGER.info("production validation complete | output=%s | rows=%s", OUTPUT_CSV, len(results))


if __name__ == "__main__":
    main()
