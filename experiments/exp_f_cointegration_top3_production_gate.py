"""Top-3 cointegration production gate.

Freezes the research universe to the top 3 validated configs and runs:
1. pair-level execution-stress validation
2. portfolio-level validation for the 3-pair book

The script reuses cached segment state from exp_f_cointegration_rigorous.py
and persists per-pair/per-scenario caches so reruns avoid recomputation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
CACHE_DIR = OUT_DIR / "cache" / "top3_production_gate"
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"
LOG_PATH = OUT_DIR / "top3_production_gate.log"

PAIR_OUTPUT_CSV = OUT_DIR / "top3_production_gate_pairs.csv"
PAIR_OUTPUT_JSON = OUT_DIR / "top3_production_gate_pairs.json"
PORT_OUTPUT_CSV = OUT_DIR / "top3_production_gate_portfolio.csv"
PORT_OUTPUT_JSON = OUT_DIR / "top3_production_gate_portfolio.json"
MONTHLY_MATRIX_CSV = OUT_DIR / "top3_production_gate_monthly_matrix.csv"
REPORT_MD = OUT_DIR / "top3_production_gate_report.md"

CACHE_VERSION = 1
LOGGER = logging.getLogger("top3_production_gate")

TOP3_CONFIGS = [
    {
        "pair": "ENA/LTC",
        "y_name": "LTC",
        "x_name": "ENA",
        "lookback_months": 2,
        "entry_z": 1.5,
        "cost_bps": 2.0,
        "config_key": "ENA/LTC|lb=2|z=1.5|c=2.0",
    },
    {
        "pair": "TAO/FET",
        "y_name": "TAO",
        "x_name": "FET",
        "lookback_months": 1,
        "entry_z": 2.0,
        "cost_bps": 10.0,
        "config_key": "TAO/FET|lb=1|z=2.0|c=10.0",
    },
    {
        "pair": "LINK/ENA",
        "y_name": "LINK",
        "x_name": "ENA",
        "lookback_months": 4,
        "entry_z": 1.5,
        "cost_bps": 2.0,
        "config_key": "LINK/ENA|lb=4|z=1.5|c=2.0",
    },
]

SCENARIOS = [
    {
        "scenario": "baseline_current",
        "fill_probability": 1.00,
        "extra_slippage_bps": 0.0,
        "entry_latency_bars": 0,
        "exit_latency_bars": 0,
    },
    {
        "scenario": "maker_good",
        "fill_probability": 0.90,
        "extra_slippage_bps": 1.0,
        "entry_latency_bars": 1,
        "exit_latency_bars": 1,
    },
    {
        "scenario": "maker_realistic",
        "fill_probability": 0.75,
        "extra_slippage_bps": 2.0,
        "entry_latency_bars": 1,
        "exit_latency_bars": 1,
    },
    {
        "scenario": "maker_harsh",
        "fill_probability": 0.60,
        "extra_slippage_bps": 3.0,
        "entry_latency_bars": 2,
        "exit_latency_bars": 2,
    },
    {
        "scenario": "forced_taker",
        "fill_probability": 1.00,
        "extra_slippage_bps": 5.0,
        "entry_latency_bars": 0,
        "exit_latency_bars": 0,
    },
]
SCENARIO_ORDER = {item["scenario"]: idx for idx, item in enumerate(SCENARIOS)}


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


def stable_seed(pair: str, scenario: str) -> int:
    digest = hashlib.sha256(f"{pair}|{scenario}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def safe_key(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("|", "_")
        .replace("=", "-")
        .replace(":", "-")
        .replace(".", "p")
    )


def annualized_sharpe(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    if std_ret <= 0 or not np.isfinite(std_ret):
        return float("nan")
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def summarize_monthly_rows(monthly_rows: list[dict[str, object]]) -> dict[str, float | int]:
    monthly = pd.DataFrame(monthly_rows)
    pnl_arr = monthly["pnl"].to_numpy(dtype=np.float64) if not monthly.empty else np.asarray([], dtype=np.float64)
    total_return_pct = float(100.0 * pnl_arr.sum()) if pnl_arr.size else float("nan")
    equity = np.cumsum(pnl_arr) if pnl_arr.size else np.asarray([], dtype=np.float64)
    peak = np.maximum.accumulate(equity) if equity.size else np.asarray([], dtype=np.float64)
    max_dd_pct = float(100.0 * np.max(peak - equity)) if equity.size else float("nan")
    worst_month_pct = float(100.0 * np.min(pnl_arr)) if pnl_arr.size else float("nan")
    return {
        "annualized_sharpe": annualized_sharpe(pnl_arr),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "positive_months_pct": float(100.0 * np.mean(pnl_arr > 0.0)) if pnl_arr.size else float("nan"),
        "worst_month_pct": worst_month_pct,
        "n_months": int(pnl_arr.size),
    }


def scenario_cache_path(pair: str, scenario: str) -> Path:
    return CACHE_DIR / f"{safe_key(pair)}__{scenario}.pkl"


def portfolio_cpcv(monthly_series: pd.Series) -> dict[str, object]:
    series = monthly_series.dropna().astype(float)
    months = series.index.tolist()
    n_months = len(months)
    if n_months >= 12:
        n_groups, k_test, scheme = 6, 2, "6x2"
    elif n_months >= 8:
        n_groups, k_test, scheme = 4, 1, "4x1"
    else:
        return {
            "portfolio_cv_scheme": "insufficient",
            "portfolio_cv_paths": 0,
            "portfolio_oos_mean_sharpe": float("nan"),
            "portfolio_oos_median_sharpe": float("nan"),
            "portfolio_oos_p10_sharpe": float("nan"),
            "portfolio_oos_positive_share": float("nan"),
        }

    groups = [chunk.tolist() for chunk in np.array_split(np.arange(n_months), n_groups)]
    scores = []
    for combo in __import__("itertools").combinations(range(n_groups), k_test):
        test_idx = np.sort(np.concatenate([np.asarray(groups[g], dtype=np.int32) for g in combo]))
        test_vals = series.iloc[test_idx].to_numpy(dtype=np.float64)
        score = annualized_sharpe(test_vals)
        if np.isfinite(score):
            scores.append(float(score))
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "portfolio_cv_scheme": scheme,
        "portfolio_cv_paths": int(arr.size),
        "portfolio_oos_mean_sharpe": float(np.mean(arr)) if arr.size else float("nan"),
        "portfolio_oos_median_sharpe": float(np.median(arr)) if arr.size else float("nan"),
        "portfolio_oos_p10_sharpe": float(np.percentile(arr, 10)) if arr.size >= 3 else float("nan"),
        "portfolio_oos_positive_share": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
    }


def simulate_scenario(
    segments: list[dict[str, object]],
    entry_z: float,
    base_cost_bps: float,
    scenario: dict[str, object],
    pair: str,
) -> dict[str, object]:
    rng = np.random.default_rng(stable_seed(pair, str(scenario["scenario"])))
    monthly_rows = []
    executed_entries = 0
    skipped_entries = 0
    entry_signals = 0
    trade_returns = []

    for seg in segments:
        zscore = np.asarray(seg["zscore"], dtype=np.float64)
        hedge_ret = np.asarray(seg["hedge_ret"], dtype=np.float64)
        beta = float(seg["beta"])
        month = str(seg["month"])
        if zscore.size != hedge_ret.size or zscore.size < 2:
            continue

        side_cost = (base_cost_bps + float(scenario["extra_slippage_bps"])) / 10_000.0 * (1.0 + abs(beta))
        month_pnl = 0.0
        position = 0
        pending_entry: dict[str, int | float] | None = None
        pending_exit_idx: int | None = None
        trade_pnl = 0.0

        for idx in range(1, len(zscore)):
            if position != 0:
                pnl_delta = position * hedge_ret[idx]
                month_pnl += pnl_delta
                trade_pnl += pnl_delta

            if position != 0 and pending_exit_idx is not None and idx >= pending_exit_idx:
                month_pnl -= side_cost
                trade_pnl -= side_cost
                trade_returns.append(float(trade_pnl))
                trade_pnl = 0.0
                position = 0
                pending_exit_idx = None

            if position == 0 and pending_entry is not None and idx >= int(pending_entry["execute_idx"]):
                position = int(pending_entry["side"])
                month_pnl -= side_cost
                trade_pnl = -side_cost
                executed_entries += 1
                pending_entry = None

            z = zscore[idx]
            if position == 0 and pending_entry is None:
                if z <= -entry_z or z >= entry_z:
                    entry_signals += 1
                    side = 1 if z <= -entry_z else -1
                    if rng.random() <= float(scenario["fill_probability"]):
                        execute_idx = min(idx + int(scenario["entry_latency_bars"]), len(zscore) - 1)
                        if execute_idx <= idx:
                            position = side
                            month_pnl -= side_cost
                            trade_pnl = -side_cost
                            executed_entries += 1
                        else:
                            pending_entry = {"side": side, "execute_idx": execute_idx}
                    else:
                        skipped_entries += 1
            elif position != 0 and pending_exit_idx is None:
                if (position == 1 and z >= 0.0) or (position == -1 and z <= 0.0):
                    execute_idx = min(idx + int(scenario["exit_latency_bars"]), len(zscore) - 1)
                    if execute_idx <= idx:
                        month_pnl -= side_cost
                        trade_pnl -= side_cost
                        trade_returns.append(float(trade_pnl))
                        trade_pnl = 0.0
                        position = 0
                    else:
                        pending_exit_idx = execute_idx

        if position != 0:
            month_pnl -= side_cost
            trade_pnl -= side_cost
            trade_returns.append(float(trade_pnl))

        monthly_rows.append({"month": month, "pnl": float(month_pnl)})

    monthly = pd.DataFrame(monthly_rows)
    pnl_arr = monthly["pnl"].to_numpy(dtype=np.float64) if not monthly.empty else np.asarray([], dtype=np.float64)
    total_return_pct = float(100.0 * pnl_arr.sum()) if pnl_arr.size else float("nan")
    equity = np.cumsum(pnl_arr) if pnl_arr.size else np.asarray([], dtype=np.float64)
    peak = np.maximum.accumulate(equity) if equity.size else np.asarray([], dtype=np.float64)
    max_dd_pct = float(100.0 * np.max(peak - equity)) if equity.size else float("nan")
    worst_month_pct = float(100.0 * np.min(pnl_arr)) if pnl_arr.size else float("nan")
    avg_trade_return_pct = float(100.0 * np.mean(trade_returns)) if trade_returns else float("nan")
    fill_rate = float(executed_entries / entry_signals) if entry_signals > 0 else float("nan")

    result = {
        "scenario": str(scenario["scenario"]),
        "fill_probability": float(scenario["fill_probability"]),
        "extra_slippage_bps": float(scenario["extra_slippage_bps"]),
        "entry_latency_bars": int(scenario["entry_latency_bars"]),
        "exit_latency_bars": int(scenario["exit_latency_bars"]),
        "monthly_pnl": monthly_rows,
        "annualized_sharpe": annualized_sharpe(pnl_arr),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd_pct,
        "positive_months_pct": float(100.0 * np.mean(pnl_arr > 0.0)) if pnl_arr.size else float("nan"),
        "total_trades": int(len(trade_returns)),
        "executed_entries": int(executed_entries),
        "skipped_entries": int(skipped_entries),
        "fill_rate": fill_rate,
        "average_trade_return_pct": avg_trade_return_pct,
        "worst_month_pct": worst_month_pct,
        "n_months": int(pnl_arr.size),
    }
    return result


def pair_scenario_pass(scenario_name: str, metrics: dict[str, object]) -> bool:
    if scenario_name == "baseline_current":
        return True
    if scenario_name in {"maker_good", "maker_realistic"}:
        return bool(np.isfinite(metrics["annualized_sharpe"]) and float(metrics["annualized_sharpe"]) > 1.0)
    return bool(np.isfinite(metrics["total_return_pct"]) and float(metrics["total_return_pct"]) > 0.0)


def apply_conservative_stress(result: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    if result["scenario"] == "baseline_current":
        return result

    monthly_rows = [dict(item) for item in result["monthly_pnl"]]
    for key in [
        "annualized_sharpe",
        "total_return_pct",
        "max_drawdown_pct",
        "positive_months_pct",
        "average_trade_return_pct",
        "worst_month_pct",
    ]:
        result[f"raw_{key}"] = result[key]

    raw_total = float(result["total_return_pct"])
    base_total = float(baseline["total_return_pct"])
    if np.isfinite(raw_total) and np.isfinite(base_total) and raw_total > base_total and monthly_rows:
        excess = (raw_total - base_total) / 100.0
        penalty = excess / len(monthly_rows)
        for row in monthly_rows:
            row["pnl"] = float(row["pnl"]) - penalty

    monthly_summary = summarize_monthly_rows(monthly_rows)
    result["monthly_pnl"] = monthly_rows
    result["annualized_sharpe"] = min(
        float(monthly_summary["annualized_sharpe"]),
        float(baseline["annualized_sharpe"]),
    )
    result["total_return_pct"] = min(
        float(monthly_summary["total_return_pct"]),
        float(baseline["total_return_pct"]),
    )
    result["max_drawdown_pct"] = max(
        float(monthly_summary["max_drawdown_pct"]),
        float(baseline["max_drawdown_pct"]),
    )
    result["positive_months_pct"] = min(
        float(monthly_summary["positive_months_pct"]),
        float(baseline["positive_months_pct"]),
    )
    result["worst_month_pct"] = min(
        float(monthly_summary["worst_month_pct"]),
        float(baseline["worst_month_pct"]),
    )
    if np.isfinite(result["average_trade_return_pct"]) and np.isfinite(baseline["average_trade_return_pct"]):
        result["average_trade_return_pct"] = min(
            float(result["average_trade_return_pct"]),
            float(baseline["average_trade_return_pct"]),
        )
    result["n_months"] = int(monthly_summary["n_months"])
    return result


def get_pair_result(config: dict[str, object], scenario: dict[str, object], rigorous) -> dict[str, object]:
    cache_path = scenario_cache_path(str(config["pair"]), str(scenario["scenario"]))
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        if cached.get("cache_version") == CACHE_VERSION:
            cached["cache_hit"] = True
            return cached

    segments = rigorous.prepare_segments(config, int(config["lookback_months"]))
    simulated = simulate_scenario(
        segments=segments,
        entry_z=float(config["entry_z"]),
        base_cost_bps=float(config["cost_bps"]),
        scenario=scenario,
        pair=str(config["pair"]),
    )
    result = {
        **config,
        **simulated,
        "scenario_pass": pair_scenario_pass(str(scenario["scenario"]), simulated),
        "cache_hit": False,
        "cache_version": CACHE_VERSION,
    }
    with cache_path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def monthly_matrix_from_pair_results(pair_results: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = []
    scenario_pair_frames: dict[str, list[pd.DataFrame]] = {}
    for result in pair_results:
        scenario = str(result["scenario"])
        pair = str(result["pair"])
        monthly = pd.DataFrame(result["monthly_pnl"])
        if monthly.empty:
            continue
        monthly["month"] = monthly["month"].astype(str)
        pair_col = pair
        monthly = monthly.rename(columns={"pnl": pair_col})[["month", pair_col]]
        scenario_pair_frames.setdefault(scenario, []).append(monthly.set_index("month"))

        wide_col = f"{scenario}|{pair}"
        frame = monthly.rename(columns={pair_col: wide_col}).set_index("month")
        frames.append(frame)

    wide_matrix = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    by_scenario = {}
    for scenario, matrices in scenario_pair_frames.items():
        by_scenario[scenario] = pd.concat(matrices, axis=1).sort_index()
    return wide_matrix, by_scenario


def mean_pair_corr(matrix: pd.DataFrame) -> float:
    if matrix.shape[1] < 2:
        return float("nan")
    corr = matrix.corr(min_periods=3)
    vals = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    return float(vals.mean()) if not vals.empty else float("nan")


def portfolio_series_from_matrix(matrix: pd.DataFrame) -> pd.Series:
    series = matrix.mean(axis=1, skipna=True)
    series = series.dropna()
    series.name = "portfolio_pnl"
    return series


def portfolio_stats(series: pd.Series) -> dict[str, float]:
    arr = series.to_numpy(dtype=np.float64)
    total_return_pct = float(100.0 * arr.sum()) if arr.size else float("nan")
    sharpe = annualized_sharpe(arr)
    equity = np.cumsum(arr) if arr.size else np.asarray([], dtype=np.float64)
    peak = np.maximum.accumulate(equity) if equity.size else np.asarray([], dtype=np.float64)
    max_dd = float(100.0 * np.max(peak - equity)) if equity.size else float("nan")
    positive = float(100.0 * np.mean(arr > 0.0)) if arr.size else float("nan")
    return {
        "portfolio_sharpe": sharpe,
        "portfolio_total_return_pct": total_return_pct,
        "portfolio_max_drawdown_pct": max_dd,
        "portfolio_positive_months_pct": positive,
        "portfolio_n_months": int(arr.size),
    }


def contribution_metrics(matrix: pd.DataFrame, full_series: pd.Series, pair: str) -> dict[str, float]:
    active_counts = matrix.notna().sum(axis=1).replace(0, np.nan)
    pair_contrib = matrix[pair].divide(active_counts).fillna(0.0)
    full_total = float(full_series.sum())
    contrib_total = float(pair_contrib.sum())
    share_pct = float(100.0 * contrib_total / full_total) if full_total != 0 else float("nan")

    loo_cols = [col for col in matrix.columns if col != pair]
    loo_series = portfolio_series_from_matrix(matrix[loo_cols]) if loo_cols else pd.Series(dtype=float)
    full_stats = portfolio_stats(full_series)
    loo_stats = portfolio_stats(loo_series) if not loo_series.empty else {
        "portfolio_sharpe": float("nan"),
        "portfolio_max_drawdown_pct": float("nan"),
    }
    standalone = matrix[pair].dropna()
    return {
        "standalone_sharpe": annualized_sharpe(standalone.to_numpy(dtype=np.float64)),
        "portfolio_marginal_sharpe": float(full_stats["portfolio_sharpe"] - loo_stats["portfolio_sharpe"])
        if np.isfinite(full_stats["portfolio_sharpe"]) and np.isfinite(loo_stats["portfolio_sharpe"])
        else float("nan"),
        "portfolio_marginal_drawdown_pct": float(full_stats["portfolio_max_drawdown_pct"] - loo_stats["portfolio_max_drawdown_pct"])
        if np.isfinite(full_stats["portfolio_max_drawdown_pct"]) and np.isfinite(loo_stats["portfolio_max_drawdown_pct"])
        else float("nan"),
        "portfolio_contribution_share_pct": share_pct,
    }


def portfolio_scenario_pass(scenario: str, full_row: dict[str, object], contribution_rows: list[dict[str, object]]) -> tuple[bool, bool]:
    concentration_fail = False
    if np.isfinite(full_row["portfolio_total_return_pct"]) and float(full_row["portfolio_total_return_pct"]) > 0:
        for row in contribution_rows:
            share = row["portfolio_contribution_share_pct"]
            if np.isfinite(share) and abs(float(share)) > 70.0:
                concentration_fail = True
                break

    if scenario == "maker_realistic":
        scenario_pass = (
            np.isfinite(full_row["portfolio_oos_mean_sharpe"])
            and float(full_row["portfolio_oos_mean_sharpe"]) > 1.0
            and np.isfinite(full_row["portfolio_oos_p10_sharpe"])
            and float(full_row["portfolio_oos_p10_sharpe"]) > 0.0
            and not concentration_fail
        )
    elif scenario in {"maker_good", "maker_harsh", "forced_taker"}:
        scenario_pass = bool(
            np.isfinite(full_row["portfolio_total_return_pct"]) and float(full_row["portfolio_total_return_pct"]) > 0.0
        ) and not concentration_fail
    else:
        scenario_pass = True
    return scenario_pass, concentration_fail


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    use_df = df.loc[:, columns].copy()
    headers = list(use_df.columns)
    rows = [headers]
    for row in use_df.itertuples(index=False):
        rows.append([str(value) for value in row])
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    lines = [fmt(rows[0]), sep]
    for row in rows[1:]:
        lines.append(fmt(row))
    return "\n".join(lines)


def build_report(
    pair_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    top3_df: pd.DataFrame,
    final_recommendation: str,
) -> str:
    pair_display = pair_df.copy()
    pair_display["annualized_sharpe"] = pair_display["annualized_sharpe"].round(2)
    pair_display["total_return_pct"] = pair_display["total_return_pct"].round(2)
    pair_display["max_drawdown_pct"] = pair_display["max_drawdown_pct"].round(2)
    pair_display["fill_rate"] = pair_display["fill_rate"].round(3)
    pair_display["implementation_shortfall_pct"] = pair_display["implementation_shortfall_pct"].round(2)

    shortfall_df = pair_display[["pair", "scenario", "implementation_shortfall_pct", "scenario_pass"]].copy()

    portfolio_display = portfolio_df.loc[portfolio_df["row_type"] == "portfolio_full"].copy()
    portfolio_display["portfolio_sharpe"] = portfolio_display["portfolio_sharpe"].round(2)
    portfolio_display["portfolio_total_return_pct"] = portfolio_display["portfolio_total_return_pct"].round(2)
    portfolio_display["portfolio_max_drawdown_pct"] = portfolio_display["portfolio_max_drawdown_pct"].round(2)
    portfolio_display["portfolio_oos_mean_sharpe"] = portfolio_display["portfolio_oos_mean_sharpe"].round(2)
    portfolio_display["portfolio_oos_p10_sharpe"] = portfolio_display["portfolio_oos_p10_sharpe"].round(2)

    loo_display = portfolio_df.loc[portfolio_df["row_type"] == "portfolio_loo"].copy()
    loo_display["portfolio_sharpe"] = loo_display["portfolio_sharpe"].round(2)
    loo_display["portfolio_total_return_pct"] = loo_display["portfolio_total_return_pct"].round(2)
    loo_display["portfolio_max_drawdown_pct"] = loo_display["portfolio_max_drawdown_pct"].round(2)

    lines = [
        "# Top-3 Cointegration Production Gate",
        "",
        f"Final recommendation: **{final_recommendation}**",
        "",
        "## Frozen Top 3",
        "",
        markdown_table(
            top3_df,
            ["pair", "y_name", "x_name", "lookback_months", "entry_z", "cost_bps", "config_key"],
        ),
        "",
        "## Pair-Level Scenario Results",
        "",
        markdown_table(
            pair_display,
            [
                "pair",
                "scenario",
                "annualized_sharpe",
                "total_return_pct",
                "max_drawdown_pct",
                "fill_rate",
                "scenario_pass",
            ],
        ),
        "",
        "## Implementation Shortfall",
        "",
        markdown_table(shortfall_df, ["pair", "scenario", "implementation_shortfall_pct", "scenario_pass"]),
        "",
        "## Portfolio Scenarios",
        "",
        markdown_table(
            portfolio_display,
            [
                "scenario",
                "portfolio_sharpe",
                "portfolio_total_return_pct",
                "portfolio_max_drawdown_pct",
                "portfolio_oos_mean_sharpe",
                "portfolio_oos_p10_sharpe",
                "scenario_pass",
                "concentration_fail",
            ],
        ),
        "",
        "## Leave-One-Out",
        "",
        markdown_table(
            loo_display,
            ["scenario", "variant", "portfolio_sharpe", "portfolio_total_return_pct", "portfolio_max_drawdown_pct"],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    top3_df = pd.DataFrame(TOP3_CONFIGS)

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        rigorous = load_rigorous_module()
        rigorous.ensure_dirs()
        rigorous._FULL_CLOSE_DF = rigorous.build_universe()

        pair_results = []
        for config in TOP3_CONFIGS:
            LOGGER.info("pair start | pair=%s", config["pair"])
            pair_specific_results = []
            baseline_result = None
            for scenario in SCENARIOS:
                result = get_pair_result(config, scenario, rigorous)
                if result["scenario"] == "baseline_current":
                    baseline_result = result
                pair_specific_results.append(result)
            assert baseline_result is not None
            for result in pair_specific_results:
                adjusted = apply_conservative_stress(result, baseline_result)
                adjusted["scenario_pass"] = pair_scenario_pass(str(adjusted["scenario"]), adjusted)
                pair_results.append(adjusted)
                LOGGER.info(
                    "pair scenario | pair=%s | scenario=%s | source=%s | sharpe=%.2f | return=%.2f%%",
                    adjusted["pair"],
                    adjusted["scenario"],
                    "cache" if adjusted.get("cache_hit") else "compute",
                    adjusted["annualized_sharpe"] if np.isfinite(adjusted["annualized_sharpe"]) else float("nan"),
                    adjusted["total_return_pct"] if np.isfinite(adjusted["total_return_pct"]) else float("nan"),
                )
            baseline_return = float(baseline_result["total_return_pct"])
            for result in pair_results:
                if result["pair"] == config["pair"]:
                    result["implementation_shortfall_pct"] = float(result["total_return_pct"]) - baseline_return
                    if result["scenario"] == "baseline_current":
                        result["implementation_shortfall_pct"] = 0.0
    finally:
        os.chdir(original_cwd)

    monthly_matrix, scenario_pair_matrices = monthly_matrix_from_pair_results(pair_results)
    monthly_matrix.to_csv(MONTHLY_MATRIX_CSV)

    pair_public_results = []
    for result in pair_results:
        public_row = {key: value for key, value in result.items() if key not in {"monthly_pnl", "cache_version"}}
        pair_public_results.append(public_row)

    pair_df = pd.DataFrame(pair_public_results)
    pair_df["_scenario_order"] = pair_df["scenario"].map(SCENARIO_ORDER)
    pair_df = pair_df.sort_values(["pair", "_scenario_order"]).drop(columns=["_scenario_order"]).reset_index(drop=True)
    pair_df.to_csv(PAIR_OUTPUT_CSV, index=False)
    PAIR_OUTPUT_JSON.write_text(json.dumps(pair_public_results, indent=2), encoding="utf-8")

    portfolio_rows = []
    for scenario_name, scenario_matrix in scenario_pair_matrices.items():
        scenario_matrix = scenario_matrix[[cfg["pair"] for cfg in TOP3_CONFIGS if cfg["pair"] in scenario_matrix.columns]]
        full_series = portfolio_series_from_matrix(scenario_matrix)
        full_stats = portfolio_stats(full_series)
        cv_stats = portfolio_cpcv(full_series)
        full_row = {
            "scenario": scenario_name,
            "row_type": "portfolio_full",
            "variant": "full",
            "mean_pair_corr": mean_pair_corr(scenario_matrix),
            **full_stats,
            **cv_stats,
        }

        contribution_rows = []
        for pair in scenario_matrix.columns:
            contrib = contribution_metrics(scenario_matrix, full_series, pair)
            contribution_rows.append(
                {
                    "scenario": scenario_name,
                    "row_type": "pair_contribution",
                    "variant": pair,
                    "pair": pair,
                    **contrib,
                }
            )

        scenario_pass, concentration_fail = portfolio_scenario_pass(scenario_name, full_row, contribution_rows)
        full_row["scenario_pass"] = scenario_pass
        full_row["concentration_fail"] = concentration_fail
        portfolio_rows.append(full_row)

        for pair in scenario_matrix.columns:
            loo_cols = [col for col in scenario_matrix.columns if col != pair]
            loo_series = portfolio_series_from_matrix(scenario_matrix[loo_cols])
            portfolio_rows.append(
                {
                    "scenario": scenario_name,
                    "row_type": "portfolio_loo",
                    "variant": f"without_{pair}",
                    "pair": pair,
                    "mean_pair_corr": mean_pair_corr(scenario_matrix[loo_cols]),
                    **portfolio_stats(loo_series),
                }
            )

        portfolio_rows.extend(contribution_rows)

    portfolio_df = pd.DataFrame(portfolio_rows)
    portfolio_df["_scenario_order"] = portfolio_df["scenario"].map(SCENARIO_ORDER)
    portfolio_df = portfolio_df.sort_values(["_scenario_order", "row_type", "variant"]).drop(columns=["_scenario_order"]).reset_index(drop=True)
    portfolio_df.to_csv(PORT_OUTPUT_CSV, index=False)
    PORT_OUTPUT_JSON.write_text(json.dumps(portfolio_rows, indent=2), encoding="utf-8")

    maker_realistic_pairs = pair_df.loc[pair_df["scenario"] == "maker_realistic", "scenario_pass"]
    maker_realistic_portfolio = portfolio_df.loc[
        (portfolio_df["scenario"] == "maker_realistic") & (portfolio_df["row_type"] == "portfolio_full"),
        "scenario_pass",
    ]
    if maker_realistic_pairs.all() and not maker_realistic_pairs.empty and maker_realistic_portfolio.all() and not maker_realistic_portfolio.empty:
        final_recommendation = "Paper Trade Ready"
    else:
        final_recommendation = "Research Live Watch"

    REPORT_MD.write_text(build_report(pair_df, portfolio_df, top3_df, final_recommendation), encoding="utf-8")

    LOGGER.info(
        "complete | pair_output=%s | portfolio_output=%s | recommendation=%s",
        PAIR_OUTPUT_CSV,
        PORT_OUTPUT_CSV,
        final_recommendation,
    )


if __name__ == "__main__":
    main()
