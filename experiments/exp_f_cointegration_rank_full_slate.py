"""Build a full ranked slate for all validated cointegration candidates.

This stage does not shrink the candidate set. It ranks the full validated list
using:
- existing validation metrics
- monthly regime robustness using BTC monthly return/vol regimes
- transparent ranking score
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
INPUT_VALIDATION = OUT_DIR / "validation_potential.csv"
CACHE_DIR = OUT_DIR / "cache" / "validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_VALIDATION)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_rigorous_module():
    spec = importlib.util.spec_from_file_location("exp_f_cointegration_rigorous", RIGOROUS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validation_key(row: pd.Series | dict[str, object]) -> str:
    return (
        f"{row['pair']}|lb={int(row['lookback_months'])}|"
        f"z={float(row['entry_z']):.2f}|cost={float(row['cost_bps']):.2f}"
    )


def monthly_pnl_table(segments: list[dict[str, object]], entry_z: float, cost_bps: float) -> pd.DataFrame:
    rows = []
    for seg in segments:
        zscore = np.asarray(seg["zscore"], dtype=np.float64)
        hedge_ret = np.asarray(seg["hedge_ret"], dtype=np.float64)
        cost = cost_bps / 10_000.0 * (1.0 + abs(float(seg["beta"])))
        pnl = 0.0
        position = 0
        trades = 0
        for i in range(1, len(zscore)):
            if position != 0:
                pnl += position * hedge_ret[i]
            z = zscore[i]
            if position == 0:
                if z <= -entry_z:
                    position = 1
                    pnl -= cost
                    trades += 1
                elif z >= entry_z:
                    position = -1
                    pnl -= cost
                    trades += 1
            elif position == 1 and z >= 0.0:
                position = 0
                pnl -= cost
            elif position == -1 and z <= 0.0:
                position = 0
                pnl -= cost
        if position != 0:
            pnl -= cost
        rows.append({"month": seg["month"], "pnl": float(pnl), "trades": trades})
    return pd.DataFrame(rows)


def annualized_sharpe(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    if std_ret <= 0 or not np.isfinite(std_ret):
        return float("nan")
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probabilistic_sharpe(monthly_pnl: pd.Series | np.ndarray, benchmark_sr: float = 0.0) -> float:
    arr = np.asarray(monthly_pnl, dtype=np.float64)
    if arr.size < 3:
        return float("nan")
    sr_hat = annualized_sharpe(arr)
    if not np.isfinite(sr_hat):
        return float("nan")
    sr_period = sr_hat / math.sqrt(12.0)
    sr_benchmark = benchmark_sr / math.sqrt(12.0)
    centered = arr - arr.mean()
    std = arr.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return float("nan")
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))
    denom = math.sqrt(max(1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period ** 2), 1e-12))
    z = ((sr_period - sr_benchmark) * math.sqrt(arr.size - 1.0)) / denom
    return float(normal_cdf(z))


def min_track_record_length(monthly_pnl: pd.Series | np.ndarray, benchmark_sr: float = 0.0, alpha: float = 0.05) -> float:
    arr = np.asarray(monthly_pnl, dtype=np.float64)
    if arr.size < 3:
        return float("inf")
    sr_hat = annualized_sharpe(arr)
    if not np.isfinite(sr_hat) or sr_hat <= benchmark_sr:
        return float("inf")
    sr_period = sr_hat / math.sqrt(12.0)
    sr_benchmark = benchmark_sr / math.sqrt(12.0)
    centered = arr - arr.mean()
    std = arr.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return float("inf")
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))
    z = 1.6448536269514722 if alpha == 0.05 else 1.959963984540054
    numer = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period ** 2)
    denom = max((sr_period - sr_benchmark) ** 2, 1e-12)
    return float(1.0 + numer * (z ** 2) / denom)


def build_btc_regime_map(rigorous) -> pd.DataFrame:
    close_df = rigorous._FULL_CLOSE_DF
    assert close_df is not None
    btc = close_df["BTC"].dropna().astype(float)
    btc_15m_ret = btc.pct_change().dropna()
    monthly_ret = btc.resample("ME").last().pct_change()
    monthly_vol = btc_15m_ret.groupby(btc_15m_ret.index.to_period("M")).std() * np.sqrt(96 * 30)
    monthly = pd.DataFrame(
        {
            "month": monthly_ret.index.to_period("M").astype(str),
            "btc_monthly_return": monthly_ret.to_numpy(dtype=np.float64),
            "btc_monthly_vol": monthly_vol.reindex(monthly_ret.index.to_period("M")).to_numpy(dtype=np.float64),
        }
    )
    vol_median = float(monthly["btc_monthly_vol"].median())
    monthly["regime_trend"] = np.where(monthly["btc_monthly_return"] >= 0, "up", "down")
    monthly["regime_vol"] = np.where(monthly["btc_monthly_vol"] >= vol_median, "high_vol", "low_vol")
    return monthly


def regime_sharpe(monthly: pd.DataFrame, regime_col: str, regime_value: str) -> tuple[float, int, float]:
    subset = monthly.loc[monthly[regime_col] == regime_value, "pnl"]
    if subset.empty:
        return float("nan"), 0, float("nan")
    return annualized_sharpe(subset), int(subset.size), float(100.0 * subset.sum())


def compute_rank_score(row: pd.Series) -> float:
    min_half = min(
        x for x in [row.get("first_half_sharpe", np.nan), row.get("second_half_sharpe", np.nan)]
        if np.isfinite(x)
    ) if np.isfinite(row.get("first_half_sharpe", np.nan)) or np.isfinite(row.get("second_half_sharpe", np.nan)) else 0.0
    regime_vals = [row.get("up_sharpe", np.nan), row.get("down_sharpe", np.nan),
                   row.get("high_vol_sharpe", np.nan), row.get("low_vol_sharpe", np.nan)]
    regime_finite = [float(x) for x in regime_vals if np.isfinite(x)]
    regime_pos_share = float(np.mean([x > 0 for x in regime_finite])) if regime_finite else 0.0
    regime_min = float(min(regime_finite)) if regime_finite else 0.0
    dd_penalty = max(float(row["base_max_dd_pct"]) - 15.0, 0.0)
    return (
        2.0 * float(row["validation_score"])
        + 0.10 * min(float(row["n_months"]), 36.0)
        + 0.75 * min(max(float(row.get("stress_cost_plus20_sharpe", 0.0)), 0.0), 3.0)
        + 0.75 * min(max(float(row.get("hood_median_sharpe", 0.0)), 0.0), 3.0)
        + 0.50 * min(max(min_half, 0.0), 3.0)
        + 2.0 * regime_pos_share
        + 0.25 * min(max(regime_min, 0.0), 2.0)
        - 0.02 * dd_penalty
    )


def evaluate_one(row_dict: dict[str, object], regime_map: pd.DataFrame, rigorous) -> dict[str, object]:
    segments = rigorous.prepare_segments(row_dict, int(row_dict["lookback_months"]))
    monthly = monthly_pnl_table(segments, float(row_dict["entry_z"]), float(row_dict["cost_bps"]))
    merged = monthly.merge(regime_map, on="month", how="left")

    up_sharpe, up_n, up_ret = regime_sharpe(merged, "regime_trend", "up")
    down_sharpe, down_n, down_ret = regime_sharpe(merged, "regime_trend", "down")
    hv_sharpe, hv_n, hv_ret = regime_sharpe(merged, "regime_vol", "high_vol")
    lv_sharpe, lv_n, lv_ret = regime_sharpe(merged, "regime_vol", "low_vol")
    trend_both_positive = bool(np.isfinite(up_sharpe) and np.isfinite(down_sharpe) and up_sharpe > 0 and down_sharpe > 0)
    vol_both_positive = bool(np.isfinite(hv_sharpe) and np.isfinite(lv_sharpe) and hv_sharpe > 0 and lv_sharpe > 0)
    psr_sr0 = probabilistic_sharpe(merged["pnl"].to_numpy(dtype=np.float64), benchmark_sr=0.0) if not merged.empty else float("nan")
    min_tl = min_track_record_length(merged["pnl"].to_numpy(dtype=np.float64), benchmark_sr=0.0, alpha=0.05) if not merged.empty else float("inf")
    afml_psr_ok = bool(np.isfinite(psr_sr0) and psr_sr0 >= 0.95)
    afml_track_record_ok = bool(np.isfinite(min_tl) and len(merged) >= math.ceil(min_tl))

    return {
        "validation_key": validation_key(row_dict),
        "pair": row_dict["pair"],
        "up_sharpe": up_sharpe,
        "up_months": up_n,
        "up_return_pct": up_ret,
        "down_sharpe": down_sharpe,
        "down_months": down_n,
        "down_return_pct": down_ret,
        "high_vol_sharpe": hv_sharpe,
        "high_vol_months": hv_n,
        "high_vol_return_pct": hv_ret,
        "low_vol_sharpe": lv_sharpe,
        "low_vol_months": lv_n,
        "low_vol_return_pct": lv_ret,
        "trend_both_positive": trend_both_positive,
        "vol_both_positive": vol_both_positive,
        "psr_sr0": psr_sr0,
        "min_track_record_months_95": min_tl,
        "afml_psr_ok": afml_psr_ok,
        "afml_track_record_ok": afml_track_record_ok,
        "monthly_pnl_mean_pct": float(100.0 * merged["pnl"].mean()) if not merged.empty else float("nan"),
        "monthly_pnl_std_pct": float(100.0 * merged["pnl"].std(ddof=1)) if len(merged) > 1 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "ranked_slate_cache.parquet"
    output_csv = OUT_DIR / "ranked_slate_full.csv"
    output_md = OUT_DIR / "ranked_slate_full.md"

    validated = pd.read_csv(args.input)
    rigorous = load_rigorous_module()
    rigorous.ensure_dirs()
    rigorous._FULL_CLOSE_DF = rigorous.build_universe()
    regime_map = build_btc_regime_map(rigorous)

    if cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
        required_cols = {
            "validation_key",
            "pair",
            "psr_sr0",
            "min_track_record_months_95",
            "afml_psr_ok",
            "afml_track_record_ok",
        }
        if not required_cols.issubset(set(cache_df.columns)):
            cache_df = pd.DataFrame()
    else:
        cache_df = pd.DataFrame()
    cached_keys = set(cache_df["validation_key"].tolist()) if not cache_df.empty else set()

    pending = validated.copy()
    pending["validation_key"] = pending.apply(validation_key, axis=1)
    pending = pending.loc[~pending["validation_key"].isin(cached_keys)].copy()

    rows: list[dict[str, object]] = []
    if not pending.empty:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_one, row.to_dict(), regime_map, rigorous) for _, row in pending.iterrows()]
            for future in as_completed(futures):
                rows.append(future.result())

    add_df = pd.DataFrame(rows)
    regime_df = pd.concat([cache_df, add_df], ignore_index=True) if not cache_df.empty else add_df
    if not regime_df.empty:
        regime_df = regime_df.drop_duplicates(subset=["validation_key"], keep="last")
    regime_df.to_parquet(cache_path, index=False)

    ranked = validated.merge(regime_df, on=["validation_key", "pair"], how="left")
    ranked["afml_regime_ok"] = ranked["trend_both_positive"].fillna(False) & ranked["vol_both_positive"].fillna(False)
    ranked["afml_validation_ok"] = ranked["validation_score"] >= 4
    ranked["afml_score"] = (
        ranked["afml_psr_ok"].fillna(False).astype(int)
        + ranked["afml_track_record_ok"].fillna(False).astype(int)
        + ranked["afml_regime_ok"].fillna(False).astype(int)
        + ranked["afml_validation_ok"].fillna(False).astype(int)
    )
    ranked["rank_score"] = ranked.apply(compute_rank_score, axis=1)
    ranked = ranked.sort_values(
        ["afml_score", "rank_score", "validation_score", "n_months", "base_sharpe"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(output_csv, index=False)

    lines = [
        "# Ranked Full Slate",
        "",
        f"- Input validation file: `{args.input.name}`",
        f"- Ranked configs: `{len(ranked)}`",
        f"- Regime cache: `{cache_path.name}`",
        "",
        "## Top 25",
        "",
    ]
    for _, row in ranked.head(25).iterrows():
        lines.append(
            f"- `#{int(row['rank'])} {row['pair']}` | AFML={int(row['afml_score'])}/4 | score={row['rank_score']:.2f} | "
            f"val={int(row['validation_score'])}/4 | months={int(row['n_months'])} | "
            f"PSR={row['psr_sr0']:.2f} | minTRL95={row['min_track_record_months_95']:.1f} | "
            f"base Sharpe={row['base_sharpe']:.2f} | up/down=({row['up_sharpe']:.2f}, {row['down_sharpe']:.2f}) | "
            f"high/low vol=({row['high_vol_sharpe']:.2f}, {row['low_vol_sharpe']:.2f})"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
