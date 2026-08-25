"""Validate potential cointegration pairs with persistent caching.

Default candidate universe: results/cointegration_rigorous/robust_pairs.csv

Validation dimensions:
- temporal split stability
- higher-cost stress
- local parameter neighborhood robustness
- sample length
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
VALIDATION_CACHE_DIR = OUT_DIR / "cache" / "validation"
DEFAULT_INPUT = OUT_DIR / "robust_pairs.csv"
DEFAULT_OUTPUT_STEM = "validation_potential"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
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


def monthly_pnl_series(segments: list[dict[str, object]], entry_z: float, cost_bps: float) -> np.ndarray:
    values = []
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
                elif z >= entry_z:
                    position = -1
                    pnl -= cost
            elif position == 1 and z >= 0.0:
                position = 0
                pnl -= cost
            elif position == -1 and z <= 0.0:
                position = 0
                pnl -= cost
        if position != 0:
            pnl -= cost
        values.append(pnl)
    return np.asarray(values, dtype=np.float64)


def annualized_sharpe(monthly_pnl: np.ndarray) -> float:
    if monthly_pnl.size < 2:
        return float("nan")
    mean_ret = float(np.mean(monthly_pnl))
    std_ret = float(np.std(monthly_pnl, ddof=1))
    if std_ret <= 0 or not np.isfinite(std_ret):
        return float("nan")
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def split_metrics(monthly_pnl: np.ndarray) -> dict[str, float]:
    if monthly_pnl.size < 4:
        return {
            "first_half_sharpe": float("nan"),
            "second_half_sharpe": float("nan"),
            "first_half_return_pct": float("nan"),
            "second_half_return_pct": float("nan"),
        }
    cut = monthly_pnl.size // 2
    first_half = monthly_pnl[:cut]
    second_half = monthly_pnl[cut:]
    return {
        "first_half_sharpe": annualized_sharpe(first_half),
        "second_half_sharpe": annualized_sharpe(second_half),
        "first_half_return_pct": float(100.0 * np.sum(first_half)),
        "second_half_return_pct": float(100.0 * np.sum(second_half)),
    }


def stress_metrics(segments: list[dict[str, object]], entry_z: float, base_cost_bps: float, rigorous) -> dict[str, float]:
    plus5 = rigorous.simulate_segments(segments, entry_z=entry_z, cost_bps=base_cost_bps + 5)
    plus10 = rigorous.simulate_segments(segments, entry_z=entry_z, cost_bps=base_cost_bps + 10)
    plus20 = rigorous.simulate_segments(segments, entry_z=entry_z, cost_bps=base_cost_bps + 20)
    return {
        "stress_cost_plus5_sharpe": float(plus5["sharpe"]) if plus5 else float("nan"),
        "stress_cost_plus10_sharpe": float(plus10["sharpe"]) if plus10 else float("nan"),
        "stress_cost_plus20_sharpe": float(plus20["sharpe"]) if plus20 else float("nan"),
        "stress_cost_plus5_return_pct": float(plus5["total_return_pct"]) if plus5 else float("nan"),
        "stress_cost_plus10_return_pct": float(plus10["total_return_pct"]) if plus10 else float("nan"),
        "stress_cost_plus20_return_pct": float(plus20["total_return_pct"]) if plus20 else float("nan"),
    }


def neighborhood_metrics(pair_rows: pd.DataFrame, lookback_months: int, entry_z: float) -> dict[str, float]:
    lookbacks = sorted(pair_rows["lookback_months"].dropna().unique().tolist())
    if lookback_months in lookbacks:
        idx = lookbacks.index(lookback_months)
        allowed_lb = set(lookbacks[max(0, idx - 1): min(len(lookbacks), idx + 2)])
    else:
        allowed_lb = {lookback_months}
    hood = pair_rows.loc[
        pair_rows["lookback_months"].isin(allowed_lb)
        & (pair_rows["entry_z"] >= entry_z - 0.5)
        & (pair_rows["entry_z"] <= entry_z + 0.5)
    ].copy()
    if hood.empty:
        return {
            "hood_configs": 0,
            "hood_median_sharpe": float("nan"),
            "hood_p25_sharpe": float("nan"),
            "hood_positive_share": float("nan"),
        }
    sharpe = hood["sharpe"].astype(float)
    return {
        "hood_configs": int(len(hood)),
        "hood_median_sharpe": float(sharpe.median()),
        "hood_p25_sharpe": float(sharpe.quantile(0.25)),
        "hood_positive_share": float((sharpe > 0.5).mean()),
    }


def validate_one(row_dict: dict[str, object], pair_backtests: pd.DataFrame, rigorous) -> dict[str, object]:
    segments = rigorous.prepare_segments(row_dict, int(row_dict["lookback_months"]))
    monthly = monthly_pnl_series(segments, float(row_dict["entry_z"]), float(row_dict["cost_bps"]))
    split = split_metrics(monthly)
    stress = stress_metrics(segments, float(row_dict["entry_z"]), float(row_dict["cost_bps"]), rigorous)
    hood = neighborhood_metrics(pair_backtests, int(row_dict["lookback_months"]), float(row_dict["entry_z"]))

    n_months = int(row_dict["n_months"])
    months_ok = n_months >= 8
    halves_ok = (
        np.isfinite(split["first_half_sharpe"])
        and np.isfinite(split["second_half_sharpe"])
        and split["first_half_sharpe"] > 0
        and split["second_half_sharpe"] > 0
    )
    stress_ok = np.isfinite(stress["stress_cost_plus10_sharpe"]) and stress["stress_cost_plus10_sharpe"] > 0.5
    hood_ok = (
        np.isfinite(hood["hood_median_sharpe"])
        and np.isfinite(hood["hood_positive_share"])
        and hood["hood_median_sharpe"] > 0.5
        and hood["hood_positive_share"] >= 0.50
    )
    score = int(months_ok) + int(halves_ok) + int(stress_ok) + int(hood_ok)

    return {
        "validation_key": validation_key(row_dict),
        "pair": row_dict["pair"],
        "y_name": row_dict["y_name"],
        "x_name": row_dict["x_name"],
        "lookback_months": int(row_dict["lookback_months"]),
        "entry_z": float(row_dict["entry_z"]),
        "cost_bps": float(row_dict["cost_bps"]),
        "base_sharpe": float(row_dict["sharpe"]),
        "base_return_pct": float(row_dict["total_return_pct"]),
        "base_win_pct": float(row_dict["positive_months_pct"]),
        "base_max_dd_pct": float(row_dict["max_drawdown_pct"]),
        "n_months": n_months,
        "n_trades": int(row_dict["n_trades"]),
        **split,
        **stress,
        **hood,
        "months_ok": months_ok,
        "halves_ok": halves_ok,
        "stress_ok": stress_ok,
        "hood_ok": hood_ok,
        "validation_score": score,
    }


def main() -> None:
    args = parse_args()
    VALIDATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_parquet = VALIDATION_CACHE_DIR / f"{args.output_stem}_cache.parquet"
    output_csv = OUT_DIR / f"{args.output_stem}.csv"
    output_md = OUT_DIR / f"{args.output_stem}.md"

    candidates = pd.read_csv(args.input)
    backtests = pd.read_csv(OUT_DIR / "backtests.csv")
    rigorous = load_rigorous_module()
    rigorous.ensure_dirs()
    rigorous._FULL_CLOSE_DF = rigorous.build_universe()

    if cache_parquet.exists():
        cache_df = pd.read_parquet(cache_parquet)
    else:
        cache_df = pd.DataFrame()
    cached_keys = set(cache_df["validation_key"].tolist()) if not cache_df.empty else set()

    pending = candidates.copy()
    pending["validation_key"] = pending.apply(validation_key, axis=1)
    pending = pending.loc[~pending["validation_key"].isin(cached_keys)].copy()

    rows: list[dict[str, object]] = []
    if not pending.empty:
        pair_groups = {pair: df.copy() for pair, df in backtests.groupby("pair")}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(validate_one, row.to_dict(), pair_groups.get(row["pair"], pd.DataFrame()), rigorous)
                for _, row in pending.iterrows()
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    new_df = pd.DataFrame(rows)
    full_df = pd.concat([cache_df, new_df], ignore_index=True) if not cache_df.empty else new_df
    if not full_df.empty:
        full_df = full_df.drop_duplicates(subset=["validation_key"], keep="last")
        full_df = full_df.sort_values(
            ["validation_score", "n_months", "base_sharpe"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    full_df.to_parquet(cache_parquet, index=False)
    full_df.to_csv(output_csv, index=False)

    lines = [
        "# Potential-Pair Validation",
        "",
        f"- Input candidates: `{args.input.name}`",
        f"- Cached results: `{cache_parquet.name}`",
        f"- Validated configs: `{len(full_df)}`",
        "",
        "## Top Ranked",
        "",
    ]
    for _, row in full_df.head(25).iterrows():
        lines.append(
            f"- `{row['pair']}` score={int(row['validation_score'])}/4 | base Sharpe={row['base_sharpe']:.2f} | "
            f"months={int(row['n_months'])} | stress+10={row['stress_cost_plus10_sharpe']:.2f} | "
            f"hood median={row['hood_median_sharpe']:.2f}"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
