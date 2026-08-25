"""Validate the current top-10 cointegration configs on stricter robustness checks.

This script does not rerun screening. It reuses:
- results/cointegration_rigorous/summary.csv
- results/cointegration_rigorous/backtests.csv
- cached monthly segment pickles from exp_f_cointegration_rigorous.py
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
VALIDATION_CSV = OUT_DIR / "validation_top10.csv"
VALIDATION_MD = OUT_DIR / "validation_top10.md"


def load_rigorous_module():
    spec = importlib.util.spec_from_file_location("exp_f_cointegration_rigorous", RIGOROUS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    cost_5 = rigorous.simulate_segments(segments, entry_z=entry_z, cost_bps=base_cost_bps + 5)
    cost_10 = rigorous.simulate_segments(segments, entry_z=entry_z, cost_bps=base_cost_bps + 10)
    return {
        "stress_cost_plus5_sharpe": float(cost_5["sharpe"]) if cost_5 else float("nan"),
        "stress_cost_plus10_sharpe": float(cost_10["sharpe"]) if cost_10 else float("nan"),
        "stress_cost_plus5_return_pct": float(cost_5["total_return_pct"]) if cost_5 else float("nan"),
        "stress_cost_plus10_return_pct": float(cost_10["total_return_pct"]) if cost_10 else float("nan"),
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


def main() -> None:
    rigorous = load_rigorous_module()
    summary = pd.read_csv(OUT_DIR / "summary.csv")
    backtests = pd.read_csv(OUT_DIR / "backtests.csv")
    top10 = summary.head(10).copy()

    rigorous.ensure_dirs()
    rigorous._FULL_CLOSE_DF = rigorous.build_universe()

    rows = []
    for _, row in top10.iterrows():
        row_dict = row.to_dict()
        segments = rigorous.prepare_segments(row_dict, int(row["lookback_months"]))
        monthly = monthly_pnl_series(segments, float(row["entry_z"]), float(row["cost_bps"]))
        split = split_metrics(monthly)
        stress = stress_metrics(segments, float(row["entry_z"]), float(row["cost_bps"]), rigorous)
        hood = neighborhood_metrics(backtests.loc[backtests["pair"] == row["pair"]], int(row["lookback_months"]), float(row["entry_z"]))

        n_months = int(row["n_months"])
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

        rows.append(
            {
                "pair": row["pair"],
                "y_name": row["y_name"],
                "x_name": row["x_name"],
                "lookback_months": int(row["lookback_months"]),
                "entry_z": float(row["entry_z"]),
                "cost_bps": float(row["cost_bps"]),
                "base_sharpe": float(row["sharpe"]),
                "base_return_pct": float(row["total_return_pct"]),
                "base_win_pct": float(row["positive_months_pct"]),
                "base_max_dd_pct": float(row["max_drawdown_pct"]),
                "n_months": n_months,
                "n_trades": int(row["n_trades"]),
                **split,
                **stress,
                **hood,
                "months_ok": months_ok,
                "halves_ok": halves_ok,
                "stress_ok": stress_ok,
                "hood_ok": hood_ok,
                "validation_score": score,
            }
        )

    validated = pd.DataFrame(rows).sort_values(
        ["validation_score", "n_months", "base_sharpe"],
        ascending=[False, False, False],
    )
    validated.to_csv(VALIDATION_CSV, index=False)

    lines = [
        "# Top-10 Validation",
        "",
        "- Small-sample next step: temporal split, cost stress, and parameter-neighborhood robustness.",
        "- Score components: `months_ok` (>=8 months), `halves_ok` (both halves positive Sharpe), `stress_ok` (+10bps Sharpe > 0.5), `hood_ok` (local parameter neighborhood robust).",
        "",
    ]
    for _, row in validated.iterrows():
        lines.append(
            f"- `{row['pair']}` score={int(row['validation_score'])}/4 | base Sharpe={row['base_sharpe']:.2f} | "
            f"months={int(row['n_months'])} | halfs=({row['first_half_sharpe']:.2f}, {row['second_half_sharpe']:.2f}) | "
            f"stress+10={row['stress_cost_plus10_sharpe']:.2f} | hood median={row['hood_median_sharpe']:.2f}"
        )
    VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
