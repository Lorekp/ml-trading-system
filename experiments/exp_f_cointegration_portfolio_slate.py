"""Portfolio-level analysis for the full annotated cointegration slate.

Builds monthly PnL series for each ranked config, measures cross-pair
dependence, and creates diversification-aware portfolio candidates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
INPUT_ANNOTATED = OUT_DIR / "ranked_slate_annotated.csv"


def load_rigorous_module():
    spec = importlib.util.spec_from_file_location("exp_f_cointegration_rigorous", RIGOROUS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def monthly_pnl_frame(segments: list[dict[str, object]], entry_z: float, cost_bps: float) -> pd.DataFrame:
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


def config_key(row: pd.Series | dict[str, object]) -> str:
    return (
        f"{int(row['rank']):03d}_{row['pair']}"
        f"_lb{int(row['lookback_months'])}_z{float(row['entry_z']):.1f}_c{int(row['cost_bps'])}"
    )


def select_diversified(
    ranked: pd.DataFrame,
    corr: pd.DataFrame,
    max_names: int = 12,
    corr_limit: float = 0.60,
    per_asset_cap: int = 2,
) -> pd.DataFrame:
    selected_rows = []
    asset_counts: dict[str, int] = {}
    selected_keys: list[str] = []

    for _, row in ranked.iterrows():
        key = row["config_key"]
        y_count = asset_counts.get(row["y_name"], 0)
        x_count = asset_counts.get(row["x_name"], 0)
        if y_count >= per_asset_cap or x_count >= per_asset_cap:
            continue

        if selected_keys:
            max_corr = corr.loc[key, selected_keys].abs().max()
            if pd.notna(max_corr) and max_corr > corr_limit:
                continue

        selected_rows.append(row.to_dict())
        selected_keys.append(key)
        asset_counts[row["y_name"]] = y_count + 1
        asset_counts[row["x_name"]] = x_count + 1
        if len(selected_rows) >= max_names:
            break

    return pd.DataFrame(selected_rows)


def portfolio_stats(pnl_df: pd.DataFrame, keys: list[str]) -> dict[str, float]:
    if not keys:
        return {
            "n_names": 0,
            "portfolio_sharpe": float("nan"),
            "portfolio_return_pct": float("nan"),
            "portfolio_max_dd_pct": float("nan"),
            "mean_pair_corr": float("nan"),
        }
    panel = pnl_df[keys].fillna(0.0)
    portfolio = panel.mean(axis=1)
    sharpe = annualized_sharpe(portfolio.to_numpy(dtype=np.float64))
    equity = portfolio.cumsum()
    peak = equity.cummax()
    max_dd = float(100.0 * (peak - equity).max()) if len(equity) else float("nan")
    corr = panel.corr()
    corr_vals = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    return {
        "n_names": len(keys),
        "portfolio_sharpe": sharpe,
        "portfolio_return_pct": float(100.0 * portfolio.sum()),
        "portfolio_max_dd_pct": max_dd,
        "mean_pair_corr": float(corr_vals.mean()) if not corr_vals.empty else float("nan"),
    }


def main() -> None:
    rigorous = load_rigorous_module()
    rigorous.ensure_dirs()
    rigorous._FULL_CLOSE_DF = rigorous.build_universe()

    ranked = pd.read_csv(INPUT_ANNOTATED)
    ranked["config_key"] = ranked.apply(config_key, axis=1)

    monthly_frames = []
    for _, row in ranked.iterrows():
        segments = rigorous.prepare_segments(row.to_dict(), int(row["lookback_months"]))
        monthly = monthly_pnl_frame(segments, float(row["entry_z"]), float(row["cost_bps"]))
        if monthly.empty:
            continue
        monthly["config_key"] = row["config_key"]
        monthly["pair"] = row["pair"]
        monthly_frames.append(monthly)

    monthly_all = pd.concat(monthly_frames, ignore_index=True)
    monthly_all.to_csv(OUT_DIR / "portfolio_monthly_pnl_long.csv", index=False)

    pnl_matrix = (
        monthly_all.pivot(index="month", columns="config_key", values="pnl")
        .sort_index()
        .sort_index(axis=1)
    )
    pnl_matrix.to_csv(OUT_DIR / "portfolio_monthly_pnl_matrix.csv")

    corr = pnl_matrix.corr(min_periods=6)
    corr.to_csv(OUT_DIR / "portfolio_pair_corr.csv")

    diversified = select_diversified(ranked, corr, max_names=12, corr_limit=0.60, per_asset_cap=2)
    concentrated = ranked.head(12).copy()

    diversified.to_csv(OUT_DIR / "portfolio_diversified_12.csv", index=False)
    concentrated.to_csv(OUT_DIR / "portfolio_top12_raw.csv", index=False)

    div_stats = portfolio_stats(pnl_matrix, diversified["config_key"].tolist())
    raw_stats = portfolio_stats(pnl_matrix, concentrated["config_key"].tolist())

    summary = pd.DataFrame(
        [
            {"portfolio": "top12_raw", **raw_stats},
            {"portfolio": "diversified_12", **div_stats},
        ]
    )
    summary.to_csv(OUT_DIR / "portfolio_summary.csv", index=False)

    lines = [
        "# Portfolio Slate",
        "",
        "## Raw Top 12",
        "",
        f"- names={int(raw_stats['n_names'])} | Sharpe={raw_stats['portfolio_sharpe']:.2f} | "
        f"return={raw_stats['portfolio_return_pct']:.2f}% | maxDD={raw_stats['portfolio_max_dd_pct']:.2f}% | "
        f"mean pair corr={raw_stats['mean_pair_corr']:.2f}",
        "",
        "## Diversified 12",
        "",
        f"- names={int(div_stats['n_names'])} | Sharpe={div_stats['portfolio_sharpe']:.2f} | "
        f"return={div_stats['portfolio_return_pct']:.2f}% | maxDD={div_stats['portfolio_max_dd_pct']:.2f}% | "
        f"mean pair corr={div_stats['mean_pair_corr']:.2f}",
        "",
        "## Diversified Members",
        "",
    ]
    for _, row in diversified.iterrows():
        lines.append(
            f"- `#{int(row['rank'])} {row['pair']}` | bucket={row['bucket']} | "
            f"AFML={int(row['afml_score'])}/4 | strengths={row['strength_tags']} | risks={row['risk_tags']}"
        )
    (OUT_DIR / "portfolio_slate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
