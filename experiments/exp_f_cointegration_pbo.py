"""Estimate PBO-style overfitting risk using CSCV on monthly PnL."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
INPUT_PNL = OUT_DIR / "portfolio_monthly_pnl_matrix.csv"
INPUT_FINAL = OUT_DIR / "final_actionable_slate.csv"

N_GROUPS = 8
MIN_MONTHS_PER_SIDE = 4


def annualized_sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    if std_ret <= 0 or not np.isfinite(std_ret):
        return float("nan")
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def split_groups(index_values: list[str], n_groups: int) -> list[list[str]]:
    arr = np.array(index_values, dtype=object)
    return [chunk.tolist() for chunk in np.array_split(arr, n_groups)]


def score_strategy(series: pd.Series, months: list[str]) -> float:
    arr = series.reindex(months).dropna().to_numpy(dtype=np.float64)
    if arr.size < MIN_MONTHS_PER_SIDE:
        return float("nan")
    return annualized_sharpe(arr)


def main() -> None:
    pnl = pd.read_csv(INPUT_PNL, index_col=0)
    months = pnl.index.tolist()
    groups = split_groups(months, N_GROUPS)
    group_ids = list(range(N_GROUPS))
    half = N_GROUPS // 2
    combos = list(itertools.combinations(group_ids, half))

    records = []
    logits = []
    win_counts: dict[str, int] = {col: 0 for col in pnl.columns}
    selected_oos_sharpes: dict[str, list[float]] = {col: [] for col in pnl.columns}

    for combo in combos:
        is_months = [m for gid in combo for m in groups[gid]]
        oos_months = [m for gid in group_ids if gid not in combo for m in groups[gid]]

        is_scores = {col: score_strategy(pnl[col], is_months) for col in pnl.columns}
        oos_scores = {col: score_strategy(pnl[col], oos_months) for col in pnl.columns}

        valid = [col for col in pnl.columns if np.isfinite(is_scores[col]) and np.isfinite(oos_scores[col])]
        if len(valid) < 5:
            continue

        best_col = max(valid, key=lambda col: is_scores[col])
        win_counts[best_col] += 1
        selected_oos_sharpes[best_col].append(float(oos_scores[best_col]))

        ordered = sorted(valid, key=lambda col: oos_scores[col])
        rank = ordered.index(best_col) + 1
        relative_rank = rank / (len(ordered) + 1.0)
        logit = math.log(relative_rank / (1.0 - relative_rank))
        logits.append(logit)
        records.append(
            {
                "is_groups": ",".join(map(str, combo)),
                "oos_groups": ",".join(map(str, [gid for gid in group_ids if gid not in combo])),
                "best_config": best_col,
                "is_sharpe": float(is_scores[best_col]),
                "oos_sharpe": float(oos_scores[best_col]),
                "oos_rank": int(rank),
                "n_valid": int(len(valid)),
                "relative_rank": float(relative_rank),
                "logit": float(logit),
            }
        )

    split_df = pd.DataFrame(records)
    split_df.to_csv(OUT_DIR / "pbo_splits.csv", index=False)

    pbo = float(np.mean(np.asarray(logits) <= 0.0)) if logits else float("nan")
    summary_rows = []
    for col in pnl.columns:
        wins = win_counts[col]
        oos = selected_oos_sharpes[col]
        summary_rows.append(
            {
                "config_key": col,
                "selected_count": int(wins),
                "mean_selected_oos_sharpe": float(np.mean(oos)) if oos else float("nan"),
                "median_selected_oos_sharpe": float(np.median(oos)) if oos else float("nan"),
            }
        )
    strategy_df = pd.DataFrame(summary_rows)
    strategy_df.to_csv(OUT_DIR / "pbo_strategy_summary.csv", index=False)

    final = pd.read_csv(INPUT_FINAL)
    final = final.merge(strategy_df, on="config_key", how="left")
    final["pbo_global"] = pbo
    final["pbo_selected_ok"] = final["mean_selected_oos_sharpe"].fillna(-np.inf) > 0.5
    final["post_pbo_bucket"] = np.select(
        [
            final["dsr_ok"] & final["pbo_selected_ok"] & (final["actionable_bucket"] != "Low Priority"),
            final["dsr_ok"] | final["pbo_selected_ok"],
        ],
        [
            "Research Live Watch",
            "Research Hold",
        ],
        default="Reject for now",
    )
    final.to_csv(OUT_DIR / "final_actionable_slate_pbo.csv", index=False)

    lines = [
        "# PBO Summary",
        "",
        f"- CSCV groups: `{N_GROUPS}`",
        f"- Combinations evaluated: `{len(split_df)}`",
        f"- Global PBO estimate: `{pbo:.3f}`",
        "",
        "## Top Post-PBO Names",
        "",
    ]
    top = final.sort_values(
        ["post_pbo_bucket", "dsr_ok", "mean_selected_oos_sharpe", "rank"],
        ascending=[True, False, False, True],
    ).head(25)
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['pair']}` | postPBO={row['post_pbo_bucket']} | DSR={row['dsr_prob']:.3f} | "
            f"mean selected OOS Sharpe={row.get('mean_selected_oos_sharpe', float('nan')):.2f} | "
            f"selected_count={int(row.get('selected_count', 0) if pd.notna(row.get('selected_count', 0)) else 0)}"
        )
    (OUT_DIR / "pbo_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
