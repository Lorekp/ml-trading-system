"""Compute AFML-style deflated Sharpe metrics for the full cointegration slate."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
INPUT_ANNOTATED = OUT_DIR / "ranked_slate_annotated.csv"
INPUT_PNL = OUT_DIR / "portfolio_monthly_pnl_matrix.csv"
INPUT_DIV = OUT_DIR / "portfolio_diversified_12.csv"


def annualized_sharpe(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    if std_ret <= 0 or not np.isfinite(std_ret):
        return float("nan")
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def probabilistic_sharpe(arr: np.ndarray, benchmark_sr_ann: float) -> float:
    if arr.size < 3:
        return float("nan")
    sr_hat_ann = annualized_sharpe(arr)
    if not np.isfinite(sr_hat_ann):
        return float("nan")
    sr_hat = sr_hat_ann / math.sqrt(12.0)
    benchmark_sr = benchmark_sr_ann / math.sqrt(12.0)
    centered = arr - arr.mean()
    std = arr.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return float("nan")
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))
    denom = math.sqrt(max(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2), 1e-12))
    z = ((sr_hat - benchmark_sr) * math.sqrt(arr.size - 1.0)) / denom
    return float(NormalDist().cdf(z))


def config_key(row: pd.Series | dict[str, object]) -> str:
    return (
        f"{int(row['rank']):03d}_{row['pair']}"
        f"_lb{int(row['lookback_months'])}_z{float(row['entry_z']):.1f}_c{int(row['cost_bps'])}"
    )


def main() -> None:
    annotated = pd.read_csv(INPUT_ANNOTATED)
    annotated["config_key"] = annotated.apply(config_key, axis=1)

    pnl = pd.read_csv(INPUT_PNL, index_col=0)
    diversified = pd.read_csv(INPUT_DIV)
    diversified_keys = set(diversified["config_key"].tolist()) if "config_key" in diversified.columns else set()

    monthly_sr = {}
    ann_sr = {}
    valid_cols = []
    for col in pnl.columns:
        arr = pnl[col].dropna().to_numpy(dtype=np.float64)
        ann = annualized_sharpe(arr)
        if np.isfinite(ann):
            ann_sr[col] = ann
            monthly_sr[col] = ann / math.sqrt(12.0)
            valid_cols.append(col)

    n_trials = len(valid_cols)
    sr_std = float(np.std([monthly_sr[k] for k in valid_cols], ddof=1)) if n_trials > 1 else 0.0
    gamma = 0.5772156649015329
    z1 = NormalDist().inv_cdf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0.0
    z2 = NormalDist().inv_cdf(1.0 - 1.0 / (n_trials * math.e)) if n_trials > 1 else 0.0
    sr_star_monthly = sr_std * ((1.0 - gamma) * z1 + gamma * z2)
    sr_star_ann = sr_star_monthly * math.sqrt(12.0)

    rows = []
    for _, row in annotated.iterrows():
        key = row["config_key"]
        arr = pnl[key].dropna().to_numpy(dtype=np.float64) if key in pnl.columns else np.array([], dtype=np.float64)
        dsr = probabilistic_sharpe(arr, benchmark_sr_ann=sr_star_ann) if arr.size else float("nan")
        rows.append(
            {
                "config_key": key,
                "dsr_prob": dsr,
                "dsr_ok": bool(np.isfinite(dsr) and dsr >= 0.95),
                "n_trials": n_trials,
                "trial_sr_benchmark_ann": sr_star_ann,
                "trial_sr_std_monthly": sr_std,
                "in_diversified_12": key in diversified_keys,
            }
        )

    dsr_df = pd.DataFrame(rows)
    final = annotated.merge(dsr_df, on="config_key", how="left")
    final["actionable_bucket"] = np.select(
        [
            (final["bucket"] == "Core") & final["dsr_ok"] & final["in_diversified_12"],
            (final["bucket"].isin(["Core", "Strong"])) & final["dsr_ok"],
            final["dsr_ok"],
        ],
        [
            "Deploy Candidate",
            "Research Priority",
            "Watchlist",
        ],
        default="Low Priority",
    )
    final = final.sort_values(
        ["dsr_ok", "in_diversified_12", "afml_score", "rank_score", "rank"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    final.to_csv(OUT_DIR / "final_actionable_slate.csv", index=False)

    lines = [
        "# Final Actionable Slate",
        "",
        f"- Number of tested configs in DSR benchmark: `{n_trials}`",
        f"- Deflated Sharpe benchmark (annualized): `{sr_star_ann:.3f}`",
        "",
        "## Top Actionable Names",
        "",
    ]
    for _, row in final.head(25).iterrows():
        lines.append(
            f"- `{row['pair']}` | action={row['actionable_bucket']} | DSR={row['dsr_prob']:.3f} | "
            f"AFML={int(row['afml_score'])}/4 | bucket={row['bucket']} | diversified12={bool(row['in_diversified_12'])}"
        )
    (OUT_DIR / "final_actionable_slate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
