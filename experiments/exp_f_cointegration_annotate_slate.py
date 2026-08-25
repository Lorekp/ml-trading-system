"""Annotate the ranked cointegration slate with strengths and risks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
RANKED_CSV = OUT_DIR / "ranked_slate_full.csv"
ANNOTATED_CSV = OUT_DIR / "ranked_slate_annotated.csv"
ANNOTATED_MD = OUT_DIR / "ranked_slate_annotated.md"


def add_tag(tags: list[str], condition: bool, label: str) -> None:
    if condition:
        tags.append(label)


def main() -> None:
    ranked = pd.read_csv(RANKED_CSV)
    top20 = ranked.head(20).copy()

    full_asset_counts = Counter()
    top20_asset_counts = Counter()
    for _, row in ranked.iterrows():
        full_asset_counts[row["y_name"]] += 1
        full_asset_counts[row["x_name"]] += 1
    for _, row in top20.iterrows():
        top20_asset_counts[row["y_name"]] += 1
        top20_asset_counts[row["x_name"]] += 1

    strength_list = []
    risk_list = []
    bucket_list = []

    for _, row in ranked.iterrows():
        strengths: list[str] = []
        risks: list[str] = []

        add_tag(strengths, row["afml_score"] >= 4, "AFML-aligned")
        add_tag(strengths, row["n_months"] >= 24, "long-sample")
        add_tag(strengths, row["base_max_dd_pct"] <= 15.0, "contained-drawdown")
        add_tag(strengths, row["stress_cost_plus20_sharpe"] >= 1.0, "cost-robust")
        add_tag(strengths, row["hood_median_sharpe"] >= 1.0, "parameter-stable")
        add_tag(strengths, bool(row["trend_both_positive"]), "trend-balanced")
        add_tag(strengths, bool(row["vol_both_positive"]), "vol-balanced")
        add_tag(strengths, row["psr_sr0"] >= 0.99, "high-psr")

        add_tag(risks, row["n_months"] < 12, "short-sample")
        add_tag(risks, row["base_max_dd_pct"] > 25.0, "high-drawdown")
        add_tag(risks, not bool(row["trend_both_positive"]), "trend-asymmetric")
        add_tag(risks, not bool(row["vol_both_positive"]), "vol-asymmetric")
        add_tag(risks, row["hood_median_sharpe"] < 0.75, "parameter-fragile")
        add_tag(risks, row["stress_cost_plus20_sharpe"] < 0.75, "cost-sensitive")
        add_tag(risks, top20_asset_counts[row["y_name"]] >= 3 or top20_asset_counts[row["x_name"]] >= 3, "top20-clustered")
        add_tag(risks, full_asset_counts[row["y_name"]] >= 10 or full_asset_counts[row["x_name"]] >= 10, "universe-clustered")

        if row["afml_score"] == 4 and row["n_months"] >= 24 and row["base_max_dd_pct"] <= 20:
            bucket = "Core"
        elif row["afml_score"] == 4 and row["n_months"] >= 12:
            bucket = "Strong"
        elif row["afml_score"] >= 3:
            bucket = "Watch"
        else:
            bucket = "Fragile"

        strength_list.append(", ".join(strengths) if strengths else "none")
        risk_list.append(", ".join(risks) if risks else "none")
        bucket_list.append(bucket)

    ranked["strength_tags"] = strength_list
    ranked["risk_tags"] = risk_list
    ranked["bucket"] = bucket_list
    ranked["y_name_full_count"] = ranked["y_name"].map(full_asset_counts)
    ranked["x_name_full_count"] = ranked["x_name"].map(full_asset_counts)
    ranked["y_name_top20_count"] = ranked["y_name"].map(top20_asset_counts)
    ranked["x_name_top20_count"] = ranked["x_name"].map(top20_asset_counts)
    ranked.to_csv(ANNOTATED_CSV, index=False)

    lines = [
        "# Ranked Slate Annotated",
        "",
        "- `bucket=Core` means AFML 4/4, longer sample, and contained drawdown.",
        "- `bucket=Strong` means robust but with a lighter sample or some concentration risk.",
        "- `bucket=Watch` means still interesting, but a senior quant would want tighter monitoring.",
        "",
        "## Top 25",
        "",
    ]
    for _, row in ranked.head(25).iterrows():
        lines.append(
            f"- `#{int(row['rank'])} {row['pair']}` | {row['bucket']} | "
            f"strengths: {row['strength_tags']} | risks: {row['risk_tags']}"
        )
    ANNOTATED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
