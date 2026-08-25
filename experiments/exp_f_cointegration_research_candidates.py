"""Build the dedicated research-candidate list from the latest post-PBO slate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
INPUT_FINAL = OUT_DIR / "final_actionable_slate_pbo.csv"


def primary_risk(risk_tags: str) -> str:
    tags = [t.strip() for t in str(risk_tags).split(",") if t.strip()]
    priority = [
        "short-sample",
        "high-drawdown",
        "trend-asymmetric",
        "vol-asymmetric",
        "parameter-fragile",
        "top20-clustered",
        "universe-clustered",
    ]
    for item in priority:
        if item in tags:
            return item
    return tags[0] if tags else "none"


def afml_next_step(row: pd.Series) -> tuple[str, str]:
    risk = primary_risk(row["risk_tags"])
    status = row["post_pbo_bucket"]

    if status == "Research Live Watch":
        if risk == "short-sample":
            return (
                "CPCV/CSCV path validation with updated monthly history",
                "AFML Ch.11-12 (PBO, CSCV, CPCV)",
            )
        return (
            "Focused CPCV plus deflated-Sharpe recalibration on local parameter neighborhood",
            "AFML Ch.11-12 and Ch.14",
        )

    if risk == "short-sample":
        return (
            "Extend track record and rerun PSR/DSR plus CPCV on the same config family",
            "AFML Ch.14 and Ch.11-12",
        )
    if risk in {"trend-asymmetric", "vol-asymmetric"}:
        return (
            "Run structural-break and regime-stability checks before any further ranking",
            "AFML Ch.17 (CUSUM / Chu-Stinchcombe-White)",
        )
    if risk == "parameter-fragile":
        return (
            "Run tighter local neighborhood sweep and CSCV selection test",
            "AFML Ch.11-12 and Ch.14",
        )
    if risk == "high-drawdown":
        return (
            "Run CPCV with drawdown-focused acceptance thresholds and stress costs",
            "AFML Ch.12 and Ch.14",
        )
    return (
        "Re-run focused PBO/DSR on finalists and compare against diversified portfolio contribution",
        "AFML Ch.11, Ch.14, Ch.16",
    )


def evidence_note(row: pd.Series) -> str:
    selected = int(row["selected_count"]) if pd.notna(row["selected_count"]) else 0
    mean_oos = row["mean_selected_oos_sharpe"]
    if selected > 0 and pd.notna(mean_oos):
        return f"PBO selected {selected}x, mean OOS Sharpe {mean_oos:.2f}"
    if bool(row["dsr_ok"]):
        return f"DSR pass {row['dsr_prob']:.3f}, but no strong CSCV selector evidence yet"
    return f"AFML/robustness hold only; DSR {row['dsr_prob']:.3f}"


def main() -> None:
    final = pd.read_csv(INPUT_FINAL)
    research = final.loc[final["post_pbo_bucket"].isin(["Research Live Watch", "Research Hold"])].copy()
    research = research.sort_values(
        ["post_pbo_bucket", "dsr_ok", "mean_selected_oos_sharpe", "rank"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    research["candidate_rank"] = range(1, len(research) + 1)
    research["main_risk"] = research["risk_tags"].map(primary_risk)
    steps = research.apply(afml_next_step, axis=1)
    research["next_afml_validation"] = [step for step, _ in steps]
    research["afml_reference"] = [ref for _, ref in steps]
    research["evidence_note"] = research.apply(evidence_note, axis=1)

    cols = [
        "candidate_rank",
        "pair",
        "post_pbo_bucket",
        "rank",
        "bucket",
        "dsr_prob",
        "selected_count",
        "mean_selected_oos_sharpe",
        "n_months",
        "base_sharpe",
        "base_return_pct",
        "base_max_dd_pct",
        "main_risk",
        "evidence_note",
        "next_afml_validation",
        "afml_reference",
    ]
    research[cols].to_csv(OUT_DIR / "research_candidates.csv", index=False)

    lines = [
        "# Research Candidates",
        "",
        "- Source: `final_actionable_slate_pbo.csv`",
        "- Inclusion rule: `post_pbo_bucket` in `{Research Live Watch, Research Hold}`",
        "",
        "## Candidates",
        "",
    ]
    for _, row in research[cols].iterrows():
        lines.append(
            f"- `#{int(row['candidate_rank'])} {row['pair']}` | status={row['post_pbo_bucket']} | "
            f"DSR={row['dsr_prob']:.3f} | months={int(row['n_months'])} | "
            f"main risk={row['main_risk']} | evidence={row['evidence_note']} | "
            f"next={row['next_afml_validation']} | ref={row['afml_reference']}"
        )
    (OUT_DIR / "research_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
