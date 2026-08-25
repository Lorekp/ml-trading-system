"""Focused CPCV/CSCV-style validation for current research candidates.

Uses each candidate's current config plus the same-pair config family from
`backtests.csv` to estimate:
- candidate OOS robustness across combinatorial time splits
- how often the candidate is selected by in-sample ranking within its family
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "cointegration_rigorous"
RIGOROUS_PATH = ROOT / "experiments" / "exp_f_cointegration_rigorous.py"

INPUT_RESEARCH = OUT_DIR / "research_candidates.csv"
INPUT_FINAL = OUT_DIR / "final_actionable_slate_pbo.csv"
INPUT_BACKTESTS = OUT_DIR / "backtests.csv"


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


def annualized_sharpe(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    if std_ret <= 0 or not np.isfinite(std_ret):
        return float("nan")
    return float(np.sqrt(12.0) * mean_ret / std_ret)


def config_key(pair: str, lookback_months: int, entry_z: float, cost_bps: float) -> str:
    return f"{pair}|lb={int(lookback_months)}|z={float(entry_z):.1f}|c={float(cost_bps):.1f}"


def monthly_pnl_frame(segments: list[dict[str, object]], entry_z: float, cost_bps: float) -> pd.DataFrame:
    rows = []
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
        rows.append({"month": seg["month"], "pnl": float(pnl)})
    return pd.DataFrame(rows)


def choose_split_scheme(n_months: int) -> tuple[int, int, str]:
    if n_months >= 12:
        return 6, 2, "afml_default"
    if n_months >= 8:
        return 4, 1, "small_sample_fallback"
    return 0, 0, "insufficient_history"


def make_groups(months: list[str], n_groups: int) -> list[list[str]]:
    arr = np.array(months, dtype=object)
    return [chunk.tolist() for chunk in np.array_split(arr, n_groups)]


def score_series(series: pd.Series, months: list[str]) -> float:
    arr = series.reindex(months).dropna().to_numpy(dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    return annualized_sharpe(arr)


def cpcv_for_candidate(candidate_row: pd.Series, family_df: pd.DataFrame, rigorous) -> dict[str, object]:
    target_key = candidate_row["config_key"]
    monthly_frames = []
    for _, row in family_df.iterrows():
        row_dict = row.to_dict()
        segments = rigorous.prepare_segments(row_dict, int(row["lookback_months"]))
        monthly = monthly_pnl_frame(segments, float(row["entry_z"]), float(row["cost_bps"]))
        if monthly.empty:
            continue
        key = config_key(row["pair"], int(row["lookback_months"]), float(row["entry_z"]), float(row["cost_bps"]))
        monthly["config_key"] = key
        monthly_frames.append(monthly)

    if not monthly_frames:
        return {
            "pair": candidate_row["pair"],
            "config_key": target_key,
            "cpcv_scheme": "no_data",
            "cpcv_paths": 0,
            "family_size": 0,
            "own_cpcv_mean_oos_sharpe": float("nan"),
            "own_cpcv_median_oos_sharpe": float("nan"),
            "own_cpcv_p10_oos_sharpe": float("nan"),
            "own_cpcv_positive_share": float("nan"),
            "family_selected_count": 0,
            "family_selected_mean_oos_sharpe": float("nan"),
            "family_selected_positive_share": float("nan"),
            "cpcv_verdict": "insufficient",
        }

    monthly_all = pd.concat(monthly_frames, ignore_index=True)
    matrix = monthly_all.pivot(index="month", columns="config_key", values="pnl").sort_index()
    months = matrix.index.tolist()
    n_groups, k_test, scheme = choose_split_scheme(len(months))
    if n_groups == 0 or target_key not in matrix.columns:
        return {
            "pair": candidate_row["pair"],
            "config_key": target_key,
            "cpcv_scheme": scheme,
            "cpcv_paths": 0,
            "family_size": int(matrix.shape[1]),
            "own_cpcv_mean_oos_sharpe": float("nan"),
            "own_cpcv_median_oos_sharpe": float("nan"),
            "own_cpcv_p10_oos_sharpe": float("nan"),
            "own_cpcv_positive_share": float("nan"),
            "family_selected_count": 0,
            "family_selected_mean_oos_sharpe": float("nan"),
            "family_selected_positive_share": float("nan"),
            "cpcv_verdict": "insufficient",
        }

    groups = make_groups(months, n_groups)
    combos = list(itertools.combinations(range(n_groups), k_test))
    own_oos = []
    selected_oos = []
    selected_count = 0

    for combo in combos:
        test_months = [m for gid in combo for m in groups[gid]]
        train_months = [m for gid in range(n_groups) if gid not in combo for m in groups[gid]]
        is_scores = {col: score_series(matrix[col], train_months) for col in matrix.columns}
        valid = [col for col, val in is_scores.items() if np.isfinite(val)]
        if target_key not in valid or len(valid) < 2:
            continue
        best_key = max(valid, key=lambda col: is_scores[col])
        target_oos = score_series(matrix[target_key], test_months)
        if np.isfinite(target_oos):
            own_oos.append(float(target_oos))
        best_oos = score_series(matrix[best_key], test_months)
        if best_key == target_key:
            selected_count += 1
            if np.isfinite(best_oos):
                selected_oos.append(float(best_oos))

    own_arr = np.asarray(own_oos, dtype=np.float64)
    selected_arr = np.asarray(selected_oos, dtype=np.float64)
    own_mean = float(np.mean(own_arr)) if own_arr.size else float("nan")
    own_median = float(np.median(own_arr)) if own_arr.size else float("nan")
    own_p10 = float(np.percentile(own_arr, 10)) if own_arr.size >= 3 else float("nan")
    own_pos = float(np.mean(own_arr > 0.0)) if own_arr.size else float("nan")
    sel_mean = float(np.mean(selected_arr)) if selected_arr.size else float("nan")
    sel_pos = float(np.mean(selected_arr > 0.0)) if selected_arr.size else float("nan")

    if own_arr.size == 0:
        verdict = "insufficient"
    elif own_mean > 1.0 and (np.isnan(own_p10) or own_p10 > 0.0) and own_pos >= 0.60:
        verdict = "strong"
    elif own_mean > 0.0 and own_pos >= 0.50:
        verdict = "provisional"
    else:
        verdict = "weak"

    return {
        "pair": candidate_row["pair"],
        "config_key": target_key,
        "cpcv_scheme": scheme,
        "cpcv_paths": int(own_arr.size),
        "family_size": int(matrix.shape[1]),
        "own_cpcv_mean_oos_sharpe": own_mean,
        "own_cpcv_median_oos_sharpe": own_median,
        "own_cpcv_p10_oos_sharpe": own_p10,
        "own_cpcv_positive_share": own_pos,
        "family_selected_count": int(selected_count),
        "family_selected_mean_oos_sharpe": sel_mean,
        "family_selected_positive_share": sel_pos,
        "cpcv_verdict": verdict,
    }


def final_status(row: pd.Series) -> str:
    if row["post_pbo_bucket"] == "Research Live Watch":
        if row["cpcv_verdict"] == "strong":
            return "Research Live Watch"
        if row["cpcv_verdict"] == "provisional":
            return "Research Hold"
        return "Reject for now"

    if row["cpcv_verdict"] == "strong":
        return "Research Live Watch"
    if row["cpcv_verdict"] == "provisional":
        return "Research Hold"
    return "Reject for now"


def main() -> None:
    research = pd.read_csv(INPUT_RESEARCH)
    final = pd.read_csv(INPUT_FINAL)
    backtests = pd.read_csv(INPUT_BACKTESTS)

    merged = research.merge(
        final[["pair", "y_name", "x_name", "lookback_months", "entry_z", "cost_bps", "validation_key", "post_pbo_bucket"]],
        on="pair",
        how="left",
        suffixes=("", "_final"),
    )
    merged["config_key"] = merged.apply(
        lambda row: config_key(row["pair"], int(row["lookback_months"]), float(row["entry_z"]), float(row["cost_bps"])),
        axis=1,
    )

    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        rigorous = load_rigorous_module()
        rigorous.ensure_dirs()
        rigorous._FULL_CLOSE_DF = rigorous.build_universe()

        rows = []
        for idx, row in merged.iterrows():
            print(f"[CPCV] {idx + 1}/{len(merged)} {row['pair']}", flush=True)
            family = backtests.loc[backtests["pair"] == row["pair"]].copy()
            rows.append(cpcv_for_candidate(row, family, rigorous))
    finally:
        os.chdir(original_cwd)

    cpcv = pd.DataFrame(rows)
    cpcv.to_csv(OUT_DIR / "research_candidates_cpcv.csv", index=False)

    updated = merged.merge(cpcv, on=["pair", "config_key"], how="left")
    updated["post_cpcv_status"] = updated.apply(final_status, axis=1)
    updated.to_csv(OUT_DIR / "research_candidates_validated.csv", index=False)

    lines = [
        "# Research Candidates CPCV",
        "",
        "- Focused same-pair family CPCV/CSCV-style validation",
        "- AFML default split: 6 groups / 2 test groups when months >= 12",
        "- Small-sample fallback: 4 groups / 1 test group when months in [8, 11]",
        "",
        "## Results",
        "",
    ]
    for _, row in updated.sort_values(["post_cpcv_status", "candidate_rank"]).iterrows():
        lines.append(
            f"- `{row['pair']}` | prior={row['post_pbo_bucket']} | CPCV={row['cpcv_verdict']} | "
            f"mean OOS Sharpe={row['own_cpcv_mean_oos_sharpe']:.2f} | "
            f"selected_count={int(row['family_selected_count'])} | final={row['post_cpcv_status']}"
        )
    (OUT_DIR / "research_candidates_validated.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
