"""Foundation alignment audit — 2026-04-17.

Compares a predictions parquet against TSRV at multiple temporal offsets to
determine which 15-min window the predictions actually target. Used both to
(a) record the pre-fix baseline (saved production) and (b) verify the post-fix
fresh retrain targets the intended forecast window [T, T+15m].

Design:
  For each feature timestamp T in the predictions parquet, look up TSRV ln(vol_ann)
  at offsets {-15m, 0m, +15m, +30m, +45m} and compute MSE vs y_pred. The offset
  with lowest MSE identifies which window the model targets.

  * offset -15 means y_pred matches vol over [T-30m, T-15m] (two bars in the past)
  * offset   0 means y_pred matches vol over [T-15m, T]    (nowcast of recent bar)
  * offset +15 means y_pred matches vol over [T,    T+15m] (intended 1-bar forecast)
  * offset +30 means y_pred matches vol over [T+15m, T+30m] (2-bar forecast)
  * offset +45 means y_pred matches vol over [T+30m, T+45m] (3-bar forecast)

TSRV parquet is END-stamped for BTC (post-2026-04-11 migration), so vol_ann at
index T represents vol over [T-15m, T].

Usage:
    python -m vol_model.run_foundation_alignment_audit \
        --predictions vol_model/output/BTCUSDT_pruned_svi_predictions.parquet \
        --tsrv TSRV/output/BTCUSDT_tsrv_15m.parquet \
        --out vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md \
        [--label "pre-fix saved production"] \
        [--append]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OFFSETS_MIN = [-15, 0, 15, 30, 45]


def compute_mse_table(
    preds: pd.DataFrame,
    tsrv: pd.DataFrame,
    y_pred_col: str = "y_pred",
    vol_col: str = "vol_ann",
    offsets_min: list[int] | None = None,
) -> pd.DataFrame:
    """MSE, mean residual, corr of y_pred vs ln(vol_ann) at each offset.

    Returns a DataFrame indexed by offset with columns [mse, mean_r, corr, n].
    """
    if offsets_min is None:
        offsets_min = OFFSETS_MIN

    rows = []
    y_pred = preds[y_pred_col].values
    feat_idx = preds.index
    for off in offsets_min:
        target_ts = feat_idx + pd.Timedelta(minutes=off)
        v = tsrv[vol_col].reindex(target_ts).values
        lv = np.log(np.clip(v, 1e-8, None))
        valid = np.isfinite(lv) & np.isfinite(y_pred)
        if valid.sum() < 10:
            rows.append({"offset_min": off, "mse": np.nan, "mean_r": np.nan, "corr": np.nan, "n": int(valid.sum())})
            continue
        r = y_pred[valid] - lv[valid]
        rows.append({
            "offset_min": off,
            "mse": float((r ** 2).mean()),
            "mean_r": float(r.mean()),
            "corr": float(np.corrcoef(y_pred[valid], lv[valid])[0, 1]),
            "n": int(valid.sum()),
        })
    return pd.DataFrame(rows).set_index("offset_min")


def monthly_mse_table(
    preds: pd.DataFrame,
    tsrv: pd.DataFrame,
    y_pred_col: str = "y_pred",
    vol_col: str = "vol_ann",
    offsets_min: list[int] | None = None,
) -> pd.DataFrame:
    """Per-month MSE at each offset. Returns DataFrame[month × offset]."""
    if offsets_min is None:
        offsets_min = OFFSETS_MIN
    out = []
    months = preds.index.to_period("M").unique()
    for m in months:
        sub = preds[preds.index.to_period("M") == m]
        if len(sub) == 0:
            continue
        t = compute_mse_table(sub, tsrv, y_pred_col, vol_col, offsets_min)
        row = {"month": str(m), "n": len(sub)}
        for off in offsets_min:
            row[f"mse_{off:+d}"] = t.loc[off, "mse"]
        row["best_offset"] = t["mse"].idxmin()
        out.append(row)
    return pd.DataFrame(out).set_index("month")


def persistence_correlations(tsrv: pd.DataFrame, vol_col: str = "vol_ann") -> dict:
    """corr(ln_vol[T-15m,T], ln_vol[T,T+15m]) etc. — the 'vol persistence ceiling'."""
    lv = np.log(np.clip(tsrv[vol_col].values, 1e-8, None))
    out = {}
    for lag in [1, 2, 3, 4]:
        x = lv[:-lag]
        y = lv[lag:]
        valid = np.isfinite(x) & np.isfinite(y)
        out[f"corr_lag_{lag}bar"] = float(np.corrcoef(x[valid], y[valid])[0, 1])
    return out


def regime_breakdown(
    preds: pd.DataFrame,
    tsrv: pd.DataFrame,
    offset_min: int = 15,
    y_pred_col: str = "y_pred",
    vol_col: str = "vol_ann",
    n_buckets: int = 3,
) -> pd.DataFrame:
    """MSE at a fixed offset, stratified by vol regime (terciles of vol_ann at T)."""
    feat_idx = preds.index
    y = preds[y_pred_col].values
    v_at_T = tsrv[vol_col].reindex(feat_idx).values
    target_ts = feat_idx + pd.Timedelta(minutes=offset_min)
    v_target = tsrv[vol_col].reindex(target_ts).values
    lv_target = np.log(np.clip(v_target, 1e-8, None))
    valid = np.isfinite(lv_target) & np.isfinite(y) & np.isfinite(v_at_T)
    df = pd.DataFrame({
        "y_pred": y[valid],
        "lv_target": lv_target[valid],
        "vol_at_T": v_at_T[valid],
    })
    df["r2"] = (df["y_pred"] - df["lv_target"]) ** 2
    df["bucket"] = pd.qcut(df["vol_at_T"], q=n_buckets, labels=[f"q{i+1}" for i in range(n_buckets)])
    grp = df.groupby("bucket", observed=True)["r2"].agg(["count", "mean"]).rename(columns={"mean": "mse"})
    return grp


def render_markdown(
    label: str,
    pred_path: Path,
    tsrv_path: Path,
    overall_table: pd.DataFrame,
    monthly_table: pd.DataFrame,
    persistence: dict,
    regime_table: pd.DataFrame,
) -> str:
    lines = []
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"Predictions: `{pred_path}`")
    lines.append(f"TSRV: `{tsrv_path}`")
    lines.append(f"Rows: {int(overall_table['n'].iloc[0])}")
    lines.append("")
    lines.append("### Overall MSE-vs-offset")
    lines.append("")
    lines.append("| offset (min) | interpretation | MSE | mean residual | corr | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    interp = {
        -15: "vol over [T-30m, T-15m] (past 2 bars)",
        0: "vol over [T-15m, T] — **nowcast** (pre-fix)",
        15: "vol over [T, T+15m] — **intended forecast** (post-fix)",
        30: "vol over [T+15m, T+30m] (2 bars ahead)",
        45: "vol over [T+30m, T+45m] (3 bars ahead)",
    }
    for off in overall_table.index:
        r = overall_table.loc[off]
        lines.append(f"| {off:+d} | {interp.get(off, '')} | {r['mse']:.4f} | {r['mean_r']:+.4f} | {r['corr']:.3f} | {int(r['n'])} |")
    best = int(overall_table["mse"].idxmin())
    lines.append("")
    lines.append(f"**Best-MSE offset: `{best:+d}` min** — "
                 f"{'NOWCAST (pre-fix)' if best == 0 else 'FORECAST of [T, T+15m] (post-fix target)' if best == 15 else f'unexpected offset {best:+d}'}")
    lines.append("")
    lines.append("### Per-month best-offset distribution")
    lines.append("")
    best_month_cts = monthly_table["best_offset"].value_counts().sort_index()
    lines.append("| best_offset | months_count |")
    lines.append("|---|---:|")
    for off, ct in best_month_cts.items():
        lines.append(f"| {off:+d} | {ct} |")
    lines.append("")
    lines.append(f"Total months covered: {len(monthly_table)}")
    lines.append("")
    lines.append("### Vol persistence ceiling (TSRV ln_vol lag correlations)")
    lines.append("")
    lines.append("| lag | corr(ln_vol[t], ln_vol[t+lag]) |")
    lines.append("|---|---:|")
    for k, v in persistence.items():
        lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    lines.append("Interpretation: `corr_lag_1bar` is how much raw autocorrelation exists in ln_vol at 15-min lag. "
                 "A poorly-aligned model still gets high corr just from persistence; the gap between persistence-only "
                 "and model MSE is the 'true skill' of the prediction.")
    lines.append("")
    lines.append("### Regime breakdown at offset +15 (vol of [T, T+15m])")
    lines.append("")
    lines.append("| vol_at_T bucket | count | MSE |")
    lines.append("|---|---:|---:|")
    for bucket, row in regime_table.iterrows():
        lines.append(f"| {bucket} | {int(row['count'])} | {row['mse']:.4f} |")
    lines.append("")
    lines.append("Expect high-vol buckets to have higher MSE (vol is harder to predict when larger). "
                 "Uniform MSE across buckets would indicate an additive-error model.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--tsrv", type=Path, default=Path("TSRV/output/BTCUSDT_tsrv_15m.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", type=str, default="Audit")
    ap.add_argument("--append", action="store_true", help="Append to existing out file (else overwrite)")
    ap.add_argument("--y-pred-col", default="y_pred")
    ap.add_argument("--vol-col", default="vol_ann")
    args = ap.parse_args()

    preds = pd.read_parquet(args.predictions)
    tsrv = pd.read_parquet(args.tsrv)

    # If predictions index is not a DatetimeIndex, try to find one
    if not isinstance(preds.index, pd.DatetimeIndex):
        # Look for a 'timestamp' or 'ts' column
        for col in ("timestamp", "ts"):
            if col in preds.columns:
                preds = preds.set_index(col)
                break

    # Only use rows that have non-null y_pred
    mask = preds[args.y_pred_col].notna()
    preds = preds[mask]
    print(f"[audit] predictions: {len(preds)} rows with non-null {args.y_pred_col!r}")
    print(f"[audit] TSRV: {len(tsrv)} rows, index: {tsrv.index.min()} -> {tsrv.index.max()}")

    overall = compute_mse_table(preds, tsrv, args.y_pred_col, args.vol_col)
    monthly = monthly_mse_table(preds, tsrv, args.y_pred_col, args.vol_col)
    persistence = persistence_correlations(tsrv, args.vol_col)
    regime = regime_breakdown(preds, tsrv, offset_min=15, y_pred_col=args.y_pred_col, vol_col=args.vol_col)

    md = render_markdown(args.label, args.predictions, args.tsrv, overall, monthly, persistence, regime)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and args.out.exists() else "w"
    if mode == "w":
        header = "# Foundation Alignment Audit — 2026-04-17\n\n"
    else:
        header = "\n---\n\n"
    with args.out.open(mode, encoding="utf-8") as f:
        f.write(header)
        f.write(md)

    print(f"[audit] wrote {args.out} (mode={mode})")
    print()
    print(overall.to_string())
    print()
    print(f"Best-MSE offset: {int(overall['mse'].idxmin()):+d} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
