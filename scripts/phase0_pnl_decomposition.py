"""Phase 0 â€” PnL decomposition audit for the canonical V2 trade log.

Investigates where the cycle-10 V2 NC PnL ($1,054,252) actually lives:
* By component: vega vs theta vs delta-hedge vs costs (fees + slippage +
  hedge_costs). Long vs short broken out separately.
* By regime: skew bucket, term-slope bucket, DVOL quintile, vol-of-vol
  regime, rolling spot-vol correlation regime.

The output drives the Phase 1 decision: which alt-structures (strangle,
risk-reversal, asymmetric strikes) might capture additional skew premium
without breaking the vol-arb edge.

Read-only â€” does not touch live state, paper.db, predictions, or aligner.

Usage::

    python scripts/phase0_pnl_decomposition.py

Outputs:

* ``vol_arb/output/PHASE0_DECOMP.md`` â€” markdown report
* ``vol_arb/output/PHASE0_DECOMP_GROUPS.parquet`` â€” full per-group ledger
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRADE_LOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"
OUT_MD = REPO_ROOT / "vol_arb" / "output" / "PHASE0_DECOMP.md"
OUT_PARQ = REPO_ROOT / "vol_arb" / "output" / "PHASE0_DECOMP_GROUPS.parquet"

NC_COLS = {
    "vega":   "vega_pnl_nc",
    "theta":  "theta_pnl_nc",
    "delta":  "delta_hedge_pnl_nc",
    "fees":   "option_fees_nc",
    "slip":   "slippage_nc",
    "hedge":  "hedge_costs_nc",
    "net":    "net_pnl_nc",
}


def df_to_md(df: pd.DataFrame) -> str:
    """Lightweight markdown-table formatter (avoids tabulate dependency)."""
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def _signed(df: pd.DataFrame, col: str) -> pd.Series:
    """Component PnL is already signed by signal in backtester output."""
    return df[col]


def component_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long/short Ã— {vega, theta, delta, fees, slip, hedge, net} table."""
    rows = []
    for sig, label in [(1, "long"), (-1, "short"), (None, "all")]:
        sub = df if sig is None else df.loc[df["signal"] == sig]
        n = int(len(sub))
        net = float(sub[NC_COLS["net"]].sum())
        per_trade = net / n if n else float("nan")
        rows.append({
            "side": label,
            "n_trades": n,
            "net_pnl": round(net, 0),
            "per_trade": round(per_trade, 1),
            "vega":  round(sub[NC_COLS["vega"]].sum(),  0),
            "theta": round(sub[NC_COLS["theta"]].sum(), 0),
            "delta": round(sub[NC_COLS["delta"]].sum(), 0),
            "fees":  round(sub[NC_COLS["fees"]].sum(),  0),
            "slip":  round(sub[NC_COLS["slip"]].sum(),  0),
            "hedge": round(sub[NC_COLS["hedge"]].sum(), 0),
        })
    return pd.DataFrame(rows)


def bucket_table(df: pd.DataFrame, by: str, bins: list, labels: list) -> pd.DataFrame:
    """Group fired trades by a continuous feature into named buckets."""
    if by not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[by]).copy()
    sub["bucket"] = pd.cut(sub[by], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for side_sig, side in [(1, "long"), (-1, "short")]:
        side_sub = sub.loc[sub["signal"] == side_sig]
        for bucket in labels:
            mask = side_sub["bucket"] == bucket
            if mask.sum() == 0:
                continue
            bk = side_sub.loc[mask]
            rows.append({
                "feature": by,
                "side": side,
                "bucket": str(bucket),
                "n": int(len(bk)),
                "net_pnl": round(bk[NC_COLS["net"]].sum(), 0),
                "per_trade": round(bk[NC_COLS["net"]].mean(), 1),
                "vega": round(bk[NC_COLS["vega"]].sum(), 0),
                "theta": round(bk[NC_COLS["theta"]].sum(), 0),
                "delta": round(bk[NC_COLS["delta"]].sum(), 0),
            })
    return pd.DataFrame(rows)


def quintile_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Group fired trades by quintile of a continuous feature."""
    if by not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=[by]).copy()
    try:
        sub["bucket"] = pd.qcut(sub[by], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for side_sig, side in [(1, "long"), (-1, "short")]:
        side_sub = sub.loc[sub["signal"] == side_sig]
        for bucket in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            mask = side_sub["bucket"] == bucket
            if mask.sum() == 0:
                continue
            bk = side_sub.loc[mask]
            rows.append({
                "feature": by,
                "side": side,
                "bucket": bucket,
                "n": int(len(bk)),
                "net_pnl": round(bk[NC_COLS["net"]].sum(), 0),
                "per_trade": round(bk[NC_COLS["net"]].mean(), 1),
                "vega": round(bk[NC_COLS["vega"]].sum(), 0),
                "theta": round(bk[NC_COLS["theta"]].sum(), 0),
                "delta": round(bk[NC_COLS["delta"]].sum(), 0),
            })
    return pd.DataFrame(rows)


def per_year_side_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year long vs short NC PnL â€” confirms the side asymmetry."""
    rows = []
    df2 = df.copy()
    df2["year"] = df2.index.year
    for year in sorted(df2["year"].unique()):
        for sig, side in [(1, "long"), (-1, "short")]:
            sub = df2.loc[(df2["year"] == year) & (df2["signal"] == sig)]
            if len(sub) == 0:
                continue
            rows.append({
                "year": int(year),
                "side": side,
                "n": int(len(sub)),
                "net_pnl": round(sub[NC_COLS["net"]].sum(), 0),
                "per_trade": round(sub[NC_COLS["net"]].mean(), 1),
                "vega": round(sub[NC_COLS["vega"]].sum(), 0),
                "theta": round(sub[NC_COLS["theta"]].sum(), 0),
                "delta": round(sub[NC_COLS["delta"]].sum(), 0),
            })
    return pd.DataFrame(rows)


def spot_vol_corr_regime(df: pd.DataFrame, window_h: int = 24*30) -> pd.Series:
    """Rolling spot-vol correlation over `window_h` hours.

    Spot returns: log(index_price).diff()
    Vol changes: svi_atm_1w.diff()
    Returns the per-bar correlation of the two series in the trailing window.
    """
    s_ret = np.log(df["index_price"]).diff()
    v_chg = df["svi_atm_1w"].diff()
    # 1 bar = 30 minutes (V2 holding) -> window_h hours = 2 * window_h bars
    w = 2 * window_h
    return s_ret.rolling(w, min_periods=w//4).corr(v_chg)


def attribution_summary(df: pd.DataFrame) -> str:
    """Component share of total NC PnL â€” answers 'where does the edge live?'"""
    total = df[NC_COLS["net"]].sum()
    parts = []
    for label, col in [("vega", "vega"), ("theta", "theta"), ("delta", "delta")]:
        s = df[NC_COLS[col]].sum()
        parts.append(f"{label}={s:>+9,.0f} ({s/total*100:>+6.1f}%)")
    costs = -(df[NC_COLS["fees"]].sum() + df[NC_COLS["slip"]].sum() + df[NC_COLS["hedge"]].sum())
    parts.append(f"net_costs={costs:>+9,.0f} ({costs/total*100:>+6.1f}%)")
    parts.append(f"net={total:>+9,.0f} (100.0%)")
    return "\n".join("  " + p for p in parts)


def main() -> int:
    print(f"[phase0] Reading {TRADE_LOG.relative_to(REPO_ROOT)} ...")
    tl = pd.read_parquet(TRADE_LOG)
    print(f"[phase0] {len(tl):,} bars, {(tl['signal']!=0).sum():,} fired trades.")

    fired = tl.loc[tl["signal"] != 0].copy()

    md = []
    md.append("# Phase 0 â€” V2 trade log PnL decomposition\n")
    md.append(f"Source: `vol_arb/output/BTC_v2_trade_log.parquet` "
              f"({fired.index.min().date()} â†’ {fired.index.max().date()})\n")
    md.append(f"Total fired trades: **{len(fired):,}**, "
              f"NC PnL **${fired[NC_COLS['net']].sum():,.0f}**\n")

    # Section 1: component breakdown
    md.append("## 1. Component breakdown (NC, signed)\n")
    md.append("```")
    md.append(attribution_summary(fired))
    md.append("```\n")
    comp = component_table(fired)
    md.append("### Long vs short Ã— component (USD, NC)\n")
    md.append(df_to_md(comp))
    md.append("")

    # Section 2: per-year by side
    md.append("\n## 2. Per-year long vs short asymmetry\n")
    yr = per_year_side_table(fired)
    md.append(df_to_md(yr))
    md.append("")

    # Section 3: skew buckets
    md.append("\n## 3. PnL by 1w 25Î” skew bucket\n")
    skew_bins = [-np.inf, -10, -5, 0, 5, 10, 20, 30, np.inf]
    skew_labels = ["<-10", "-10..-5", "-5..0", "0..5", "5..10", "10..20", "20..30", ">30"]
    sk = bucket_table(fired, "svi_25d_skew_1w", skew_bins, skew_labels)
    md.append(df_to_md(sk))
    md.append("")

    # Section 4: term slope buckets
    md.append("\n## 4. PnL by 1w-2w term slope bucket\n")
    ts_bins = [-np.inf, -5, -2, 0, 2, 5, np.inf]
    ts_labels = ["<-5", "-5..-2", "-2..0", "0..2", "2..5", ">5"]
    ts = bucket_table(fired, "svi_term_slope_1w_2w", ts_bins, ts_labels)
    md.append(df_to_md(ts))
    md.append("")

    # Section 5: DVOL quintile
    md.append("\n## 5. PnL by DVOL quintile\n")
    dv = quintile_table(fired, "dvol_close")
    md.append(df_to_md(dv))
    md.append("")

    # Section 6: vol-of-vol regime (rolling 30d std of svi_atm_1w)
    md.append("\n## 6. PnL by vol-of-vol regime (rolling 30d std of svi_atm_1w)\n")
    if "svi_atm_1w" in tl.columns:
        # Compute vol-of-vol on the FULL bar series, then attach to fired
        atm_1w = tl["svi_atm_1w"]
        # 30d * 48 bars/day = 1440 bars
        vov_full = atm_1w.diff().rolling(1440, min_periods=240).std()
        fired = fired.copy()
        fired["vov_30d"] = vov_full.reindex(fired.index)
        vov = quintile_table(fired, "vov_30d")
        md.append(df_to_md(vov))
    else:
        md.append("(svi_atm_1w not present)\n")
    md.append("")

    # Section 7: spot-vol correlation regime
    md.append("\n## 7. PnL by rolling spot-vol correlation regime (30d window)\n")
    if "index_price" in tl.columns and "svi_atm_1w" in tl.columns:
        sv_corr = spot_vol_corr_regime(tl, window_h=24*30)
        fired["spot_vol_corr_30d"] = sv_corr.reindex(fired.index)
        # Bin: < -0.3 (neg regime), -0.3..0 (mild neg), 0..0.3 (mild pos), > 0.3 (pos regime)
        corr_bins = [-1.0, -0.3, 0.0, 0.3, 1.0]
        corr_labels = ["neg(<-0.3)", "mild_neg(-0.3..0)", "mild_pos(0..0.3)", "pos(>0.3)"]
        cv = bucket_table(fired, "spot_vol_corr_30d", corr_bins, corr_labels)
        md.append(df_to_md(cv))
    else:
        md.append("(index_price or svi_atm_1w missing)\n")
    md.append("")

    # Section 8: GATE summary
    md.append("\n## 8. Phase-1 gate summary\n")
    long_per_trade = fired.loc[fired["signal"]==1, NC_COLS["net"]].mean()
    short_per_trade = fired.loc[fired["signal"]==-1, NC_COLS["net"]].mean()
    asym_ratio = long_per_trade / short_per_trade if short_per_trade != 0 else float("nan")
    md.append(f"- Long per-trade ${long_per_trade:.1f} vs short per-trade ${short_per_trade:.1f}: "
              f"**{asym_ratio:.2f}x** asymmetry\n")
    # Find max skew/regime concentration
    if not sk.empty:
        sk_sub = sk.assign(per=lambda x: x["per_trade"])
        idx = sk_sub["per"].abs().idxmax()
        b = sk_sub.loc[idx]
        md.append(f"- Max skew-bucket concentration: side={b['side']} "
                  f"bucket={b['bucket']} per_trade=${b['per']}\n")
    if not cv.empty:
        cv_sub = cv.assign(per=lambda x: x["per_trade"])
        idx2 = cv_sub["per"].abs().idxmax()
        b2 = cv_sub.loc[idx2]
        md.append(f"- Max spot-vol-corr-regime concentration: side={b2['side']} "
                  f"bucket={b2['bucket']} per_trade=${b2['per']}\n")

    # Save
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[phase0] wrote {OUT_MD.relative_to(REPO_ROOT)}")

    # Save group parquet for later programmatic access
    all_groups = pd.concat([
        sk.assign(grouping="skew_1w_25d"),
        ts.assign(grouping="term_slope_1w_2w"),
        dv.assign(grouping="dvol_quintile"),
    ], ignore_index=True)
    all_groups.to_parquet(OUT_PARQ)
    print(f"[phase0] wrote {OUT_PARQ.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
