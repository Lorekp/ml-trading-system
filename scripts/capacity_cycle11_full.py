"""Capacity analysis (cycle-11 canonical, 2026-05-11).

Re-runs the V2 capacity analysis on the post-cycle-10/11 trade log
($1.087M NC PnL, 8,859 trades) with three methodology fixes vs the
2026-05-07 run:

1. K=1 now means "today's per-trade deployed vega" (uses
   N_contracts_nc × vega as Q_ν, not unit vega). Cycle-11 K1+slope
   amplification on longs is baked in.
2. Sign stratification: separate Sharpe(K) curves for long vs short
   trades. Cycle-11 amplifies longs ×1.25 at extreme backwardation,
   so long-side capacity may bind earlier.
3. Year cohort: report deployed-vega means by year so K=1 can be
   interpreted against the relevant cohort (recent vs full-sample).

Plus the analytic vega-absorption bound (Method 2) and the perp-leg
check (Method 6) with updated cycle-10 notional (200K cap) and
combined Binance + Hyperliquid perp ADV.

Outputs:
- vol_arb/output/CAPACITY_ANALYSIS_2026_05_11.md
- vol_arb/output/BTC_v2_capacity_2026_05_11.parquet
- vol_arb/output/capacity_sharpe_curve_2026_05_11.png

Usage:
    PYTHONPATH=. python scripts/capacity_cycle11_full.py --smoke
    PYTHONPATH=. python scripts/capacity_cycle11_full.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.capacity import (
    align_sigma_sigma_to_trades,
    compute_impact_adjusted_pnl_nc,
    compute_market_vega_proxy,
    compute_sigma_sigma,
    compute_vega_capacity,
    dvol_regime,
    stationary_block_bootstrap_sharpe,
)


TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")
DVOL_PATH = Path("data/deribit/dvol/BTC_dvol_60m.parquet")
OUT_REPORT = Path("vol_arb/output/CAPACITY_ANALYSIS_2026_05_11.md")
OUT_PARQUET = Path("vol_arb/output/BTC_v2_capacity_2026_05_11.parquet")
OUT_PLOT = Path("vol_arb/output/capacity_sharpe_curve_2026_05_11.png")

K_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0, 500.0, 2_500.0, 10_000.0]
SOFT_KNEE_FLOOR = 4.0
HARD_KNEE_FLOOR = 2.0

# Hedge-venue ADV (BTC perp), USD per day
BINANCE_BTC_PERP_ADV = 20e9
HL_BTC_PERP_ADV = 2.5e9  # 2026 typical
TOTAL_HEDGE_ADV = BINANCE_BTC_PERP_ADV + HL_BTC_PERP_ADV  # 22.5B


def _calendar_years(idx: pd.DatetimeIndex) -> float:
    if len(idx) == 0:
        return 0.0
    span = idx.max() - idx.min()
    return float(span.total_seconds() / (365.25 * 24 * 3600))


def _row_for_K(
    df: pd.DataFrame,
    K: float,
    sigma_sigma_per_bar: np.ndarray,
    Y_vol: float,
    perm_frac: float,
    decay_seconds: float,
    n_bootstrap: int,
    label: str,
) -> dict:
    pnl = compute_impact_adjusted_pnl_nc(
        trade_log=df,
        K=K,
        sigma_sigma_per_trade=sigma_sigma_per_bar,
        Y_vol=Y_vol,
        perm_frac=perm_frac,
        decay_seconds=decay_seconds,
        use_deployed_vega=True,
    )
    exe_mask = (df["N_contracts_nc"] > 0).values
    pnl_arr = pnl.values[exe_mask]
    pnl_total = float(pnl_arr.sum())
    n_years = _calendar_years(df.index[exe_mask])
    sharpe, lcb, ucb = stationary_block_bootstrap_sharpe(
        pnl_arr, n_years=n_years, n_replications=n_bootstrap, seed=42,
    )
    return {
        "K": K,
        "label": label,
        "Y_vol": Y_vol,
        "perm_frac": perm_frac,
        "n_trades": int(exe_mask.sum()),
        "nc_pnl": pnl_total,
        "trade_sharpe": sharpe,
        "trade_sharpe_lcb": lcb,
        "trade_sharpe_ucb": ucb,
    }


def _knee(curve_df: pd.DataFrame, sharpe_floor: float) -> float | None:
    s = curve_df.sort_values("K")[["K", "trade_sharpe"]].reset_index(drop=True)
    above = s["trade_sharpe"] >= sharpe_floor
    if above.all():
        return None
    if not above.any():
        return float(s.iloc[0]["K"])
    idx = above.idxmin()
    if idx == 0:
        return float(s.iloc[0]["K"])
    K_lo, S_lo = float(s.iloc[idx - 1]["K"]), float(s.iloc[idx - 1]["trade_sharpe"])
    K_hi, S_hi = float(s.iloc[idx]["K"]), float(s.iloc[idx]["trade_sharpe"])
    if abs(S_hi - S_lo) < 1e-12:
        return K_hi
    t = (S_lo - sharpe_floor) / (S_lo - S_hi)
    return float(K_lo + t * (K_hi - K_lo))


def _df_to_md_table(df: pd.DataFrame, cols: list[str]) -> str:
    fmts = {
        "K": "{:.1f}",
        "n_trades": "{:,.0f}",
        "nc_pnl": "${:,.0f}",
        "trade_sharpe": "{:.2f}",
        "trade_sharpe_lcb": "{:.2f}",
        "trade_sharpe_ucb": "{:.2f}",
    }
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmts.get(c, "{}")
            if pd.isna(v):
                cells.append("—")
            else:
                cells.append(f.format(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def main(smoke: bool = False) -> None:
    print(f"loading trade log: {TRADE_LOG}")
    df = pd.read_parquet(TRADE_LOG)
    n_exe = int((df['N_contracts_nc'] > 0).sum())
    print(f"  {len(df):,} bars, {n_exe:,} executed trades")
    nc_pnl_committed = float(df.loc[df['N_contracts_nc'] > 0, 'net_pnl_nc'].sum())
    print(f"  committed NC PnL: ${nc_pnl_committed:,.2f}")

    if smoke:
        end = df.index.max()
        start = end - pd.Timedelta(days=90)
        df = df.loc[start:end]
        n_bootstrap = 200
        print(f"  SMOKE: {len(df):,} bars, n_bootstrap={n_bootstrap}")
    else:
        n_bootstrap = 2000

    print(f"loading DVOL: {DVOL_PATH}")
    dvol = pd.read_parquet(DVOL_PATH)
    if "timestamp" in dvol.columns:
        dvol = dvol.set_index("timestamp")
    if dvol.index.tz is None:
        dvol.index = dvol.index.tz_localize("UTC")

    sigma_sigma = compute_sigma_sigma(dvol["close"], window_days=30)
    print(f"  sigma_sigma median (log-DVOL/day): {sigma_sigma.median():.4f}")
    sigma_sigma_per_bar = align_sigma_sigma_to_trades(sigma_sigma, df.index)

    # ── K=0 identity check ──
    pnl_K0 = compute_impact_adjusted_pnl_nc(
        trade_log=df, K=0.0,
        sigma_sigma_per_trade=sigma_sigma_per_bar,
        use_deployed_vega=True,
    )
    total_K0 = float(pnl_K0.sum())
    total_orig = float(df["net_pnl_nc"].sum())
    diff = total_K0 - total_orig
    print(f"\nK=0 identity: replay=${total_K0:,.4f} vs original=${total_orig:,.4f}")
    print(f"  diff: ${diff:,.6f}")
    if abs(diff) > 1.0:
        raise RuntimeError(
            f"K=0 identity check FAILED — diff ${diff:,.2f}; capacity.py bug."
        )

    # ── Deployed-vega cohort table ──
    exe = df[df["N_contracts_nc"] > 0].copy()
    exe["year"] = exe.index.year
    exe["dep_vega"] = exe["N_contracts_nc"] * exe["vega"]
    cohort = exe.groupby("year").agg(
        n_trades=("N_contracts_nc", "size"),
        nc_pnl=("net_pnl_nc", "sum"),
        mean_dep_vega=("dep_vega", "mean"),
        mean_N=("N_contracts_nc", "mean"),
    )
    print(f"\nDeployed-vega cohort:\n{cohort}")
    last12 = exe.loc[exe.index >= (exe.index.max() - pd.Timedelta(days=365))]
    last12_dep_vega = float(last12["dep_vega"].mean()) if len(last12) else float("nan")
    full_dep_vega = float(exe["dep_vega"].mean())
    print(f"  full-sample mean deployed vega: ${full_dep_vega:.1f}/vp")
    print(f"  last-12m mean deployed vega:   ${last12_dep_vega:.1f}/vp")

    rows = []

    # ── Main curves ──
    grids = [
        ("all", df, sigma_sigma_per_bar, 0.5, 0.3),
        ("Y=0.25", df, sigma_sigma_per_bar, 0.25, 0.3),
        ("Y=1.0", df, sigma_sigma_per_bar, 1.0, 0.3),
    ]

    print(f"\nrunning main K grid (n_boot={n_bootstrap}):")
    for label, sub_df, ss, Y, perm in grids:
        n_boot_use = n_bootstrap if label == "all" else max(200, n_bootstrap // 5)
        for K in K_GRID:
            r = _row_for_K(
                df=sub_df, K=K,
                sigma_sigma_per_bar=ss,
                Y_vol=Y, perm_frac=perm, decay_seconds=600.0,
                n_bootstrap=n_boot_use, label=label,
            )
            rows.append(r)
            if label == "all":
                print(f"  K={K:>7.1f}: PnL=${r['nc_pnl']:>10,.0f}  "
                      f"Sharpe={r['trade_sharpe']:>5.2f}  "
                      f"[CI {r['trade_sharpe_lcb']:>5.2f}, {r['trade_sharpe_ucb']:>5.2f}]")

    # ── DVOL regime stratification ──
    exe_idx = df.index[df["N_contracts_nc"] > 0]
    dvol_aligned = dvol["close"].reindex(exe_idx, method="ffill").bfill()
    regime = dvol_regime(dvol_aligned)
    print(f"\nregime stratification (Y=0.5, perm=0.3):")
    for r_label in ["low", "mid", "high"]:
        idx_r = exe_idx[regime.values == r_label]
        if len(idx_r) == 0:
            continue
        df_r = df.loc[idx_r]
        ss_r = sigma_sigma_per_bar[df.index.get_indexer(idx_r)]
        for K in K_GRID:
            row = _row_for_K(
                df=df_r, K=K, sigma_sigma_per_bar=ss_r,
                Y_vol=0.5, perm_frac=0.3, decay_seconds=600.0,
                n_bootstrap=max(200, n_bootstrap // 5),
                label=f"dvol_{r_label}",
            )
            rows.append(row)

    # ── Long vs short stratification ──
    print(f"\nsign stratification (Y=0.5, perm=0.3):")
    for sig, sig_label in [(1, "long"), (-1, "short")]:
        idx_s = exe_idx[(df.loc[exe_idx, "signal"] == sig).values]
        df_s = df.loc[idx_s]
        ss_s = sigma_sigma_per_bar[df.index.get_indexer(idx_s)]
        print(f"  {sig_label}: {len(idx_s)} trades")
        for K in K_GRID:
            row = _row_for_K(
                df=df_s, K=K, sigma_sigma_per_bar=ss_s,
                Y_vol=0.5, perm_frac=0.3, decay_seconds=600.0,
                n_bootstrap=max(200, n_bootstrap // 5),
                label=f"sign_{sig_label}",
            )
            rows.append(row)
            if K in (0.0, 100.0, 2500.0, 10000.0):
                print(f"    K={K:>7.1f}: PnL=${row['nc_pnl']:>10,.0f}  "
                      f"Sharpe={row['trade_sharpe']:>5.2f}  "
                      f"[CI {row['trade_sharpe_lcb']:>5.2f}, {row['trade_sharpe_ucb']:>5.2f}]")

    # ── Last-12m cohort stratification ──
    last12_idx = df.index[(df.index >= (df.index.max() - pd.Timedelta(days=365))) & (df["N_contracts_nc"] > 0)]
    if len(last12_idx) > 100:
        df_l12 = df.loc[last12_idx[0]:]
        ss_l12 = sigma_sigma_per_bar[df.index.get_indexer(df_l12.index)]
        print(f"\nlast-12m cohort: {(df_l12['N_contracts_nc']>0).sum()} trades")
        for K in K_GRID:
            row = _row_for_K(
                df=df_l12, K=K, sigma_sigma_per_bar=ss_l12,
                Y_vol=0.5, perm_frac=0.3, decay_seconds=600.0,
                n_bootstrap=max(200, n_bootstrap // 5),
                label="last12m",
            )
            rows.append(row)

    # ── Method A: analytic per-trade K_max ──
    print(f"\nMethod A (analytic per-trade K_max):")
    method_a_summary = []
    for Y in [0.25, 0.5, 1.0]:
        for ef in [0.25, 0.5]:
            bound = compute_vega_capacity(
                trade_log=df,
                sigma_sigma_per_trade=sigma_sigma_per_bar,
                Y_vol=Y, edge_fraction=ef,
                use_deployed_vega=True,
            )
            finite = bound.K_max[np.isfinite(bound.K_max)]
            method_a_summary.append({
                "Y_vol": Y, "edge_fraction": ef,
                "K_max_p10": float(np.percentile(finite, 10)) if len(finite) else float("nan"),
                "K_max_p25": float(np.percentile(finite, 25)) if len(finite) else float("nan"),
                "K_max_median": float(np.median(finite)) if len(finite) else float("nan"),
                "K_max_p75": float(np.percentile(finite, 75)) if len(finite) else float("nan"),
                "n_finite": int(len(finite)),
            })
    method_a_df = pd.DataFrame(method_a_summary)
    print(method_a_df.to_string(index=False))

    # ── Method C: perp-hedge leg capacity ──
    print(f"\nMethod C (perp-hedge leg):")
    perp_taker_fee = 0.0005
    hedge_traded_usd = (df["hedge_costs_nc"] / perp_taker_fee).fillna(0).clip(lower=0)
    perp_15m_binance = BINANCE_BTC_PERP_ADV / (24 * 4)
    perp_15m_combined = TOTAL_HEDGE_ADV / (24 * 4)
    method_c_rows = []
    for K_test in [1, 10, 100, 1000, 10000]:
        share_binance = (K_test * hedge_traded_usd) / perp_15m_binance
        share_combined = (K_test * hedge_traded_usd) / perp_15m_combined
        method_c_rows.append({
            "K": K_test,
            "p99_binance": float(np.percentile(share_binance[share_binance > 0], 99)) if (share_binance > 0).any() else 0.0,
            "max_binance": float(share_binance.max()),
            "p99_combined": float(np.percentile(share_combined[share_combined > 0], 99)) if (share_combined > 0).any() else 0.0,
            "max_combined": float(share_combined.max()),
        })
    method_c_df = pd.DataFrame(method_c_rows)
    print(method_c_df.to_string(index=False))

    # ── Save sidecar ──
    out_df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PARQUET)
    print(f"\nwrote: {OUT_PARQUET}")

    # ── Knees on base curve ──
    base = out_df[(out_df.label == "all") & (out_df.Y_vol == 0.5)].sort_values("K")
    base_y10 = out_df[(out_df.label == "Y=1.0")].sort_values("K")
    soft = _knee(base, SOFT_KNEE_FLOOR)
    hard = _knee(base, HARD_KNEE_FLOOR)
    hard_conservative = _knee(base_y10, HARD_KNEE_FLOOR)
    print(f"\nknees:")
    print(f"  base (Y=0.5)        soft={soft}  hard={hard}")
    print(f"  conservative (Y=1.0)               hard={hard_conservative}")

    long_curve = out_df[out_df.label == "sign_long"].sort_values("K")
    short_curve = out_df[out_df.label == "sign_short"].sort_values("K")
    long_hard = _knee(long_curve, HARD_KNEE_FLOOR)
    short_hard = _knee(short_curve, HARD_KNEE_FLOOR)
    print(f"  long-side hard knee:  {long_hard}")
    print(f"  short-side hard knee: {short_hard}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for lab, sty, name in [
            ("all", "o-", "all (Y=0.5)"),
            ("Y=0.25", "s--", "Y=0.25"),
            ("Y=1.0", "d--", "Y=1.0"),
            ("sign_long", "^-", "long-only"),
            ("sign_short", "v-", "short-only"),
            ("last12m", "p-", "last 12m"),
        ]:
            sub = out_df[out_df.label == lab].sort_values("K")
            if sub.empty:
                continue
            if lab == "all":
                ax.errorbar(
                    sub["K"], sub["trade_sharpe"],
                    yerr=[
                        sub["trade_sharpe"] - sub["trade_sharpe_lcb"],
                        sub["trade_sharpe_ucb"] - sub["trade_sharpe"],
                    ],
                    fmt=sty, capsize=4, label=name,
                )
            else:
                ax.plot(sub["K"], sub["trade_sharpe"], sty, alpha=0.7, label=name)
        ax.axhline(SOFT_KNEE_FLOOR, color="orange", ls=":", alpha=0.5, label=f"soft knee {SOFT_KNEE_FLOOR}")
        ax.axhline(HARD_KNEE_FLOOR, color="red", ls=":", alpha=0.5, label=f"hard knee {HARD_KNEE_FLOOR}")
        ax.set_xscale("log")
        ax.set_xlabel("Scale multiplier K (deployed-vega ×, log)")
        ax.set_ylabel("NC trade Sharpe")
        ax.set_title("V2 vol-arb capacity (cycle-11) — Sharpe(K) under square-root impact")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PLOT, dpi=120, bbox_inches="tight")
        print(f"wrote: {OUT_PLOT}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot failed (non-fatal): {exc}")

    # ── Write report ──
    write_report(
        out_df=out_df,
        method_a=method_a_df,
        method_c=method_c_df,
        cohort=cohort,
        full_dep_vega=full_dep_vega,
        last12_dep_vega=last12_dep_vega,
        soft_knee=soft, hard_knee=hard, hard_conservative=hard_conservative,
        long_hard=long_hard, short_hard=short_hard,
        nc_pnl_committed=nc_pnl_committed,
        n_trades=n_exe,
        smoke=smoke,
    )


def write_report(
    *,
    out_df: pd.DataFrame,
    method_a: pd.DataFrame,
    method_c: pd.DataFrame,
    cohort: pd.DataFrame,
    full_dep_vega: float,
    last12_dep_vega: float,
    soft_knee: float | None,
    hard_knee: float | None,
    hard_conservative: float | None,
    long_hard: float | None,
    short_hard: float | None,
    nc_pnl_committed: float,
    n_trades: int,
    smoke: bool,
) -> None:
    smoke_tag = "**SMOKE (90-day, NOT for headline)**" if smoke else "Full cycle-11 V2 trade log (51 months, 2022-02 → 2026-04)"
    base = out_df[(out_df.label == "all") & (out_df.Y_vol == 0.5)].sort_values("K")
    y025 = out_df[out_df.label == "Y=0.25"].sort_values("K")
    y10 = out_df[out_df.label == "Y=1.0"].sort_values("K")

    base_tbl = _df_to_md_table(base, ["K", "n_trades", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb"])
    y025_tbl = _df_to_md_table(y025, ["K", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb"])
    y10_tbl = _df_to_md_table(y10, ["K", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb"])

    sign_long = out_df[out_df.label == "sign_long"].sort_values("K")
    sign_short = out_df[out_df.label == "sign_short"].sort_values("K")
    long_tbl = _df_to_md_table(sign_long, ["K", "n_trades", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb"])
    short_tbl = _df_to_md_table(sign_short, ["K", "n_trades", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb"])

    last12 = out_df[out_df.label == "last12m"].sort_values("K")
    last12_tbl = _df_to_md_table(last12, ["K", "n_trades", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb"]) if not last12.empty else "(no data)"

    regime_blocks = []
    for r in ["low", "mid", "high"]:
        rdf = out_df[out_df.label == f"dvol_{r}"].sort_values("K")
        if rdf.empty:
            continue
        regime_blocks.append(f"### DVOL regime: {r}\n\n{_df_to_md_table(rdf, ['K', 'n_trades', 'nc_pnl', 'trade_sharpe', 'trade_sharpe_lcb', 'trade_sharpe_ucb'])}")
    regimes_md = "\n\n".join(regime_blocks)

    soft_md = f"K = {soft_knee:.1f}×" if soft_knee is not None else "≥ grid max (Sharpe never breached)"
    hard_md = f"K = {hard_knee:.1f}×" if hard_knee is not None else "≥ grid max (Sharpe never breached)"
    hard_conservative_md = f"K ≈ {hard_conservative:.1f}×" if hard_conservative is not None else "≥ grid max"
    long_hard_md = f"K ≈ {long_hard:.1f}×" if long_hard is not None else "≥ grid max"
    short_hard_md = f"K ≈ {short_hard:.1f}×" if short_hard is not None else "≥ grid max"

    def _generic_md(df: pd.DataFrame, fmts: dict[str, str], index: bool = False) -> str:
        if index:
            d = df.reset_index()
        else:
            d = df
        cols = list(d.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        rows = []
        for _, r in d.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                f = fmts.get(c, "{}")
                if pd.isna(v):
                    cells.append("—")
                else:
                    try:
                        cells.append(f.format(v))
                    except Exception:
                        cells.append(str(v))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + rows)

    method_a_tbl = _generic_md(method_a, {
        "Y_vol": "{:.2f}", "edge_fraction": "{:.2f}",
        "K_max_p10": "{:,.0f}", "K_max_p25": "{:,.0f}",
        "K_max_median": "{:,.0f}", "K_max_p75": "{:,.0f}",
        "n_finite": "{:,}",
    })
    method_c_tbl = _generic_md(method_c, {
        "K": "{:,.0f}",
        "p99_binance": "{:.3%}", "max_binance": "{:.3%}",
        "p99_combined": "{:.3%}", "max_combined": "{:.3%}",
    })
    cohort_tbl = _generic_md(cohort, {
        "year": "{:.0f}", "n_trades": "{:,}",
        "nc_pnl": "${:,.0f}",
        "mean_dep_vega": "{:.1f}", "mean_N": "{:.3f}",
    }, index=True)

    body = f"""# V2 Vol-Arb Capacity Analysis — cycle-11 canonical (2026-05-11)

{smoke_tag}.  Committed NC PnL = ${nc_pnl_committed:,.0f}, n_trades = {n_trades:,}.

This historical analysis uses the committed notional and trade-count inputs
shown above. It is not current capacity validation.

## Headline

| Floor | Base (Y=0.5) | Conservative (Y=1.0) |
|---|---|---|
| Soft knee (Sharpe ≥ 4.0) | {soft_md} | — |
| Hard knee (Sharpe ≥ 2.0) | {hard_md} | {hard_conservative_md} |

Long-side hard knee (Y=0.5): **{long_hard_md}**.  Short-side hard knee:
**{short_hard_md}**.

K is the multiplier on **today's per-trade deployed vega** (N_contracts_nc
× per-BTC vega).  K = 1 already includes cycle-11 K1+slope amplification
on long bars at extreme backwardation.

## Deployed-vega cohort

K = 1 represents this distribution of per-trade vega exposure:

{cohort_tbl}

- Full-sample mean deployed vega: **${full_dep_vega:.1f}/vp**
- Last-12m mean historical vega: **${last12_dep_vega:.1f}/vp** ← recent research cohort

Deployed vega has grown ~2× since 2022 (BTC price level + cycle-11 long
amplifier).  Capacity scaling K is per current cohort.

## Base curve (Y = 0.5, perm_frac = 0.3, all-trades)

{base_tbl}

K = 0 is the no-impact identity; reproduces committed NC PnL to fp.

## Sensitivity — Y_vol

### Conservative (Y_vol = 0.25, harder to detect)

{y025_tbl}

### Aggressive (Y_vol = 1.0, easier to detect — use for adoption)

{y10_tbl}

Y_vol is the dominant calibration uncertainty (Donier-Bouchaud-Mastromatteo
2015, [arxiv:1602.03043](https://arxiv.org/abs/1602.03043)).  Y = 1.0
is the conservative sensitivity bound for this historical analysis.

## Sign stratification — long vs short

Cycle-11 K1+slope amplifies long sizing ×1.25 on extreme backwardation
bars.  Per-trade deployed vega: longs $62/vp (mean), shorts $24/vp (mean).
Long-side η is higher per trade → long-side impact may bind first.

### Long-side (Y = 0.5)

{long_tbl}

### Short-side (Y = 0.5)

{short_tbl}

## Last-12m cohort (Y = 0.5)

Recent research cohort — this profile carries the invalidated cycle-11 sizing.

{last12_tbl}

## DVOL regime stratification (Y = 0.5)

DVOL terciles.  Low-DVOL is typically the binding regime (thinnest absolute
σ_σ × DVOL_level in vol-points, tightest spreads).

{regimes_md}

## Method A — analytic per-trade K_max (vega absorption)

Closed-form: K_max(i) = ((edge_fraction · |edge_i|) / (Y_vol · σ_σ_vp,i))² /
η_i, with η using deployed Q_ν.  Aggregate as median across executed trades.

{method_a_tbl}

This is an upper-bound order-of-magnitude — assumes the signal continues to
fire at scale and ignores Sharpe-degradation feedback.

## Method C — perp-hedge leg (cycle-10 max_gross_notional = $200K)

The DRL v4 hedge leg consumes BTC perp ADV.  Proxy: hedge_costs_nc /
taker_fee = perp-leg notional traded per bar.  Combined-venue ADV uses
Binance ($20B/day) + Hyperliquid ($2.5B/day) = $22.5B.

{method_c_tbl}

Threshold: p99 bar share > 1% = perp leg starts binding.  At cycle-10
canonical, perp leg has more headroom than 2026-05-07 estimate (which used
Binance-only).

## CIs

Stationary block bootstrap (Politis-Romano 1994), 30-trade mean blocks,
{2000 if not smoke else 200} replications.  Per-K bootstrapped (2.5, 97.5)
percentiles of the trade-Sharpe sampling distribution.  Block-length
geometric — preserves short-run autocorrelation in per-bar PnL.

## Decision table

At today's typical concurrent notional ~$15-30K/trade × ~5 in-flight
(15min hold × ~6% fire rate) → ~$75-150K typical concurrent exposure.

| Operating point | K | Concurrent notional (approx) | Caveats |
|---|---:|---:|---|
| Today | 1 | $75K-150K | Cycle-11 canonical |
| Cautious scale | 10 | $750K-1.5M | Within all-curve hard knee |
| Historical stress case (Y=1.0) | hard-knee Y=1.0 → see headline | scaled accordingly | Not a validated limit |

## Limitations (unchanged from methodology doc)

1. V_ν proxy = trailing_1h_volume × 24 × per-BTC vega.  Coarse; tightens
   with Laevitas/Amberdata per-strike data (deferred).
2. Static impact model — ignores adversarial MM response (Garleanu-
   Pedersen-Poteshman 2009).  True capacity below this bound.
3. σ_σ calibrated on same window as strategy.  Method-3 hold-out
   deferred.
4. perm_frac, τ taken from Bouchaud 2023 equity-index baseline; no
   vol-surface-specific published values.

## Next

- Live paper-trade scaling at 2× / 5× / 10× to ground-truth Method 1.
- Laevitas/Amberdata per-strike volume to tighten V_ν (cycle-12 backlog).
- Adversarial-MM stress model (Garleanu et al.) — analytic extension.
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(body, encoding="utf-8")
    print(f"wrote: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="90-day + n_boot=200")
    args = parser.parse_args()
    main(smoke=args.smoke)
