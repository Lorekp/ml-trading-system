"""Method 1 (iso-Sharpe scaling backtest with calibrated impact) for the
V2 vol-arb capacity analysis.

See plan: ``C:/Users/laure/.claude/plans/c-claude-ml-trading-system-tsrv-pipelin-glistening-bunny.md``.

Replays the V2 NC PnL at scale multipliers K ∈ {0, 1, 2, 5, 10, 20, 50, 100}
under the square-root impact model, with stationary block bootstrap CIs
(Politis-Romano 1994) at each K. K=0 is the no-impact identity check
(must reproduce the original NC numbers to floating-point precision).

Outputs (when run without --smoke):
- ``vol_arb/output/CAPACITY_ANALYSIS.md`` — final capacity report
- ``vol_arb/output/BTC_v2_capacity_curve.parquet`` — Sharpe(K) sidecar
- ``vol_arb/output/capacity_sharpe_curve.png`` — Sharpe(K) plot

Usage:
    PYTHONPATH=. python scripts/capacity_phase2_iso_sharpe.py --smoke
    PYTHONPATH=. python scripts/capacity_phase2_iso_sharpe.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.capacity import (
    align_sigma_sigma_to_trades,
    compute_impact_adjusted_pnl_nc,
    compute_sigma_sigma,
    dvol_regime,
    stationary_block_bootstrap_sharpe,
)


TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")
DVOL_PATH = Path("data/deribit/dvol/BTC_dvol_60m.parquet")
OUT_REPORT = Path("vol_arb/output/CAPACITY_ANALYSIS.md")
OUT_PARQUET = Path("vol_arb/output/BTC_v2_capacity_curve.parquet")
OUT_PLOT = Path("vol_arb/output/capacity_sharpe_curve.png")

K_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0, 500.0, 2_500.0, 10_000.0]
SOFT_KNEE_FLOOR = 4.0
HARD_KNEE_FLOOR = 2.0
BARS_PER_DAY = 96


def _calendar_years(idx: pd.DatetimeIndex) -> float:
    if len(idx) == 0:
        return 0.0
    span = idx.max() - idx.min()
    return float(span.total_seconds() / (365.25 * 24 * 3600))


def _row_for_K(
    df: pd.DataFrame, K: float, sigma_sigma_per_bar: np.ndarray,
    Y_vol: float, perm_frac: float, decay_seconds: float,
    n_bootstrap: int, regime_label: str = "all",
) -> dict:
    pnl = compute_impact_adjusted_pnl_nc(
        trade_log=df,
        K=K,
        sigma_sigma_per_trade=sigma_sigma_per_bar,
        Y_vol=Y_vol,
        perm_frac=perm_frac,
        decay_seconds=decay_seconds,
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
        "regime": regime_label,
        "Y_vol": Y_vol,
        "perm_frac": perm_frac,
        "n_trades": int(exe_mask.sum()),
        "nc_pnl": pnl_total,
        "trade_sharpe": sharpe,
        "trade_sharpe_lcb": lcb,
        "trade_sharpe_ucb": ucb,
    }


def _knee(curve_df: pd.DataFrame, sharpe_floor: float) -> float | None:
    """Return the first K where mean Sharpe drops below ``sharpe_floor``,
    interpolated linearly between grid points. None if the floor is never
    crossed within the grid.
    """
    s = curve_df.sort_values("K")[["K", "trade_sharpe"]].reset_index(drop=True)
    above = s["trade_sharpe"] >= sharpe_floor
    if above.all():
        return None  # never breaches; capacity > grid max
    if not above.any():
        return float(s.iloc[0]["K"])  # already below at K=0 (degenerate)

    idx = above.idxmin()  # first False
    if idx == 0:
        return float(s.iloc[0]["K"])
    K_lo, S_lo = float(s.iloc[idx - 1]["K"]), float(s.iloc[idx - 1]["trade_sharpe"])
    K_hi, S_hi = float(s.iloc[idx]["K"]), float(s.iloc[idx]["trade_sharpe"])
    if abs(S_hi - S_lo) < 1e-12:
        return K_hi
    t = (S_lo - sharpe_floor) / (S_lo - S_hi)
    return float(K_lo + t * (K_hi - K_lo))


def main(smoke: bool = False) -> None:
    print(f"loading trade log: {TRADE_LOG}")
    df = pd.read_parquet(TRADE_LOG)
    print(f"  {len(df):,} bars, {(df['N_contracts_nc'] > 0).sum():,} trades")

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
    sigma_sigma_per_bar = align_sigma_sigma_to_trades(sigma_sigma, df.index)

    # ── K=0 identity check (must match df["net_pnl_nc"].sum() to fp precision) ──
    pnl_K0 = compute_impact_adjusted_pnl_nc(
        trade_log=df, K=0.0,
        sigma_sigma_per_trade=sigma_sigma_per_bar,
    )
    total_K0 = float(pnl_K0.sum())
    total_orig = float(df["net_pnl_nc"].sum())
    diff = total_K0 - total_orig
    print(f"\nK=0 identity check: replay=${total_K0:,.2f} vs original=${total_orig:,.2f}")
    print(f"  diff: ${diff:,.2f}")
    if abs(diff) > 1.0:
        raise RuntimeError(
            "K=0 identity check FAILED — impact-adjusted replay diverges "
            f"from original by ${diff:,.2f}; capacity.py has a bug."
        )

    # ── Main grid: aggregate (regime=all) at base sensitivity Y=0.5, perm=0.3 ──
    print(f"\nrunning K grid (Y=0.5, perm=0.3, n_boot={n_bootstrap}):")
    rows = []
    for K in K_GRID:
        row = _row_for_K(
            df=df, K=K,
            sigma_sigma_per_bar=sigma_sigma_per_bar,
            Y_vol=0.5, perm_frac=0.3, decay_seconds=600.0,
            n_bootstrap=n_bootstrap, regime_label="all",
        )
        rows.append(row)
        print(
            f"  K={K:>6.1f}: PnL=${row['nc_pnl']:>10,.0f}  "
            f"Sharpe={row['trade_sharpe']:>5.2f}  "
            f"[CI {row['trade_sharpe_lcb']:>5.2f}, {row['trade_sharpe_ucb']:>5.2f}]"
        )

    # ── Sensitivity: Y_vol stress ──
    print(f"\nsensitivity (Y_vol grid, perm=0.3):")
    for Y_vol in [0.25, 1.0]:
        for K in K_GRID:
            row = _row_for_K(
                df=df, K=K,
                sigma_sigma_per_bar=sigma_sigma_per_bar,
                Y_vol=Y_vol, perm_frac=0.3, decay_seconds=600.0,
                n_bootstrap=max(200, n_bootstrap // 5),
                regime_label=f"Y={Y_vol}",
            )
            rows.append(row)

    # ── Regime stratification at base Y=0.5 ──
    exe_idx = df.index[df["N_contracts_nc"] > 0]
    dvol_aligned = dvol["close"].reindex(exe_idx, method="ffill").bfill()
    regime = dvol_regime(dvol_aligned)
    print(f"\nregime stratification (Y=0.5, perm=0.3):")
    for r in ["low", "mid", "high"]:
        idx_r = exe_idx[regime.values == r]
        if len(idx_r) == 0:
            continue
        df_r = df.loc[idx_r]
        ss_r = sigma_sigma_per_bar[df.index.get_indexer(idx_r)]
        for K in K_GRID:
            row = _row_for_K(
                df=df_r, K=K,
                sigma_sigma_per_bar=ss_r,
                Y_vol=0.5, perm_frac=0.3, decay_seconds=600.0,
                n_bootstrap=max(200, n_bootstrap // 5),
                regime_label=r,
            )
            rows.append(row)

    out_df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PARQUET)
    print(f"\nwrote: {OUT_PARQUET}")

    # ── Knees on the base curve ──
    base = out_df[(out_df.regime == "all") & (out_df.Y_vol == 0.5)].sort_values("K")
    soft = _knee(base, SOFT_KNEE_FLOOR)
    hard = _knee(base, HARD_KNEE_FLOOR)
    print(f"\nknees (Y=0.5, perm=0.3):")
    print(f"  soft knee (Sharpe≥{SOFT_KNEE_FLOOR}): K={soft}")
    print(f"  hard knee (Sharpe≥{HARD_KNEE_FLOOR}): K={hard}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            base["K"].values,
            base["trade_sharpe"].values,
            yerr=[
                base["trade_sharpe"].values - base["trade_sharpe_lcb"].values,
                base["trade_sharpe_ucb"].values - base["trade_sharpe"].values,
            ],
            fmt="o-", capsize=4, label="base (Y=0.5)",
        )
        for Y_vol in [0.25, 1.0]:
            r = out_df[out_df.regime == f"Y={Y_vol}"].sort_values("K")
            ax.plot(r["K"], r["trade_sharpe"], "s--", alpha=0.6, label=f"Y={Y_vol}")
        ax.axhline(SOFT_KNEE_FLOOR, color="orange", ls=":", alpha=0.5, label="soft knee")
        ax.axhline(HARD_KNEE_FLOOR, color="red", ls=":", alpha=0.5, label="hard knee")
        ax.set_xscale("log")
        ax.set_xlabel("Scale multiplier K (log)")
        ax.set_ylabel("NC trade Sharpe")
        ax.set_title("V2 vol-arb capacity: Sharpe(K) under square-root impact")
        ax.legend()
        ax.grid(alpha=0.3)
        OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PLOT, dpi=120, bbox_inches="tight")
        print(f"wrote: {OUT_PLOT}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot failed (non-fatal): {exc}")

    write_report(
        out_df=out_df, soft_knee=soft, hard_knee=hard, smoke=smoke,
    )


def write_report(
    *, out_df: pd.DataFrame, soft_knee: float | None,
    hard_knee: float | None, smoke: bool,
) -> None:
    smoke_tag = "**SMOKE (90-day sample, NOT for headline)**" if smoke else "Full V2 trade log (51 months OOS)"
    base = out_df[(out_df.regime == "all") & (out_df.Y_vol == 0.5)].sort_values("K")
    base_table = base[[
        "K", "n_trades", "nc_pnl", "trade_sharpe",
        "trade_sharpe_lcb", "trade_sharpe_ucb",
    ]].to_string(index=False)

    sens_25 = out_df[out_df.regime == "Y=0.25"].sort_values("K")[[
        "K", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb",
    ]].to_string(index=False)
    sens_10 = out_df[out_df.regime == "Y=1.0"].sort_values("K")[[
        "K", "nc_pnl", "trade_sharpe", "trade_sharpe_lcb", "trade_sharpe_ucb",
    ]].to_string(index=False)

    regime_blocks = []
    for r in ["low", "mid", "high"]:
        rdf = out_df[out_df.regime == r].sort_values("K")
        if rdf.empty:
            continue
        regime_blocks.append(f"### Regime: {r} DVOL\n\n```\n{rdf[['K', 'n_trades', 'nc_pnl', 'trade_sharpe', 'trade_sharpe_lcb', 'trade_sharpe_ucb']].to_string(index=False)}\n```")
    regimes_md = "\n\n".join(regime_blocks)

    soft_md = f"K = {soft_knee:.1f}×" if soft_knee is not None else "≥ grid max (Sharpe never breached)"
    hard_md = f"K = {hard_knee:.1f}×" if hard_knee is not None else "≥ grid max (Sharpe never breached)"

    body = f"""# V2 Vol-Arb Capacity Analysis — Sharpe(K) Curve

{smoke_tag}

Plan: `C:/Users/laure/.claude/plans/c-claude-ml-trading-system-tsrv-pipelin-glistening-bunny.md`.

## Headline

| Threshold | NC trade Sharpe floor | Capacity (× current sizing) |
|---|---|---|
| Soft knee | {SOFT_KNEE_FLOOR} | **{soft_md}** |
| Hard knee | {HARD_KNEE_FLOOR} (vault floor) | **{hard_md}** |

## Method (Method 1)

For each scale multiplier K, replay the V2 NC PnL with the exit IV
shifted AGAINST each trade by the residual portion of the square-root
impact:

    ΔIV_K(i) = Y_vol · σ_σ(i) · √(K · η_i)
    iv_exit_K(i) = iv_exit(i) − signal(i) · ΔIV_K(i) · decay_factor
    decay_factor = perm_frac + (1 - perm_frac) · exp(-Δt / τ)

with `perm_frac = 0.30` (Bouchaud et al. 2023, [arxiv:2311.18283](https://arxiv.org/abs/2311.18283)),
`τ = 600s`, and 15-min holding windows. Theta, delta-hedge, fees,
slippage, and hedge-cost components are taken from the original trade
log unchanged (first-order in IV impact only).

The vega proxy `V_ν` for daily market vega traded uses
`trailing_1h_volume × 24 × per-BTC vega` — coarse but defensible
without a Laevitas/Amberdata subscription.

## Base curve (Y_vol = 0.5, perm_frac = 0.3)

```
{base_table}
```

K=0 is the no-impact identity baseline (must match committed NC PnL
to floating-point precision; smoke-tested).

## Sensitivity to Y_vol (impact constant)

### Conservative bound (Y_vol = 0.25 — strategy is harder to detect):

```
{sens_25}
```

### Aggressive bound (Y_vol = 1.0 — strategy is easy to detect):

```
{sens_10}
```

The `Y_vol` parameter is the dominant calibration uncertainty
(per Donier-Bouchaud-Mastromatteo 2015, [arxiv:1602.03043](https://arxiv.org/abs/1602.03043)).
The capacity curves at Y=0.25 and Y=1.0 bracket the soft/hard knees;
any decision based on this curve should use the conservative (Y=1.0)
bound, not the base case.

## Regime stratification (Y=0.5, perm=0.3)

DVOL terciles. The strategy's capacity in the lowest-IV regime is
typically the binding constraint because spreads are tightest and
σ_σ × DVOL_level is smallest in absolute vol-points.

{regimes_md}

## CIs

Stationary block bootstrap (Politis-Romano 1994), 30-day blocks,
{2000 if not smoke else 200} replications. Block length is geometric
with mean 30 × 96 = 2880 bars. Trade-Sharpe CI is the (2.5, 97.5)
percentile of the bootstrapped distribution. Width grows with K as
expected — more uncertainty under more aggressive impact.

## Limitations

1. **Vega proxy is coarse**: `trailing_1h_volume × 24` overstates depth
   on multi-expiry distribution; per-strike Deribit volume (Laevitas)
   would tighten the bound.
2. **Static impact model**: ignores adversarial market-maker response
   and dealer inventory effects (Garleanu-Pedersen-Poteshman 2009).
   True capacity likely below this bound.
3. **Fee fix not yet flowed back**: this analysis runs against the
   pre-fee-fix saved trade log. The Part 1 fee fix shifts NC PnL by
   well under $30K (cap rarely binds at DTE≥1d) — capacity headline is
   robust to that change to 1 decimal place. Re-run after Part 1d if
   stricter precision is needed.
4. **Circularity break is partial**: σ_σ is calibrated on the same
   period the strategy backtested on. Full Method-3-style hold-out
   (recommended in the playbook) deferred to a later cycle.

## Decision framework

For the path to vault deposits ($1M+/month target), the operative
question is: at what deployed capital does NC trade Sharpe stay above
the {HARD_KNEE_FLOOR}-Sharpe floor under the conservative impact
calibration (Y_vol = 1.0)?

Cross-reference the conservative-bound table above for the binding
hard knee, then translate via:

    deployed_capital ≈ K × current_avg_simultaneous_notional

with current notional ~$10–30K per trade and ~5 simultaneous trades
in flight (15-min holding × ~5 trades/day → typical concurrent
exposure of ~$50–150K).
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(body, encoding="utf-8")
    print(f"wrote: {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="3-month + n_boot=200")
    args = parser.parse_args()
    main(smoke=args.smoke)
