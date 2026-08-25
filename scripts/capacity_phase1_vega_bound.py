"""Method 2 (analytic vega-absorption bound) + Method 6 quick-check
(perp leg non-binding) for the V2 vol-arb capacity analysis.

Outputs (when run without --smoke):
- ``vol_arb/output/CAPACITY_PHASE1_VEGA_BOUND.md`` — methodology + result
- ``vol_arb/output/BTC_v2_capacity_phase1.parquet`` — per-trade K_max,
  η, σ_σ, regime label

Sensitivity stress: Y_vol ∈ {0.25, 0.5, 1.0}; edge_fraction ∈ {0.25, 0.5}.
Regime split: DVOL terciles (low / mid / high). Aggregate K_max =
median of per-trade values within each regime, then min across regimes
weighted by historical time share.

Usage:
    PYTHONPATH=. python scripts/capacity_phase1_vega_bound.py --smoke
    PYTHONPATH=. python scripts/capacity_phase1_vega_bound.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.capacity import (
    align_sigma_sigma_to_trades,
    compute_market_vega_proxy,
    compute_sigma_sigma,
    compute_vega_capacity,
    dvol_regime,
)


TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")
DVOL_PATH = Path("data/deribit/dvol/BTC_dvol_60m.parquet")
OUT_REPORT = Path("vol_arb/output/CAPACITY_PHASE1_VEGA_BOUND.md")
OUT_PARQUET = Path("vol_arb/output/BTC_v2_capacity_phase1.parquet")


def main(smoke: bool = False) -> None:
    print(f"loading trade log: {TRADE_LOG}")
    df = pd.read_parquet(TRADE_LOG)
    print(f"  {len(df):,} bars, {(df['N_contracts_nc'] > 0).sum():,} executed trades")

    if smoke:
        # Bound smoke-test runtime to the latest three months.
        end = df.index.max()
        start = end - pd.Timedelta(days=90)
        df = df.loc[start:end]
        print(f"  SMOKE: trimmed to {len(df):,} bars over last 90d")

    print(f"loading DVOL: {DVOL_PATH}")
    dvol = pd.read_parquet(DVOL_PATH)
    if "timestamp" in dvol.columns:
        dvol = dvol.set_index("timestamp")
    if dvol.index.tz is None:
        dvol.index = dvol.index.tz_localize("UTC")

    print("computing sigma_sigma (30-day rolling std of log-DVOL daily changes)")
    sigma_sigma = compute_sigma_sigma(dvol["close"], window_days=30)
    print(f"  sigma_sigma median (log-DVOL/day): {sigma_sigma.median():.4f}")

    sigma_sigma_per_bar = align_sigma_sigma_to_trades(sigma_sigma, df.index)

    # Sensitivity grid
    Y_grid = [0.25, 0.5, 1.0]
    ef_grid = [0.25, 0.5]
    rows = []
    for Y_vol in Y_grid:
        for edge_fraction in ef_grid:
            bound = compute_vega_capacity(
                trade_log=df,
                sigma_sigma_per_trade=sigma_sigma_per_bar,
                Y_vol=Y_vol,
                edge_fraction=edge_fraction,
            )
            finite = bound.K_max[np.isfinite(bound.K_max)]
            rows.append({
                "Y_vol": Y_vol,
                "edge_fraction": edge_fraction,
                "K_max_p25": float(np.percentile(finite, 25)) if len(finite) else np.nan,
                "K_max_median": float(np.percentile(finite, 50)) if len(finite) else np.nan,
                "K_max_p75": float(np.percentile(finite, 75)) if len(finite) else np.nan,
                "K_max_p10": float(np.percentile(finite, 10)) if len(finite) else np.nan,
                "n_trades_finite": int(len(finite)),
            })
    sens = pd.DataFrame(rows)
    print()
    print("Sensitivity table (K_max percentiles across trades):")
    print(sens.to_string(index=False))

    # Regime stratification (using base case Y=0.5, edge_fraction=0.5)
    print()
    bound = compute_vega_capacity(
        trade_log=df,
        sigma_sigma_per_trade=sigma_sigma_per_bar,
        Y_vol=0.5,
        edge_fraction=0.5,
    )
    exe_idx = df.index[df["N_contracts_nc"] > 0]
    dvol_aligned = dvol["close"].reindex(exe_idx, method="ffill").bfill()
    regime = dvol_regime(dvol_aligned)
    per_trade = pd.DataFrame({
        "K_max": bound.K_max,
        "eta_current": bound.eta_current,
        "delta_iv_at_K1": bound.delta_iv_at_K1,
        "regime": regime.values,
        "dvol_close": dvol_aligned.values,
    }, index=exe_idx)

    regime_summary = (
        per_trade.replace({"K_max": {np.inf: np.nan}})
        .groupby("regime")["K_max"]
        .agg(["count", "median", lambda s: s.quantile(0.10), lambda s: s.quantile(0.25)])
        .rename(columns={"<lambda_0>": "p10", "<lambda_1>": "p25"})
    )
    print("Regime-stratified K_max (Y=0.5, edge_fraction=0.5):")
    print(regime_summary.to_string())

    # Method 6 quick-check — perp leg non-binding
    # Per-bar perp-side notional traded ≈ 2 × |delta_hedge_pnl_nc|/dIV_underlying
    # is not directly available; use a conservative proxy: hedge_costs_nc /
    # perp_taker_fee = total_usd_traded on the perp.
    # At K=100, max bar perp share = K · (|hedge_traded| / 15min_perp_volume).
    # Without bar-level Binance perp volume, we use 24h ADV (BTCUSDT perp,
    # 2024 average $20B). 15-min slice ≈ $20B / (24·4) = $208M / 15min.
    perp_taker_fee = 0.0005
    hedge_traded_usd = (df["hedge_costs_nc"] / perp_taker_fee).fillna(0).clip(lower=0)
    perp_15m_volume_usd = 20e9 / (24 * 4)
    K_test = 100
    perp_share = (K_test * hedge_traded_usd) / perp_15m_volume_usd
    p99 = float(np.percentile(perp_share[perp_share > 0], 99)) if (perp_share > 0).any() else 0.0
    p_max = float(perp_share.max())
    print()
    print(f"Method 6 (perp non-binding) at K=100:")
    print(f"  p99 bar perp share: {p99:.4%}")
    print(f"  max bar perp share: {p_max:.4%}")
    print(f"  threshold (1%): {'PASS' if p99 < 0.01 else 'FAIL'}")

    # Save sidecar parquet
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    per_trade.to_parquet(OUT_PARQUET)
    print(f"\nwrote: {OUT_PARQUET}")

    # Write report
    base_K = float(np.nanmedian(per_trade.replace({"K_max": {np.inf: np.nan}})["K_max"]))
    write_report(
        sens=sens,
        regime_summary=regime_summary,
        method6_p99=p99,
        method6_max=p_max,
        base_K_max=base_K,
        smoke=smoke,
    )


def write_report(
    *, sens: pd.DataFrame, regime_summary: pd.DataFrame,
    method6_p99: float, method6_max: float,
    base_K_max: float, smoke: bool,
) -> None:
    out = OUT_REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    smoke_tag = "**SMOKE (90-day sample, NOT for headline)**" if smoke else "Full V2 trade log"
    body = f"""# Capacity Analysis Phase 1 — Vega-Absorption Bound (Method 2)

{smoke_tag}

## Headline

Median per-trade `K_max` (Y=0.5, edge_fraction=0.5): **{base_K_max:.1f}×** current sizing.

This is the analytic upper bound from Donier-Bouchaud-Mastromatteo et al. (2015,
[arxiv:1602.03043](https://arxiv.org/abs/1602.03043)) square-root impact law,
solved per-trade. Method 1 (iso-Sharpe scaling backtest, Phase 2) will refine
this with a calibrated Sharpe(K) curve and bootstrap CIs.

## Method

For each executed trade i:
- η_i = vega_i / market_vega_per_day_i
- ΔIV_i(K) = Y_vol · σ_σ(i) · √(K · η_i)   [vol points]
- K_max(i) such that ΔIV_i(K_max) = edge_fraction · |edge_i|

`σ_σ` = 30-day rolling std of daily log-DVOL changes, multiplied by the bar's
DVOL level to convert from log to vol-points.

`market_vega_per_day` is proxied as `trailing_1h_volume × 24 × per-BTC vega`.
This is a known coarse estimate — true per-strike Deribit volume requires a
Laevitas/Amberdata subscription (see plan Methods 3 / future work).

## Sensitivity

K_max percentiles across trades, varying Y_vol and edge_fraction:

```
{sens.to_string(index=False)}
```

Higher Y_vol → larger impact per trade → lower K_max (more conservative).
Higher edge_fraction → tolerate more impact before knee → higher K_max.

## Regime-stratified K_max (Y=0.5, edge_fraction=0.5)

DVOL terciles (low / mid / high). Capacity should be lowest in the low-IV
regime (thinner spreads, lower σ_σ in vol points) and highest in the high-IV
regime (deeper market, larger absolute ΔIV tolerance).

```
{regime_summary.to_string()}
```

Aggregate capacity for the strategy = min across regimes weighted by historical
time share.

## Method 6 quick-check — perp leg non-binding

At K=100× current sizing, the perp hedge participation rate against a
proxy 15-min Binance BTCUSDT perp slice ($208M, derived from $20B daily ADV):
- p99 bar share: {method6_p99:.4%}
- max bar share: {method6_max:.4%}
- 1% threshold: {'PASS' if method6_p99 < 0.01 else 'FAIL — escalate'}

If PASS (expected), the perp leg does not constrain capacity at any realistic
K — the option leg binds first by orders of magnitude. This justifies focusing
the Method 1 scaling backtest exclusively on the option-leg impact channel.

## Caveats

1. **Vega proxy is coarse**: assumes trailing 1h ATM volume × 24 = daily
   market vega traded. Concentrates risk on near-expiry strikes; true depth
   varies with moneyness and DTE. Treat K_max as an order-of-magnitude bound.
2. **σ_σ → vol-points conversion**: multiplying by current DVOL level is a
   linearization. Stress with ±50% in Phase 2.
3. **Static impact model**: ignores adversarial market-maker response and
   inventory effects (Garleanu-Pedersen-Poteshman). True capacity is likely
   below this bound; live paper-trade scaling (Method 7) is the only way to
   ground-truth.

## Next

Phase 2 (Method 1, `scripts/capacity_phase2_iso_sharpe.py`): impact-adjusted
Sharpe(K) curve with stationary block bootstrap CIs and circularity-break
calibration on a held-out window.
"""
    out.write_text(body, encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="3-month sample")
    args = parser.parse_args()
    main(smoke=args.smoke)
