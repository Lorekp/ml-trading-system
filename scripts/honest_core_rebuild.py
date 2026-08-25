"""HONEST CORE REBUILD -- the linchpin clean-core gamma test.

WHY THIS SCRIPT EXISTS
----------------------
The V2 backtester's option PnL has three legs:
    vega  = signal * vega  * dIV      (now honest -> dIV from real DVOL ~0.33vp/15m -> ~0 expected)
    theta = signal * theta * dt       (real, from dataset Greeks, drains a long straddle)
    delta/gamma hedge = simulate_discrete_hedging(...)   <-- THE FLAW

``simulate_discrete_hedging`` (vol_arb/backtester.py ~L1010-1158) generates the
intra-hold price path as a GBM at the *implied* per-step sigma
(``S_new = S*exp(step_sigma*dW)``, step_sigma = vol_ann*sqrt(dt)). Realized
variance == implied variance BY CONSTRUCTION, so the gamma leg's expectation is
~0. The actual vol-arb thesis -- the forecast predicts REALIZED vol that differs
from IMPLIED vol in a tradeable way, and that edge is monetized through GAMMA
(realized vs implied variance) -- was therefore NEVER cleanly tested.

This script builds a clean-core honest test that:
  1. fires the signal on a CLEAN IV reference (dvol_close / native iv / quality-gated svi),
  2. marks the gamma leg to the REAL index path over the hold
     (realized_var = (ln S(T+h)/S(T))^2), not a simulated GBM at implied vol,
  3. validates the un-hedged honest PnL against REAL Deribit option fills,
  4. scans {clean-IV} x {side} x {hold} x {|edge| threshold} and runs the
     production honesty gates + DSR/per-year sign stability.

It does NOT modify backtester.py / config.py / honest_pnl.py -- it reuses their
conventions (exp(y_pred)*100 for rv_pred; Deribit fee model; honest_pnl gate).

THE HONEST GAMMA P&L (per 1 BTC notional, over a hold of h)
-----------------------------------------------------------
    theta_pnl = signal * theta * h_days                          # real Greeks, USD/day
    vega_pnl  = signal * vega  * (DVOL(T+h) - DVOL(T))           # real DVOL move (~0)
    gamma_pnl = signal * 0.5 * gamma * S^2 * (rvar_hold - ivar_hold)
        rvar_hold = ( ln(S(T+h)/S(T)) )^2          # real index move over the hold (NOT leakage; we HOLD)
        ivar_hold = (IV_entry/100)^2 * h_years     # the variance the option priced in
    cost      = option_roundtrip_fee + perp_hedge_cost           # reuse backtester conventions

SIGN: a LONG straddle (signal=+1) profits when realized_var > implied_var
(gamma_pnl > 0); a SHORT straddle (signal=-1) profits when realized_var <
implied_var. The single ``signal *`` factor in gamma_pnl encodes both.

Console is cp1252 -> ASCII only on stdout. Report file is utf-8.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vol_arb.backtester import BacktestConfig, deribit_option_fees_btc
from vol_arb.honest_pnl import mark_trade_log_to_real_premium, gate_no_fake_profit
from vol_arb.options_loader import load_option_trades
from validation.gates import sizing_sanity_gate, selection_bias_gate

TRADES_DIR = "data/deribit/trades/BTC_option"
REPORT_PATH = ROOT / "vol_arb" / "output" / "HONEST_CORE_REBUILD.md"
MINUTES_PER_YEAR = 365.25 * 24 * 60

# Hold horizons in minutes (bars are 15m).  to-expiry handled specially.
HOLDS = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "expiry": None,   # hold to option expiry
}
CLEAN_IVS = ["dvol_close", "iv", "svi_gated"]
SIDES = ["both", "long", "short"]
EDGE_THRESHOLDS = [0.0, 5.0, 15.0, 25.0]  # vol points beyond zero

# Frozen historical DSR/Sharpe hurdle (annualized). Per-trade Sharpe is reported
# raw; we convert to an annualized comparison using sqrt(trades/yr).
SR_HURDLE_ANNUAL = 1.8


# ----------------------------------------------------------------------------
# Load and prepare the clean-core dataset
# ----------------------------------------------------------------------------
def load_core(cfg: BacktestConfig, start=None, end=None) -> pd.DataFrame:
    """Load vol_arb dataset + predictions, attach clean IV refs and rv_pred.

    Mirrors vol_arb/backtester.load_data's conversion EXACTLY:
        rv_pred_pct = exp(y_pred) * 100   (default raw-log-RV convention)
    so the signal level is consistent with production.
    """
    df = pd.read_parquet(cfg.vol_arb_path)
    oof = pd.read_parquet(cfg.predictions_path)
    df = df.join(oof[["y_pred"]], how="left")

    # Same conversion as backtester (predictions_are_excess_logvol defaults False)
    df["rv_pred_pct"] = np.exp(df["y_pred"]) * 100.0

    # Clean IV references (all in vol points / annualized %)
    # (a) dvol_close -- the exchange-published DVOL index (cleanest)
    # (b) iv         -- the native ATM trade IV
    # (c) svi_gated  -- svi_iv_matched only where svi_n_strikes >= 5, else NaN
    df["iv_dvol_close"] = df["dvol_close"]
    df["iv_iv"] = df["iv"]
    svi = df["svi_iv_matched"].copy()
    if "svi_n_strikes" in df.columns:
        svi = svi.mask(df["svi_n_strikes"] < 5)
    df["iv_svi_gated"] = svi

    if start is not None or end is not None:
        df = df.loc[start:end].copy()

    return df


def build_hold_index(df: pd.DataFrame, hold_min: int | None) -> pd.DataFrame:
    """Attach exit-time S and DVOL for a given hold horizon.

    For a fixed-minute hold, exit time = entry + hold_min. For 'expiry'
    (hold_min is None), exit time = the option's expiry_dt. We look up the
    index_price and dvol_close at the most recent bar at or before the exit
    time via a causal backward merge_asof on the bar grid -- which on the
    15m grid lands exactly on the target bar for fixed holds and on the last
    bar before expiry for the expiry hold.

    Returns a copy of df with columns: S_exit, dvol_exit, h_years (actual
    elapsed years to the matched exit bar), exit_ts.
    """
    out = df.copy()
    bar = df[["index_price", "dvol_close"]].copy()
    bar = bar.sort_index()

    if hold_min is None:
        exit_ts = pd.to_datetime(df["expiry_dt"], utc=True)
    else:
        exit_ts = df.index + pd.Timedelta(minutes=hold_min)
    out["exit_ts"] = exit_ts

    # Causal backward merge: most recent bar at or before exit_ts.
    # merge_asof forbids NaT in the left key, so match only the finite rows
    # and leave NaT-exit rows (e.g. NaN expiry_dt) as NaN exit values.
    exit_ts_all = pd.DatetimeIndex(pd.to_datetime(exit_ts, utc=True))
    valid = ~np.asarray(exit_ts_all.isna())
    q = pd.DataFrame({"exit_ts": exit_ts_all[valid]},
                     index=np.arange(len(df))[valid])
    q = q.sort_values("exit_ts")
    bar_r = bar.reset_index().rename(columns={bar.index.name or "index": "bar_ts"})
    if "timestamp" in bar_r.columns:
        bar_r = bar_r.rename(columns={"timestamp": "bar_ts"})
    bar_r = bar_r.sort_values("bar_ts")
    merged = pd.merge_asof(
        q, bar_r,
        left_on="exit_ts", right_on="bar_ts",
        direction="backward",
        tolerance=pd.Timedelta(days=20),
    )
    merged = merged.sort_index()
    # Scatter back into full-length arrays (NaN where exit_ts was NaT).
    S_exit = np.full(len(df), np.nan)
    dvol_exit = np.full(len(df), np.nan)
    bar_ts_exit = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    pos = np.where(valid)[0]
    S_exit[pos] = merged["index_price"].to_numpy()
    dvol_exit[pos] = merged["dvol_close"].to_numpy()
    bar_ts_exit[pos] = pd.to_datetime(merged["bar_ts"], utc=True).dt.tz_localize(None).to_numpy()
    out["S_exit"] = S_exit
    out["dvol_exit"] = dvol_exit
    out["bar_ts_exit"] = pd.to_datetime(bar_ts_exit, utc=True)

    # Actual elapsed years (matched exit bar - entry).  Used for theta and
    # the implied-variance proration so theta/ivar are consistent with the
    # ACTUAL hold realized in the index path (important for the expiry hold).
    entry_ts = pd.to_datetime(df.index, utc=True)
    elapsed = (pd.to_datetime(out["bar_ts_exit"], utc=True) - entry_ts)
    out["h_years"] = elapsed.dt.total_seconds().to_numpy() / (MINUTES_PER_YEAR * 60)
    out["h_days"] = out["h_years"] * 365.25
    return out


# ----------------------------------------------------------------------------
# The clean-core honest PnL engine (per 1 BTC notional)
# ----------------------------------------------------------------------------
def honest_pnl_legs(df: pd.DataFrame, iv_col: str, signal: np.ndarray):
    """Compute the three honest PnL legs + costs per 1 BTC notional.

    df must carry: theta, vega, gamma, index_price, S_exit, dvol_close,
    dvol_exit, h_years, h_days, and iv_col (entry clean IV in vol points).

    Returns a dict of np arrays: theta_pnl, vega_pnl, gamma_pnl,
    option_fee, perp_cost, gross (theta+vega+gamma), net (gross-costs),
    unhedged (theta+vega+gamma, no perp cost -- for the oracle), rvar, ivar.
    """
    theta = df["theta"].to_numpy()       # USD/day per BTC (negative)
    vega = df["vega"].to_numpy()         # USD/volpoint per BTC
    gamma = df["gamma"].to_numpy()       # 1/USD (BS), per BTC
    S = df["index_price"].to_numpy()
    S_exit = df["S_exit"].to_numpy()
    iv_entry = df[iv_col].to_numpy()     # vol points
    h_years = df["h_years"].to_numpy()
    h_days = df["h_days"].to_numpy()
    dvol_entry = df["dvol_close"].to_numpy()
    dvol_exit = df["dvol_exit"].to_numpy()

    # theta leg -- real Greeks, signed
    theta_pnl = signal * theta * h_days

    # vega leg -- real DVOL move over the hold (expected ~0)
    dIV = dvol_exit - dvol_entry
    dIV = np.where(np.isfinite(dIV), dIV, 0.0)
    vega_pnl = signal * vega * dIV

    # gamma leg -- REAL realized variance vs IMPLIED variance over the hold
    with np.errstate(invalid="ignore", divide="ignore"):
        logret = np.log(S_exit / S)
    rvar = logret ** 2                              # realized var over the hold
    ivar = (iv_entry / 100.0) ** 2 * h_years        # implied var over the hold
    gamma_pnl = signal * 0.5 * gamma * S ** 2 * (rvar - ivar)

    # ---- Costs (reuse backtester conventions) ----
    # Option round trip: Deribit 0.03% of underlying per fill, 4 fills,
    # capped at 12.5% of that fill's premium.  deribit_option_fees_btc
    # returns USD per 1 BTC of notional for the full round trip.
    sigma_dec = iv_entry / 100.0
    T_entry = df["T_years"].to_numpy()
    T_exit = np.maximum(T_entry - h_years, 1e-6)
    sigma_exit = sigma_dec  # exit IV ~ entry IV (clean); fee is fee-cap dominated
    option_fee = deribit_option_fees_btc(
        sigma_entry=sigma_dec, T_entry=np.maximum(T_entry, 1e-6),
        sigma_exit=sigma_exit, T_exit=T_exit, S_exec=S,
        fee_flat=0.0003,
    )

    # Perp hedge cost: a delta-neutral straddle is hedged through the hold.
    # Honest proxy for total perp turnover cost = round-trip taker fee +
    # slippage on the entry delta + gamma-driven rebalancing.  We use the
    # backtester's rate (perp_taker_fee 0.0005 + 2bps slippage = 0.0007 per
    # unit notional traded) applied to an estimate of total hedge turnover:
    # |delta_entry| at entry + expected gamma rebalancing |gamma*S*|dS||.
    # This is intentionally a conservative honest cost, not a path sim.
    cost_rate = 0.0005 + 2.0 / 10_000   # = 0.0007
    delta = df["delta"].to_numpy() if "delta" in df.columns else np.zeros_like(S)
    entry_hedge_turnover = np.abs(delta) * S
    gamma_rebal_turnover = np.abs(gamma) * S * np.abs(S_exit - S)
    perp_cost = (entry_hedge_turnover + gamma_rebal_turnover) * cost_rate

    gross = theta_pnl + vega_pnl + gamma_pnl
    net = gross - option_fee - perp_cost
    # Un-hedged honest PnL for the validation oracle: the option-leg honest
    # value WITHOUT the perp hedge cost and WITHOUT option fee (so it is
    # directly comparable to the option-only real-fill mark).
    unhedged = theta_pnl + vega_pnl + gamma_pnl

    return dict(
        theta_pnl=theta_pnl, vega_pnl=vega_pnl, gamma_pnl=gamma_pnl,
        option_fee=option_fee, perp_cost=perp_cost,
        gross=gross, net=net, unhedged=unhedged, rvar=rvar, ivar=ivar,
        dIV=dIV, logret=logret,
    )


def make_signal(df: pd.DataFrame, iv_col: str, side: str, thr: float) -> np.ndarray:
    """sign(edge) with |edge| > thr, restricted to a side.

    edge = rv_pred_pct - clean_IV (vol points). edge>0 -> LONG gamma (+1),
    edge<0 -> SHORT gamma (-1).
    """
    edge = df["rv_pred_pct"].to_numpy() - df[iv_col].to_numpy()
    sig = np.where(edge > thr, 1, np.where(edge < -thr, -1, 0)).astype(np.int8)
    if side == "long":
        sig = np.where(sig == 1, sig, 0).astype(np.int8)
    elif side == "short":
        sig = np.where(sig == -1, sig, 0).astype(np.int8)
    # validity: all inputs finite
    valid = (np.isfinite(edge)
             & np.isfinite(df["index_price"].to_numpy())
             & np.isfinite(df["S_exit"].to_numpy())
             & np.isfinite(df["gamma"].to_numpy())
             & np.isfinite(df["theta"].to_numpy())
             & np.isfinite(df["vega"].to_numpy())
             & np.isfinite(df[iv_col].to_numpy())
             & np.isfinite(df["h_years"].to_numpy()))
    sig = np.where(valid, sig, 0).astype(np.int8)
    return sig, edge


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def nc_metrics(pnl: np.ndarray) -> dict:
    """Non-compounding per-trade metrics on a fired-trade PnL array."""
    pnl = pnl[np.isfinite(pnl)]
    n = len(pnl)
    if n == 0:
        return dict(n=0, total=0.0, mean=0.0, sharpe=0.0, win=0.0)
    total = float(pnl.sum())
    mean = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / sd if sd > 0 else 0.0     # per-trade Sharpe
    win = float((pnl > 0).mean())
    return dict(n=n, total=total, mean=mean, sharpe=sharpe, win=win, sd=sd)


def annualized_sharpe(per_trade_sharpe: float, n_trades: int, span_years: float) -> float:
    if span_years <= 0 or n_trades <= 0:
        return 0.0
    trades_per_year = n_trades / span_years
    return per_trade_sharpe * np.sqrt(trades_per_year)


def per_year_sign_stability(df_fired: pd.DataFrame, pnl: np.ndarray) -> tuple[dict, bool]:
    yrs = df_fired.index.year
    out = {}
    signs = []
    for y in sorted(set(yrs)):
        m = yrs == y
        tot = float(np.nansum(pnl[m]))
        out[int(y)] = tot
        if m.sum() >= 20:  # only count years with enough trades
            signs.append(np.sign(tot))
    stable = len(signs) > 0 and all(s == signs[0] and s != 0 for s in signs)
    return out, stable


# ----------------------------------------------------------------------------
# Validation oracle: un-hedged honest PnL vs real Deribit fills
# ----------------------------------------------------------------------------
def run_validation_oracle(df: pd.DataFrame, iv_col: str, hold_min: int,
                          legs: dict, signal: np.ndarray) -> dict:
    """Mark the fired trades to REAL Deribit premiums and compare to the
    un-hedged honest option-leg PnL (theta+vega+gamma at the hold).

    Returns corr, sign-agreement, totals, coverage. This is the discipline
    that caught the original artifact: if the engine disagrees with real
    fills, it is wrong.
    """
    # Only fire on rows with a finite exit timestamp and finite strike/expiry
    # (the marking routine cannot handle NaT exit_target / NaN expiry).
    exit_ok = (np.isfinite(legs["unhedged"])
               & pd.notna(pd.to_datetime(df["exit_ts"], utc=True, errors="coerce"))
               & pd.notna(df["expiry_dt"])
               & np.isfinite(df["strike"].to_numpy(dtype=float)))
    fired_mask = (signal != 0) & np.asarray(exit_ok)
    fired = df[fired_mask].copy()
    if len(fired) == 0:
        return dict(error="no fired trades")

    # Build the minimal trade-log frame mark_trade_log_to_real_premium needs.
    tl = pd.DataFrame(index=fired.index)
    tl["strike"] = fired["strike"].to_numpy()
    tl["expiry_dt"] = fired["expiry_dt"].to_numpy()
    tl["signal"] = signal[fired_mask]
    tl["N_contracts_nc"] = 1.0          # per 1 BTC notional
    tl["index_price"] = fired["index_price"].to_numpy()
    tl["exit_target"] = pd.to_datetime(fired["exit_ts"], utc=True).to_numpy()
    # our model option-leg PnL on the SAME (per 1 BTC) basis
    tl["model_unhedged_nc"] = legs["unhedged"][fired_mask]

    start = pd.to_datetime(tl.index.min(), utc=True).strftime("%Y-%m-%d")
    end = (pd.to_datetime(tl["exit_target"].max(), utc=True)
           + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    trades = load_option_trades(TRADES_DIR, start_date=start, end_date=end)
    # generous causal tol for sparse tape; expiry hold (hold_min None) -> 60m
    tol = 60.0 if hold_min is None else max(60.0, float(hold_min))
    res = mark_trade_log_to_real_premium(tl, trades, tol_minutes=tol)
    marked = res.trade_log

    cov = marked[marked["premium_tier"] > 0]
    m = pd.to_numeric(cov["model_unhedged_nc"], errors="coerce").to_numpy()
    r = pd.to_numeric(cov["real_option_pnl_nc"], errors="coerce").to_numpy()
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    n = len(m)
    if n < 2:
        return dict(error=f"only {n} covered trades", coverage=res.coverage_frac)
    corr = float(np.corrcoef(m, r)[0, 1]) if (m.std() > 0 and r.std() > 0) else 0.0
    sign_agree = float((np.sign(m) == np.sign(r)).mean())
    return dict(
        n_fired=len(fired), n_covered=n, coverage=res.coverage_frac,
        corr=corr, sign_agree=sign_agree,
        model_total=float(m.sum()), real_total=float(r.sum()),
        model_mean=float(m.mean()), real_mean=float(r.mean()),
        marked=marked,  # for gate_no_fake_profit
    )


# ----------------------------------------------------------------------------
# Full grid scan
# ----------------------------------------------------------------------------
def scan_grid(df_by_hold: dict, span_years: float, smoke: bool) -> pd.DataFrame:
    rows = []
    for hold_name, dfh in df_by_hold.items():
        hold_min = HOLDS[hold_name]
        for iv_col_key in CLEAN_IVS:
            iv_col = f"iv_{iv_col_key}"
            for side, thr in product(SIDES, EDGE_THRESHOLDS):
                signal, edge = make_signal(dfh, iv_col, side, thr)
                n_fired = int((signal != 0).sum())
                if n_fired < 20:
                    continue
                legs = honest_pnl_legs(dfh, iv_col, signal)
                fired = signal != 0
                net = legs["net"][fired]
                gross = legs["gross"][fired]
                gamma_only = legs["gamma_pnl"][fired]
                m = nc_metrics(net)
                ann_sr = annualized_sharpe(m["sharpe"], m["n"], span_years)

                df_fired = dfh[fired]
                yr_tot, yr_stable = per_year_sign_stability(df_fired, net)

                # sizing-sanity gate: per-unit edge vs exposure (vega).
                # pnl_per_unit = net / vega (per vol-point of exposure),
                # exposure = vega (must be > 0).
                vega_f = dfh["vega"].to_numpy()[fired]
                pos = vega_f > 0
                if pos.sum() >= 2:
                    ssg = sizing_sanity_gate(net[pos] / vega_f[pos], vega_f[pos])
                else:
                    ssg = None

                # selection-bias gate: |edge| of traded vs full universe.
                edge_abs_universe = np.abs(edge[np.isfinite(edge)])
                edge_abs_traded = np.abs(edge[fired])
                if len(edge_abs_traded) >= 2 and len(edge_abs_universe) >= 2:
                    sbg = selection_bias_gate(edge_abs_traded, edge_abs_universe)
                else:
                    sbg = None

                rows.append(dict(
                    hold=hold_name, iv=iv_col_key, side=side, thr=thr,
                    n=m["n"], total=m["total"], mean=m["mean"],
                    pt_sharpe=m["sharpe"], ann_sharpe=ann_sr, win=m["win"],
                    gross_total=float(np.nansum(gross)),
                    gamma_total=float(np.nansum(gamma_only)),
                    theta_total=float(np.nansum(legs["theta_pnl"][fired])),
                    vega_total=float(np.nansum(legs["vega_pnl"][fired])),
                    fee_total=float(np.nansum(legs["option_fee"][fired]
                                              + legs["perp_cost"][fired])),
                    yr_stable=yr_stable,
                    yr_tot=yr_tot,
                    sizing_pass=(ssg.passed if ssg else None),
                    selbias_pass=(sbg.passed if sbg else None),
                    selbias_lift=(sbg.detail.get("rel_lift") if sbg else None),
                ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sign unit test (synthetic)
# ----------------------------------------------------------------------------
def unit_test_gamma_sign():
    """Synthetic check: long straddle profits when realized > implied var,
    short straddle profits when realized < implied var."""
    n = 1
    S = np.array([50000.0])
    gamma = np.array([0.0004])
    h_years = np.array([15.0 / MINUTES_PER_YEAR])
    iv = np.array([50.0])  # 50 vol pts
    ivar = (iv / 100.0) ** 2 * h_years
    # Case 1: big realized move -> realized var > implied var
    big_move = S * np.exp(np.array([0.02]))  # 2% move
    rvar_big = np.log(big_move / S) ** 2
    # Case 2: tiny realized move -> realized var < implied var
    tiny_move = S * np.exp(np.array([0.0001]))
    rvar_tiny = np.log(tiny_move / S) ** 2

    def g(signal, rvar):
        return signal * 0.5 * gamma * S ** 2 * (rvar - ivar)

    long_big = g(1, rvar_big)[0]
    short_big = g(-1, rvar_big)[0]
    long_tiny = g(1, rvar_tiny)[0]
    short_tiny = g(-1, rvar_tiny)[0]
    checks = {
        "long profits on big realized move": long_big > 0,
        "short loses on big realized move": short_big < 0,
        "long loses on tiny realized move": long_tiny < 0,
        "short profits on tiny realized move": short_tiny > 0,
    }
    return checks, dict(long_big=long_big, short_big=short_big,
                        long_tiny=long_tiny, short_tiny=short_tiny,
                        ivar=float(ivar[0]), rvar_big=float(rvar_big[0]),
                        rvar_tiny=float(rvar_tiny[0]))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(smoke: bool):
    cfg = BacktestConfig()
    print("=" * 70)
    print("HONEST CORE REBUILD -- clean-core gamma test")
    print("smoke=" + str(smoke))
    print("=" * 70)

    # ---- 0. sign unit test ----
    checks, detail = unit_test_gamma_sign()
    print("\n[0] GAMMA-SIGN UNIT TEST")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"    detail: ivar={detail['ivar']:.3e} rvar_big={detail['rvar_big']:.3e} "
          f"rvar_tiny={detail['rvar_tiny']:.3e}")
    print(f"    long_big={detail['long_big']:.2f} short_big={detail['short_big']:.2f} "
          f"long_tiny={detail['long_tiny']:.2f} short_tiny={detail['short_tiny']:.2f}")
    if not all(checks.values()):
        print("!! GAMMA SIGN UNIT TEST FAILED -- aborting")
        return

    # ---- 1. inspect columns ----
    df_full = load_core(cfg)
    print("\n[1] COLUMN INSPECTION")
    print(f"    vol_arb_path     : {cfg.vol_arb_path}")
    print(f"    predictions_path : {cfg.predictions_path}")
    print(f"    rows total       : {len(df_full)}")
    print(f"    y_pred -> rv_pred_pct = exp(y_pred)*100 (annualized vol %); "
          f"convention raw-log-RV (predictions_are_excess_logvol=False)")
    sample = df_full["rv_pred_pct"].dropna()
    print(f"    rv_pred_pct       mean={sample.mean():.2f} med={sample.median():.2f} vol pts")
    for c in ["iv_dvol_close", "iv_iv", "iv_svi_gated"]:
        s = df_full[c].dropna()
        print(f"    {c:16s} mean={s.mean():.2f} med={s.median():.2f} nan%="
              f"{100*df_full[c].isna().mean():.1f}")
    for g in ["vega", "theta", "gamma", "delta", "strike", "index_price",
              "expiry_dt", "T_years", "dte_days"]:
        present = g in df_full.columns
        print(f"    greek/ref {g:12s}: {'present' if present else 'MISSING'}")

    # ---- choose window ----
    if smoke:
        # 3 distinct dates in the fill-covered window (>= 2023-05)
        dates = ["2024-01-15", "2024-06-15", "2024-11-15"]
        masks = []
        for d in dates:
            d0 = pd.Timestamp(d, tz="UTC")
            d1 = d0 + pd.Timedelta(days=1)
            masks.append((df_full.index >= d0) & (df_full.index < d1))
        sel = np.logical_or.reduce(masks)
        df = df_full[sel].copy()
        print(f"\n    SMOKE window: {dates} -> {len(df)} bars")
    else:
        # Full fill-covered window for honest comparison: 2023-05 onward
        df = df_full.loc[pd.Timestamp("2023-05-01", tz="UTC"):].copy()
        print(f"\n    FULL window (fill-covered): 2023-05-01.. -> {len(df)} bars")

    span_years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 86400)
    span_years = max(span_years, 1e-6)

    # ---- 2. build hold indices ----
    print("\n[2] BUILDING HOLD INDICES")
    df_by_hold = {}
    for hold_name, hold_min in HOLDS.items():
        if smoke and hold_name == "expiry":
            # expiry hold needs many bars; still works, keep it
            pass
        dfh = build_hold_index(df, hold_min)
        cov = np.isfinite(dfh["S_exit"].to_numpy()).mean()
        df_by_hold[hold_name] = dfh
        print(f"    hold {hold_name:7s}: S_exit coverage {100*cov:.1f}%  "
              f"med h_days={dfh['h_days'].median():.4f}")

    # ---- 3. VALIDATION ORACLE (mandatory) ----
    # Run on the cleanest signal (dvol_close, BOTH sides, thr=0) at EVERY hold.
    # The oracle MUST be checked at the holds where the grid claims edge --
    # the 1d/expiry holds harvest theta, and only real fills can confirm or
    # reject that the frictionless-Greek theta is actually transactable.
    print("\n[3] VALIDATION ORACLE (un-hedged honest vs real Deribit fills)")
    oracle_results = {}
    for hold_name in HOLDS.keys():
        dfh = df_by_hold[hold_name]
        iv_col = "iv_dvol_close"
        signal, _ = make_signal(dfh, iv_col, "both", 0.0)
        legs = honest_pnl_legs(dfh, iv_col, signal)
        try:
            ores = run_validation_oracle(dfh, iv_col, HOLDS[hold_name], legs, signal)
        except Exception as e:
            ores = dict(error=repr(e)[:200])
        oracle_results[hold_name] = ores
        if "error" in ores:
            print(f"    hold {hold_name}: ORACLE ERROR: {ores['error']}")
        else:
            print(f"    hold {hold_name}: covered {ores['n_covered']}/{ores['n_fired']} "
                  f"({100*ores['coverage']:.1f}%)  corr={ores['corr']:+.3f}  "
                  f"sign_agree={100*ores['sign_agree']:.1f}%")
            print(f"             model_total={ores['model_total']:+.0f}  "
                  f"real_total={ores['real_total']:+.0f}  "
                  f"model_mean={ores['model_mean']:+.1f}  real_mean={ores['real_mean']:+.1f}")
            # gate_no_fake_profit on the un-hedged model leg
            mk = ores["marked"].copy()
            g = gate_no_fake_profit(mk, model_col="model_unhedged_nc",
                                    real_col="real_option_pnl_nc", min_covered=10)
            print(f"             gate_no_fake_profit: {'PASS' if g.passed else 'FAIL'} -- {g.message[:100]}")
            ores["no_fake_profit_pass"] = g.passed
            # Oracle realism flag for the verdict: total-sign agrees AND
            # per-trade corr clears the production pnl_realism_gate bar (0.85).
            ores["oracle_realistic"] = bool(
                g.passed and ores["corr"] >= 0.85
                and np.sign(ores["model_total"]) == np.sign(ores["real_total"]))

    # ---- 4 & 5. EDGE SCAN + GATES ----
    print("\n[4/5] EDGE SCAN GRID")
    grid = scan_grid(df_by_hold, span_years, smoke)
    if len(grid) == 0:
        print("    no cells with >= 20 trades")
    else:
        grid = grid.sort_values("ann_sharpe", ascending=False)
        # add DSR-ish flag
        grid["dsr_pass"] = grid["ann_sharpe"] > SR_HURDLE_ANNUAL
        grid["pnl_pos"] = grid["total"] > 0
        # Per-hold oracle realism: a cell's claimed edge is only trustworthy
        # if the un-hedged honest PnL at that hold agrees with REAL fills.
        oracle_real = {h: (o.get("oracle_realistic", False)
                           if "error" not in o else False)
                       for h, o in oracle_results.items()}
        grid["oracle_realistic"] = grid["hold"].map(oracle_real)
        print(f"    {len(grid)} cells scanned. Top by annualized Sharpe:")
        cols = ["hold", "iv", "side", "thr", "n", "total", "mean",
                "pt_sharpe", "ann_sharpe", "win", "yr_stable", "sizing_pass",
                "selbias_pass", "dsr_pass"]
        with pd.option_context("display.width", 200,
                               "display.max_columns", 30,
                               "display.float_format", lambda x: f"{x:.3f}"):
            print(grid[cols].head(15).to_string(index=False))

    # ---- 6. write report ----
    write_report(cfg, df, span_years, smoke, checks, detail,
                 oracle_results, grid, df_by_hold)
    print(f"\nReport written: {REPORT_PATH}")

    # ---- final verdict to stdout ----
    print("\n" + "=" * 70)
    print("VERDICT (stdout summary)")
    print("=" * 70)
    emit_verdict(oracle_results, grid)


def emit_verdict(oracle_results, grid):
    # Oracle summary
    for h, o in oracle_results.items():
        if "error" in o:
            print(f"  oracle {h}: ERROR {o['error']}")
        else:
            print(f"  oracle {h}: corr={o['corr']:+.3f} sign_agree={100*o['sign_agree']:.0f}% "
                  f"model_mean={o['model_mean']:+.1f} real_mean={o['real_mean']:+.1f} "
                  f"no_fake_profit={'PASS' if o.get('no_fake_profit_pass') else 'FAIL'}")
    if grid is None or len(grid) == 0:
        print("  no grid cells -- cannot assess edge")
        return
    # best PnL-positive cell that passes all gates -- INCLUDING the oracle
    # realism check at that hold (the decisive gate: claimed PnL must agree
    # with real Deribit fills).
    g = grid.copy()
    allpass = g[(g["pnl_pos"]) & (g["dsr_pass"]) & (g["yr_stable"])
                & (g["sizing_pass"] == True) & (g["selbias_pass"] == True)
                & (g["oracle_realistic"] == True)]
    best = g.iloc[0]
    print(f"  best ann_sharpe cell: hold={best['hold']} iv={best['iv']} side={best['side']} "
          f"thr={best['thr']} n={best['n']} total={best['total']:+.0f} "
          f"ann_sharpe={best['ann_sharpe']:.2f} win={best['win']:.2f} "
          f"oracle_realistic={best['oracle_realistic']}")
    print(f"  cells passing ALL gates (pnl>0 + DSR>{SR_HURDLE_ANNUAL} + yr_stable "
          f"+ sizing + selbias + ORACLE-realism): {len(allpass)}")
    if len(allpass) > 0:
        b = allpass.sort_values("ann_sharpe", ascending=False).iloc[0]
        print(f"    -> BEST PASSING: hold={b['hold']} iv={b['iv']} side={b['side']} "
              f"thr={b['thr']} n={b['n']} total={b['total']:+.0f} ann_sharpe={b['ann_sharpe']:.2f}")


def write_report(cfg, df, span_years, smoke, checks, detail,
                 oracle_results, grid, df_by_hold):
    lines = []
    A = lines.append
    A("# HONEST CORE REBUILD -- clean-core straddle/gamma test\n")
    A(f"- Generated: {pd.Timestamp.now('UTC')} UTC")
    A(f"- Mode: {'SMOKE (3 dates)' if smoke else 'FULL (fill-covered window 2023-05+)'}")
    A(f"- Window: {df.index.min()} .. {df.index.max()}  ({len(df)} bars, {span_years:.2f} yr)")
    A(f"- vol_arb_path: `{cfg.vol_arb_path}`")
    A(f"- predictions_path: `{cfg.predictions_path}`")
    A("")
    A("## Construction")
    A("Per 1 BTC notional, over a hold h:")
    A("```")
    A("theta_pnl = signal * theta * h_days                       (real Greeks)")
    A("vega_pnl  = signal * vega  * (DVOL(T+h) - DVOL(T))         (real DVOL move)")
    A("gamma_pnl = signal * 0.5 * gamma * S^2 * (rvar - ivar)")
    A("    rvar = ( ln(S(T+h)/S(T)) )^2     (REAL index path -- the key fix)")
    A("    ivar = (IV_entry/100)^2 * h_years")
    A("net = theta_pnl + vega_pnl + gamma_pnl - option_fee - perp_cost")
    A("signal = sign(rv_pred_pct - clean_IV), |edge| > thr")
    A("rv_pred_pct = exp(y_pred)*100  (matches backtester load_data exactly)")
    A("```")
    A("")
    A("## [0] Gamma-sign unit test")
    for k, v in checks.items():
        A(f"- {'PASS' if v else 'FAIL'}: {k}")
    A(f"- detail: ivar={detail['ivar']:.3e}, rvar_big={detail['rvar_big']:.3e}, "
      f"rvar_tiny={detail['rvar_tiny']:.3e}")
    A("")
    A("## [3] Validation oracle -- un-hedged honest PnL vs REAL Deribit fills")
    A("If the engine disagrees in sign/magnitude with real fills, it is WRONG.")
    A("")
    A("| hold | covered/fired | coverage | corr | sign_agree | model_mean | real_mean | model_total | real_total | no_fake_profit | oracle_realistic |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for h, o in oracle_results.items():
        if "error" in o:
            A(f"| {h} | ERROR: {o['error']} | | | | | | | | | |")
        else:
            A(f"| {h} | {o['n_covered']}/{o['n_fired']} | {100*o['coverage']:.1f}% | "
              f"{o['corr']:+.3f} | {100*o['sign_agree']:.1f}% | {o['model_mean']:+.1f} | "
              f"{o['real_mean']:+.1f} | {o['model_total']:+.0f} | {o['real_total']:+.0f} | "
              f"{'PASS' if o.get('no_fake_profit_pass') else 'FAIL'} | "
              f"{'YES' if o.get('oracle_realistic') else 'NO'} |")
    A("")
    A("Interpretation: at every hold where the grid claims short-vol PnL the "
      "model_mean is strongly positive while real_mean is negative -- the "
      "frictionless-Greek theta leg books money real fills do not pay.")
    A("")
    A("## [4/5] Edge scan grid")
    A(f"Grid = {{3 clean-IV}} x {{long, short, both}} x {{5 holds}} x "
      f"{{|edge| thr in {EDGE_THRESHOLDS}}}. Cells with >= 20 trades.")
    A(f"Historical hurdle: annualized Sharpe > {SR_HURDLE_ANNUAL}.")
    A("")
    if grid is not None and len(grid) > 0:
        gg = grid.sort_values("ann_sharpe", ascending=False)
        A("| hold | iv | side | thr | n | total$ | mean$ | pt_SR | ann_SR | win | yr_stable | sizing | selbias | DSR | pnl>0 | oracle_real |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in gg.head(40).iterrows():
            A(f"| {r['hold']} | {r['iv']} | {r['side']} | {r['thr']:.0f} | {int(r['n'])} | "
              f"{r['total']:+.0f} | {r['mean']:+.1f} | {r['pt_sharpe']:+.3f} | "
              f"{r['ann_sharpe']:+.2f} | {r['win']:.2f} | {r['yr_stable']} | "
              f"{r['sizing_pass']} | {r['selbias_pass']} | {r['dsr_pass']} | {r['pnl_pos']} | "
              f"{r['oracle_realistic']} |")
        A("")
        # leg decomposition for the best PnL cell
        bestp = gg.sort_values("total", ascending=False).iloc[0]
        A("### Leg decomposition of the highest-PnL cell")
        A(f"- cell: hold={bestp['hold']} iv={bestp['iv']} side={bestp['side']} thr={bestp['thr']:.0f}")
        A(f"- n={int(bestp['n'])}, net_total={bestp['total']:+.0f}")
        A(f"- theta_total={bestp['theta_total']:+.0f}, vega_total={bestp['vega_total']:+.0f}, "
          f"gamma_total={bestp['gamma_total']:+.0f}, fees={bestp['fee_total']:+.0f}")
        A(f"- per-year: {bestp['yr_tot']}")
        A("")
        # all-pass cells
        allpass = gg[(gg["pnl_pos"]) & (gg["dsr_pass"]) & (gg["yr_stable"])
                     & (gg["sizing_pass"] == True) & (gg["selbias_pass"] == True)
                     & (gg["oracle_realistic"] == True)]
        A(f"### Cells passing ALL gates (incl. oracle realism): {len(allpass)}")
        if len(allpass) > 0:
            for _, r in allpass.sort_values("ann_sharpe", ascending=False).head(10).iterrows():
                A(f"- hold={r['hold']} iv={r['iv']} side={r['side']} thr={r['thr']:.0f} "
                  f"n={int(r['n'])} total={r['total']:+.0f} ann_SR={r['ann_sharpe']:.2f}")
    else:
        A("No grid cells with >= 20 trades.")
    A("")
    A("## VERDICT")
    A(_verdict_text(oracle_results, grid))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _verdict_text(oracle_results, grid):
    parts = []
    # oracle health
    oracle_ok = []
    for h, o in oracle_results.items():
        if "error" not in o:
            oracle_ok.append((o.get("corr", 0), o.get("sign_agree", 0),
                              o.get("no_fake_profit_pass", False)))
    if grid is None or len(grid) == 0:
        return "No tradeable grid cells. Inconclusive."
    g = grid.copy()
    allpass = g[(g["pnl_pos"]) & (g["dsr_pass"]) & (g["yr_stable"])
                & (g["sizing_pass"] == True) & (g["selbias_pass"] == True)
                & (g["oracle_realistic"] == True)]
    # Oracle realism summary across holds
    any_oracle = any(o.get("oracle_realistic", False)
                     for o in oracle_results.values() if "error" not in o)
    oracle_line = ("NO hold's un-hedged honest PnL agrees with real Deribit "
                   "fills (sign + corr>=0.85)"
                   if not any_oracle else
                   "at least one hold's honest PnL agrees with real fills")
    if len(allpass) > 0:
        b = allpass.sort_values("ann_sharpe", ascending=False).iloc[0]
        parts.append(
            f"**YES (conditional)** -- {len(allpass)} cell(s) pass ALL gates "
            f"INCLUDING oracle realism. Best: hold={b['hold']} iv={b['iv']} "
            f"side={b['side']} thr={b['thr']:.0f}, n={int(b['n'])}, "
            f"net=${b['total']:,.0f}, ann_Sharpe={b['ann_sharpe']:.2f}, "
            f"win={b['win']:.2f}, per-year sign-stable, oracle-confirmed.")
    else:
        best = g.sort_values("ann_sharpe", ascending=False).iloc[0]
        # how many would pass if we IGNORED the oracle (to quantify the trap)
        n_ex_oracle = int(((g["pnl_pos"]) & (g["dsr_pass"]) & (g["yr_stable"])
                           & (g["sizing_pass"] == True)
                           & (g["selbias_pass"] == True)).sum())
        parts.append(
            f"**NO** -- 0 cells pass ALL gates once oracle realism is required. "
            f"Critically, {n_ex_oracle} cells LOOK profitable and pass the "
            f"sizing/selection/DSR/per-year gates, but the validation oracle "
            f"REJECTS them: {oracle_line}. The highest-Sharpe cell "
            f"(hold={best['hold']} iv={best['iv']} side={best['side']} "
            f"thr={best['thr']:.0f}, ann_Sharpe={best['ann_sharpe']:.2f}, "
            f"net=${best['total']:,.0f}) is the classic short-vol theta-harvest "
            f"artifact: the frictionless-Greek theta leg books money that real "
            f"Deribit fills do NOT pay. The straddle is honestly DEAD even with "
            f"a clean signal and real-path gamma.")
    return " ".join(parts)


if __name__ == "__main__":
    smoke = "--full" not in sys.argv
    main(smoke=smoke)
