"""BUILD CANDIDATE TABLE -- the expensive, one-time honest-objective cache.

WHY THIS EXISTS
---------------
The greedy add-back experiment (`scripts/greedy_addback.py`) needs, for every bar
where an ATM straddle could fire, the HONEST objective (real-Deribit-fill-marked
straddle PnL) at every tradeable horizon, plus every component feature the
contaminated/tuned items key on. Marking to real fills (4 option legs x 5
horizons via causal merge_asof over the sparse option tape) is the slow part;
we do it ONCE here and cache to parquet so the greedy search is cheap.

THE HONEST OBJECTIVE (real fills, the ground truth)
---------------------------------------------------
For each horizon h in {15m, 1h, 4h, 1d, expiry} we mark the SHORT straddle to
real Deribit premiums via `vol_arb.honest_pnl.mark_trade_log_to_real_premium`:
    real_short_pnl_h = (premium_in - premium_out) * N * index - fees   (covered rows)
The LONG payoff is exactly the negation of the option leg:
    real_long_pnl_h  = -(premium_in - premium_out) * N * index - fees
(the fee is the same 4-leg cost either way; sign only flips the premium diff).
We store `real_short_pnl_nc_<h>` and `premium_tier_<h>` (0=uncovered). The greedy
search keeps only fill-covered rows for whichever horizon a strategy trades.

COMPONENT FEATURES (everything the add-back pool keys on)
--------------------------------------------------------
forecast_edge_dvol = rv_pred_pct - dvol_close      (vs exchange DVOL, cleanest)
forecast_edge_iv   = rv_pred_pct - iv              (vs native ATM IV)
forecast_edge_svi  = rv_pred_pct - svi_gated       (vs quality-gated SVI, NaN if n<5)
svi_n_strikes, jump_fraction, term_slope (svi_atm_2w - svi_atm_1w; contango>0),
svi_25d_skew_1w, hour_utc, dte_days, vega, gamma, theta, index_price,
realized_var_h, implied_var_h, dvol_pct_1y (trailing-1y DVOL percentile = crisis
marker). rv_pred is reused from the predictions file -- NOT recomputed.

Console is cp1252 -> ASCII only on stdout. Cache + report are utf-8.

USAGE
-----
    .venv/Scripts/python.exe scripts/build_candidate_table.py          # SMOKE (3 dates)
    .venv/Scripts/python.exe scripts/build_candidate_table.py --full   # FULL window
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vol_arb.backtester import BacktestConfig
from vol_arb.honest_pnl import mark_trade_log_to_real_premium
from vol_arb.options_loader import load_option_trades

TRADES_DIR = "data/deribit/trades/BTC_option"
OUT_DIR = ROOT / "vol_arb" / "output"
CACHE_SMOKE = OUT_DIR / "candidate_table_smoke.parquet"
CACHE_FULL = OUT_DIR / "candidate_table.parquet"
MINUTES_PER_YEAR = 365.25 * 24 * 60

# Hold horizons in minutes; expiry handled specially (hold to option expiry).
HOLDS = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "expiry": None,
}

# Fill-covered window starts 2023-05 (the option tape only covers straddle
# marking densely from there -- matches HONEST_CORE_REBUILD).
FULL_START = "2023-05-01"
SMOKE_DATES = ["2024-01-15", "2024-06-15", "2024-11-15"]


# ----------------------------------------------------------------------------
def load_core(cfg: BacktestConfig) -> pd.DataFrame:
    """Load vol_arb dataset + predictions; attach rv_pred and clean-IV refs.

    Mirrors vol_arb/backtester.load_data EXACTLY: rv_pred_pct = exp(y_pred)*100.
    """
    df = pd.read_parquet(cfg.vol_arb_path)
    oof = pd.read_parquet(cfg.predictions_path)
    df = df.join(oof[["y_pred"]], how="left")
    df["rv_pred_pct"] = np.exp(df["y_pred"]) * 100.0

    df["iv_dvol_close"] = df["dvol_close"]
    df["iv_iv"] = df["iv"]
    svi = df["svi_iv_matched"].copy()
    if "svi_n_strikes" in df.columns:
        svi = svi.mask(df["svi_n_strikes"] < 5)
    df["iv_svi_gated"] = svi
    return df


def attach_exit(df: pd.DataFrame, hold_min: int | None) -> pd.DataFrame:
    """Attach exit-time S and DVOL for a hold horizon via causal backward
    merge_asof on the bar grid (mirrors honest_core_rebuild.build_hold_index)."""
    out = pd.DataFrame(index=df.index)
    bar = df[["index_price", "dvol_close"]].sort_index()

    if hold_min is None:
        exit_ts = pd.to_datetime(df["expiry_dt"], utc=True)
    else:
        exit_ts = df.index + pd.Timedelta(minutes=hold_min)
    out["exit_ts"] = exit_ts

    exit_ts_all = pd.DatetimeIndex(pd.to_datetime(exit_ts, utc=True))
    valid = ~np.asarray(exit_ts_all.isna())
    q = pd.DataFrame({"exit_ts": exit_ts_all[valid]},
                     index=np.arange(len(df))[valid]).sort_values("exit_ts")
    bar_r = bar.reset_index().rename(columns={"timestamp": "bar_ts"})
    if "index" in bar_r.columns and "bar_ts" not in bar_r.columns:
        bar_r = bar_r.rename(columns={"index": "bar_ts"})
    bar_r = bar_r.sort_values("bar_ts")
    merged = pd.merge_asof(
        q, bar_r, left_on="exit_ts", right_on="bar_ts",
        direction="backward", tolerance=pd.Timedelta(days=20),
    ).sort_index()

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
    entry_ts = pd.to_datetime(df.index, utc=True)
    elapsed = (pd.to_datetime(out["bar_ts_exit"], utc=True) - entry_ts)
    out["h_years"] = elapsed.dt.total_seconds().to_numpy() / (MINUTES_PER_YEAR * 60)
    out["h_days"] = out["h_years"] * 365.25
    return out


def mark_horizon(df: pd.DataFrame, exit_df: pd.DataFrame, trades: pd.DataFrame,
                 hold_min: int | None) -> pd.DataFrame:
    """Mark the SHORT straddle to real fills for one horizon. Returns a frame
    aligned to df.index with real_short_pnl_nc and premium_tier (0=uncovered).

    Only rows with a valid straddle (finite strike/expiry) and finite exit
    timestamp are markable; the rest get tier=0 / NaN.
    """
    markable = (np.isfinite(df["strike"].to_numpy(dtype=float))
                & pd.notna(df["expiry_dt"])
                & pd.notna(pd.to_datetime(exit_df["exit_ts"], utc=True, errors="coerce"))
                & np.isfinite(exit_df["S_exit"].to_numpy()))
    sub = df[markable]
    exsub = exit_df[markable]
    out = pd.DataFrame(index=df.index)
    out["real_short_pnl_nc"] = np.nan
    out["premium_tier"] = 0
    if len(sub) == 0:
        return out

    tl = pd.DataFrame(index=sub.index)
    tl["strike"] = sub["strike"].to_numpy()
    tl["expiry_dt"] = sub["expiry_dt"].to_numpy()
    tl["signal"] = -1                      # SHORT; long payoff = -short option leg
    tl["N_contracts_nc"] = 1.0             # per 1 BTC notional
    tl["index_price"] = sub["index_price"].to_numpy()
    tl["exit_target"] = pd.to_datetime(exsub["exit_ts"], utc=True).to_numpy()

    tol = 60.0 if hold_min is None else max(60.0, float(hold_min))
    res = mark_trade_log_to_real_premium(tl, trades, tol_minutes=tol)
    marked = res.trade_log
    out.loc[marked.index, "real_short_pnl_nc"] = marked["real_option_pnl_nc"].to_numpy()
    out.loc[marked.index, "premium_tier"] = marked["premium_tier"].to_numpy()
    return out


def trailing_dvol_pct(dvol: pd.Series, window_bars: int) -> np.ndarray:
    """Trailing percentile rank of current DVOL within the past `window_bars`
    (causal, shift(1) so the current bar is excluded -> no lookahead)."""
    s = dvol.astype(float)
    # rolling rank of the last value within the trailing window
    def _rank(x):
        # fraction of past values <= current (the last element)
        cur = x[-1]
        past = x[:-1]
        past = past[np.isfinite(past)]
        if len(past) == 0:
            return np.nan
        return float((past <= cur).mean())
    # window+1 so we have `window_bars` past values plus the current
    r = s.rolling(window_bars + 1, min_periods=20).apply(_rank, raw=True)
    return r.to_numpy()


def build(df: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    """Assemble the candidate table over df (already windowed)."""
    t0 = time.time()
    # Load the option tape ONCE over the full window (+1d pad for expiry exits).
    start = df.index.min().strftime("%Y-%m-%d")
    # expiry exits can be up to ~30d past entry; pad generously
    end = (pd.to_datetime(df["expiry_dt"], utc=True).max()
           + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"    loading option tape {start}..{end} ...")
    trades = load_option_trades(TRADES_DIR, start_date=start, end_date=end)
    print(f"    option tape rows: {len(trades):,}  ({time.time()-t0:.1f}s)")

    out = pd.DataFrame(index=df.index)
    # --- component features (entry-time, causal) ---
    out["rv_pred_pct"] = df["rv_pred_pct"]
    out["dvol_close"] = df["dvol_close"]
    out["iv"] = df["iv"]
    out["svi_iv_matched"] = df["svi_iv_matched"]
    out["svi_gated"] = df["iv_svi_gated"]
    out["forecast_edge_dvol"] = df["rv_pred_pct"] - df["dvol_close"]
    out["forecast_edge_iv"] = df["rv_pred_pct"] - df["iv"]
    out["forecast_edge_svi"] = df["rv_pred_pct"] - df["iv_svi_gated"]
    out["svi_n_strikes"] = df.get("svi_n_strikes", np.nan)
    out["jump_fraction"] = df.get("jump_fraction", np.nan)
    # contango term slope: 2w - 1w ATM SVI (positive = upward sloping = contango)
    out["term_slope"] = df.get("svi_atm_2w", np.nan) - df.get("svi_atm_1w", np.nan)
    out["svi_25d_skew_1w"] = df.get("svi_25d_skew_1w", np.nan)
    out["hour_utc"] = df.index.hour
    out["dte_days"] = df.get("dte_days", np.nan)
    out["vega"] = df.get("vega", np.nan)
    out["gamma"] = df.get("gamma", np.nan)
    out["theta"] = df.get("theta", np.nan)
    out["delta"] = df.get("delta", np.nan)
    out["index_price"] = df["index_price"]
    out["strike"] = df.get("strike", np.nan)
    out["T_years"] = df.get("T_years", np.nan)
    out["year"] = df.index.year

    # crisis/regime marker: trailing-1y DVOL percentile (causal)
    bars_1y = int(365.25 * 24 * 60 / 15)   # 15-min bars in a year
    if smoke:
        bars_1y = 50  # smoke window too short for 1y; use a short trailing window
    print(f"    computing trailing DVOL percentile (window {bars_1y} bars) ...")
    out["dvol_pct_1y"] = trailing_dvol_pct(df["dvol_close"], bars_1y)

    # --- honest objective: real-fill-marked straddle PnL per horizon ---
    for hold_name, hold_min in HOLDS.items():
        th = time.time()
        exit_df = attach_exit(df, hold_min)
        # entry/exit realized & implied variance (for sizing features & ref)
        with np.errstate(invalid="ignore", divide="ignore"):
            logret = np.log(exit_df["S_exit"].to_numpy() / df["index_price"].to_numpy())
        out[f"realized_var_{hold_name}"] = logret ** 2
        out[f"implied_var_{hold_name}"] = (df["dvol_close"].to_numpy() / 100.0) ** 2 * exit_df["h_years"].to_numpy()
        out[f"h_years_{hold_name}"] = exit_df["h_years"].to_numpy()
        mk = mark_horizon(df, exit_df, trades, hold_min)
        out[f"real_short_pnl_nc_{hold_name}"] = mk["real_short_pnl_nc"]
        out[f"premium_tier_{hold_name}"] = mk["premium_tier"]
        cov = (mk["premium_tier"] > 0).mean()
        print(f"    horizon {hold_name:7s}: covered {100*cov:5.1f}%  ({time.time()-th:.1f}s)")

    print(f"    candidate table built in {time.time()-t0:.1f}s")
    return out


def main(smoke: bool):
    cfg = BacktestConfig()
    print("=" * 70)
    print("BUILD CANDIDATE TABLE  smoke=" + str(smoke))
    print("=" * 70)
    df_full = load_core(cfg)
    print(f"loaded core: {len(df_full):,} bars  "
          f"{df_full.index.min()} .. {df_full.index.max()}")

    if smoke:
        masks = []
        for d in SMOKE_DATES:
            d0 = pd.Timestamp(d, tz="UTC")
            d1 = d0 + pd.Timedelta(days=1)
            masks.append((df_full.index >= d0) & (df_full.index < d1))
        sel = np.logical_or.reduce(masks)
        df = df_full[sel].copy()
        cache = CACHE_SMOKE
        print(f"SMOKE window {SMOKE_DATES} -> {len(df)} bars")
    else:
        df = df_full.loc[pd.Timestamp(FULL_START, tz="UTC"):].copy()
        cache = CACHE_FULL
        print(f"FULL window {FULL_START}.. -> {len(df):,} bars")

    tbl = build(df, smoke)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(cache)
    print(f"\nwrote cache: {cache}  ({len(tbl):,} rows, {tbl.shape[1]} cols)")

    # coverage summary
    print("\nCOVERAGE SUMMARY (fill-covered rows per horizon):")
    for h in HOLDS:
        tier = tbl[f"premium_tier_{h}"]
        cov = (tier > 0)
        t1 = (tier == 1)
        pnl = tbl.loc[cov, f"real_short_pnl_nc_{h}"]
        print(f"  {h:7s}: covered {cov.sum():6d} ({100*cov.mean():5.1f}%)  "
              f"tier1 {t1.sum():6d}  short_mean=${pnl.mean():+.1f}  "
              f"short_total=${pnl.sum():+,.0f}")


if __name__ == "__main__":
    smoke = "--full" not in sys.argv
    main(smoke=smoke)
