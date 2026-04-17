"""Vol-arb merge pipeline: options + DVOL + ML features.

Orchestrates the point-in-time merge using pd.merge_asof (backward direction)
so each 15m dollar-bar window sees only the most recent option market state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.config import VolArbConfig
from vol_arb.greeks import compute_greeks
from vol_arb.options_loader import load_option_trades, filter_atm

logger = logging.getLogger(__name__)

# Default slippage surface bucket edges (BTC; ETH overrides via VolArbConfig).
# Kept as module-level BTC baseline so existing callers without a config arg
# reproduce the legacy behavior exactly.
_SIZE_EDGES = [0, 1, 3, 10, np.inf]
_VOL_EDGES = [0, 45, 65, np.inf]
_N_SIZE = len(_SIZE_EDGES) - 1  # 4
_N_VOL = len(_VOL_EDGES) - 1    # 3
SLIP_COLS = [f"slip_s{s}_v{v}" for s in range(_N_SIZE) for v in range(_N_VOL)]


def compute_slippage_surface(
    atm: pd.DataFrame,
    size_edges: list[float] | None = None,
    vol_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Trailing 30-day 2D slippage surface: size_bucket x vol_bucket.

    Computes per-trade dollar slippage (abs(price - mark_price) * index_price),
    then rolling 30-day median per (size_bucket, vol_bucket) at daily granularity.
    Fully vectorized, lookahead-free.

    Args:
        atm: ATM trade tape with price, mark_price, index_price, amount, iv.
        size_edges: bucket edges for the trade ``amount`` column. Default: BTC
            `[0, 1, 3, 10, ∞]`. Pass ETH edges `[0, 10, 30, 100, ∞]` when
            building an ETH surface — matches the ~10× larger ETH notional/trade.
        vol_edges: bucket edges for IV %. Default: BTC `[0, 45, 65, ∞]`; ETH
            should use `[0, 55, 85, ∞]` to reflect its higher IV baseline.

    Returns daily DataFrame with timestamp + (n_size × n_vol) slippage columns
    named ``slip_s{i}_v{j}``.
    """
    if size_edges is None:
        size_edges = _SIZE_EDGES
    if vol_edges is None:
        vol_edges = _VOL_EDGES
    n_size = len(size_edges) - 1
    n_vol = len(vol_edges) - 1

    df = atm[["timestamp", "price", "mark_price", "index_price", "amount", "iv"]].copy()
    df["slippage_usd"] = (df["price"] - df["mark_price"]).abs() * df["index_price"]
    df["date"] = df["timestamp"].dt.normalize()

    df["size_bucket"] = pd.cut(
        df["amount"], bins=size_edges, labels=range(n_size), right=False,
    ).astype(int)
    df["vol_bucket"] = pd.cut(
        df["iv"], bins=vol_edges, labels=range(n_vol), right=False,
    ).astype(int)

    # Daily median slippage per (size_bucket, vol_bucket).
    daily = (
        df.groupby(["date", "size_bucket", "vol_bucket"])["slippage_usd"]
        .median()
        .unstack(["size_bucket", "vol_bucket"])
    )

    # Reindex to full (size × vol) cartesian so every configured cell exists.
    # Empty buckets (e.g. any bucket whose edges are outside the observed
    # distribution) are resolved below by within-vol-stratum interpolation —
    # not a global row-median hack — so neighbouring size buckets propagate
    # their slippage estimate to empty size buckets at the same vol level.
    full_cols = pd.MultiIndex.from_product(
        [range(n_size), range(n_vol)], names=["size_bucket", "vol_bucket"],
    )
    daily = daily.reindex(columns=full_cols)
    daily.columns = [f"slip_s{s}_v{v}" for s, v in daily.columns]

    # Rolling 30-day median per cell. min_periods=1 keeps day-N estimate
    # strictly trailing (lookahead-free). ffill handles mid-series gaps.
    surface = daily.rolling(30, min_periods=1).median().ffill()

    # Within-vol-stratum fallback: for any size bucket that stayed NaN because
    # no trades populated that (size, vol) cell, interpolate across size
    # buckets at the same vol level. This is elegant because slippage is
    # monotonically increasing in size within a given vol regime, so
    # neighbouring buckets are the best available estimator for missing ones.
    for v in range(n_vol):
        stratum = [f"slip_s{s}_v{v}" for s in range(n_size)]
        surface[stratum] = (
            surface[stratum]
            .interpolate(axis=1, limit_direction="both")
        )

    surface = surface.reset_index().rename(columns={"date": "timestamp"})
    sample_s0_v0 = f"slip_s0_v{max(0, n_vol // 2)}"
    sample_s_last = f"slip_s{n_size - 1}_v{max(0, n_vol // 2)}"
    logger.info(
        "Slippage surface: %d days, %s=%.2f %s=%.2f USD/underlying",
        len(surface), sample_s0_v0, surface[sample_s0_v0].median(),
        sample_s_last, surface[sample_s_last].median(),
    )
    return surface


def forward_merge_taker_prints(
    merged: pd.DataFrame,
    atm: pd.DataFrame,
    tolerance_seconds: int = 5,
) -> pd.DataFrame:
    """Forward merge_asof: next taker sell/buy within tolerance window.

    For each 15m bar, finds the very next ATM taker sell and taker buy
    that printed within ``tolerance_seconds``.  Used by the backtester's
    deterministic SOR to decide maker vs taker execution.

    Adds columns: next_sell_price_5s, next_sell_mark_5s,
                  next_buy_price_5s,  next_buy_mark_5s   (NaN if none).
    """
    tol = pd.Timedelta(seconds=tolerance_seconds)

    # Split by direction, keep only needed columns
    sells = (
        atm.loc[atm["direction"] == "sell", ["timestamp", "price", "mark_price"]]
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .rename(columns={"price": "next_sell_price_5s",
                         "mark_price": "next_sell_mark_5s"})
    )
    buys = (
        atm.loc[atm["direction"] == "buy", ["timestamp", "price", "mark_price"]]
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .rename(columns={"price": "next_buy_price_5s",
                         "mark_price": "next_buy_mark_5s"})
    )

    grid = merged.reset_index().sort_values("timestamp")
    grid = pd.merge_asof(grid, sells, on="timestamp",
                         direction="forward", tolerance=tol)
    grid = pd.merge_asof(grid, buys, on="timestamp",
                         direction="forward", tolerance=tol)

    sell_hits = grid["next_sell_price_5s"].notna().sum()
    buy_hits = grid["next_buy_price_5s"].notna().sum()
    logger.info("SOR forward merge: %d sell hits, %d buy hits / %d bars (%.1f%%, %.1f%%)",
                sell_hits, buy_hits, len(grid),
                100 * sell_hits / max(len(grid), 1),
                100 * buy_hits / max(len(grid), 1))

    return grid.set_index("timestamp").sort_index()


# Columns to carry from composite instruments into the merged dataset.
# ATM straddle columns keep original names (backward compatible).
# Strangle columns get _bear / _bull suffixes.
_GREEK_COLS = ["iv", "strike", "mark_price", "moneyness", "delta", "gamma", "vega", "theta"]
_ATM_COLS = ["iv", "strike", "index_price", "mark_price",
             "moneyness", "dte_days", "T_years", "delta", "gamma", "vega", "theta",
             "expiry_dt"]
_BEAR_COLS = [f"{c}_bear" for c in _GREEK_COLS]
_BULL_COLS = [f"{c}_bull" for c in _GREEK_COLS]

OPT_MERGE_COLS = ["timestamp"] + _ATM_COLS + _BEAR_COLS + _BULL_COLS


def load_tsrv(path: str) -> pd.DataFrame:
    """Load TSRV 15m, ensure sorted DatetimeIndex with UTC."""
    df = pd.read_parquet(path)
    df = df.sort_index()
    df.index.name = "timestamp"
    return df


def load_dvol(path: str) -> pd.DataFrame:
    """Load DVOL hourly, rename close -> dvol_close, shift to END-stamped.

    Deribit's ``get_volatility_index_data`` API returns START-stamped candles:
    the timestamp marks the OPEN of the hour, and the ``close`` field is the
    DVOL value at ``timestamp + 1h``.  Without correction, a feature at
    time T would merge_asof(backward) to a DVOL row whose close is up to
    1 hour in the future — a ~1h lookahead leak into the backtester's
    vol bucketing and dynamic margin calc.

    Fix: shift timestamps forward by one bar (1 hour) so that ``dvol_close``
    at timestamp T represents the volatility index value AT T (end of the
    hour that just closed).  After this shift, merge_asof(backward) at
    feature time T returns only DVOL values known at or before T.

    Verified empirically on BTC_dvol_60m.parquet: row[N+1].open == row[N].close,
    which proves that row[N]'s close is the value at row[N+1]'s timestamp,
    i.e., at the START of the next hour.  Shifting by +1h makes the stored
    timestamp match the value's observation time.
    """
    df = pd.read_parquet(path)
    df = df.rename(columns={"close": "dvol_close"})
    df = df[["timestamp", "dvol_close"]].sort_values("timestamp").copy()
    # Shift to END-stamped convention (lookahead-safe for merge_asof backward)
    df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=1)
    return df


def attach_greeks(df: pd.DataFrame, r: float = 0.0) -> pd.DataFrame:
    """Compute BS Greeks and attach as columns. Vectorized, in-place."""
    sigma = df["iv"].values / 100.0  # percentage -> decimal
    greeks = compute_greeks(
        S=df["index_price"].values,
        K=df["strike"].values,
        T=df["T_years"].values,
        sigma=sigma,
        is_call=(df["option_type"].values == "C"),
        r=r,
    )
    df["delta"] = greeks["delta"]
    df["gamma"] = greeks["gamma"]
    df["vega"] = greeks["vega"]
    df["theta"] = greeks["theta"]
    return df


def _composite_greeks(
    paired: pd.DataFrame, suffix: str = "",
) -> pd.DataFrame:
    """Compute composite Greeks for a call+put pair DataFrame.

    ``paired`` must have columns ending in ``_c`` and ``_p`` (from a merge
    with those suffixes).  Returns composite columns, optionally with a
    suffix (e.g. ``_bear``) for strangle variants.
    """
    s = suffix
    total_vega = paired["vega_c"] + paired["vega_p"]
    paired[f"vega{s}"] = total_vega
    paired[f"gamma{s}"] = paired["gamma_c"] + paired["gamma_p"]
    paired[f"theta{s}"] = paired["theta_c"] + paired["theta_p"]
    paired[f"delta{s}"] = paired["delta_c"] + paired["delta_p"]
    # Vega-weighted IV (falls back to simple avg if total_vega ~ 0)
    safe_vega = total_vega.replace(0, np.nan)
    paired[f"iv{s}"] = (
        (paired["iv_c"] * paired["vega_c"] + paired["iv_p"] * paired["vega_p"])
        / safe_vega
    ).fillna((paired["iv_c"] + paired["iv_p"]) / 2.0)
    paired[f"mark_price{s}"] = paired["mark_price_c"] + paired["mark_price_p"]
    return paired


def build_composite_instruments(
    trades: pd.DataFrame,
    bucket_freq: str = "15min",
) -> pd.DataFrame:
    """Build ATM straddle + bearish/bullish strangle composites per bucket.

    For each 15m bucket and expiry:
      - ATM Straddle:      closest-to-ATM call + closest-to-ATM put
      - Bearish Strangle:  ATM call + 25-delta OTM put
      - Bullish Strangle:  25-delta OTM call + ATM put

    All three share the same expiry (chosen by the ATM straddle's best
    moneyness).  Strangle columns get ``_bear`` / ``_bull`` suffixes;
    ATM columns keep their original names for backward compatibility.

    Fully vectorized — no loops or .apply().

    Parameters
    ----------
    trades : Option trades with Greeks attached (from attach_greeks).
             Must include both ATM and OTM trades (wider moneyness filter).
    bucket_freq : time bucket frequency for pairing (default 15min)

    Returns
    -------
    DataFrame with one row per bucket.  Columns match ``OPT_MERGE_COLS``.

    Timestamp convention
    --------------------
    The output ``timestamp`` column is END-of-bucket: a row at
    ``timestamp = T`` was built from trades in the half-open window
    ``[T - bucket_freq, T)``.  This makes the row observable AT ``T``
    (no forward peek) and matches the end-stamping convention used by
    the SVI calibrator.  Upstream merges (``merge_asof(direction="backward")``)
    at feature time ``T`` will return options built only from trades that
    happened before ``T``.
    """
    df = trades.copy()
    df["bucket"] = df["timestamp"].dt.floor(bucket_freq)
    df["abs_moneyness"] = (df["moneyness"] - 1.0).abs()

    # ── ATM legs: closest to moneyness = 1.0 ────────────────────
    atm_best = (
        df.sort_values("abs_moneyness")
        .groupby(["bucket", "expiry_dt", "option_type"], observed=True)
        .first()
        .reset_index()
    )
    atm_calls = atm_best[atm_best["option_type"] == "C"]
    atm_puts = atm_best[atm_best["option_type"] == "P"]

    # ── 25-delta legs: closest to |delta| = 0.25, OTM only ─────
    otm_calls = df[(df["option_type"] == "C") & (df["moneyness"] > 1.0)].copy()
    otm_puts = df[(df["option_type"] == "P") & (df["moneyness"] < 1.0)].copy()

    otm_calls["delta_dist_25d"] = (otm_calls["delta"] - 0.25).abs()
    otm_puts["delta_dist_25d"] = (otm_puts["delta"].abs() - 0.25).abs()

    best_25d_call = (
        otm_calls.sort_values("delta_dist_25d")
        .groupby(["bucket", "expiry_dt"], observed=True)
        .first()
        .reset_index()
    )
    best_25d_put = (
        otm_puts.sort_values("delta_dist_25d")
        .groupby(["bucket", "expiry_dt"], observed=True)
        .first()
        .reset_index()
    )

    # ── Pair: ATM straddle ──────────────────────────────────────
    straddle = atm_calls.merge(
        atm_puts, on=["bucket", "expiry_dt"], suffixes=("_c", "_p"), how="inner",
    )
    logger.info("Straddle pairing: %d calls, %d puts -> %d pairs",
                len(atm_calls), len(atm_puts), len(straddle))

    if straddle.empty:
        logger.warning("No call+put pairs found!")
        return pd.DataFrame(columns=OPT_MERGE_COLS)

    _composite_greeks(straddle, suffix="")
    straddle["strike"] = (straddle["strike_c"] + straddle["strike_p"]) / 2.0
    straddle["moneyness"] = (straddle["moneyness_c"] + straddle["moneyness_p"]) / 2.0
    straddle["index_price"] = straddle["index_price_c"]
    straddle["dte_days"] = straddle["dte_days_c"]
    straddle["T_years"] = straddle["T_years_c"]
    straddle["abs_moneyness_avg"] = (
        straddle["abs_moneyness_c"] + straddle["abs_moneyness_p"]
    ) / 2.0

    # Per bucket: keep the expiry with smallest avg abs_moneyness
    best = (
        straddle.sort_values("abs_moneyness_avg")
        .groupby("bucket", observed=True)
        .first()
        .reset_index()
    )
    # Keep the winning (bucket, expiry_dt) pairs for strangle joins
    best_keys = best[["bucket", "expiry_dt"]].copy()

    # ── Pair: bearish strangle (ATM call + 25d put) ─────────────
    bear = atm_calls.merge(
        best_25d_put, on=["bucket", "expiry_dt"], suffixes=("_c", "_p"), how="inner",
    )
    _composite_greeks(bear, suffix="_bear")
    bear["strike_bear"] = bear["strike_p"]  # put determines skew strike
    bear["moneyness_bear"] = bear["moneyness_p"]
    # Restrict to same expiry chosen by ATM straddle
    bear = best_keys.merge(bear, on=["bucket", "expiry_dt"], how="inner")
    bear = bear.drop_duplicates(subset="bucket", keep="first")

    # ── Pair: bullish strangle (25d call + ATM put) ─────────────
    bull = best_25d_call.merge(
        atm_puts, on=["bucket", "expiry_dt"], suffixes=("_c", "_p"), how="inner",
    )
    _composite_greeks(bull, suffix="_bull")
    bull["strike_bull"] = bull["strike_c"]  # call determines skew strike
    bull["moneyness_bull"] = bull["moneyness_c"]
    # Restrict to same expiry
    bull = best_keys.merge(bull, on=["bucket", "expiry_dt"], how="inner")
    bull = bull.drop_duplicates(subset="bucket", keep="first")

    # ── Merge strangles onto straddle base ──────────────────────
    bear_cols = ["bucket"] + _BEAR_COLS
    bull_cols = ["bucket"] + _BULL_COLS
    result = best.merge(
        bear[[c for c in bear_cols if c in bear.columns]],
        on="bucket", how="left",
    ).merge(
        bull[[c for c in bull_cols if c in bull.columns]],
        on="bucket", how="left",
    )

    # END-of-bucket stamping: a bar at T was built from trades in [T - freq, T).
    # This is the source-level fix for the lookahead documented in
    # scripts/forensic/01_verify_bar_timing.py and TSRV/PIPELINE_DOCUMENTATION.md
    # Stage 14.  Must stay in sync with SVI (vol_arb/svi.py:1482, 1494).
    result["timestamp"] = result["bucket"] + pd.Timedelta(bucket_freq)

    # Stats
    n_bear = result[_BEAR_COLS[0]].notna().sum()
    n_bull = result[_BULL_COLS[0]].notna().sum()
    logger.info("Composite instruments: %d buckets", len(result))
    logger.info("  ATM straddle: %d (100%%)", len(result))
    logger.info("  Bearish strangle: %d (%.1f%%)", n_bear, 100 * n_bear / len(result))
    logger.info("  Bullish strangle: %d (%.1f%%)", n_bull, 100 * n_bull / len(result))
    logger.info("  ATM delta: %.4f (expect ~0)", result["delta"].mean())
    if n_bear > 0:
        logger.info("  Bear delta: %.4f (ATM call + OTM put -> positive)",
                    result["delta_bear"].dropna().mean())
    if n_bull > 0:
        logger.info("  Bull delta: %.4f (OTM call + ATM put -> negative)",
                    result["delta_bull"].dropna().mean())

    # Ensure all OPT_MERGE_COLS exist (NaN for missing strangles)
    for col in OPT_MERGE_COLS:
        if col not in result.columns:
            result[col] = np.nan

    return result[OPT_MERGE_COLS].copy()


def merge_options_onto_features(
    features: pd.DataFrame,
    options_atm: pd.DataFrame,
    dvol: pd.DataFrame | None = None,
    trailing_vol: pd.DataFrame | None = None,
    slippage_surface: pd.DataFrame | None = None,
    ffill_limit: int | None = None,
) -> pd.DataFrame:
    """Backward merge_asof: latest ATM option trade before each 15m bar.

    Parameters
    ----------
    features : TSRV 15m DataFrame (DatetimeIndex)
    options_atm : ATM-filtered options with Greeks attached
    dvol : optional DVOL hourly DataFrame
    trailing_vol : optional trailing 1h ATM volume DataFrame
    slippage_surface : optional 2D slippage surface DataFrame (12 slip_s*_v* cols)
    ffill_limit : forward-fill limit for option columns (None = no fill)
    """
    # Prepare features: index -> column for merge_asof
    feat = features.reset_index()
    feat = feat.sort_values("timestamp")

    # Prepare options: sort, deduplicate (keep last trade at each timestamp)
    sort_cols = ["timestamp"]
    if "trade_seq" in options_atm.columns:
        sort_cols.append("trade_seq")
    opt = (
        options_atm.sort_values(sort_cols)
        [OPT_MERGE_COLS]
        .drop_duplicates(subset="timestamp", keep="last")
    )
    logger.info("Options for merge: %d unique timestamps", len(opt))

    # Merge 1: ATM options -> features (backward)
    merged = pd.merge_asof(
        feat,
        opt,
        on="timestamp",
        direction="backward",
        suffixes=("", "_opt"),
    )

    # Optional forward-fill for sparse option data (ATM cols only;
    # strangle cols are NaN where no 25d strike existed and should stay NaN)
    atm_opt_cols = ["iv", "strike", "index_price", "moneyness",
                    "dte_days", "T_years", "delta", "gamma", "vega", "theta",
                    "expiry_dt"]
    if ffill_limit is not None:
        merged[atm_opt_cols] = merged[atm_opt_cols].ffill(limit=ffill_limit)

    # Merge 2: DVOL -> merged (backward)
    if dvol is not None:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            dvol,
            on="timestamp",
            direction="backward",
        )

    # Merge 3: Slippage surface -> merged (backward, daily granularity)
    if slippage_surface is not None:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            slippage_surface.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    # Merge 4: Trailing ATM volume -> merged (backward)
    if trailing_vol is not None:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            trailing_vol.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    # Derived signal columns
    merged["rv_pct"] = merged["vol_ann"] * 100  # annualized RV in %
    if "iv" in merged.columns:
        merged["rv_iv_spread"] = merged["rv_pct"] - merged["iv"]
    if "dvol_close" in merged.columns:
        merged["rv_dvol_spread"] = merged["rv_pct"] - merged["dvol_close"]
        if "iv" in merged.columns:
            merged["iv_premium"] = merged["iv"] - merged["dvol_close"]

    # Restore timestamp index
    merged = merged.set_index("timestamp").sort_index()

    return merged


def build_vol_arb_dataset(config: VolArbConfig) -> pd.DataFrame:
    """Full pipeline: load, parse, filter, compute Greeks, merge.

    Returns the merged DataFrame with all TSRV columns + option Greeks + DVOL.
    """
    # 1. Load TSRV features
    logger.info("Loading TSRV from %s", config.tsrv_path)
    tsrv = load_tsrv(config.tsrv_path)
    start_date = tsrv.index.min().strftime("%Y-%m-%d")
    end_date = tsrv.index.max().strftime("%Y-%m-%d")
    logger.info("TSRV: %d rows, %s to %s", len(tsrv), start_date, end_date)

    # 2. Load option trades (only overlap period)
    logger.info("Loading option trades from %s", config.option_trade_dir)
    trades = load_option_trades(config.option_trade_dir, start_date, end_date)

    # 2b. SVI surface calibration (all strikes, all expiries)
    svi_features = None
    svi_params = None
    if config.svi_enabled:
        from vol_arb.svi import calibrate_surface, SVIConfig
        from vol_arb.paths import svi_features_path, svi_surface_params_path
        svi_config = SVIConfig(
            bucket_freq=config.svi_bucket_freq,
            min_strikes=config.svi_min_strikes,
            fallback_min_strikes=config.svi_fallback_min_strikes,
            cache_params=config.svi_cache,
        )
        svi_features, svi_params = calibrate_surface(trades, svi_config)
        if not svi_features.empty:
            logger.info("SVI calibration: %d buckets, %.1f%% full fit (level 0), %d param rows",
                        len(svi_features),
                        100 * (svi_features["svi_calibration_level"] == 0).mean(),
                        len(svi_params))
            # Persist to per-currency paths so downstream pruned+SVI training
            # and DRL hedging can load the correct surface. BTC writes to the
            # legacy unsuffixed filename (preserving byte-compat); ETH and
            # future currencies get a {currency}_ prefix via the resolver.
            feat_out = svi_features_path(config.currency)
            params_out = svi_surface_params_path(config.currency)
            feat_out.parent.mkdir(parents=True, exist_ok=True)
            svi_features.to_parquet(feat_out)
            svi_params.to_parquet(params_out)
            logger.info("SVI persisted: features=%s, params=%s", feat_out, params_out)
        else:
            logger.warning("SVI calibration produced no results")
            svi_features = None
            svi_params = None

    # 3. Filter: ATM for slippage/volume, wider for strangle legs
    atm = filter_atm(
        trades,
        moneyness_lo=config.moneyness_lo,
        moneyness_hi=config.moneyness_hi,
        dte_min=config.dte_min_days,
        dte_max=config.dte_max_days,
    )
    wide = filter_atm(
        trades,
        moneyness_lo=config.strangle_moneyness_lo,
        moneyness_hi=config.strangle_moneyness_hi,
        dte_min=config.dte_min_days,
        dte_max=config.dte_max_days,
    )

    # 4. Compute Greeks on the wider set (ATM is a subset)
    logger.info("Computing BS Greeks on %d trades (wide filter)...", len(wide))
    wide = attach_greeks(wide, r=config.risk_free_rate)
    # ATM subset (for slippage surface and volume, which should use tight filter)
    atm = wide[(wide["moneyness"] >= config.moneyness_lo)
               & (wide["moneyness"] <= config.moneyness_hi)].copy()

    # 4b. 2D slippage surface (trailing 30-day, lookahead-free, ATM only).
    # Per-currency bucket edges flow from VolArbConfig → ETH uses shifted
    # size/vol ranges that match its larger notional-per-trade and higher IV.
    logger.info("Computing 2D slippage surface...")
    slippage_df = compute_slippage_surface(
        atm,
        size_edges=config.slippage_size_edges,
        vol_edges=config.slippage_vol_edges,
    )

    # 4c. Trailing 1h ATM volume (lookahead-free)
    vol_15m = atm.set_index("timestamp").resample("15min")["amount"].sum()
    trailing_vol = vol_15m.rolling(4, min_periods=1).sum().rename("trailing_1h_volume")
    trailing_vol_df = trailing_vol.reset_index()
    logger.info("Trailing 1h ATM volume: median=%.1f BTC, mean=%.1f BTC (%d buckets)",
                trailing_vol.median(), trailing_vol.mean(), len(trailing_vol))

    # 5. Build composite instruments (ATM straddle + bear/bull strangles)
    logger.info("Building composite instruments (straddle + strangles)...")
    straddles = build_composite_instruments(wide, bucket_freq=config.bucket_freq)

    # 6. Load DVOL
    dvol = None
    dvol_path = Path(config.dvol_path)
    if dvol_path.exists():
        logger.info("Loading DVOL from %s", config.dvol_path)
        dvol = load_dvol(config.dvol_path)

    # 7. Merge straddle composites onto TSRV grid
    logger.info("Merging %d straddles onto %d TSRV bars...",
                len(straddles), len(tsrv))
    merged = merge_options_onto_features(
        tsrv, straddles, dvol=dvol, trailing_vol=trailing_vol_df,
        slippage_surface=slippage_df, ffill_limit=config.ffill_limit_bars,
    )

    # 7b. Forward-merge taker prints for SOR fill detection
    logger.info("Forward-merging taker prints for SOR...")
    merged = forward_merge_taker_prints(
        merged, atm, tolerance_seconds=config.sor_tolerance_seconds,
    )

    # 7c. Merge SVI surface features (backward, 1h granularity)
    if svi_features is not None and not svi_features.empty:
        from vol_arb.svi import evaluate_surface_batch

        logger.info("Merging SVI surface features (%d buckets)...", len(svi_features))
        merged_reset = merged.reset_index().sort_values("timestamp")
        merged_reset = pd.merge_asof(
            merged_reset,
            svi_features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        # Maturity-matched IV: evaluate SVI surface at each straddle's EXACT expiry.
        # Uses vectorized merge_asof(by='_exp_key') to find the slice calibrated
        # for the straddle's exact expiry date — no T_years interpolation or
        # extrapolation. This is how real options desks compute market IV.
        if svi_params is not None and not svi_params.empty:
            logger.info("Computing svi_iv_matched at straddle expiry_dt for %d bars...", len(merged_reset))
            query_mask = (
                merged_reset["strike"].notna()
                & merged_reset["expiry_dt"].notna()
                & merged_reset["index_price"].notna()
            )
            query_df = merged_reset.loc[query_mask, [
                "timestamp", "strike", "expiry_dt", "index_price"
            ]].copy()
            matched = evaluate_surface_batch(svi_params, query_df)
            merged_reset["svi_iv_matched"] = np.nan
            merged_reset.loc[query_mask, "svi_iv_matched"] = matched.values
            n_matched = merged_reset["svi_iv_matched"].notna().sum()
            logger.info("svi_iv_matched: %d / %d bars (%.1f%%)",
                        n_matched, len(merged_reset), 100 * n_matched / max(len(merged_reset), 1))

        merged = merged_reset.set_index("timestamp").sort_index()

        cov_cm = 100 * merged["svi_atm_1w"].notna().mean() if "svi_atm_1w" in merged.columns else 0
        logger.info("SVI constant-maturity features merged: %.1f%% coverage (svi_atm_1w)", cov_cm)

    # 8. Optionally join full ML features
    if config.features_path and Path(config.features_path).exists():
        logger.info("Joining full features from %s", config.features_path)
        feat_full = pd.read_parquet(config.features_path)
        # Only add columns not already in merged
        new_cols = [c for c in feat_full.columns if c not in merged.columns]
        if new_cols:
            merged = merged.join(feat_full[new_cols], how="left")
            logger.info("Added %d feature columns", len(new_cols))

    return merged


def verify_alignment(merged: pd.DataFrame) -> None:
    """Print alignment check and coverage stats."""
    total = len(merged)
    iv_coverage = merged["iv"].notna().sum()
    dvol_coverage = merged["dvol_close"].notna().sum() if "dvol_close" in merged.columns else 0

    print("\n=== Vol-Arb Alignment Check ===")
    print(f"Total bars: {total:,}")
    print(f"Trade IV coverage: {iv_coverage:,} / {total:,} ({100*iv_coverage/total:.1f}%)")
    print(f"DVOL coverage:     {dvol_coverage:,} / {total:,} ({100*dvol_coverage/total:.1f}%)")

    # Show a sample row with option data
    with_iv = merged[merged["iv"].notna()]
    if len(with_iv) > 0:
        row = with_iv.iloc[len(with_iv) // 2]
        print(f"\nSample row (index {len(with_iv)//2} of {len(with_iv)} with IV):")
        fields = [
            ("timestamp", row.name),
            ("vol_ann", f"{row['vol_ann']:.4f}"),
            ("jump_fraction", f"{row.get('jump_fraction', np.nan):.4f}"),
            ("iv", f"{row['iv']:.2f}%"),
            ("dvol_close", f"{row.get('dvol_close', np.nan):.2f}%"),
            ("delta (net)", f"{row['delta']:.4f}"),
            ("vega (straddle)", f"{row['vega']:.2f}"),
            ("theta (straddle)", f"{row['theta']:.2f}"),
            ("strike (avg)", f"{row['strike']:.0f}"),
            ("moneyness", f"{row['moneyness']:.4f}"),
            ("dte_days", f"{row['dte_days']:.1f}"),
        ]
        for name, val in fields:
            print(f"  {name:18s} {val}")

        # Sanity checks (straddle-level)
        print("\nSanity checks (straddle composites):")
        mean_delta = with_iv["delta"].mean()
        print(f"  Mean net delta:  {mean_delta:.4f} (expect ~0 for ATM straddle)")
        print(f"  All vega > 0:    {(with_iv['vega'] > 0).all()}")
        print(f"  All theta < 0:   {(with_iv['theta'] < 0).all()}")
    else:
        print("\nWARNING: No bars with trade-level IV data!")
    print()
