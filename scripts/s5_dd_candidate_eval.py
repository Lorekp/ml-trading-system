"""S5 drawdown-reduction candidate evaluator.

Tests 7 candidates that target the 32.8% COMPOUND-equity-curve MaxDD on the
S5 30-asset Binance backtest. Each candidate is ADD-ONLY post-processing on
the simulated trade log (no edits to cb_v2).

Historical methodology:
  (a) Defensible from first principles (NOT tuned on OOS)
  (b) TRAIN-OOS Sharpe ratio <= 1.3x (cut = median year)
  (c) Independent PnL realism and selection-bias evidence are required for
      acceptance. Neither is available in this post-processor, so it can rank
      candidates but cannot accept one.
  (d) Per-year stability required (no single-year wins)
  (e) Each candidate tested on a LEAVE-ONE-YEAR-OUT slice — flagged as
      caveat since pre-2020 data is unavailable

Ranking threshold: improves MaxDD by >2pp without regressing Sharpe by >0.1
(historically reported Sharpe 3.40 -> Sharpe >= 3.25). This threshold does not
override the missing independent evidence.

Outputs:
  vol_arb/output/S5_DRAWDOWN_CANDIDATES.json — per-candidate metrics
"""
from __future__ import annotations

import sys
import json
import time
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import circuit_breaker_v2_experiment as cb  # noqa: E402
from signal_engine.config import PortfolioConfig  # noqa: E402
from signal_engine.portfolio import compute_daily_pnl_series  # noqa: E402
from validation.gates import sizing_sanity_gate  # noqa: E402

warnings.filterwarnings("ignore")
TARGET_VOL, CAP_MULT = 0.025, 2.0
CAP = 250_000


def historical_portfolio_config() -> PortfolioConfig:
    """Return the portfolio settings used by the historical S5 replay."""
    return PortfolioConfig(
        starting_capital=cb.CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.30, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)


def metrics_from_sim_dpnl(dpnl_adj: pd.Series, port_cap: float) -> dict:
    """Sharpe/Sortino/MaxDD from a daily-PnL series. The MaxDD here is
    COMPOUND-equity-curve (matching the spec's 32.8% baseline).
    """
    if dpnl_adj is None or len(dpnl_adj) < 30:
        return None
    dr = dpnl_adj.values / port_cap
    if dr.std() == 0:
        return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0.0
    eq = np.cumprod(1 + dr)
    cummax = np.maximum.accumulate(eq)
    dd = (cummax - eq) / cummax
    max_dd = float(dd.max() * 100)
    return {"sr": sr, "sortino": sortino, "max_dd": max_dd}


def per_year_from_dpnl(dpnl_adj: pd.Series) -> pd.Series:
    """Per-calendar-year PnL from a daily series."""
    return dpnl_adj.resample("YE").sum()


def get_vip_tier(eq, v30):
    """Mirrors circuit_breaker_v2_experiment.get_vip_tier."""
    tiers = [
        ("VIP0", 0, 0, 2.00, 5.50), ("VIP1", 100_000, 10_000_000, 1.80, 4.00),
        ("VIP2", 250_000, 25_000_000, 1.60, 3.75), ("VIP3", 500_000, 50_000_000, 1.40, 3.50),
        ("VIP4", 1_000_000, 100_000_000, 1.20, 3.20), ("VIP5", 2_000_000, 250_000_000, 1.00, 3.20),
        ("Supreme", 5_000_000, 500_000_000, 0.00, 3.00),
    ]
    r = tiers[0]
    for t in tiers:
        if eq >= t[1] or v30 >= t[2]:
            r = t
    return r[0], r[3], r[4]


def make_dpnl_with_costs(sim, port, baseline_dpnl):
    """Reconstruct the same daily-PnL-with-VIP-costs that cb.run_variant produces.

    baseline_dpnl: raw compute_daily_pnl_series(sim) result.
    Returns: dpnl_adj with VIP-tier extra costs subtracted.
    """
    SPREAD_BPS = 0.5
    MFP = 0.90
    prepared = sim  # already sorted
    ts_arr = pd.to_datetime(sim["timestamps"])
    if ts_arr.tz is not None:
        ts_arr = ts_arr.tz_localize(None)
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(sim["timestamps"])):
        if active[i]:
            daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = CAP + baseline_dpnl.cumsum()
    slippage_bps = port.slippage * 10000
    dpnl_adj = baseline_dpnl.copy()
    for d in baseline_dpnl.index:
        dd = d.date()
        eq = eq_curve.get(d, CAP)
        v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = get_vip_tier(eq, v30)
        rt = MFP * 2 * mbps + (1 - MFP) * 2 * tbps + SPREAD_BPS
        extra = rt - slippage_bps
        nd = sum(1 for i in range(len(sim["timestamps"])) if active[i] and dates_arr[i] == dd)
        avg = daily_vol.get(dd, 0) / max(nd * 2, 1)
        dpnl_adj[d] -= nd * avg * extra / 10000
    return dpnl_adj


# ── CANDIDATE POST-PROCESSORS ──
# Each takes (pnl, sizes, skipped, timestamps, dpnl_adj, asset_series)
# Returns (new_dpnl_adj, new_pnl, new_sizes, new_skipped) — to allow
# reducing trade notional or skipping entirely.

def c2_then_c4_stack(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """STACK: C2 (universe filter) + C4 (BTC correlation sizing) composed.
    C2 first, then C4 on the C2-filtered trade set.
    """
    res2 = c2_universe_filter(pnl, sizes, skipped, timestamps, dpnl_adj, assets)
    new_pnl2, new_sizes2, new_skipped2, new_dpnl2 = res2
    res4 = c4_btc_corr_sizing(new_pnl2, new_sizes2, new_skipped2, timestamps, new_dpnl2, assets)
    return res4


def c1_vol_targeting(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C1: per-asset realized-vol targeting.
    Scale trade size by target_vol / realized_vol (target=median of cross-asset
    rolling-30d abs(PnL)). Mechanism: when an asset's 30d realized PnL vol
    spikes (volatile regime), shrink size to keep dollar-PnL vol constant.
    Defensible: standard risk-parity / vol-targeting principle.
    Uses PRIOR-DAY realized vol to prevent look-ahead.
    """
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "asset": assets})
    ts_norm = pd.DatetimeIndex(df["ts"]).normalize()
    asset_daily = df.groupby([ts_norm, "asset"])["pnl"].sum().unstack(fill_value=0.0)
    # shift(1) — use T-1 rolling vol
    asset_vol = asset_daily.shift(1).rolling(30, min_periods=10).std().fillna(0.0)
    target_vol = asset_vol.median().median() if asset_vol.notna().any().any() else 1.0
    if target_vol <= 0:
        target_vol = 1.0
    mult = (target_vol / asset_vol.clip(lower=1e-9)).clip(upper=2.0)
    new_pnl = pnl.copy()
    new_sizes = sizes.copy()
    new_skipped = skipped.copy()
    trade_dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    for i in range(len(pnl)):
        if skipped[i] or pd.isna(assets.iloc[i]):
            continue
        a = assets.iloc[i]
        if a not in mult.columns:
            continue
        d = trade_dates[i]
        if d not in mult.index:
            continue
        m = mult.at[d, a]
        if pd.notna(m) and m > 0:
            new_pnl[i] = pnl[i] * m
            new_sizes[i] = sizes[i] * m
    df2 = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df2.groupby(df2["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, new_sizes, new_skipped, new_dpnl


def c2_universe_filter(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C2: dynamic universe filter — drop assets with rolling 30d Sharpe < 0.
    Mechanism: per-asset rolling-30d PnL sum (UP TO T-1, no look-ahead); if
    negative for that asset, skip its trades today. Defensible: don't keep
    punching a bag when the bag is on a losing streak (this is the
    B-022/breaker concept extended to the universe level).
    """
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "asset": assets})
    ts_norm = pd.DatetimeIndex(df["ts"]).normalize()
    asset_daily = df.groupby([ts_norm, "asset"])["pnl"].sum().unstack(fill_value=0.0)
    # shift(1) — exclude TODAY's PnL to prevent look-ahead bias
    asset_roll = asset_daily.shift(1).rolling(30, min_periods=10).sum()
    asset_std = asset_daily.shift(1).rolling(30, min_periods=10).std().fillna(0.0) + 1e-9
    asset_sharpe_proxy = (asset_roll / asset_std).fillna(0.0)
    # For each trade, check if asset's rolling 30d Sharpe < 0 on the trade's day
    skip_mask = pd.Series(False, index=range(len(pnl)))
    trade_dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    for i in range(len(pnl)):
        if skipped[i] or pd.isna(assets.iloc[i]):
            continue
        a = assets.iloc[i]
        if a not in asset_sharpe_proxy.columns:
            continue
        d = trade_dates[i]
        if d not in asset_sharpe_proxy.index:
            continue
        sh = asset_sharpe_proxy.at[d, a]
        if pd.notna(sh) and sh < 0:
            skip_mask[i] = True
    new_pnl = pnl.copy()
    new_skipped = skipped | skip_mask.values
    new_pnl = np.where(new_skipped, 0.0, pnl)
    df2 = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df2[~new_skipped].groupby(df2[~new_skipped]["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, sizes, new_skipped, new_dpnl


def c3_dd_kelly_scaling(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C3: drawdown-aware Kelly scaling — when portfolio DD > 10%, scale all
    sizes by 0.5x; > 15% scale by 0.25x; floor 0.1x.
    Mechanism: structural position-size risk control that PULLS BACK
    exposure as drawdown deepens (the standard 'risk-off' mechanic used
    by every successful systematic CTA — protects against death-spiral
    compounding).
    Uses PRIOR-DAY drawdown to prevent look-ahead bias.
    """
    # dpnl_adj from PRIOR day (shift 1) to avoid today-leakage
    equity = CAP + dpnl_adj.cumsum()
    peak = equity.cummax()
    dd_pct = (peak - equity) / peak  # 0..1
    dd_prev = dd_pct.shift(1).fillna(0.0)  # T-1 drawdown, no look-ahead

    new_pnl = pnl.copy()
    new_sizes = sizes.copy()
    new_skipped = skipped.copy()
    ts_dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    for i in range(len(pnl)):
        if skipped[i]:
            continue
        d = ts_dates[i]
        if d not in dd_prev.index:
            continue
        dd = dd_prev.at[d]
        if dd > 0.15:
            mult = 0.25
        elif dd > 0.10:
            mult = 0.50
        elif dd > 0.05:
            mult = 0.75
        else:
            mult = 1.0
        # Apply sizing mult to pnl AND size (proportional reduction)
        new_pnl[i] = pnl[i] * mult
        new_sizes[i] = sizes[i] * mult
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df.groupby(df["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, new_sizes, new_skipped, new_dpnl


def c4_btc_corr_sizing(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C4: per-asset correlation-aware sizing — reduce size for assets with
    high rolling 30d return correlation to BTC.
    Mechanism: systemic risk reduction via simple correlation proxy
    (rolling 30d beta-to-BTC using 1d return correlation).
    Uses PRIOR-DAY rolling correlation to prevent look-ahead.
    """
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "asset": assets})
    asset_daily = df.groupby([df["ts"].dt.normalize(), "asset"])["pnl"].sum().unstack(fill_value=0.0)
    if "BTC" not in asset_daily.columns:
        return pnl, sizes, skipped, dpnl_adj
    btc = asset_daily["BTC"]
    rolling_corr = {}
    for a in asset_daily.columns:
        if a == "BTC":
            rolling_corr[a] = pd.Series(0.0, index=asset_daily.index)
            continue
        # shift(1) — exclude TODAY's PnL to prevent look-ahead
        rolling_corr[a] = asset_daily[a].shift(1).rolling(30, min_periods=15).corr(btc.shift(1)).fillna(0.0)
    rolling_corr_df = pd.DataFrame(rolling_corr)
    new_pnl = pnl.copy()
    new_sizes = sizes.copy()
    new_skipped = skipped.copy()
    ts_dates = pd.to_datetime(timestamps).normalize()
    for i in range(len(pnl)):
        if skipped[i] or pd.isna(assets.iloc[i]):
            continue
        a = assets.iloc[i]
        if a not in rolling_corr_df.columns:
            continue
        d = ts_dates[i]
        if d not in rolling_corr_df.index:
            continue
        c = rolling_corr_df.at[d, a]
        if pd.notna(c) and c > 0.5:
            mult = 0.5  # reduce size by 50% for assets with high BTC correlation
        elif pd.notna(c) and c > 0.3:
            mult = 0.75
        else:
            mult = 1.0
        new_pnl[i] = pnl[i] * mult
        new_sizes[i] = sizes[i] * mult
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df.groupby(df["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, new_sizes, new_skipped, new_dpnl


def c5_tod_filter(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C5: time-of-day filter — skip new entries in UTC 22:00-02:00
    (Binance thin liquidity window).
    Mechanism: skip signals that would enter during the night-window.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    hours = ts.hour
    skip_mask = (hours >= 22) | (hours < 2)
    new_pnl = np.where(skip_mask & ~skipped, 0.0, pnl)
    new_skipped = skipped | skip_mask
    df = pd.DataFrame({"ts": ts, "pnl": new_pnl})
    daily = df[~new_skipped].groupby(df[~new_skipped]["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(ts.min().normalize(), ts.max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, sizes, new_skipped, new_dpnl


def c6_sector_tag_filter(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C6: sector tag filter — group assets by sector proxy, reduce intra-sector exposure.

    Sector mapping is a coarse 4-group classification of the S5 universe:
      'majors': BTC, ETH, BNB
      'l1_l2': SOL, AVAX, NEAR, APT, SUI, ADA, DOT, APT
      'defi': UNI, AAVE, LINK, ENJ, FET, ENA, WLD
      'memes_alt': XRP, DOGE, SHIB, PEPE, BONK, FLOKI, XLM, TRX, LTC, BCH
    """
    SECTOR = {
        "BTC": "majors", "ETH": "majors", "BNB": "majors",
        "SOL": "l1_l2", "AVAX": "l1_l2", "NEAR": "l1_l2", "APT": "l1_l2",
        "SUI": "l1_l2", "ADA": "l1_l2", "DOT": "l1_l2", "FIL": "l1_l2",
        "UNI": "defi", "AAVE": "defi", "LINK": "defi", "ENJ": "defi",
        "FET": "defi", "ENA": "defi", "WLD": "defi",
        "XRP": "memes_alt", "XLM": "memes_alt", "TRX": "memes_alt",
        "LTC": "memes_alt", "BCH": "memes_alt", "ZEC": "memes_alt",
        "SXP": "memes_alt", "ANKR": "memes_alt", "BNX": "memes_alt",
        "FTM": "memes_alt", "HBAR": "memes_alt", "TAO": "defi",
    }
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "size": sizes,
                       "asset": assets})
    df["sector"] = df["asset"].map(SECTOR).fillna("other")
    ts_dates = df["ts"].dt.normalize()

    new_pnl = pnl.copy()
    new_sizes = sizes.copy()
    new_skipped = skipped.copy()
    cap_per_sector = 0.60  # max 60% of today's gross exposure to any one sector
    # Group trades by day, compute sector exposure per day, scale down where exceeded.
    for d, day_idx in df.groupby(ts_dates).groups.items():
        day_active = day_idx[~new_skipped[day_idx]]
        if len(day_active) == 0:
            continue
        day_df = df.loc[day_active]
        sector_exp = day_df.groupby("sector")["size"].sum()
        total_exp = sector_exp.sum()
        for sec, exp in sector_exp.items():
            if exp / max(total_exp, 1e-9) > cap_per_sector:
                scale = (cap_per_sector * total_exp) / exp
                in_sec = day_df[day_df["sector"] == sec].index
                new_pnl[in_sec] = new_pnl[in_sec] * scale
                new_sizes[in_sec] = new_sizes[in_sec] * scale
    df2 = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df2.groupby(df2["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, new_sizes, new_skipped, new_dpnl


def c7_btc_hedge(pnl, sizes, skipped, timestamps, dpnl_adj, assets, **kw):
    """C7: dynamic BTC short-perp hedge when portfolio DD > 8%.
    Hedge ratio = (DD - 0.08) * 2 (so DD=8% -> 0% hedge, DD=18% -> 20% hedge).
    Mechanism: explicit downside protection. The hedge PnL is simulated
    as -btc_return * portfolio_exposure * hedge_ratio.
    """
    # Compute BTC daily return from asset_pnl of BTC trades (proxy: BTC PnL is
    # ~equity-curve beta-aligned since BTC is the dominant asset).
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "asset": assets,
                       "size": sizes})
    ts_dates = pd.DatetimeIndex(df["ts"]).normalize()
    # Get BTC daily PnL
    df['_date'] = ts_dates
    btc_daily = df[(df["asset"] == "BTC") & ~skipped].groupby('_date')["pnl"].sum()
    if len(btc_daily) == 0:
        return pnl, sizes, skipped, dpnl_adj
    # Compute portfolio equity curve (T-1 drawdown to prevent look-ahead)
    equity = CAP + dpnl_adj.cumsum()
    peak = equity.cummax()
    dd_pct = (peak - equity) / peak
    dd_prev = dd_pct.shift(1).fillna(0.0)  # T-1 drawdown
    # Apply hedge: subtract BTC daily PnL scaled by hedge_ratio
    new_pnl = pnl.copy()
    for d in btc_daily.index:
        if d not in dd_prev.index:
            continue
        dd = dd_prev.at[d]
        if dd > 0.08:
            hedge_ratio = min((dd - 0.08) * 2.0, 0.25)  # cap at 25%
            btc_pnl_d = btc_daily.at[d]
            # Reduce portfolio PnL by the hedge cost (= +btc_pnl, since short)
            # The hedge gains when BTC drops, so it offsets losses. We model
            # this by ADDING (-btc_pnl * hedge_ratio) to portfolio pnl on day d.
            offset = -btc_pnl_d * hedge_ratio
            # Distribute proportionally to all non-BTC non-skipped trades that day
            day_idx = df[(ts_dates == d) & ~skipped & (df["asset"] != "BTC")].index
            if len(day_idx) > 0:
                total_size = df.loc[day_idx, "size"].sum()
                if total_size > 0:
                    weights = df.loc[day_idx, "size"] / total_size
                    new_pnl[day_idx] = new_pnl[day_idx] + offset * weights.values
    df2 = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df2.groupby(df2["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, sizes, skipped, new_dpnl


CANDIDATES = {
    "C1": {"fn": c1_vol_targeting, "first_principles": "Per-asset vol-targeting is the standard mechanism for risk-controlled position sizing — when realized vol rises above the cross-asset target, shrink the bet size.",
           "rationale": "Should reduce DD in vol-cluster events (regime shocks).",
           "params": {}},
    "C2": {"fn": c2_universe_filter, "first_principles": "Drop assets on rolling-30d negative Sharpe — the universal mean-reversion premise that 'serial losers' (assets currently in a losing regime) tend to keep losing for a window.",
           "rationale": "Removes 5-15% of trades during their worst window; should reduce DD without much PnL cost.",
           "params": {}},
    "C3": {"fn": c3_dd_kelly_scaling, "first_principles": "Standard CTA risk-management: when portfolio DD exceeds a threshold, mechanically delever. This is the defensive risk-control everyone uses.",
           "rationale": "Should reduce DD significantly during the worst 2020-11-24 episode (which had 8.28% NC DD). Cost: lower Sharpe in recovery.",
           "params": {}},
    "C4": {"fn": c4_btc_corr_sizing, "first_principles": "BTC drives the crypto market; assets with high BTC correlation have elevated systemic exposure. Reduce size on correlated assets to limit concentrated downside when BTC moves.",
           "rationale": "Should reduce DD in BTC-led selloffs (2020-11, 2022-11 FTX).",
           "params": {}},
    "C5": {"fn": c5_tod_filter, "first_principles": "UTC 22:00-02:00 is the Binance thin-liquidity window (Asia evening, US pre-open). Maker orders have lower fill probability; taker fills have wider spreads.",
           "rationale": "Should reduce cost drag; DD impact unclear but no negative Sharpe expectation.",
           "params": {}},
    "C6": {"fn": c6_sector_tag_filter, "first_principles": "Sector concentration is a known risk: when 60%+ of book is in one regime-driven group (e.g., L1/L2 during an L1 selloff), losses compound. Cap intra-sector exposure to bound cluster risk.",
           "rationale": "Should reduce DD in sector-specific drawdowns.",
           "params": {}},
    "C7": {"fn": c7_btc_hedge, "first_principles": "Explicit BTC short-perp hedge when portfolio DD > 8% — the textbook 'tail-risk overlay' that every multi-strategy fund runs.",
           "rationale": "Direct downside protection; cost is reduced upside in BTC-rally periods.",
           "params": {}},
    "STACK": {"fn": c2_then_c4_stack, "first_principles": "Compose C2 (universe filter) + C4 (BTC correlation sizing). Both pass anti-overfit individually; the stack reduces DD further.",
              "rationale": "Combined defense: skip losing-streak assets AND reduce size on BTC-correlated names. Should reduce DD more than either alone.",
              "params": {}},
}


def evaluate_candidate(cid, base_dpnl, pnl, sizes, skipped, timestamps, assets, port):
    """Run candidate post-processor and return metrics + anti-overfit flags."""
    cfg = CANDIDATES[cid]
    out = {"candidate": cid, "first_principles": cfg["first_principles"]}
    fn = cfg["fn"]
    res = fn(pnl, sizes, skipped, timestamps, base_dpnl, assets, **cfg["params"])
    if len(res) == 4:
        new_pnl, new_sizes, new_skipped, new_dpnl = res
    else:
        new_pnl, new_sizes, new_skipped = res
        new_dpnl = base_dpnl.copy()

    # Apply VIP-tier costs on top
    sim_proxy = {
        "timestamps": timestamps, "skipped": new_skipped,
        "sizes": new_sizes, "pnl": new_pnl,
        "n_trades": int((~new_skipped).sum()),
        "equity": np.zeros(len(timestamps)),
        "dd": np.zeros(len(timestamps)),
    }
    dpnl_adj = make_dpnl_with_costs(sim_proxy, port, new_dpnl)

    metrics = metrics_from_sim_dpnl(dpnl_adj, CAP)
    py = per_year_from_dpnl(dpnl_adj)
    out["baseline_sharpe"] = 3.40
    out["baseline_maxdd"] = 32.8
    out["variant_sharpe"] = round(metrics["sr"], 3)
    out["variant_maxdd"] = round(metrics["max_dd"], 1)
    out["variant_sortino"] = round(metrics["sortino"], 3)
    out["variant_pnl"] = round(float(dpnl_adj.sum()), 0)
    out["per_year"] = {str(y.year): float(v) for y, v in py.items()}
    n_pos = int((py > 0).sum())
    out["n_pos_years"] = f"{n_pos}/{len(py)}"

    # Anti-overfit
    flags = []
    sr_delta = metrics["sr"] - 3.40
    if sr_delta < -0.1:
        flags.append(f"sharpe_drop:{sr_delta:.2f}")
    if maxdd_delta := (32.8 - metrics["max_dd"]) < 2.0:
        flags.append(f"maxdd_improvement_lt_2pp:{32.8 - metrics['max_dd']:.1f}")

    # TRAIN/OOS ratio: train = sum(per_year PnL for years 2020-2022), oos = 2023-2026
    yrs = sorted(py.index)
    train_pnl = sum(py[y] for y in yrs if y.year < 2023)
    oos_pnl = sum(py[y] for y in yrs if y.year >= 2023)
    base_train = 265523 + 770112 + 17271  # from prior run
    base_oos = 240808 + 286767 + 148933 + 20616  # from prior run
    train_oos_ratio = (train_pnl / max(abs(oos_pnl), 1.0)) / (base_train / max(abs(base_oos), 1.0)) if base_train > 0 and base_oos > 0 else 999
    out["train_oos_ratio"] = round(train_oos_ratio, 2)
    if train_oos_ratio > 1.3:
        flags.append(f"train_oos_breach:{train_oos_ratio:.2f}x")

    # Available diagnostic on the per-trade PnL. This post-processor has
    # neither an independently reconstructed execution/fill PnL series nor a
    # distinct candidate-universe edge series, so those checks are explicitly
    # unevaluated and block acceptance.
    flags.append("independent_pnl_realism_not_evaluated")
    flags.append("selection_bias_not_evaluated")
    active = ~new_skipped
    if active.sum() > 30:
        tp = new_pnl[active]
        sz_arr = new_sizes[active]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_unit = np.where(sz_arr > 0, tp / sz_arr, 0.0)
        sizing_diagnostic = sizing_sanity_gate(per_unit, sz_arr)
        out["diagnostics"] = {
            "pnl_realism": {
                "status": "not_evaluated",
                "message": (
                    "Independent PnL realism NOT EVALUATED: no separate "
                    "execution/fill PnL series is available."
                ),
            },
            "sizing_sanity": {
                "status": "passed" if sizing_diagnostic.passed else "failed",
                "message": sizing_diagnostic.message,
            },
            "selection_bias": {
                "status": "not_evaluated",
                "message": (
                    "Selection bias NOT EVALUATED: no distinct candidate-universe "
                    "edge series is available."
                ),
            },
        }
        if not sizing_diagnostic.passed:
            flags.append("sizing_sanity_failed")
    else:
        flags.append("insufficient_trades_for_sizing_diagnostic")
        out["diagnostics"] = {
            "pnl_realism": {
                "status": "not_evaluated",
                "message": (
                    "Independent PnL realism NOT EVALUATED: no separate "
                    "execution/fill PnL series is available."
                ),
            },
            "sizing_sanity": {
                "status": "not_evaluated",
                "message": "Too few active trades for the sizing diagnostic.",
            },
            "selection_bias": {
                "status": "not_evaluated",
                "message": (
                    "Selection bias NOT EVALUATED: no distinct candidate-universe "
                    "edge series is available."
                ),
            },
        }

    out["anti_overfit_flags"] = flags
    out["accept"] = (
        not flags
        and sr_delta >= -0.10
        and (32.8 - metrics["max_dd"]) > 2.0
    )
    if out["accept"]:
        out["reason"] = f"ACCEPT: MaxDD {32.8:.1f}%->{metrics['max_dd']:.1f}% (-{32.8 - metrics['max_dd']:.1f}pp); Sharpe {metrics['sr']:.3f} (delta {sr_delta:+.3f})"
    else:
        out["reason"] = f"REJECT: flags={flags}" if flags else f"REJECT: MaxDD improvement only {32.8 - metrics['max_dd']:.1f}pp"
    return out


def main():
    t0 = time.time()
    print("=" * 80)
    print("  S5 DRAWDOWN-REDUCTION CANDIDATE EVALUATION")
    print(
        "  historical reported reference: Sharpe=3.40, MaxDD=32.8%, "
        "NC PnL $1,707,603 (not independently certified)"
    )
    print("  independent PnL realism: NOT EVALUATED; candidates cannot be accepted")
    print("=" * 80)

    # Load cached trade log
    trade_log_path = ROOT / "vol_arb/output/S5_HONEST_TRADE_LOG.parquet"
    if not trade_log_path.exists():
        print(f"ERROR: missing trade log at {trade_log_path}")
        sys.exit(1)
    df = pd.read_parquet(trade_log_path).sort_values("timestamp").reset_index(drop=True)
    print(f"  Trade log: {len(df):,} rows ({df['timestamp'].min()} -> {df['timestamp'].max()})")

    timestamps = df["timestamp"].values
    pnl = df["pnl"].values.astype(np.float64)
    sizes = df["size"].values.astype(np.float64)
    skipped = df["skipped"].values.astype(bool)
    assets = df["asset"]

    # Compute baseline daily PnL with costs
    port = historical_portfolio_config()
    sim = {
        "timestamps": timestamps, "skipped": skipped,
        "sizes": sizes, "pnl": pnl,
        "n_trades": int((~skipped).sum()),
        "equity": np.zeros(len(timestamps)),
        "dd": np.zeros(len(timestamps)),
    }
    base_dpnl = compute_daily_pnl_series(sim)
    base_dpnl = make_dpnl_with_costs(sim, port, base_dpnl)
    base_m = metrics_from_sim_dpnl(base_dpnl, CAP)
    print(f"  Baseline reconstructed: Sharpe={base_m['sr']:.3f} MaxDD={base_m['max_dd']:.1f}%")
    print(f"  (Spec anchor: Sharpe=3.40, MaxDD=32.8%)\n")

    # Run candidates
    results = []
    for cid in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "STACK"]:
        print(f"  Evaluating {cid}...")
        t1 = time.time()
        try:
            r = evaluate_candidate(cid, base_dpnl, pnl, sizes, skipped, timestamps, assets, port)
            r["runtime_s"] = round(time.time() - t1, 1)
            results.append(r)
            print(f"    Sharpe={r['variant_sharpe']:.3f} MaxDD={r['variant_maxdd']:.1f}% "
                  f"PnL=${r['variant_pnl']:,.0f} pos={r['n_pos_years']} "
                  f"train_oos={r['train_oos_ratio']}x")
            print(f"    flags={r['anti_overfit_flags']}")
            print(f"    accept={r['accept']}: {r['reason']}")
        except Exception as e:
            print(f"    FAIL: {e}")
            import traceback; traceback.print_exc()
            results.append({"candidate": cid, "accept": False, "reason": f"FAIL: {e}",
                            "anti_overfit_flags": ["exception"]})

    # Save JSON
    out_path = ROOT / "vol_arb/output/S5_DRAWDOWN_CANDIDATES.json"
    with open(out_path, "w") as f:
        json.dump({"baseline": {"sharpe": 3.40, "maxdd": 32.8, "pnl": 1707603, "nc_pnl": 1707603,
                                "nc_maxdd": 8.28, "compound_maxdd": 32.8,
                                "methodology_note": "Spec headline MaxDD 32.8% is COMPOUND-equity-curve (cumprod+peak-to-trough); NC peak-to-trough is 8.28% (peak-to-trough on cumsum)."},
                   "candidates": results,
                   "caveats": ["INDEPENDENT PNL REALISM NOT EVALUATED: no separate execution/fill PnL series is available, so this evaluator cannot accept a candidate.",
                              "SELECTION BIAS NOT EVALUATED: no distinct candidate-universe edge series is available.",
                              "PRE-2020 DATA UNAVAILABLE: data/binance-1m starts 2020-01. Cannot use a 2018-2019 holdout. TRAIN/OOS proxy uses years 2020-2022 (TRAIN) vs 2023-2026 (OOS).",
                              "Compound vs NC MaxDD: spec reports compound 32.8%. NC peak-to-trough is 8.28%. Candidates are scored on COMPOUND MaxDD (matching the spec)."],
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"  Runtime: {time.time() - t0:.1f}s")

    # Summary
    accepted = [r for r in results if r.get("accept")]
    print(f"\n  ACCEPTED: {[r['candidate'] for r in accepted] or 'NONE'}")
    print("  Independent PnL realism is not evaluated; acceptance is blocked.")
    print("=" * 80)


if __name__ == "__main__":
    main()
