"""Dynamic asset selection — multi-factor scoring with diversification.

Literature-grounded universe construction:
- Multi-factor score: volume + spread + stability (Harris Ch. 19, Johnson Ch. 10)
- Correlation-based diversification filter (ML for AM Ch. 2-4)
- Data quality gate: history, stationarity, volume consistency (AFML Ch. 2)
- Rebalance smoothing: buffer zone + position dampening (Harris Ch. 23)

For backtesting: computes the eligible asset set per month using only data
available at the start of that month.

For live trading: recomputes monthly on the 1st, using the prior 30 days.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# ============================================================
# Core metrics
# ============================================================

def compute_monthly_dollar_volume(ohlcv_15m: pd.DataFrame) -> pd.Series:
    """30-day rolling MEDIAN daily dollar volume (robust to spikes)."""
    dollar_vol = (ohlcv_15m["close"] * ohlcv_15m["volume"]).resample("1D").sum()
    return dollar_vol.rolling(30, min_periods=20).median()


def compute_rolling_spread(ohlcv_15m: pd.DataFrame, window: int = 30) -> pd.Series:
    """30-day rolling median Corwin-Schultz spread from H/L data.

    Lower = tighter spread = better liquidity.
    Ref: AFML Ch. 19, Corwin & Schultz (2012).
    """
    high = ohlcv_15m["high"]
    low = ohlcv_15m["low"]

    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2

    # Two-bar high-low range
    high_2 = high.rolling(2).max()
    low_2 = low.rolling(2).min()
    gamma = np.log(high_2 / low_2) ** 2

    beta = log_hl_sq.rolling(2).sum()
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    alpha = alpha.clip(lower=0)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Daily median, then 30-day rolling median
    daily_spread = spread.resample("1D").median()
    return daily_spread.rolling(window, min_periods=window // 2).median()


def compute_volume_stability(ohlcv_15m: pd.DataFrame, window: int = 30) -> pd.Series:
    """Volume consistency: 30d_median / 30d_mean. Closer to 1 = more stable.

    Penalizes assets with spike-driven volume (mean >> median).
    """
    dollar_vol = (ohlcv_15m["close"] * ohlcv_15m["volume"]).resample("1D").sum()
    rolling_median = dollar_vol.rolling(window, min_periods=window // 2).median()
    rolling_mean = dollar_vol.rolling(window, min_periods=window // 2).mean()
    stability = rolling_median / rolling_mean.replace(0, np.nan)
    return stability.clip(0, 1)


# ============================================================
# Asset scoring
# ============================================================

def compute_asset_score(
    ohlcv_15m: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Compute all scoring components for one asset.

    Returns dict with keys: volume, spread, stability (all daily Series).
    """
    return {
        "volume": compute_monthly_dollar_volume(ohlcv_15m),
        "spread": compute_rolling_spread(ohlcv_15m),
        "stability": compute_volume_stability(ohlcv_15m),
    }


# ============================================================
# Data quality validation
# ============================================================

def validate_asset(
    ohlcv_15m: pd.DataFrame,
    lookback_end: pd.Timestamp,
    min_months: int = 17,
    min_volume: float = 20_000_000,
) -> tuple[bool, str]:
    """Validate asset data quality before selection.

    Checks: history length, volume consistency, stationarity, halted status.

    Returns (pass, reason).
    """
    # 1. Min history
    available = ohlcv_15m.loc[:lookback_end]
    n_days = (available.index[-1] - available.index[0]).days if len(available) > 0 else 0
    if n_days < min_months * 30:
        return False, f"too_short ({n_days}d < {min_months * 30}d)"

    # 2. Not halted (latest bar within 2 days of lookback_end)
    if len(available) > 0:
        gap = (lookback_end - available.index[-1]).days
        if gap > 2:
            return False, f"halted (last bar {gap}d ago)"

    # 3. Volume consistency — no month in last 3 with median < $5M
    dollar_vol = (available["close"] * available["volume"]).resample("1D").sum()
    if len(dollar_vol) >= 60:
        monthly_median = dollar_vol.iloc[-90:].resample("ME").median()
        if (monthly_median < 5_000_000).any():
            return False, "volume_collapse (month < $5M)"

    # 4. Return stationarity (ADF test on 90d returns)
    if len(available) >= 90 * 96:  # 90 days of 15min bars
        returns = available["close"].pct_change().dropna().iloc[-90 * 96:]
        try:
            adf_stat, adf_pval, *_ = adfuller(returns, maxlag=10, regression="c", autolag="AIC")
            if adf_pval > 0.05:
                return False, f"non_stationary (ADF p={adf_pval:.3f})"
        except Exception:
            pass  # skip if ADF fails

    return True, "ok"


# ============================================================
# Simple monthly asset selection (for experiments)
# ============================================================

def select_top_n_for_month(
    asset_volumes: dict[str, pd.Series],
    asset_first_dates: dict[str, pd.Timestamp],
    month_start: pd.Timestamp,
    top_n: int = 30,
    min_history_days: int = 180,
    spike_ratio: float = 5.0,
    max_vol_cv: float = 0.0,
) -> list[str]:
    """Select top-N assets by volume with age and stability filters.

    This is the canonical selection function — all experiments and live
    trading should use this instead of inline selection logic.

    Filters (applied in order):
        1. Minimum history: asset must have existed for min_history_days
           at the time of selection (not at backtest time)
        2. Volume available: at least 45 days of volume data
        3. Volume positive: latest 30d median > 0
        4. Spike filter: 30d median / 90d median < spike_ratio
        5. Volume stability (optional): coefficient of variation < max_vol_cv
        6. Rank by 30d median dollar volume, take top_n

    Args:
        asset_volumes: {symbol: pd.Series of 30d rolling median dollar volume}
        asset_first_dates: {symbol: first available timestamp}
        month_start: Start of the selection month.
        top_n: Number of assets to select.
        min_history_days: Minimum days since asset first appeared.
        spike_ratio: Max allowed 30d/90d volume ratio.
        max_vol_cv: Max volume coefficient of variation (0 = disabled).

    Returns:
        List of selected symbols, sorted by volume descending.
    """
    lookback_end = month_start - pd.Timedelta(days=1)
    candidates = {}

    for symbol, vol_s in asset_volumes.items():
        # 1. Age filter
        first_date = asset_first_dates.get(symbol)
        if first_date is not None:
            # Normalize tz
            ms = month_start
            fd = first_date
            if hasattr(fd, "tz") and fd.tz is not None and ms.tz is None:
                ms = ms.tz_localize(fd.tz)
            elif hasattr(ms, "tz") and ms.tz is not None and (not hasattr(fd, "tz") or fd.tz is None):
                fd = fd.tz_localize(ms.tz) if hasattr(fd, "tz_localize") else fd
            age_days = (ms - fd).days if ms > fd else 0
            if age_days < min_history_days:
                continue

        # 2. Volume data available
        avail = vol_s.loc[:lookback_end]
        if len(avail) < 45:
            continue

        # 3. Latest volume positive
        latest = avail.iloc[-1]
        if np.isnan(latest) or latest <= 0:
            continue

        # 4. Spike filter
        if len(avail) >= 90:
            m90 = avail.iloc[-90:].median()
            if m90 > 0 and latest / m90 > spike_ratio:
                continue

        # 5. Volume stability (CV filter)
        if max_vol_cv > 0 and len(avail) >= 30:
            recent = avail.iloc[-30:]
            mean_v = recent.mean()
            if mean_v > 0:
                cv = recent.std() / mean_v
                if cv > max_vol_cv:
                    continue

        candidates[symbol] = latest

    # 6. Rank and select top-N
    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    return [s for s, _ in ranked[:top_n]]


# ============================================================
# Diversification filter
# ============================================================

def diversification_filter(
    ranked_candidates: list[str],
    daily_returns: dict[str, pd.Series],
    max_corr: float = 0.85,
    target_n: int = 30,
) -> list[str]:
    """Greedy diversification: reject candidates too correlated with already-selected.

    Ref: ML for Asset Managers Ch. 2-4 (simplified — Pearson, no denoising).

    Args:
        ranked_candidates: Assets sorted by composite score (best first).
        daily_returns: Dict of asset -> daily return Series.
        max_corr: Maximum allowed correlation with any selected asset.
        target_n: Target number of assets.

    Returns:
        Filtered list of assets (may be < target_n if correlation too high).
    """
    selected = []
    for candidate in ranked_candidates:
        if candidate not in daily_returns:
            continue
        cand_ret = daily_returns[candidate]
        if len(cand_ret) < 30:
            continue

        # Check correlation with all already-selected
        too_correlated = False
        for sel in selected:
            if sel not in daily_returns:
                continue
            sel_ret = daily_returns[sel]
            # Align indices
            common = cand_ret.index.intersection(sel_ret.index)
            if len(common) < 30:
                continue
            corr = cand_ret.loc[common].corr(sel_ret.loc[common])
            if abs(corr) > max_corr:
                too_correlated = True
                logger.debug(f"  {candidate} rejected: corr={corr:.3f} with {sel}")
                break

        if not too_correlated:
            selected.append(candidate)

        if len(selected) >= target_n:
            break

    return selected


# ============================================================
# Main selection function
# ============================================================

def select_assets_for_month(
    asset_scores: dict[str, dict[str, pd.Series]],
    month_start: pd.Timestamp,
    top_n: int = 30,
    min_history_days: int = 90,
    previous_assets: list[str] | None = None,
    asset_ohlcv: dict[str, pd.DataFrame] | None = None,
    daily_returns: dict[str, pd.Series] | None = None,
    w_vol: float = 0.5,
    w_spread: float = 0.3,
    w_stability: float = 0.2,
    entry_n: int | None = None,
    exit_n: int | None = None,
    max_corr: float = 0.85,
) -> tuple[list[str], dict[str, float]]:
    """Select top-N assets by composite score with diversification.

    Args:
        asset_scores: Dict of asset -> {volume, spread, stability} Series.
        month_start: First day of the month.
        top_n: Target number of assets.
        min_history_days: Minimum days of data required.
        previous_assets: Assets selected last month (for buffer logic).
        asset_ohlcv: Dict of asset -> OHLCV DataFrame (for validation).
        daily_returns: Dict of asset -> daily return Series (for diversification).
        w_vol, w_spread, w_stability: Scoring weights.
        entry_n: Score rank required to enter (default: top_n + 5).
        exit_n: Score rank at which asset exits (default: top_n + 15).
        max_corr: Max correlation for diversification filter.

    Returns:
        (selected_assets, rebalance_weights) where weights are 0.5 for
        first/last month, 1.0 otherwise.
    """
    if entry_n is None:
        entry_n = top_n + 5
    if exit_n is None:
        exit_n = top_n + 15

    lookback_end = month_start - pd.Timedelta(days=1)
    prev_set = set(previous_assets) if previous_assets else set()

    # Compute scores at lookback_end
    raw_scores = {}
    for asset, scores in asset_scores.items():
        vol_s = scores.get("volume")
        if vol_s is None:
            continue
        available = vol_s.loc[:lookback_end]
        if len(available) < min_history_days // 2:
            continue

        vol_val = available.iloc[-1]
        if np.isnan(vol_val) or vol_val <= 0:
            continue

        # Spike check (30d vs 90d)
        if len(available) >= 90:
            median_90d = available.iloc[-90:].median()
            if median_90d > 0 and vol_val / median_90d > 5.0:
                continue

        # Spread score (lower = better, invert for ranking)
        spread_s = scores.get("spread")
        spread_val = 0.01  # default
        if spread_s is not None:
            s_avail = spread_s.loc[:lookback_end].dropna()
            if len(s_avail) > 10:
                spread_val = max(s_avail.iloc[-1], 1e-6)

        # Stability score
        stab_s = scores.get("stability")
        stab_val = 0.5
        if stab_s is not None:
            st_avail = stab_s.loc[:lookback_end].dropna()
            if len(st_avail) > 10:
                stab_val = st_avail.iloc[-1]

        # Data quality gate
        if asset_ohlcv and asset in asset_ohlcv:
            passed, reason = validate_asset(asset_ohlcv[asset], lookback_end)
            if not passed:
                logger.debug(f"  {asset}: validation failed ({reason})")
                continue

        raw_scores[asset] = {
            "volume": vol_val,
            "spread": spread_val,
            "stability": stab_val,
        }

    if not raw_scores:
        return [], {}

    # Rank each factor (percentile rank, 0-1)
    assets = list(raw_scores.keys())
    vol_vals = np.array([raw_scores[a]["volume"] for a in assets])
    spread_vals = np.array([raw_scores[a]["spread"] for a in assets])
    stab_vals = np.array([raw_scores[a]["stability"] for a in assets])

    n = len(assets)
    vol_rank = np.argsort(np.argsort(-vol_vals)) / max(n - 1, 1)  # higher = better
    spread_rank = np.argsort(np.argsort(spread_vals)) / max(n - 1, 1)  # lower = better
    stab_rank = np.argsort(np.argsort(-stab_vals)) / max(n - 1, 1)  # higher = better

    composite = w_vol * (1 - vol_rank) + w_spread * (1 - spread_rank) + w_stability * (1 - stab_rank)

    # Sort by composite (lower rank = better)
    order = np.argsort(composite)
    ranked = [assets[i] for i in order]

    # Buffer zone logic
    if prev_set:
        # Assets already in universe: only exit if rank > exit_n
        keep = [a for a in ranked[:exit_n] if a in prev_set]
        # New assets: only enter if rank <= entry_n
        new = [a for a in ranked[:entry_n] if a not in prev_set]
        # Combine: existing first (preserve), then new
        pool = keep + [a for a in new if a not in keep]
    else:
        pool = ranked[:entry_n]

    # Diversification filter
    if daily_returns:
        pool = diversification_filter(pool, daily_returns, max_corr=max_corr, target_n=top_n)
    else:
        pool = pool[:top_n]

    # Rebalance weights: 0.5 for first/last month in universe
    weights = {}
    for a in pool:
        if a not in prev_set:
            weights[a] = 0.5  # new entry → half size first month
        else:
            weights[a] = 1.0
    # Assets leaving: not in pool but were in prev_set → handled by caller

    return pool, weights


# ============================================================
# Schedule builder
# ============================================================

def build_monthly_asset_schedule(
    asset_scores: dict[str, dict[str, pd.Series]],
    start_month: str,
    end_month: str | None = None,
    top_n: int = 30,
    min_history_days: int = 90,
    asset_ohlcv: dict[str, pd.DataFrame] | None = None,
    daily_returns: dict[str, pd.Series] | None = None,
    max_corr: float = 0.85,
) -> dict[str, dict]:
    """Build the full schedule with buffer logic and rebalance weights.

    Returns dict of month -> {assets: [...], weights: {asset: float}, added: [...], removed: [...]}.
    """
    all_dates = set()
    for scores in asset_scores.values():
        vol_s = scores.get("volume")
        if vol_s is not None:
            all_dates.update(vol_s.index)
    if not all_dates:
        return {}

    start = pd.Timestamp(start_month + "-01")
    end = pd.Timestamp(end_month + "-01") + pd.offsets.MonthEnd(0) if end_month else max(all_dates)

    months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")

    schedule = {}
    prev_assets = []

    for month in months:
        ms = pd.Timestamp(month.start_time)
        # Timezone alignment
        sample_vol = next(iter(asset_scores.values())).get("volume")
        if sample_vol is not None and sample_vol.index.tz is not None:
            ms = ms.tz_localize(sample_vol.index.tz)

        selected, weights = select_assets_for_month(
            asset_scores, ms, top_n=top_n, min_history_days=min_history_days,
            previous_assets=prev_assets, asset_ohlcv=asset_ohlcv,
            daily_returns=daily_returns, max_corr=max_corr,
        )

        prev_set = set(prev_assets)
        cur_set = set(selected)
        schedule[str(month)] = {
            "assets": selected,
            "weights": weights,
            "added": sorted(cur_set - prev_set),
            "removed": sorted(prev_set - cur_set),
        }
        prev_assets = selected

    return schedule


# ============================================================
# Reporting
# ============================================================

def report_asset_schedule(
    schedule: dict[str, dict],
    all_assets: dict[str, str] | None = None,
) -> None:
    """Print the asset schedule with changes and rebalance weights."""
    all_selected = set()
    for month, info in sorted(schedule.items()):
        assets = info["assets"] if isinstance(info, dict) else info
        all_selected.update(assets)

    sample = next(iter(schedule.values()))
    n_per = len(sample["assets"]) if isinstance(sample, dict) else len(sample)
    print(f"\nAsset Schedule ({len(schedule)} months, target {n_per} per month)")
    print(f"  Total unique assets: {len(all_selected)}")
    print()

    for month, info in sorted(schedule.items()):
        if isinstance(info, dict):
            assets = info["assets"]
            added = info.get("added", [])
            removed = info.get("removed", [])
            changes = ""
            if added:
                changes += f" +{','.join(added)}"
            if removed:
                changes += f" -{','.join(removed)}"
            print(f"  {month}: {len(assets)} assets{changes}")
        else:
            print(f"  {month}: {len(info)} assets")

    if all_assets:
        missing = all_selected - set(all_assets.keys())
        if missing:
            print(f"\n  MISSING DATA: {', '.join(sorted(missing))}")
        else:
            print(f"\n  All {len(all_selected)} assets have data.")
