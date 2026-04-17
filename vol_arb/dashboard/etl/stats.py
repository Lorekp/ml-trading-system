"""Heavy stats for the backtest report — bootstrap, permutation, walk-forward, rolling."""
from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple


def returns_from_eq(eq: np.ndarray) -> np.ndarray:
    eq = np.asarray(eq, dtype=float)
    base = eq[:-1]
    base = np.where(base == 0, np.nan, base)
    r = np.diff(eq) / base
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def bars_per_year(bar_hours: float) -> float:
    return 365.0 * 24.0 / bar_hours


def headline_stats(eq: np.ndarray, bar_hours: float, rf_annual: float = 0.045) -> Dict:
    eq = np.asarray(eq, dtype=float)
    if len(eq) < 3 or eq[0] <= 0:
        return _empty_stats()
    r = returns_from_eq(eq)
    bpy = bars_per_year(bar_hours)
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    ann_vol = sig * math.sqrt(bpy)
    cagr = (1 + mu) ** bpy - 1 if abs(mu) < 1 else float('nan')
    sharpe = (mu * bpy - rf_annual) / ann_vol if ann_vol > 0 else 0.0
    down = r[r < 0]
    down_vol = math.sqrt(np.mean(down ** 2)) * math.sqrt(bpy) if len(down) else 0.0
    sortino = (mu * bpy - rf_annual) / down_vol if down_vol > 0 else 0.0
    # drawdown
    peak = eq[0]
    dd = 0.0
    cur_dur = 0
    max_dur = 0
    for v in eq:
        if v > peak:
            peak = v
            cur_dur = 0
        else:
            cur_dur += 1
            cd = v / peak - 1
            if cd < dd:
                dd = cd
        if cur_dur > max_dur:
            max_dur = cur_dur
    calmar = cagr / abs(dd) if dd < 0 else 0.0
    hit = float(np.sum(r > 0) / len(r)) if len(r) else 0.0
    sharpe_ci = bootstrap_sharpe_ci(r, bar_hours, rf_annual)
    return dict(
        cagr=cagr, vol=ann_vol, sharpe=sharpe, sortino=sortino, calmar=calmar,
        maxdd=dd, maxdd_dur_days=max_dur * bar_hours / 24.0,
        hit=hit, n_obs=len(r),
        sharpe_ci=list(sharpe_ci),
        total_ret=float(eq[-1] / eq[0] - 1),
    )


def _empty_stats() -> Dict:
    return dict(cagr=0.0, vol=0.0, sharpe=0.0, sortino=0.0, calmar=0.0,
                maxdd=0.0, maxdd_dur_days=0.0, hit=0.0, n_obs=0,
                sharpe_ci=[0.0, 0.0], total_ret=0.0)


def bootstrap_sharpe_ci(r: np.ndarray, bar_hours: float, rf_annual: float = 0.045,
                         B: int = 400, block: int = 24, seed: int = 20260417) -> Tuple[float, float]:
    samples = bootstrap_sharpe(r, bar_hours, rf_annual, B=B, block=block, seed=seed)
    samples_sorted = np.sort(samples)
    lo = float(samples_sorted[int(0.025 * B)])
    hi = float(samples_sorted[int(0.975 * B)])
    return lo, hi


def bootstrap_sharpe(r: np.ndarray, bar_hours: float, rf_annual: float = 0.045,
                      B: int = 800, block: int = 24, seed: int = 20260417) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(r)
    bpy = bars_per_year(bar_hours)
    out = np.empty(B)
    if n <= block:
        return np.full(B, 0.0)
    for b in range(B):
        sample = []
        k = 0
        while k < n:
            st = rng.integers(0, n - block)
            sample.append(r[st:st + block])
            k += block
        s = np.concatenate(sample)[:n]
        m = s.mean()
        sd = s.std(ddof=1) if len(s) > 1 else 0.0
        out[b] = (m * bpy - rf_annual) / (sd * math.sqrt(bpy)) if sd > 0 else 0.0
    return np.sort(out)


def permutation_null(r: np.ndarray, bar_hours: float,
                     B: int = 800, seed: int = 20260417) -> np.ndarray:
    """Sign-randomization null: distribution of *raw* (rf-unadjusted) Sharpe
    under sign(r)~Bernoulli. Under H0 the mean is 0 by symmetry, so the null
    is centered at 0 — compare against the observed raw Sharpe, not the
    rf-adjusted Sharpe shown on the Overview tab.
    """
    rng = np.random.default_rng(seed + 1)
    bpy = bars_per_year(bar_hours)
    n = len(r)
    if n == 0:
        return np.zeros(B)
    abs_r = np.abs(r)
    out = np.empty(B)
    for b in range(B):
        signs = rng.choice([-1.0, 1.0], size=n)
        x = signs * abs_r
        m = x.mean()
        v = x.var(ddof=1) if len(x) > 1 else 0.0
        sd = math.sqrt(v)
        out[b] = (m * bpy) / (sd * math.sqrt(bpy)) if sd > 0 else 0.0
    return np.sort(out)


def raw_sharpe(r: np.ndarray, bar_hours: float) -> float:
    """Annualized Sharpe without rf adjustment — matches the scale of permutation_null."""
    if len(r) < 2:
        return 0.0
    bpy = bars_per_year(bar_hours)
    m = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return 0.0
    return (m * bpy) / (sd * math.sqrt(bpy))


def walk_forward(eq: np.ndarray, ts: np.ndarray, bar_hours: float, k: int = 8) -> List[Dict]:
    n = len(eq)
    folds = []
    for i in range(k):
        a = i * n // k
        b = (i + 1) * n // k
        slice_eq = eq[a:b]
        s = headline_stats(slice_eq, bar_hours)
        folds.append(dict(
            i=i + 1,
            ts_a=str(ts[a])[:10],
            ts_b=str(ts[max(a, b - 1)])[:10],
            sharpe=float(s['sharpe']),
            cagr=float(s['cagr']),
            dd=float(s['maxdd']),
            is_=i < k // 2,
            oos=i >= k // 2,
        ))
    return folds


def walk_forward_summary(eq: np.ndarray, bar_hours: float, k: int = 8,
                          rf_annual: float = 0.045) -> Dict:
    """Summaries that avoid the Jensen artifact of averaging per-fold Sharpes.

    - mean_fold_sharpe: simple mean of per-fold Sharpes (what the chart shows)
    - full_oos_sharpe:  Sharpe on the *concatenated* returns of OOS folds (i>=k//2).
      Averaging Sharpes gives a biased-up estimate when fold volatilities differ;
      this is the canonical OOS number.
    """
    n = len(eq)
    r = returns_from_eq(eq)
    bpy = bars_per_year(bar_hours)
    fold_sharpes = []
    oos_segments = []
    for i in range(k):
        a = i * n // k
        b = (i + 1) * n // k
        slice_eq = eq[a:b]
        s = headline_stats(slice_eq, bar_hours, rf_annual)
        fold_sharpes.append(float(s['sharpe']))
        if i >= k // 2:
            # Map eq-slice indices to return indices (r = diff of eq, len n-1)
            ra, rb = max(a - 1, 0), max(b - 1, 0)
            if rb > ra:
                oos_segments.append(r[ra:rb])
    if oos_segments:
        concat = np.concatenate(oos_segments)
        if len(concat) > 1 and np.std(concat, ddof=1) > 0:
            m = float(np.mean(concat))
            sd = float(np.std(concat, ddof=1))
            full_oos = (m * bpy - rf_annual) / (sd * math.sqrt(bpy))
        else:
            full_oos = 0.0
    else:
        full_oos = 0.0
    oos_folds = [s for s in fold_sharpes[k // 2:]]
    return dict(
        mean_fold_sharpe=float(np.mean(fold_sharpes)),
        mean_oos_fold_sharpe=float(np.mean(oos_folds)) if oos_folds else 0.0,
        full_oos_sharpe=float(full_oos),
    )


def rolling_sharpe(r: np.ndarray, bar_hours: float, window_bars: int) -> List[float | None]:
    n = len(r)
    bpy = bars_per_year(bar_hours)
    out: List[float | None] = [None] * n
    if n < window_bars:
        return out
    # Vectorized rolling mean/std via cumulative sums
    cs = np.cumsum(np.insert(r, 0, 0.0))
    cs2 = np.cumsum(np.insert(r * r, 0, 0.0))
    for i in range(window_bars, n):
        s = cs[i] - cs[i - window_bars]
        s2 = cs2[i] - cs2[i - window_bars]
        m = s / window_bars
        v = max(0.0, s2 / window_bars - m * m)
        sd = math.sqrt(v)
        out[i] = (m * bpy) / (sd * math.sqrt(bpy)) if sd > 0 else 0.0
    return out


def rolling_ic(edge: np.ndarray, fwd_ret: np.ndarray, window_bars: int, step: int = 12) -> List[Dict]:
    n = min(len(edge), len(fwd_ret))
    out: List[Dict] = []
    if n < window_bars:
        return out
    abs_edge = np.abs(edge[:n])
    fr = fwd_ret[:n]
    for i in range(window_bars, n, step):
        a = abs_edge[i - window_bars:i]
        b = fr[i - window_bars:i]
        ma, mb = a.mean(), b.mean()
        num = float(np.sum((a - ma) * (b - mb)))
        da = float(np.sum((a - ma) ** 2))
        db = float(np.sum((b - mb) ** 2))
        denom = math.sqrt(da * db) if da * db > 0 else 1.0
        out.append(dict(i=int(i), ic=num / denom))
    return out


def drawdown_series(eq: np.ndarray) -> List[float]:
    out = np.empty(len(eq))
    peak = eq[0]
    for i, v in enumerate(eq):
        if v > peak:
            peak = v
        out[i] = float(v / peak - 1.0) if peak > 0 else 0.0
    return out.tolist()


def decile_table(episodes: List[Dict]) -> List[Dict]:
    if not episodes:
        return []
    pairs = [(e['edge_entry'], e['net_pnl'] / max(1, e['N_contracts'])) for e in episodes]
    pairs.sort(key=lambda p: p[0])
    n = len(pairs)
    out = []
    for d in range(10):
        lo = d * n // 10
        hi = (d + 1) * n // 10
        sl = pairs[lo:hi]
        if not sl:
            out.append(dict(d=d + 1, mean=0.0, n=0, se=0.0, edge_lo=0.0, edge_hi=0.0))
            continue
        rs = np.array([p[1] for p in sl])
        m = float(rs.mean())
        sd = float(rs.std(ddof=1)) if len(rs) > 1 else 0.0
        out.append(dict(
            d=d + 1, mean=m, n=len(sl),
            se=sd / math.sqrt(len(sl)) if len(sl) > 0 else 0.0,
            edge_lo=float(sl[0][0]), edge_hi=float(sl[-1][0]),
        ))
    return out


def monthly_returns(eq: np.ndarray, ts: np.ndarray) -> List[Dict]:
    """Calendar-month compound returns from per-bar equity."""
    if len(eq) < 2:
        return []
    import pandas as pd
    s = pd.Series(eq, index=pd.DatetimeIndex(ts))
    monthly = s.resample('MS').last().ffill()
    rets = monthly.pct_change().dropna()
    return [dict(k=ts.strftime('%Y-%m'), r=float(r)) for ts, r in rets.items()]


def cost_sensitivity(per_bar_pnl_no_costs: np.ndarray, per_bar_costs: np.ndarray,
                     edge: np.ndarray, signal: np.ndarray, bar_hours: float,
                     E0: float = 100_000.0,
                     thresholds: List[float] = [3, 5, 7, 9, 12, 15],
                     cost_mults: List[float] = [0, 0.5, 1, 1.5, 2, 3, 5]) -> List[Dict]:
    """Recompute Sharpe across (threshold, cost_mult) grid.

    For each threshold T: keep bars where active signal corresponds to |edge|>=T at trade entry,
    approximated here by masking PnL on bars where |edge_at_signal|<T. Costs scaled by mult.
    """
    bpy = bars_per_year(bar_hours)
    out = []
    abs_edge = np.abs(edge)
    active = signal != 0
    for thr in thresholds:
        # Mask bars where signal active but |edge| < thr  -> drop those PnL contributions
        keep = (~active) | (abs_edge >= thr)
        base_pnl = per_bar_pnl_no_costs * keep
        base_cost = per_bar_costs * keep
        for mult in cost_mults:
            net = base_pnl + base_cost * mult
            eq = E0 + np.cumsum(net)
            r = returns_from_eq(eq)
            if len(r) < 2 or r.std() == 0:
                sharpe = 0.0
            else:
                sharpe = float((r.mean() * bpy - 0.045) / (r.std(ddof=1) * math.sqrt(bpy)))
            out.append(dict(thr=float(thr), cost=float(mult), sharpe=sharpe))
    return out


# Hardcoded named events -> compute PnL on a ±5 day window around each.
STRESS_EVENT_DEFS = [
    ("Mar-23 banking",   "2023-03-13"),
    ("Aug-23 liq flush", "2023-08-17"),
    ("Oct-23 rally",     "2023-10-24"),
    ("Jan-24 ETF",       "2024-01-11"),
    ("Apr-24 halving",   "2024-04-20"),
    ("Aug-24 yen carry", "2024-08-05"),
    ("Nov-24 election",  "2024-11-06"),
    ("Feb-25 flash",     "2025-02-28"),
    ("Jul-25 squeeze",   "2025-07-14"),
    ("Apr-26 chop",      "2026-04-08"),
]


def stress_events(eq: np.ndarray, bench: np.ndarray, ts: np.ndarray) -> List[Dict]:
    import pandas as pd
    eq_s = pd.Series(eq, index=pd.DatetimeIndex(ts))
    bench_s = pd.Series(bench, index=pd.DatetimeIndex(ts))
    out = []
    for name, date in STRESS_EVENT_DEFS:
        d0 = pd.Timestamp(date, tz='UTC')
        a = d0 - pd.Timedelta(days=2)
        b = d0 + pd.Timedelta(days=5)
        eq_window = eq_s.loc[a:b]
        bench_window = bench_s.loc[a:b]
        if len(eq_window) < 2 or len(bench_window) < 2:
            continue
        pnl = float(eq_window.iloc[-1] / eq_window.iloc[0] - 1)
        bm = float(bench_window.iloc[-1] / bench_window.iloc[0] - 1)
        out.append(dict(name=name, date=date, pnl=pnl, bench=bm))
    return out
