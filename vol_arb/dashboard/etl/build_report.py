"""Build report.json for the BTC vol-arb dashboard from a per-bar trade log parquet.

Usage:
  python etl/build_report.py --in ../output/BTC_v2_trade_log.parquet --out public/report.json
"""
from __future__ import annotations
import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from stats import (
    headline_stats, bootstrap_sharpe, permutation_null, walk_forward,
    rolling_sharpe, rolling_ic, drawdown_series, decile_table,
    monthly_returns, cost_sensitivity, stress_events, returns_from_eq,
    walk_forward_summary, raw_sharpe,
)

# Columns we actually use from the 128-col parquet.
USE_COLS = [
    'ts', 'index_price',
    'iv', 'tsrv', 'rv_pred_pct', 'edge', 'signal',
    # comp PnL legs
    'vega_pnl', 'theta_pnl', 'delta_hedge_pnl',
    'hedge_costs', 'option_fees', 'slippage',
    'gross_pnl', 'net_pnl', 'equity', 'N_contracts',
    # NC PnL legs
    'vega_pnl_nc', 'theta_pnl_nc', 'delta_hedge_pnl_nc',
    'hedge_costs_nc', 'option_fees_nc', 'slippage_nc',
    'gross_pnl_nc', 'net_pnl_nc', 'equity_nc', 'N_contracts_nc',
    'exit_target',
]

E0 = 100_000.0

# Feature-maturity boundary, not a leakage boundary.
# The model is genuine walk-forward from 2022-02 onwards (each month uses only prior
# data — see vol_model/run_walkforward_pruned_svi.py + extract_pruned_svi_predictions.py),
# so there is no IS/OOS leak. However, OFI features are zero-filled before 2023-04
# (see vol_model/pipeline.py:104-111), so pre-2023-04 predictions run on a degraded
# feature set. 2023-04-01 is the cleanest "representative-quality" cut.
WALK_FORWARD_OOS_START = '2023-04-01'


def load_parquet(path: Path) -> pd.DataFrame:
    print(f"[etl] reading {path}", flush=True)
    t0 = time.time()
    df = pq.read_table(str(path), columns=USE_COLS).to_pandas()
    df = df.sort_values('ts').reset_index(drop=True)
    # Naive (UTC) timestamps for downstream resampling
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    print(f"[etl]   {len(df):,} rows, {len(df.columns)} cols, {time.time()-t0:.2f}s", flush=True)
    return df


def derive_episodes(df: pd.DataFrame) -> List[Dict]:
    """Walk the per-bar trade log and group consecutive same-side trades into episodes.

    Entry: prev signal == 0 (or different sign) and current signal != 0 with N_contracts > 0.
    Exit:  signal returns to 0 OR sign flips OR exit_target is reached.
    """
    print("[etl] deriving episodes from signal transitions", flush=True)
    sig = df['signal'].to_numpy(dtype=np.int64)
    nc = df['N_contracts'].to_numpy(dtype=float)
    iv = df['iv'].to_numpy(dtype=float)
    edge = df['edge'].to_numpy(dtype=float)
    px = df['index_price'].to_numpy(dtype=float)
    # NC versions for headline NC reporting
    vega = df['vega_pnl_nc'].to_numpy(dtype=float)
    theta = df['theta_pnl_nc'].to_numpy(dtype=float)
    hedge = df['delta_hedge_pnl_nc'].to_numpy(dtype=float)
    hedge_c = df['hedge_costs_nc'].to_numpy(dtype=float)
    fees = df['option_fees_nc'].to_numpy(dtype=float)
    slip = df['slippage_nc'].to_numpy(dtype=float)
    gross = df['gross_pnl_nc'].to_numpy(dtype=float)
    net = df['net_pnl_nc'].to_numpy(dtype=float)
    ts = df['ts'].dt.strftime('%Y-%m-%d %H:%M').to_numpy()

    n = len(df)
    episodes: List[Dict] = []
    i = 0
    eid = 0
    while i < n:
        s = sig[i]
        if s == 0 or nc[i] <= 0:
            i += 1
            continue
        # Start of an episode
        entry = i
        side = int(np.sign(s))
        j = i
        # Walk until signal flips, becomes flat, or N_contracts=0
        while j + 1 < n:
            ns = sig[j + 1]
            if ns == 0 or np.sign(ns) != side or nc[j + 1] <= 0:
                break
            j += 1
        exit_ = j  # last bar where the position was held
        # The position closes at the END of bar exit_ (equivalently, the open of exit_+1).
        # Entry/exit snapshots should read from the close-time index, not the last-held bar.
        close_idx = min(exit_ + 1, n - 1)
        # cost-inclusive sums to compute net for *this* episode
        ep_vega = float(vega[entry:exit_ + 1].sum())
        ep_theta = float(theta[entry:exit_ + 1].sum())
        ep_hedge = float(hedge[entry:exit_ + 1].sum())
        ep_hcost = float(hedge_c[entry:exit_ + 1].sum())
        ep_fees = float(fees[entry:exit_ + 1].sum())
        ep_slip = float(slip[entry:exit_ + 1].sum())
        ep_gross = float(gross[entry:exit_ + 1].sum())
        ep_net = float(net[entry:exit_ + 1].sum())
        # Total signed contract exposure (sum of per-bar N_contracts as a proxy)
        ep_contracts = float(nc[entry:exit_ + 1].sum())
        iv0 = float(iv[entry])
        iv1 = float(iv[close_idx])
        px0 = float(px[entry])
        px1 = float(px[close_idx])
        ret = float(math.log(max(px1, 1e-9) / max(px0, 1e-9)))
        # Regime tag
        # rv_realized in vol points: derive from index price log-return std over the
        # [entry, close_idx] window; for 1-bar holds this yields a single 15m log-return.
        rv_window = px[entry:close_idx + 1]
        if len(rv_window) > 1:
            log_r = np.diff(np.log(np.maximum(rv_window, 1e-9)))
            rv_realized = float(np.sqrt(np.mean(log_r ** 2) * (365 * 24 * 4)) * 100)  # 15min cadence
        else:
            rv_realized = 0.0
        rv_entry = rv_realized  # best available proxy without full history scan
        if abs(ret) > 0.08:
            regime = 'stress'
        elif iv0 - rv_entry > 8:
            regime = 'rich'
        elif iv0 - rv_entry < 0:
            regime = 'cheap'
        else:
            regime = 'neutral'
        eid += 1
        episodes.append(dict(
            id=eid,
            entry=int(entry),
            exit=int(exit_),
            side=side,
            dur=int(exit_ - entry + 1),
            ts_entry=str(ts[entry]),
            ts_exit=str(ts[close_idx]),
            iv_entry=iv0,
            iv_exit=iv1,
            rv_entry=rv_entry,
            rv_realized=rv_realized,
            edge_entry=float(edge[entry]),
            vrp=iv0 - rv_entry,
            vega_pnl=ep_vega,
            theta_pnl=ep_theta,
            delta_hedge_pnl=ep_hedge,
            hedge_costs=ep_hcost,
            option_fees=ep_fees,
            slippage=ep_slip,
            gross_pnl=ep_gross,
            net_pnl=ep_net,
            N_contracts=ep_contracts,
            regime=regime,
            price_entry=px0,
            price_exit=px1,
            ret_btc=ret,
        ))
        i = exit_ + 1
    print(f"[etl]   {len(episodes):,} episodes derived", flush=True)
    return episodes


def downsample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15m bars to 1h. State series take last; PnL legs sum."""
    print("[etl] downsampling 15m -> 1h", flush=True)
    df = df.set_index('ts')
    agg = {
        'index_price': 'last',
        'iv': 'last',
        'tsrv': 'last',
        'rv_pred_pct': 'last',
        'edge': 'last',
        'signal': 'last',
        'equity': 'last',
        'equity_nc': 'last',
        'N_contracts': 'sum',
        'N_contracts_nc': 'sum',
        # comp legs
        'vega_pnl': 'sum',
        'theta_pnl': 'sum',
        'delta_hedge_pnl': 'sum',
        'hedge_costs': 'sum',
        'option_fees': 'sum',
        'slippage': 'sum',
        'net_pnl': 'sum',
        'gross_pnl': 'sum',
        # NC legs
        'vega_pnl_nc': 'sum',
        'theta_pnl_nc': 'sum',
        'delta_hedge_pnl_nc': 'sum',
        'hedge_costs_nc': 'sum',
        'option_fees_nc': 'sum',
        'slippage_nc': 'sum',
        'net_pnl_nc': 'sum',
        'gross_pnl_nc': 'sum',
    }
    h = df.resample('1h').agg(agg)
    # Forward-fill price/iv/equity to handle bars with no trades
    for c in ['index_price', 'iv', 'tsrv', 'rv_pred_pct', 'equity', 'equity_nc']:
        h[c] = h[c].ffill()
    # Edge can stay NaN -> fill 0 for downstream math
    h['edge'] = h['edge'].fillna(0.0)
    h['signal'] = h['signal'].fillna(0).astype(int)
    h = h.reset_index()
    print(f"[etl]   {len(h):,} 1h bars", flush=True)
    return h


def build_bench(price: np.ndarray, E0: float) -> Dict[str, np.ndarray]:
    p0 = price[0]
    btc_hold = E0 * price / p0
    n = len(price)
    bar_hours = 1.0
    bar_apr = (1 + 0.05) ** (1 / (365 * 24 / bar_hours)) - 1
    cash = 0.4 * E0
    btc_units = 0.6 * E0 / p0
    bench_6040 = np.empty(n)
    for i in range(n):
        cash *= 1 + bar_apr
        bench_6040[i] = cash + btc_units * price[i]
    cash_only = E0 * (1 + bar_apr) ** np.arange(n)
    return dict(btc_hold=btc_hold, bench_6040=bench_6040, cash=cash_only)


def package_bars(h: pd.DataFrame, mode: str) -> Dict:
    """Build per-bar series object for one mode (nc or comp)."""
    if mode == 'nc':
        eq = h['equity_nc'].to_numpy(dtype=float)
        vega = h['vega_pnl_nc'].to_numpy()
        theta = h['theta_pnl_nc'].to_numpy()
        hedge = h['delta_hedge_pnl_nc'].to_numpy()
        costs = -(h['hedge_costs_nc'].to_numpy() + h['option_fees_nc'].to_numpy() + h['slippage_nc'].to_numpy())
    else:
        eq = h['equity'].to_numpy(dtype=float)
        vega = h['vega_pnl'].to_numpy()
        theta = h['theta_pnl'].to_numpy()
        hedge = h['delta_hedge_pnl'].to_numpy()
        costs = -(h['hedge_costs'].to_numpy() + h['option_fees'].to_numpy() + h['slippage'].to_numpy())

    # BTC-denominated equity: idle capital held in BTC instead of USD cash.
    # eq_btc[i] = eq_btc[i-1] * (price[i]/price[i-1]) + bar_trade_pnl[i]
    # where bar_trade_pnl is the USD PnL delta of the cash-denominated equity.
    price = h['index_price'].ffill().to_numpy(dtype=float)
    eq_btc = np.empty_like(eq)
    eq_btc[0] = eq[0]
    for i in range(1, len(eq)):
        p_prev = price[i - 1] if price[i - 1] > 0 else price[i]
        ret_btc = (price[i] / p_prev) if p_prev > 0 else 1.0
        trade_pnl = eq[i] - eq[i - 1]
        eq_btc[i] = eq_btc[i - 1] * ret_btc + trade_pnl

    return dict(
        equity=[float(x) for x in eq],
        equity_btc_denom=[float(x) for x in eq_btc],
        vega_pnl_bar=[float(x) for x in vega],
        theta_pnl_bar=[float(x) for x in theta],
        delta_hedge_pnl_bar=[float(x) for x in hedge],
        costs_bar=[float(x) for x in costs],
    )


def daily_volume(episodes: List[Dict]) -> List[Dict]:
    """Group episodes by entry date; return daily contract count + episode count."""
    agg: Dict[str, Dict[str, float]] = {}
    for e in episodes:
        d = e['ts_entry'][:10]  # 'YYYY-MM-DD'
        row = agg.setdefault(d, {'n_contracts': 0.0, 'n_episodes': 0})
        row['n_contracts'] += float(e.get('N_contracts', 0.0))
        row['n_episodes'] += 1
    return [
        {'date': d, 'n_contracts': v['n_contracts'], 'n_episodes': int(v['n_episodes'])}
        for d, v in sorted(agg.items())
    ]


def fire_rate_by_year(df: pd.DataFrame) -> List[Dict]:
    """Episode fire rate per calendar year on the native 15m series.

    Uses signal transitions: an episode *starts* when the previous bar's signal was
    0 (or flat) and the current bar signals and N_contracts > 0. Divides those starts
    by the count of 15m bars in the year to give a fire rate.
    """
    sig = df['signal'].to_numpy(dtype=int)
    nc = df['N_contracts'].to_numpy(dtype=float)
    active = (sig != 0) & (nc > 0)
    # Starts = active AND previous bar was not active-same-side
    prev_active = np.zeros_like(active)
    prev_active[1:] = active[:-1]
    prev_sig = np.zeros_like(sig)
    prev_sig[1:] = sig[:-1]
    starts = active & (~prev_active | (np.sign(sig) != np.sign(prev_sig)))
    years = df['ts'].dt.year.to_numpy()
    out = []
    for y in sorted(set(int(v) for v in years)):
        mask = (years == y)
        n_bars = int(mask.sum())
        n_starts = int(starts[mask].sum())
        rate = (n_starts / n_bars) if n_bars > 0 else 0.0
        out.append(dict(year=int(y), n_bars=n_bars, n_starts=n_starts, rate=float(rate)))
    return out


def build_report(parquet_path: Path) -> Dict:
    df = load_parquet(parquet_path)

    # Episodes derived from native 15m series
    episodes = derive_episodes(df)
    fire_yr = fire_rate_by_year(df)
    daily_vol = daily_volume(episodes)

    # IS vs OOS PnL split — computed on native 15m series for accuracy
    oos_cut = pd.Timestamp(WALK_FORWARD_OOS_START, tz='UTC')
    is_mask = df['ts'] < oos_cut
    oos_mask = df['ts'] >= oos_cut
    split_pnl = dict(
        oos_start=WALK_FORWARD_OOS_START,
        is_bars=int(is_mask.sum()),
        oos_bars=int(oos_mask.sum()),
        is_net_pnl=float(df.loc[is_mask, 'net_pnl_nc'].sum()),
        oos_net_pnl=float(df.loc[oos_mask, 'net_pnl_nc'].sum()),
    )

    # Downsample for display
    h = downsample_to_1h(df)
    n_bars = len(h)
    bar_hours = 1.0

    ts = h['ts']
    price = h['index_price'].ffill().to_numpy(dtype=float)
    iv = h['iv'].ffill().to_numpy(dtype=float)
    # Model IV in vol points: prefer rv_pred_pct, fallback to sqrt(tsrv * annualization) * 100
    rv_pred = h['rv_pred_pct'].to_numpy(dtype=float)
    tsrv = h['tsrv'].to_numpy(dtype=float)
    iv_model = np.where(np.isnan(rv_pred) | (rv_pred <= 0),
                        np.sqrt(np.maximum(tsrv, 0) * (365 * 24 * 4)) * 100,
                        rv_pred)
    edge = h['edge'].to_numpy(dtype=float)
    signal = h['signal'].to_numpy(dtype=int)

    bench = build_bench(price, E0)

    # Headline stats
    print("[etl] computing headline stats", flush=True)
    eq_nc = h['equity_nc'].ffill().to_numpy(dtype=float)
    eq_co = h['equity'].ffill().to_numpy(dtype=float)
    hs_nc = headline_stats(eq_nc, bar_hours)
    hs_co = headline_stats(eq_co, bar_hours)

    # Returns for stat batteries (use NC by convention)
    r_nc = returns_from_eq(eq_nc)

    print("[etl] bootstrap (B=400)", flush=True)
    boot = bootstrap_sharpe(r_nc, bar_hours, B=400, block=24)
    print("[etl] permutation (B=400)", flush=True)
    perm = permutation_null(r_nc, bar_hours, B=400)

    print("[etl] walk-forward (k=8)", flush=True)
    wf = walk_forward(eq_nc, ts.dt.strftime('%Y-%m-%d').to_numpy(), bar_hours, k=8)
    wf_summary = walk_forward_summary(eq_nc, bar_hours, k=8)
    # Raw (rf-unadjusted) Sharpe — matches the scale of permutation_null
    obs_raw_sharpe = raw_sharpe(r_nc, bar_hours)

    print("[etl] rolling sharpe (90d)", flush=True)
    rol_sh = rolling_sharpe(r_nc, bar_hours, window_bars=90 * 24)

    print("[etl] rolling IC (180d)", flush=True)
    # IC defined as corr(edge, forward IV change over next 5 bars).
    # This is the standard vol-forecast IC — does the signal predict where IV goes?
    # The previous version used forward-strategy-equity which has selection bias
    # because strategy equity only moves when the signal fires.
    n_eq = len(eq_nc)
    fwd_iv_chg = np.zeros(n_eq - 5)
    for i in range(n_eq - 5):
        fwd_iv_chg[i] = float(iv[i + 5] - iv[i])
    ric = rolling_ic(edge[:len(fwd_iv_chg)], fwd_iv_chg, window_bars=180 * 24, step=24)

    print("[etl] drawdown", flush=True)
    dd_nc = drawdown_series(eq_nc)
    dd_co = drawdown_series(eq_co)

    print("[etl] decile table", flush=True)
    deciles = decile_table(episodes)

    print("[etl] monthly returns", flush=True)
    monthly_strat = monthly_returns(eq_nc, ts.to_numpy())
    monthly_bench = monthly_returns(bench['bench_6040'], ts.to_numpy())

    print("[etl] stress events", flush=True)
    sev = stress_events(eq_nc, bench['bench_6040'], ts.to_numpy())

    print("[etl] cost sensitivity grid", flush=True)
    pnl_no_costs = (h['vega_pnl_nc'] + h['theta_pnl_nc'] + h['delta_hedge_pnl_nc']).to_numpy(dtype=float)
    cost_per_bar = -(h['hedge_costs_nc'] + h['option_fees_nc'] + h['slippage_nc']).to_numpy(dtype=float)
    cost_sens = cost_sensitivity(pnl_no_costs, cost_per_bar, edge, signal, bar_hours, E0=E0)

    bars_common = dict(
        ts=[t.isoformat() for t in ts],
        price=[float(x) for x in price],
        iv_mkt=[float(x) for x in iv],
        iv_model=[float(x) for x in iv_model],
        edge=[float(x) for x in edge],
        signals=[int(x) for x in signal],
    )

    report = dict(
        meta=dict(
            strategy='BTC vol-arb (production v2)',
            subject='BTC',
            n_bars=int(n_bars),
            bar_hours=float(bar_hours),
            start=str(ts.iloc[0]),
            end=str(ts.iloc[-1]),
            generated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            E0=float(E0),
        ),
        bars_common=bars_common,
        bars_nc=package_bars(h, 'nc'),
        bars_comp=package_bars(h, 'comp'),
        episodes=episodes,
        bench_6040=[float(x) for x in bench['bench_6040']],
        bench_btc_hold=[float(x) for x in bench['btc_hold']],
        bench_cash=[float(x) for x in bench['cash']],
        decile_table=deciles,
        monthly_strat=monthly_strat,
        monthly_bench=monthly_bench,
        rolling_ic=ric,
        rolling_sharpe=[None if v is None else float(v) for v in rol_sh],
        drawdown_nc=dd_nc,
        drawdown_comp=dd_co,
        stress_events=sev,
        cost_sensitivity=cost_sens,
        bootstrap_sharpe=[float(x) for x in boot],
        permutation_null=[float(x) for x in perm],
        permutation_obs=float(obs_raw_sharpe),
        walk_forward=wf,
        walk_forward_summary=wf_summary,
        fire_rate_by_year=fire_yr,
        daily_volume=daily_vol,
        split_pnl=split_pnl,
        headline_stats_nc=hs_nc,
        headline_stats_comp=hs_co,
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='input', required=True, type=Path)
    ap.add_argument('--out', dest='output', required=True, type=Path)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args.input)

    print(f"[etl] writing {args.output}", flush=True)
    payload = json.dumps(report, separators=(',', ':'), allow_nan=False, default=str)
    args.output.write_text(payload, encoding='utf-8')
    raw_mb = len(payload) / 1024 / 1024

    gz_path = args.output.with_suffix(args.output.suffix + '.gz')
    with gzip.open(gz_path, 'wb', compresslevel=6) as f:
        f.write(payload.encode('utf-8'))
    gz_mb = gz_path.stat().st_size / 1024 / 1024

    print(f"[etl] done. raw={raw_mb:.2f} MB  gz={gz_mb:.2f} MB", flush=True)
    print(f"       NC sharpe = {report['headline_stats_nc']['sharpe']:.2f}  "
          f"NC total return = {report['headline_stats_nc']['total_ret']*100:.1f}%  "
          f"episodes = {len(report['episodes']):,}")


if __name__ == '__main__':
    main()
