"""Fill-marked PnL — mark backtested straddles to recorded Deribit option
trades, mirroring the paper engine's ``realized_pnl_usd`` (deribit_paper/engine.py
:2781-2857).

WHY THIS EXISTS
---------------
The V2 backtester books the straddle's option PnL as ``signal * vega * d(iv_used)``
where ``iv_used = svi_iv_matched`` is SVI calibration noise (15-min change std
9.58 vol-points vs the real Deribit DVOL index 0.33vp). That harvests fit-noise
mean-reversion and booked +$1.6M; a historical paper-engine observation marked
to recorded option trades lost 14/14 trades. This module computes a
fill-marked option-leg PnL from the trade tape for PnL-realism diagnostics.
Sparse trade-tape coverage and backward matching mean this is evidence, not a
complete reconstruction of executable prices.

FILL-MARKED OPTION PnL (per the paper engine, deribit_paper/engine.py:2789-2792)
-------------------------------------------------------------------------------
    premium_in  = entry_call_price + entry_put_price        # BTC, matched trades
    premium_out = exit_call_price  + exit_put_price          # BTC, matched trades
    pnl_btc     = (premium_in - premium_out) * N_contracts   # SHORT payoff
    if long (signal == +1): pnl_btc = -pnl_btc
    option_pnl_usd = pnl_btc * index_price

COVERAGE
--------
The Deribit feed is a TRADE tape, not a book/quote feed, so a given ATM contract
trades sparsely around any 15-min bar boundary. Requiring a real fill for BOTH
legs at BOTH ends within a causal backward tolerance covers only a minority of
trades. We mark in tiers, but only tier 1 is admissible to the strict
historical-result gate. Tier 2 is diagnostic evidence and its coverage must not
be presented as real-fill coverage.

    tier 1 — recorded same-contract trade ``price`` on all four legs
    tier 2 — Deribit server ``mark_price`` (real published mark, model-smoothed)
    tier 0 — no same-contract fill within tolerance (uncovered; excluded)

All matching is causal: ``merge_asof(direction="backward")`` finds the most
recent real trade at or before the query time. Forward matching would peek.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Deribit option fee: 0.03% of underlying per contract, capped at 12.5% of the
# option's premium, applied per fill (mirrors deribit_paper/engine.py:2799-2806).
_DERIBIT_OPTION_FLAT_FEE = 0.0003
_DERIBIT_FEE_PREMIUM_CAP = 0.125


@dataclass
class HonestMarkResult:
    """Per-trade fill-marked option-leg PnL plus coverage diagnostics."""
    trade_log: pd.DataFrame          # input with honest columns attached
    n_fired: int
    n_tier1: int                     # both legs both ends real fills
    n_tier2: int                     # filled via mark_price fallback
    n_covered: int                   # tier1 + tier2
    coverage_frac: float

    def summary(self) -> str:
        return (
            f"honest premium mark: {self.n_covered}/{self.n_fired} covered "
            f"({self.coverage_frac:.1%}) — tier1 {self.n_tier1}, "
            f"tier2(mark) {self.n_tier2}"
        )


def _match_leg(
    queries: pd.DataFrame,
    trades_side: pd.DataFrame,
    query_ts_col: str,
    tol: pd.Timedelta,
) -> pd.DataFrame:
    """Causal backward match of each query row to the most recent same-contract
    trade at or before ``query_ts_col`` within ``tol``.

    ``trades_side`` is one option_type's trades, pre-sorted by timestamp, with
    integer-ns ``expiry_key`` and float ``strike`` ``by`` keys. Returns a frame
    aligned to ``queries`` carrying ``price`` and ``mark_price`` (NaN if no match).
    """
    left = queries.sort_values(query_ts_col).reset_index(drop=False)
    merged = pd.merge_asof(
        left,
        trades_side[["timestamp", "strike", "expiry_key", "price", "mark_price"]],
        left_on=query_ts_col,
        right_on="timestamp",
        by=["strike", "expiry_key"],
        direction="backward",
        tolerance=tol,
        suffixes=("", "_tr"),
        allow_exact_matches=True,
    )
    # restore original order
    merged = merged.set_index("index").sort_index()
    return merged[["price", "mark_price"]]


def mark_trade_log_to_real_premium(
    fired: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    tol_minutes: float = 15.0,
    flat_fee: float = _DERIBIT_OPTION_FLAT_FEE,
) -> HonestMarkResult:
    """Mark each fired straddle to real entry/exit option premiums.

    Parameters
    ----------
    fired : the fired-trade rows of a backtest trade log. Index is the entry
        timestamp. Must carry: strike, expiry_dt, signal, N_contracts_nc,
        index_price, exit_target.
    trades : the raw option trade tape (from ``load_option_trades``), with
        timestamp, strike, expiry_dt, option_type, price, mark_price.
    tol_minutes : causal backward match tolerance for a real fill.

    Returns
    -------
    HonestMarkResult with these columns attached to a copy of ``fired``:
        entry_call_real, entry_put_real, exit_call_real, exit_put_real (price),
        premium_tier (1 real-fill / 2 mark_price / 0 uncovered),
        real_option_pnl_nc  — fill-marked option-leg USD PnL (covered rows only;
                              NaN elsewhere), already net of the option fee.
    """
    fired = fired.copy()
    n_fired = len(fired)
    tol = pd.Timedelta(minutes=float(tol_minutes))

    # Causal key prep — float strike + int64-ns expiry on BOTH sides so the
    # merge_asof `by` keys match exactly (microsecond vs nanosecond expiry
    # resolution otherwise silently mismatches -> 0% coverage false negative).
    # Expiry `by` key: the expiry DATE string. All Deribit options settle at
    # 08:00 UTC on a date, so the date uniquely identifies the expiry — and a
    # string key is immune to the datetime64 us-vs-ns resolution mismatch
    # (tape expiry is [us], trade-log expiry is [ns]; a naive .astype('int64')
    # silently differs by 1000x -> 0% coverage false negative).
    tr = trades.copy()
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True)
    # Expiry `by` key: the UTC expiry DATE as datetime64[D]. This is the
    # vectorized equivalent of the prior strftime("%Y-%m-%d") string key —
    # bijective with it (verified elementwise-identical + identical cardinality
    # on the full tape) but ~180x faster (strftime on the 5M-row tape is ~20s
    # each). datetime64[D] floors to the UTC day; all Deribit options settle at
    # 08:00 UTC on a date so the day uniquely identifies the expiry. Must be the
    # SAME dtype on BOTH sides of the merge_asof `by`, see `base` below.
    tr["expiry_key"] = pd.to_datetime(tr["expiry_dt"], utc=True).values.astype("datetime64[D]")
    tr["strike"] = tr["strike"].astype(np.float64)
    tr = tr.sort_values("timestamp")
    calls = tr[tr["option_type"] == "C"].reset_index(drop=True)
    puts = tr[tr["option_type"] == "P"].reset_index(drop=True)

    entry_ts = pd.to_datetime(fired.index, utc=True)
    exit_ts = pd.to_datetime(fired["exit_target"].values, utc=True)
    base = pd.DataFrame({
        "strike": fired["strike"].astype(np.float64).values,
        "expiry_key": pd.to_datetime(fired["expiry_dt"], utc=True).values.astype("datetime64[D]"),
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
    }, index=np.arange(n_fired))

    legs = {
        ("entry", "C"): _match_leg(base.assign(q=base["entry_ts"]), calls, "q", tol),
        ("entry", "P"): _match_leg(base.assign(q=base["entry_ts"]), puts, "q", tol),
        ("exit", "C"): _match_leg(base.assign(q=base["exit_ts"]), calls, "q", tol),
        ("exit", "P"): _match_leg(base.assign(q=base["exit_ts"]), puts, "q", tol),
    }

    # Tier 1 = real fill `price` present on all four legs.
    # Tier 2 = fall back to `mark_price` where a fill price is missing but a
    #          same-contract mark exists (still a real Deribit-published number).
    def leg_value(key):
        px = legs[key]["price"].to_numpy(dtype=float)
        mk = legs[key]["mark_price"].to_numpy(dtype=float)
        tier1 = np.isfinite(px)
        val = np.where(tier1, px, mk)               # price else mark
        return val, tier1, np.isfinite(val)

    ec, ec_t1, ec_ok = leg_value(("entry", "C"))
    ep, ep_t1, ep_ok = leg_value(("entry", "P"))
    xc, xc_t1, xc_ok = leg_value(("exit", "C"))
    xp, xp_t1, xp_ok = leg_value(("exit", "P"))

    all_t1 = ec_t1 & ep_t1 & xc_t1 & xp_t1
    all_ok = ec_ok & ep_ok & xc_ok & xp_ok          # covered (tier 1 or 2)
    tier = np.where(all_t1, 1, np.where(all_ok, 2, 0)).astype(np.int8)

    fired["entry_call_real"] = ec
    fired["entry_put_real"] = ep
    fired["exit_call_real"] = xc
    fired["exit_put_real"] = xp
    fired["premium_tier"] = tier

    # Fill-marked option-leg PnL, using the paper engine's accounting formula.
    signal = fired["signal"].to_numpy(dtype=float)
    N = fired["N_contracts_nc"].to_numpy(dtype=float)
    index = fired["index_price"].to_numpy(dtype=float)

    premium_in = ec + ep                            # BTC
    premium_out = xc + xp
    pnl_btc = (premium_in - premium_out) * N        # SHORT payoff
    pnl_btc = np.where(signal == 1, -pnl_btc, pnl_btc)   # long flips sign
    option_pnl_usd = pnl_btc * index

    # Real per-fill fee (capped at 12.5% of that fill's premium), 4 legs.
    def fee(prem):
        return np.minimum(flat_fee, _DERIBIT_FEE_PREMIUM_CAP * prem)
    option_fees_usd = (fee(ec) + fee(ep) + fee(xc) + fee(xp)) * N * index

    real_pnl = option_pnl_usd - option_fees_usd
    real_pnl = np.where(all_ok, real_pnl, np.nan)   # only covered rows
    fired["real_option_pnl_nc"] = real_pnl

    n_tier1 = int(all_t1.sum())
    n_covered = int(all_ok.sum())
    n_tier2 = n_covered - n_tier1
    res = HonestMarkResult(
        trade_log=fired,
        n_fired=n_fired,
        n_tier1=n_tier1,
        n_tier2=n_tier2,
        n_covered=n_covered,
        coverage_frac=n_covered / max(n_fired, 1),
    )
    logger.info(res.summary())
    return res


def gate_no_fake_profit(
    marked: pd.DataFrame,
    *,
    model_col: str = "net_pnl_nc",
    real_col: str = "real_option_pnl_nc",
    min_covered: int = 30,
):
    """Run the legacy sign-only ``no_fake_profit`` smoke check.

    The check fails when a backtest reports aggregate profit while the same
    fill-covered trades show an aggregate loss. It catches the V2 artifact: the legacy
    model booked +$1.6M of ``vega x d(svi_iv)`` while real fills lost ~-$44/trade
    (historical live-paper observation: 14/14 losers).

    Passing means only that this aggregate sign contradiction was absent. The
    check deliberately ignores magnitude and per-trade correlation and does not
    validate costs, out-of-sample performance, or deployability. Use
    :func:`strict_pnl_realism_gate` when deciding whether a historical result may
    be written as a maintained research artifact.

    ``marked`` must already carry ``premium_tier`` and ``real_option_pnl_nc``
    (from :func:`mark_trade_log_to_real_premium`). ``model_col`` is the backtest's
    reported per-trade NC PnL.

    Returns a ``validation.gates.GateResult``; ``passed`` describes only this
    sign-only smoke check.
    """
    from validation.gates import GateResult

    cov = marked[marked["premium_tier"] > 0]
    m = pd.to_numeric(cov[model_col], errors="coerce").to_numpy()
    r = pd.to_numeric(cov[real_col], errors="coerce").to_numpy()
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    n = len(m)
    if n < min_covered:
        return GateResult("no_fake_profit", False,
                          f"only {n} fill-covered trades (<{min_covered}); "
                          "cannot run sign-only smoke check", {"n_covered": n})
    tot_m, tot_r = float(m.sum()), float(r.sum())
    corr = float(np.corrcoef(m, r)[0, 1]) if (m.std() > 0 and r.std() > 0) else 0.0
    false_profit = (tot_m > 0) and (tot_r <= 0)
    passed = not false_profit
    detail = {
        "n_covered": n, "model_total": tot_m, "real_fill_total": tot_r,
        "model_mean": tot_m / n, "real_fill_mean": tot_r / n,
        "per_trade_corr": corr, "false_profit": false_profit,
    }
    if false_profit:
        msg = (f"FALSE PROFIT — model reports +${tot_m:,.0f} on the covered "
               f"subset but real Deribit fills lose ${tot_r:,.0f} "
               f"(${tot_r/n:.1f}/trade). The aggregate signs contradict.")
    else:
        sign = "loss" if tot_r <= 0 else "profit"
        msg = (f"no false profit — model ${tot_m:,.0f} agrees in sign with "
               f"real fills ${tot_r:,.0f} ({sign}, ${tot_r/n:.1f}/trade). "
               f"per-trade corr {corr:.3f}; sign-only smoke check passed, "
               "but magnitude and correlation are not validated.")
    return GateResult("no_fake_profit", passed, msg, detail)


def strict_pnl_realism_gate(
    marked: pd.DataFrame,
    *,
    model_col: str = "net_pnl_nc",
    real_col: str = "real_option_pnl_nc",
    min_covered: int = 30,
    min_coverage: float = 0.05,
    min_corr: float = 0.85,
    max_total_rel_err: float = 0.30,
):
    """Strict comparison against tier-1 Deribit fills.

    Unlike :func:`gate_no_fake_profit`, this gate rejects sign, magnitude,
    or per-trade-shape disagreement. It also fails when the real-fill sample is
    too small or covers too little of the fired universe. Mark-price fallbacks
    (``premium_tier == 2``) are intentionally excluded.
    """
    from validation.gates import GateResult, pnl_realism_gate

    required = {"premium_tier", model_col, real_col}
    missing = sorted(required.difference(marked.columns))
    if missing:
        return GateResult(
            "strict_pnl_realism",
            False,
            f"missing required columns: {missing}",
            {"missing": missing},
        )

    tier1 = marked[marked["premium_tier"] == 1]
    model = pd.to_numeric(tier1[model_col], errors="coerce").to_numpy()
    real = pd.to_numeric(tier1[real_col], errors="coerce").to_numpy()
    finite = np.isfinite(model) & np.isfinite(real)
    model, real = model[finite], real[finite]
    n_fired = len(marked)
    n_covered = len(model)
    coverage = n_covered / max(n_fired, 1)

    coverage_detail = {
        "n_fired": n_fired,
        "n_tier1_finite": n_covered,
        "tier1_coverage": coverage,
        "min_covered": min_covered,
        "min_coverage": min_coverage,
    }
    if n_covered < min_covered or coverage < min_coverage:
        return GateResult(
            "strict_pnl_realism",
            False,
            (
                f"insufficient tier-1 fill evidence: {n_covered}/{n_fired} "
                f"({coverage:.1%}); require at least {min_covered} trades and "
                f"{min_coverage:.1%} coverage"
            ),
            coverage_detail,
        )

    result = pnl_realism_gate(
        model,
        real,
        min_corr=min_corr,
        max_total_rel_err=max_total_rel_err,
    )
    return GateResult(
        "strict_pnl_realism",
        result.passed,
        result.message,
        {**coverage_detail, **result.detail},
    )


def attach_honest_pnl(
    trade_log: pd.DataFrame,
    trades_dir: str | Path,
    *,
    tol_minutes: float = 15.0,
) -> HonestMarkResult:
    """Convenience wrapper: load the option tape for the trade-log window and
    mark the fired trades. Use in run_backtest / audit scripts.

    The model option-leg PnL to compare against is ``vega_pnl_nc + theta_pnl_nc``
    (the backtester's option legs); the honest counterpart is
    ``real_option_pnl_nc``. Feed both to ``validation.pnl_realism_gate`` on the
    covered subset.
    """
    from vol_arb.options_loader import load_option_trades

    fired = trade_log[trade_log.get("gross_pnl", pd.Series(1, index=trade_log.index)) != 0]
    if "N_contracts_nc" in fired.columns:
        fired = fired[fired["N_contracts_nc"] > 0]
    if len(fired) == 0:
        raise ValueError("no fired trades in trade_log")

    start = pd.to_datetime(fired.index.min(), utc=True).strftime("%Y-%m-%d")
    end = (pd.to_datetime(fired["exit_target"].max(), utc=True)
           + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Loading option tape %s..%s for honest mark of %d trades",
                start, end, len(fired))
    trades = load_option_trades(trades_dir, start_date=start, end_date=end)
    return mark_trade_log_to_real_premium(fired, trades, tol_minutes=tol_minutes)
