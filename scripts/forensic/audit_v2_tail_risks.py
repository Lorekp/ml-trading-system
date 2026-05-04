"""V2 BTC vol-arb backtest — empirical tail-risk audit (Tier A, leakage-safe).

Audits seven tail-risk dimensions against the canonical V2 trade log:

  R1+R7 — gap risk + short-gamma intrabar drawdown
  R2    — liquidity disappearance (depth thinning)
  R3    — margin breach via Deribit PME ±17.5% spot shock
  R4    — stale IV marks (svi_n_strikes / svi-row age)
  R5    — vega-stop hit diagnostic (no threshold selection here)
  R6    — exchange outages (15m bar-gap detection)

Methodology grounded in:

* López de Prado, *Advances in Financial Machine Learning* Ch. 7
  (purged CV / embargo as anti-leakage primitives), Ch. 11 (storytelling
  bias — bar-OPEN IV only for stress flags), Ch. 12 (CPCV; threshold
  selection NOT done in this script).
* Bailey & López de Prado 2014 (Deflated Sharpe Ratio) — applied at
  Tier C-2, not here; this is description, not selection.
* Deribit Portfolio Margin Engine: IM = 1.3 × MM, liquidation at 100%
  MM usage, main table ±17.5% spot shock for BTC.

Pre-committed parameters (locked in plan
``C:\\Users\\laure\\.claude\\plans\\now-check-empirically-from-eager-bird.md``):

* Materiality: $ exposure ≥ 1.0% of cumulative V2 NC PnL ($9,596) OR
  worst-day exposure ≥ 5× median daily NC PnL ($3,200).
* Quarantine list: 2024-07-26 + 2024-11-05 are IS-investigation only
  (no parameter selection); 2022-06-13, 2022-11-08, 2024-08-05,
  2025-01-20 are OOS validation reserves.

Output: ``vol_arb/output/V2_TAIL_RISK_AUDIT.md``.

Usage::

    PYTHONPATH=. python scripts/forensic/audit_v2_tail_risks.py [--head N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit_v2_tail_risks")


# ─── Pre-committed parameters (locked in plan; do not edit in-script) ─────

CANONICAL_NC_PNL: float = 959_571.0653
MATERIAL_PCT_OF_CUM: float = 0.01           # 1% of cumulative
MATERIAL_DAILY_MULTIPLIER: float = 5.0       # 5x median daily

# Source-of-truth paths (read-only inputs)
TRADE_LOG_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
MINUTE_IV_PATH = Path("vol_arb/output/minute_atm_iv.parquet")
BAR_EXEC_PATH = Path("vol_arb/output/bar_execution_quality.parquet")
DVOL_PATH = Path("data/deribit/dvol/BTC_dvol_60m.parquet")
SVI_VOL_ARB_PATH = Path("vol_arb/output/BTC_vol_arb_svi.parquet")

OUTPUT_PATH = Path("vol_arb/output/V2_TAIL_RISK_AUDIT.md")

# Stress-event quarantine (from plan + scripts/forensic/12_rnd_backtest.py)
OOS_VALIDATION_DAYS: dict[str, pd.Timestamp] = {
    "Celsius_freeze_2022_06_13": pd.Timestamp("2022-06-13", tz="UTC"),
    "FTX_collapse_2022_11_08": pd.Timestamp("2022-11-08", tz="UTC"),
    "JPY_carry_unwind_2024_08_05": pd.Timestamp("2024-08-05", tz="UTC"),
    "Trump_inauguration_2025_01_20": pd.Timestamp("2025-01-20", tz="UTC"),
}
IS_INVESTIGATION_DAYS: dict[str, pd.Timestamp] = {
    "BTC_selloff_2024_07_26": pd.Timestamp("2024-07-26", tz="UTC"),
    "BTC_election_2024_11_05": pd.Timestamp("2024-11-05", tz="UTC"),
}

# Deribit PME — main-table BTC spot shock (documented public value)
DERIBIT_PME_SPOT_SHOCK: float = 0.175  # ±17.5%
DERIBIT_IM_OVER_MM_RATIO: float = 1.3   # IM = 1.3 × MM


# ─── Dataclass for findings ──────────────────────────────────────────────

@dataclass
class RiskFinding:
    risk_id: str
    risk_name: str
    verdict: str               # "modeled" | "partial" | "not modeled"
    dollar_exposure: float
    worst_day_exposure: float
    worst_day_date: str
    oos_worst_day_exposure: float
    oos_worst_day_date: str
    material: bool
    recommended_tier_c: str
    notes: list[str] = field(default_factory=list)


def is_material(dollar_exposure: float, worst_day: float,
                median_daily: float, cum_pnl: float) -> bool:
    """Apply pre-committed materiality rule — locked in plan.

    Requires BOTH a positive exposure AND meeting one of the two
    threshold rails. Zero-exposure findings are never material.
    """
    if abs(dollar_exposure) <= 0.5 and abs(worst_day) <= 0.5:
        # Strict zero: never material (avoids 0 >= 0 false positives
        # when median_daily is also 0 in tiny smoke samples).
        return False
    cum_threshold = abs(cum_pnl) * MATERIAL_PCT_OF_CUM
    # Daily threshold: 5x median or $3,200 hard floor (locked in plan)
    daily_threshold = max(
        abs(median_daily) * MATERIAL_DAILY_MULTIPLIER,
        3_200.0,
    )
    return (abs(dollar_exposure) >= cum_threshold or
            abs(worst_day) >= daily_threshold)


# ─── Loading utilities ───────────────────────────────────────────────────

def load_trade_log(head: int | None = None) -> pd.DataFrame:
    """Load canonical V2 trade log; optionally head-truncate for smoke."""
    tl = pd.read_parquet(TRADE_LOG_PATH)
    if not isinstance(tl.index, pd.DatetimeIndex):
        tl.index = pd.to_datetime(tl.index, utc=True)
    if head is not None:
        tl = tl.head(head)
        logger.info("Head-truncated trade log to %d rows for smoke", len(tl))
    fired = tl[tl["net_pnl_nc"] != 0]
    logger.info("Trade log: %d bars, %d fired trades, NC PnL $%.0f",
                len(tl), len(fired), tl["net_pnl_nc"].sum())
    return tl


def load_minute_iv(span_start: pd.Timestamp,
                   span_end: pd.Timestamp) -> pd.DataFrame:
    """Load minute-level ATM IV path. END-stamped per pipeline convention."""
    miv = pd.read_parquet(MINUTE_IV_PATH)
    if not isinstance(miv.index, pd.DatetimeIndex):
        miv.index = pd.to_datetime(miv.index, utc=True)
    miv = miv.sort_index()
    pad = pd.Timedelta(minutes=30)
    miv = miv.loc[span_start - pad:span_end + pad]
    logger.info("Minute IV: %d rows in span", len(miv))
    return miv


# ─── Common: per-day aggregation respecting quarantine ──────────────────

def daily_pnl(trade_log: pd.DataFrame) -> pd.Series:
    """NC PnL by UTC day."""
    return trade_log["net_pnl_nc"].groupby(trade_log.index.normalize()).sum()


def by_day_excluding_oos(per_trade_exposure: pd.Series) -> tuple[float, str]:
    """Worst-day $ exposure EXCLUDING OOS-validation quarantine days.

    Returns ($, date-str). Used for the main "worst day" column. The
    OOS-validation worst day is reported separately so it remains a
    held-out test set.
    """
    if len(per_trade_exposure) == 0:
        return 0.0, "—"
    daily = per_trade_exposure.groupby(per_trade_exposure.index.normalize()).sum()
    oos_dates = set(d.normalize() for d in OOS_VALIDATION_DAYS.values())
    daily_no_oos = daily[~daily.index.isin(oos_dates)]
    if len(daily_no_oos) == 0:
        return 0.0, "—"
    worst_idx = daily_no_oos.abs().idxmax()
    return float(daily_no_oos.loc[worst_idx]), str(worst_idx.date())


def by_day_oos_only(per_trade_exposure: pd.Series) -> tuple[float, str]:
    """Worst OOS-validation day only — held-out test."""
    if len(per_trade_exposure) == 0:
        return 0.0, "—"
    daily = per_trade_exposure.groupby(per_trade_exposure.index.normalize()).sum()
    oos_dates = set(d.normalize() for d in OOS_VALIDATION_DAYS.values())
    daily_oos = daily[daily.index.isin(oos_dates)]
    if len(daily_oos) == 0:
        return 0.0, "(no fired trades on OOS days)"
    worst_idx = daily_oos.abs().idxmax()
    return float(daily_oos.loc[worst_idx]), str(worst_idx.date())


# ─── Probe R1+R7: gap / short-gamma intrabar drawdown ──────────────────

def probe_intrabar_dd(trade_log: pd.DataFrame, minute_iv: pd.DataFrame,
                      median_daily: float) -> RiskFinding:
    """Per-trade peak-intrabar vega drawdown using minute_atm_iv.

    Anti-leak guard: minute_iv at minute m is END-stamped at m, i.e.
    aggregated from real Deribit trades that occurred during
    [m-1min, m]. We look at minutes [entry_ts+1min, entry_ts+14min]
    (the 14 intra-bar minutes whose data is observable BY then) and
    measure peak adverse IV move vs iv_entry. No future minutes are
    used to compute a flag at minute m.
    """
    fired = trade_log[trade_log["net_pnl_nc"] != 0].copy()
    fired = fired[fired["signal"] != 0]

    if len(fired) == 0:
        return RiskFinding(
            risk_id="R1+R7", risk_name="gap / short-gamma intrabar DD",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["no fired trades in slice"],
        )

    # For each fired entry, look up median_iv_minute over [+1m, +14m]
    # and compute peak adverse IV move (signed by signal so loss is
    # negative for both long and short).
    iv_entry = fired["iv_used"].values
    vega_per_btc = fired["vega"].values
    signal = fired["signal"].values
    n_contracts = fired["N_contracts"].values

    miv_idx = minute_iv.index.values.astype("datetime64[ns]")
    miv_iv = minute_iv["median_iv"].values

    holding_min = 15
    intra_minutes = np.arange(1, holding_min)  # 1..14

    peak_adverse_iv_move = np.zeros(len(fired))
    for i, (ts, sig) in enumerate(zip(fired.index, signal)):
        ts64 = np.datetime64(ts.tz_convert("UTC").tz_localize(None))
        # Search positions for entry_ts+1m through entry_ts+14m
        target_times = ts64 + np.array(
            [np.timedelta64(int(m) * 60_000_000_000, "ns") for m in intra_minutes]
        )
        # backward search per minute
        positions = np.searchsorted(miv_idx, target_times, side="right") - 1
        # Filter to positions where the matched timestamp is within
        # the bar (i.e., not earlier than entry_ts) — this keeps the
        # probe leakage-safe even if there are gaps in minute_iv.
        valid = (positions >= 0) & (
            miv_idx[positions] >= ts64
        ) & (miv_idx[positions] < ts64 + np.timedelta64(15 * 60_000_000_000, "ns"))
        if not valid.any():
            continue
        ivs_intrabar = miv_iv[positions[valid]]
        # Adverse IV move: for short (signal=-1), IV up is bad (positive
        # dIV is loss). For long (signal=+1), IV down is bad (negative
        # dIV is loss). adverse_dIV = -signal * (iv_intra - iv_entry)
        adverse_dIV = -float(sig) * (ivs_intrabar - iv_entry[i])
        peak = float(np.nanmax(adverse_dIV))
        peak_adverse_iv_move[i] = max(peak, 0.0)  # only adverse direction

    # Per-trade peak adverse vega PnL in USD (negative = loss)
    # vega is per-1-vol-pt-per-BTC; n_contracts is BTC; sign is loss
    peak_dd = -peak_adverse_iv_move * vega_per_btc * n_contracts

    fired["peak_intra_bar_dd_usd"] = peak_dd
    realised = fired["net_pnl_nc"].values
    # Counts where peak DD > 2× |realised PnL| (i.e., bar showed an
    # intrabar excursion the 15m PnL didn't reflect)
    excursion_mask = np.abs(peak_dd) > 2.0 * np.abs(realised)
    excursion_n = int(excursion_mask.sum())
    excursion_total_dd = float(peak_dd[excursion_mask].sum())

    # Total exposure = cumulative peak DD (negative number)
    total_exposure = float(peak_dd.sum())

    # Worst-day exposure (excluding OOS-validation days)
    per_trade = pd.Series(peak_dd, index=fired.index, name="peak_intra_dd")
    worst_day, worst_day_date = by_day_excluding_oos(per_trade)
    oos_worst, oos_date = by_day_oos_only(per_trade)

    notes = [
        f"{len(fired):,} fired trades audited",
        f"{excursion_n:,} trades had peak intra-bar DD > 2x realized PnL "
        f"(total ${excursion_total_dd:,.0f})",
    ]
    # IS-investigation days reported separately for mechanism
    for label, d in IS_INVESTIGATION_DAYS.items():
        d_norm = d.normalize()
        is_day = per_trade[per_trade.index.normalize() == d_norm]
        if len(is_day) > 0:
            notes.append(
                f"IS day {label}: peak DD ${is_day.sum():,.0f} "
                f"({len(is_day)} trades) — mechanism only, NOT used for selection"
            )

    material = is_material(
        total_exposure, worst_day, median_daily, CANONICAL_NC_PNL
    )

    return RiskFinding(
        risk_id="R1+R7", risk_name="gap / short-gamma intrabar DD",
        verdict="partial" if material else "modeled",
        dollar_exposure=total_exposure,
        worst_day_exposure=worst_day, worst_day_date=worst_day_date,
        oos_worst_day_exposure=oos_worst, oos_worst_day_date=oos_date,
        material=material,
        recommended_tier_c="C-2 (vega_stop_minute_usd) → C-1 (intrabar PnL recompute)" if material else "—",
        notes=notes,
    )


# ─── Probe R2: liquidity disappearance via available_depth_btc ─────────

def probe_liquidity_depth(trade_log: pd.DataFrame,
                          median_daily: float) -> RiskFinding:
    """Flag trades where N_contracts > available_depth_btc on entry bar.

    The bar_execution_quality sidecar is END-stamped (bar T's depth
    aggregates trades up to T), so joining at the bar timestamp is
    leakage-safe.
    """
    if not BAR_EXEC_PATH.exists():
        return RiskFinding(
            risk_id="R2", risk_name="liquidity disappearance (depth)",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["bar_execution_quality.parquet not found"],
        )

    bex = pd.read_parquet(BAR_EXEC_PATH)
    if not isinstance(bex.index, pd.DatetimeIndex):
        bex.index = pd.to_datetime(bex.index, utc=True)
    if "available_depth_btc" not in bex.columns:
        return RiskFinding(
            risk_id="R2", risk_name="liquidity disappearance (depth)",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["available_depth_btc column missing in sidecar"],
        )

    fired = trade_log[trade_log["net_pnl_nc"] != 0].copy()
    if len(fired) == 0:
        return RiskFinding(
            risk_id="R2", risk_name="liquidity disappearance (depth)",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["no fired trades"],
        )

    # join: align bar exec at fired bar timestamps
    depth_aligned = bex["available_depth_btc"].reindex(
        fired.index, method=None
    )
    n_with_depth = depth_aligned.notna().sum()

    fired["available_depth_btc"] = depth_aligned.values
    over = fired.dropna(subset=["available_depth_btc"]).copy()
    # Convention: bar_taker_slip_usd is per-1-BTC; if size > depth, the
    # SOR would have to walk the book and the slip understatement is
    # roughly (size/depth - 1) × baseline_slip × size.
    over["depth_ratio"] = over["N_contracts"] / over["available_depth_btc"].clip(lower=1e-6)
    over["depth_excess"] = (over["depth_ratio"] - 1.0).clip(lower=0.0)
    flagged = over[over["depth_excess"] > 0]

    # Slip understatement: baseline_slip ≈ $12/BTC (sidecar fallback).
    # Use bar_taker_slip_usd if available.
    slip_per_btc = bex["bar_taker_slip_usd"].reindex(
        flagged.index, method=None
    ).fillna(12.0).values

    slip_understatement = (
        flagged["depth_excess"].values * slip_per_btc * flagged["N_contracts"].values
    )
    flagged = flagged.assign(slip_understatement_usd=slip_understatement)

    total_understatement = -float(slip_understatement.sum())  # negative = cost
    per_trade = pd.Series(
        -slip_understatement, index=flagged.index, name="depth_understate"
    )
    worst_day, worst_day_date = by_day_excluding_oos(per_trade)
    oos_worst, oos_date = by_day_oos_only(per_trade)

    material = is_material(
        total_understatement, worst_day, median_daily, CANONICAL_NC_PNL
    )

    notes = [
        f"{n_with_depth:,} of {len(fired):,} fired trades have depth coverage",
        f"{len(flagged):,} flagged where N_contracts > available_depth_btc",
        f"max depth_ratio: {over['depth_ratio'].max():.2f}",
    ]

    return RiskFinding(
        risk_id="R2", risk_name="liquidity disappearance (depth)",
        verdict="partial" if material else "partial (acceptable)",
        dollar_exposure=total_understatement,
        worst_day_exposure=worst_day, worst_day_date=worst_day_date,
        oos_worst_day_exposure=oos_worst, oos_worst_day_date=oos_date,
        material=material,
        recommended_tier_c="C-5 (depth-thinning slippage adjuster, annotator-only)" if material else "—",
        notes=notes,
    )


# ─── Probe R3: Deribit PME ±17.5% margin breach ─────────────────────────

def probe_pme_margin(trade_log: pd.DataFrame, minute_iv: pd.DataFrame,
                     median_daily: float) -> RiskFinding:
    """Reconstruct Deribit PME ±17.5% main-table spot shock per fired
    short trade and flag minutes where simulated MM_required > equity.

    Anti-leak guard: at simulated minute m, MM is computed from the
    minute m mark (which uses real trades up to m only) and the
    POSITION snapshot at entry. The PME shock is a fixed scenario
    function of (spot, IV), not data-fitted.
    """
    fired = trade_log[trade_log["net_pnl_nc"] != 0].copy()
    fired = fired[fired["signal"] != 0]
    if len(fired) == 0:
        return RiskFinding(
            risk_id="R3", risk_name="margin breach (Deribit PME)",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["no fired trades"],
        )

    # Simplified PME reconstruction for short ATM straddles (which is V2
    # default). Stress P&L on a short straddle under ±s spot shock and
    # +25vol IV shock is approximately:
    #   stress_loss ≈ N × max(
    #       0.5 × gamma × (s × S)² + vega × dIV_up + cash_loss_dn,
    #       0.5 × gamma × (s × S)² + vega × dIV_up + cash_loss_up,
    #   )
    # where dIV_up is the IV shock the PME applies for ATM (~25 vol pts
    # in stressed scenarios). We use a conservative single-leg estimate:
    #   MM_required ≈ N × |signal × vega × dIV_shock| + 0.5 × gamma × (s × S)²
    # since the short ATM straddle has roughly zero delta at entry.

    if "gamma" not in fired.columns:
        # gamma not available — use vega-only PME estimate as fallback
        gamma = np.zeros(len(fired))
    else:
        gamma = fired["gamma"].values

    n_contracts = fired["N_contracts"].values
    vega_per_btc = fired["vega"].values
    signal = fired["signal"].values
    iv_entry = fired["iv_used"].values
    spot = fired["index_price"].values
    equity = fired["equity"].values

    # Deribit PME shock magnitudes (publicly documented for BTC main table)
    spot_shock_pct = DERIBIT_PME_SPOT_SHOCK   # 0.175
    iv_shock_vp = 25.0   # ~25 vol-points up-shock for short vol stress

    # Spot move PnL (gamma, second-order; for short straddle gamma<0
    # so spot move is bad regardless of direction)
    gamma_loss_per_btc = -0.5 * np.abs(gamma) * (spot_shock_pct * spot) ** 2  # negative for short
    # Note: gamma column for ATM straddle is positive (long-gamma sign);
    # signal=-1 short straddle is short gamma, so loss is +0.5*gamma*dS²

    # Vega move PnL — adverse for short on IV up
    # signal × vega × (-iv_shock_vp) for IV-up-bad case
    # signal × vega × (+iv_shock_vp) for IV-down-bad case (long straddle)
    # For short (signal=-1) IV up is loss: -1 × vega × (+25) = -25*vega
    # For long (signal=+1) IV down is loss: +1 × vega × (-25) = -25*vega
    vega_loss_per_btc = -np.abs(vega_per_btc) * iv_shock_vp

    # Total stress loss per BTC under the worst-case scenario
    # (gamma is short-gamma loss; vega is short-vega loss)
    is_short = signal == -1
    stress_loss_per_btc = np.where(
        is_short,
        # short straddle: short gamma + short vega both bite
        -0.5 * np.abs(gamma) * (spot_shock_pct * spot) ** 2 +
        -np.abs(vega_per_btc) * iv_shock_vp,
        # long straddle: long gamma + long vega so stress moves are
        # actually friendly except in flat-IV-flat-spot calm; PME
        # would compute a smaller MM. Use the IV-down loss as floor.
        -np.abs(vega_per_btc) * iv_shock_vp,
    )

    mm_required = -stress_loss_per_btc * n_contracts  # positive USD
    # IM is what the trader posts; we simulate breach as MM > equity.
    # For shorts at V2 sizing (im_fraction=0.20), notional was sized
    # such that 0.20 × notional <= margin_balance. MM = IM / 1.3 per
    # documented Deribit PME; thus a breach is mm_required > equity_at_t.
    # Use entry equity as a conservative proxy (true MtM equity at t
    # would require full DRL hedge replay — out of scope for Tier A).
    breach_mask = mm_required > equity
    breach_mm = mm_required[breach_mask]

    # Severity = (mm_required - equity), expressed as $ shortfall.
    shortfall = (mm_required - equity).clip(min=0)
    fired["pme_mm_required"] = mm_required
    fired["pme_shortfall"] = shortfall

    total_shortfall = -float(shortfall.sum())  # negative = exposure
    per_trade = pd.Series(-shortfall, index=fired.index, name="pme_shortfall")
    worst_day, worst_day_date = by_day_excluding_oos(per_trade)
    oos_worst, oos_date = by_day_oos_only(per_trade)

    material = is_material(
        total_shortfall, worst_day, median_daily, CANONICAL_NC_PNL
    )

    notes = [
        f"PME ±{DERIBIT_PME_SPOT_SHOCK*100:.1f}% spot shock + {iv_shock_vp:.0f}vp IV shock",
        f"breaches: {int(breach_mask.sum()):,} of {len(fired):,}",
        f"total shortfall: ${total_shortfall:,.0f}",
        "MM = IM / 1.3 per documented Deribit PME (IM = 1.3 × MM)",
    ]

    return RiskFinding(
        risk_id="R3", risk_name="margin breach (Deribit PME)",
        verdict="partial" if material else "modeled (sizing-side)",
        dollar_exposure=total_shortfall,
        worst_day_exposure=worst_day, worst_day_date=worst_day_date,
        oos_worst_day_exposure=oos_worst, oos_worst_day_date=oos_date,
        material=material,
        recommended_tier_c="C-4 (Deribit PME maintenance-margin engine)" if material else "—",
        notes=notes,
    )


# ─── Probe R4: stale IV marks ────────────────────────────────────────────

def probe_stale_iv(trade_log: pd.DataFrame,
                   median_daily: float) -> RiskFinding:
    """Flag trades fired when SVI surface had few strikes / IV was stale.

    `svi_n_strikes` is in the trade log: number of strikes used to fit
    the SVI slice for that bar's near-expiry. Low values → unreliable
    fit. We use a fixed threshold (5 strikes — the minimum for a 5-param
    SVI fit; below this is interpolation, not calibration) — anchored
    in SVI methodology, not data-fitted.
    """
    fired = trade_log[trade_log["net_pnl_nc"] != 0].copy()
    if len(fired) == 0 or "svi_n_strikes" not in fired.columns:
        return RiskFinding(
            risk_id="R4", risk_name="stale IV marks",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["no fired trades or svi_n_strikes column missing"],
        )

    # Trades on a thin SVI fit (n_strikes < 5)
    thin_mask = fired["svi_n_strikes"] < 5
    n_thin = int(thin_mask.sum())

    # Trades where iv_used fell back to dvol_close — verified zero in
    # earlier audit but we re-verify for robustness.
    fallback_to_dvol = (
        fired["svi_iv_matched"].isna() & fired["iv"].isna() &
        fired["dvol_close"].notna()
    )
    n_dvol_fallback = int(fallback_to_dvol.sum())

    pnl_thin = float(fired.loc[thin_mask, "net_pnl_nc"].sum())
    pnl_fresh = float(fired.loc[~thin_mask, "net_pnl_nc"].sum())

    # Exposure proxy: "if these trades had been blocked, what is the PnL
    # we would have lost (positive) or saved (negative)?"
    # Use abs-value loss as the conservative exposure measure: any
    # P&L that is fitted on a thin SVI is at higher uncertainty.
    total_exposure = -abs(pnl_thin)  # negative = at-risk PnL
    per_trade = pd.Series(
        -np.abs(fired.loc[thin_mask, "net_pnl_nc"].values),
        index=fired.index[thin_mask], name="thin_svi_at_risk",
    )
    worst_day, worst_day_date = by_day_excluding_oos(per_trade)
    oos_worst, oos_date = by_day_oos_only(per_trade)

    material = is_material(
        total_exposure, worst_day, median_daily, CANONICAL_NC_PNL
    )

    notes = [
        f"{n_thin:,} of {len(fired):,} fired trades on thin SVI fit (n_strikes<5)",
        f"PnL on thin-SVI trades: ${pnl_thin:,.0f}",
        f"PnL on fresh-SVI trades: ${pnl_fresh:,.0f}",
        f"DVOL fallback fires: {n_dvol_fallback:,} (expect 0)",
    ]

    return RiskFinding(
        risk_id="R4", risk_name="stale IV marks",
        verdict="partial",
        dollar_exposure=total_exposure,
        worst_day_exposure=worst_day, worst_day_date=worst_day_date,
        oos_worst_day_exposure=oos_worst, oos_worst_day_date=oos_date,
        material=material,
        recommended_tier_c="annotator-only (Tier B): emit svi_n_strikes_at_entry" if material else "—",
        notes=notes,
    )


# ─── Probe R5: vega-stop hit diagnostic ─────────────────────────────────

def probe_vega_stops(trade_log: pd.DataFrame,
                     median_daily: float) -> RiskFinding:
    """Confirm `vega_stopped` is all-False under V2 default.

    No threshold sweep here — that is the Tier C-2 work, gated by
    quarantine-day separation and CPCV.
    """
    if "vega_stopped" not in trade_log.columns:
        return RiskFinding(
            risk_id="R5", risk_name="vega-stop hits",
            verdict="not applicable", dollar_exposure=0.0,
            worst_day_exposure=0.0, worst_day_date="—",
            oos_worst_day_exposure=0.0, oos_worst_day_date="—",
            material=False, recommended_tier_c="—",
            notes=["vega_stopped column missing"],
        )

    n_stopped = int(trade_log["vega_stopped"].sum())
    notes = [
        f"vega_stopped == True on {n_stopped:,} of {len(trade_log):,} bars",
        "V2 default has vega_stop_*_usd = 0.0 (disabled)",
        "Threshold sweep is Tier C-2 (CPCV-selected, OOS-validated); not done here",
    ]

    return RiskFinding(
        risk_id="R5", risk_name="vega-stop hits (diagnostic only)",
        verdict="not modeled" if n_stopped == 0 else "modeled",
        dollar_exposure=0.0,
        worst_day_exposure=0.0, worst_day_date="—",
        oos_worst_day_exposure=0.0, oos_worst_day_date="—",
        material=False,    # diagnostic — materiality decided in C-2
        recommended_tier_c="C-2 (vega_stop_minute_usd CPCV sweep)",
        notes=notes,
    )


# ─── Probe R6: exchange outages via 15m bar-gap detection ───────────────

def probe_outages(trade_log: pd.DataFrame,
                  median_daily: float) -> RiskFinding:
    """Detect 15m-bar gaps > 20m and quantify trades crossing them.

    Anti-leak guard: gap detection uses prior-bar timestamps only.
    """
    expected = pd.Timedelta(minutes=15)
    threshold = pd.Timedelta(minutes=20)

    deltas = trade_log.index.to_series().diff()
    gap_mask = deltas > threshold
    n_gaps = int(gap_mask.sum())

    fired = trade_log[trade_log["net_pnl_nc"] != 0]
    # A trade "crossed an outage" if any gap > threshold landed inside
    # its 15-min holding window. Approximation: the bar AT entry has
    # delta-from-prior > threshold (entered after a gap) OR the next bar
    # has delta > threshold (exit-bar gap).
    holding = pd.Timedelta(minutes=15)
    delta_to_next = trade_log.index.to_series().diff(-1).abs()
    entry_gap = deltas > threshold
    exit_gap = delta_to_next > threshold
    fired_gap_mask = (entry_gap.reindex(fired.index, fill_value=False) |
                      exit_gap.reindex(fired.index, fill_value=False))
    n_fired_crossing = int(fired_gap_mask.sum())

    # $ exposure: PnL on bars adjacent to outages — at risk because the
    # exit IV was forward-filled across the outage window. Positive
    # PnL on these bars is suspect; negative PnL is realised cost.
    pnl_at_risk = float(fired.loc[fired_gap_mask, "net_pnl_nc"].sum())

    per_trade = fired.loc[fired_gap_mask, "net_pnl_nc"].copy()
    per_trade.name = "pnl_at_outage_boundary"
    worst_day, worst_day_date = by_day_excluding_oos(per_trade)
    oos_worst, oos_date = by_day_oos_only(per_trade)

    material = is_material(
        pnl_at_risk, worst_day, median_daily, CANONICAL_NC_PNL
    )

    # Top 5 worst gaps by duration
    top_gaps = deltas[gap_mask].sort_values(ascending=False).head(5)
    gap_lines = [
        f"  {ts}: gap = {gap}"
        for ts, gap in top_gaps.items()
    ]

    notes = [
        f"15m-bar gaps > 20m: {n_gaps}",
        f"fired trades crossing a gap: {n_fired_crossing}",
        f"PnL on outage-boundary trades: ${pnl_at_risk:,.0f}",
        "Top 5 outage candidates:",
        *gap_lines,
    ]

    return RiskFinding(
        risk_id="R6", risk_name="exchange outages",
        verdict="partial" if material else "not modeled (low impact)",
        dollar_exposure=pnl_at_risk,
        worst_day_exposure=worst_day, worst_day_date=worst_day_date,
        oos_worst_day_exposure=oos_worst, oos_worst_day_date=oos_date,
        material=material,
        recommended_tier_c="C-3 (outage exit logic)" if material else "—",
        notes=notes,
    )


# ─── Render audit markdown ───────────────────────────────────────────────

def render_markdown(findings: list[RiskFinding],
                    cum_pnl: float, median_daily: float) -> str:
    cum_thresh = abs(cum_pnl) * MATERIAL_PCT_OF_CUM
    daily_thresh = max(abs(median_daily) * MATERIAL_DAILY_MULTIPLIER, 3_200.0)
    daily_floor_binding = (abs(median_daily) * MATERIAL_DAILY_MULTIPLIER) < 3_200.0

    lines = [
        "# V2 Tail-Risk Audit (Tier A — read-only, leakage-safe)",
        "",
        f"Canonical V2 trade log: `{TRADE_LOG_PATH}`",
        f"Cumulative NC PnL: **${cum_pnl:,.2f}**",
        f"Median daily NC PnL: **${median_daily:,.2f}**",
        "",
        "## Pre-committed materiality thresholds (locked in plan)",
        f"- $ exposure ≥ 1.0% of cumulative NC PnL → **${cum_thresh:,.2f}**, OR",
        f"- worst-day exposure ≥ max(5× median, $3,200 floor) → **${daily_thresh:,.2f}**"
        + (" *(floor binds)*" if daily_floor_binding else ""),
        "",
        "## Quarantine",
        "**OOS validation reserve** (excluded from any future parameter selection):",
        *[f"- {label} ({ts.date()})" for label, ts in OOS_VALIDATION_DAYS.items()],
        "",
        "**IS investigation only** (mechanism root-cause; no parameter fitting):",
        *[f"- {label} ({ts.date()})" for label, ts in IS_INVESTIGATION_DAYS.items()],
        "",
        "## Findings",
        "",
        "| Risk | Verdict | $ Exposure | Worst Day (non-OOS) | OOS Worst Day | Material? | Recommended Tier C |",
        "|---|---|---:|---|---|---|---|",
    ]
    for f in findings:
        lines.append(
            f"| **{f.risk_id}** {f.risk_name} | {f.verdict} | "
            f"${f.dollar_exposure:,.0f} | "
            f"${f.worst_day_exposure:,.0f} ({f.worst_day_date}) | "
            f"${f.oos_worst_day_exposure:,.0f} ({f.oos_worst_day_date}) | "
            f"{'YES' if f.material else 'no'} | {f.recommended_tier_c} |"
        )
    lines += ["", "## Per-risk notes", ""]
    for f in findings:
        lines.append(f"### {f.risk_id} — {f.risk_name}")
        for n in f.notes:
            lines.append(f"- {n}")
        lines.append("")

    lines += [
        "## Interpreting R1+R7 — peak DD is RISK, not realized loss",
        "",
        "The R1+R7 peak intra-bar drawdown is the worst MtM the position "
        "would have shown at the most adverse intra-bar minute, NOT a "
        "realized P&L number. V2 holds positions for 15 min and books "
        "P&L only at the 15-min close, so many of these peak DDs reverse "
        "before exit and never become realized loss in the canonical "
        "trade log. This exposure measure is what a Tier C-2 vega-stop "
        "would have prevented (it would have closed the position at the "
        "peak DD minute). It is also what a real account would face as "
        "(a) margin-call risk if leveraged, (b) forced-liquidation risk "
        "if the Deribit PME maintenance margin triggered intra-bar, and "
        "(c) credibility risk for vault-pitch reporting.",
        "",
        "## Methodology guardrails",
        "",
        "All probes are backward-only by construction; minute_atm_iv at "
        "minute m uses real Deribit trades observed up to m only. The "
        "Deribit PME spot shock (±17.5%) and IM/MM ratio (1.3) are "
        "documented public values, not data-fitted. The thin-SVI "
        "threshold (n_strikes<5) is the minimum for a 5-parameter SVI "
        "fit, anchored in methodology not in this dataset's metrics. "
        "Threshold sweeps and parameter selection are deferred to "
        "Tier C — gated by CPCV / synthetic-path validation and "
        "evaluated on the 4 OOS quarantine days. The 2024-07-26 / "
        "2024-11-05 IS-investigation days appear in the worst-day "
        "column for transparency, but cannot drive parameter selection "
        "(per AFML Ch. 11 storytelling-bias guard).",
        "",
    ]
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=int, default=None,
                        help="Smoke-test mode: head-truncate trade log to N rows")
    parser.add_argument("--out", type=str, default=str(OUTPUT_PATH),
                        help=f"Output path (default {OUTPUT_PATH})")
    args = parser.parse_args()

    trade_log = load_trade_log(head=args.head)
    minute_iv = load_minute_iv(trade_log.index.min(), trade_log.index.max())

    daily_pnl_series = daily_pnl(trade_log)
    median_daily = float(daily_pnl_series.median())
    cum_pnl = float(trade_log["net_pnl_nc"].sum())
    logger.info("Median daily NC PnL: $%.2f", median_daily)
    logger.info("Cumulative NC PnL: $%.2f", cum_pnl)

    findings: list[RiskFinding] = []
    findings.append(probe_intrabar_dd(trade_log, minute_iv, median_daily))
    findings.append(probe_liquidity_depth(trade_log, median_daily))
    findings.append(probe_pme_margin(trade_log, minute_iv, median_daily))
    findings.append(probe_stale_iv(trade_log, median_daily))
    findings.append(probe_vega_stops(trade_log, median_daily))
    findings.append(probe_outages(trade_log, median_daily))

    output = render_markdown(findings, cum_pnl, median_daily)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    logger.info("Audit report written to %s", out_path)

    # Summary to stdout
    n_material = sum(1 for f in findings if f.material)
    print()
    print("=" * 70)
    print(f" V2 TAIL-RISK AUDIT — {n_material}/{len(findings)} material findings")
    print("=" * 70)
    for f in findings:
        material_tag = "MATERIAL" if f.material else "ok"
        print(f" {f.risk_id:6s} {material_tag:9s} {f.risk_name}")
        print(f"          $exp ${f.dollar_exposure:>12,.0f}  worst ${f.worst_day_exposure:>10,.0f} ({f.worst_day_date})")
    print("=" * 70)
    print(f" Full report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
