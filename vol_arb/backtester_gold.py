"""Gold futures options vol-arb backtester.

Simplified from BTC backtester — uses:
  - Vol model predictions (predicted RV)
  - Reconstructed ATM IV from option trades
  - IBKR fee model
  - COMEX margin model
  - Delta hedging with GC futures

Signal: edge = predicted_RV - market_IV
  Buy straddle when edge > +threshold (vol underpriced)
  Sell straddle when edge < -threshold (vol overpriced)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from vol_arb.greeks import compute_greeks, bs_price
from vol_arb.fees.ibkr_gold import IBKRGoldFees, IBKR_TIERED
from vol_arb.fees.comex_margin import COMEXMargin, COMEX_GC

logger = logging.getLogger(__name__)

CONTRACT_SIZE = 100  # oz per GC contract
SECONDS_PER_YEAR = 365.25 * 24 * 3600
BAR_SECONDS = 900  # 15-min bars


@dataclass
class GoldBacktestConfig:
    """Configuration for gold vol-arb backtest."""
    # Signal
    entry_threshold: float = 3.0   # vol points edge required
    # Sizing
    initial_equity: float = 100_000.0
    margin_utilization: float = 0.30  # fraction of equity as margin
    short_sizing_penalty: float = 0.50  # halve N for short straddles
    # Fees
    fees: IBKRGoldFees = None
    margin: COMEXMargin = None
    # Hedging
    delta_hedge_band: float = 0.30  # delta threshold for hedge (in contracts)
    hedge_sim_steps: int = 10
    # Filtering
    min_iv: float = 5.0   # don't trade when IV < 5%
    min_edge_ratio: float = 0.10  # edge must be > 10% of IV
    # Data
    predictions_path: str = "vol_model/output/XAUUSD_walkforward_predictions.parquet"
    iv_path: str = "data/databento/gold_1yr/GC_atm_iv_15m.parquet"

    def __post_init__(self):
        if self.fees is None:
            self.fees = IBKR_TIERED
        if self.margin is None:
            self.margin = COMEX_GC


class GoldBacktester:
    def __init__(self, cfg: GoldBacktestConfig):
        self.cfg = cfg

    def run(self) -> tuple[dict, pd.DataFrame]:
        cfg = self.cfg

        # ── Load data ──
        preds = pd.read_parquet(cfg.predictions_path)
        preds.index = pd.to_datetime(preds.index, utc=True)
        preds["rv_pred"] = np.exp(preds["y_pred"])  # predicted RV in %

        iv_df = pd.read_parquet(cfg.iv_path)
        iv_df["timestamp"] = pd.to_datetime(iv_df["timestamp"], utc=True)
        iv_df = iv_df.set_index("timestamp")

        # ── Align on common timestamps ──
        preds.index = preds.index.floor("15min")
        iv_df.index = iv_df.index.floor("15min")
        preds = preds[~preds.index.duplicated(keep="last")]
        iv_df = iv_df[~iv_df.index.duplicated(keep="last")]

        common = preds.index.intersection(iv_df.index)
        logger.info("Overlap: %d bars", len(common))

        if len(common) < 100:
            logger.error("Insufficient overlap (%d bars)", len(common))
            return {}, pd.DataFrame()

        rv_pred = preds.loc[common, "rv_pred"].values    # predicted RV (%)
        rv_true = np.exp(preds.loc[common, "y_true"].values)  # actual RV (%)
        market_iv = iv_df.loc[common, "iv_median"].values      # market IV (%)
        straddle_iv = iv_df.loc[common, "straddle_iv"].values
        futures_price = iv_df.loc[common, "futures_price"].values
        n_opt_trades = iv_df.loc[common, "n_trades"].values
        timestamps = common

        n_bars = len(common)

        # Use straddle_iv where available, else iv_median
        iv_used = np.where(np.isfinite(straddle_iv), straddle_iv, market_iv)

        # ── Signal ──
        edge = rv_pred - iv_used  # vol points
        signal = np.zeros(n_bars, dtype=int)
        signal[edge > cfg.entry_threshold] = 1    # buy straddle (vol underpriced)
        signal[edge < -cfg.entry_threshold] = -1  # sell straddle (vol overpriced)

        # Filter: don't trade when IV is too low or edge ratio too small
        too_low_iv = iv_used < cfg.min_iv
        too_small_edge = np.abs(edge) / np.maximum(iv_used, 1) < cfg.min_edge_ratio
        signal[too_low_iv | too_small_edge] = 0

        # Filter: need minimum option trades for reliable IV
        signal[n_opt_trades < 3] = 0

        n_trades = (signal != 0).sum()
        logger.info("Signals: %d buy, %d sell, %d skip",
                     (signal == 1).sum(), (signal == -1).sum(), (signal == 0).sum())

        # ── Greeks (ATM approximation) ──
        # For ATM straddle: delta ≈ 0, vega = 2 × call_vega
        # Approximate T from median DTE in IV data
        avg_dte = iv_df.loc[common, "avg_dte"].values
        T_years = avg_dte / 365.25
        sigma_dec = iv_used / 100.0

        # ATM straddle vega per oz = 2 × S × phi(d1) × sqrt(T) × 0.01
        # For ATM: d1 ≈ sigma*sqrt(T)/2, phi(d1) ≈ 0.399
        phi_d1 = 0.399  # ATM approximation
        vega_per_oz_per_vp = futures_price * phi_d1 * np.sqrt(np.maximum(T_years, 1e-6)) * 0.01
        vega_per_contract_per_vp = vega_per_oz_per_vp * CONTRACT_SIZE  # USD per vol point per contract

        # Theta per oz per day ≈ -S × sigma × phi(d1) / (2 × sqrt(T)) / 365.25
        theta_per_oz_day = -(futures_price * sigma_dec * phi_d1) / (2 * np.sqrt(np.maximum(T_years, 1e-6))) / 365.25
        theta_per_contract_day = theta_per_oz_day * CONTRACT_SIZE
        theta_per_contract_bar = theta_per_contract_day * (15 / 1440)  # per 15-min bar

        # ── Straddle premium (Brenner-Subrahmanyam ATM approximation) ──
        premium_per_oz = 0.8 * futures_price * sigma_dec * np.sqrt(np.maximum(T_years, 1e-6))
        premium_per_contract = premium_per_oz * CONTRACT_SIZE

        # ── Sizing ──
        N = np.zeros(n_bars)
        equity_arr = np.zeros(n_bars)
        trade_pnl = np.zeros(n_bars)
        vega_pnl_arr = np.zeros(n_bars)
        theta_pnl_arr = np.zeros(n_bars)
        fee_arr = np.zeros(n_bars)
        hedge_cost_arr = np.zeros(n_bars)

        equity = cfg.initial_equity

        for i in range(n_bars):
            if signal[i] != 0 and equity > 0 and futures_price[i] > 0:
                # Position size
                is_long = signal[i] == 1
                if is_long:
                    im_per = premium_per_contract[i]  # long = premium only
                else:
                    im_per = cfg.margin.short_atm_im_per_contract
                    im_per = max(im_per, futures_price[i] * CONTRACT_SIZE * cfg.margin.futures_im_pct)

                if im_per > 0:
                    n_contracts = int(equity * cfg.margin_utilization / im_per)
                else:
                    n_contracts = 0

                if not is_long:
                    n_contracts = int(n_contracts * cfg.short_sizing_penalty)

                n_contracts = max(n_contracts, 0)
                n_contracts = min(n_contracts, 20)  # hard cap
                N[i] = n_contracts

                if n_contracts > 0:
                    # ── Vega PnL ──
                    # IV change over 15-min bar (approximate from RV realization)
                    # In reality: IV at exit ≈ IV at entry + (realized_edge * decay)
                    # Simplified: vega_pnl = edge * vega * discount
                    # The "discount" accounts for imperfect edge capture
                    edge_capture = 0.30  # we capture ~30% of edge per bar
                    iv_change = edge[i] * edge_capture  # vol points
                    vega_pnl = signal[i] * iv_change * vega_per_contract_per_vp[i] * n_contracts
                    vega_pnl_arr[i] = vega_pnl

                    # ── Theta PnL ──
                    # Long straddle pays theta, short earns it
                    theta_pnl = signal[i] * theta_per_contract_bar[i] * n_contracts
                    theta_pnl_arr[i] = theta_pnl

                    # ── Fees ──
                    straddle_fee = cfg.fees.straddle_round_trip(n_contracts)
                    fee_arr[i] = straddle_fee

                    # ── Delta hedge cost ──
                    # Approximate: 2-3 hedge trades per bar at futures RT cost
                    n_hedges = 2
                    hedge_cost = n_hedges * cfg.fees.hedge_round_trip(max(n_contracts // 2, 1))
                    hedge_cost_arr[i] = hedge_cost

                    # ── Net PnL ──
                    trade_pnl[i] = vega_pnl + theta_pnl - straddle_fee - hedge_cost

            equity += trade_pnl[i]
            equity = max(equity, 0)
            equity_arr[i] = equity

        # ── Build results ──
        result_df = pd.DataFrame({
            "timestamp": timestamps,
            "signal": signal,
            "edge": edge,
            "rv_pred": rv_pred,
            "iv_used": iv_used,
            "futures_price": futures_price,
            "N_contracts": N,
            "vega_pnl": vega_pnl_arr,
            "theta_pnl": theta_pnl_arr,
            "fees": fee_arr,
            "hedge_cost": hedge_cost_arr,
            "net_pnl": trade_pnl,
            "equity": equity_arr,
        }).set_index("timestamp")

        metrics = self._compute_metrics(result_df)
        return metrics, result_df

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        active = df[df["signal"] != 0]
        n_trades = len(active)
        final_eq = df["equity"].iloc[-1]
        total_pnl = final_eq - cfg.initial_equity

        # Daily PnL for Sharpe
        daily = df["net_pnl"].resample("D").sum()
        daily_ret = daily / cfg.initial_equity
        sharpe = 0.0
        if len(daily) > 30 and daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)

        # Max drawdown
        eq = df["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / peak, 0)
        max_dd = dd.max() * 100

        # Win rate
        win_rate = (active["net_pnl"] > 0).mean() * 100 if n_trades > 0 else 0

        # Fee analysis
        total_fees = df["fees"].sum()
        total_hedge = df["hedge_cost"].sum()
        total_vega = df["vega_pnl"].sum()
        total_theta = df["theta_pnl"].sum()
        gross_pnl = total_vega + total_theta

        return {
            "total_pnl": total_pnl,
            "final_equity": final_eq,
            "sharpe": sharpe,
            "max_dd_pct": max_dd,
            "n_trades": n_trades,
            "n_buy": (df["signal"] == 1).sum(),
            "n_sell": (df["signal"] == -1).sum(),
            "win_rate": win_rate,
            "total_vega_pnl": total_vega,
            "total_theta_pnl": total_theta,
            "gross_pnl": gross_pnl,
            "total_fees": total_fees,
            "total_hedge_cost": total_hedge,
            "fee_pct_of_gross": total_fees / max(abs(gross_pnl), 1) * 100,
            "avg_edge_vp": active["edge"].mean() if n_trades > 0 else 0,
            "avg_contracts": active["N_contracts"].mean() if n_trades > 0 else 0,
        }


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Sweep thresholds
    print("=" * 80)
    print("  GOLD VOL-ARB BACKTEST — IBKR Fees, COMEX Margin")
    print("=" * 80)

    results = []
    for thresh in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        cfg = GoldBacktestConfig(entry_threshold=thresh)
        bt = GoldBacktester(cfg)
        metrics, df = bt.run()

        if not metrics:
            continue

        results.append({"threshold": thresh, **metrics})
        print(f"\n  Threshold={thresh:.0f} vp:")
        print(f"    PnL=${metrics['total_pnl']:>+10,.0f}  Sharpe={metrics['sharpe']:>6.2f}  "
              f"MaxDD={metrics['max_dd_pct']:>5.1f}%  Trades={metrics['n_trades']:>5,}")
        print(f"    Vega=${metrics['total_vega_pnl']:>+10,.0f}  Theta=${metrics['total_theta_pnl']:>+10,.0f}  "
              f"Fees=${metrics['total_fees']:>8,.0f}  Hedge=${metrics['total_hedge_cost']:>8,.0f}")
        print(f"    Fee% of gross: {metrics['fee_pct_of_gross']:.1f}%  Win: {metrics['win_rate']:.0f}%  "
              f"AvgEdge: {metrics['avg_edge_vp']:+.1f}vp  AvgN: {metrics['avg_contracts']:.1f}")

    # Best config
    if results:
        best = max(results, key=lambda r: r["sharpe"])
        print(f"\n{'='*80}")
        print(f"  BEST: threshold={best['threshold']:.0f}vp  "
              f"PnL=${best['total_pnl']:+,.0f}  Sharpe={best['sharpe']:.2f}  MaxDD={best['max_dd_pct']:.1f}%")
        print(f"  Fee drag: {best['fee_pct_of_gross']:.1f}% of gross (vs 55% on Deribit BTC)")


if __name__ == "__main__":
    main()
