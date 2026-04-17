"""Generic CME futures options vol-arb backtester.

Verbatim copy of backtester_gold.py with only the inputs parameterized
(instrument, contract size, fees, margin, paths). All PnL / sizing /
signal logic identical to the gold version.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from vol_arb.greeks import compute_greeks, bs_price
from vol_arb.fees.ibkr_futures import INSTRUMENT_FEES, IBKRFuturesFees
from vol_arb.fees.futures_margin import INSTRUMENT_MARGIN, FuturesMargin

logger = logging.getLogger(__name__)

SECONDS_PER_YEAR = 365.25 * 24 * 3600
BAR_SECONDS = 900


@dataclass
class FuturesBacktestConfig:
    instrument: str = "GC"
    entry_threshold: float = 3.0
    initial_equity: float = 100_000.0
    margin_utilization: float = 0.30
    short_sizing_penalty: float = 0.50
    fees: IBKRFuturesFees = None
    margin: FuturesMargin = None
    delta_hedge_band: float = 0.30
    hedge_sim_steps: int = 10
    min_iv: float = 5.0
    min_edge_ratio: float = 0.10
    predictions_path: str = ""
    iv_path: str = ""

    def __post_init__(self):
        if self.fees is None:
            self.fees = INSTRUMENT_FEES[self.instrument]
        if self.margin is None:
            self.margin = INSTRUMENT_MARGIN[self.instrument]
        if not self.predictions_path:
            pred_asset = {"GC": "XAUUSD"}.get(self.instrument, self.instrument)
            self.predictions_path = f"vol_model/output/{pred_asset}_walkforward_predictions.parquet"
        if not self.iv_path:
            dir_map = {"GC": "gold_1yr", "SI": "si_1yr", "CL": "cl_1yr", "ES": "es_1yr"}
            self.iv_path = f"data/databento/{dir_map[self.instrument]}/{self.instrument}_atm_iv_15m.parquet"


class FuturesBacktester:
    def __init__(self, cfg: FuturesBacktestConfig):
        self.cfg = cfg

    def run(self) -> tuple[dict, pd.DataFrame]:
        cfg = self.cfg
        CONTRACT_SIZE = cfg.fees.contract_size   # oz/bbl/multiplier per contract

        # ── Load data ──
        preds = pd.read_parquet(cfg.predictions_path)
        preds.index = pd.to_datetime(preds.index, utc=True)
        preds["rv_pred"] = np.exp(preds["y_pred"])  # predicted RV in % (training labels ln(vol_%))

        iv_df = pd.read_parquet(cfg.iv_path)
        iv_df["timestamp"] = pd.to_datetime(iv_df["timestamp"], utc=True)
        iv_df = iv_df.set_index("timestamp")

        # ── Align on common timestamps ──
        preds.index = preds.index.floor("15min")
        iv_df.index = iv_df.index.floor("15min")
        preds = preds[~preds.index.duplicated(keep="last")]
        iv_df = iv_df[~iv_df.index.duplicated(keep="last")]

        common = preds.index.intersection(iv_df.index)
        logger.info("[%s] Overlap: %d bars", cfg.instrument, len(common))
        if len(common) < 100:
            logger.error("[%s] Insufficient overlap (%d bars)", cfg.instrument, len(common))
            return {}, pd.DataFrame()

        rv_pred = preds.loc[common, "rv_pred"].values
        rv_true = np.exp(preds.loc[common, "y_true"].values)
        market_iv = iv_df.loc[common, "iv_median"].values
        straddle_iv = iv_df.loc[common, "straddle_iv"].values
        futures_price = iv_df.loc[common, "futures_price"].values
        n_opt_trades = iv_df.loc[common, "n_trades"].values
        timestamps = common

        n_bars = len(common)

        iv_used = np.where(np.isfinite(straddle_iv), straddle_iv, market_iv)

        # ── Signal ──
        edge = rv_pred - iv_used
        signal = np.zeros(n_bars, dtype=int)
        signal[edge > cfg.entry_threshold] = 1
        signal[edge < -cfg.entry_threshold] = -1

        too_low_iv = iv_used < cfg.min_iv
        too_small_edge = np.abs(edge) / np.maximum(iv_used, 1) < cfg.min_edge_ratio
        signal[too_low_iv | too_small_edge] = 0
        signal[n_opt_trades < 3] = 0

        n_trades = (signal != 0).sum()
        logger.info("[%s] Signals: %d buy, %d sell, %d skip",
                    cfg.instrument,
                    int((signal == 1).sum()), int((signal == -1).sum()), int((signal == 0).sum()))

        # ── Greeks (ATM approximation) ──
        avg_dte = iv_df.loc[common, "avg_dte"].values
        T_years = avg_dte / 365.25
        sigma_dec = iv_used / 100.0

        phi_d1 = 0.399
        vega_per_unit_per_vp = futures_price * phi_d1 * np.sqrt(np.maximum(T_years, 1e-6)) * 0.01
        vega_per_contract_per_vp = vega_per_unit_per_vp * CONTRACT_SIZE

        theta_per_unit_day = -(futures_price * sigma_dec * phi_d1) / (2 * np.sqrt(np.maximum(T_years, 1e-6))) / 365.25
        theta_per_contract_day = theta_per_unit_day * CONTRACT_SIZE
        theta_per_contract_bar = theta_per_contract_day * (15 / 1440)

        premium_per_unit = 0.8 * futures_price * sigma_dec * np.sqrt(np.maximum(T_years, 1e-6))
        premium_per_contract = premium_per_unit * CONTRACT_SIZE

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
                is_long = signal[i] == 1
                if is_long:
                    im_per = premium_per_contract[i]
                else:
                    im_per = cfg.margin.short_atm_im_per_contract
                    im_per = max(im_per, futures_price[i] * CONTRACT_SIZE * cfg.margin.futures_im_pct)

                n_contracts = int(equity * cfg.margin_utilization / im_per) if im_per > 0 else 0
                if not is_long:
                    n_contracts = int(n_contracts * cfg.short_sizing_penalty)
                n_contracts = max(n_contracts, 0)
                n_contracts = min(n_contracts, 20)
                N[i] = n_contracts

                if n_contracts > 0:
                    edge_capture = 0.30
                    iv_change = edge[i] * edge_capture
                    vega_pnl = signal[i] * iv_change * vega_per_contract_per_vp[i] * n_contracts
                    vega_pnl_arr[i] = vega_pnl

                    theta_pnl = signal[i] * theta_per_contract_bar[i] * n_contracts
                    theta_pnl_arr[i] = theta_pnl

                    straddle_fee = cfg.fees.straddle_round_trip(n_contracts)
                    fee_arr[i] = straddle_fee

                    n_hedges = 2
                    hedge_cost = n_hedges * cfg.fees.hedge_round_trip(max(n_contracts // 2, 1))
                    hedge_cost_arr[i] = hedge_cost

                    trade_pnl[i] = vega_pnl + theta_pnl - straddle_fee - hedge_cost

            equity += trade_pnl[i]
            equity = max(equity, 0)
            equity_arr[i] = equity

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
        metrics["instrument"] = cfg.instrument
        return metrics, result_df

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        active = df[df["signal"] != 0]
        n_trades = len(active)
        final_eq = df["equity"].iloc[-1]
        total_pnl = final_eq - cfg.initial_equity

        daily = df["net_pnl"].resample("D").sum()
        daily_ret = daily / cfg.initial_equity
        sharpe = 0.0
        if len(daily) > 30 and daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)

        eq = df["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / peak, 0)
        max_dd = dd.max() * 100

        win_rate = (active["net_pnl"] > 0).mean() * 100 if n_trades > 0 else 0

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
            "n_buy": int((df["signal"] == 1).sum()),
            "n_sell": int((df["signal"] == -1).sum()),
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
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--inst", choices=list(INSTRUMENT_FEES.keys()), required=True)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    args = p.parse_args()

    print("=" * 80)
    print(f"  {args.inst} VOL-ARB BACKTEST — IBKR Fees, CME Margin")
    print("=" * 80)

    results = []
    for thresh in args.thresholds:
        cfg = FuturesBacktestConfig(instrument=args.inst, entry_threshold=thresh)
        bt = FuturesBacktester(cfg)
        try:
            metrics, _ = bt.run()
        except FileNotFoundError as e:
            print(f"  [skip t={thresh}] missing: {e}")
            continue
        if not metrics:
            continue
        results.append({"threshold": thresh, **metrics})
        print(f"\n  Threshold={thresh:.0f} vp:")
        print(f"    PnL=${metrics['total_pnl']:>+10,.0f}  Sharpe={metrics['sharpe']:>6.2f}  "
              f"MaxDD={metrics['max_dd_pct']:>5.1f}%  Trades={metrics['n_trades']:>5,}")
        print(f"    Vega=${metrics['total_vega_pnl']:>+10,.0f}  "
              f"Theta=${metrics['total_theta_pnl']:>+10,.0f}  "
              f"Fees=${metrics['total_fees']:>8,.0f}  Hedge=${metrics['total_hedge_cost']:>8,.0f}")
        print(f"    Fee% of gross: {metrics['fee_pct_of_gross']:.1f}%  Win: {metrics['win_rate']:.0f}%  "
              f"AvgEdge: {metrics['avg_edge_vp']:+.1f}vp  AvgN: {metrics['avg_contracts']:.1f}")

    if results:
        out = Path(f"vol_arb/output/{args.inst}_sweep.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_parquet(out)
        best = max(results, key=lambda r: r["sharpe"])
        print(f"\n{'='*80}")
        print(f"  BEST: threshold={best['threshold']:.0f}vp  PnL=${best['total_pnl']:+,.0f}  "
              f"Sharpe={best['sharpe']:.2f}  MaxDD={best['max_dd_pct']:.1f}%")


if __name__ == "__main__":
    main()
