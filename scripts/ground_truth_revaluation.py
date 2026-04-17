"""Phase D — Ground-truth attribution test for the vol-arb backtester.

Resolves whether the strategy is **fixed-contract** (interpretation A, what
the recent fix imposes) or **rolling-ATM** (interpretation B, what the
legacy code implies). The test uses *real same-contract trades* within a
tight tolerance around the exit time as the ground truth, then compares
how well each interpretation's exit IV matches that ground truth.

Method:
  For every entry bar in the backtester's filtered df, query the raw
  Deribit trade tape for any trade of the entry's exact (strike,
  expiry_dt, option_type) within [T + holding - tol, T + holding + tol].
  If a real trade is available, that is the ground-truth (GT) exit IV
  for that bar. Compare:
    - GT vs interp A (surface-reconstructed at entry K)
    - GT vs interp B (next-bar's iv_used, possibly different K)

Output: per-asset CSV with the per-trade comparison, plus a summary
showing correlation, RMSE, and aggregate sums.

Decision rule:
  - Higher correlation with GT + smaller RMSE = the right interpretation.
  - If interp A wins → keep the fix as default.
  - If interp B wins → revert to legacy as default.

Usage:
  PYTHONPATH=. python scripts/ground_truth_revaluation.py --currency BTC
  PYTHONPATH=. python scripts/ground_truth_revaluation.py --currency ETH

Notes:
  - Restricts to the subset of trades where a real same-contract trade
    exists near exit_target. Subset is biased toward liquid contracts;
    representativeness is documented in the report.
  - GT tolerance defaults to ±10 min around T+holding (so any same-
    contract trade between T+5min and T+25min counts). Tightening this
    increases ground-truth fidelity but reduces sample size.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester
from vol_arb.options_loader import load_option_trades
from vol_arb.svi import evaluate_surface_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ground_truth")


def _build_input_frame(currency: str) -> tuple[pd.DataFrame, BacktestConfig]:
    """Run the backtester far enough to get the aligned df, then return it."""
    cfg = (
        BacktestConfig.for_currency(currency)
        .with_realistic_costs()
        .model_copy(update={"entry_threshold": 25.0, "entry_threshold_long": 25.0})
    )
    bt = VolArbBacktester(cfg)
    df = bt._prepare()
    # df now has columns: iv_used (entry), iv_exit (post-fix or legacy depending
    # on cfg.legacy_exit_iv), iv_exit_legacy (always tier-3), strike, expiry_dt,
    # exit_target (from align_exits), price_exit, signal, etc.
    # We want both the fix's iv_exit AND the legacy iv_exit on the SAME df —
    # rerun align_exits with legacy on a copy to get both.
    df_fix = df.copy()
    bt.cfg.legacy_exit_iv = True
    df_legacy = bt._prepare()  # rebuilds df with legacy attribution
    bt.cfg.legacy_exit_iv = False  # restore
    return df_fix, df_legacy, cfg


def _attach_ground_truth(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    holding_minutes: int = 15,
    tol_minutes: int = 10,
) -> pd.DataFrame:
    """For each row in df, find a real same-(K, D, type) trade within
    [exit_target - tol, exit_target + tol]. Returns df augmented with
    GT_iv_exit (NaN where no trade in window).
    """
    # Trades must be sorted by timestamp; merge_asof requires it for `on`.
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades["expiry_dt"] = pd.to_datetime(trades["expiry_dt"], utc=True)
    trades["strike"] = trades["strike"].astype(np.float64)
    # The straddle is a call+put pair at the same K. Either leg's IV at exit
    # is a fair representative under put-call parity at ATM (~5% moneyness).
    # Use calls for the GT lookup (consistent with tier-1 in the backtester).
    calls = trades[trades["option_type"] == "C"][
        ["timestamp", "strike", "expiry_dt", "iv"]
    ].sort_values("timestamp").reset_index(drop=True)

    exit_ts = pd.to_datetime(df["exit_target"].values, utc=True) \
        if "exit_target" in df.columns \
        else pd.to_datetime(df.index.values, utc=True) + pd.Timedelta(minutes=holding_minutes)

    left = pd.DataFrame({
        "timestamp": exit_ts,
        "strike": np.asarray(df["strike"].values, dtype=np.float64),
        "expiry_dt": pd.to_datetime(df["expiry_dt"].values, utc=True),
        "_row_idx": np.arange(len(df)),
    }).sort_values("timestamp").reset_index(drop=True)

    # Use direction='nearest' with tolerance so we get the closest trade in
    # either direction within ±tol_minutes — gives us the best estimate of
    # the same-contract market mid at exit_target.
    tol = pd.Timedelta(minutes=tol_minutes)
    merged = pd.merge_asof(
        left, calls,
        on="timestamp",
        by=["strike", "expiry_dt"],
        direction="nearest",
        tolerance=tol,
        suffixes=("", "_tr"),
    )

    # Map the merged iv values back to df.index in original order.
    out = pd.Series(np.nan, index=df.index, dtype=float)
    idx_array = df.index.to_numpy()
    out.loc[idx_array[merged["_row_idx"].values]] = merged["iv"].values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ground-truth attribution test")
    parser.add_argument("--currency", default="BTC", help="BTC or ETH")
    parser.add_argument(
        "--tol-minutes", type=int, default=10,
        help="Tolerance around exit_target for matching real same-contract "
             "trade. Tighter → higher GT fidelity, fewer GT-available rows.",
    )
    parser.add_argument(
        "--output-csv", default="",
        help="Output path for the per-trade CSV (default: "
             "vol_arb/output/{currency}_ground_truth_attribution.csv).",
    )
    args = parser.parse_args()

    ccy = args.currency.upper()
    out_csv = Path(args.output_csv) if args.output_csv else Path(
        f"vol_arb/output/{ccy}_ground_truth_attribution.csv"
    )

    logger.info("Building backtester aligned df for %s (this loads predictions + 20M trades)...", ccy)
    df_fix, df_legacy, cfg = _build_input_frame(ccy)
    logger.info("Aligned df: %d rows", len(df_fix))

    logger.info("Loading raw trades for ground-truth lookup from %s...", cfg.option_trades_dir)
    trades = load_option_trades(Path(cfg.option_trades_dir))
    logger.info("Loaded %d trades", len(trades))

    logger.info("Attaching ground-truth IV (real same-contract trade within ±%d min)...", args.tol_minutes)
    df = df_fix.copy()
    df["iv_exit_fix"] = df["iv_exit"].values  # interp A (post-fix)
    df["iv_exit_legacy_attr"] = df_legacy["iv_exit"].values  # interp B
    df["iv_exit_GT"] = _attach_ground_truth(df, trades, tol_minutes=args.tol_minutes)

    # Subset to rows where GT is available
    has_gt = df["iv_exit_GT"].notna()
    n_total = len(df)
    n_gt = int(has_gt.sum())
    logger.info("GT-available rows: %d / %d (%.1f%%)", n_gt, n_total, 100 * n_gt / max(n_total, 1))

    if n_gt < 100:
        logger.error("Too few GT-available rows (%d) — widen --tol-minutes or check trade data.", n_gt)
        return

    sub = df.loc[has_gt, ["iv_used", "iv_exit_fix", "iv_exit_legacy_attr", "iv_exit_GT", "strike", "expiry_dt"]].copy()
    sub["dIV_GT"] = sub["iv_exit_GT"] - sub["iv_used"]
    sub["dIV_fix"] = sub["iv_exit_fix"] - sub["iv_used"]
    sub["dIV_legacy"] = sub["iv_exit_legacy_attr"] - sub["iv_used"]
    sub["err_fix"] = sub["dIV_fix"] - sub["dIV_GT"]
    sub["err_legacy"] = sub["dIV_legacy"] - sub["dIV_GT"]

    # Correlation of attribution dIV with GT dIV
    corr_fix = float(sub["dIV_fix"].corr(sub["dIV_GT"]))
    corr_legacy = float(sub["dIV_legacy"].corr(sub["dIV_GT"]))
    rmse_fix = float(np.sqrt((sub["err_fix"] ** 2).mean()))
    rmse_legacy = float(np.sqrt((sub["err_legacy"] ** 2).mean()))
    bias_fix = float(sub["err_fix"].mean())
    bias_legacy = float(sub["err_legacy"].mean())

    logger.info("=" * 70)
    logger.info("  PHASE D — GROUND-TRUTH ATTRIBUTION TEST — %s", ccy)
    logger.info("=" * 70)
    logger.info("  N rows with GT available: %d / %d (%.1f%%)", n_gt, n_total, 100 * n_gt / n_total)
    logger.info("  GT dIV stats: mean=%+.3f, std=%.3f, |median|=%.3f",
                sub["dIV_GT"].mean(), sub["dIV_GT"].std(), sub["dIV_GT"].abs().median())
    logger.info("")
    logger.info("  Interpretation A (FIX, contract-pinned):")
    logger.info("    corr(dIV_fix, dIV_GT)  = %+.4f", corr_fix)
    logger.info("    RMSE(dIV_fix - dIV_GT) = %.4f", rmse_fix)
    logger.info("    bias                   = %+.4f", bias_fix)
    logger.info("")
    logger.info("  Interpretation B (LEGACY, next-bar iv_used):")
    logger.info("    corr(dIV_legacy, dIV_GT)  = %+.4f", corr_legacy)
    logger.info("    RMSE(dIV_legacy - dIV_GT) = %.4f", rmse_legacy)
    logger.info("    bias                      = %+.4f", bias_legacy)
    logger.info("")
    logger.info("  WINNER (higher corr, lower RMSE): %s",
                "FIX (Interp A)" if (corr_fix > corr_legacy and rmse_fix < rmse_legacy)
                else ("LEGACY (Interp B)" if (corr_legacy > corr_fix and rmse_legacy < rmse_fix)
                      else "AMBIGUOUS"))

    sub.to_csv(out_csv)
    logger.info("Per-trade CSV saved to %s", out_csv)


if __name__ == "__main__":
    main()
