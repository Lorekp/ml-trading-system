"""Phase 2 prerequisite check — validate the NC-PnL approximation against
a real backtester run with a slope-amplified Kelly sidecar.

Pipeline:

1. Load existing K1 sidecar (`BTC_v2_kelly_mult_K1.parquet`).
2. Build a NEW sidecar that multiplies K1 by `MULT` on bars where
   `signal == 1` AND `svi_term_slope_1w_2w < THR`.
3. Save as `BTC_v2_kelly_mult_K1_slope_amp.parquet`.
4. Run the backtester with V2 preset + this new sidecar.
5. Compare results vs:
   - Cycle-10 baseline (canonical V2 trade log)
   - Phase 2 NC-PnL approximation (offline 1.25x multiplication)

If backtester PnL matches NC-PnL approximation within ±5%, the offline
analysis was honest. If it diverges, margin caps / realistic-slippage
non-linearity is binding and the offline approximation overstates the
real gain.
"""
from __future__ import annotations

import sys
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vol_arb.backtester import BacktestConfig, VolArbBacktester  # noqa: E402

K1_PATH = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_kelly_mult_K1.parquet"
SLOPE_AMP_PATH = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_kelly_mult_K1_slope_amp.parquet"
BASELINE_TRADELOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"

THR = -5.0
MULT = 1.25  # conservative — least Sharpe damage


def build_amped_sidecar(thr: float, mult: float) -> Path:
    print(f"[validate] Loading K1 sidecar: {K1_PATH.name}")
    k1 = pd.read_parquet(K1_PATH)
    print(f"[validate]  K1: {len(k1):,} rows, mean={k1['size_mult'].mean():.4f}")

    # Need term_slope from trade log to know which bars to amp
    print(f"[validate] Reading trade log for term_slope feature...")
    tl = pd.read_parquet(BASELINE_TRADELOG)
    if k1.index.tz is None:
        k1.index = pd.to_datetime(k1.index, utc=True)
    # Align: K1 index should be subset/superset of tl.index. Should match.
    aligned = k1.join(
        tl[["signal", "svi_term_slope_1w_2w"]], how="left",
    )
    amp_mask = (
        (aligned["signal"] == 1) &
        (aligned["svi_term_slope_1w_2w"] < thr)
    )
    n_amp = int(amp_mask.sum())
    print(f"[validate]  Amplifying {n_amp:,} bars (signal=1 AND slope<{thr})")

    # New sidecar: multiply K1 by `mult` on amp_mask
    new_mult = aligned["size_mult"].copy()
    new_mult.loc[amp_mask] = new_mult.loc[amp_mask] * mult
    # Existing K1 has cap at 2.0 — preserve cap on new sidecar too
    new_mult = new_mult.clip(upper=2.5)  # raise cap slightly to allow amp
    out = pd.DataFrame({"size_mult": new_mult})
    out.index.name = k1.index.name
    out.to_parquet(SLOPE_AMP_PATH)
    print(f"[validate]  wrote {SLOPE_AMP_PATH.relative_to(REPO_ROOT)}")
    print(f"[validate]  new sidecar: mean={out['size_mult'].mean():.4f}, "
          f"max={out['size_mult'].max():.4f}, "
          f"n_at_cap_2.5={(out['size_mult']>=2.49).sum()}")
    return SLOPE_AMP_PATH


def run_with_sidecar(sidecar_path: Path) -> tuple[pd.DataFrame, dict]:
    cfg = BacktestConfig().with_btc_production_filters_v2().with_realistic_costs()
    cfg.kelly_mult_path = str(sidecar_path)
    print(f"[validate] Running V2 backtest with sidecar: {sidecar_path.name}")
    t0 = time.time()
    _, tl = VolArbBacktester(cfg).run_backtest()
    elapsed = time.time() - t0
    fired = tl.loc[tl["signal"] != 0]
    summary = {
        "name": sidecar_path.stem,
        "n_trades": int(len(fired)),
        "n_long": int((fired["signal"]==1).sum()),
        "n_short": int((fired["signal"]==-1).sum()),
        "nc_pnl_total": float(fired["net_pnl_nc"].sum()),
        "nc_pnl_long": float(fired.loc[fired["signal"]==1, "net_pnl_nc"].sum()),
        "nc_pnl_short": float(fired.loc[fired["signal"]==-1, "net_pnl_nc"].sum()),
    }
    print(f"[validate]  Done in {elapsed:.1f}s  trades={summary['n_trades']:,}  "
          f"NC PnL=${summary['nc_pnl_total']:,.0f}")
    return tl, summary


def offline_nc_approximation(thr: float, mult: float) -> dict:
    """Compute the NC-PnL approximation (Phase 2 method)."""
    tl = pd.read_parquet(BASELINE_TRADELOG)
    is_long = tl["signal"] == 1
    is_extreme = tl["svi_term_slope_1w_2w"] < thr
    amp_mask = is_long & is_extreme
    NC_COLS = [
        "vega_pnl_nc", "theta_pnl_nc", "delta_hedge_pnl_nc",
        "option_fees_nc", "slippage_nc", "hedge_costs_nc", "net_pnl_nc",
    ]
    tl2 = tl.copy()
    for col in NC_COLS:
        tl2.loc[amp_mask, col] = tl2.loc[amp_mask, col] * mult
    fired = tl2.loc[tl2["signal"] != 0]
    return {
        "name": "offline_nc_approximation",
        "n_trades": int(len(fired)),
        "n_amp": int(amp_mask.sum()),
        "nc_pnl_total": float(fired["net_pnl_nc"].sum()),
        "nc_pnl_long": float(fired.loc[fired["signal"]==1, "net_pnl_nc"].sum()),
        "nc_pnl_short": float(fired.loc[fired["signal"]==-1, "net_pnl_nc"].sum()),
    }


def main() -> int:
    # ── 0. Baseline = on-disk canonical (already built with K1) ──
    tl_base = pd.read_parquet(BASELINE_TRADELOG)
    fired_base = tl_base.loc[tl_base["signal"] != 0]
    base_summary = {
        "name": "S0_baseline_canonical",
        "n_trades": int(len(fired_base)),
        "n_long": int((fired_base["signal"]==1).sum()),
        "n_short": int((fired_base["signal"]==-1).sum()),
        "nc_pnl_total": float(fired_base["net_pnl_nc"].sum()),
        "nc_pnl_long": float(fired_base.loc[fired_base["signal"]==1, "net_pnl_nc"].sum()),
        "nc_pnl_short": float(fired_base.loc[fired_base["signal"]==-1, "net_pnl_nc"].sum()),
    }
    print(f"[validate] Baseline (canonical V2): NC PnL=${base_summary['nc_pnl_total']:,.0f}")

    # ── 1. Offline NC approximation ──
    offline_summary = offline_nc_approximation(THR, MULT)
    delta_offline = offline_summary["nc_pnl_total"] - base_summary["nc_pnl_total"]
    print(f"[validate] Offline NC approx: NC PnL=${offline_summary['nc_pnl_total']:,.0f} "
          f"(delta=${delta_offline:+,.0f})")

    # ── 2. Build amped sidecar ──
    build_amped_sidecar(THR, MULT)

    # ── 3. Run backtester with amped sidecar ──
    tl_amped, amped_summary = run_with_sidecar(SLOPE_AMP_PATH)
    delta_real = amped_summary["nc_pnl_total"] - base_summary["nc_pnl_total"]
    print(f"[validate] Real backtest: NC PnL=${amped_summary['nc_pnl_total']:,.0f} "
          f"(delta=${delta_real:+,.0f})")

    # ── 4. Compare ──
    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'metric':<30} {'baseline':>14} {'offline':>14} {'real':>14} {'real-offline':>14}")
    for label, key in [
        ("n_trades", "n_trades"),
        ("n_long", "n_long"),
        ("n_short", "n_short"),
        ("nc_pnl_total", "nc_pnl_total"),
        ("nc_pnl_long", "nc_pnl_long"),
        ("nc_pnl_short", "nc_pnl_short"),
    ]:
        b = base_summary[key]
        o = offline_summary.get(key, "n/a")
        r = amped_summary[key]
        if isinstance(o, (int, float)) and isinstance(r, (int, float)):
            diff = r - o
            print(f"{label:<30} {b:>14,.0f} {o:>14,.0f} {r:>14,.0f} {diff:>+14,.0f}")
        else:
            print(f"{label:<30} {b:>14,.0f} {'n/a':>14} {r:>14,.0f}")
    print()
    print(f"Offline-approximation dPnL: ${delta_offline:+,.0f}")
    print(f"Real-backtest    dPnL:     ${delta_real:+,.0f}")
    if delta_offline != 0:
        ratio = delta_real / delta_offline
        print(f"Ratio (real/offline):      {ratio:.3f}")
        if abs(ratio - 1.0) < 0.05:
            print("VERDICT: Offline NC-PnL approximation is HONEST (within ±5%).")
        else:
            print(f"VERDICT: Offline approximation diverges from real by {(ratio-1)*100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
