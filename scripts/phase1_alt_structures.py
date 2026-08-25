"""Historical Phase 1 alternative-structure tests under invalidated V2.

The backtester contains optional strangle infrastructure
(``use_instrument_routing=True``) that selects between ATM straddle, bear
strangle (ATM call + 25Δ OTM put), and bull strangle (25Δ OTM call + ATM
put) based on a skew gate. Earlier cycle-9 runs found this routing strictly
worse than ATM, so the historical V2 default is ``False``.

An earlier decomposition suggested that long-side results concentrated in
extreme positive 1w 25-delta skew (>20). This script preserves the subsequent
historical routing comparison:

* TransVol + LGBM retrained
* skew>25 short filter active
* thr_S = 20 (was 22)

If routing wins ANY pass under cycle-10, it's a structural change worth
shipping. If it still loses, the routing question is closed for cycle-10
too and the section can move to Phase 3 (asymmetric hedging / corridor
variance).

The script uses the existing backtester with cfg flags only — no source
changes. Output:

* ``vol_arb/output/PHASE1_ALT_STRUCTURES.md``
* ``vol_arb/output/PHASE1_ALT_STRUCTURES_TRADELOGS/`` per-config trade logs
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vol_arb.backtester import BacktestConfig, VolArbBacktester  # noqa: E402

OUT_DIR = REPO_ROOT / "vol_arb" / "output" / "PHASE1_ALT_STRUCTURES_TRADELOGS"
OUT_MD = REPO_ROOT / "vol_arb" / "output" / "PHASE1_ALT_STRUCTURES.md"
BASELINE_TRADELOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"


@dataclass
class ConfigCase:
    name: str
    desc: str
    use_routing: bool
    routing_mode: str = "svi_skew"
    svi_thresh: float = 5.0
    jump_thresh: float = 0.05


CASES: list[ConfigCase] = [
    ConfigCase(name="S0_baseline", desc="V2 cycle-10 (ATM only, no routing)",
               use_routing=False),
    ConfigCase(name="S1_svi_thr5",
               desc="V2 + svi_skew routing thr=5 (T2.2 retest)",
               use_routing=True, svi_thresh=5.0),
    ConfigCase(name="S2_svi_thr10",
               desc="V2 + svi_skew routing thr=10",
               use_routing=True, svi_thresh=10.0),
    ConfigCase(name="S3_svi_thr20",
               desc="V2 + svi_skew routing thr=20 (Phase-0-driven)",
               use_routing=True, svi_thresh=20.0),
    ConfigCase(name="S4_jump_thr05",
               desc="V2 + jump_skew routing thr=0.05 (T2.2b retest)",
               use_routing=True, routing_mode="jump_skew", jump_thresh=0.05),
]
# NOTE: An inverted-routing variant (skew-fading direction = buy the cheap
# wing) was tested in this spike via a temporary cfg.invert_routing flag.
# All inverted variants also strictly lost vs ATM baseline (-$266K to
# -$320K NC PnL, 0/5 per-year strict pass). The flag has been removed
# since both routing directions are now ruled out under cycle-10.
# The historical run rejected both routing directions under cycle-10.


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def daily_sharpe_nc(trade_log: pd.DataFrame) -> float:
    """Daily NC Sharpe (per-bar PnL summed by day)."""
    daily = trade_log["net_pnl_nc"].resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return float("nan")
    return float(np.sqrt(365) * daily.mean() / daily.std())


def trade_sharpe_nc(trade_log: pd.DataFrame) -> float:
    """Per-trade NC Sharpe."""
    fired = trade_log.loc[trade_log["signal"] != 0, "net_pnl_nc"]
    if len(fired) < 5 or fired.std() == 0:
        return float("nan")
    return float(fired.mean() / fired.std())


def max_drawdown_nc(trade_log: pd.DataFrame) -> float:
    """Max NC drawdown via cumulative bar-PnL."""
    cum = trade_log["net_pnl_nc"].cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


def per_year_nc(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Per-year fired-trade NC PnL."""
    fired = trade_log.loc[trade_log["signal"] != 0].copy()
    fired["year"] = fired.index.year
    return fired.groupby("year").agg(
        n=("net_pnl_nc", "count"),
        net=("net_pnl_nc", "sum"),
        per_trade=("net_pnl_nc", "mean"),
    ).round({"net": 0, "per_trade": 1})


def summarize_run(name: str, tl: pd.DataFrame) -> dict:
    fired = tl.loc[tl["signal"] != 0]
    return {
        "name": name,
        "n_trades": int(len(fired)),
        "nc_pnl": round(float(fired["net_pnl_nc"].sum()), 0),
        "nc_pnl_long": round(float(fired.loc[fired["signal"]==1, "net_pnl_nc"].sum()), 0),
        "nc_pnl_short": round(float(fired.loc[fired["signal"]==-1, "net_pnl_nc"].sum()), 0),
        "n_long": int((fired["signal"]==1).sum()),
        "n_short": int((fired["signal"]==-1).sum()),
        "trade_sharpe": round(trade_sharpe_nc(tl), 3),
        "daily_sharpe": round(daily_sharpe_nc(tl), 3),
        "max_dd": round(max_drawdown_nc(tl), 0),
        "n_atm": int((tl.get("instrument_type", pd.Series([0]*len(tl))) == 0).sum()) if "instrument_type" in tl.columns else None,
        "n_bear": int((tl.get("instrument_type", pd.Series([0]*len(tl))) == 1).sum()) if "instrument_type" in tl.columns else None,
        "n_bull": int((tl.get("instrument_type", pd.Series([0]*len(tl))) == -1).sum()) if "instrument_type" in tl.columns else None,
    }


def run_case(case: ConfigCase) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    cfg.use_instrument_routing = case.use_routing
    cfg.routing_mode = case.routing_mode
    cfg.svi_skew_threshold = case.svi_thresh
    cfg.jump_skew_threshold = case.jump_thresh
    print(f"[phase1] running {case.name}: {case.desc} ...")
    t0 = time.time()
    bt = VolArbBacktester(cfg)
    _, tl = bt.run_backtest()
    print(f"[phase1]   done in {time.time()-t0:.1f}s, {(tl['signal']!=0).sum():,} trades, "
          f"NC PnL ${tl['net_pnl_nc'].sum():,.0f}")
    summary = summarize_run(case.name, tl)
    py = per_year_nc(tl)
    return summary, py, tl


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Use baseline trade log on disk (cycle-10 canonical) instead of re-running
    print(f"[phase1] reading cached baseline {BASELINE_TRADELOG.name} (no re-run)")
    baseline_tl = pd.read_parquet(BASELINE_TRADELOG)
    cases_summaries = [summarize_run(CASES[0].name, baseline_tl)]
    cases_per_year = [per_year_nc(baseline_tl)]
    cases_tls = [baseline_tl]

    # 2) Run alternative cases
    for case in CASES[1:]:
        summary, py, tl = run_case(case)
        cases_summaries.append(summary)
        cases_per_year.append(py)
        cases_tls.append(tl)
        out_path = OUT_DIR / f"{case.name}_trade_log.parquet"
        tl.to_parquet(out_path)
        print(f"[phase1]   saved {out_path.relative_to(REPO_ROOT)}")

    # 3) Build markdown report
    md = []
    md.append("# Phase 1 - Alternative structure backtests (cycle-10 V2)\n")
    md.append("Historical test of whether the strangle routing infrastructure ")
    md.append("(`use_instrument_routing=True`) wins under cycle-10's V2 ")
    md.append("(TransVol + LGBM retrain + skew>25 short filter + thr_S=20). ")
    md.append("Earlier cycle-9 runs found routing strictly worse; ")
    md.append("Phase 0 finding that long-side edge concentrates 4-5x at skew>20 ")
    md.append("motivates a re-test.\n")
    md.append("Baseline (S0) is the invalidated historical V2 trade log on disk; ")
    md.append("alternatives are fresh runs with `use_instrument_routing` and the ")
    md.append("specified gating.\n")

    md.append("\n## 1. Headline results\n")
    summary_df = pd.DataFrame(cases_summaries)
    md.append(df_to_md(summary_df))
    md.append("")

    md.append("\n## 2. Per-year NC PnL (n_trades, total, per_trade)\n")
    for case, py in zip(CASES, cases_per_year):
        md.append(f"### {case.name} - {case.desc}\n")
        py2 = py.reset_index()
        md.append(df_to_md(py2))
        md.append("")

    md.append("\n## 3. Decision-gate scorecard (vs S0 baseline)\n")
    base = cases_summaries[0]
    base_per_year = cases_per_year[0]["net"]
    rows = []
    for cs, py in zip(cases_summaries[1:], cases_per_year[1:]):
        # Strict per-year: each year non-negative dPnL vs baseline
        merged = pd.concat({"alt": py["net"], "base": base_per_year}, axis=1).fillna(0)
        per_year_dpnl = merged["alt"] - merged["base"]
        n_pos = int((per_year_dpnl >= 0).sum())
        n_total = int(len(per_year_dpnl))
        rows.append({
            "name": cs["name"],
            "d_pnl": cs["nc_pnl"] - base["nc_pnl"],
            "d_trade_sharpe": round(cs["trade_sharpe"] - base["trade_sharpe"], 3),
            "d_daily_sharpe": round(cs["daily_sharpe"] - base["daily_sharpe"], 3),
            "d_max_dd": round(cs["max_dd"] - base["max_dd"], 0),
            "per_year_strict_pass": f"{n_pos}/{n_total}",
            "per_year_dpnl_min": round(per_year_dpnl.min(), 0) if len(per_year_dpnl) else None,
        })
    md.append(df_to_md(pd.DataFrame(rows)))
    md.append("")

    md.append("\n## 4. Routing-application stats (n_atm vs n_bear vs n_bull)\n")
    md.append("Confirms how often the routing gate fires per case.\n")
    routing_df = summary_df[["name", "n_atm", "n_bear", "n_bull", "n_trades"]].copy()
    md.append(df_to_md(routing_df))
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[phase1] wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
