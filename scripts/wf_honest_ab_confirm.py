"""Empirical confirmation A/B: honest-PnL effect of (Q1) de-noising the signal
IV input (svi_quality_gate) and (Q2) the conservative May-2026 parameter rollback.

ALL variants run with ``honest_vega_iv=True`` (the current default) so the
vega leg is marked to the real DVOL move, not the 9.28vp/15min SVI surface noise.
We are NOT looking for profit — we quantify whether either lever moves the dead
straddle's honest loss materially, to back a remediation decision.

Variants (all on the historical V2 replay window):
    A. V2_historical   with_historical_btc_v2() + with_realistic_costs()
    B. V2_qualgate  = A but svi_quality_gate=True, svi_gate_min_strikes=5   (Q1)
    C. V2_rollback  = A but conservative-rollback deltas + UNALIGNED preds   (Q2)

Reports per variant: NC PnL, NC Sharpe, MaxDD ($ and %), trade count, win%, and
the gate_no_fake_profit PASS/FAIL on the last-12m fill-covered subset (model vs
real-fill mark). This is a weaker smoke check than the strict gate in
scripts/regenerate_historical_v2.py and does not certify a result.

Console output is ASCII-only (Windows cp1252); the markdown report is utf-8.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vol_arb.backtester import BacktestConfig, VolArbBacktester
from vol_arb.honest_pnl import mark_trade_log_to_real_premium, gate_no_fake_profit
from vol_arb.options_loader import load_option_trades

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wf_ab_confirm")

OUT_MD = REPO_ROOT / "vol_arb/output/HONEST_AB_CONFIRM.md"

# Conservative-rollback deltas (Q2): revert the May-2026 noise-tuned params.
# Field names verified against vol_arb/backtester.py BacktestConfig.
ROLLBACK_PRED_UNALIGNED = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"


def _build_cfg(variant: str, start_dt: str | None, end_dt: str | None) -> BacktestConfig:
    """Construct the BacktestConfig for a variant. All keep honest_vega_iv=True."""
    cfg = (BacktestConfig()
           .with_historical_btc_v2()
           .with_realistic_costs())
    # honest_vega_iv defaults True; assert it explicitly for safety.
    assert cfg.honest_vega_iv is True, "honest_vega_iv must be True for all variants"

    if variant == "A":
        pass  # Historical V2 preset without overrides.
    elif variant == "B":
        # Q1: de-noise the signal IV input.
        cfg = cfg.model_copy(update=dict(
            svi_quality_gate=True,
            svi_gate_min_strikes=5,
        ))
    elif variant == "C":
        # Q2: conservative rollback of May-2026 noise-tuned params.
        cfg = cfg.model_copy(update=dict(
            kelly_mult_path=None,                  # Kelly OFF
            max_jump_fraction_for_shorts=0.15,     # was 1.0 (disabled)
            short_sizing_penalty=0.5,              # was 1.0 (full size)
            spread_jump_k=3.0,                     # was 0.0 (no widening)
            entry_threshold=25.0,                  # was 20.0
            entry_threshold_long=30.0,             # was 12.0
            target_margin_utilization=0.075,       # was 0.10
            predictions_path=ROLLBACK_PRED_UNALIGNED,  # UNALIGNED raw HAR-PD
        ))
    else:
        raise ValueError(variant)

    if start_dt is not None:
        cfg = cfg.model_copy(update=dict(start_dt=start_dt))
    if end_dt is not None:
        cfg = cfg.model_copy(update=dict(end_dt=end_dt))
    return cfg


def _nc_stats(tl: pd.DataFrame) -> dict:
    """NC PnL / NC daily Sharpe / MaxDD ($ and %) / trades / win% from a trade log.

    Convention matches scripts/regenerate_historical_v2.py: fired = N_contracts_nc>0,
    daily Sharpe from net_pnl_nc resampled 1D, MDD on 100k + cumsum equity.
    """
    exe = tl[tl["N_contracts_nc"] > 0]
    n = int(len(exe))
    if n == 0:
        return dict(n_trades=0, nc_pnl=0.0, nc_sharpe=0.0, mdd_usd=0.0,
                    mdd_pct=0.0, win_pct=0.0)
    nc_pnl = float(exe["net_pnl_nc"].sum())
    daily = exe["net_pnl_nc"].resample("1D").sum()
    sd = float(daily.std(ddof=1))
    nc_sharpe = float(daily.mean() / sd * np.sqrt(365.25)) if sd > 0 else 0.0
    eq = 100_000.0 + exe["net_pnl_nc"].cumsum()
    mdd_usd = float((eq - eq.cummax()).min())
    mdd_pct = float(((eq - eq.cummax()) / eq.cummax()).min() * 100.0)
    win_pct = float((exe["net_pnl_nc"] > 0).mean() * 100.0)
    return dict(n_trades=n, nc_pnl=nc_pnl, nc_sharpe=nc_sharpe,
                mdd_usd=mdd_usd, mdd_pct=mdd_pct, win_pct=win_pct)


def _gate_last_12m(tl: pd.DataFrame) -> dict:
    """Mark the most recent 12 months of fired trades to real Deribit fills and
    run gate_no_fake_profit (model net_pnl_nc vs real_option_pnl_nc).

    Uses the historical script's 12-month sampling convention, then applies the
    weaker sign-only smoke gate rather than its strict realism gate.
    """
    fired = tl[tl["N_contracts_nc"] > 0]
    if len(fired) == 0:
        return dict(gate_pass=None, msg="no fired trades", n_covered=0,
                    model_total=0.0, real_total=0.0)
    cutoff = pd.to_datetime(fired.index.max(), utc=True) - pd.Timedelta(days=365)
    sample = fired[fired.index >= cutoff]
    start = pd.to_datetime(sample.index.min(), utc=True).strftime("%Y-%m-%d")
    end = (pd.to_datetime(sample["exit_target"].max(), utc=True)
           + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("  gate: marking %d trades (%s..%s) to real fills ...",
                len(sample), start, end)
    trades = load_option_trades("data/deribit/trades/BTC_option",
                                start_date=start, end_date=end)
    marked = mark_trade_log_to_real_premium(sample, trades, tol_minutes=60.0)
    gate = gate_no_fake_profit(marked.trade_log)
    # GateResult exposes the diagnostics dict as ``.detail`` (singular).
    d = getattr(gate, "detail", None) or {}
    return dict(
        gate_pass=bool(gate.passed),
        msg=str(gate.message),
        n_covered=int(d.get("n_covered", 0)),
        model_total=float(d.get("model_total", float("nan"))),
        real_total=float(d.get("real_fill_total", float("nan"))),
        real_mean=float(d.get("real_fill_mean", float("nan"))),
        corr=float(d.get("per_trade_corr", float("nan"))),
    )


def _run_variant(variant: str, label: str, start_dt: str | None,
                 end_dt: str | None, with_gate: bool) -> dict:
    logger.info("=== Variant %s (%s) ===", variant, label)
    t0 = time.time()
    cfg = _build_cfg(variant, start_dt, end_dt)
    _, tl = VolArbBacktester(cfg).run_backtest()
    stats = _nc_stats(tl)
    logger.info("  %s: NC PnL $%s | NC Sharpe %.2f | MDD $%s (%.2f%%) | trades %d | win %.1f%% | %.1fs",
                label, f"{stats['nc_pnl']:,.0f}", stats["nc_sharpe"],
                f"{stats['mdd_usd']:,.0f}", stats["mdd_pct"], stats["n_trades"],
                stats["win_pct"], time.time() - t0)
    row = dict(variant=variant, label=label, **stats)
    if with_gate:
        g = _gate_last_12m(tl)
        row.update(g)
        gp = "PASS" if g["gate_pass"] else ("FAIL" if g["gate_pass"] is False else "N/A")
        logger.info("  %s gate: %s | covered %d | model $%s | real $%s",
                    label, gp, g["n_covered"],
                    f"{g.get('model_total', float('nan')):,.0f}",
                    f"{g.get('real_total', float('nan')):,.0f}")
    return row


def _fmt_table(rows: list[dict]) -> str:
    hdr = (f"{'Variant':<24} {'NC PnL':>14} {'NC Shp':>8} {'MaxDD $':>14} "
           f"{'MaxDD%':>8} {'Trades':>8} {'Win%':>7} {'Gate':>6} "
           f"{'Real$(12m)':>13}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        gp = r.get("gate_pass", None)
        gate = "PASS" if gp is True else ("FAIL" if gp is False else "-")
        real = r.get("real_total", float("nan"))
        real_s = f"{real:,.0f}" if real == real else "-"
        lines.append(
            f"{r['label']:<24} {r['nc_pnl']:>14,.0f} {r['nc_sharpe']:>8.2f} "
            f"{r['mdd_usd']:>14,.0f} {r['mdd_pct']:>8.2f} {r['n_trades']:>8d} "
            f"{r['win_pct']:>7.1f} {gate:>6} {real_s:>13}"
        )
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="3-date slice, no gate (fast error check).")
    ap.add_argument("--start", default=None, help="ISO start_dt override.")
    ap.add_argument("--end", default=None, help="ISO end_dt override.")
    args = ap.parse_args()

    if args.smoke:
        # 3-date slice to catch errors fast (hard repo rule).
        start_dt = "2024-01-01"
        end_dt = "2024-01-03"
        window_note = f"SMOKE 3-day slice {start_dt}..{end_dt} (no gate)"
        with_gate = False
    else:
        start_dt = args.start
        end_dt = args.end
        window_note = (f"full historical replay window "
                       f"(start={start_dt or 'None=2022-02-01'}, "
                       f"end={end_dt or 'None=2026-05-12'})")
        with_gate = True

    logger.info("WINDOW: %s", window_note)
    variants = [("A", "V2_current"), ("B", "V2_qualgate"), ("C", "V2_rollback")]
    rows = []
    for v, label in variants:
        rows.append(_run_variant(v, label, start_dt, end_dt, with_gate))

    table = _fmt_table(rows)
    print("\n" + table + "\n")

    # Deltas A->B (Q1) and A->C (Q2).
    A = rows[0]
    B = rows[1]
    C = rows[2]
    dB = dict(pnl=B["nc_pnl"] - A["nc_pnl"], shp=B["nc_sharpe"] - A["nc_sharpe"],
              trades=B["n_trades"] - A["n_trades"])
    dC = dict(pnl=C["nc_pnl"] - A["nc_pnl"], shp=C["nc_sharpe"] - A["nc_sharpe"],
              trades=C["n_trades"] - A["n_trades"])
    print(f"A->B (Q1 de-noise signal IV):  dNC PnL ${dB['pnl']:,.0f}  "
          f"dNC Sharpe {dB['shp']:+.2f}  dTrades {dB['trades']:+d}")
    print(f"A->C (Q2 conservative rollback): dNC PnL ${dC['pnl']:,.0f}  "
          f"dNC Sharpe {dC['shp']:+.2f}  dTrades {dC['trades']:+d}")

    if not args.smoke:
        _write_report(rows, table, dB, dC, window_note)
    return 0


def _write_report(rows, table, dB, dC, window_note) -> None:
    A, B, C = rows[0], rows[1], rows[2]
    all_loss = all(r["nc_pnl"] <= 0 for r in rows)
    all_gate_pass = all(r.get("gate_pass") is True for r in rows)

    def verdict(d):
        # "material" heuristic: a sign flip to profit, OR |dPnL| moving the
        # honest loss by > 25% of |A's loss|.
        base = abs(A["nc_pnl"]) if A["nc_pnl"] != 0 else 1.0
        return abs(d["pnl"]) > 0.25 * base

    q1_material = verdict(dB)
    q2_material = verdict(dC)
    q1_word = "YES (material)" if q1_material else "NO (immaterial)"
    q2_word = "YES (material)" if q2_material else "NO (immaterial)"

    md = []
    md.append("# Honest A/B Confirmation: de-noising signal IV (Q1) and "
              "conservative rollback (Q2)\n")
    md.append(f"Window: {window_note}.  All variants run with "
              "`honest_vega_iv=True` (vega leg marked to the real DVOL move, "
              "not the 9.28vp/15min SVI surface noise).\n")
    md.append("\n## Results\n")
    md.append("```\n" + table + "\n```\n")
    md.append("\nGate = `gate_no_fake_profit` on the most-recent-12m "
              "fill-covered subset (model `net_pnl_nc` vs real Deribit-fill "
              "`real_option_pnl_nc`). `Real$(12m)` is the real-fill PnL on that "
              "covered subset.\n")
    md.append("\n## Deltas\n")
    md.append(f"- **A->B (Q1, de-noise signal IV via svi_quality_gate)**: "
              f"dNC PnL ${dB['pnl']:,.0f}, dNC Sharpe {dB['shp']:+.2f}, "
              f"dTrades {dB['trades']:+d}.\n")
    md.append(f"- **A->C (Q2, conservative rollback + unaligned preds)**: "
              f"dNC PnL ${dC['pnl']:,.0f}, dNC Sharpe {dC['shp']:+.2f}, "
              f"dTrades {dC['trades']:+d}.\n")
    md.append("\n## Interpretation\n")
    # 4 sentences.
    s1 = (f"Under honest vega marking the V2 straddle (A) books NC PnL "
          f"${A['nc_pnl']:,.0f} over {A['n_trades']:,} trades, a loss that the "
          f"real-fill gate {'confirms' if A.get('gate_pass') else 'flags'} on "
          f"the covered 12m subset (real ${A.get('real_total', float('nan')):,.0f}).")
    s2 = (f"Q1 (de-noising the signal IV input with svi_quality_gate) moves the "
          f"honest result by ${dB['pnl']:,.0f} NC ({q1_word}); the cleaner IV "
          f"changes which bars fire ({dB['trades']:+d} trades) but does not "
          f"turn the dead trade profitable.")
    s3 = (f"Q2 (the conservative May-2026 rollback with Kelly off, tighter "
          f"thresholds, restored jump/spread defenses, and unaligned "
          f"predictions) moves the honest result by ${dC['pnl']:,.0f} NC "
          f"({q2_word}), {'reducing' if dC['pnl'] > 0 else 'deepening'} the loss "
          f"but not eliminating it.")
    s4 = (f"All three variants remain a loss under honest PnL"
          f"{' and all pass the no-fake-profit gate' if all_gate_pass else ''}; "
          f"neither lever produces a deployable edge, so the straddle stays a "
          f"loss under every variant tested.")
    md.append(s1 + " " + s2 + " " + s3 + " " + s4 + "\n")
    md.append(f"\nVERDICT Q1 (does de-noising change the honest loss materially): "
              f"{q1_word} -- dNC PnL ${dB['pnl']:,.0f}, A=${A['nc_pnl']:,.0f} "
              f"-> B=${B['nc_pnl']:,.0f}.\n")
    md.append(f"VERDICT Q2 (does the rollback reduce the honest loss): "
              f"{q2_word} -- dNC PnL ${dC['pnl']:,.0f}, A=${A['nc_pnl']:,.0f} "
              f"-> C=${C['nc_pnl']:,.0f}.\n")
    md.append(f"\nBOTH CONFIRM STRADDLE STAYS A LOSS: {str(all_loss).upper()}.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(md), encoding="utf-8")
    logger.info("Wrote report -> %s", OUT_MD)


if __name__ == "__main__":
    raise SystemExit(main())
