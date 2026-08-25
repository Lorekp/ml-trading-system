"""Testnet reconciliation — testnet-net PnL vs the backtest's twin.

Reads the ``FillRecord`` JSONL emitted by ``TestnetExecutor`` and produces a
single report comparing:

  - **Testnet-real PnL** (when the executor is wired with testnet creds)
  - **Paper-calibrated PnL** (the executor's internal fallback baseline)
  - **Historical replay reference** (the uncertified result reproduced by
    ``scripts/evaluate_s5_historical.py``)

Per-fill metrics reported:
  - slippage (bps, decision_mid → fill_price)
  - fee (bps, attributed by VIP2)
  - notional (USD)
  - method (maker / taker / paper_calibrated)

The reconciliation quantifies divergence between testnet fills and a historical
replay reference. It does not validate that reference, certify a Sharpe ratio,
or authorize deployment. Per-fill output exists so a running delta can be
inspected at any time.

Output: ``vol_arb/output/S5_TESTNET_RECONCILIATION.md``.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ── uncertified historical replay reference ──
# Reported 2026-05-18 result, 32-asset Binance, $250k sleeve,
# 7/7 positive years, non-compound, $1,707,603. This is a comparison anchor,
# not a validation result or deployment gate.
BACKTEST_NC_SHARPE = 3.40
BACKTEST_PNL_USD = 1_707_603.0
BACKTEST_MAXDD_PCT = 32.8
BACKTEST_AVG_COST_RT_BPS = 3.9   # historical VIP2 round-trip assumption
BACKTEST_PER_YEAR_USD = {
    2020: 280_979, 2021: 811_633, 2022: 16_563,
    2023: 247_601, 2024: 294_222, 2025: 162_632, 2026: 21_314,
}


@dataclass
class FillMetrics:
    """Aggregate fill-level metrics for one run."""
    n_fills: int = 0
    n_maker: int = 0
    n_taker: int = 0
    n_paper: int = 0
    n_skipped_poisoned: int = 0  # NEW (B-002): NaN/inf inputs counted but not summed
    sum_notional_usd: float = 0.0
    sum_slippage_bps: float = 0.0
    sum_fee_bps: float = 0.0
    sum_slippage_usd: float = 0.0    # slip_bps/1e4 * notional
    sum_fee_usd: float = 0.0
    by_method_slippage_bps: dict = field(default_factory=dict)
    by_symbol: dict = field(default_factory=dict)

    def add(self, rec: dict) -> None:
        method = rec.get("method", "?")
        slip_bps = float(rec.get("slippage_bps", 0.0) or 0.0)
        fee_bps = float(rec.get("fee_bps", 0.0) or 0.0)
        notional = float(rec.get("notional_usd", 0.0) or 0.0)
        fee_usd = float(rec.get("fee_usd", 0.0) or 0.0)
        # NEW (B-002): guard against NaN/inf poisoning the running sum. Without
        # this guard a single poisoned fill makes sum_slippage_usd NaN, which
        # propagates to cost_delta_bps -> adjusted_live_sharpe -> gate check
        # fails cryptically with no surface-level error.
        if not (math.isfinite(slip_bps) and math.isfinite(fee_bps)
                and math.isfinite(notional) and math.isfinite(fee_usd)):
            self.n_skipped_poisoned += 1
            logger.warning(
                "FillMetrics.add: skipped poisoned fill (non-finite inputs) "
                "symbol=%s method=%s slip_bps=%s notional=%s",
                rec.get("symbol", "?"), method, slip_bps, notional)
            return
        self.n_fills += 1
        if method == "maker":
            self.n_maker += 1
        elif method == "taker":
            self.n_taker += 1
        elif method == "paper_calibrated":
            self.n_paper += 1
        self.sum_notional_usd += notional
        self.sum_slippage_bps += slip_bps
        self.sum_fee_bps += fee_bps
        self.sum_slippage_usd += slip_bps / 1e4 * notional
        self.sum_fee_usd += fee_usd
        self.by_method_slippage_bps.setdefault(method, []).append(slip_bps)
        sym = rec.get("symbol", "?")
        b = self.by_symbol.setdefault(sym, {"n": 0, "notional": 0.0, "slip": 0.0, "fee": 0.0})
        b["n"] += 1
        b["notional"] += notional
        b["slip"] += slip_bps
        b["fee"] += fee_bps


def load_fills(path: Path) -> Iterable[dict]:
    """Read a JSONL fill log into a list of dicts."""
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def aggregate(fills: Iterable[dict]) -> FillMetrics:
    m = FillMetrics()
    for rec in fills:
        m.add(rec)
    return m


def reconcile(
    fills: Iterable[dict],
    backtest_nc_sharpe: float = BACKTEST_NC_SHARPE,
    backtest_avg_cost_rt_bps: float = BACKTEST_AVG_COST_RT_BPS,
) -> dict:
    """Compute the reconciliation numbers from a list of fills.

    Fill logs record one-way per-fill costs, while the historical assumption
    is round-trip. The comparison therefore doubles the mean per-fill cost.
    Returns a flat dict suitable for the Markdown report.
    """
    m = aggregate(fills)
    if m.n_fills == 0:
        return {
            "n_fills": 0,
            "n_skipped_poisoned": m.n_skipped_poisoned,
            "msg": "no fills recorded — driver may not have hit the bar close window, "
                   "or testnet is rate-limiting.",
        }

    avg_slip_bps = m.sum_slippage_bps / m.n_fills
    avg_fee_bps = m.sum_fee_bps / m.n_fills
    avg_total_one_way_bps = avg_slip_bps + avg_fee_bps
    avg_total_round_trip_bps = 2.0 * avg_total_one_way_bps
    slip_delta_bps = 2.0 * avg_slip_bps - 1.0
    cost_delta_bps = avg_total_round_trip_bps - backtest_avg_cost_rt_bps

    # Heuristic: each 1 bps of additional cost on $250k notional per year
    # (assume ~2 round-trips/day * 252 days = 504 RT/year) drags Sharpe by
    # ~0.04 in the historical replay. This is an unvalidated sensitivity
    # heuristic, not a forward Sharpe estimate.
    sharpe_drag_per_bps = 0.04
    est_sharpe_drag = max(0.0, cost_delta_bps) * sharpe_drag_per_bps
    adjusted_live_sharpe = max(0.0, backtest_nc_sharpe - est_sharpe_drag)

    return {
        "n_fills": m.n_fills,
        "n_maker": m.n_maker,
        "n_taker": m.n_taker,
        "n_paper": m.n_paper,
        "n_skipped_poisoned": m.n_skipped_poisoned,
        "sum_notional_usd": m.sum_notional_usd,
        "avg_slippage_bps": avg_slip_bps,
        "avg_fee_bps": avg_fee_bps,
        "avg_total_cost_bps_one_way": avg_total_one_way_bps,
        "avg_total_cost_bps_round_trip": avg_total_round_trip_bps,
        "slippage_delta_vs_backtest_bps": slip_delta_bps,
        "cost_delta_vs_backtest_bps": cost_delta_bps,
        "est_sharpe_drag_from_costs": est_sharpe_drag,
        "backtest_nc_sharpe": backtest_nc_sharpe,
        "backtest_avg_cost_rt_bps": backtest_avg_cost_rt_bps,
        "adjusted_live_sharpe": adjusted_live_sharpe,
        "sum_slippage_usd": m.sum_slippage_usd,
        "sum_fee_usd": m.sum_fee_usd,
    }


def render_report(
    summary: dict,
    fills: Iterable[dict],
    out_path: Path,
) -> None:
    """Render a single markdown report to ``out_path``."""
    m = aggregate(fills)
    lines = ["# S5 Testnet Reconciliation",
             "",
             "Driven by `src/live/testnet/cli.py` + `reconciler.py`.",
             "Per-fill JSONL → aggregated costs, slippage, fees.",
             ""]
    if not summary or summary.get("n_fills", 0) == 0:
        lines.append("> **No fills observed.** Driver did not execute any trades; "
                     "re-run after the engine's bar-close window triggers signals, "
                     "or after the testnet account is funded.")
        out_path.write_text("\n".join(lines))
        return

    lines += [
        "## Headline numbers",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Fills (n) | {summary['n_fills']:,} |",
        f"| Maker / Taker / Paper | {summary['n_maker']:,} / "
        f"{summary['n_taker']:,} / {summary['n_paper']:,} |",
        f"| Total notional | ${summary['sum_notional_usd']:,.0f} |",
        f"| Avg slippage (bps) | {summary['avg_slippage_bps']:.3f} |",
        f"| Avg fee (bps) | {summary['avg_fee_bps']:.3f} |",
        f"| Avg total cost, one way (bps) | {summary['avg_total_cost_bps_one_way']:.3f} |",
        f"| Avg total cost, round trip (bps) | {summary['avg_total_cost_bps_round_trip']:.3f} |",
        f"| **Historical round-trip assumption (bps)** | "
        f"**{summary['backtest_avg_cost_rt_bps']:.2f}** (VIP2 model) |",
        f"| **Cost delta vs backtest (bps)** | **{summary['cost_delta_vs_backtest_bps']:+.3f}** |",
        f"| **Slippage delta vs backtest (bps)** | "
        f"**{summary['slippage_delta_vs_backtest_bps']:+.3f}** (1.0 bps base) |",
        "",
        "## Historical sensitivity calculation",
        "",
        f"- Historical replay NC Sharpe: **{summary['backtest_nc_sharpe']:.2f}** "
        f"(reported 2026-05-18, 32-asset Binance, $250k sleeve; uncertified)",
        f"- Cost-drag heuristic: ~0.04 Sharpe per 1 bps of incremental cost on $250k "
        f"notional × ~504 RT/yr (conservative; based on 2 RT/day × 252 days).",
        f"- Estimated Sharpe drag from cost delta: "
        f"**{summary['est_sharpe_drag_from_costs']:.2f}**",
        f"- **Cost-adjusted historical sensitivity: "
        f"{summary['adjusted_live_sharpe']:.2f}**",
        "",
        "This heuristic does not validate the historical reference and is not "
        "a go/no-go or capital-allocation gate.",
        "",
    ]

    # Per-symbol breakdown.
    if m.by_symbol:
        lines += ["## Per-symbol breakdown (top 10 by notional)",
                  "",
                  "| Symbol | Fills | Notional ($) | Avg slip (bps) | Avg fee (bps) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        ranked = sorted(m.by_symbol.items(),
                        key=lambda kv: kv[1]["notional"], reverse=True)[:10]
        for sym, b in ranked:
            n = b["n"]
            slip = b["slip"] / n if n else 0.0
            fee = b["fee"] / n if n else 0.0
            lines.append(f"| {sym} | {n:,} | {b['notional']:,.0f} | "
                         f"{slip:.3f} | {fee:.3f} |")
        lines.append("")

    lines += [
        "## Mandate / next steps",
        "",
        "- Compare observed costs with the historical assumptions and document "
        "the difference.",
        "- Diagnose material differences in the fill model, fee tier, slippage "
        "model, universe, or signal source before further research claims.",
        "- Do not use this report to authorize deployment or capital allocation.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Reconcile testnet fills vs backtest")
    p.add_argument("--fills", default="vol_arb/output/S5_TESTNET_FILLS.jsonl")
    p.add_argument("--out", default="vol_arb/output/S5_TESTNET_RECONCILIATION.md")
    args = p.parse_args()
    fills = load_fills(Path(args.fills))
    summary = reconcile(fills)
    render_report(summary, fills, Path(args.out))
    print(f"wrote {args.out} ({len(fills)} fills)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
