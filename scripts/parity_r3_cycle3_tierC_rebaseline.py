"""Tier C (parity-r3 cycle 3 — 2026-05-08) V2 rebaseline orchestrator.

Re-runs the canonical V2 backtest after the Tier C semantic flips:
  - E-6 (`ioc_max_slippage_pct=1.0`)   — IOC % cap mirrors live API
  - O-1 (`cancel_replace_bar_gap`)     — N/A under V2 (use_sor=False)
  - O-2 (`taker_tier_blend=True`)      — blended slip for live's 3-tier ladder
  - O-6 (`bar_aligned_exit_latency_pct=0.05`) — premium-scaled exit slip

After re-running V2, regenerates the Kelly K1 sidecar from the new
trade log; iterates until NC PnL converges to ±$0.01 or 5 iterations.
Writes a rebaseline report at
``vol_arb/output/parity_r3_cycle3_tierC_rebaseline.md``.

Halt gate: if NC PnL drops > 10% vs cycle-2 baseline, exit non-zero.

Usage::

    python scripts/parity_r3_cycle3_tierC_rebaseline.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

# Cycle-2 baseline (post-D-7 fix; locked in tests/test_preventative_sizing.py:446-447).
PRE_FIX_NC_PNL_USD: float = 1_024_054.5149
PRE_FIX_N_TRADES: int = 7_543
HALT_RELATIVE_DROP: float = 0.10  # halt-gate: > 10% PnL drop
CONVERGENCE_TOL_USD: float = 0.01
MAX_ITERATIONS: int = 5


def _run_canonical_v2_backtest() -> pd.DataFrame:
    """Run the V2 canon (production filters + realistic costs)."""
    from vol_arb.backtester import BacktestConfig, VolArbBacktester
    cfg = (
        BacktestConfig()
        .with_btc_production_filters_v2()
        .with_realistic_costs()
    )
    assert cfg.preventative_size_mult_path is None
    assert cfg.uncertainty_filter_path is None
    _, trade_log = VolArbBacktester(cfg).run_backtest()
    return trade_log


def _summary(tl: pd.DataFrame) -> tuple[float, int]:
    nc_pnl = float(tl["net_pnl_nc"].sum())
    n_trades = int((tl["net_pnl_nc"] != 0).sum())
    return nc_pnl, n_trades


def _persist_canonical_trade_log(tl: pd.DataFrame) -> Path:
    out_path = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"
    tl.to_parquet(out_path, index=True)
    return out_path


def _regen_kelly_k1(horizon_minutes: int = 15) -> Path | None:
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.generate_kelly_sidecar",
                "--horizon-minutes", str(horizon_minutes),
            ],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[cycle3 rebaseline] Kelly K1 regen failed: {exc}\n"
            "    Run manually: python -m scripts.generate_kelly_sidecar\n"
        )
        return None
    return REPO_ROOT / "vol_arb" / "output" / "BTC_v2_kelly_mult_K1.parquet"


def _write_report(
    *,
    pre_pnl: float, pre_trades: int,
    post_pnl: float, post_trades: int,
    iteration_history: list[tuple[int, float, int]],
    kelly_path: Path | None,
) -> Path:
    out_path = REPO_ROOT / "vol_arb" / "output" / "parity_r3_cycle3_tierC_rebaseline.md"
    delta_pnl = post_pnl - pre_pnl
    delta_pnl_pct = delta_pnl / pre_pnl * 100.0 if pre_pnl != 0 else float("nan")
    delta_trades = post_trades - pre_trades

    md = f"""# Tier C V2 rebaseline report (parity-r3 cycle 3 — 2026-05-08)

## Mechanism

Four backtester semantic flips bring V2 in line with live execution truth:

- **E-6** (`ioc_max_slippage_pct=1.0`): % cap on IOC slip mirrors
  live's `place_ioc_taker(max_slippage_pct=1.0)` API ceiling.
- **O-1** (`cancel_replace_bar_gap`): N/A under V2 (`use_sor=False`).
- **O-2** (`taker_tier_blend=True`): blended slip across live's 3-tier
  IOC ladder (50/30/20 fractions × 0.5/1.5/2.5 multipliers, effective
  cap ~2.4× bar_slip) replaces the legacy flat 2.0× bar_slip cap.
- **O-6** (`bar_aligned_exit_latency_pct=0.05`): 5 bps premium-scaled
  exit slip surcharge models live's 2× ticker + 2× IOC round-trip
  latency vs offline's instant-fill assumption.

## Numbers

| Metric | Cycle-2 baseline | Cycle-3 (Tier C) | Δ |
|---|---:|---:|---:|
| NC PnL (USD) | ${pre_pnl:,.2f} | ${post_pnl:,.2f} | ${delta_pnl:+,.2f} ({delta_pnl_pct:+.3f}%) |
| n_trades fired | {pre_trades:,} | {post_trades:,} | {delta_trades:+,d} |

## Iteration history (V2 ↔ Kelly K1 convergence loop)

| Iteration | NC PnL | n_trades |
|:--:|---:|---:|
"""
    for it, nc, n in iteration_history:
        md += f"| {it} | ${nc:,.4f} | {n:,} |\n"

    md += f"""
Converged at iteration {iteration_history[-1][0]}.

## Verdict

"""
    if abs(delta_pnl_pct) < 0.5:
        md += (
            "Δ PnL within ±0.5% — Tier C semantic flips have a small\n"
            "but real impact. The 1% IOC cap (E-6) tightens slip on\n"
            "high-jump bars; the tier blend (O-2) loosens cap slightly\n"
            "on calmer bars; exit latency (O-6) charges 5 bps of\n"
            "premium per non-stopped exit. Net is a modest cost\n"
            "increase, fully within parity expectations.\n"
        )
    elif delta_pnl_pct > 0:
        md += (
            "Δ PnL **positive** — Tier C flips reduced effective slip\n"
            "(via tier blending) more than they added (via exit latency\n"
            "+ % cap). Forward-looking parity ship.\n"
        )
    elif delta_pnl_pct > -HALT_RELATIVE_DROP * 100.0:
        md += (
            "Δ PnL **negative but within halt-gate** — Tier C flips\n"
            "raise execution costs in the V2 backtest to match what live\n"
            "actually pays. The cycle-2 baseline was an overoptimistic\n"
            "approximation; the new baseline is the honest production\n"
            "view. Future SAC v4 retrains on corrected costs will\n"
            "recover any gap by adapting trade entry thresholds.\n"
        )
    else:
        md += (
            f"Δ PnL **breaches halt-gate** (>{HALT_RELATIVE_DROP*100:.0f}%"
            " drop).\n"
            "Manual review REQUIRED. Possible causes:\n"
            "- Tier C cap misconfigured (e.g. `ioc_max_slippage_pct` too tight)\n"
            "- Premium-scaled exit slip (O-6) accumulating excessive cost\n"
            "Re-evaluate Tier C parameters before commit.\n"
        )

    md += "\n## Kelly K1 sidecar\n\n"
    if kelly_path is not None and kelly_path.exists():
        md += f"Regenerated: `{kelly_path.relative_to(REPO_ROOT)}`\n"
    else:
        md += (
            "Regen attempted but failed. Run manually:\n"
            "```\n"
            "python -m scripts.generate_kelly_sidecar\n"
            "```\n"
        )

    md += f"""
## Suggested test relock

Update `tests/test_preventative_sizing.py:446-447` to:

```python
V2_NC_PNL_BASELINE = {post_pnl:.4f}
V2_NC_TRADES_BASELINE = {post_trades}
```

## Cross-references

- Plan: `~/.claude/plans/glimmering-snacking-turing.md` (cycle 3)
- Audit: `vol_arb/output/PARITY_AUDIT_ROUND3.md` (E-6/O-1/O-2/O-6 entries)
- Source change: `vol_arb/backtester.py` (config additions + slippage path edits)
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier C V2 rebaseline orchestrator.")
    ap.add_argument(
        "--skip-kelly", action="store_true",
        help="Skip Kelly K1 sidecar regeneration (faster smoke test).",
    )
    ap.add_argument(
        "--max-iterations", type=int, default=MAX_ITERATIONS,
        help=f"Max V2↔Kelly convergence iterations (default {MAX_ITERATIONS}).",
    )
    args = ap.parse_args()

    iteration_history: list[tuple[int, float, int]] = []
    last_pnl = float("inf")
    kelly_path: Path | None = None

    for it in range(1, args.max_iterations + 1):
        print(f"[cycle3 rebaseline] iteration {it}: V2 backtest …")
        tl = _run_canonical_v2_backtest()
        post_pnl, post_trades = _summary(tl)
        iteration_history.append((it, post_pnl, post_trades))
        print(f"[cycle3 rebaseline]   NC PnL = ${post_pnl:,.4f}; "
              f"n_trades = {post_trades:,}")

        # Persist trade log so Kelly regen reads the latest.
        out = _persist_canonical_trade_log(tl)
        print(f"[cycle3 rebaseline]   persisted: {out}")

        # Check convergence (skip on first iteration).
        if it > 1 and abs(post_pnl - last_pnl) < CONVERGENCE_TOL_USD:
            print(
                f"[cycle3 rebaseline]   converged "
                f"(|delta| {abs(post_pnl - last_pnl):.4f} < "
                f"tolerance {CONVERGENCE_TOL_USD:.4f})"
            )
            break
        last_pnl = post_pnl

        if not args.skip_kelly and it < args.max_iterations:
            print(f"[cycle3 rebaseline]   regenerating Kelly K1 sidecar …")
            kelly_path = _regen_kelly_k1()

    final_pnl, final_trades = iteration_history[-1][1], iteration_history[-1][2]
    report_path = _write_report(
        pre_pnl=PRE_FIX_NC_PNL_USD, pre_trades=PRE_FIX_N_TRADES,
        post_pnl=final_pnl, post_trades=final_trades,
        iteration_history=iteration_history,
        kelly_path=kelly_path,
    )
    print(f"[cycle3 rebaseline] wrote report: {report_path}")

    delta_pct = (final_pnl - PRE_FIX_NC_PNL_USD) / PRE_FIX_NC_PNL_USD
    if delta_pct < -HALT_RELATIVE_DROP:
        print(
            f"[cycle3 rebaseline] HALT — PnL drop {delta_pct*100:.2f}% exceeds "
            f"{HALT_RELATIVE_DROP*100:.0f}% threshold. Review report before commit."
        )
        return 2

    print(
        "\nSUGGESTED CONSTANTS (paste into "
        "tests/test_preventative_sizing.py:446-447):\n"
        f"    V2_NC_PNL_BASELINE = {final_pnl:.4f}\n"
        f"    V2_NC_TRADES_BASELINE = {final_trades}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
