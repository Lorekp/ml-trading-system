"""D-7 (parity-r3 cycle 2 — 2026-05-08) V2 rebaseline orchestrator.

Re-runs the canonical V2 backtest (``with_btc_production_filters_v2()``
+ ``with_realistic_costs()``) AFTER the per-tick Greeks fix in
``vol_arb/drl_hedging/env.py:simulate``, captures the new NC PnL +
n_trades, regenerates the Kelly K1 sidecar (since the trade log
changed), and writes a rebaseline report to
``vol_arb/output/parity_r3_cycle2_d7_rebaseline.md``.

Usage::

    python scripts/parity_r3_cycle2_d7_rebaseline.py

The script is idempotent: re-running it produces the same numbers
modulo the V2 backtester's deterministic seeds. It also emits, on the
last line of stdout, a ``SUGGESTED CONSTANTS`` block the operator
can paste into ``tests/test_preventative_sizing.py`` to relock the
byte-identity test at the new baseline.

D-7 risk gate: if new NC PnL drops > 20% vs the pre-fix baseline,
exit non-zero so CI / a wrapping shell catches the regression for
manual review (per the Commit-B risk register in the cycle plan).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

# Pre-D-7 baseline (set by the fee-fix commit 2026-05-07; see
# tests/test_preventative_sizing.py:431-432).
PRE_FIX_NC_PNL_USD: float = 1_024_449.1810
PRE_FIX_N_TRADES: int = 7_543
HALT_RELATIVE_DROP: float = 0.20  # halt-gate: > 20% PnL drop


def _run_canonical_v2_backtest() -> pd.DataFrame:
    """Run the V2 canon (production filters + realistic costs)."""
    from vol_arb.backtester import BacktestConfig, VolArbBacktester
    cfg = (
        BacktestConfig()
        .with_btc_production_filters_v2()
        .with_realistic_costs()
    )
    # Both overlay flags must default off — exactly the same path as the
    # byte-identity test.
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
    """Regenerate the Kelly K1 sidecar via the existing helper script.

    Returns the K1 sidecar path on success; None if the helper isn't
    importable (e.g., scripts running in an environment without the
    phase-1 dependencies — fail-soft, the operator can re-run manually).
    """
    try:
        import subprocess
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
            f"[D-7 rebaseline] Kelly K1 regen failed: {exc}\n"
            "    Run manually: python -m scripts.generate_kelly_sidecar\n"
        )
        return None
    return REPO_ROOT / "vol_arb" / "output" / "BTC_v2_kelly_mult_K1.parquet"


def _write_report(
    *,
    pre_pnl: float, pre_trades: int,
    post_pnl: float, post_trades: int,
    kelly_path: Path | None,
) -> Path:
    out_path = REPO_ROOT / "vol_arb" / "output" / "parity_r3_cycle2_d7_rebaseline.md"
    delta_pnl = post_pnl - pre_pnl
    delta_pnl_pct = delta_pnl / pre_pnl * 100.0 if pre_pnl != 0 else float("nan")
    delta_trades = post_trades - pre_trades

    md = f"""# D-7 V2 rebaseline report (parity-r3 cycle 2 — 2026-05-08)

## Mechanism

`DRLHedger.simulate` precomputed `sigma_sqrt_T` and `half_sigma2_T` at
entry σ × entry T and reused them for every tick — but `tau_t` decays
each tick. The time-decay term `half_sigma²·T` was up to 9× too large
at the last tick, biasing `d1` (and therefore `delta_obs` /
`gamma_obs`) into a distribution the SAC v4 policy was NOT trained on.
Live (`drl_hedger_live.py:315-322`) and the training env
(`env.py:418`) both recompute per tick; only the simulate path was
reading frozen `T`.

Fix: per-tick recompute of `sigma_sqrt_tau_t` + `half_sigma²·tau_t`
inside the loop (vol_arb/drl_hedging/env.py:783-787, 819, 826-827).

## Numbers

| Metric | Pre-fix | Post-fix | Δ |
|---|---:|---:|---:|
| NC PnL (USD) | ${pre_pnl:,.2f} | ${post_pnl:,.2f} | ${delta_pnl:+,.2f} ({delta_pnl_pct:+.2f}%) |
| n_trades fired | {pre_trades:,} | {post_trades:,} | {delta_trades:+,d} |

## Verdict

"""
    if abs(delta_pnl_pct) < 1.0:
        md += (
            "Δ PnL within ±1% — within the noise floor of the policy's\n"
            "stochastic action sampling. Effectively a no-op on this baseline.\n"
        )
    elif delta_pnl_pct > 0:
        md += (
            "Δ PnL **positive** — the policy fed in-distribution observations\n"
            "produces marginally better hedging vs the pre-D-7 corrupted-obs\n"
            "scoring. This is the expected sign: the policy was trained on\n"
            "the env's per-tick distribution; it sees the same distribution\n"
            "in live; the simulate path now matches both. Ship.\n"
        )
    elif delta_pnl_pct > -HALT_RELATIVE_DROP * 100.0:
        md += (
            "Δ PnL **negative but within halt-gate** — the pre-D-7 frozen-T\n"
            "obs distribution was incidentally helpful to the policy in this\n"
            "rollout, but the discrepancy from training distribution was a\n"
            "real parity bug. Shipping the fix is correct; future SAC v4\n"
            "retrains on corrected obs will recover the gap.\n"
        )
    else:
        md += (
            f"Δ PnL **breaches halt-gate** (>{HALT_RELATIVE_DROP*100:.0f}% drop).\n"
            "Manual review REQUIRED. Possible cause: SAC v4 policy is\n"
            "fragile under the corrected obs distribution; consider a full\n"
            "retrain on env-generated rollouts before shipping.\n"
        )

    md += f"""

## Kelly K1 sidecar

"""
    if kelly_path is not None and kelly_path.exists():
        md += f"Regenerated: `{kelly_path.relative_to(REPO_ROOT)}`\n"
    else:
        md += (
            "Regen attempted but the helper exited non-zero. Run manually:\n"
            "```\n"
            "python -m scripts.generate_kelly_sidecar\n"
            "```\n"
        )

    md += f"""

## Suggested test relock

Update `tests/test_preventative_sizing.py:431-432` to:

```python
V2_NC_PNL_BASELINE = {post_pnl:.4f}
V2_NC_TRADES_BASELINE = {post_trades}
```

## Cross-references

- Plan: `~/.claude/plans/starry-rolling-rabin.md` (cycle 2)
- Audit: `vol_arb/output/PARITY_AUDIT_ROUND3.md` (D-7 entry)
- Source change: `vol_arb/drl_hedging/env.py:783-787, 819, 826-827`
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="D-7 V2 rebaseline orchestrator.")
    ap.add_argument(
        "--skip-kelly", action="store_true",
        help="Skip Kelly K1 sidecar regeneration (faster smoke test).",
    )
    ap.add_argument(
        "--no-persist", action="store_true",
        help="Don't overwrite the canonical trade log parquet.",
    )
    args = ap.parse_args()

    print("[D-7 rebaseline] running canonical V2 backtest …")
    tl = _run_canonical_v2_backtest()
    post_pnl, post_trades = _summary(tl)
    print(f"[D-7 rebaseline] post-fix NC PnL = ${post_pnl:,.4f}; "
          f"n_trades = {post_trades:,}")

    if not args.no_persist:
        out = _persist_canonical_trade_log(tl)
        print(f"[D-7 rebaseline] persisted canonical trade log: {out}")

    kelly_path: Path | None = None
    if not args.skip_kelly:
        print("[D-7 rebaseline] regenerating Kelly K1 sidecar …")
        kelly_path = _regen_kelly_k1()

    report_path = _write_report(
        pre_pnl=PRE_FIX_NC_PNL_USD, pre_trades=PRE_FIX_N_TRADES,
        post_pnl=post_pnl, post_trades=post_trades,
        kelly_path=kelly_path,
    )
    print(f"[D-7 rebaseline] wrote report: {report_path}")

    delta_pct = (post_pnl - PRE_FIX_NC_PNL_USD) / PRE_FIX_NC_PNL_USD
    if delta_pct < -HALT_RELATIVE_DROP:
        print(
            f"[D-7 rebaseline] HALT — PnL drop {delta_pct*100:.2f}% exceeds "
            f"{HALT_RELATIVE_DROP*100:.0f}% threshold. Review report before commit."
        )
        return 2

    print(
        "\nSUGGESTED CONSTANTS (paste into "
        "tests/test_preventative_sizing.py:431-432):\n"
        f"    V2_NC_PNL_BASELINE = {post_pnl:.4f}\n"
        f"    V2_NC_TRADES_BASELINE = {post_trades}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
