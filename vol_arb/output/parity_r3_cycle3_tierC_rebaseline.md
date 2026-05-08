# Tier C V2 rebaseline report (parity-r3 cycle 3 — 2026-05-08)

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
| NC PnL (USD) | $1,024,054.51 | $1,010,891.50 | $-13,163.02 (-1.285%) |
| n_trades fired | 7,543 | 7,543 | +0 |

## Iteration history (V2 ↔ Kelly K1 convergence loop)

| Iteration | NC PnL | n_trades |
|:--:|---:|---:|
| 1 | $1,010,891.4966 | 7,543 |
| 2 | $1,010,891.4966 | 7,543 |

Converged at iteration 2.

## Verdict

Δ PnL **negative but within halt-gate** — Tier C flips
raise execution costs in the V2 backtest to match what live
actually pays. The cycle-2 baseline was an overoptimistic
approximation; the new baseline is the honest production
view. Future SAC v4 retrains on corrected costs will
recover any gap by adapting trade entry thresholds.

## Kelly K1 sidecar

Regenerated: `vol_arb\output\BTC_v2_kelly_mult_K1.parquet`

## Suggested test relock

Update `tests/test_preventative_sizing.py:446-447` to:

```python
V2_NC_PNL_BASELINE = 1010891.4966
V2_NC_TRADES_BASELINE = 7543
```

## Cross-references

- Plan: `~/.claude/plans/glimmering-snacking-turing.md` (cycle 3)
- Audit: `vol_arb/output/PARITY_AUDIT_ROUND3.md` (E-6/O-1/O-2/O-6 entries)
- Source change: `vol_arb/backtester.py` (config additions + slippage path edits)
