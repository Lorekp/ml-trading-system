# D-7 V2 rebaseline report (parity-r3 cycle 2 — 2026-05-08)

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
| NC PnL (USD) | $1,024,449.18 | $1,024,054.51 | $-394.67 (-0.04%) |
| n_trades fired | 7,543 | 7,543 | +0 |

## Verdict

Δ PnL within ±1% — within the noise floor of the policy's
stochastic action sampling. Effectively a no-op on this baseline.


## Kelly K1 sidecar

Regenerated: `vol_arb\output\BTC_v2_kelly_mult_K1.parquet`


## Suggested test relock

Update `tests/test_preventative_sizing.py:431-432` to:

```python
V2_NC_PNL_BASELINE = 1024054.5149
V2_NC_TRADES_BASELINE = 7543
```

## Cross-references

- Plan: `~/.claude/plans/starry-rolling-rabin.md` (cycle 2)
- Audit: `vol_arb/output/PARITY_AUDIT_ROUND3.md` (D-7 entry)
- Source change: `vol_arb/drl_hedging/env.py:783-787, 819, 826-827`
