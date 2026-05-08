# Forward-looking improvement tickets

Tickets discovered during parity audits but explicitly OUT OF SCOPE for
parity work — they are within-system design decisions, not parity
divergences.

These should be revisited as standalone risk-management or modeling
improvements at the operator's discretion.

---

## R-3 — perp-leg notional inclusion in margin gate

**Source**: parity-r3 cycle 3 (2026-05-08), originally flagged in
`vol_arb/output/PARITY_AUDIT_ROUND3.md` Surface 5 R-3.

**Status**: REJECT in parity ledger (both systems consistent), OPEN as
a forward-looking design ticket.

### Observation

Both the live margin gate (`deribit_paper/risk/limits.py:165-169`) and
the offline backtester sizing (`vol_arb/sizing.py:31-55`) compute
margin notional from the **option leg only**. The perp delta-hedge
leg's notional is NOT counted.

```
margin_used_usd = im_fraction × (option_notional_btc × spot_usd)
```

The hedge inventory (`hedge_inventory_btc` in the position tracker) is
maintained but never feeds the margin envelope.

### Mechanism

A long 0.5 BTC option straddle + 0.25 BTC perp hedge → margin gate
treats this as 0.5 BTC notional (`im_fraction × 0.5 × spot`). Real
exchange margin requirements typically account for both legs (the
perp hedge offsets some delta risk but adds gross notional).

### Why it's not a parity divergence

Live and offline AGREE on the formula. "Fixing" live to include perp
would CREATE a divergence with the established offline V2 sizing,
breaking byte-identity at the K1 sidecar level. The fix would need
to land on BOTH sides simultaneously, which is a backtester semantic
flip plus a V2 rebaseline.

### Why it's worth revisiting

In stressed scenarios (sustained high-vol regime, large delta
deviations), the perp hedge can grow material relative to the option
leg. The current formula under-states margin used → gate clears
trades the operator wouldn't intend at high utilisation. Failure mode
is silent: trades fire that the operator's mental model says
shouldn't.

### Suggested forward-looking work

1. **Offline study**: instrument the V2 backtester to log
   `margin_with_hedge` vs `margin_option_only` per bar. Quantify the
   gap distribution under V2 production filters. If 99th percentile
   gap > 10% of equity, this is a real risk leakage.
2. **Decision**: should V2 sizing include perp leg in the margin
   formula? If yes, plan the V2 rebaseline (Kelly K1 + byte-identity
   relock) and the live `RiskLimits.check_signal` symmetric update.
3. **Live-only safety belt**: if (1) shows the gap matters but a full
   V2 rebaseline is expensive, ship a **monitoring-only** delta —
   live tracks `margin_with_hedge` and alerts when it diverges from
   `margin_option_only` by > X%; gates remain on the legacy formula
   until the rebaseline lands.

### Cost estimate

- Offline study: ~50 LOC + 1-day compute on V2 logs
- V2 rebaseline (option 2): ~30 LOC sizing edit + Kelly K1 regen +
  byte-identity relock (~3-iteration convergence loop)
- Live monitoring (option 3): ~20 LOC + Telegram alert wire-up

### Cross-references

- Parity audit (R-3): `vol_arb/output/PARITY_AUDIT_ROUND3.md`
- Live gate: `deribit_paper/risk/limits.py:163-186`
- Offline gate: `vol_arb/sizing.py:31-55`

---

## (Future tickets append below)
