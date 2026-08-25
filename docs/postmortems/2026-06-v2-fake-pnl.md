# V2 false-PnL incident

**Status:** historical V2 PnL/Sharpe **invalidated**; honest result
**negative**; the known false-PnL publication route is now blocked by a strict
gate.

## Summary

The 15-minute Deribit V2 backtest booked a vega leg from the change in
svi_iv_matched. That fitted series moved by roughly 9.58 volatility points over
15-minute intervals in the audit sample, while Deribit DVOL moved roughly
0.33–0.36 points.

The signal selected extreme fitted IV values. Subsequent calibration
mean-reversion was therefore recorded as profit even when no executable option
premium made the corresponding move. The backtest harvested surface-fit noise.

## Impact

The committed audit at bb509ad reported:

| Mark | Result | Interpretation |
| --- | ---: | --- |
| Legacy SVI-change mark | +$1,623,295; Sharpe 8.50 | Artifact |
| IV-frozen mark | -$586,939; Sharpe -20.13 | Negative check |
| Pinned-exit-IV mark | +$29,188; Sharpe 0.73 | No deployable evidence |
| Fourteen live paper trades | -$619 total; -$44/trade | Execution warning |

A later local Phase-6 run used different evaluation details and reported legacy
+$1.579M/8.41 versus DVOL-marked -$266K/-18.74, with its fill-covered subset
losing $18.8/trade. The exact current data-dependent baseline has since drifted
again while remaining negative; it is an unresolved reproducibility issue, not
a number to silently re-baseline.

All positive V2 PnL/Sharpe headlines and all choices justified only by that PnL
are invalid. This includes Kelly sizing, PnL-derived classifier labels, and
risk/filter changes selected from the fake trade log.

## Root cause

SVI is a cross-sectional surface fit. Its change between independently
calibrated bars is not an option fill and is especially unstable when trade
coverage changes. The system failed to keep these claims separate:

1. a useful fitted surface at one timestamp;
2. a forecast of future volatility;
3. an executable change in option premium.

The third was inferred from the first without independent price validation.

## What remains usable

The incident does not show that TSRV, feature engineering, forecast loss, or
SVI as a same-timestamp surface model is intrinsically invalid. Those assets do
not inherit validation either. Forecast QLIKE/R2 and any new options structure
must be reproduced against their own causal contracts.

## Corrective controls

- The backtester defaults to honest DVOL movement for the vega mark.
- Real-premium marking causally matches the same call and put contracts.
- Tier-1 real fills are separated from tier-2 published mark fallbacks.
- The strict gate requires a minimum tier-1 sample and coverage plus sign,
  magnitude, and per-trade correlation agreement.
- Historical V2 artifact regeneration executes that strict gate before
  writing.
- The former command-line override that could write a failed forensic artifact
  was removed.
- The weaker sign-only smoke check is documented as non-certifying.
- The paper CLI no longer presents V2 as a deployment preset or offers a
  real-order path.

## Remaining work

- Remove or retune historical parity settings selected against invalid PnL.
- Pin and hash data artifacts used by data-dependent regression baselines.
- Add independent option-price validation to every future PnL promotion path,
  not only the historical V2 script.
- Close the active-path issues in
  [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) with direct regressions.

## Evidence

- Commit bb509ad: honest-PnL audit and primary sign-reversal diagnosis.
- Current source: vol_arb/honest_pnl.py, vol_arb/backtester.py, and
  scripts/regenerate_historical_v2.py.
- Local ignored raw reports listed in [VALIDATION.md](../VALIDATION.md).

The numeric audit rows above describe their named historical runs. They are
not current performance claims.
