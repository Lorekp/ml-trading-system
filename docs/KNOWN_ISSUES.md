# Known issues

This is the canonical correctness backlog. An item is not fixed merely because
a helper, report, or isolated test demonstrates a proposed behavior.

## Blocking result promotion

| Area | Problem | Required closure |
| --- | --- | --- |
| Historical V2 replay | Kelly, aligned-prediction, filters, and risk settings were selected against invalidated V2 PnL. The preset is isolated from the CLI but remains available for forensic parity. | Never interpret the historical preset as a current paper profile; derive any future configuration from independently validated honest evidence. |
| S5 validation | No independent execution/cost series currently supports a PnL-realism gate. The historical evaluators now report this as not evaluated and block acceptance. | Add an independent check appropriate to the execution model and rerun the full universe. |
| Forecast metrics | Headline QLIKE/R2 and TransVol comparisons are not backed by one pinned, persisted current artifact. | Rerun corrected labels/folds, persist inputs and metrics, and update the ledger. |
| SVI feature artifacts | The current local `svi_features_15m.parquet` fails multiple locked integration gates: `svi_atm_spot_d1h` coverage is 56.7% vs 88%; `svi_atm_spot` coverage is 70.8% vs 80%; the spot series varies inside an hour and begins before its declared +1h end-shift floor. | Regenerate or repair the sidecar, audit its timestamp contract, and pass `tests/test_features_svi_surface.py` and `tests/test_svi_spot_feature.py` explicitly with the integration marker. Do not weaken the gates to bless the current artifact. |
| Honest V2 provenance | Historical honest reruns remain negative but drift because their large local inputs were not immutably identified. | Pin and hash input artifacts and store a separately versioned integration result; do not silently bless a new headline number. |

## Canonical-path defects requiring direct fixes

June research left add-only wrappers under vol_arb/_patches/. They were not
imported by maintained callers and therefore created false confidence. The
wrappers and wrapper-only tests are not part of the maintained system.

Before changing each canonical path, reproduce the defect and add a
mechanism-specific regression against the canonical module:

| ID | Suspected defect | Required invariant |
| --- | --- | --- |
| B-009 | A broad half-spread cap may hide bad reconstructed quotes. | Reject or explicitly classify unusable quotes; do not silently turn them into plausible fills. |
| B-010 | Non-finite funding rates can propagate into PnL and later disappear through nansum. | Validate funding on load and fail or explicitly mark coverage; never convert missing economics into zero implicitly. |
| B-012 | Real-fill coverage may include mark-price fallback tiers. | A real-fill result uses only the declared real-fill tier and reports the denominator. |
| B-014 | An enabled minute-vega stop can depend on a missing sidecar. | Validate required artifacts before the run starts; an enabled control must not degrade silently. |
| B-046 | Missing routed Greeks can mix two hedge conventions inside one result. | Select one declared hedge path for the run or fail coverage; never splice conventions bar by bar. |
| B-047 | Missing slippage-grid columns can trigger an undocumented dollar default. | Make required columns explicit and fail validation when absent. |
| B-048 | Backfilling the first lagged realized-vol value uses future data. | Use a predeclared initial value or start after the first observable lag; never bfill a causal lag. |
| B-049 | A recorded Kelly multiplier may describe the pre-cap rather than effective position size. | Derive diagnostics and NC sizing from final executed notional. |
| B-050–052 | SOR spread signs and maker-fill gate units may be inconsistent. | Express all costs from one order-side convention and compare quantities with identical units in replay tests. |
| B-053 | End-of-sample exits can be dropped; the proposed wrapper fabricated forward-filled future prices. | End the eligible entry window before unavailable exits or extend with real subsequent data—never synthetic fills. |

## Structural debt

- Separate data-dependent integration tests from fresh-clone unit tests.
- Move root workflow packages into one installable namespace in a dedicated
  refactor; do not add import shims.
- Move shared artifact/path resolution out of the vol_model to vol_arb
  dependency cycle.
- Curate the flat experiment tree only after recording references and
  provenance; generated outputs remain ignored.
- Migrate large local artifact roots only as a checksum-verified mechanical
  operation.

## Security follow-up

Hard-coded Deribit and Bybit demo credential values were removed from the
current tree during the 2026-08 cleanup. Both existed in Git history. Revoke or
rotate them; a cleanup commit does not erase prior objects or remote clones.
