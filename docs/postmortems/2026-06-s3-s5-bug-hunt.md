# S3/S5 bug-hunt and validation incident

**Status:** S3 baseline deployment validation **negative**; S3 E2/H2 rescue
**invalidated**; S5 headline **uncertain** pending a clean full rerun.

## Summary

The June bug hunt mixed genuine fixes, add-only patch helpers, incomplete
validation, and later attempts to rescue failed statistics. The raw reports are
useful forensic material, but their strongest deployment language is not
supported by the executable paths now present.

The main bug-fix report counted 37 critical/high issues in scope, described 22
as fixed and verified, and left 15 high issues skipped or test-only. Several
proposed fixes were add-only wrappers that production callers never imported.
Those wrappers and their wrapper-only tests were removed during consolidation;
their claimed defects remain in the canonical known-issues backlog.

## S3: what survived and what did not

The frozen S3 short-vol-to-expiry revalidation found a profitable real-fill
covered subset and passed its fill-realism and sizing checks. It nevertheless
failed the binding statistical gate:

- selection-bias lift: +65%;
- non-overlapping sample: 52 trades;
- deflated-Sharpe p(skill): 0.7168;
- required pass: greater than 0.99;
- final frozen-protocol verdict: **FAIL**.

Source: `vol_arb/output/S3_REVALIDATION.md:30-63`. This is raw evidence; the
canonical conclusion is that S3 is research-only and not deployment-validated.

### Invalid E2/H2 rescue

Later documents claimed the E2/H2 stack raised DSR to approximately 1.0. The
implementation does not simulate executable stops:

- `experiments/invalidated/s3_variants/D_statistical.py:71-88` computes sigma from the full
  sample and clips terminal losses to a chosen sigma threshold. It does not
  establish when the stop was touched, the fill price, or costs.
- `experiments/invalidated/s3_variants/H_tail_survival.py:97-139` caps terminal losses again and
  adds `0.5 * sigma` to every terminal winner above `1 * sigma`. It does not use
  an intratrade peak path to earn that addition.

Therefore `vol_arb/output/S3_GO_LIVE_SPEC.md` and
`vol_arb/output/S3_RESEARCH_VARIANT_REPORT.md` are withdrawn as result claims.
Any stop or trailing overlay must be rerun on causal intratrade marks with
train-only parameters, trigger timestamps, executable fills, and fees.

## S5: promising, not currently certified

The historical full-universe Binance result reported Sharpe 3.40 and positive
PnL in seven years. Current evidence does not establish that value on the final
code:

- `scripts/evaluate_s5_historical.py` now reports independent PnL realism as
  **not evaluated** because the replay has no separate execution/fill PnL
  series. Its sizing and selection-bias diagnostics remain useful, but they do
  not certify execution realism.
- `experiments/invalidated/s5_self_validating_variants.py` repeats that
  identity gate while applying proxy terminal-PnL post-processors. It is
  retained only as forensic provenance, not as a validation runner.
- `vol_arb/output/S3_S5_BUG_FIX_REPORT.md:13` says a full post-fix rerun is the
  binding validation gate and had not been run.
- `vol_arb/output/S5_REGRESSION_DIAGNOSIS.md:178-184` restores the small smoke
  run after a timestamp-normalization bug but still requires the full run; it
  also retracts the earlier “0 delta” statement.
- The correction in `src/signal_engine/portfolio.py:249-268` is present in the
  consolidated source, but a fresh full-universe result has not been persisted.

S5 may retain real potential, but the sober status is **Uncertain / paper-only**
until a clean full-universe artifact is produced from pinned code and its
execution assumptions are checked independently.

## Required closure

1. Reproduce each suspected defect against its canonical caller, implement the
   fix directly there, and add a mechanism-specific regression.
2. Run the full S5 evaluation from a clean commit and persist the command,
   input hashes, per-year table, costs, and result artifact.
3. Add an execution-ledger or independently reconstructed fill-PnL series before
   evaluating S5 PnL realism; the historical runner deliberately leaves that
   result unevaluated today.
4. Re-test S3 overlays with causal path simulation and then reapply the frozen
   DSR and selection-bias gates without relaxing thresholds.

Raw sources such as `vol_arb/output/S3_S5_BUG_FIX_REPORT.md`,
`vol_arb/output/MED_BUG_SPRINT_REPORT.md`, and
`vol_arb/output/S5_REGRESSION_DIAGNOSIS.md` are ignored generated evidence, not
canonical status documents.
