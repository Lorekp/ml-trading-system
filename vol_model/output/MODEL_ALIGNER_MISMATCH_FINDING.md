# Live engine vs aligner β* — model mismatch finding (2026-05-07)

## What this is

While auditing sidecars after the VPIN bug, I noticed the live engine
loads a different LightGBM model than the one the aligner β* was
calibrated against. The β* values are still applied as if the
prediction came from the calibrated model. This is a known-direction
miscalibration that has been live since the V2 preset was adopted.

## The two models

| Directory | Features | Walk-forward months | Latest training month | Engine config |
|---|---:|---:|---|---|
| `vol_model/output/models/expanding_pruned_svi_v1/` | 44 | 50 (2022-02 → **2026-04** as of this commit) | 2026-04 | **active** |
| `vol_model/output/models/expanding_pruned_svi_har_pd_v1/` | 50 | 51 (2022-02 → 2026-04) | 2026-04 | research-only |

The 50-feature variant adds 6 HAR-PD path-dependent features:
`har_pd_R1_a005`, `har_pd_R1_a01`, `har_pd_R1_a03`, `har_pd_R2_b005`,
`har_pd_R2_b01`, `har_pd_R2_b03`. Mean per-month QLIKE delta:

- `_pruned_svi_v1`: −10.0% vs baseline (44 feat, full WF)
- `_pruned_svi_har_pd_v1`: −12.7% vs baseline (50 feat, full WF)

So the HAR-PD variant is ~2.7pp better in QLIKE on average.

## The mismatch

`PIPELINE_DOCUMENTATION.md` line 11 declares the production model is
`_pruned_svi_har_pd_v1`. The V2 backtest's aligned predictions parquet
(`BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet`) is
generated from `_pruned_svi_har_pd_v1` raw predictions. The aligner
β* JSON is extracted from THAT parquet via:

```python
# scripts/extract_aligner_betas.py
β*_month = mean(y_pred - y_pred_raw) per month
```

where `y_pred_raw` and `y_pred` are both from `_pruned_svi_har_pd_v1`.

But `deribit_paper/config.py:183` pins:

```python
lgbm_models_dir: str = "vol_model/output/models/expanding_pruned_svi_v1"
```

So the live engine's call chain is:

1. **Predict raw**: `pred_ln_raw = _pruned_svi_v1.predict(features)` ← 44-feature model
2. **Apply bias**: `pred_ln = pred_ln_raw + β*_month` ← β* calibrated for `_har_pd_v1`
3. **Convert + signal**: `rv_pred = exp(pred_ln) × 100`, compare to IV, fire signal

Step 2 is the problem: the β* values were fit to a model that wasn't
the model used to produce `pred_ln_raw`. The two raw distributions
differ (different feature sets, different QLIKE optima); their
per-month residuals against `y_true` are ~70-90% correlated (same
target, similar architectures), so the misapplied β* is approximately
right but systematically off.

## How big is the bias?

Empirically, β* values clip to `[-0.5, +0.5]` (load_aligner_betas
enforces). The live engine's most recent month had β* = +0.047 (memory:
`project_prediction_aligner_2026_04_20.md`). Applied as `pred_ln_raw +
0.047`, this is a +5% shift in the volatility forecast.

The mismatch produces an EXTRA error of order:

```
err ≈ β*_har_pd − β*_v1 ≈ (μ_v1 − μ_har_pd) + (σ²_v1 − σ²_har_pd)
```

per the QLIKE-optimal scalar formula `β* = μ + σ²` (Gaussian residuals
case, from the same memory entry). Without re-extracting β* against
`_pruned_svi_v1`'s residuals, we can't quantify this exactly. Order of
magnitude: probably 0.005-0.02 in log-vol units (0.5-2% in volatility).

## Why this hasn't been caught

- Both models predict the same target (log-vol over the next 15min).
- Both have similar mean QLIKE.
- The β* miscalibration is small relative to other prediction errors.
- The V2 backtest uses the predictions parquet directly (no live
  inference path), so backtest numbers don't reflect this miscalibration.
- The live engine's PnL hasn't been compared in a parity-check that
  would surface the bias (the harness writer was added but no daily
  diff vs backtest replay yet).

## Why I didn't auto-fix

The cleanest fix is to switch the engine to `_pruned_svi_har_pd_v1`,
matching the documented production model and the aligner calibration.
This requires:

1. Update `deribit_paper/config.py:183` →
   `expanding_pruned_svi_har_pd_v1`.
2. **Implement live HAR-PD feature computation.** The 6 HAR-PD features
   are path-dependent and currently only exist in a separate sidecar
   (`vol_model/output/BTCUSDT_features_har_pd_15m.parquet`). The live
   engine has NO live HAR-PD computation — `FeatureAssembler.build_row`
   would receive NaN for the 6 slots, the predictor would return a flat
   signal ("feature pipeline returned NaN"), and the engine would never
   trade.
3. Add HAR-PD to the FeatureAssembler manifest, including a live
   computation function (similar to how live TSRV is computed).
4. Verify against the backtester's HAR-PD computation for byte-identity
   (the harness writer comparison would catch this).
5. Test, deploy, validate.

This is multi-day work + a new validation gate. Doing it as part of
fixing the staleness bug would be over-scope.

## Alternative fix — re-extract β* for `_pruned_svi_v1`

Smaller alternative: regenerate the β* JSON from `_pruned_svi_v1`
predictions instead of `_har_pd_v1` predictions. Steps:

1. Run `vol_model/extract_pruned_svi_predictions.py` to produce a
   `BTCUSDT_pruned_svi_predictions.parquet` from `_pruned_svi_v1`
   walk-forward outputs (this script exists; the output parquet is
   currently 26 days stale).
2. Run a v1-specific aligner step to bias-correct it (or extend
   `prediction_aligner.py` to handle the v1 source).
3. Run `scripts/extract_aligner_betas.py --input
   BTCUSDT_pruned_svi_predictions_aligned_bias.parquet` to produce a
   v1-calibrated β* JSON.
4. Point `cfg.aligner_betas_json` at the new JSON.

This is about half a day of work + a validation cycle.

## What I did fix (this commit)

The simpler, contained fix: brought the live engine's loaded LightGBM
model up to date by retraining the 2026-04 month in
`_pruned_svi_v1`. Result: `qlike=0.5283` vs baseline `0.7638`, **−30.8%
QLIKE delta**, RankIC 0.624 (in line with historical months). The
engine now loads `model=2026-04` after deployment. This does NOT fix
the model/aligner mismatch but removes the 1-month staleness on the
loaded model, which was the contributing reason this audit started.

## Recommended next step

Decide between:

- **Option A (proper)**: Implement live HAR-PD computation, switch
  engine to `_pruned_svi_har_pd_v1`. Aligns engine with docs and with
  aligner calibration. ~2-3 days.
- **Option B (smaller)**: Re-extract β* for `_pruned_svi_v1`. Removes
  the calibration mismatch without touching model variants. ~half day.
  Predictions parquet at `BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet`
  stays as the V2 backtest source (no impact on offline metrics).
- **Option C (defer)**: Accept the small known miscalibration. Add an
  explicit warning at engine startup that the loaded model differs
  from the aligner calibration source. Re-prioritize when live-vs-
  backtest harness reveals materiality.

My instinct: Option B. The bias is small but real, and re-extracting
β* against the engine's actual model is the cheapest way to close the
gap without a model migration.

## Files

- This finding: `vol_model/output/MODEL_ALIGNER_MISMATCH_FINDING.md`
- Retrain log: `vol_model/output/pruned_svi_v1_2026_04_retrain.log`
- New model files: `vol_model/output/models/expanding_pruned_svi_v1/2026-04.txt` + `_features.json`
- HAR-PD walk-forward (research-only, full WF): `vol_model/output/models/expanding_pruned_svi_har_pd_v1/`
- Aligner script: `scripts/extract_aligner_betas.py`
- HAR-PD sidecar: `vol_model/output/BTCUSDT_features_har_pd_15m.parquet`
