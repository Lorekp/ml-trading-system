# TransVol Optuna-best vs production defaults — cross-month validation

Optuna-best params (from sweep on 2024-08): `{"lr": 7.475992999956501e-05, "dropout": 0.19810364221551063, "weight_decay": 0.001238513729886093}`
Production defaults: `{"lr": 3e-05, "dropout": 0.15, "weight_decay": 0.01}`

Both trained with max_epochs=50, patience=12, identical train/val splits.

## Per-month QLIKE

| month | regime | default | optuna | delta | delta% | winner |
|---|---|---:|---:|---:|---:|---|
| 2022-06 | LUNA collapse aftermath + 3AC unwind | 0.4922 | 0.4884 | -0.0038 | -0.77% | optuna |
| 2022-11 | FTX collapse + contagion | 0.5799 | 0.5516 | -0.0283 | -4.88% | optuna |
| 2023-08 | mid-2023 range-bound calm | 0.5633 | 0.5601 | -0.0031 | -0.56% | optuna |
| 2023-12 | late-2023 ETF speculation | 0.6360 | 0.6216 | -0.0144 | -2.27% | optuna |
| 2024-04 | BTC halving cycle / transition | 0.4340 | 0.4267 | -0.0072 | -1.67% | optuna |
| 2024-08 | yen-carry unwind | 0.3270 | 0.3273 | +0.0003 | +0.10% | default |
| 2025-04 | recent vol shock | 0.2562 | 0.2489 | -0.0074 | -2.87% | optuna |
| 2026-01 | recent OOS | 0.3158 | 0.3067 | -0.0091 | -2.88% | optuna |

## Verdict
- Optuna wins in **7/8** months
- Default wins in **1/8** months
- Average delta: **-1.97%** (negative = Optuna better)
- Median delta: **-1.97%**

**Optuna params are robust** — win in ≥6/8 months, not just the sweep month.