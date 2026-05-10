# Optuna-tuned SHAP audit (cycle 9, 2026-05-08)

Production-matched Optuna 200 trials / 50 patience per month. Supersedes the fixed-hyperparam audit at `shap_audit_6_regimes_*.csv`.

## Pre-OI run

| month | regime | n_train | n_test | test QLIKE | trials |
|---|---|---:|---:|---:|---:|
| 2022-06 | LUNA collapse aftermath + 3AC unwind (post-crash high vol) | 49,154 | 2,880 | 0.4929 | 154 |
| 2022-11 | FTX collapse + contagion (acute crisis) | 63,266 | 2,688 | 0.5931 | 162 |
| 2023-08 | mid-2023 range-bound calm (low realized vol) | 89,186 | 2,974 | 0.7731 | 200 |
| 2024-04 | BTC halving cycle / transition | 112,608 | 2,880 | 0.3494 | 94 |
| 2024-08 | yen-carry unwind (vol spike + cross-asset shock) | 124,319 | 2,976 | 0.4085 | 200 |
| 2026-01 | recent OOS (most recent regime baseline) | 174,047 | 2,976 | 0.5068 | 172 |

## Post-OI run

| month | regime | n_train | n_test | test QLIKE | trials |
|---|---|---:|---:|---:|---:|
| 2023-08 | mid-2023 range-bound calm (low realized vol) | 89,186 | 2,974 | 0.7731 | 200 |
| 2023-12 | late-2023 ETF-speculation / GBTC discount narrowing | 100,896 | 2,976 | 0.4538 | 113 |
| 2024-04 | BTC halving cycle / transition | 112,608 | 2,880 | 0.3494 | 94 |
| 2024-08 | yen-carry unwind (vol spike + cross-asset shock) | 124,319 | 2,976 | 0.4085 | 200 |
| 2025-04 | recent vol shock | 147,647 | 2,880 | 0.3162 | 141 |
| 2026-01 | recent OOS (most recent regime baseline) | 174,047 | 2,976 | 0.5068 | 172 |

## Union decision: KEEP=62 / total=126

Features that needed UNION (one-audit-only KEEP):

- `ffd_rv_ema50`: pre=KEEP (shap=0.0060) post=REVIEW (shap=0.0020)
- `vov_lag1`: pre=KEEP (shap=0.0030) post=REVIEW (shap=0.0020)
- `noise_level_lag3`: pre=REVIEW (shap=0.0029) post=KEEP (shap=0.0051)
- `parkinson_1d_lag1`: pre=REVIEW (shap=0.0028) post=KEEP (shap=0.0030)
- `imbalance_lag1`: pre=KEEP (shap=0.0022) post=REVIEW (shap=0.0020)
- `tsrv_ffd_ema10`: pre=KEEP (shap=0.0019) post=REVIEW (shap=0.0018)