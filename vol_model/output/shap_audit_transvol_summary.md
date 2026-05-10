# TransVol per-regime SHAP audit (production-matched)

Decision: KEEP=100 / REVIEW=1 / DROP=0

## Per-month metadata

| month | regime | n_test_windows | shap_s |
|---|---|---:|---:|
| 2022-06 | LUNA collapse aftermath + 3AC unwind | 512 | 68 |
| 2022-11 | FTX collapse + contagion | 512 | 69 |
| 2023-08 | mid-2023 range-bound calm | 512 | 67 |
| 2023-12 | late-2023 ETF speculation | 512 | 67 |
| 2024-04 | BTC halving cycle / transition | 512 | 67 |
| 2024-08 | yen-carry unwind | 512 | 67 |
| 2025-04 | recent vol shock | 512 | 67 |
| 2026-01 | recent OOS | 512 | 68 |

## Top 30 features by mean |SHAP|

| rank | feature | mean SHAP | best rank | worst rank | months>1% | decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | `ffd_rv_lag1` | 0.0044 | 1 | 35 | 8/8 | KEEP |
| 2 | `ffd_rv_lag3` | 0.0032 | 2 | 50 | 8/8 | KEEP |
| 3 | `duration_s_ema50` | 0.0030 | 1 | 63 | 8/8 | KEEP |
| 4 | `vpin_lag3` | 0.0028 | 1 | 71 | 8/8 | KEEP |
| 5 | `bpv_tsrv_ratio` | 0.0026 | 2 | 56 | 8/8 | KEEP |
| 6 | `ffd_rv_ema10` | 0.0025 | 2 | 83 | 8/8 | KEEP |
| 7 | `duration_s_lag3` | 0.0025 | 6 | 58 | 8/8 | KEEP |
| 8 | `ffd_rv` | 0.0024 | 1 | 79 | 8/8 | KEEP |
| 9 | `n_trades_ema50` | 0.0024 | 2 | 66 | 8/8 | KEEP |
| 10 | `up_jump_frac_ema10` | 0.0024 | 2 | 75 | 8/8 | KEEP |
| 11 | `time_to_funding_lag1` | 0.0022 | 5 | 75 | 8/8 | KEEP |
| 12 | `ffd_rv_ema50` | 0.0022 | 4 | 90 | 8/8 | KEEP |
| 13 | `tsrv_mean_1d_lag1` | 0.0022 | 2 | 75 | 8/8 | KEEP |
| 14 | `vpin` | 0.0022 | 4 | 72 | 8/8 | KEEP |
| 15 | `rv_zscore_1w_lag1` | 0.0021 | 7 | 88 | 8/8 | KEEP |
| 16 | `n_bars_in_window` | 0.0020 | 3 | 92 | 8/8 | KEEP |
| 17 | `down_jump_frac_ema10` | 0.0020 | 2 | 84 | 8/8 | KEEP |
| 18 | `rv_zscore_1d` | 0.0020 | 4 | 87 | 8/8 | KEEP |
| 19 | `down_jump_frac_lag3` | 0.0019 | 1 | 85 | 8/8 | KEEP |
| 20 | `duration_s` | 0.0019 | 1 | 70 | 8/8 | KEEP |
| 21 | `duration_s_ema10` | 0.0019 | 10 | 80 | 8/8 | KEEP |
| 22 | `tsrv_mean_1d` | 0.0019 | 9 | 92 | 8/8 | KEEP |
| 23 | `up_jump_frac_lag1` | 0.0019 | 4 | 64 | 8/8 | KEEP |
| 24 | `bpv_tsrv_ratio_ema10` | 0.0019 | 8 | 63 | 8/8 | KEEP |
| 25 | `hurst_100` | 0.0018 | 4 | 61 | 8/8 | KEEP |
| 26 | `imbalance_ema10` | 0.0018 | 3 | 80 | 8/8 | KEEP |
| 27 | `n_trades_ema10` | 0.0018 | 5 | 77 | 8/8 | KEEP |
| 28 | `time_to_funding_lag3` | 0.0018 | 3 | 87 | 8/8 | KEEP |
| 29 | `hurst_100_ema50` | 0.0018 | 4 | 85 | 8/8 | KEEP |
| 30 | `down_jump_frac_lag1` | 0.0017 | 14 | 67 | 8/8 | KEEP |