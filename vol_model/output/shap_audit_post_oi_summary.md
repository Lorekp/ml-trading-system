# Multi-Regime SHAP Audit (6 months)

## Months audited

| month | regime | n_train | n_test | test QLIKE |
|---|---|---:|---:|---:|
| 2023-08 | mid-2023 range-bound calm (low realized vol) | 89,186 | 2,974 | 0.9777 |
| 2023-12 | late-2023 ETF-speculation / GBTC discount narrowing | 100,896 | 2,976 | 0.4621 |
| 2024-04 | BTC halving cycle / transition | 112,608 | 2,880 | 0.3527 |
| 2024-08 | yen-carry unwind (vol spike + cross-asset shock) | 124,319 | 2,976 | 0.4145 |
| 2025-04 | recent vol shock (regime characterised post-hoc) | 147,647 | 2,880 | 0.3319 |
| 2026-01 | recent OOS (most recent regime baseline) | 174,047 | 2,976 | 0.5172 |

## Decision summary
- KEEP: 50 features (median rank <=35 OR best rank <=25 OR non-trivial SHAP in >=4 of 6 months)
- DROP: 36 features (best rank >60 AND non-trivial in zero months)
- REVIEW: 40 features (judgment call — see CSV)

## Top 50 features by mean SHAP across regimes

| rank | feature | mean SHAP | median rank | best rank | worst rank | months >1% | decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `duration_s_ema50` | 0.2169 | 1 | 1 | 1 | 6/6 | KEEP |
| 2 | `n_bars_in_window` | 0.1232 | 2 | 2 | 3 | 6/6 | KEEP |
| 3 | `vov_ema50` | 0.1182 | 3 | 2 | 4 | 6/6 | KEEP |
| 4 | `n_trades_ema50` | 0.0539 | 4 | 3 | 6 | 6/6 | KEEP |
| 5 | `duration_s_ema10` | 0.0495 | 5 | 4 | 5 | 6/6 | KEEP |
| 6 | `vov_ema10` | 0.0289 | 6 | 6 | 9 | 6/6 | KEEP |
| 7 | `tsrv_mean_1w` | 0.0204 | 9 | 6 | 12 | 6/6 | KEEP |
| 8 | `noise_level_ema50` | 0.0200 | 9 | 7 | 17 | 6/6 | KEEP |
| 9 | `tsrv_mean_1w_lag1` | 0.0187 | 20 | 4 | 61 | 5/6 | KEEP |
| 10 | `noise_level` | 0.0183 | 9 | 7 | 18 | 6/6 | KEEP |
| 11 | `tsrv_mean_1d` | 0.0153 | 11 | 9 | 19 | 6/6 | KEEP |
| 12 | `vpin_ema50` | 0.0148 | 11 | 8 | 15 | 6/6 | KEEP |
| 13 | `noise_level_lag1` | 0.0147 | 18 | 7 | 61 | 5/6 | KEEP |
| 14 | `tsrv_mean_1d_lag3` | 0.0128 | 15 | 9 | 23 | 6/6 | KEEP |
| 15 | `tsrv_mean_1d_ema50` | 0.0126 | 15 | 7 | 26 | 6/6 | KEEP |
| 16 | `time_to_funding_lag1` | 0.0125 | 13 | 9 | 19 | 6/6 | KEEP |
| 17 | `noise_level_ema10` | 0.0111 | 19 | 10 | 30 | 6/6 | KEEP |
| 18 | `parkinson_1d` | 0.0100 | 16 | 10 | 30 | 6/6 | KEEP |
| 19 | `vpin_ema10` | 0.0088 | 18 | 13 | 44 | 6/6 | KEEP |
| 20 | `tsrv_mean_1w_ema50` | 0.0087 | 25 | 10 | 51 | 6/6 | KEEP |
| 21 | `time_to_funding` | 0.0081 | 26 | 10 | 42 | 6/6 | KEEP |
| 22 | `n_trades_ema10` | 0.0079 | 20 | 17 | 31 | 6/6 | KEEP |
| 23 | `vpin` | 0.0078 | 25 | 13 | 39 | 6/6 | KEEP |
| 24 | `duration_s` | 0.0076 | 22 | 20 | 25 | 6/6 | KEEP |
| 25 | `vpin_lag1` | 0.0070 | 26 | 13 | 37 | 6/6 | KEEP |
| 26 | `time_to_funding_ema50` | 0.0064 | 26 | 17 | 40 | 6/6 | KEEP |
| 27 | `ffd_rv_lag1` | 0.0061 | 33 | 12 | 66 | 5/6 | KEEP |
| 28 | `imbalance_ema50` | 0.0060 | 29 | 15 | 48 | 5/6 | KEEP |
| 29 | `tsrv_mean_1d_ema10` | 0.0059 | 28 | 24 | 33 | 6/6 | KEEP |
| 30 | `tsrv_mean_1d_lag1` | 0.0057 | 26 | 17 | 54 | 5/6 | KEEP |
| 31 | `parkinson_1d_ema50` | 0.0052 | 36 | 20 | 42 | 5/6 | KEEP |
| 32 | `vpin_lag3` | 0.0046 | 42 | 22 | 45 | 6/6 | KEEP |
| 33 | `down_jump_frac_ema50` | 0.0044 | 34 | 24 | 50 | 5/6 | KEEP |
| 34 | `time_to_funding_lag3` | 0.0043 | 35 | 29 | 51 | 6/6 | KEEP |
| 35 | `tsrv_ffd` | 0.0042 | 34 | 31 | 49 | 5/6 | KEEP |
| 36 | `parkinson_1d_ema10` | 0.0042 | 37 | 23 | 48 | 6/6 | KEEP |
| 37 | `vov` | 0.0041 | 40 | 25 | 53 | 6/6 | KEEP |
| 38 | `bpv_tsrv_ratio_ema50` | 0.0041 | 39 | 21 | 79 | 3/6 | KEEP |
| 39 | `vov_lag3` | 0.0039 | 46 | 21 | 52 | 6/6 | KEEP |
| 40 | `duration_s_lag1` | 0.0038 | 41 | 29 | 49 | 5/6 | KEEP |
| 41 | `tsrv_mean_1w_lag3` | 0.0036 | 45 | 21 | 74 | 4/6 | KEEP |
| 42 | `time_to_funding_ema10` | 0.0033 | 43 | 31 | 56 | 5/6 | KEEP |
| 43 | `tsrv_ffd_lag1` | 0.0032 | 42 | 29 | 62 | 4/6 | KEEP |
| 44 | `up_jump_frac_ema50` | 0.0031 | 48 | 30 | 57 | 5/6 | KEEP |
| 45 | `tsrv_mean_1w_ema10` | 0.0031 | 53 | 22 | 86 | 4/6 | KEEP |
| 46 | `ffd_rv` | 0.0029 | 49 | 29 | 73 | 4/6 | KEEP |
| 47 | `rv_zscore_1d` | 0.0027 | 56 | 23 | 74 | 4/6 | KEEP |
| 48 | `bpv_tsrv_ratio` | 0.0026 | 52 | 35 | 64 | 3/6 | REVIEW |
| 49 | `duration_s_lag3` | 0.0026 | 47 | 40 | 73 | 3/6 | REVIEW |
| 50 | `up_jump_frac` | 0.0026 | 46 | 36 | 85 | 2/6 | REVIEW |

## Bottom 30 features by mean SHAP

| rank | feature | mean SHAP | median rank | best rank | worst rank | months >1% | decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 97 | `hurst_100_ema10` | 0.000542 | 87 | 78 | 107 | 0/6 | DROP |
| 98 | `oi_zscore_lag1` | 0.000506 | 99 | 60 | 113 | 1/6 | REVIEW |
| 99 | `rv_zscore_1w_lag1` | 0.000503 | 96 | 75 | 104 | 0/6 | DROP |
| 100 | `rv_zscore_1m_ema10` | 0.000468 | 98 | 82 | 106 | 0/6 | DROP |
| 101 | `hurst_100_lag3` | 0.000410 | 92 | 76 | 108 | 0/6 | DROP |
| 102 | `oi_zscore` | 0.000404 | 96 | 69 | 105 | 0/6 | DROP |
| 103 | `rv_zscore_1m_lag3` | 0.000375 | 101 | 85 | 111 | 0/6 | DROP |
| 104 | `oi_zscore_lag3` | 0.000363 | 97 | 83 | 110 | 0/6 | DROP |
| 105 | `oi_zscore_ema10` | 0.000346 | 95 | 86 | 113 | 0/6 | DROP |
| 106 | `oi_zscore_ema50` | 0.000290 | 101 | 73 | 111 | 0/6 | DROP |
| 107 | `down_jump_frac_lag1` | 0.000282 | 94 | 80 | 107 | 0/6 | DROP |
| 108 | `rv_zscore_1w_ema10` | 0.000265 | 100 | 94 | 113 | 0/6 | DROP |
| 109 | `bpv_tsrv_ratio_lag3` | 0.000249 | 97 | 81 | 111 | 0/6 | DROP |
| 110 | `up_jump_frac_lag3` | 0.000232 | 111 | 69 | 115 | 0/6 | DROP |
| 111 | `oi_change_5_lag3` | 0.000226 | 102 | 74 | 114 | 0/6 | DROP |
| 112 | `oi_change_5_lag1` | 0.000105 | 108 | 99 | 114 | 0/6 | DROP |
| 113 | `oi_acceleration_lag1` | 0.000103 | 107 | 99 | 113 | 0/6 | DROP |
| 114 | `funding_basis_lag3` | 0.000082 | 109 | 98 | 113 | 0/6 | DROP |
| 115 | `oi_change_5` | 0.000077 | 106 | 99 | 116 | 0/6 | DROP |
| 116 | `down_jump_frac_lag3` | 0.000073 | 109 | 98 | 116 | 0/6 | DROP |
| 117 | `ofi_normalized` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 118 | `ofi_normalized_lag1` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 119 | `ofi_normalized_lag3` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 120 | `depth_mid_spread_lag1` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 121 | `depth_mid_spread` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 122 | `ofi_normalized_ema10` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 123 | `depth_mid_spread_ema10` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 124 | `depth_mid_spread_lag3` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 125 | `depth_mid_spread_ema50` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |
| 126 | `ofi_normalized_ema50` | 0.000000 | 113 | 99 | 117 | 0/6 | DROP |