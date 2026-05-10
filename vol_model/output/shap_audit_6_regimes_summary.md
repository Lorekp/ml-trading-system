# Multi-Regime SHAP Audit (6 months)

## Months audited

| month | regime | n_train | n_test | test QLIKE |
|---|---|---:|---:|---:|
| 2022-06 | LUNA collapse aftermath + 3AC unwind (post-crash high vol) | 49,154 | 2,880 | 0.4895 |
| 2022-11 | FTX collapse + contagion (acute crisis) | 63,266 | 2,688 | 0.5953 |
| 2023-08 | mid-2023 range-bound calm (low realized vol) | 89,186 | 2,974 | 0.9777 |
| 2024-04 | BTC halving cycle / transition | 112,608 | 2,880 | 0.3527 |
| 2024-08 | yen-carry unwind (vol spike + cross-asset shock) | 124,319 | 2,976 | 0.4145 |
| 2026-01 | recent OOS (most recent regime baseline) | 174,047 | 2,976 | 0.5172 |

## Decision summary
- KEEP: 50 features (median rank <=35 OR best rank <=25 OR non-trivial SHAP in >=4 of 6 months)
- DROP: 43 features (best rank >60 AND non-trivial in zero months)
- REVIEW: 33 features (judgment call — see CSV)

## Top 50 features by mean SHAP across regimes

| rank | feature | mean SHAP | median rank | best rank | worst rank | months >1% | decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `duration_s_ema50` | 0.1867 | 1 | 1 | 3 | 6/6 | KEEP |
| 2 | `vov_ema50` | 0.1407 | 2 | 1 | 3 | 6/6 | KEEP |
| 3 | `n_bars_in_window` | 0.1264 | 2 | 2 | 3 | 6/6 | KEEP |
| 4 | `duration_s_ema10` | 0.0480 | 4 | 4 | 5 | 6/6 | KEEP |
| 5 | `n_trades_ema50` | 0.0429 | 5 | 4 | 7 | 6/6 | KEEP |
| 6 | `vov_ema10` | 0.0222 | 7 | 6 | 39 | 6/6 | KEEP |
| 7 | `noise_level_ema50` | 0.0209 | 9 | 7 | 17 | 6/6 | KEEP |
| 8 | `noise_level` | 0.0198 | 8 | 6 | 18 | 6/6 | KEEP |
| 9 | `tsrv_mean_1w` | 0.0191 | 9 | 7 | 15 | 6/6 | KEEP |
| 10 | `tsrv_mean_1d` | 0.0190 | 10 | 5 | 15 | 6/6 | KEEP |
| 11 | `tsrv_mean_1w_lag1` | 0.0169 | 26 | 4 | 61 | 5/6 | KEEP |
| 12 | `noise_level_lag1` | 0.0155 | 16 | 7 | 61 | 5/6 | KEEP |
| 13 | `tsrv_mean_1d_ema50` | 0.0153 | 13 | 7 | 26 | 6/6 | KEEP |
| 14 | `tsrv_mean_1w_ema50` | 0.0138 | 11 | 6 | 32 | 6/6 | KEEP |
| 15 | `tsrv_mean_1d_lag3` | 0.0121 | 15 | 9 | 23 | 6/6 | KEEP |
| 16 | `vpin_ema50` | 0.0114 | 14 | 11 | 28 | 6/6 | KEEP |
| 17 | `time_to_funding_lag1` | 0.0105 | 14 | 13 | 37 | 6/6 | KEEP |
| 18 | `noise_level_ema10` | 0.0090 | 19 | 10 | 98 | 4/6 | KEEP |
| 19 | `parkinson_1d` | 0.0089 | 19 | 10 | 30 | 6/6 | KEEP |
| 20 | `ffd_rv_lag1` | 0.0078 | 25 | 12 | 53 | 6/6 | KEEP |
| 21 | `n_trades_ema10` | 0.0076 | 21 | 17 | 31 | 6/6 | KEEP |
| 22 | `tsrv_mean_1d_ema10` | 0.0075 | 25 | 13 | 28 | 6/6 | KEEP |
| 23 | `duration_s` | 0.0074 | 21 | 19 | 25 | 6/6 | KEEP |
| 24 | `time_to_funding` | 0.0071 | 23 | 13 | 42 | 6/6 | KEEP |
| 25 | `vpin_ema10` | 0.0068 | 18 | 15 | 60 | 6/6 | KEEP |
| 26 | `time_to_funding_ema50` | 0.0062 | 25 | 17 | 34 | 6/6 | KEEP |
| 27 | `tsrv_mean_1d_lag1` | 0.0057 | 26 | 17 | 54 | 5/6 | KEEP |
| 28 | `vpin_lag1` | 0.0055 | 36 | 13 | 47 | 6/6 | KEEP |
| 29 | `parkinson_1d_ema50` | 0.0050 | 32 | 20 | 42 | 5/6 | KEEP |
| 30 | `imbalance_ema50` | 0.0049 | 31 | 20 | 45 | 5/6 | KEEP |
| 31 | `vpin` | 0.0049 | 26 | 24 | 50 | 6/6 | KEEP |
| 32 | `ffd_rv_lag3` | 0.0043 | 40 | 16 | 79 | 4/6 | KEEP |
| 33 | `parkinson_1d_lag3` | 0.0042 | 40 | 14 | 87 | 3/6 | KEEP |
| 34 | `time_to_funding_lag3` | 0.0042 | 33 | 27 | 51 | 6/6 | KEEP |
| 35 | `ffd_rv_ema50` | 0.0042 | 50 | 17 | 70 | 3/6 | KEEP |
| 36 | `up_jump_frac_ema50` | 0.0040 | 44 | 16 | 51 | 5/6 | KEEP |
| 37 | `duration_s_lag1` | 0.0040 | 33 | 29 | 48 | 5/6 | KEEP |
| 38 | `tsrv_ffd` | 0.0038 | 37 | 31 | 43 | 5/6 | KEEP |
| 39 | `vpin_lag3` | 0.0038 | 44 | 22 | 67 | 5/6 | KEEP |
| 40 | `parkinson_1d_ema10` | 0.0036 | 37 | 23 | 81 | 5/6 | KEEP |
| 41 | `vov_lag3` | 0.0034 | 45 | 21 | 68 | 5/6 | KEEP |
| 42 | `time_to_funding_ema10` | 0.0031 | 42 | 30 | 56 | 5/6 | KEEP |
| 43 | `ffd_rv` | 0.0030 | 42 | 25 | 73 | 4/6 | KEEP |
| 44 | `vov` | 0.0029 | 48 | 25 | 68 | 4/6 | KEEP |
| 45 | `duration_s_lag3` | 0.0029 | 43 | 32 | 73 | 4/6 | KEEP |
| 46 | `up_jump_frac` | 0.0029 | 40 | 36 | 66 | 3/6 | REVIEW |
| 47 | `tsrv_ffd_lag1` | 0.0028 | 46 | 29 | 59 | 4/6 | KEEP |
| 48 | `bpv_tsrv_ratio_ema50` | 0.0027 | 51 | 30 | 74 | 3/6 | REVIEW |
| 49 | `down_jump_frac_ema50` | 0.0027 | 49 | 24 | 81 | 3/6 | KEEP |
| 50 | `rv_zscore_1d` | 0.0025 | 53 | 23 | 65 | 3/6 | KEEP |

## Bottom 30 features by mean SHAP

| rank | feature | mean SHAP | median rank | best rank | worst rank | months >1% | decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 97 | `oi_acceleration` | 0.000236 | 98 | 68 | 105 | 0/6 | DROP |
| 98 | `oi_zscore_lag3` | 0.000209 | 98 | 83 | 107 | 0/6 | DROP |
| 99 | `oi_acceleration_ema10` | 0.000205 | 98 | 65 | 102 | 0/6 | DROP |
| 100 | `rv_zscore_1w_ema10` | 0.000198 | 96 | 89 | 113 | 0/6 | DROP |
| 101 | `oi_zscore_ema10` | 0.000198 | 97 | 86 | 101 | 0/6 | DROP |
| 102 | `down_jump_frac_lag1` | 0.000185 | 93 | 80 | 107 | 0/6 | DROP |
| 103 | `oi_acceleration_lag3` | 0.000142 | 96 | 93 | 101 | 0/6 | DROP |
| 104 | `oi_change_5_ema10` | 0.000138 | 98 | 77 | 106 | 0/6 | DROP |
| 105 | `bpv_tsrv_ratio_lag3` | 0.000137 | 96 | 81 | 111 | 0/6 | DROP |
| 106 | `oi_zscore` | 0.000131 | 97 | 93 | 101 | 0/6 | DROP |
| 107 | `oi_zscore_lag1` | 0.000126 | 99 | 87 | 113 | 0/6 | DROP |
| 108 | `oi_zscore_ema50` | 0.000094 | 100 | 90 | 104 | 0/6 | DROP |
| 109 | `funding_basis_lag3` | 0.000093 | 103 | 86 | 113 | 0/6 | DROP |
| 110 | `oi_change_5_ema50` | 0.000078 | 100 | 98 | 108 | 0/6 | DROP |
| 111 | `oi_acceleration_lag1` | 0.000069 | 102 | 98 | 109 | 0/6 | DROP |
| 112 | `oi_change_5` | 0.000063 | 100 | 98 | 108 | 0/6 | DROP |
| 113 | `oi_acceleration_ema50` | 0.000062 | 100 | 88 | 110 | 0/6 | DROP |
| 114 | `oi_change_5_lag3` | 0.000052 | 100 | 95 | 109 | 0/6 | DROP |
| 115 | `oi_change_5_lag1` | 0.000020 | 104 | 98 | 114 | 0/6 | DROP |
| 116 | `down_jump_frac_lag3` | 0.000019 | 100 | 98 | 116 | 0/6 | DROP |
| 117 | `ofi_normalized` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 118 | `ofi_normalized_lag1` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 119 | `ofi_normalized_lag3` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 120 | `depth_mid_spread_lag1` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 121 | `depth_mid_spread` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 122 | `ofi_normalized_ema10` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 123 | `depth_mid_spread_ema10` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 124 | `depth_mid_spread_lag3` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 125 | `depth_mid_spread_ema50` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |
| 126 | `ofi_normalized_ema50` | 0.000000 | 105 | 98 | 117 | 0/6 | DROP |