# Optuna-tuned SHAP audit — second pass on the 62-feature pruned list

Total pruned features audited: 62
KEEP in BOTH runs: 60
DROP in BOTH runs: 0
REVIEW in BOTH runs: 0

## Per-month Optuna config + QLIKE

| month | regime | n_train | n_test | test QLIKE | trials |
|---|---|---:|---:|---:|---:|
| 2022-06 | LUNA collapse aftermath + 3AC unwind (post-crash high vol) | 49,154 | 2,880 | 0.5009 | 161 |
| 2022-11 | FTX collapse + contagion (acute crisis) | 63,266 | 2,688 | 0.5971 | 104 |
| 2023-08 | mid-2023 range-bound calm (low realized vol) | 89,186 | 2,974 | 0.7664 | 200 |
| 2024-04 | BTC halving cycle / transition | 112,608 | 2,880 | 0.3491 | 138 |
| 2024-08 | yen-carry unwind (vol spike + cross-asset shock) | 124,319 | 2,976 | 0.4115 | 200 |
| 2026-01 | recent OOS (most recent regime baseline) | 174,047 | 2,976 | 0.5152 | 67 |
| 2023-12 | late-2023 ETF-speculation / GBTC discount narrowing | 100,896 | 2,976 | 0.4599 | 153 |
| 2025-04 | recent vol shock | 147,647 | 2,880 | 0.3183 | 200 |

## All 62 pruned features land in KEEP in at least one of the two runs

No further pruning indicated. Prune list is stable.