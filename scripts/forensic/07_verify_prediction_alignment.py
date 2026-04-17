"""Forensic test 7: verify y_pred alignment with vol_arb's vol_ann.

Oracle (y_pred = y_true) should produce HIGHER PnL than the model, because
a perfect forecast is an upper bound. The model beats the oracle, which is
impossible unless:
  (a) y_pred is not actually aligned to the bar's forward variance, OR
  (b) y_true in OOF differs from vol_ann in vol_arb parquet, OR
  (c) the model is using leaked info during training

Check the actual data to find which.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

oof = pd.read_parquet("vol_model/output/BTCUSDT_hybrid_predictions.parquet")
va = pd.read_parquet("vol_arb/output/BTC_vol_arb_svi_shifted.parquet")

print(f"OOF shape: {oof.shape}")
print(f"OOF columns: {list(oof.columns)}")
print(f"OOF index: {oof.index[0]} ... {oof.index[-1]}")
print()
print(f"vol_arb shape: {va.shape}")
print(f"vol_arb index: {va.index[0]} ... {va.index[-1]}")
print()

# Join and compare
joined = va[["vol_ann"]].join(oof[["y_true", "y_pred"]], how="inner")
joined["ln_vol_va"] = np.log(joined["vol_ann"].clip(lower=1e-8))
joined = joined.dropna()

print(f"Joined rows: {len(joined):,}")
print()

print("Sample rows:")
print(joined.head(10).to_string())
print()

# Compare y_true (from OOF) with ln(vol_ann) (from vol_arb)
diff_true = joined["y_true"] - joined["ln_vol_va"]
print("y_true (OOF) vs ln(vol_ann) (vol_arb):")
print(f"  Correlation: {joined['y_true'].corr(joined['ln_vol_va']):.6f}")
print(f"  Mean diff: {diff_true.mean():.6f}")
print(f"  Max abs diff: {diff_true.abs().max():.6f}")
print()

# If they match perfectly, y_true == ln_vol_va and we can trust the oracle
if diff_true.abs().max() < 1e-6:
    print("-> y_true matches ln(vol_ann): oracle was correctly formulated.")
    print("-> Model beating oracle means LEAKAGE in y_pred itself.")
else:
    print("-> MISMATCH: y_true and ln(vol_ann) differ — oracle was testing wrong target.")
    print()
    # What's the actual alignment? Try shift
    for shift in [-2, -1, 0, 1, 2]:
        shifted = joined["ln_vol_va"].shift(shift)
        corr = joined["y_true"].corr(shifted)
        print(f"  corr(y_true, ln_vol_va.shift({shift})) = {corr:.6f}")

# Also compare model y_pred correlation with various shifts
print()
print("Model y_pred correlation with ln_vol_va at various shifts:")
for shift in [-2, -1, 0, 1, 2]:
    shifted = joined["ln_vol_va"].shift(shift)
    valid = shifted.notna() & joined["y_pred"].notna()
    corr = joined.loc[valid, "y_pred"].corr(shifted[valid])
    print(f"  corr(y_pred, ln_vol_va.shift({shift})) = {corr:.6f}")

# Per-bar: if model is "better than oracle", maybe model is correlated with NEXT bar's vol
# Check: is y_pred correlated with vol_ann at T+1 (future)?
print()
print("Oracle test diagnosis:")
print(f"  OOF y_pred correlation with OOF y_true: {joined['y_pred'].corr(joined['y_true']):.4f}")
print(f"  OOF y_pred correlation with ln(vol_arb.vol_ann): {joined['y_pred'].corr(joined['ln_vol_va']):.4f}")

# Index alignment sanity check
ts_oof = set(oof.index[:20].tolist())
ts_va = set(va.index[:20].tolist())
common_prefix = ts_oof & ts_va
print(f"\nIndex type: OOF={type(oof.index[0])}, vol_arb={type(va.index[0])}")
print(f"First 20 OOF timestamps: {sorted(oof.index[:20])[:3]}")
print(f"First 20 vol_arb timestamps: {sorted(va.index[:20])[:3]}")
