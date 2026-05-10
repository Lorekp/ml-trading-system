"""
Cycle-10 vs Cycle-9 aligner audit (read-only).

Investigates whether the cycle-10 aligner (β* per month) is mis-calibrated
for cycle-10 predictions, or if the β* fit is fine and the QLIKE-vs-PnL gap
comes from elsewhere.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Claude\ml-trading-system")
C9_DIR = ROOT / "backups" / "2026-05-09_pre_transvol_optuna_retrain"
C10_DIR = ROOT / "vol_model" / "output"

# Files
c9_betas_path = C9_DIR / "aligner_betas_by_month.json"
c10_betas_path = C10_DIR / "aligner_betas_by_month.json"
c9_pred_path = C9_DIR / "BTCUSDT_pruned_svi_har_pd_predictions.parquet"
c10_pred_path = C10_DIR / "BTCUSDT_pruned_svi_har_pd_predictions.parquet"
c9_aligned_path = C9_DIR / "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
c10_aligned_path = C10_DIR / "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
c9_trade_post = C9_DIR / "BTC_v2_trade_log_post_retrain.parquet"
c9_trade_pre = C9_DIR / "BTC_v2_trade_log_pre_retrain.parquet"

# V2 thresholds
THR_LONG = 12
THR_SHORT = 22


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------- 1. β* JSON DIFF ----------------
section("1. β* JSON DIFF (cycle-10 minus cycle-9)")
c9_b = json.load(open(c9_betas_path))
c10_b = json.load(open(c10_betas_path))

print(f"  cycle-9  months: {len(c9_b)}, range {min(c9_b)}..{max(c9_b)}")
print(f"  cycle-10 months: {len(c10_b)}, range {min(c10_b)}..{max(c10_b)}")

months = sorted(set(c9_b) & set(c10_b))
df_b = pd.DataFrame({
    "month": months,
    "c9_beta": [c9_b[m] for m in months],
    "c10_beta": [c10_b[m] for m in months],
})
df_b["delta"] = df_b["c10_beta"] - df_b["c9_beta"]
df_b["abs_delta"] = df_b["delta"].abs()

print(f"\n  Mean β* cycle-9 :  {df_b.c9_beta.mean():+.5f}")
print(f"  Mean β* cycle-10:  {df_b.c10_beta.mean():+.5f}")
print(f"  Median β* c9    :  {df_b.c9_beta.median():+.5f}")
print(f"  Median β* c10   :  {df_b.c10_beta.median():+.5f}")
print(f"  Std β* cycle-9  :  {df_b.c9_beta.std():.5f}")
print(f"  Std β* cycle-10 :  {df_b.c10_beta.std():.5f}")
print(f"\n  Delta distribution:")
print(f"    mean   : {df_b.delta.mean():+.5f}")
print(f"    median : {df_b.delta.median():+.5f}")
print(f"    std    : {df_b.delta.std():.5f}")
print(f"    min    : {df_b.delta.min():+.5f}")
print(f"    max    : {df_b.delta.max():+.5f}")

n_signflip = ((df_b.c9_beta * df_b.c10_beta) < 0).sum()
print(f"  Sign-flips: {n_signflip} / {len(df_b)} months")

print("\n  Top-5 biggest |delta| months:")
top5 = df_b.sort_values("abs_delta", ascending=False).head(5)
for _, r in top5.iterrows():
    print(f"    {r['month']}  c9={r['c9_beta']:+.5f}  c10={r['c10_beta']:+.5f}  delta={r['delta']:+.5f}")

print("\n  Period averages (β*):")
def period(m):
    y, mo = m.split("-")
    y, mo = int(y), int(mo)
    if y == 2022: return "2022"
    if y == 2023: return "2023"
    if y == 2024: return "2024"
    if y == 2025 and mo <= 3: return "2025-Q1"
    return "2025-Q2+"
df_b["period"] = df_b["month"].map(period)
print(df_b.groupby("period")[["c9_beta", "c10_beta", "delta"]].mean().round(5).to_string())


# ---------------- 2. RESIDUAL DISTRIBUTION DIFF ----------------
section("2. RESIDUAL DISTRIBUTION (y_true - y_pred_raw) per period")

def load_pred(path):
    df = pd.read_parquet(path)
    # Standardize: timestamp / y_true / y_pred
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    # Find pred / true columns
    cand_pred = [c for c in df.columns if "pred" in c.lower()]
    cand_true = [c for c in df.columns if c.lower() in ("y_true", "true", "y", "target", "y_target")]
    return df, cand_pred, cand_true

c9_df, c9_pcols, c9_tcols = load_pred(c9_pred_path)
c10_df, c10_pcols, c10_tcols = load_pred(c10_pred_path)

print(f"  cycle-9 columns sample : {list(c9_df.columns)[:10]}")
print(f"  cycle-10 columns sample: {list(c10_df.columns)[:10]}")
print(f"  cycle-9 pred cols      : {c9_pcols}")
print(f"  cycle-9 true cols      : {c9_tcols}")
print(f"  cycle-10 pred cols     : {c10_pcols}")
print(f"  cycle-10 true cols     : {c10_tcols}")
print(f"  cycle-9 rows: {len(c9_df):,}, cycle-10 rows: {len(c10_df):,}")

# Build residual frames
def build_resid(df):
    d = df[["timestamp", "y_pred", "y_true"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.dropna(subset=["y_pred", "y_true"])
    d["residual"] = d["y_true"] - d["y_pred"]
    d["sq_err"] = d["residual"] ** 2
    # QLIKE in log-vol space: y_true/y_pred is exp(2*resid); QLIKE = exp(2r) - 2r - 1
    d["qlike"] = np.exp(2 * d["residual"]) - 2 * d["residual"] - 1
    d["month"] = d["timestamp"].dt.to_period("M").astype(str)
    return d

c9_res = build_resid(c9_df)
c10_res = build_resid(c10_df)
print(f"  cycle-9 valid rows : {len(c9_res):,}")
print(f"  cycle-10 valid rows: {len(c10_res):,}")

# Period assign for residuals
def period_dt(ts):
    y, m = ts.year, ts.month
    if y == 2022: return "2022"
    if y == 2023: return "2023"
    if y == 2024: return "2024"
    if y == 2025 and m <= 3: return "2025-Q1"
    return "2025-Q2+"

c9_res["period"] = c9_res["timestamp"].apply(period_dt)
c10_res["period"] = c10_res["timestamp"].apply(period_dt)

print("\n  Residual distribution by PERIOD:")
def stats(g):
    return pd.Series({
        "n": len(g),
        "mean": g["residual"].mean(),
        "std": g["residual"].std(),
        "skew": g["residual"].skew(),
        "p05": g["residual"].quantile(0.05),
        "p95": g["residual"].quantile(0.95),
        "MSE": g["sq_err"].mean(),
        "QLIKE": g["qlike"].mean(),
    })

stats9 = c9_res.groupby("period").apply(stats, include_groups=False).round(6)
stats10 = c10_res.groupby("period").apply(stats, include_groups=False).round(6)
print("\n  CYCLE-9 residuals by period:")
print(stats9.to_string())
print("\n  CYCLE-10 residuals by period:")
print(stats10.to_string())

# Diff
diff_stats = (stats10 - stats9).round(6)
diff_stats["MSE_pct"] = (stats10["MSE"] / stats9["MSE"] - 1).round(4) * 100
diff_stats["QLIKE_pct"] = (stats10["QLIKE"] / stats9["QLIKE"] - 1).round(4) * 100
print("\n  DIFF (cycle-10 - cycle-9), and % change of MSE/QLIKE:")
print(diff_stats[["mean", "std", "skew", "MSE", "QLIKE", "MSE_pct", "QLIKE_pct"]].to_string())


# ---------------- 3. QLIKE-vs-MSE per month ----------------
section("3. QLIKE vs MSE divergence per month")

m9 = c9_res.groupby("month").agg(MSE9=("sq_err", "mean"), QLIKE9=("qlike", "mean"), n9=("residual", "size"), mean9=("residual", "mean"), std9=("residual", "std")).reset_index()
m10 = c10_res.groupby("month").agg(MSE10=("sq_err", "mean"), QLIKE10=("qlike", "mean"), n10=("residual", "size"), mean10=("residual", "mean"), std10=("residual", "std")).reset_index()
m_join = m9.merge(m10, on="month", how="inner")
m_join["dMSE"] = m_join["MSE10"] - m_join["MSE9"]
m_join["dQLIKE"] = m_join["QLIKE10"] - m_join["QLIKE9"]
m_join["dMSE_pct"] = (m_join["MSE10"] / m_join["MSE9"] - 1) * 100
m_join["dQLIKE_pct"] = (m_join["QLIKE10"] / m_join["QLIKE9"] - 1) * 100

n_qwin_mloss = ((m_join["dQLIKE"] < 0) & (m_join["dMSE"] > 0)).sum()
n_qwin_mwin = ((m_join["dQLIKE"] < 0) & (m_join["dMSE"] < 0)).sum()
n_qloss_mwin = ((m_join["dQLIKE"] > 0) & (m_join["dMSE"] < 0)).sum()
n_qloss_mloss = ((m_join["dQLIKE"] > 0) & (m_join["dMSE"] > 0)).sum()
print(f"  Quadrant counts (n={len(m_join)}):")
print(f"    QLIKE win + MSE win  : {n_qwin_mwin}")
print(f"    QLIKE win + MSE loss : {n_qwin_mloss}  <- divergence: cycle-10 wins QLIKE but worse MSE")
print(f"    QLIKE loss + MSE win : {n_qloss_mwin}")
print(f"    QLIKE loss + MSE loss: {n_qloss_mloss}")

print("\n  10 months with biggest QLIKE improvement but MSE regression:")
div = m_join[(m_join["dQLIKE"] < 0) & (m_join["dMSE"] > 0)].sort_values("dQLIKE").head(10)
print(div[["month", "n9", "MSE9", "MSE10", "dMSE_pct", "QLIKE9", "QLIKE10", "dQLIKE_pct"]].round(5).to_string(index=False))

print("\n  Aggregate (all months):")
print(f"    Total QLIKE: c9={m_join['QLIKE9'].mean():.5f}  c10={m_join['QLIKE10'].mean():.5f}  pct={(m_join['QLIKE10'].mean()/m_join['QLIKE9'].mean()-1)*100:+.2f}%")
print(f"    Total MSE  : c9={m_join['MSE9'].mean():.5f}  c10={m_join['MSE10'].mean():.5f}  pct={(m_join['MSE10'].mean()/m_join['MSE9'].mean()-1)*100:+.2f}%")


# ---------------- 4. ALIGNED PREDICTION CHECK ----------------
section("4. ALIGNED PREDICTION CHECK (post-aligner shifts vs entry thresholds)")
# Check what y_pred values look like (these are log-vol predictions; threshold 12 / 22 isn't
# directly comparable). The thresholds 12/22 in V2 are confidence thresholds in BPS or some
# scaled units. Let me first examine the aligned files.

c9_al = pd.read_parquet(c9_aligned_path)
c10_al = pd.read_parquet(c10_aligned_path)
print(f"  cycle-9 aligned cols : {list(c9_al.columns)}")
print(f"  cycle-10 aligned cols: {list(c10_al.columns)}")
print(f"  cycle-9 aligned head:")
print(c9_al.head(3).to_string())
print(f"\n  cycle-10 aligned head:")
print(c10_al.head(3).to_string())

# Show the actual β* shift (timestamp is the index)
c9_al = c9_al.reset_index()
c10_al = c10_al.reset_index()
c9_al["timestamp"] = pd.to_datetime(c9_al["timestamp"])
c10_al["timestamp"] = pd.to_datetime(c10_al["timestamp"])
c9_pred_raw = c9_df[["timestamp", "y_pred"]].rename(columns={"y_pred": "y_pred_raw"})
c9_pred_raw["timestamp"] = pd.to_datetime(c9_pred_raw["timestamp"])
c10_pred_raw = c10_df[["timestamp", "y_pred"]].rename(columns={"y_pred": "y_pred_raw"})
c10_pred_raw["timestamp"] = pd.to_datetime(c10_pred_raw["timestamp"])

# Aligned files already have y_pred_raw column; just use that
c9_join = c9_al.copy()
c10_join = c10_al.copy()
print(f"\n  cycle-9 aligned-vs-raw join rows : {len(c9_join):,}")
print(f"  cycle-10 aligned-vs-raw join rows: {len(c10_join):,}")

# Find aligned column
ac9 = [c for c in c9_al.columns if "pred" in c.lower() and c != "y_pred_raw"]
ac10 = [c for c in c10_al.columns if "pred" in c.lower() and c != "y_pred_raw"]
print(f"  cycle-9 aligned pred col(s) : {ac9}")
print(f"  cycle-10 aligned pred col(s): {ac10}")

# Use y_pred (after aligner overwrites it)
c9_join["shift"] = c9_join["y_pred"] - c9_join["y_pred_raw"]
c10_join["shift"] = c10_join["y_pred"] - c10_join["y_pred_raw"]

print(f"\n  Aligner shift distribution (y_pred_aligned - y_pred_raw):")
print(f"    cycle-9 :  mean={c9_join['shift'].mean():+.5f}  median={c9_join['shift'].median():+.5f}  std={c9_join['shift'].std():.5f}")
print(f"    cycle-10:  mean={c10_join['shift'].mean():+.5f}  median={c10_join['shift'].median():+.5f}  std={c10_join['shift'].std():.5f}")


# ---------------- 5. THRESHOLD-CROSS COUNT ----------------
section("5. RAW vs ALIGNED y_pred shift in log-vol space")
# In log-vol space, β* shifts y_pred UP. exp(y_pred) is the predicted vol level.
# Higher pred_vol -> larger pred_iv -> changes whether market IV is above (short signal)
# or below (long signal) the prediction.
# We can't compute V2 edge here without IV data, but we CAN bound the impact.

# How big is the typical y_pred change in vol-level units?
# vol_level_aligned = exp(y_pred_aligned), vol_level_raw = exp(y_pred_raw)
# Pct change in vol level = exp(beta*) - 1
print(f"  Typical aligner shift in log-vol space:")
print(f"    cycle-9 mean β*  = {df_b.c9_beta.mean():+.5f}  ->  vol-level pct = {(np.exp(df_b.c9_beta.mean())-1)*100:+.2f}%")
print(f"    cycle-10 mean β* = {df_b.c10_beta.mean():+.5f}  ->  vol-level pct = {(np.exp(df_b.c10_beta.mean())-1)*100:+.2f}%")
print(f"  -> cycle-10 aligner pushes pred_vol UP by ~{(np.exp(df_b.c10_beta.mean())-1)*100:.1f}% on avg, vs ~{(np.exp(df_b.c9_beta.mean())-1)*100:.1f}% in cycle-9.")
print(f"  -> Each extra +1pp upward shift on pred_vol means MORE shorts will fire (mkt_iv-pred_iv larger)")
print(f"     and FEWER longs (or same longs but with smaller edge).")

# Per-month, how often does aligner push y_pred above raw y_true (i.e., turn an under-prediction into over-prediction)?
c9_join["over_after_align"] = (c9_join["y_pred"] > c9_join["y_true"])
c9_join["over_before_align"] = (c9_join["y_pred_raw"] > c9_join["y_true"])
c9_join["flipped_to_over"] = c9_join["over_after_align"] & (~c9_join["over_before_align"])
c10_join["over_after_align"] = (c10_join["y_pred"] > c10_join["y_true"])
c10_join["over_before_align"] = (c10_join["y_pred_raw"] > c10_join["y_true"])
c10_join["flipped_to_over"] = c10_join["over_after_align"] & (~c10_join["over_before_align"])

print(f"\n  Direction-flip stats (raw under-truth -> aligned over-truth):")
print(f"    cycle-9 :  {c9_join['flipped_to_over'].sum():,} / {len(c9_join):,} bars ({c9_join['flipped_to_over'].mean()*100:.2f}%)")
print(f"    cycle-10:  {c10_join['flipped_to_over'].sum():,} / {len(c10_join):,} bars ({c10_join['flipped_to_over'].mean()*100:.2f}%)")

# How does the aligner ANSWER the residual question? β* should equal μ + σ² of residual (Gaussian) -- check
print(f"\n  β* sanity: β* should equal μ + σ² of residual on training (3-month) cal-window")
print(f"  Whole-sample as-if check (NOT walk-forward, just descriptive):")
for label, res in [("cycle-9", c9_res), ("cycle-10", c10_res)]:
    mu, sigma = res["residual"].mean(), res["residual"].std()
    optimal_b = -mu + sigma * sigma  # to UNDO the under-prediction we'd shift by -mu, then add safety σ²
    # Actually β* moves y_pred -> y_pred + β. If residual = y_true - y_pred has mean μ, then to make
    # mean residual zero we'd add μ. The QLIKE-optimal shift is μ + σ² per the user's prompt.
    optimal_b_qlike = mu + sigma * sigma
    print(f"    {label}: μ_resid={mu:+.5f}  σ_resid={sigma:.5f}  μ+σ²={optimal_b_qlike:+.5f}  observed mean β*={c9_b if label=='cycle-9' else c10_b}".replace(str(c9_b), f"{df_b.c9_beta.mean():+.5f}").replace(str(c10_b), f"{df_b.c10_beta.mean():+.5f}"))


# ---------------- 6. ALIGNER VARIANCE INCREASE ON V2 ENTRY EDGES ----------------
section("6. PROXY: how much does the bigger aligner change short-vs-long mix?")

# After aligner: pred_vol *= exp(β*). For a short signal, edge ~= mkt_iv - pred_iv*. Higher
# pred_iv* means smaller short edge. For longs, smaller pred_iv* makes long edge larger.
# Cycle-10 has bigger β* -> pushes pred even higher -> MORE short signals fire, FEWER longs.

# We can use mean β* deltas per period as proxy for "how much harder is it to fire a long"
print("  Per-period: cycle-10 minus cycle-9 mean β* (positive = bigger upward shift = MORE shorts, FEWER longs)")
print(df_b.groupby("period")[["delta"]].mean().round(5).to_string())

# Match V2 trade logs from the backup if available
print("\n  V2 trade logs comparison (pre_retrain = cycle-9 result, post_retrain = cycle-10 result):")
def summarize_tl(p, label):
    tl = pd.read_parquet(p)
    sig = tl["signal"]  # +1 long, -1 short, 0 no trade
    longs = (sig == 1).sum()
    shorts = (sig == -1).sum()
    no_trade = (sig == 0).sum()
    pnl_nc = tl["net_pnl_nc"].sum()
    pnl_c = tl["net_pnl"].sum()
    fired = sig != 0
    long_pnl = tl.loc[sig == 1, "net_pnl_nc"].sum() if longs else 0
    short_pnl = tl.loc[sig == -1, "net_pnl_nc"].sum() if shorts else 0
    print(f"  {label}:")
    print(f"    rows={len(tl):,}  trades={fired.sum():,}  longs={longs:,}  shorts={shorts:,}  no_trade={no_trade:,}")
    print(f"    NC PnL total: ${pnl_nc:,.0f}   (long ${long_pnl:,.0f} / short ${short_pnl:,.0f})")
    print(f"    Compound PnL total: ${pnl_c:,.0f}")
    return tl

c9_tl = summarize_tl(c9_trade_pre, "pre_retrain (cycle-9 era)")
c10_tl = summarize_tl(c9_trade_post, "post_retrain (cycle-10 era)")

print("\n  DIFF (cycle-10 - cycle-9):")
def sig_count(tl, v): return (tl["signal"] == v).sum()
print(f"    longs : {sig_count(c10_tl,1) - sig_count(c9_tl,1):+,d}")
print(f"    shorts: {sig_count(c10_tl,-1) - sig_count(c9_tl,-1):+,d}")
print(f"    NC PnL: ${c10_tl['net_pnl_nc'].sum() - c9_tl['net_pnl_nc'].sum():+,.0f}")
print(f"    NC PnL longs : ${c10_tl.loc[c10_tl.signal==1,'net_pnl_nc'].sum() - c9_tl.loc[c9_tl.signal==1,'net_pnl_nc'].sum():+,.0f}")
print(f"    NC PnL shorts: ${c10_tl.loc[c10_tl.signal==-1,'net_pnl_nc'].sum() - c9_tl.loc[c9_tl.signal==-1,'net_pnl_nc'].sum():+,.0f}")

# How many bars have y_pred where the aligner pushed them across the entry threshold?
# Approximate: for each bar in c9_tl/c10_tl, check signal change
# Match by ts
c9_tl_idx = c9_tl.reset_index()
c10_tl_idx = c10_tl.reset_index()
# Find timestamp column
ts_col_c9 = next((c for c in c9_tl_idx.columns if c in ("ts", "timestamp", "index")), c9_tl_idx.columns[0])
ts_col_c10 = next((c for c in c10_tl_idx.columns if c in ("ts", "timestamp", "index")), c10_tl_idx.columns[0])
print(f"  c9 ts col: {ts_col_c9}  c10 ts col: {ts_col_c10}")
c9_tl_idx = c9_tl_idx.rename(columns={ts_col_c9: "_ts"})
c10_tl_idx = c10_tl_idx.rename(columns={ts_col_c10: "_ts"})
both = c9_tl_idx[["_ts", "signal", "edge", "y_pred", "iv_used"]].merge(
    c10_tl_idx[["_ts", "signal", "edge", "y_pred", "iv_used"]],
    on="_ts", how="inner", suffixes=("_c9", "_c10"))
print(f"\n  Bars matched: {len(both):,}")
both["sig_change"] = both["signal_c9"] != both["signal_c10"]
print(f"  Bars with signal change cycle-9 -> cycle-10: {both.sig_change.sum():,} ({both.sig_change.mean()*100:.2f}%)")
flip_table = pd.crosstab(both["signal_c9"], both["signal_c10"], rownames=["c9"], colnames=["c10"])
print(f"  Signal-flip table (rows=cycle-9, cols=cycle-10):")
print(flip_table.to_string())

# Pred shift distribution at trade bars
trade_bars = both[(both["signal_c9"] != 0) | (both["signal_c10"] != 0)].copy()
trade_bars["dy_pred"] = trade_bars["y_pred_c10"] - trade_bars["y_pred_c9"]
trade_bars["dedge"] = trade_bars["edge_c10"] - trade_bars["edge_c9"]
print(f"\n  Among trade-eligible bars (any-side fired in either cycle), dy_pred and dedge:")
print(f"    dy_pred  : mean={trade_bars['dy_pred'].mean():+.5f}  median={trade_bars['dy_pred'].median():+.5f}  std={trade_bars['dy_pred'].std():.5f}")
print(f"    dedge    : mean={trade_bars['dedge'].mean():+.5f}  median={trade_bars['dedge'].median():+.5f}  std={trade_bars['dedge'].std():.5f}")


# ---------------- 7. CAL-WINDOW RESIDUAL (TRAILING 3-MONTH) ----------------
section("7. Per-month walk-forward β* sanity (trailing 3-month cal-window)")
# QLIKE-optimal β* on trailing 3 months: should equal mean of residual + variance
# Compute it from BOTH cycles' raw predictions and compare to JSON

def trailing_optimal(res_df, cal_months=3):
    """For each month, compute QLIKE-optimal β* on trailing cal_months."""
    res_df = res_df.copy()
    res_df["month_p"] = res_df["timestamp"].dt.to_period("M")
    months_sorted = sorted(res_df["month_p"].unique())
    out = []
    for i, m in enumerate(months_sorted):
        if i < cal_months:
            out.append({"month": str(m), "n_cal": 0, "mu_cal": np.nan, "sigma_cal": np.nan, "beta_qlike": np.nan})
            continue
        cal_start = months_sorted[i - cal_months]
        cal_end = months_sorted[i - 1]
        cal = res_df[(res_df["month_p"] >= cal_start) & (res_df["month_p"] <= cal_end)]
        if len(cal) == 0:
            out.append({"month": str(m), "n_cal": 0, "mu_cal": np.nan, "sigma_cal": np.nan, "beta_qlike": np.nan})
            continue
        mu = cal["residual"].mean()
        sigma = cal["residual"].std()
        # QLIKE-optimal closed-form: 0.5 * log(mean(exp(2*resid)))
        beta_q = 0.5 * np.log(np.mean(np.exp(2 * cal["residual"])))
        out.append({"month": str(m), "n_cal": len(cal), "mu_cal": mu, "sigma_cal": sigma, "beta_qlike": beta_q})
    return pd.DataFrame(out)

c9_recompute = trailing_optimal(c9_res, cal_months=3)
c10_recompute = trailing_optimal(c10_res, cal_months=3)

# Merge with JSON β*
c9_check = c9_recompute.merge(pd.DataFrame({"month": list(c9_b.keys()), "json_beta": list(c9_b.values())}), on="month", how="left")
c10_check = c10_recompute.merge(pd.DataFrame({"month": list(c10_b.keys()), "json_beta": list(c10_b.values())}), on="month", how="left")
c9_check["json_minus_recompute"] = c9_check["json_beta"] - c9_check["beta_qlike"]
c10_check["json_minus_recompute"] = c10_check["json_beta"] - c10_check["beta_qlike"]

print("  cycle-9 (recompute = closed-form on trailing 3mo, vs JSON):")
print(f"    matched months : {c9_check.json_minus_recompute.notna().sum()}")
print(f"    max abs diff   : {c9_check.json_minus_recompute.abs().max():.6f}")
print(f"    median abs diff: {c9_check.json_minus_recompute.abs().median():.6f}")
print("  cycle-10:")
print(f"    matched months : {c10_check.json_minus_recompute.notna().sum()}")
print(f"    max abs diff   : {c10_check.json_minus_recompute.abs().max():.6f}")
print(f"    median abs diff: {c10_check.json_minus_recompute.abs().median():.6f}")

# Show worst-fit month
print(f"\n  cycle-10 top-5 worst json-vs-recompute mismatch:")
worst = c10_check.sort_values("json_minus_recompute", key=abs, ascending=False).head(5)
print(worst[["month", "n_cal", "mu_cal", "sigma_cal", "beta_qlike", "json_beta", "json_minus_recompute"]].round(5).to_string(index=False))



