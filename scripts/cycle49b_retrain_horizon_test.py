"""Cycle-49b (Phase 5/6, 2026-05-18): retrain-at-horizon make-or-break test.

The probe (cycle-49) showed the production model — which forecasts
15-MIN realized vol — barely beats raw IV at forecasting multi-day RV
(+1.1% incremental R2). But that model is HANDICAPPED: trained for the
wrong horizon, used as a multi-day proxy.

This test removes the handicap: train LightGBM DIRECTLY on a multi-day
RV label, MONTHLY walk-forward (matches the production retrain cadence
so the model is never stale), head-to-head against IV.

Sanity check: the same harness runs at H=15min, where the production
model is known to score R2~0.59 — if the harness reproduces that, the
multi-day result is trustworthy.

Decision:
  - horizon-matched model beats IV / adds large incremental R2
        -> real ML vol alpha; full production retrain justified.
  - model only ties or loses to IV
        -> ML-vol direction dead; edge (if any) is pure IV carry.

Features: BTCUSDT_features_15m.parquet — 126 pure historical features
(TSRV / HAR-PD / microstructure), NO IV -> clean head-to-head.
Label: log(annualised RV over [T, T+H]) from TSRV vol_ann.

Usage:  PYTHONPATH=. python scripts/cycle49b_retrain_horizon_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    raise SystemExit("lightgbm required")

FEAT = "vol_model/output/BTCUSDT_features_15m.parquet"
SVI = "vol_arb/output/BTC_vol_arb_svi.parquet"
BARS_PER_DAY = 96

LGB_PARAMS = dict(
    objective="regression", metric="l2", num_leaves=63, learning_rate=0.04,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50, verbosity=-1, n_jobs=-1,
)


def fwd_log_rv(vol_ann: pd.Series, h_bars: int) -> pd.Series:
    var = vol_ann.astype(float) ** 2
    fwd = var.rolling(h_bars).mean().shift(-h_bars)
    out = np.log(np.sqrt(fwd) * 100.0)
    return out.replace([np.inf, -np.inf], np.nan)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def fit_lgb(tr: pd.DataFrame, va: pd.DataFrame, cols: list[str]) -> lgb.Booster:
    ds_tr = lgb.Dataset(tr[cols], label=tr["y"])
    ds_va = lgb.Dataset(va[cols], label=va["y"])
    return lgb.train(LGB_PARAMS, ds_tr, num_boost_round=700,
                     valid_sets=[ds_va],
                     callbacks=[lgb.early_stopping(50, verbose=False),
                                lgb.log_evaluation(0)])


def walk_forward(df: pd.DataFrame, feat_cols: list[str], h_bars: int,
                 with_iv: bool) -> dict:
    """Monthly expanding-window walk-forward. Returns pooled OOF arrays."""
    embargo = pd.Timedelta(minutes=15 * h_bars)
    months = sorted({(ts.year, ts.month) for ts in df.index})
    oof = {"y": [], "m_feat": [], "m_both": [], "iv": []}
    for yr, mo in months:
        test = df[(df.index.year == yr) & (df.index.month == mo)]
        if len(test) < 400:
            continue
        train = df[df.index < test.index[0] - embargo]
        if len(train) < 20000:
            continue
        cut = int(len(train) * 0.85)
        tr, va = train.iloc[:cut], train.iloc[cut:]
        m_feat = fit_lgb(tr, va, feat_cols)
        oof["y"].append(test["y"].to_numpy())
        oof["m_feat"].append(m_feat.predict(test[feat_cols]))
        b = np.polyfit(np.log(train["iv"]), train["y"], 1)
        oof["iv"].append(np.polyval(b, np.log(test["iv"].to_numpy())))
        if with_iv:
            m_both = fit_lgb(tr, va, feat_cols + ["iv"])
            oof["m_both"].append(m_both.predict(test[feat_cols + ["iv"]]))
    return {k: np.concatenate(v) for k, v in oof.items() if v}


def main() -> int:
    X = pd.read_parquet(FEAT)
    X.index = pd.to_datetime(X.index, utc=True)
    svi = pd.read_parquet(SVI)
    svi.index = pd.to_datetime(svi.index, utc=True)
    vol_ann = svi["vol_ann"].astype(float)
    iv = svi["iv"].astype(float)
    feat_cols = [c for c in X.columns if X[c].dtype.kind in "fiu"]

    print("=" * 78)
    print("CYCLE-49b RETRAIN-AT-HORIZON TEST  (monthly walk-forward, vs IV)")
    print("=" * 78)
    print(f"\n{len(feat_cols)} historical features, no IV. Monthly retrain.\n")

    for h_label, h_bars, with_iv in [("15min-SANITY", 1, False),
                                     ("1-day", BARS_PER_DAY, True),
                                     ("3-day", 3 * BARS_PER_DAY, True)]:
        y = fwd_log_rv(vol_ann, h_bars)
        df = (X[feat_cols].join(y.rename("y"), how="inner")
              .join(iv.rename("iv"), how="inner"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "iv"])
        df = df[df["iv"] > 0].sort_index()

        oof = walk_forward(df, feat_cols, h_bars, with_iv)
        yt = oof["y"]
        r_feat = r2(yt, oof["m_feat"])
        r_iv = r2(yt, oof["iv"])
        print(f"--- H = {h_label}  (pooled OOS, n={len(yt)}, log-space R2) ---")
        print(f"  IV alone (option market's forecast) : R2 = {r_iv:7.4f}")
        print(f"  ML model, {len(feat_cols)} historical features  : R2 = {r_feat:7.4f}"
              f"   ({'BEATS' if r_feat > r_iv else 'loses to'} IV)")
        if with_iv and "m_both" in oof:
            r_both = r2(yt, oof["m_both"])
            print(f"  ML model + IV (combined)            : R2 = {r_both:7.4f}")
            print(f"  incremental R2 of ML over IV        : {r_both - r_iv:+7.4f}")
        print()

    print("=" * 78)
    print("Read: 15min-SANITY should land near R2~0.5-0.6 (production gets")
    print("0.59) — if so the harness is trustworthy. Then: does the 1-day")
    print("model beat IV? If yes -> ML vol alpha exists. If no -> ML-vol dead.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
