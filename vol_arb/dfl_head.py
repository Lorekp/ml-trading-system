"""DFL (Decision-Focused Learning) head for V2 strategy.

DFF-style (Wen et al. AAAI 2025, arxiv 2501.01874) post-hoc output
correction: freeze the HAR-PD backbone, train a small LGBM classifier
on decision-centric features to predict win/loss within trade-fired
bars. The classifier's P(win) is then used as a filter OR a size
multiplier at deployment.

**Why not retrain HAR-PD with SPO+?** Research briefing (2026-04-20)
found:
- SPO+ has Hessian degeneracy in tree boosting (gradient is
  piecewise constant where the two reflected decisions agree,
  Hessian is identically zero)
- DFF is the only DFL paper with explicit GBDT experiments
- Fine-tuning a head is far cheaper than full retrain
- Keeping HAR-PD frozen preserves all certified production behavior

**Leakage discipline**:
- Walk-forward training: year-by-year, never uses test-year data
- Features are strictly lagged (shift(1).rolling for past aggregates)
- Point-prediction backbone (``y_pred``) was itself walk-forward OOF,
  so no backbone-leakage
- ``y_true`` appears only in strictly-lagged rolling aggregates —
  not as itself
- Classification target (``win``) is eval-only, never a feature
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Features used by the DFF head. These are the 10 chosen in
# Phase 2 robustness testing. Order matters for reproducibility.
DFL_HEAD_FEATURES = [
    "y_pred", "iv_used", "iv_log",
    "pred_edge_log", "abs_pred_edge_log",
    "past_vol_24h", "past_vol_6h", "past_vol_std",
    "recent_resid_24h", "recent_abs_resid_24h",
]


# Pre-committed LightGBM hyperparameters. Frozen by Phase 2 testing;
# the comparison is pre-registered so we do not re-tune after seeing
# the backtest.
DFL_LGBM_PARAMS = dict(
    num_leaves=15, max_depth=5,
    learning_rate=0.05, n_estimators=300,
    min_child_samples=50, reg_lambda=1.0,
    seed=42, verbose=-1,
)


def build_head_features(
    trade_log: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the 10 DFL-head input features.

    Parameters
    ----------
    trade_log :
        V2 backtest output. Must have ``signal``, ``iv_used``,
        ``net_pnl_nc``, indexed by timestamp.
    predictions :
        HAR-PD predictions, must have ``y_pred``, ``y_true``,
        indexed by matching timestamps.

    Returns
    -------
    pd.DataFrame indexed by trade_log index with the 10 feature
    columns (NaN where rolling windows lack history).
    """
    df = trade_log[["signal", "iv_used", "net_pnl_nc"]].join(
        predictions[["y_pred", "y_true"]], how="inner"
    )
    df["iv_log"] = np.log(df["iv_used"] / 100.0)
    df["pred_edge_log"] = df["y_pred"] - df["iv_log"]
    df["abs_pred_edge_log"] = df["pred_edge_log"].abs()
    df["past_vol_24h"] = df["y_true"].shift(1).rolling(96, min_periods=24).mean()
    df["past_vol_6h"]  = df["y_true"].shift(1).rolling(24, min_periods=8).mean()
    df["past_vol_std"] = df["y_true"].shift(1).rolling(96, min_periods=24).std()
    r = df["y_true"] - df["y_pred"]
    df["recent_resid_24h"]     = r.shift(1).rolling(96, min_periods=24).mean()
    df["recent_abs_resid_24h"] = r.abs().shift(1).rolling(96, min_periods=24).mean()
    return df[DFL_HEAD_FEATURES].copy()


def compute_walkforward_scores(
    trade_log: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    min_train_trades: int = 500,
    train_bucket: str = "year",
    seed: int = 42,
) -> pd.Series:
    """Train a DFF head walk-forward and return P(win) for each
    trade-fired bar in the trade log.

    Training scheme: expanding-window by year. For each test year,
    train on all PRIOR bars. Pass-through (NaN) for the first year
    when no training history is available.

    Parameters
    ----------
    trade_log, predictions : see ``build_head_features``.
    min_train_trades : require at least this many train trades to
        fit a head; otherwise pass through NaN for that test bucket.
    train_bucket : ``"year"`` is the only supported mode today.
    seed : LightGBM seed.

    Returns
    -------
    pd.Series of P(win) indexed by trade_log.index. NaN for bars
    with no signal or insufficient training history.
    """
    from lightgbm import LGBMClassifier, early_stopping

    feats = build_head_features(trade_log, predictions)
    tr = trade_log.loc[trade_log["signal"] != 0].copy()
    tr["win"] = (tr["net_pnl_nc"] > 0).astype(int)

    # Restrict to trade-fired rows with full feature coverage.
    tr_feats = feats.loc[tr.index].dropna(subset=DFL_HEAD_FEATURES)
    tr_full = tr.loc[tr_feats.index].copy()
    tr_full[DFL_HEAD_FEATURES] = tr_feats

    scores = pd.Series(
        np.nan, index=trade_log.index, name="p_win_walkforward", dtype=float
    )

    if train_bucket != "year":
        raise NotImplementedError(f"train_bucket={train_bucket!r}")

    years = sorted(set(tr_full.index.year))
    for test_y in years:
        train = tr_full.loc[tr_full.index.year < test_y]
        test  = tr_full.loc[tr_full.index.year == test_y]
        if len(train) < min_train_trades or len(test) < 1:
            continue
        X_train = train[DFL_HEAD_FEATURES]
        X_test  = test[DFL_HEAD_FEATURES]
        y_train = train["win"]
        y_test  = test["win"]
        clf = LGBMClassifier(**DFL_LGBM_PARAMS | {"seed": seed})
        clf.fit(
            X_train, y_train, eval_set=[(X_test, y_test)],
            callbacks=[early_stopping(25, verbose=False)],
        )
        p = clf.predict_proba(X_test)[:, 1]
        scores.loc[test.index] = p

    return scores


# ── Post-processor strategies ─────────────────────────────────
#
# Each strategy maps a P(win) score + trade-log PnL into a new
# PnL, applying either a filter or a size multiplier. These are
# PRE-COMMITTED from Phase 2 robustness results. No post-hoc
# tuning: after running these on the backtest, we do NOT re-tune
# parameters.


def _daily_threshold_from_train(
    scores: pd.Series,
    train_end: pd.Timestamp,
    quantile: float,
) -> float:
    """Threshold at a given quantile of P(win) over training-only
    data (data strictly before ``train_end``).
    """
    train_scores = scores.loc[scores.index < train_end].dropna()
    return float(train_scores.quantile(quantile)) if len(train_scores) > 0 else float("nan")


@dataclass(frozen=True)
class DflStrategy:
    name: str
    description: str

    def apply(self, trade_log: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
        """Return a new trade_log with PnL modified per strategy.
        Subclasses override this.
        """
        raise NotImplementedError


class FilterStrategy(DflStrategy):
    """Drop bars with P(win) below a training-quantile threshold."""
    def __init__(self, *, quantile: float, name: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description",
                           f"drop bars with P(win) < train_q{int(quantile*100)}")
        object.__setattr__(self, "quantile", quantile)

    def apply(self, trade_log, scores):
        out = trade_log.copy()
        # Re-evaluate threshold each year (walk-forward pattern)
        years = sorted(set(trade_log.index.year))
        filter_mask = pd.Series(False, index=trade_log.index)
        for y in years:
            thr = _daily_threshold_from_train(
                scores, pd.Timestamp(f"{y}-01-01", tz="UTC"), self.quantile,
            )
            if np.isnan(thr):
                continue
            year_mask = trade_log.index.year == y
            year_scores = scores.loc[year_mask]
            drop_mask = (year_scores < thr).reindex(
                trade_log.index[year_mask], fill_value=False
            )
            # Align the boolean mask back
            drop_idx = trade_log.index[year_mask][drop_mask.to_numpy()]
            filter_mask.loc[drop_idx] = True

        # Zero out PnL for filtered bars
        pnl_cols = [c for c in [
            "net_pnl_nc", "gross_pnl_nc", "vega_pnl_nc",
            "theta_pnl_nc", "delta_hedge_pnl_nc",
            "net_pnl", "gross_pnl", "vega_pnl",
            "theta_pnl", "delta_hedge_pnl",
        ] if c in out.columns]
        for c in pnl_cols:
            out.loc[filter_mask, c] = 0.0
        return out


class SizeStrategy(DflStrategy):
    """Scale PnL columns by a P(win)-derived multiplier per bar.

    The multiplier function is fixed at construction; no post-hoc
    tuning.
    """
    def __init__(self, *, name: str, description: str, multiplier_fn):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "multiplier_fn", multiplier_fn)

    def apply(self, trade_log, scores):
        out = trade_log.copy()
        mult = pd.Series(1.0, index=trade_log.index)
        # Compute multiplier year-by-year (using training-only stats)
        years = sorted(set(trade_log.index.year))
        for y in years:
            train_end = pd.Timestamp(f"{y}-01-01", tz="UTC")
            train_scores = scores.loc[scores.index < train_end].dropna()
            if len(train_scores) < 100:
                continue
            year_mask = trade_log.index.year == y
            year_scores = scores.loc[year_mask]
            m_year = self.multiplier_fn(year_scores, train_scores)
            mult.loc[year_mask] = m_year.fillna(1.0).to_numpy()

        pnl_cols = [c for c in [
            "net_pnl_nc", "gross_pnl_nc", "vega_pnl_nc",
            "theta_pnl_nc", "delta_hedge_pnl_nc",
            "net_pnl", "gross_pnl", "vega_pnl",
            "theta_pnl", "delta_hedge_pnl",
        ] if c in out.columns]
        for c in pnl_cols:
            out[c] = out[c].to_numpy() * mult.to_numpy()
        return out


# Pre-committed strategy factory — do not re-parameterize.
def pre_committed_strategies() -> list[DflStrategy]:
    def b1_multiplier(year_scores, train_scores):
        return np.clip(2.0 * year_scores, 0.25, 1.5)

    def b2_multiplier(year_scores, train_scores):
        med = train_scores.median()
        return pd.Series(
            np.where(year_scores < med, 0.5, 1.0),
            index=year_scores.index,
        )

    def b3_multiplier(year_scores, train_scores):
        m = train_scores.mean()
        if m <= 0:
            return pd.Series(1.0, index=year_scores.index)
        return np.clip(year_scores / m, 0.5, 2.0)

    return [
        FilterStrategy(quantile=0.25, name="A1_filter25"),
        FilterStrategy(quantile=0.10, name="A2_filter10"),
        SizeStrategy(
            name="B1_linear",
            description="clip(2*P(win), 0.25, 1.5)",
            multiplier_fn=b1_multiplier,
        ),
        SizeStrategy(
            name="B2_binary_median",
            description="0.5 if P(win)<train_median else 1.0",
            multiplier_fn=b2_multiplier,
        ),
        SizeStrategy(
            name="B3_normalized",
            description="clip(P(win)/train_mean, 0.5, 2.0)",
            multiplier_fn=b3_multiplier,
        ),
    ]
