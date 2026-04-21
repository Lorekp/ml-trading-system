"""Label construction for volatility prediction.

Label: y = ln(vol_ann) from TSRV output, OR y = ln(vol_ann) - ln(IV_at_T)
(excess log-vol) when ``config.predict_excess_logvol`` is True.
Alignment: features at time t predict label at time t+1 (one-step-ahead).
"""

from __future__ import annotations

from pathlib import Path

import logging
import numpy as np
import pandas as pd

from vol_arb.paths import vol_arb_path as _vol_arb_path, CURRENCY_FOR_ASSET
from vol_model.config import VolModelConfig

logger = logging.getLogger(__name__)


def _load_iv_at_decision_time(asset: str, index: pd.Index, freq_s: int) -> pd.Series:
    """Load iv_used at the *decision* time for each label bar.

    At label bar ``t_i`` the model's decision is made at ``t_{i-1}`` (the
    feature's time), so iv_used(t_{i-1}) is the relevant reference IV for
    excess-logvol target construction. This function returns a Series indexed
    at the LABEL timestamps with the iv_used value that was observable at
    the corresponding decision time — i.e., pre-shifted by one freq.

    Uses the same fallback chain the backtester applies:
    ``svi_iv_matched.fillna(iv).fillna(dvol_close)`` so training and
    inference see the same IV definition.

    ``freq_s`` is the bar size in seconds (= ``config.freq_s``); the shift
    must equal the label horizon, so callers MUST pass the same freq_s the
    rest of the label pipeline uses. See horizon SSOT in
    ``VolModelConfig.horizon_minutes`` (plan: starry-rolling-rabin.md).
    """
    ccy = CURRENCY_FOR_ASSET[asset]
    path = _vol_arb_path(ccy, svi=True)
    if not path.exists():
        raise FileNotFoundError(
            f"Excess-logvol target needs {path}; build it via "
            f"`python -m vol_arb.run_pipeline --currency {ccy} --svi`."
        )
    va = pd.read_parquet(path, columns=["svi_iv_matched", "iv", "dvol_close"])
    iv_used = (
        va["svi_iv_matched"].fillna(va["iv"]).fillna(va["dvol_close"])
    ) / 100.0  # pct -> decimal
    # Shift +1 freq so that at label index t_i, the aligned value is the
    # iv_used observed at the decision time t_{i-1}. Horizon-derived from
    # freq_s — at h=15 this is +15m (legacy behaviour); at h=60 it's +60m.
    iv_used.index = iv_used.index + pd.Timedelta(seconds=freq_s)
    # Align to the label index (forward-fill is NOT appropriate here — if a
    # decision-time iv_used is missing, the row should drop out of training).
    return iv_used.reindex(index)


def load_labels(config: VolModelConfig) -> pd.DataFrame:
    """Load TSRV output and construct prediction labels.

    Returns DataFrame indexed by timestamp with columns:
        vol_ann: raw annualized volatility
        ln_vol: ln(vol_ann) — the regression target
        t1: label end time (index + freq, for PurgedKFold)

    When ``config.predict_excess_logvol`` is True, ``ln_vol`` is reparam'd to
    ``ln(vol_ann) - ln(iv_used_at_decision_time)``, i.e., excess log-vol.
    """
    tsrv = pd.read_parquet(config.get_tsrv_path())

    # Normalize TSRV timestamp convention to START-stamped (Stage 18 semantics):
    # at any label key T, `result.ln_vol[T]` must equal `log(vol of [T, T+freq))`.
    # The BTC TSRV parquet was migrated from START-stamped to END-stamped on
    # 2026-04-11 (see .pre_end_stamp_bak backup + LABEL_SHIFT_AUDIT.md). Under
    # END-stamp, bar at index T = vol of [T-freq, T); shifting the index by
    # -freq restores the START-stamp semantic assumed by align_features_labels'
    # iloc[1:] (one-bar-ahead forecast). Without this shift the label is a
    # *nowcast* of the already-closed bar — features trivially correlate with
    # their own trade window's realized vol. Per-asset flag in config.
    if getattr(config, "tsrv_end_stamped", False):
        freq_td = pd.Timedelta(seconds=config.freq_s)
        tsrv = tsrv.copy()
        tsrv.index = tsrv.index - freq_td

    # Drop zero-vol windows (ln(0) = -inf)
    valid = tsrv[tsrv["vol_ann"] > 0].copy()

    ln_vol = np.log(valid["vol_ann"])

    # Optional target reparameterization: subtract log(IV_at_decision_time)
    # BEFORE clipping, so the clip bounds reflect the reparam'd distribution.
    if getattr(config, "predict_excess_logvol", False):
        iv_decision = _load_iv_at_decision_time(
            config.asset, valid.index, config.freq_s
        )
        log_iv = np.log(iv_decision)
        ln_vol = ln_vol - log_iv
        # Rows where iv_used is missing at decision time become NaN — drop
        # them from the training set rather than fill (avoids silent bias).
        mask = ln_vol.notna()
        n_dropped = (~mask).sum()
        if n_dropped:
            logger.info(
                "[excess-logvol] Dropped %d / %d rows with missing "
                "decision-time iv_used (feature timestamps before SVI "
                "coverage begins, mostly).",
                n_dropped, len(ln_vol),
            )
        valid = valid[mask]
        ln_vol = ln_vol[mask]

    # Clip extremes to prevent QLIKE loss explosion
    lo = ln_vol.quantile(1 - config.label_clip_quantile)
    hi = ln_vol.quantile(config.label_clip_quantile)
    ln_vol = ln_vol.clip(lo, hi)

    # Raw variance components for TransVol ratio-form QLIKE (MTL targets)
    # Clipped at 99.9th percentile to prevent extreme outliers
    continuous_var = valid["continuous_var"].clip(lower=1e-20)
    hi_c = continuous_var.quantile(config.label_clip_quantile)
    continuous_var = continuous_var.clip(upper=hi_c)

    jump_var = valid["jump_var"].clip(lower=0.0)
    hi_j = jump_var.quantile(config.label_clip_quantile)
    jump_var = jump_var.clip(upper=hi_j)

    freq = pd.Timedelta(seconds=config.freq_s)

    result = pd.DataFrame({
        "vol_ann": valid["vol_ann"],
        "ln_vol": ln_vol,
        "continuous_var": continuous_var,
        "jump_var": jump_var,
        "t1": valid.index + freq,
    }, index=valid.index)

    return result


def align_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Align features and labels for one-step-ahead prediction.

    Features at time t predict label at time t+1.

    Returns:
        X: features (rows aligned to prediction timestamps)
        y: ln_vol labels (one step ahead of X)
        t1: label end times for PurgedKFold
    """
    # Find common timestamps
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    # One-step-ahead: features[:-1] predict labels[1:]
    X = features.iloc[:-1].copy()
    y = labels["ln_vol"].iloc[1:].copy()
    t1 = labels["t1"].iloc[1:].copy()

    # Re-align indices so X[i] and y[i] share the same index
    # The index represents the LABEL timestamp (what we're predicting)
    X.index = y.index

    return X, y, t1
