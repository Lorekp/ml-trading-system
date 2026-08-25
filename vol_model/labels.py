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


def detect_stamping(index: pd.DatetimeIndex, freq_s: int) -> str:
    """Detect whether a bar parquet is START- or END-stamped from its first bar.

    A bar covering the realised window ``[t, t+freq)`` is either keyed at its
    window START (``t``, START-stamped) or its window END (``t+freq``,
    END-stamped). For a series that begins at midnight UTC, the first bar's
    offset-from-midnight is ``0`` when START-stamped and ``freq`` when
    END-stamped.

    This is the SSOT for the stamping-aware alignment fix (MF-2). The global
    ``config.tsrv_end_stamped`` flag is NOT trusted: it is set ``True`` for
    BTCUSDT but only ``BTCUSDT_tsrv_15m.parquet`` is actually END-stamped; all
    other-horizon TSRV + HAR-PD sidecars remain START-stamped. Detecting the
    true stamping per-file makes the alignment shift correct at every horizon
    with a pure code change (no data re-stamp). The expected per-file
    distinction is covered by ``tests/test_label_semantics.py``.

    Returns ``"END"``, ``"START"``, or ``"OTHER"`` (anything else — caller
    should treat as ambiguous).
    """
    if len(index) == 0:
        return "OTHER"
    first = index[0]
    off_s = (first - first.normalize()).total_seconds()
    if abs(off_s - freq_s) < 1e-6:
        return "END"
    if abs(off_s) < 1e-6:
        return "START"
    return "OTHER"


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
    rest of the label pipeline uses. The horizon source of truth is
    ``VolModelConfig.horizon_minutes``.
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


def load_labels(
    config: VolModelConfig, *, clip: bool = True,
) -> pd.DataFrame:
    """Load TSRV output and construct prediction labels.

    Returns DataFrame indexed by timestamp with columns:
        vol_ann: raw annualized volatility
        ln_vol: ln(vol_ann) — the regression target
        t1: label end time (index + freq, for PurgedKFold)

    When ``config.predict_excess_logvol`` is True, ``ln_vol`` is reparam'd to
    ``ln(vol_ann) - ln(iv_used_at_decision_time)``, i.e., excess log-vol.

    Parameters
    ----------
    clip : bool, default True
        If True, clip ln_vol / continuous_var / jump_var at the
        ``config.label_clip_quantile`` quantiles fitted on the FULL series.
        This is structurally a mild leak (test-set tails influence the
        clip bounds applied to training labels) — kept on by default for
        backward compatibility with non-walk-forward callers. Walk-forward
        callers should pass ``clip=False`` and apply
        :func:`clip_label_quantiles` on their per-fold training slice
        instead. Cycle 9 fix (2026-05-08).
    """
    tsrv = pd.read_parquet(config.get_tsrv_path())

    # Normalize TSRV timestamp convention to START-stamped (Stage 18 semantics):
    # at any label key T, `result.ln_vol[T]` must equal `log(vol of [T, T+freq))`.
    # The BTC 15m TSRV parquet was migrated from START-stamped to END-stamped on
    # 2026-04-11. Under
    # END-stamp, bar at index T = vol of [T-freq, T); shifting the index by
    # -freq restores the START-stamp semantic assumed by align_features_labels'
    # iloc[1:] (one-bar-ahead forecast). Without this shift the label is a
    # *nowcast* of the already-closed bar — features trivially correlate with
    # their own trade window's realized vol.
    #
    # MF-2 STAMPING-AWARE FIX (2026-06-21): the global config flag
    # `tsrv_end_stamped` is True for BTCUSDT but only the 15m TSRV parquet was
    # actually migrated; all other horizons (5/30/60/240/480/1440m) remain
    # START-stamped. Trusting the flag unconditionally shifted START-stamped
    # labels one bar FORWARD -> the target became vol of [T+freq, T+2*freq)
    # instead of [T, T+freq) (proven at h=60). We now DETECT the true stamping
    # per-file and shift -freq ONLY when the parquet is genuinely END-stamped.
    # h=15 (the only END-stamped file) stays byte-identical; every other
    # horizon now correctly targets [T, T+freq). The config flag is retained as
    # an opt-OUT safety belt: if a future migration end-stamps everything, set
    # it False to suppress the shift; we never shift when it is False.
    freq_td = pd.Timedelta(seconds=config.freq_s)
    stamping = detect_stamping(tsrv.index, config.freq_s)
    if getattr(config, "tsrv_end_stamped", False) and stamping == "END":
        tsrv = tsrv.copy()
        tsrv.index = tsrv.index - freq_td
    elif getattr(config, "tsrv_end_stamped", False) and stamping != "END":
        logger.info(
            "[labels] config.tsrv_end_stamped=True but %s is %s-stamped "
            "(freq=%ds) — NOT applying -freq shift (MF-2 stamping-aware). "
            "Label target is [T, T+freq) as required.",
            config.get_tsrv_path(), stamping, config.freq_s,
        )

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

    # Variance components for TransVol ratio-form QLIKE (MTL targets).
    # Lower clips at 1e-20 / 0.0 are not data-snooping (constants).
    continuous_var = valid["continuous_var"].clip(lower=1e-20)
    jump_var = valid["jump_var"].clip(lower=0.0)

    if clip:
        # Global clip — fitted on the FULL series. STRUCTURALLY DATA-
        # SNOOPING (test-set tails leak into the clip bounds applied to
        # training labels), but kept on by default for backward compat.
        # Magnitude is small (only worst 0.1% of bars hit the clip at
        # the default label_clip_quantile=0.999). Cycle 9 (2026-05-08):
        # walk-forward callers should opt out via ``clip=False`` and
        # apply ``clip_label_quantiles`` on their per-fold training
        # slice instead. See ``run_walkforward_pruned_svi_har_pd.py``
        # for the canonical leak-safe pattern.
        lo = ln_vol.quantile(1 - config.label_clip_quantile)
        hi = ln_vol.quantile(config.label_clip_quantile)
        ln_vol = ln_vol.clip(lo, hi)

        hi_c = continuous_var.quantile(config.label_clip_quantile)
        continuous_var = continuous_var.clip(upper=hi_c)

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


def clip_label_quantiles(
    labels: pd.DataFrame,
    *,
    train_idx: pd.Index,
    quantile: float,
) -> pd.DataFrame:
    """Clip ln_vol / continuous_var / jump_var using quantiles fitted on
    ``train_idx`` ONLY, then applied to the full label DataFrame.

    Leakage-safe replacement for the inline clip in :func:`load_labels`
    (which fits quantiles on the full series, mildly leaking test-set
    tails into the training clip bounds). Use this in walk-forward
    training: compute clip bounds on the training fold's quantiles,
    apply to both train AND test (test clipping is fine — only the
    bound estimation must be train-only).

    Parameters
    ----------
    labels : DataFrame
        From ``load_labels(config, clip=False)``. Must contain ln_vol;
        continuous_var / jump_var clipped if present.
    train_idx : Index
        Timestamps in the training fold. Quantiles fit on
        ``labels.loc[labels.index.intersection(train_idx)]``.
    quantile : float
        Upper quantile (e.g. 0.999). Lower bound for ln_vol uses
        ``1 - quantile`` symmetrically.

    Returns
    -------
    DataFrame
        Copy of ``labels`` with ln_vol / continuous_var / jump_var
        clipped at train-fold quantiles.
    """
    out = labels.copy()
    train_overlap = labels.index.intersection(train_idx)
    if len(train_overlap) == 0:
        raise ValueError(
            "clip_label_quantiles: train_idx has no overlap with labels.index"
        )
    train_slice = labels.loc[train_overlap]

    train_ln_vol = train_slice["ln_vol"]
    lo = float(train_ln_vol.quantile(1.0 - quantile))
    hi = float(train_ln_vol.quantile(quantile))
    out["ln_vol"] = labels["ln_vol"].clip(lo, hi)

    if "continuous_var" in labels.columns:
        hi_c = float(train_slice["continuous_var"].quantile(quantile))
        out["continuous_var"] = labels["continuous_var"].clip(
            lower=1e-20, upper=hi_c,
        )

    if "jump_var" in labels.columns:
        hi_j = float(train_slice["jump_var"].quantile(quantile))
        out["jump_var"] = labels["jump_var"].clip(
            lower=0.0, upper=hi_j,
        )

    return out


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
