"""Walk-forward model: monthly expanding-window logistic regression.

Extracted from experiments/multi_approach_comparison.py (lines 541-735)
and experiments/expanded_portfolio_test.py (lines 485-665).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from afml.bet_sizing import discretize_signal, get_signal

from signal_engine.config import SystemConfig
from signal_engine.weights import compute_sample_weights

logger = logging.getLogger(__name__)


class WalkForwardModel:
    """Walk-forward monthly-retrain direction predictor.

    Produces raw (unfiltered) signals for a single asset using an
    expanding-window logistic regression with L1 regularisation, AFML
    sample weights, and per-window winsorisation.
    """

    def __init__(self, config: SystemConfig, force_monthly_retrain: bool = False) -> None:
        self._config = config
        self._force_monthly_retrain = force_monthly_retrain
        self._cached_model: LogisticRegression | None = None
        self._cached_scaler: StandardScaler | None = None
        # Last-retrain thresholds captured so the training script can bake them
        # into a sidecar metadata file. Live inference reads these to match
        # the backtest exactly (see src/live/signal_generator.py).
        self._cached_conf_threshold: float | None = None
        self._cached_vol_threshold: float | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> LogisticRegression | None:
        """Currently fitted model (for Phase 2 streaming)."""
        return self._cached_model

    @property
    def current_scaler(self) -> StandardScaler | None:
        """Currently fitted scaler (for Phase 2 streaming)."""
        return self._cached_scaler

    @property
    def last_conf_threshold(self) -> float | None:
        """Confidence threshold (70th pct of train |proba-0.5|) from the most
        recent retrain. None if no retrain has happened yet."""
        return self._cached_conf_threshold

    @property
    def last_vol_threshold(self) -> float | None:
        """Vol-filter threshold (50th pct of EWM-std returns over the training
        window) from the most recent retrain. None if vol filter did not run."""
        return self._cached_vol_threshold

    # ------------------------------------------------------------------
    # Single-bar inference (Phase 2 live)
    # ------------------------------------------------------------------

    def predict_single_bar(self, features_row: pd.DataFrame) -> dict:
        """Single-bar prediction using the cached model.

        Args:
            features_row: Single-row DataFrame with feature columns,
                matching the columns used during training.

        Returns:
            Dict with keys: probability, direction, confidence, signal_strength.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._cached_model is None or self._cached_scaler is None:
            raise RuntimeError(
                "Model not trained. Run generate_signals_batch first."
            )

        X = self._cached_scaler.transform(features_row.values)
        prob = float(self._cached_model.predict_proba(X)[0, 1])
        direction = 1.0 if prob > 0.5 else -1.0
        confidence = abs(prob - 0.5)

        sig = get_signal(prob, num_classes=2)
        sig_disc = discretize_signal(sig, step_size=0.1)

        return {
            "probability": prob,
            "direction": direction,
            "confidence": confidence,
            "signal_strength": abs(float(sig_disc)),
        }

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    def generate_signals_batch(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        asset_name: str,
    ) -> pd.DataFrame:
        """Walk-forward monthly retrain with expanding window.

        Returns a DataFrame of raw signals (no confidence filter applied)
        with columns: ``timestamp, direction, fwd_return, confidence,
        probability, signal_strength, asset``.

        Args:
            ohlcv: OHLCV data for the asset (used to compute forward returns
                and the volatility filter).
            features: Pre-computed feature DataFrame aligned to *ohlcv*.
            asset_name: Human-readable asset identifier (e.g. ``"BTC"``).

        Returns:
            Signals DataFrame, possibly empty if insufficient history.
        """
        cfg = self._config
        mcfg = cfg.model
        fcfg = cfg.features
        scfg = cfg.signal

        feature_cols = fcfg.feature_list
        fwd_bars = mcfg.fwd_bars

        close = ohlcv["close"]
        fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

        # Volatility for vol filter
        vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

        # Determine month range from the actual data
        start_date = ohlcv.index.min()
        end_date = ohlcv.index.max()

        # Need at least 6 months of data to start walk-forward
        if (end_date - start_date).days < 180:
            logger.info(
                "%s: insufficient history (%d days < 180)",
                asset_name,
                (end_date - start_date).days,
            )
            return pd.DataFrame()

        # Generate monthly periods from actual data range
        first_test_month = start_date + pd.DateOffset(months=6)
        all_months = pd.period_range(
            start=first_test_month.to_period("M"),
            end=end_date.to_period("M"),
            freq="M",
        )

        if len(all_months) == 0:
            return pd.DataFrame()

        # Retraining optimisation: quarterly for large datasets
        # (force_monthly_retrain overrides for backtest accuracy)
        retrain_every = 1
        if not self._force_monthly_retrain:
            if len(ohlcv) > 500_000:
                retrain_every = 3
            elif len(ohlcv) > 200_000:
                retrain_every = 2

        # Compute embargo timedelta.  A label at bar t depends on bars
        # [t+1 ... t+fwd_bars], so the last training sample whose label
        # is safe is at t_cutoff − (fwd_bars + 1) * bar_gap.
        if len(ohlcv) > 1:
            median_gap = ohlcv.index.to_series().diff().dropna().median()
            embargo_td = median_gap * (fwd_bars + 1)
        else:
            embargo_td = pd.Timedelta(hours=2)

        rows: list[pd.DataFrame] = []
        cached_model: LogisticRegression | None = None
        cached_scaler: StandardScaler | None = None
        cached_conf_threshold: float = 0.0
        last_vol_threshold: float | None = None
        last_retrain_idx = -999

        for month_idx, month in enumerate(all_months):
            ms = pd.Timestamp(month.start_time, tz="UTC")
            me = pd.Timestamp(month.end_time, tz="UTC")

            # Ensure timezone compatibility
            if features.index.tz is None:
                ms = ms.tz_localize(None)
                me = me.tz_localize(None)
            embargo_cutoff = ms - embargo_td

            # Train cutoff with embargo
            train_cutoff = embargo_cutoff

            # Select features available up to train_cutoff
            feat_cols = [c for c in feature_cols if c in features.columns]
            if len(feat_cols) < 8:
                continue  # need at least 8 features

            feat = features[feat_cols].copy()

            # Per-window winsorisation of returns using training-period quantiles
            train_mask = feat.index < train_cutoff
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    train_vals = feat.loc[train_mask, col]
                    if len(train_vals.dropna()) > 10:
                        lo = train_vals.quantile(fcfg.winsor_lo)
                        hi = train_vals.quantile(fcfg.winsor_hi)
                        feat[col] = feat[col].clip(lo, hi)

            # Clean: drop rows with any NaN feature
            fc = feat.dropna()
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)

            # Train / test split
            tr = valid[valid < train_cutoff]
            te = valid[(valid >= ms) & (valid <= me)]

            if len(tr) < 200 or len(te) < 10:
                continue

            # Vol filter: top N% volatility bars in test set
            if vol is not None and len(vol) > 0:
                tv = vol.loc[vol.index.intersection(tr)].dropna()
                if len(tv) < 20:
                    continue
                vt = tv.quantile(fcfg.vol_filter_quantile)
                last_vol_threshold = float(vt)
                vol_aligned = vol.reindex(te)
                hv = (vol_aligned > vt).fillna(False)
                te = te[hv.reindex(te).fillna(False).values]
            else:
                continue

            if len(te) < 5:
                continue

            X_train = fc.loc[tr]
            y_train = (y_all.loc[tr] > 0).astype(int)
            X_test = fc.loc[te]

            # Only retrain every retrain_every months
            need_retrain = (
                (month_idx - last_retrain_idx) >= retrain_every
                or cached_model is None
            )

            if need_retrain:
                # Sample weights (AFML Ch4)
                train_timestamps = X_train.index.values.astype(np.int64)
                sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars)

            # Fit LogReg L1 (or reuse cached model for intermediate months)
            try:
                if need_retrain:
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_train)
                    # sklearn >=1.8 deprecates penalty= in favour of l1_ratio=
                    import sklearn
                    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8):
                        l1_ratio = 1.0 if mcfg.penalty == "l1" else (0.5 if mcfg.penalty == "elasticnet" else 0.0)
                        model = LogisticRegression(
                            C=mcfg.C,
                            l1_ratio=l1_ratio,
                            solver=mcfg.solver,
                            max_iter=mcfg.max_iter,
                            random_state=mcfg.seed,
                        )
                    else:
                        model = LogisticRegression(
                            C=mcfg.C,
                            penalty=mcfg.penalty,
                            solver=mcfg.solver,
                            max_iter=mcfg.max_iter,
                            random_state=mcfg.seed,
                        )
                    model.fit(X_tr_scaled, y_train, sample_weight=sample_w)
                    cached_model = model
                    cached_scaler = scaler
                    last_retrain_idx = month_idx
                else:
                    scaler = cached_scaler  # type: ignore[assignment]
                    model = cached_model  # type: ignore[assignment]

                X_te_scaled = scaler.transform(X_test)  # type: ignore[union-attr]
                yp = model.predict_proba(X_te_scaled)[:, 1]  # type: ignore[union-attr]

                # Per-window confidence threshold from TRAINING predictions
                # (avoids leakage — threshold based only on past data)
                if need_retrain:
                    train_proba = model.predict_proba(X_tr_scaled)[:, 1]  # type: ignore[union-attr]
                    train_conf = np.abs(train_proba - 0.5)
                    cached_conf_threshold = float(np.percentile(train_conf, 70.0))

            except Exception:
                continue

            conf = np.abs(yp - 0.5)
            fwd_ret_vals = fwd_ret.reindex(te)

            directions = np.where(yp > 0.5, 1.0, -1.0)
            signals = get_signal(yp, num_classes=2)
            signals_disc = discretize_signal(signals, step_size=scfg.step_size)

            # Per-window confidence filter (using training-derived threshold)
            conf_mask = conf >= cached_conf_threshold
            # Combine with finite forward return check
            fr_vals = (
                fwd_ret_vals.values
                if hasattr(fwd_ret_vals, "values")
                else np.array(fwd_ret_vals)
            )
            finite_mask = np.isfinite(fr_vals) & conf_mask
            if finite_mask.any():
                sig_str = (
                    signals_disc
                    if np.ndim(signals_disc) > 0
                    else np.full(len(yp), float(signals_disc))
                )
                batch = pd.DataFrame(
                    {
                        "timestamp": te[finite_mask],
                        "direction": directions[finite_mask],
                        "fwd_return": fr_vals[finite_mask],
                        "confidence": conf[finite_mask],
                        "probability": yp[finite_mask],
                        "signal_strength": np.abs(sig_str[finite_mask]),
                    }
                )
                rows.append(batch)

        if not rows:
            return pd.DataFrame()

        df = pd.concat(rows, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["asset"] = asset_name

        # Persist latest model for Phase 2 streaming
        self._cached_model = cached_model
        self._cached_scaler = cached_scaler
        # Persist last-retrain thresholds for metadata bake
        self._cached_conf_threshold = (
            float(cached_conf_threshold) if cached_conf_threshold else None
        )
        self._cached_vol_threshold = last_vol_threshold

        logger.info(
            "%s: generated %d raw signals over %d months",
            asset_name,
            len(df),
            len(all_months),
        )
        return df
