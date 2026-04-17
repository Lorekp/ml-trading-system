"""Per-asset per-TF signal generation for live trading.

Mirrors circuit_breaker_v2_experiment.py:91-113 (_gen_one), but uses
predict_single_bar() for real-time inference instead of batch processing.

Model inference is delegated to a ModelBackend (default: SklearnBackend).
To use a different model type, implement the ModelBackend protocol and
pass it to SignalGenerator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig

from live.config import OHLCV_FEATURES
from live.core.interfaces import ModelBackend

logger = logging.getLogger(__name__)

# AFML bet sizing — same imports as model.py
try:
    from afml.bet_sizing import discretize_signal, get_signal
except ImportError:
    def get_signal(prob, num_classes=2):
        return 2.0 * prob - 1.0
    def discretize_signal(sig, step_size=0.1):
        return round(sig / step_size) * step_size


@dataclass
class Signal:
    """Signal output matching experiment format."""
    asset: str
    timeframe: str
    direction: float        # +1.0 or -1.0
    probability: float      # [0, 1]
    confidence: float       # abs(prob - 0.5)
    signal_strength: float  # discretized AFML signal
    hold_bars: int


class SklearnBackend:
    """Default model backend: joblib-serialized sklearn/xgboost models.

    Loads {ASSET}_model.joblib + {ASSET}_scaler.joblib pairs per timeframe.
    Works with any model that implements predict_proba() (LogisticRegression,
    XGBClassifier, RandomForest, etc.) and any scaler with transform().
    """

    def __init__(self, *, require_metadata: bool = False) -> None:
        self._models: dict[tuple[str, str], tuple] = {}  # (tf, asset) -> (model, scaler)
        # Baked per-(tf, asset) metadata from training. Each entry is the dict
        # loaded from `{asset}_meta.json`; absent entries fall back to defaults.
        self._meta: dict[tuple[str, str], dict] = {}
        # Fail-safe: if True, refuse to load models that lack a sidecar
        # `*_meta.json`. Prevents silent fallback to the hardcoded
        # `conf_threshold=0.08` and rolling vol-filter quantile for any legacy
        # model that hasn't been re-baked with training-time thresholds.
        self._require_metadata = bool(require_metadata)

    def load_models(self, model_dir: Path) -> int:
        loaded = 0
        skipped_no_meta = 0
        # Reset metadata on reload so stale entries do not linger
        self._meta = {}
        if not model_dir.exists():
            logger.error("Model dir not found: %s", model_dir)
            return 0

        for tf_dir in model_dir.iterdir():
            if not tf_dir.is_dir():
                continue
            tf_label = tf_dir.name
            for model_path in tf_dir.glob("*_model.joblib"):
                asset = model_path.stem.replace("_model", "")
                scaler_path = tf_dir / f"{asset}_scaler.joblib"
                if not scaler_path.exists():
                    continue

                meta_path = tf_dir / f"{asset}_meta.json"
                meta = None
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                    except Exception as e:
                        logger.warning("Failed to load metadata %s/%s: %s",
                                        tf_label, asset, e)
                        meta = None

                if meta is None and self._require_metadata:
                    # Hard fail-safe: no baked thresholds → refuse to load.
                    skipped_no_meta += 1
                    logger.warning(
                        "SKIP %s/%s: require_metadata=True and %s_meta.json missing; "
                        "model will NOT be loaded until re-baked.",
                        tf_label, asset, asset)
                    continue

                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                except Exception as e:
                    logger.warning("Failed to load %s/%s: %s", tf_label, asset, e)
                    continue

                self._models[(tf_label, asset)] = (model, scaler)
                loaded += 1
                if meta is not None:
                    self._meta[(tf_label, asset)] = meta

        msg = ("Loaded %d sklearn models from %s (%d with metadata"
               + (", %d skipped: no meta" if skipped_no_meta else "")
               + ")")
        if skipped_no_meta:
            logger.info(msg, loaded, model_dir, len(self._meta), skipped_no_meta)
        else:
            logger.info(msg, loaded, model_dir, len(self._meta))
        return loaded

    def get_meta(self, asset: str, tf: str) -> dict | None:
        """Return baked metadata for (tf, asset), or None if not available."""
        return self._meta.get((tf, asset))

    def predict(self, asset: str, tf: str, features: np.ndarray) -> float | None:
        key = (tf, asset)
        if key not in self._models:
            return None
        model, scaler = self._models[key]
        try:
            X = scaler.transform(features)
            return float(model.predict_proba(X)[0, 1])
        except Exception as e:
            logger.debug("Prediction failed for %s/%s: %s", asset, tf, e)
            return None

    def has_model(self, asset: str, tf: str) -> bool:
        return (tf, asset) in self._models

    def available_assets(self, tf: str) -> list[str]:
        return [asset for (t, asset) in self._models if t == tf]


class SignalGenerator:
    """Generates signals per-asset per-TF.

    Pipeline:
    1. compute_features(ohlcv) → feature matrix
    2. Vol filter (skip low-vol bars — matches backtest model.py:219-227)
    3. backend.predict(features) → probability
    4. AFML bet sizing → signal_strength
    5. Confidence filter
    6. Regime filter (4H EMA trend alignment)

    The model backend is pluggable — pass any ModelBackend implementation
    to use different model types (ONNX, PyTorch, ensemble, etc.).
    """

    def __init__(
        self,
        model_dir: str = "data/live/models",
        conf_threshold: float = 0.08,
        regime_ema_span: int = 20,
        feature_config_overrides: dict | None = None,
        backend: ModelBackend | None = None,
        vol_filter_enabled: bool = True,
        require_metadata: bool = False,
    ):
        self._model_dir = Path(model_dir)
        self._conf_threshold = conf_threshold
        self._regime_ema_span = regime_ema_span
        # If no backend is supplied, create the default and thread the flag
        # through. An externally-supplied backend is assumed to be configured
        # already.
        self._backend = backend or SklearnBackend(require_metadata=require_metadata)
        self._vol_filter_enabled = vol_filter_enabled

        # Read feature list from model metadata if available
        feature_list = list(OHLCV_FEATURES)
        metadata_path = self._model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                if "features" in meta and meta["features"]:
                    feature_list = meta["features"]
            except Exception:
                pass

        overrides = feature_config_overrides or {}
        self._feature_config = FeatureConfig(
            feature_list=feature_list,
            winsor_lo=overrides.get("winsor_lo", 0.01),
            winsor_hi=overrides.get("winsor_hi", 0.99),
            vol_filter_quantile=overrides.get("vol_filter_quantile", 0.50),
            vol_zscore_window=overrides.get("vol_zscore_window", 30),
            swing_lookback=overrides.get("swing_lookback", 20),
        )

    def load_models(self) -> int:
        """Load models via backend. Returns count loaded."""
        return self._backend.load_models(self._model_dir)

    def generate(
        self,
        asset: str,
        tf_label: str,
        ohlcv_df: pd.DataFrame,
        ohlcv_4h: Optional[pd.DataFrame],
        hold_bars: int,
    ) -> Optional[Signal]:
        """Generate one signal for one asset on one TF."""
        if not self._backend.has_model(asset, tf_label):
            return None

        if ohlcv_df is None or len(ohlcv_df) < 30:
            return None

        # 1. Vol filter: skip low-vol bars (matches backtest model.py:219-227)
        # Backtest only generates signals on bars in the top 50% of volatility
        # (measured as EWM std of returns, span=50, min_periods=20).
        #
        # The backtest computes the threshold over the full TRAINING WINDOW
        # (months of history). We prefer the baked training-time threshold from
        # the model sidecar metadata. If metadata is absent (legacy models)
        # we fall back to the rolling-lookback approach — strictly worse but
        # keeps the live trader running during a staged rollout.
        if self._vol_filter_enabled and "close" in ohlcv_df.columns:
            vol = ohlcv_df["close"].pct_change().dropna().ewm(
                span=50, min_periods=20).std()
            if len(vol) >= 20:
                current_vol = float(vol.iloc[-1])

                # Prefer baked threshold from training
                baked_vt: float | None = None
                baked_source: str = "rolling"
                meta = self._get_baked_meta(asset, tf_label)
                if meta is not None:
                    mv = meta.get("vol_threshold")
                    if mv is not None and np.isfinite(mv):
                        baked_vt = float(mv)
                        baked_source = "baked"

                if baked_vt is not None:
                    vol_threshold = baked_vt
                else:
                    vol_threshold = float(
                        vol.quantile(self._feature_config.vol_filter_quantile))

                # Backtest uses strict `>`; match it here (was `<=` previously)
                if (np.isfinite(current_vol) and np.isfinite(vol_threshold)
                        and not (current_vol > vol_threshold)):
                    logger.info("VOL_FILTER %s/%s: vol=%.6f <= thr=%.6f (%s)",
                                asset, tf_label, current_vol, vol_threshold,
                                baked_source)
                    return None

        # 2. Compute features
        try:
            feat = compute_features(ohlcv_df, self._feature_config)
        except Exception as e:
            logger.debug("Feature computation failed for %s/%s: %s", asset, tf_label, e)
            return None

        if len(feat) == 0:
            return None

        # 3. Extract last valid row
        last_row = feat.iloc[[-1]]
        feature_cols = [c for c in self._feature_config.feature_list if c in last_row.columns]
        if len(feature_cols) < len(self._feature_config.feature_list) * 0.8:
            return None

        row_values = last_row[feature_cols]
        if row_values.isna().any().any():
            return None

        # 4. Predict via backend
        prob = self._backend.predict(asset, tf_label, row_values.values)
        if prob is None:
            return None

        direction = 1.0 if prob > 0.5 else -1.0
        confidence = abs(prob - 0.5)
        logger.debug("PREDICT %s/%s: prob=%.4f conf=%.4f", asset, tf_label, prob, confidence)

        # 5. AFML bet sizing
        sig = get_signal(prob, num_classes=2)
        sig_disc = discretize_signal(sig, step_size=0.1)
        strength = abs(float(sig_disc))

        # 6. Confidence filter — prefer baked 70th-pct threshold from training
        # (matches backtest model.py:286-302). Fall back to the config default
        # (0.08) only when metadata is missing.
        conf_thr = self._conf_threshold
        conf_src = "default"
        meta = self._get_baked_meta(asset, tf_label)
        if meta is not None:
            mc = meta.get("conf_threshold")
            if mc is not None and np.isfinite(mc):
                conf_thr = float(mc)
                conf_src = "baked"

        if confidence < conf_thr:
            logger.info("FILTERED %s/%s: conf=%.4f (thr=%.4f %s) prob=%.4f dir=%+.0f",
                        asset, tf_label, confidence, conf_thr, conf_src, prob, direction)
            return None

        # 7. Regime filter (4H EMA trend alignment)
        if ohlcv_4h is not None and not self._check_regime(direction, ohlcv_4h):
            logger.info("REGIME_BLOCK %s/%s: dir=%+.0f conf=%.4f — 4H trend opposes",
                        asset, tf_label, direction, confidence)
            return None

        return Signal(
            asset=asset,
            timeframe=tf_label,
            direction=direction,
            probability=prob,
            confidence=confidence,
            signal_strength=strength,
            hold_bars=hold_bars,
        )

    def _check_regime(self, direction: float, ohlcv_4h: pd.DataFrame) -> bool:
        """4H EMA regime filter — exact logic from signal_engine/filters.py.

        Uses shift(1) to prevent look-ahead bias: the regime for a bar is
        only known AFTER it closes. Without shift, we'd see the current bar's
        close before it's complete.
        """
        if len(ohlcv_4h) < self._regime_ema_span + 5:
            return True

        close = ohlcv_4h["close"]
        ema = close.ewm(span=self._regime_ema_span, adjust=False).mean()

        regime_raw = np.where(close > ema, 1.0, -1.0)
        regime_shifted = pd.Series(regime_raw, index=ohlcv_4h.index).shift(1)

        latest_regime = regime_shifted.iloc[-1]
        if pd.isna(latest_regime):
            return True

        if direction > 0 and latest_regime < 0:
            return False
        if direction < 0 and latest_regime > 0:
            return False

        return True

    def has_model(self, asset: str, tf: str) -> bool:
        """Check if a model exists for this (asset, tf) pair."""
        return self._backend.has_model(asset, tf)

    def _get_baked_meta(self, asset: str, tf: str) -> dict | None:
        """Return baked training-time metadata from the backend, or None if
        the backend does not support it (e.g., legacy ONNX backend)."""
        getter = getattr(self._backend, "get_meta", None)
        if getter is None:
            return None
        try:
            return getter(asset, tf)
        except Exception:
            return None

    def available_assets(self, tf_label: str) -> list[str]:
        """Return list of assets with models for a given TF."""
        return self._backend.available_assets(tf_label)
