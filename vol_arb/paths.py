"""Asset-aware artifact path resolver for the vol-arb + vol-model stack.

Single source of truth so BTC ↔ ETH runs produce zero path collisions. BTC
keeps its legacy unsuffixed paths for backward compatibility; non-BTC
currencies (ETH today, potentially SOL/XRP later) get a currency/asset
token baked into every artifact name.

Rules:
  - Functions that are PER-CURRENCY take `currency` (e.g. "BTC", "ETH").
    These name shared-surface artifacts: SVI features, bar-exec quality,
    vol-arb dataset, DVOL, SAC checkpoint dirs, etc.
  - Functions that are PER-ASSET take `asset` (e.g. "BTCUSDT", "ETHUSDT").
    These name model artifacts tied to a specific training target:
    features, predictions, SHAP pruning, TransVol checkpoints, etc.
  - `CURRENCY_FOR_ASSET` and `ASSET_FOR_CURRENCY` keep the mapping in one
    place; call sites don't hand-roll string conversions.
"""

from __future__ import annotations

from pathlib import Path


CURRENCY_FOR_ASSET = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
ASSET_FOR_CURRENCY = {v: k for k, v in CURRENCY_FOR_ASSET.items()}


def _ccy(asset: str | None = None, currency: str | None = None) -> str:
    if currency is not None:
        return currency
    if asset is not None:
        return CURRENCY_FOR_ASSET[asset]
    raise ValueError("must provide asset or currency")


def _asset(asset: str | None = None, currency: str | None = None) -> str:
    if asset is not None:
        return asset
    if currency is not None:
        return ASSET_FOR_CURRENCY[currency]
    raise ValueError("must provide asset or currency")


# ── Vol-arb shared-surface artifacts (per currency) ─────────────────────

def vol_arb_path(currency: str = "BTC", svi: bool = True) -> Path:
    """Merged vol-arb dataset (option IV × TSRV × DVOL + optional SVI)."""
    stem = f"{currency}_vol_arb_svi" if svi else f"{currency}_vol_arb"
    return Path(f"vol_arb/output/{stem}.parquet")


def svi_features_path(currency: str = "BTC", horizon_minutes: int = 15) -> Path:
    """SVI surface calibration output, bucketed features (e.g. svi_atm_1d).

    h=15 returns the legacy unsuffixed path (byte-identical to pre-Phase 2h5
    call sites). h!=15 lands at ``<currency>_svi_features_h{N}m.parquet``
    so the h=5m regen does not overwrite the production h=15 file.
    """
    if horizon_minutes == 15:
        if currency == "BTC":
            return Path("vol_arb/output/svi_features_15m.parquet")  # legacy
        return Path(f"vol_arb/output/{currency}_svi_features_15m.parquet")
    # h!=15: always per-currency-suffixed (no legacy path exists for h!=15)
    if currency == "BTC":
        return Path(f"vol_arb/output/svi_features_h{horizon_minutes}m.parquet")
    return Path(
        f"vol_arb/output/{currency}_svi_features_h{horizon_minutes}m.parquet"
    )


def svi_surface_params_path(currency: str = "BTC", freq: str = "1h") -> Path:
    """Per-bucket SVI (a, b, rho, m, sigma) parameters for reconstruction.

    The ``freq`` argument is the bucket cadence the SVI calibration ran at
    (e.g. ``"1h"`` for production, ``"5min"`` for the h=5m regen). Distinct
    from the downstream bar-grid horizon (``horizon_minutes`` elsewhere),
    because SVI params can be calibrated at one cadence and interpolated
    onto a finer bar grid.
    """
    if currency == "BTC":
        return Path(f"vol_arb/output/svi_surface_params_{freq}.parquet")
    return Path(f"vol_arb/output/{currency}_svi_surface_params_{freq}.parquet")


def bar_exec_quality_path(
    currency: str = "BTC", horizon_minutes: int = 15,
) -> Path:
    """Per-bar realistic-taker slippage surface (from trade tape).

    h=15 returns legacy unsuffixed path; h!=15 suffixed with ``_h{N}m`` so
    per-horizon slippage surfaces do not collide.
    """
    if horizon_minutes == 15:
        if currency == "BTC":
            return Path("vol_arb/output/bar_execution_quality.parquet")
        return Path(f"vol_arb/output/{currency}_bar_execution_quality.parquet")
    # h!=15: always per-currency-suffixed
    if currency == "BTC":
        return Path(
            f"vol_arb/output/bar_execution_quality_h{horizon_minutes}m.parquet"
        )
    return Path(
        f"vol_arb/output/{currency}_bar_execution_quality_h{horizon_minutes}m.parquet"
    )


def minute_atm_iv_path(currency: str = "BTC") -> Path:
    """Minute-resolution ATM IV for vega-stop monitoring (optional)."""
    if currency == "BTC":
        return Path("vol_arb/output/minute_atm_iv.parquet")
    return Path(f"vol_arb/output/{currency}_minute_atm_iv.parquet")


def dvol_path(currency: str = "BTC", freq: str = "60m") -> Path:
    """Deribit DVOL time series (input data, already asset-prefixed)."""
    return Path(f"data/deribit/dvol/{currency}_dvol_{freq}.parquet")


def option_trade_dir(currency: str = "BTC") -> Path:
    """Raw Deribit option trade JSON directory."""
    return Path(f"data/deribit/trades/{currency}_option")


def vpin_path(currency: str = "BTC", bar_minutes: int = 15) -> Path:
    """VPIN time series (input to DRL hedging env).

    Matches the on-disk convention used by ``scripts/recompute_vpin.py``:
    ``vol_arb/output/vpin_{currency_lower}_{bar_minutes}m.parquet``.
    """
    return Path(f"vol_arb/output/vpin_{currency.lower()}_{bar_minutes}m.parquet")


# ── Vol-model per-asset artifacts ──────────────────────────────────────

def tsrv_path(asset: str = "BTCUSDT", tf_min: int = 15) -> Path:
    """Two-scale realized variance per-asset bars (already asset-prefixed)."""
    return Path(f"TSRV/output/{asset}_tsrv_{tf_min}m.parquet")


def features_path(asset: str = "BTCUSDT", tf: str = "15m") -> Path:
    """Engineered ML features (already asset-prefixed)."""
    return Path(f"vol_model/output/{asset}_features_{tf}.parquet")


def pruned_features_path(asset: str = "BTCUSDT") -> Path:
    """SHAP-derived pruned feature list."""
    if asset == "BTCUSDT":
        return Path("vol_model/output/pruned_features.csv")  # legacy
    return Path(f"vol_model/output/{asset}_pruned_features.csv")


def shap_features_path(asset: str = "BTCUSDT") -> Path:
    """Raw SHAP feature-importance output (pre-prune)."""
    if asset == "BTCUSDT":
        return Path("vol_model/output/shap_126_features.csv")  # legacy
    return Path(f"vol_model/output/{asset}_shap_126_features.csv")


def drop_features_path(asset: str = "BTCUSDT") -> Path:
    """SHAP-tail DROP list (bottom-N inside top-N pruned). Stage 15.9 equivalent."""
    return Path(f"vol_model/output/{asset}_drop_features.csv")


def transvol_dir(asset: str = "BTCUSDT", horizon_minutes: int = 15) -> Path:
    """Walk-forward TransVol encoder checkpoints (50 months).

    h=15 keeps the legacy unsuffixed path for both BTC and non-BTC
    assets (byte-identical to the pre-horizon-SSOT layout). h!=15
    appends ``_h{N}m`` before any per-asset suffix so a single asset
    can have multiple horizon-specific TransVol checkpoint trees
    without collision.
    """
    if asset == "BTCUSDT":
        stem = "expanding_hybrid"  # legacy BTC
    else:
        stem = f"expanding_hybrid_{asset}"
    stem = _suffix_horizon(stem, horizon_minutes)
    return Path(f"vol_model/output/models/{stem}")


def pruned_svi_model_dir(asset: str = "BTCUSDT") -> Path:
    """LightGBM pruned+SVI per-month models."""
    if asset == "BTCUSDT":
        return Path("vol_model/output/models/expanding_pruned_svi_v1")  # legacy
    return Path(f"vol_model/output/models/expanding_pruned_svi_v1_{asset}")


def pruned_svi_predictions_path(asset: str = "BTCUSDT") -> Path:
    """Walk-forward OOS predictions from pruned+SVI model (backtester input)."""
    return Path(f"vol_model/output/{asset}_pruned_svi_predictions.parquet")


def oof_predictions_path(asset: str = "BTCUSDT") -> Path:
    """Walk-forward OOF predictions (simpler LGBM model, no TransVol)."""
    return Path(f"vol_model/output/{asset}_oof_predictions.parquet")


# ── Horizon-aware artifact paths ────────────────────────────────────────
#
# All horizon-suffixed helpers follow the same convention:
#   - h=15 (legacy production) returns the existing UNSUFFIXED path so
#     refactor-time byte-identity is automatic.
#   - h≠15 returns the same path with a `_h{N}m` suffix appended before
#     the .parquet extension.
#
# The convention lives ONLY here so any place that needs a
# horizon-tagged file path goes through these helpers. The horizon
# comes from `vol_model.config.VolModelConfig.horizon_minutes`
# (the SSOT). See plan: starry-rolling-rabin.md Phase 1c.

def _suffix_horizon(stem: str, horizon_minutes: int) -> str:
    """Return ``stem`` with ``_h{N}m`` appended when N != 15.

    At h=15 the stem is returned unchanged so production paths stay byte-
    identical to the pre-horizon-SSOT layout.
    """
    if horizon_minutes == 15:
        return stem
    return f"{stem}_h{horizon_minutes}m"


def har_pd_features_path(
    asset: str = "BTCUSDT", horizon_minutes: int = 15,
) -> Path:
    """HAR-PD raw feature sidecar (per-horizon).

    h=15: ``vol_model/output/{asset}_features_har_pd_15m.parquet``
          (legacy name, kept for byte-identity).
    h=N : ``vol_model/output/{asset}_features_har_pd_{N}m.parquet``.

    Note: the legacy 15m suffix was always literal "_15m"; the helper
    keeps that exact stem at h=15 and uses "_{N}m" at other horizons.
    The "_h{N}m" convention used by other helpers does NOT apply here
    because the on-disk legacy name embeds the bar size directly.
    """
    return Path(
        f"vol_model/output/{asset}_features_har_pd_{horizon_minutes}m.parquet"
    )


def har_pd_predictions_path(
    asset: str = "BTCUSDT",
    horizon_minutes: int = 15,
    *,
    aligned_bias: bool = False,
) -> Path:
    """HAR-PD walk-forward OOS predictions (per-horizon).

    h=15 (aligned_bias=False): ``vol_model/output/{asset}_pruned_svi_har_pd_predictions.parquet``
    h=15 (aligned_bias=True):  ``vol_model/output/{asset}_pruned_svi_har_pd_predictions_aligned_bias.parquet``
    h=N : same with ``_h{N}m`` appended before .parquet.
    """
    base = f"{asset}_pruned_svi_har_pd_predictions"
    if aligned_bias:
        base = f"{base}_aligned_bias"
    stem = _suffix_horizon(base, horizon_minutes)
    return Path(f"vol_model/output/{stem}.parquet")


def har_pd_model_dir(
    asset: str = "BTCUSDT", horizon_minutes: int = 15,
) -> Path:
    """LightGBM HAR-PD per-month walk-forward checkpoints (per-horizon).

    h=15 (BTC, legacy): ``vol_model/output/models/expanding_pruned_svi_har_pd_v1``
    h=15 (other asset): ``vol_model/output/models/expanding_pruned_svi_har_pd_v1_{asset}``
    h=N : same with ``_h{N}m`` appended.
    """
    if asset == "BTCUSDT":
        stem = "expanding_pruned_svi_har_pd_v1"
    else:
        stem = f"expanding_pruned_svi_har_pd_v1_{asset}"
    stem = _suffix_horizon(stem, horizon_minutes)
    return Path(f"vol_model/output/models/{stem}")


def trade_log_path(
    currency: str = "BTC",
    horizon_minutes: int = 15,
    *,
    variant: str = "v2",
    suffix: str = "",
) -> Path:
    """Backtester trade log (per-horizon, per-variant).

    Legacy h=15 paths preserved verbatim:
      - variant="v2", suffix="":         BTC_v2_trade_log.parquet
      - variant="v2", suffix="baseline": BTC_v2_trade_log_baseline.parquet
      - variant="v2", suffix="aligned_bias": BTC_v2_trade_log_aligned_bias.parquet
      - variant="v2", suffix="dfl_A1_filter25": BTC_v2_trade_log_dfl_A1_filter25.parquet
      - variant="v2", suffix="throttled_static": BTC_v2_trade_log_throttled_static.parquet

    h=N appends ``_h{N}m`` before .parquet.
    """
    stem = f"{currency}_{variant}_trade_log"
    if suffix:
        stem = f"{stem}_{suffix}"
    stem = _suffix_horizon(stem, horizon_minutes)
    return Path(f"vol_arb/output/{stem}.parquet")


def kelly_sidecar_path(
    currency: str = "BTC",
    horizon_minutes: int = 15,
    *,
    variant: str = "K1",
) -> Path:
    """Kelly multiplier sidecar (per-horizon).

    h=15: ``vol_arb/output/{currency}_v2_kelly_mult_{variant}.parquet`` (legacy)
    h=N : same with ``_h{N}m`` appended before .parquet.
    """
    stem = f"{currency}_v2_kelly_mult_{variant}"
    stem = _suffix_horizon(stem, horizon_minutes)
    return Path(f"vol_arb/output/{stem}.parquet")


# ── DRL hedging per-asset artifacts ────────────────────────────────────

def sac_model_dir(asset: str = "BTCUSDT", tag: str = "v4") -> Path:
    """SAC DRL hedging agent checkpoint dir."""
    if asset == "BTCUSDT":
        return Path(f"vol_arb/drl_hedging/models/sac_risk_{tag}")  # legacy
    return Path(f"vol_arb/drl_hedging/models/sac_risk_{tag}_{asset}")


def sac_tb_dir(asset: str = "BTCUSDT", tag: str = "v4") -> Path:
    """SAC TensorBoard log dir."""
    if asset == "BTCUSDT":
        return Path(f"vol_arb/drl_hedging/tb_logs/sac_risk_{tag}")  # legacy
    return Path(f"vol_arb/drl_hedging/tb_logs/sac_risk_{tag}_{asset}")
