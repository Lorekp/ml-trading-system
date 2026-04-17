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


def svi_features_path(currency: str = "BTC") -> Path:
    """SVI surface calibration output, 15m bucketed features (e.g. svi_atm_1d)."""
    if currency == "BTC":
        return Path("vol_arb/output/svi_features_15m.parquet")  # legacy
    return Path(f"vol_arb/output/{currency}_svi_features_15m.parquet")


def svi_surface_params_path(currency: str = "BTC", freq: str = "1h") -> Path:
    """Per-bucket SVI (a, b, rho, m, sigma) parameters for reconstruction."""
    if currency == "BTC":
        return Path(f"vol_arb/output/svi_surface_params_{freq}.parquet")
    return Path(f"vol_arb/output/{currency}_svi_surface_params_{freq}.parquet")


def bar_exec_quality_path(currency: str = "BTC") -> Path:
    """Per-bar realistic-taker slippage surface (from trade tape)."""
    if currency == "BTC":
        return Path("vol_arb/output/bar_execution_quality.parquet")
    return Path(f"vol_arb/output/{currency}_bar_execution_quality.parquet")


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


def transvol_dir(asset: str = "BTCUSDT") -> Path:
    """Walk-forward TransVol encoder checkpoints (50 months)."""
    if asset == "BTCUSDT":
        return Path("vol_model/output/models/expanding_hybrid")  # legacy
    return Path(f"vol_model/output/models/expanding_hybrid_{asset}")


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
