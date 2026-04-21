"""Phase 1 regression locks for the horizon-parametrized pipeline.

See plan: ``C:/Users/laure/.claude/plans/starry-rolling-rabin.md`` Phase 1g.

These tests guarantee that the horizon SSOT introduced in Phase 1
preserves byte-identity at h=15 (current production) AND that future
edits cannot reintroduce the hardcoded-15 anti-pattern that the
refactor systematically eliminated.

What "byte-identity at h=15" means in practice:

- Plain ``BacktestConfig().with_btc_production_filters_v2()`` returns
  the same field values it did pre-refactor (holding_minutes=15,
  predictions_path = legacy aligned-bias parquet, kelly_mult_path =
  legacy K1 sidecar).
- ``VolModelConfig().freq_s == 900``; the new ``horizon_minutes``
  property returns 15 by default.
- Every horizon-aware path helper in ``vol_arb.paths`` returns the
  legacy unsuffixed path when called with ``horizon_minutes=15``.

What the new generalized invariant tests cover:

- For every preset (h=5 / h=15 / h=30 / h=60), the V2 BTC stack pairs
  ``holding_minutes`` with the matching horizon-tagged
  ``predictions_path`` and ``kelly_mult_path``. This is the LIFTED
  foundation-fix invariant ("model horizon == backtester hold")
  generalised from the original "literal 15" form.
"""
from __future__ import annotations

from pathlib import Path
import re

import pytest

from vol_arb.backtester import BacktestConfig
from vol_arb.paths import (
    bar_exec_quality_path, har_pd_features_path, har_pd_predictions_path,
    kelly_sidecar_path, svi_features_path, trade_log_path, vol_arb_path,
)
from vol_model.config import VolModelConfig


# ─────────────────────────────────────────────────────────────────────
# 1. SSOT defaults
# ─────────────────────────────────────────────────────────────────────


def test_volmodel_default_horizon_is_15m():
    """``VolModelConfig`` default freq_s == 900s == 15m. The horizon
    SSOT properties must agree.
    """
    cfg = VolModelConfig()
    assert cfg.freq_s == 900
    assert cfg.horizon_minutes == 15
    assert cfg.bars_per_day == 96
    assert cfg.bars_per_24h == 96
    assert cfg.tf_str == "15min"
    assert cfg.horizon_suffix == "h15m"


@pytest.mark.parametrize("h,expected_bpd,expected_tf,expected_suffix", [
    (5,  288, "5min",  "h5m"),
    (15, 96,  "15min", "h15m"),
    (30, 48,  "30min", "h30m"),
    (60, 24,  "60min", "h60m"),
])
def test_volmodel_horizon_properties(h, expected_bpd, expected_tf, expected_suffix):
    """Derived properties stay consistent with freq_s for every supported horizon."""
    cfg = VolModelConfig(freq_s=h * 60)
    assert cfg.horizon_minutes == h
    assert cfg.bars_per_day == expected_bpd
    assert cfg.bars_per_24h == expected_bpd
    assert cfg.tf_str == expected_tf
    assert cfg.horizon_suffix == expected_suffix


# ─────────────────────────────────────────────────────────────────────
# 2. Path-helper byte-identity at h=15 (legacy unsuffixed paths)
# ─────────────────────────────────────────────────────────────────────


def test_har_pd_features_path_h15_legacy():
    """h=15 returns the existing on-disk filename verbatim."""
    p = har_pd_features_path("BTCUSDT", 15)
    assert p == Path("vol_model/output/BTCUSDT_features_har_pd_15m.parquet")


@pytest.mark.parametrize("h,expected_stem", [
    (5,  "BTCUSDT_features_har_pd_5m"),
    (30, "BTCUSDT_features_har_pd_30m"),
    (60, "BTCUSDT_features_har_pd_60m"),
])
def test_har_pd_features_path_other_horizons(h, expected_stem):
    p = har_pd_features_path("BTCUSDT", h)
    assert p.stem == expected_stem


def test_har_pd_predictions_path_h15_aligned_bias_legacy():
    """h=15 aligned_bias predictions match the production default exactly."""
    p = har_pd_predictions_path("BTCUSDT", 15, aligned_bias=True)
    assert p == Path(
        "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
    )


def test_har_pd_predictions_path_h15_unaligned_legacy():
    p = har_pd_predictions_path("BTCUSDT", 15, aligned_bias=False)
    assert p == Path(
        "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"
    )


@pytest.mark.parametrize("h", [5, 30, 60])
def test_har_pd_predictions_path_other_horizons_have_h_suffix(h):
    """h≠15 paths get the ``_h{N}m`` suffix appended."""
    p = har_pd_predictions_path("BTCUSDT", h, aligned_bias=True)
    assert p.stem.endswith(f"_h{h}m"), p
    p2 = har_pd_predictions_path("BTCUSDT", h, aligned_bias=False)
    assert p2.stem.endswith(f"_h{h}m"), p2


def test_trade_log_path_h15_legacy_variants():
    """h=15 trade-log paths match every production / sidecar location."""
    assert trade_log_path("BTC", 15) == Path("vol_arb/output/BTC_v2_trade_log.parquet")
    assert trade_log_path("BTC", 15, suffix="baseline") == Path(
        "vol_arb/output/BTC_v2_trade_log_baseline.parquet"
    )
    assert trade_log_path("BTC", 15, suffix="aligned_bias") == Path(
        "vol_arb/output/BTC_v2_trade_log_aligned_bias.parquet"
    )
    assert trade_log_path("BTC", 15, suffix="dfl_A1_filter25") == Path(
        "vol_arb/output/BTC_v2_trade_log_dfl_A1_filter25.parquet"
    )
    assert trade_log_path("BTC", 15, suffix="throttled_static") == Path(
        "vol_arb/output/BTC_v2_trade_log_throttled_static.parquet"
    )


@pytest.mark.parametrize("h", [5, 30, 60])
def test_trade_log_path_other_horizons_have_h_suffix(h):
    p = trade_log_path("BTC", h)
    assert p.stem == f"BTC_v2_trade_log_h{h}m"


def test_kelly_sidecar_path_h15_legacy():
    p = kelly_sidecar_path("BTC", 15, variant="K1")
    assert p == Path("vol_arb/output/BTC_v2_kelly_mult_K1.parquet")


@pytest.mark.parametrize("h,variant", [(5, "K1"), (30, "K1"), (60, "K1"),
                                       (60, "K2"), (60, "K3")])
def test_kelly_sidecar_path_other_horizons_have_h_suffix(h, variant):
    p = kelly_sidecar_path("BTC", h, variant=variant)
    assert p.stem == f"BTC_v2_kelly_mult_{variant}_h{h}m"


# ── Phase 2h5.0c: SVI features + bar_exec_quality horizon-aware paths ──

def test_svi_features_path_h15_legacy():
    """h=15 preserves the literal legacy BTC SVI features path."""
    p = svi_features_path("BTC", horizon_minutes=15)
    assert p == Path("vol_arb/output/svi_features_15m.parquet")


def test_svi_features_path_h15_eth_legacy():
    """h=15 non-BTC uses the pre-horizon per-currency layout."""
    p = svi_features_path("ETH", horizon_minutes=15)
    assert p == Path("vol_arb/output/ETH_svi_features_15m.parquet")


@pytest.mark.parametrize("h", [5, 30, 60])
def test_svi_features_path_other_horizons_have_h_suffix(h):
    p = svi_features_path("BTC", horizon_minutes=h)
    assert p.stem == f"svi_features_h{h}m", p


def test_bar_exec_quality_path_h15_legacy():
    """h=15 preserves the literal legacy BTC bar-exec-quality path."""
    p = bar_exec_quality_path("BTC", horizon_minutes=15)
    assert p == Path("vol_arb/output/bar_execution_quality.parquet")


@pytest.mark.parametrize("h", [5, 30, 60])
def test_bar_exec_quality_path_other_horizons_have_h_suffix(h):
    p = bar_exec_quality_path("BTC", horizon_minutes=h)
    assert p.stem == f"bar_execution_quality_h{h}m", p


def test_svi_features_path_default_kwarg_is_h15():
    """Calling the helper WITHOUT horizon_minutes must still return the
    legacy h=15 path — byte-identity guarantee for pre-Phase-2h5 callers.
    """
    assert svi_features_path("BTC") == svi_features_path("BTC", horizon_minutes=15)
    assert bar_exec_quality_path("BTC") == bar_exec_quality_path(
        "BTC", horizon_minutes=15,
    )


def test_vol_arb_path_h15_legacy():
    """h=15 returns the literal legacy BTC merged dataset path."""
    assert vol_arb_path("BTC", svi=True, horizon_minutes=15) == Path(
        "vol_arb/output/BTC_vol_arb_svi.parquet"
    )
    assert vol_arb_path("BTC", svi=False, horizon_minutes=15) == Path(
        "vol_arb/output/BTC_vol_arb.parquet"
    )


@pytest.mark.parametrize("h", [5, 30, 60])
def test_vol_arb_path_other_horizons_have_h_suffix(h):
    p = vol_arb_path("BTC", svi=True, horizon_minutes=h)
    assert p.stem == f"BTC_vol_arb_svi_h{h}m", p


def test_vol_arb_path_default_kwarg_is_h15():
    """Bare `vol_arb_path("BTC", svi=True)` must equal the h=15 helper
    result — byte-identity guarantee for pre-Phase-2h5 callers.
    """
    assert vol_arb_path("BTC", svi=True) == vol_arb_path(
        "BTC", svi=True, horizon_minutes=15,
    )


# ─────────────────────────────────────────────────────────────────────
# 3. V2 preset / per-horizon presets — pairing invariant
# ─────────────────────────────────────────────────────────────────────


def test_v2_h15_preset_byte_identity():
    """``with_btc_production_filters_v2()`` (the legacy h=15 entry point)
    preserves every production field exactly (regression lock).

    Refactor in Phase 1 factored shared invariants into
    ``_v2_horizon_invariants()`` and added per-horizon factories. The
    h=15 named entry point MUST remain byte-identical to its
    pre-refactor state — otherwise downstream sidecar generators
    silently drift.
    """
    cfg = BacktestConfig().with_btc_production_filters_v2()
    assert cfg.holding_minutes == 15
    assert cfg.use_drl_hedging is True
    assert cfg.use_profit_scaling is True
    assert cfg.profit_scaling_floor == 0.3
    assert cfg.target_margin_utilization == 0.15
    assert cfg.entry_threshold_long == 12.0
    assert cfg.skip_long_if_term_slope_gt == 0.0
    assert cfg.kelly_mult_path == "vol_arb/output/BTC_v2_kelly_mult_K1.parquet"
    assert cfg.predictions_path == (
        "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
    )


@pytest.mark.parametrize("preset_method,expected_h", [
    ("with_btc_production_h5m",  5),
    ("with_btc_production_h30m", 30),
    ("with_btc_production_h60m", 60),
])
def test_per_horizon_preset_pairs_holding_minutes_with_paths(
    preset_method, expected_h,
):
    """Per-horizon V2 preset must pair holding_minutes with the
    horizon-tagged predictions_path AND kelly_mult_path.

    This is the LIFTED foundation-fix invariant: "model horizon ==
    backtester hold == sidecar horizon". A drift here re-introduces
    the failure mode of `project_foundation_fix_2026_04_17.md` (the
    $150K phantom-PnL bug from holding 30m on 15m predictions).
    """
    cfg = getattr(BacktestConfig, preset_method)()
    assert cfg.holding_minutes == expected_h, preset_method
    assert cfg.vrp_window == (1440 // expected_h) * 7, preset_method
    assert cfg.vrp_min_periods == (1440 // expected_h), preset_method

    expected_predictions = har_pd_predictions_path(
        "BTCUSDT", expected_h, aligned_bias=True,
    ).as_posix()
    assert cfg.predictions_path == expected_predictions, (
        f"{preset_method}: predictions_path={cfg.predictions_path!r} "
        f"does not match horizon {expected_h} expectation "
        f"{expected_predictions!r}"
    )
    expected_kelly = kelly_sidecar_path(
        "BTC", expected_h, variant="K1",
    ).as_posix()
    assert cfg.kelly_mult_path == expected_kelly, (
        f"{preset_method}: kelly_mult_path={cfg.kelly_mult_path!r} "
        f"does not match horizon {expected_h} expectation "
        f"{expected_kelly!r}"
    )

    # V2 invariants must come along regardless of horizon.
    assert cfg.use_drl_hedging is True, preset_method
    assert cfg.use_profit_scaling is True, preset_method
    assert cfg.profit_scaling_floor == 0.3, preset_method
    assert cfg.target_margin_utilization == 0.15, preset_method
    assert cfg.entry_threshold_long == 12.0, preset_method
    assert cfg.skip_long_if_term_slope_gt == 0.0, preset_method


@pytest.mark.parametrize("h", [5, 30, 60])
def test_with_btc_production_for_horizon_dispatch_matches_named(h):
    """The generic factory and the named per-horizon presets must agree."""
    named = {
        5: BacktestConfig.with_btc_production_h5m,
        30: BacktestConfig.with_btc_production_h30m,
        60: BacktestConfig.with_btc_production_h60m,
    }[h]()
    generic = BacktestConfig.with_btc_production_for_horizon(h)
    assert named.holding_minutes == generic.holding_minutes
    assert named.predictions_path == generic.predictions_path
    assert named.kelly_mult_path == generic.kelly_mult_path
    assert named.vrp_window == generic.vrp_window
    assert named.vrp_min_periods == generic.vrp_min_periods


def test_with_btc_production_for_horizon_rejects_non_divisor_of_1440():
    """Catch caller errors at build time, not at runtime."""
    for bad_h in (7, 13, 27, 100, 1000):
        with pytest.raises(ValueError, match="does not divide 1440"):
            BacktestConfig.with_btc_production_for_horizon(bad_h)


# ─────────────────────────────────────────────────────────────────────
# 4. No-hardcoded-15 lint — defense against future drift
# ─────────────────────────────────────────────────────────────────────
#
# After Phase 1, every horizon-coupled bar count or time literal in the
# touched modules should read from a SSOT helper, NOT hardcode "15" or
# "minutes=15". Whitelist:
#  - Comments / docstrings / log messages: free text is fine.
#  - `freq_s: int = 900  # 15m = 900s` field default in VolModelConfig.
#  - Default-value annotations like `horizon_minutes: int = 15`.
#  - Path components like `"_15m"` ARE allowed in path helpers because
#    the legacy disk layout is what they encode.
#  - The `with_btc_production_filters_v2()` h=15 named entry point
#    explicitly hard-codes `holding_minutes=15` to lock the invariant —
#    that is INTENDED (it's the named "h=15" preset).


_HARDCODED_TIMEDELTA_MIN_15 = re.compile(r"Timedelta\s*\(\s*minutes\s*=\s*15\b")


@pytest.mark.parametrize("path", [
    "vol_model/labels.py",
    "vol_arb/dfl_head.py",
    "vol_model/features/derivatives.py",
    "vol_model/features/har_pd.py",
])
def test_no_hardcoded_timedelta_minutes_15(path):
    """Modules that build features / labels / sidecars MUST NOT use a
    bare ``pd.Timedelta(minutes=15)`` literal — that was the exact
    failure mode Phase 1 fixed. Any reintroduction = silent
    horizon-mismatch bug.
    """
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / path).read_text(encoding="utf-8")
    matches = _HARDCODED_TIMEDELTA_MIN_15.findall(text)
    assert not matches, (
        f"{path}: found {len(matches)} hardcoded `Timedelta(minutes=15)` "
        f"literal(s); use `pd.Timedelta(seconds=config.freq_s)` or pass "
        f"`horizon_minutes` through the function signature."
    )


_HARDCODED_RESAMPLE_15MIN = re.compile(r"""\.resample\(\s*['"]15min['"]""")


@pytest.mark.parametrize("path", [
    "vol_model/features/derivatives.py",
    # Phase 2h5.0a/b lint: trailing-volume + VPIN resample must derive
    # cadence from config / kwarg, not a hardcoded "15min" literal.
    "vol_arb/pipeline.py",
    "vol_arb/drl_hedging/vpin.py",
])
def test_no_hardcoded_resample_15min(path):
    """Resampling targets must come from a cadence parameter — never a
    hardcoded ``"15min"`` literal — in modules that feed the per-horizon
    pipeline. Catches Phase 2h5.0 regressions.
    """
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / path).read_text(encoding="utf-8")
    matches = _HARDCODED_RESAMPLE_15MIN.findall(text)
    assert not matches, (
        f"{path}: found hardcoded `.resample(\"15min\")` — derive "
        f"cadence from `config.bucket_freq` / `config.tf_str` / "
        f"`tf_str` kwarg instead."
    )
