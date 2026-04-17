"""Tests for the W4.1 directional classifier head gate in the backtester.

The gate applies an optional confidence filter after the primary threshold
signal is generated: long-vol trades survive only when the classifier's
``direction_prob`` (= P[y_true > y_pred_base]) is at least the threshold,
and short-vol trades survive only when ``direction_prob`` is at most
``1 - threshold``.  Missing (NaN) classifier outputs always mute.

These tests build a synthetic classifier predictions parquet on a module-
scoped temp path, aligned to the full base predictions index, so they
don't depend on the real W4.1 trained artifacts existing on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_arb.backtester import BacktestConfig, VolArbBacktester


# ── Fixture helpers ───────────────────────────────────────────────────

def _write_constant_prob_parquet(
    path: Path, index: pd.DatetimeIndex, prob: float,
) -> None:
    """Write a synthetic classifier parquet with a constant direction_prob."""
    df = pd.DataFrame(
        {
            "direction_prob": np.full(len(index), float(prob)),
            "direction_true": np.zeros(len(index), dtype=np.int8),
        },
        index=index,
    )
    df.index.name = "timestamp"
    df.to_parquet(path)


def _write_prob_series_parquet(
    path: Path, probs: pd.Series,
) -> None:
    """Write a synthetic classifier parquet from an arbitrary prob series."""
    probs = probs.rename("direction_prob")
    df = pd.DataFrame(
        {
            "direction_prob": probs.values,
            "direction_true": np.zeros(len(probs), dtype=np.int8),
        },
        index=probs.index,
    )
    df.index.name = "timestamp"
    df.to_parquet(path)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline_run():
    """Run the backtester with no classifier gate — reference trade log."""
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=20.0,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


@pytest.fixture(scope="module")
def base_preds_index(baseline_run):
    """Index over which the synthetic classifier parquets are built.

    Uses the default base predictions parquet path (the same one the
    backtester consumes) so the synthetic classifier predictions align
    with exactly the rows the backtester joins against.
    """
    cfg, _, _ = baseline_run
    preds = pd.read_parquet(cfg.predictions_path)
    return preds.index


@pytest.fixture(scope="module")
def tmp_module_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("classifier_gate")


@pytest.fixture(scope="module")
def all_long_path(tmp_module_dir, base_preds_index):
    """Classifier that strongly favours long-vol: prob=0.9 everywhere.

    At threshold 0.55:
      - long-vol signals (need ≥ 0.55) → survive (0.9 ≥ 0.55)
      - short-vol signals (need ≤ 0.45) → muted (0.9 > 0.45)
    """
    path = tmp_module_dir / "all_long_prob.parquet"
    _write_constant_prob_parquet(path, base_preds_index, prob=0.9)
    return str(path)


@pytest.fixture(scope="module")
def all_short_path(tmp_module_dir, base_preds_index):
    """Classifier that strongly favours short-vol: prob=0.1 everywhere."""
    path = tmp_module_dir / "all_short_prob.parquet"
    _write_constant_prob_parquet(path, base_preds_index, prob=0.1)
    return str(path)


@pytest.fixture(scope="module")
def half_nan_path(tmp_module_dir, base_preds_index):
    """Classifier with half NaN rows — muted bars should match NaN mask."""
    path = tmp_module_dir / "half_nan_prob.parquet"
    probs = pd.Series(0.9, index=base_preds_index)
    probs.iloc[::2] = np.nan  # Every other row NaN
    _write_prob_series_parquet(path, probs)
    return str(path)


@pytest.fixture(scope="module")
def gated_long_run(all_long_path):
    """Baseline config + classifier gate loaded from the long-favour parquet."""
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=20.0,
        classifier_path=all_long_path,
        classifier_threshold=0.55,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


@pytest.fixture(scope="module")
def gated_short_run(all_short_path):
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=20.0,
        classifier_path=all_short_path,
        classifier_threshold=0.55,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


@pytest.fixture(scope="module")
def gated_half_nan_mute_run(half_nan_path):
    """Gate with classifier_nan_is_pass=False (validation A/B mode)."""
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=20.0,
        classifier_path=half_nan_path,
        classifier_threshold=0.55,
        classifier_nan_is_pass=False,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


@pytest.fixture(scope="module")
def gated_half_nan_pass_run(half_nan_path):
    """Gate with classifier_nan_is_pass=True (production default)."""
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=20.0,
        classifier_path=half_nan_path,
        classifier_threshold=0.55,
        classifier_nan_is_pass=True,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


# ── Tests ─────────────────────────────────────────────────────────────

def test_classifier_path_none_is_baseline(baseline_run):
    """With classifier_path=None the gate is a no-op: there should be
    trades and the standard columns should be present (sanity check that
    the baseline run itself is valid)."""
    _, metrics, tl = baseline_run
    assert metrics["n_trades"] > 0
    assert (tl["signal"] != 0).sum() > 0


def test_classifier_gate_strictly_selective(baseline_run, gated_long_run):
    """Long-favouring classifier mutes every short-vol trade but preserves
    every long-vol trade exactly. Result: gated trade count ≤ baseline."""
    _, base_metrics, base_tl = baseline_run
    _, gated_metrics, gated_tl = gated_long_run

    base_trades = int((base_tl["signal"] != 0).sum())
    gated_trades = int((gated_tl["signal"] != 0).sum())
    assert gated_trades <= base_trades, (
        "classifier gate must be strictly more selective than baseline"
    )
    # Long-vol (+1) trades must be exactly preserved (prob=0.9 ≥ 0.55).
    base_long = int((base_tl["signal"] == 1).sum())
    gated_long = int((gated_tl["signal"] == 1).sum())
    assert gated_long == base_long, (
        f"long-vol trades must be preserved: base={base_long} gated={gated_long}"
    )
    # Short-vol (-1) trades must be fully muted (prob=0.9 > 0.45 → disagree).
    gated_short = int((gated_tl["signal"] == -1).sum())
    assert gated_short == 0, (
        f"short-vol trades must be fully muted at prob=0.9, got {gated_short}"
    )


def test_classifier_gate_long_vs_short_asymmetry(
    baseline_run, gated_short_run,
):
    """Short-favouring classifier (prob=0.1) mutes long and preserves short."""
    _, _, base_tl = baseline_run
    _, _, gated_tl = gated_short_run

    base_short = int((base_tl["signal"] == -1).sum())
    gated_short = int((gated_tl["signal"] == -1).sum())
    gated_long = int((gated_tl["signal"] == 1).sum())

    assert gated_short == base_short, (
        "short-vol trades must be preserved at prob=0.1"
    )
    assert gated_long == 0, (
        "long-vol trades must be fully muted at prob=0.1"
    )


def test_classifier_gate_mutes_nan_bars_when_configured(
    baseline_run, gated_half_nan_mute_run,
):
    """With classifier_nan_is_pass=False, NaN bars are muted (validation
    A/B mode). The fixture's prob series has ``probs.iloc[::2] = NaN``
    and non-NaN=0.9. So: NaN rows mute all signals, non-NaN rows preserve
    long-vol and mute short-vol. Gated long-vol count must be strictly
    less than baseline long-vol count.
    """
    _, _, base_tl = baseline_run
    _, _, gated_tl = gated_half_nan_mute_run

    gated_active_mask = gated_tl["signal"] != 0
    assert gated_active_mask.sum() > 0, "expected some active gated trades"
    # Short-vol mutes regardless of NaN (prob 0.9 > 0.45).
    assert int((gated_tl["signal"] == -1).sum()) == 0
    base_long_count = int((base_tl["signal"] == 1).sum())
    gated_long_count = int((gated_tl["signal"] == 1).sum())
    assert gated_long_count < base_long_count, (
        f"NaN mute should kill some long-vol trades: "
        f"base_long={base_long_count}, gated_long={gated_long_count}"
    )


def test_classifier_gate_nan_pass_preserves_nan_bars(
    baseline_run, gated_half_nan_pass_run,
):
    """With classifier_nan_is_pass=True (default), NaN bars bypass the gate
    and trade as usual. Short-vol is still muted by the non-NaN 0.9 probs;
    long-vol count must match baseline exactly (non-NaN rows survive by
    prob, NaN rows survive by pass-through).
    """
    _, _, base_tl = baseline_run
    _, _, gated_tl = gated_half_nan_pass_run

    base_long_count = int((base_tl["signal"] == 1).sum())
    gated_long_count = int((gated_tl["signal"] == 1).sum())
    assert gated_long_count == base_long_count, (
        f"pass-through must preserve all long-vol trades: "
        f"base_long={base_long_count}, gated_long={gated_long_count}"
    )
    # Short-vol still muted on non-NaN rows; NaN short-vol passes through.
    assert int((gated_tl["signal"] == -1).sum()) > 0, (
        "NaN short-vol bars must pass through with classifier_nan_is_pass=True"
    )
    assert int((gated_tl["signal"] == -1).sum()) < int(
        (base_tl["signal"] == -1).sum(),
    ), "some non-NaN short-vol should still be muted"


def test_classifier_gate_muted_bars_have_zero_pnl(gated_long_run):
    """All muted bars (signal==0 after gate) must have zero PnL and
    zero contract counts, just like any other inactive bar."""
    _, _, tl = gated_long_run
    inactive = tl[tl["signal"] == 0]
    assert (inactive["N_contracts"] == 0).all()
    assert (inactive["net_pnl"] == 0).all()
    # NC columns too
    assert (inactive["N_contracts_nc"] == 0).all()
    assert (inactive["net_pnl_nc"] == 0).all()


@pytest.mark.parametrize("bad_threshold", [0.3, 0.5, 1.0, 1.2, -0.1])
def test_classifier_threshold_out_of_range_raises(bad_threshold):
    """Classifier threshold must lie in the open interval (0.5, 1.0)."""
    with pytest.raises(Exception):  # pydantic ValidationError or ValueError
        BacktestConfig(classifier_threshold=bad_threshold)


def test_classifier_threshold_valid_values_accepted():
    """Sanity: valid thresholds construct without raising."""
    for good in [0.51, 0.55, 0.75, 0.99]:
        cfg = BacktestConfig(classifier_threshold=good)
        assert cfg.classifier_threshold == good
