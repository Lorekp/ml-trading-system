"""Tests for vol_model.spot_check_regimes.pick_regime_months."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vol_model.spot_check_regimes import MACRO_MONTHS, pick_regime_months


def _fixture(tmp_path: Path, per_month: list[dict]) -> Path:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"per_month": per_month}))
    return p


def test_picks_hardest_by_absolute_qlike(tmp_path: Path) -> None:
    """Pick_regime_months should pick the months with the highest QLIKE."""
    per_month = [
        {"month": "2024-01", "pruned_svi_qlike": 0.30},
        {"month": "2024-02", "pruned_svi_qlike": 0.80},  # hardest
        {"month": "2024-03", "pruned_svi_qlike": 0.60},
        {"month": "2024-04", "pruned_svi_qlike": 0.75},  # 2nd hardest
        {"month": "2024-05", "pruned_svi_qlike": 0.40},
    ]
    months = pick_regime_months(
        _fixture(tmp_path, per_month),
        k_hardest=2,
        include_easiest=0,
        macro_months=(),
    )
    assert set(months) >= {"2024-02", "2024-04"}
    assert sorted(months) == months  # sorted output


def test_unions_macro_months(tmp_path: Path) -> None:
    """Macro hand-picks should always appear in the output."""
    per_month = [
        {"month": "2025-01", "pruned_svi_qlike": 0.30},
        {"month": "2025-02", "pruned_svi_qlike": 0.40},
    ]
    months = pick_regime_months(
        _fixture(tmp_path, per_month),
        k_hardest=1,
        include_easiest=1,
        macro_months=("2022-11", "2024-11"),
    )
    assert "2022-11" in months
    assert "2024-11" in months


def test_deduplicates_overlap(tmp_path: Path) -> None:
    """Month appearing in both hardest and macro should appear once."""
    per_month = [
        {"month": "2022-11", "pruned_svi_qlike": 0.99},  # hardest AND macro
        {"month": "2023-01", "pruned_svi_qlike": 0.30},
    ]
    months = pick_regime_months(
        _fixture(tmp_path, per_month),
        k_hardest=1,
        include_easiest=0,
        macro_months=("2022-11",),
    )
    assert months.count("2022-11") == 1


def test_deterministic(tmp_path: Path) -> None:
    """Same summary.json + same params = byte-identical output."""
    per_month = [
        {"month": f"2024-{m:02d}", "pruned_svi_qlike": 0.5 + m * 0.01}
        for m in range(1, 13)
    ]
    fixture = _fixture(tmp_path, per_month)
    a = pick_regime_months(fixture, k_hardest=3, include_easiest=1)
    b = pick_regime_months(fixture, k_hardest=3, include_easiest=1)
    assert a == b


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pick_regime_months(tmp_path / "does_not_exist.json")


def test_malformed_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"wrong_key": []}))
    with pytest.raises(ValueError, match="per_month"):
        pick_regime_months(p)


def test_macro_months_defaults_present() -> None:
    """Sanity: the default MACRO_MONTHS constant is non-empty and well-formed."""
    assert len(MACRO_MONTHS) >= 3
    for m in MACRO_MONTHS:
        # YYYY-MM format
        assert len(m) == 7
        assert m[4] == "-"
        assert int(m[:4]) >= 2022
        assert 1 <= int(m[5:]) <= 12
