"""Tests for model_loader — finds the latest month + loads all 3 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deribit_paper.signal.model_loader import _find_latest_month


def test_find_latest_month_picks_lex_max(tmp_path):
    # Create synthetic walk-forward layout: txt + features.json + transvol.pt
    for month in ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03"]:
        (tmp_path / f"{month}.txt").write_text("dummy")
        (tmp_path / f"{month}_features.json").write_text("[]")
        (tmp_path / f"{month}_transvol.pt").write_text("dummy")

    result = _find_latest_month(tmp_path)
    assert result == "2026-03"


def test_find_latest_month_ignores_incomplete(tmp_path):
    # 2026-03 is incomplete (missing transvol.pt), should fall back to 2026-02
    (tmp_path / "2026-02.txt").write_text("x")
    (tmp_path / "2026-02_features.json").write_text("[]")
    (tmp_path / "2026-02_transvol.pt").write_text("x")

    (tmp_path / "2026-03.txt").write_text("x")
    (tmp_path / "2026-03_features.json").write_text("[]")
    # NO transvol.pt — incomplete

    result = _find_latest_month(tmp_path)
    assert result == "2026-02"


def test_find_latest_month_raises_if_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="No complete"):
        _find_latest_month(tmp_path)


def test_find_latest_month_split_directories(tmp_path):
    """The T2.1 production layout has LightGBM files in one directory
    (``expanding_pruned_svi_v1``) and TransVol checkpoints in another
    (``expanding_hybrid``).  ``_find_latest_month`` must check both and
    only return months that have all three artifacts across the pair.
    """
    lgbm_dir = tmp_path / "pruned_svi_v1"
    tv_dir = tmp_path / "hybrid"
    lgbm_dir.mkdir()
    tv_dir.mkdir()

    # 2026-01 has everything
    (lgbm_dir / "2026-01.txt").write_text("x")
    (lgbm_dir / "2026-01_features.json").write_text("[]")
    (tv_dir / "2026-01_transvol.pt").write_text("x")

    # 2026-02 is incomplete — LGBM present but no transvol in tv_dir
    (lgbm_dir / "2026-02.txt").write_text("x")
    (lgbm_dir / "2026-02_features.json").write_text("[]")
    # Deliberately no transvol.pt

    result = _find_latest_month(lgbm_dir, tv_dir)
    assert result == "2026-01"  # 2026-02 rejected because transvol is missing


def test_find_latest_month_same_directory_still_works(tmp_path):
    """Passing transvol_dir=None should fall back to single-dir layout."""
    for month in ["2025-11", "2025-12"]:
        (tmp_path / f"{month}.txt").write_text("x")
        (tmp_path / f"{month}_features.json").write_text("[]")
        (tmp_path / f"{month}_transvol.pt").write_text("x")
    assert _find_latest_month(tmp_path, None) == "2025-12"


def test_load_model_bundle_with_real_artifacts():
    """Smoke test: load the real production artifacts on disk.

    Cycle-9 production default is the pruned+SVI+HAR-PD LightGBM
    variant (85 features = 16 TransVol embeddings + pruned base + SVI
    + HAR-PD decomposition) with TransVol weights from the hybrid run
    (16-dim embeddings).  Skipped if the production models don't exist
    (e.g., fresh clone).
    """
    lgbm_dir = Path("vol_model/output/models/expanding_pruned_svi_har_pd_v1")
    tv_dir = Path("vol_model/output/models/expanding_hybrid")
    drl_path = Path("vol_arb/drl_hedging/models/production_sac_v4_final.zip")
    if (
        not lgbm_dir.exists()
        or not tv_dir.exists()
        or not drl_path.exists()
    ):
        pytest.skip("Production model artifacts not present on this machine")

    from deribit_paper.signal.model_loader import load_model_bundle

    bundle = load_model_bundle(
        lgbm_models_dir=lgbm_dir,
        transvol_models_dir=tv_dir,
        drl_policy_path=drl_path,
    )
    # Walk-forward lex-max month depends on what's on disk; don't hardcode.
    assert bundle.month_key
    # Pruned+SVI+HAR-PD has 70-90 features depending on the SHAP-tail
    # drop list active at training time (cycle-9 = 85, cycle-10 = 77,
    # cycle-9 dropped 8 noise_level* / ffd_rv_lag3* tail features).
    # Check structural invariants instead of an exact count.
    n_feat = len(bundle.lgbm_features)
    assert 60 <= n_feat <= 100, f"unexpected n_features={n_feat}"
    assert bundle.lgbm_features[:16] == [f"tv_{i}" for i in range(16)]
    assert "svi_atm_1d" in bundle.lgbm_features
    assert any(f.startswith("har_pd_") for f in bundle.lgbm_features)
    assert bundle.lgbm_model.num_feature() == n_feat
    # TransVol should be in eval mode
    assert not bundle.transvol_model.training
    # DRL policy should have a policy attribute (from stable_baselines3)
    assert hasattr(bundle.drl_policy, "predict")
    # VecNormalize should have loaded (production file is committed)
    assert bundle.drl_vec_normalize_mean is not None
    assert bundle.drl_vec_normalize_var is not None
    assert bundle.drl_vec_normalize_mean.shape == (8,)
