"""Tests for the TQC alternative algorithm path (Phase D of SAC hardening).

TQC (Truncated Quantile Critics, Kuznetsov et al. 2020) is the modern
distributional-SAC alternative to our dual-critic RiskAverseSAC.  It lives
in ``sb3-contrib`` and supports the same SB3 training/eval/save/load
contracts, so it drops into the existing pipeline via the
``algorithm="tqc"`` config switch.

These tests use:

- **CPU device** throughout (the foreground 300k training may be holding
  GPU resources; TQC tests run on CPU to avoid contention)
- **Tiny networks and buffers** for speed (<30s per test)
- **Synthetic market fixture** (same as test_env.py / test_sac_risk.py)
  so no real parquet load is required
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# sb3-contrib is a soft dependency — skip these tests if it's not installed.
sb3_contrib = pytest.importorskip("sb3_contrib")
from sb3_contrib import TQC  # noqa: E402

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.env import DRLHedger, GymnasiumHedgingEnv, MarketDataView


def _build_synthetic_market(n_bars: int = 200) -> MarketDataView:
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    expiry = idx[0] + pd.Timedelta(days=7)

    df = pd.DataFrame({
        "iv": rng.uniform(60.0, 100.0, n_bars),
        "strike": rng.uniform(49_000, 51_000, n_bars),
        "expiry_dt": [expiry] * n_bars,
        "index_price": rng.uniform(49_000, 51_000, n_bars),
        "vol_ann": rng.uniform(0.4, 1.2, n_bars),
        "trailing_1h_volume": rng.uniform(50.0, 500.0, n_bars),
        "jump_fraction": rng.uniform(0.0, 0.2, n_bars),
        "T_years": np.full(n_bars, 7.0 / 365.25),
    }, index=idx)

    data = MarketDataView.__new__(MarketDataView)
    data.period = ("2024-01-01", "2024-01-05")
    data.df = df
    data.active_idx = df.index
    data.sample_weights = None
    svi_params = pd.DataFrame({
        "timestamp": [idx[0] - pd.Timedelta(hours=1)],
        "expiry_dt": [expiry],
        "T": [7.0 / 365.25],
        "F": [50_000.0],
        "a": [0.001],
        "b": [0.1],
        "rho": [-0.2],
        "m": [0.0],
        "sigma": [0.3],
    })
    svi_params["_exp_key"] = pd.to_datetime(
        svi_params["expiry_dt"], utc=True
    ).dt.strftime("%Y-%m-%d")
    data.svi_params = svi_params
    data._svi_by_expiry = {
        svi_params["_exp_key"].iloc[0]: svi_params.reset_index(drop=True),
    }
    data.vpin = pd.Series(0.5, index=df.index, name="vpin")
    return data


def _tiny_tqc_kwargs():
    """Small hyperparameters for fast tests."""
    return dict(
        learning_rate=3e-4,
        buffer_size=500,
        batch_size=32,
        learning_starts=50,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.2,              # fixed (matches our production default)
        top_quantiles_to_drop_per_net=2,
        device="cpu",              # avoid contending with any GPU training
        policy_kwargs=dict(
            net_arch=[16, 16],
            activation_fn=torch.nn.ReLU,
        ),
        verbose=0,
    )


# ─── Construction & training ─────────────────────────────────────────

def test_tqc_constructs_and_trains_without_nan():
    """TQC must build with our standard kwargs and complete a short training
    run with all parameters finite."""
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 5
    cfg.high_vol_upsample_threshold = 0.0

    data = _build_synthetic_market(n_bars=200)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=1)
    vec_env = DummyVecEnv([lambda: env])

    model = TQC("MlpPolicy", vec_env, **_tiny_tqc_kwargs())
    model.learn(total_timesteps=200, log_interval=10_000)

    for name, p in model.policy.named_parameters():
        assert torch.isfinite(p).all(), f"NaN in policy.{name}"


def test_tqc_save_and_load_round_trip(tmp_path):
    """TQC model saves and loads via SB3's standard API."""
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 5
    cfg.high_vol_upsample_threshold = 0.0

    data = _build_synthetic_market(n_bars=200)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=2)
    vec_env = DummyVecEnv([lambda: env])

    model = TQC("MlpPolicy", vec_env, **_tiny_tqc_kwargs())
    model.learn(total_timesteps=100, log_interval=10_000)

    path = tmp_path / "tqc.zip"
    model.save(str(path))
    assert path.exists()

    loaded = TQC.load(str(path), device="cpu")
    obs = env.reset()[0].reshape(1, -1)
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (1, 1)
    assert np.isfinite(action).all()
    assert -1.0 <= action.item() <= 1.0


# ─── DRLHedger integration with TQC ──────────────────────────────────

def test_drl_hedger_loads_tqc_policy(tmp_path):
    """DRLHedger should load a TQC policy when cfg.algorithm = 'tqc'."""
    cfg = DRLHedgingConfig()
    cfg.algorithm = "tqc"
    cfg.n_steps_per_episode = 5
    cfg.high_vol_upsample_threshold = 0.0

    data = _build_synthetic_market(n_bars=200)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=3)
    vec_env = DummyVecEnv([lambda: env])

    model = TQC("MlpPolicy", vec_env, **_tiny_tqc_kwargs())
    model.learn(total_timesteps=100, log_interval=10_000)

    policy_path = tmp_path / "tqc_policy.zip"
    model.save(str(policy_path))

    svi_path = tmp_path / "svi.parquet"
    data.svi_params.to_parquet(svi_path)

    hedger = DRLHedger(
        policy_path=str(policy_path),
        svi_params_path=str(svi_path),
        config=cfg,
    )
    from sb3_contrib import TQC as _TQC
    assert isinstance(hedger._model, _TQC)


def test_drl_hedger_tqc_simulate_returns_finite(tmp_path):
    """End-to-end: train tiny TQC, save, load via DRLHedger, call simulate()."""
    cfg = DRLHedgingConfig()
    cfg.algorithm = "tqc"
    cfg.n_steps_per_episode = 5
    cfg.high_vol_upsample_threshold = 0.0

    data = _build_synthetic_market(n_bars=200)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=4)
    vec_env = DummyVecEnv([lambda: env])

    model = TQC("MlpPolicy", vec_env, **_tiny_tqc_kwargs())
    model.learn(total_timesteps=100, log_interval=10_000)

    policy_path = tmp_path / "tqc.zip"
    model.save(str(policy_path))
    svi_path = tmp_path / "svi.parquet"
    data.svi_params.to_parquet(svi_path)

    hedger = DRLHedger(
        policy_path=str(policy_path),
        svi_params_path=str(svi_path),
        config=cfg,
    )

    n = 20
    rng = np.random.default_rng(7)
    S = rng.uniform(45_000, 55_000, n)
    K = S * rng.uniform(0.98, 1.02, n)
    sigma_dec = rng.uniform(0.4, 1.0, n)
    T_years = rng.uniform(1.0 / 365, 7.0 / 365, n)
    vol_ann = rng.uniform(0.4, 1.2, n)
    signal = rng.choice([1, -1, 0], n, p=[0.4, 0.4, 0.2])

    dpnl, costs = hedger.simulate(
        S_entry=S, K=K, sigma=sigma_dec, T_years=T_years,
        vol_ann=vol_ann, signal=signal, n_steps=cfg.n_steps_per_episode,
    )

    assert dpnl.shape == (n,)
    assert costs.shape == (n,)
    assert np.isfinite(dpnl).all()
    assert np.isfinite(costs).all()
    assert (costs >= 0.0).all()
    inactive = signal == 0
    if inactive.any():
        assert (dpnl[inactive] == 0.0).all()
        assert (costs[inactive] == 0.0).all()


# ─── TQC + VecNormalize interaction ──────────────────────────────────

def test_tqc_with_vec_normalize_trains_cleanly():
    """VecNormalize is algorithm-agnostic — confirm TQC interoperates."""
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 5
    cfg.high_vol_upsample_threshold = 0.0

    data = _build_synthetic_market(n_bars=200)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=5)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = TQC("MlpPolicy", vec_env, **_tiny_tqc_kwargs())
    model.learn(total_timesteps=200, log_interval=10_000)

    for name, p in model.policy.named_parameters():
        assert torch.isfinite(p).all(), f"NaN in policy.{name} after VecNorm+TQC train"

    assert vec_env.obs_rms.count > 0
    assert not np.allclose(vec_env.obs_rms.mean, 0.0)
