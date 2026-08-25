"""Tests for GymnasiumHedgingEnv using a synthetic in-memory MarketDataView.

We don't load the real vol_arb parquet here — these tests run in < 1s and
validate the env contract (reset determinism, step shape, reward finite,
termination, observation bounds).  A separate integration test hits the
real parquet end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.env import (
    GymnasiumHedgingEnv,
    MarketDataView,
    _SVISlice,
)


def _build_synthetic_market(n_bars: int = 200) -> MarketDataView:
    """Build a stub MarketDataView without touching the filesystem."""
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    expiry = idx[0] + pd.Timedelta(days=7)

    df = pd.DataFrame({
        "iv": rng.uniform(60.0, 100.0, n_bars),  # vol points
        "strike": rng.uniform(49_000, 51_000, n_bars),
        "expiry_dt": [expiry] * n_bars,
        "index_price": rng.uniform(49_000, 51_000, n_bars),
        "vol_ann": rng.uniform(0.4, 1.2, n_bars),
        "trailing_1h_volume": rng.uniform(50.0, 500.0, n_bars),
        "jump_fraction": rng.uniform(0.0, 0.2, n_bars),
        "T_years": np.full(n_bars, 7.0 / 365.25),
    }, index=idx)

    # Mirror MarketDataView.__init__: the environment must simulate from
    # information available before the sampled bar, never the bar's own RV.
    df["vol_ann_lagged"] = df["vol_ann"].shift(1)

    # Bypass __init__ — construct an instance manually
    data = MarketDataView.__new__(MarketDataView)
    data.period = ("2024-01-01", "2024-01-05")
    data.df = df
    data.active_idx = df.index[df["vol_ann_lagged"].notna()]
    data.sample_weights = None
    # Single synthetic SVI slice covering every bar
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
    svi_params["_exp_key"] = pd.to_datetime(svi_params["expiry_dt"], utc=True).dt.strftime("%Y-%m-%d")
    data.svi_params = svi_params
    data._svi_by_expiry = {
        svi_params["_exp_key"].iloc[0]: svi_params.reset_index(drop=True),
    }
    # Flat neutral VPIN
    data.vpin = pd.Series(0.5, index=df.index, name="vpin")
    return data


@pytest.fixture
def env():
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 5  # smaller for fast tests
    cfg.high_vol_upsample_threshold = 0.0
    data = _build_synthetic_market(n_bars=200)
    return GymnasiumHedgingEnv(cfg, data, worker_seed=123)


def test_reset_returns_correct_shape(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert isinstance(info, dict)


def test_reset_is_deterministic_with_same_seed(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2, atol=0.0)


def test_step_returns_finite_reward(env):
    env.reset(seed=42)
    obs, reward, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == (env.OBS_DIM,)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "cost" in info and "iv" in info


def test_episode_terminates_at_n_steps(env):
    env.reset(seed=42)
    done = False
    n = 0
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.5], dtype=np.float32))
        done = term or trunc
        n += 1
        assert n <= env.cfg.n_steps_per_episode + 1
    assert n == env.cfg.n_steps_per_episode


def test_action_clipped_to_unit_interval(env):
    env.reset(seed=42)
    # Out-of-bounds action must not crash or produce NaN
    obs, r, _, _, info = env.step(np.array([5.0], dtype=np.float32))
    assert np.isfinite(r)
    obs, r, _, _, info = env.step(np.array([-10.0], dtype=np.float32))
    assert np.isfinite(r)


def test_zero_action_keeps_inventory_zero(env):
    env.reset(seed=42)
    for _ in range(env.cfg.n_steps_per_episode):
        _, _, term, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert abs(info["trade_size"]) < 1e-12
        if term:
            break
    assert abs(env._H) < 1e-12


def test_cost_nonnegative(env):
    env.reset(seed=42)
    total_cost = 0.0
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        assert info["cost"] >= 0.0
        total_cost += info["cost"]
        done = term or trunc
    assert total_cost > 0.0  # we traded every tick, so cost should accumulate


def test_svi_slice_iv_formula():
    """_SVISlice.iv() should return reasonable IV for a well-behaved slice."""
    s = _SVISlice(a=0.001, b=0.1, rho=-0.2, m=0.0, sigma=0.3, T=7/365.25, F=50_000.0)
    iv = s.iv(50_000.0)  # ATM
    assert 0.0 < iv < 5.0
    iv_otm = s.iv(55_000.0)
    assert iv_otm > 0.0


def test_svi_slice_invalid_T_returns_zero():
    s = _SVISlice(a=0.1, b=0.1, rho=0.0, m=0.0, sigma=0.1, T=0.0, F=50_000.0)
    assert s.iv(50_000.0) == 0.0


def test_env_samples_signal_uniformly_when_enabled():
    """With sample_signal_at_reset=True, over many resets the signal
    distribution should be roughly 50/50 between -1 and +1."""
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 3
    cfg.sample_signal_at_reset = True
    cfg.high_vol_upsample_threshold = 0.0
    data = _build_synthetic_market(n_bars=200)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=12345)

    signs = []
    for _ in range(200):
        env.reset()
        signs.append(env._trade.signal)
    signs = np.array(signs)
    # Expect both values present and roughly balanced
    assert (signs == -1).any()
    assert (signs == +1).any()
    frac_long = (signs == +1).mean()
    assert 0.35 < frac_long < 0.65, f"signal sampling is biased: frac_long = {frac_long}"


def test_env_always_short_when_signal_sampling_disabled():
    """With sample_signal_at_reset=False (legacy), all episodes short."""
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 3
    cfg.sample_signal_at_reset = False
    cfg.high_vol_upsample_threshold = 0.0
    data = _build_synthetic_market(n_bars=100)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=99)
    for _ in range(20):
        env.reset()
        assert env._trade.signal == -1


def test_env_residual_reward_bounded_by_cost():
    """When the env hedges 'perfectly' (action always full hedge in the
    right direction), the residual drops to ~0 quickly and the reward
    stream should be dominated by cost (negative)."""
    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 5
    cfg.reward_mode = "residual"
    cfg.sample_signal_at_reset = False
    cfg.high_vol_upsample_threshold = 0.0
    data = _build_synthetic_market(n_bars=150)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=2024)
    env.reset(seed=42)

    total_r = 0.0
    total_cost = 0.0
    done = False
    while not done:
        _, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        total_r += r
        total_cost += info["cost"]
        done = term or trunc

    # With reward_scale=100 and full-hedge actions, reward should be
    # approximately -cost/scale plus small residual P&L noise.
    expected_cost_component = -total_cost / cfg.reward_scale
    # Allow residual error from tracking lag
    assert total_r < 0.1, f"full-hedge should give non-positive reward, got {total_r}"
    # And cost-dominated
    assert abs(total_r - expected_cost_component) < abs(expected_cost_component) + 0.5


# ─── DRLHedger.simulate() contract ─────────────────────────────────────

def test_drl_hedger_simulate_contract(tmp_path):
    """Contract test: DRLHedger.simulate returns correct shape/values on a
    tiny policy. Verifies the vectorised rollout integrates with SB3's
    batched predict() and respects the signal==0 active mask.
    """
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from vol_arb.drl_hedging.env import DRLHedger
    from vol_arb.drl_hedging.sac_risk import RiskAverseSAC

    cfg = DRLHedgingConfig()
    cfg.n_steps_per_episode = 5
    cfg.buffer_size = 500
    cfg.batch_size = 32

    data = _build_synthetic_market(n_bars=300)
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=3)
    vec_env = DummyVecEnv([lambda: env])

    model = RiskAverseSAC(
        "MlpPolicy", vec_env,
        lambda_risk=0.3,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        policy_kwargs=dict(net_arch=[16, 16], activation_fn=torch.nn.ReLU),
        verbose=0,
    )
    model.learn(total_timesteps=100)

    policy_path = tmp_path / "tiny.zip"
    model.save(str(policy_path))

    # Build a fake SVI params parquet so DRLHedger can load *something*
    # (simulate() doesn't actually use it yet — the stub loads and caches
    # it for forward compatibility).
    svi_path = tmp_path / "svi.parquet"
    data.svi_params.to_parquet(svi_path)

    hedger = DRLHedger(policy_path=str(policy_path), svi_params_path=str(svi_path))

    n = 20
    rng = np.random.default_rng(0)
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
    # signal==0 trades must have exactly zero P&L and cost
    inactive = signal == 0
    if inactive.any():
        assert (dpnl[inactive] == 0.0).all()
        assert (costs[inactive] == 0.0).all()
