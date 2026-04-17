"""
Evaluation Metrics — AFML Chapter 14 + industry standard.

RankIC: Spearman correlation between predictions and forward returns.
DSR: Deflated Sharpe Ratio (corrects for multiple testing + non-normality).
Net-of-cost Sharpe: Sharpe after realistic transaction costs.

Reference: López de Prado (2018), Ch 14. Bailey & LdP (2014) SSRN 2460551.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def rank_ic(
    predictions: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """
    Rank Information Coefficient (Spearman correlation).

    Primary metric for evaluating cross-sectional signal quality.
    Thresholds: 0.02-0.03 marginal, 0.04-0.06 good, >0.06 strong.

    Args:
        predictions: Model signal/prediction values.
        forward_returns: Realized future returns.

    Returns:
        Spearman rank correlation coefficient.
    """
    common = predictions.dropna().index.intersection(forward_returns.dropna().index)
    if len(common) < 3:
        return np.nan
    rho, _ = spearmanr(predictions.loc[common], forward_returns.loc[common])
    return rho


def rank_ic_series(
    predictions: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute RankIC per time period (for cross-sectional signals).

    Args:
        predictions: DataFrame (time × assets) of predictions.
        forward_returns: DataFrame (time × assets) of forward returns.

    Returns:
        Series of per-period RankIC values.
    """
    common_times = predictions.index.intersection(forward_returns.index)
    common_assets = predictions.columns.intersection(forward_returns.columns)

    ic_values = pd.Series(index=common_times, dtype=float)
    for t in common_times:
        pred = predictions.loc[t, common_assets].dropna()
        ret = forward_returns.loc[t, common_assets].dropna()
        common = pred.index.intersection(ret.index)
        if len(common) < 5:
            ic_values[t] = np.nan
            continue
        rho, _ = spearmanr(pred[common], ret[common])
        ic_values[t] = rho

    return ic_values


def directional_accuracy(
    predictions: pd.Series,
    actuals: pd.Series,
) -> float:
    """
    Fraction of correct directional predictions.

    Args:
        predictions: Predicted direction (sign matters, magnitude ignored).
        actuals: Actual returns (sign matters).

    Returns:
        Accuracy in [0, 1]. 0.5 = random.
    """
    common = predictions.dropna().index.intersection(actuals.dropna().index)
    if len(common) == 0:
        return np.nan
    pred_sign = np.sign(predictions.loc[common])
    actual_sign = np.sign(actuals.loc[common])
    return (pred_sign == actual_sign).mean()


def net_sharpe(
    returns: pd.Series,
    cost_per_trade: float = 0.0015,
    trades_per_bar: float = 1.0,
    annualization: float = 1.0,
) -> float:
    """
    Sharpe ratio after transaction costs.

    Args:
        returns: Strategy returns per bar (BEFORE costs).
        cost_per_trade: Round-trip cost as fraction. Default 0.15% (taker on HL).
        trades_per_bar: Average number of trades per bar. Default 1.
        annualization: Multiply Sharpe by sqrt(annualization) for annualized.
                       For 4H bars with ~6 bars/day: annualization = 6*365 ≈ 2190.
                       For 1m bars: annualization = 1440*365 ≈ 525600.

    Returns:
        Annualized Sharpe ratio after costs.
    """
    net_returns = returns - cost_per_trade * trades_per_bar
    if net_returns.std() == 0:
        return 0.0
    sharpe = net_returns.mean() / net_returns.std()
    return sharpe * np.sqrt(annualization)


def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    num_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: float | None = None,
) -> float:
    """
    Deflated Sharpe Ratio — corrects for multiple testing.

    Answers: "Given that I tested N strategies and picked the best one,
    what's the probability that the observed Sharpe is genuinely above zero?"

    DSR = Phi((SR_hat - SR_0) * sqrt(T-1) / sigma_SR)

    where SR_0 = expected maximum Sharpe from N random trials.

    Args:
        observed_sr: Observed (unannualized!) Sharpe ratio of best strategy.
        num_trials: Number of strategies tested (including failed ones).
        num_observations: Number of return observations in backtest.
        skewness: Return skewness (negative inflates Sharpe).
        kurtosis: Return kurtosis (fat tails inflate Sharpe). Normal = 3.
        sr_variance: Variance across all trial Sharpe ratios.
                     If None, estimated as 1.0 (conservative).

    Returns:
        DSR probability in [0, 1]. Reject null (no skill) if DSR > 0.95.
    """
    if sr_variance is None:
        # NOTE: Default 1.0 is almost always too high and makes DSR ~0 for
        # any realistic strategy. Callers should pass the actual variance of
        # Sharpe ratios across their K trials. If unavailable, 0.1 is a
        # more reasonable assumption (trials cluster within ~0.3 SR of each other).
        sr_variance = 1.0

    # Expected maximum Sharpe under null (Bailey & LdP formula)
    gamma = 0.5772  # Euler-Mascheroni constant
    sr_0 = np.sqrt(sr_variance) * (
        (1 - gamma) * norm.ppf(1 - 1.0 / num_trials)
        + gamma * norm.ppf(1 - 1.0 / (num_trials * np.e))
    )

    # Standard error of Sharpe estimate (accounting for non-normality)
    sigma_sr = np.sqrt(
        (1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr ** 2)
        / (num_observations - 1)
    )

    if sigma_sr == 0:
        return 0.0

    # DSR = probability that true SR > 0 given observed SR and num_trials
    dsr = norm.cdf((observed_sr - sr_0) / sigma_sr)
    return dsr


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown of an equity curve.

    Args:
        equity_curve: Cumulative returns or equity curve.

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.15 = 15% drawdown).
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())


def hhi_concentration(returns: pd.Series) -> float:
    """
    Herfindahl-Hirschman Index for return concentration.

    If profits come from a few outlier trades, HHI is high → strategy is fragile.
    If profits are evenly distributed, HHI is low → strategy is robust.

    Args:
        returns: Per-trade returns.

    Returns:
        HHI in [0, 1]. 0 = perfectly distributed, 1 = single trade.
    """
    positive = returns[returns > 0]
    if len(positive) == 0:
        return 1.0
    weights = positive / positive.sum()
    return (weights ** 2).sum()
