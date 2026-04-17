"""
Bet Sizing — AFML Chapter 10.

Maps model probability to position size using a sigmoid-like function.
The key insight: bet size should be a SMOOTH function of confidence,
not a binary all-or-nothing decision.

Reference: López de Prado (2018), Ch 10, Snippet 10.1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def get_signal(
    probability: float | np.ndarray,
    num_classes: int = 2,
) -> float | np.ndarray:
    """
    Convert predicted probability to bet signal in [-1, 1].

    For binary classification (num_classes=2):
        p > 0.5 → positive signal (long), p < 0.5 → negative signal (short)
        Signal magnitude scales smoothly with distance from 0.5.

    Formula: signal = 2 * Phi(z) - 1
    where z = (p - 1/K) / sqrt(p * (1-p))
    K = number of classes, Phi = standard normal CDF

    Args:
        probability: Predicted probability of the positive class.
        num_classes: Number of label classes.

    Returns:
        Signal in [-1, 1]. 0 = no bet, ±1 = maximum bet.
    """
    p = np.asarray(probability, dtype=float)

    # Clip to avoid division by zero at p=0 or p=1
    p = np.clip(p, 1e-6, 1 - 1e-6)

    z = (p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
    signal = 2.0 * norm.cdf(z) - 1.0
    return float(signal) if np.ndim(probability) == 0 else signal


def discretize_signal(
    signal: float | np.ndarray,
    step_size: float = 0.1,
) -> float | np.ndarray:
    """
    Discretize continuous signal to nearest step.

    Avoids excessive rebalancing from tiny signal changes.

    Args:
        signal: Continuous signal in [-1, 1].
        step_size: Discretization step. 0.1 = round to nearest 0.1.

    Returns:
        Discretized signal.
    """
    return np.round(np.asarray(signal) / step_size) * step_size


def kelly_size(
    win_prob: float,
    win_return: float,
    loss_return: float,
) -> float:
    """
    Kelly criterion for optimal position sizing.

    f* = (b*p - q) / b
    where b = win_return / |loss_return|, p = win_prob, q = 1-p.

    Half-Kelly (f*/2) is recommended for practical use.

    Args:
        win_prob: Probability of winning.
        win_return: Average winning return (positive).
        loss_return: Average losing return (negative, will be abs'd).

    Returns:
        Optimal fraction of capital to risk. Negative = don't bet.
    """
    if loss_return == 0:
        return 0.0
    b = win_return / abs(loss_return)
    q = 1.0 - win_prob
    f_star = (b * win_prob - q) / b
    return f_star
