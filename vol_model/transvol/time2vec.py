"""Time2Vec positional encoding (Kazemi et al. 2019).

Learnable periodic + linear encoding for asynchronous time features.
Input: scalar time features (e.g., bar duration, bar count).
Output: vector of linear + sin(learned_freq * t + learned_phase).
"""

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Time2Vec: learnable time representation.

    For each of `in_features` time inputs, produces:
        - 1 linear component: w * t + b
        - periodic_dim sinusoidal components: sin(w_i * t + b_i)
    Total output dim: in_features * (periodic_dim + 1)
    """

    def __init__(self, in_features: int = 2, periodic_dim: int = 16):
        super().__init__()
        self.in_features = in_features
        self.periodic_dim = periodic_dim
        self.out_dim = in_features * (periodic_dim + 1)

        # Linear component: one weight + bias per input feature
        self.w_linear = nn.Parameter(torch.randn(in_features))
        self.b_linear = nn.Parameter(torch.zeros(in_features))

        # Periodic components: (in_features, periodic_dim) weights + biases
        self.w_periodic = nn.Parameter(torch.randn(in_features, periodic_dim))
        self.b_periodic = nn.Parameter(torch.zeros(in_features, periodic_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, T, in_features) — time-related inputs

        Returns:
            (B, T, in_features * (periodic_dim + 1))
        """
        # Linear: (B, T, in_features)
        linear = t * self.w_linear + self.b_linear

        # Periodic: (B, T, in_features, 1) * (in_features, periodic_dim)
        # -> (B, T, in_features, periodic_dim)
        t_expanded = t.unsqueeze(-1)  # (B, T, in_features, 1)
        periodic = torch.sin(t_expanded * self.w_periodic + self.b_periodic)

        # Flatten: (B, T, in_features * periodic_dim)
        B, T, F, P = periodic.shape
        periodic = periodic.reshape(B, T, F * P)

        # Concat linear + periodic
        return torch.cat([linear, periodic], dim=-1)
