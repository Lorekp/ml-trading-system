"""TransVol V2 encoder — ratio-form QLIKE, softplus links, persistence shortcut.

Architecture:
    Input projection (F -> d_model) + Time2Vec + persistence skip connection
    -> N pre-norm Transformer blocks with causal self-attention
    -> Last-token pooling + skip residual
    -> Head 1: continuous_var via softplus link, ratio-form QLIKE loss
    -> Head 2: jump_var via softplus link, scaled MSE loss
    -> Loss-balanced dynamic weighting
    -> Compact bottleneck embedding (16-dim) for LightGBM
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.time2vec import Time2Vec


# ────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ────────────────────────────────────────────────────────────

def qlike_ratio_loss(c_true: torch.Tensor, c_pred_raw: torch.Tensor) -> torch.Tensor:
    """QLIKE in ratio form (Patton 2011). Operates in variance space.

    L = c / ĉ - log(c / ĉ) - 1

    c_pred_raw is unbounded head output. Softplus maps to (0, inf).
    Gradient w.r.t. head output h: dL/dh = 1 - c/ĉ (no exp, inherently stable).
    """
    c_hat = F.softplus(c_pred_raw) + 1e-8
    ratio = (c_true / c_hat).clamp(max=100.0)  # cap extreme ratios to prevent gradient spikes
    return torch.mean(ratio - torch.log(ratio) - 1.0)


def huber_jump_loss(
    j_true: torch.Tensor, j_pred_raw: torch.Tensor, delta: float = 0.01,
) -> torch.Tensor:
    """Huber loss on jump variance with softplus link.

    Softplus ensures non-negative predictions. Huber clips gradient for
    outlier jumps (prevents 15,000x MSE spikes from extreme events).
    """
    j_hat = F.softplus(j_pred_raw) + 1e-10
    return F.huber_loss(j_hat, j_true, delta=delta)


# ────────────────────────────────────────────────────────────
# TRANSFORMER BLOCKS
# ────────────────────────────────────────────────────────────

class TransVolBlock(nn.Module):
    """Pre-norm Transformer encoder block with causal attention."""

    def __init__(self, config: TransVolConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


# ────────────────────────────────────────────────────────────
# MAIN MODEL
# ────────────────────────────────────────────────────────────

class TransVolEncoder(nn.Module):
    """TransVol V2: causal Transformer with ratio-form QLIKE + persistence skip.

    Changes from V1:
        - Ratio-form QLIKE (variance space, no exp in gradient)
        - Softplus output links (both heads, no dead ReLU)
        - Persistence shortcut (HAR-style skip from last input window)
        - Loss-balanced dynamic weighting (not Kendall)
        - Compact 16-dim bottleneck for LightGBM embedding
    """

    def __init__(self, config: TransVolConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        # Time2Vec positional encoding
        self.time2vec = Time2Vec(config.n_time_inputs, config.time2vec_dim)
        t2v_out_dim = config.n_time_inputs * (config.time2vec_dim + 1)
        self.time_proj = nn.Linear(t2v_out_dim, config.d_model)

        # Persistence shortcut (HAR-style: raw features of last window -> d_model)
        self.skip_proj = nn.Linear(config.n_features, config.d_model)

        # Transformer stack
        self.layers = nn.ModuleList([
            TransVolBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        # MTL heads (output raw values, softplus applied in loss functions)
        self.head_continuous = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        self.head_jump = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

        # Compact bottleneck for LightGBM embedding extraction
        self.bottleneck = nn.Sequential(
            nn.Linear(config.d_model, 16),
            nn.Tanh(),
        )

        self._init_weights()

        # Initialize output biases so softplus starts near target means
        # continuous_var ~ 1.6e-5, softplus(x) ≈ x for x >> 0, so init bias ≈ log(exp(1.6e-5) - 1) ≈ -11
        # jump_var is normalized to mean=1.0, softplus(0) = 0.69, close enough
        with torch.no_grad():
            self.head_continuous[-1].bias.fill_(-11.0)
            self.head_jump[-1].bias.fill_(0.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Strict upper-triangular causal mask for nn.MultiheadAttention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.float().masked_fill(mask.bool(), float("-inf"))

    def forward(
        self, x: torch.Tensor, time_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, F) feature sequences (z-normalized)
            time_features: (B, T, 2) [duration_s, n_bars_in_window]

        Returns dict with:
            'continuous_var_raw': (B,) — raw head output (apply softplus for prediction)
            'jump_var_raw': (B,) — raw head output (apply softplus for prediction)
            'embedding': (B, d_model) — full embedding
            'embedding_compact': (B, 16) — bottleneck for LightGBM
        """
        B, T, F = x.shape

        # Project features + Time2Vec encoding
        h = self.input_proj(x)
        t2v = self.time2vec(time_features)
        h = h + self.time_proj(t2v)
        h = self.input_dropout(h)

        # Causal Transformer stack
        mask = self.create_causal_mask(T, x.device)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.final_norm(h)

        # Last-token pooling + persistence shortcut
        attn_out = h[:, -1, :]                    # (B, d_model)
        skip = self.skip_proj(x[:, -1, :])        # (B, d_model)
        emb = attn_out + skip                     # HAR-style residual

        # Prediction heads (raw outputs, softplus applied in loss)
        cont_raw = self.head_continuous(emb).squeeze(-1)  # (B,)
        jump_raw = self.head_jump(emb).squeeze(-1)        # (B,)

        return {
            "continuous_var_raw": cont_raw,
            "jump_var_raw": jump_raw,
            "embedding": emb,
            "embedding_compact": self.bottleneck(emb),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """MTL loss with fixed weights. QLIKE dominates (0.8), jump regularizes (0.2)."""
        loss_qlike = qlike_ratio_loss(
            targets["continuous_var"], outputs["continuous_var_raw"],
        )
        loss_jump = huber_jump_loss(
            targets["jump_var"], outputs["jump_var_raw"],
            delta=self.config.huber_delta,
        )

        total = (self.config.loss_weight_qlike * loss_qlike
                 + self.config.loss_weight_jump * loss_jump)

        return {
            "total": total,
            "qlike": loss_qlike,
            "huber_jump": loss_jump,
        }
