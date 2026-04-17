"""PatchTST encoder with cross-feature attention (drop-in for TransVol).

Interface contract:
    forward(x, time_features) -> dict with keys:
        "continuous_var_raw", "jump_var_raw", "embedding", "embedding_compact"
    where x has shape (B, T, F) and time_features has shape (B, T, 2).

    Same losses as TransVol (ratio-form QLIKE + Huber jump), reused via
    the losses defined in ``vol_model.transvol.model``.

**Causal guarantee**: attention mask is strictly upper-triangular on the
patch axis. A patch at position p attends only to patches [0, p]. Test
``tests/test_patchtst.py::test_patchtst_attention_is_causal`` verifies
this numerically by zero-masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vol_model.patchtst.config import PatchTSTConfig
from vol_model.transvol.model import huber_jump_loss, qlike_ratio_loss
from vol_model.transvol.time2vec import Time2Vec


class PatchEmbedding(nn.Module):
    """Unfold (B, T, F) into (B, n_patches, patch_len * F) then project."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.stride = config.patch_stride
        self.n_patches = config.n_patches
        patch_dim = config.patch_len * config.n_features
        self.proj = nn.Linear(patch_dim, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x of shape (B, T, F)
        Returns: (B, n_patches, d_model)
        """
        B, T, F_ = x.shape
        # Unfold along time dim: (B, n_patches, patch_len, F)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold returns (B, n_patches, F, patch_len); permute to
        # (B, n_patches, patch_len, F) then flatten.
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.view(B, self.n_patches, -1)  # (B, P, patch_len*F)
        return self.proj(patches)                       # (B, P, d_model)


class PatchTSTBlock(nn.Module):
    """Pre-norm Transformer block, masked self-attention across patches."""

    def __init__(self, config: PatchTSTConfig):
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


class PatchTSTEncoder(nn.Module):
    """PatchTST-style encoder producing TransVol-compatible embeddings.

    Pipeline:
      1. Patch embedding: (B, T=32, F=101) -> (B, P=8, d_model)
      2. Time features (summed over each patch): (B, T, 2) -> (B, P, 2)
         then Time2Vec -> projected to d_model and added.
      3. N causal Transformer blocks.
      4. Last-patch pooling + persistence skip from the raw last patch.
      5. Two MTL heads + 16-dim bottleneck.

    Matches TransVol's public interface so `_extract_compact_embeddings`
    works without modification.
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        self.patch_emb = PatchEmbedding(config)
        self.input_dropout = nn.Dropout(config.dropout)

        # Time2Vec over per-patch mean of time features.
        self.time2vec = Time2Vec(config.n_time_inputs, config.time2vec_dim)
        t2v_out_dim = config.n_time_inputs * (config.time2vec_dim + 1)
        self.time_proj = nn.Linear(t2v_out_dim, config.d_model)

        # Persistence shortcut: flatten the last patch and project.
        self.skip_proj = nn.Linear(config.patch_len * config.n_features, config.d_model)

        self.layers = nn.ModuleList([
            PatchTSTBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

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
        self.bottleneck = nn.Sequential(
            nn.Linear(config.d_model, 16),
            nn.Tanh(),
        )
        self._init_weights()
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
    def create_causal_mask(n_patches: int, device: torch.device) -> torch.Tensor:
        """Strict upper-triangular mask on the patch axis.

        A patch at position p attends to patches [0, p] (inclusive).
        No attention to positions > p -- guarantees no future-patch leakage.
        """
        mask = torch.triu(torch.ones(n_patches, n_patches, device=device), diagonal=1)
        return mask.float().masked_fill(mask.bool(), float("-inf"))

    def _patch_time_features(self, time_features: torch.Tensor) -> torch.Tensor:
        """Collapse (B, T, 2) time features into (B, P, 2) via per-patch mean."""
        B, T, C = time_features.shape
        # Same unfold as the patch embedding.
        patches = time_features.unfold(
            dimension=1,
            size=self.config.patch_len,
            step=self.config.patch_stride,
        )  # (B, P, 2, patch_len)
        return patches.mean(dim=-1)  # (B, P, 2)

    def forward(
        self, x: torch.Tensor, time_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T=seq_len, F=n_features)
            time_features: (B, T, 2)

        Returns:
            dict matching TransVol's keys:
                continuous_var_raw, jump_var_raw, embedding, embedding_compact
        """
        # Patch + project
        h = self.patch_emb(x)                           # (B, P, d_model)
        t_patches = self._patch_time_features(time_features)
        t2v = self.time2vec(t_patches)                  # (B, P, t2v_out)
        h = h + self.time_proj(t2v)
        h = self.input_dropout(h)

        # Masked patch-axis attention
        P = h.size(1)
        mask = self.create_causal_mask(P, x.device)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.final_norm(h)

        # Last-patch pooling + persistence skip from the raw last patch
        last_patch_raw = x[:, -self.config.patch_len:, :].reshape(x.size(0), -1)
        emb = h[:, -1, :] + self.skip_proj(last_patch_raw)

        cont_raw = self.head_continuous(emb).squeeze(-1)
        jump_raw = self.head_jump(emb).squeeze(-1)

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
        loss_qlike = qlike_ratio_loss(
            targets["continuous_var"], outputs["continuous_var_raw"],
        )
        loss_jump = huber_jump_loss(
            targets["jump_var"], outputs["jump_var_raw"],
            delta=self.config.huber_delta,
        )
        total = (
            self.config.loss_weight_qlike * loss_qlike
            + self.config.loss_weight_jump * loss_jump
        )
        return {"total": total, "qlike": loss_qlike, "huber_jump": loss_jump}
