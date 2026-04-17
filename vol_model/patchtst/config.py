"""PatchTST encoder configuration."""

from dataclasses import dataclass


@dataclass
class PatchTSTConfig:
    # Architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3         # one fewer than TransVol; patching concentrates signal
    d_ff: int = 512
    dropout: float = 0.15

    # Input -- matches TransVol contract for drop-in swap
    seq_len: int = 32         # T: consecutive 15m windows
    n_features: int = 101     # F: features per window
    n_time_inputs: int = 2

    # Patching
    patch_len: int = 4        # 4 bars = 1 hour, natural for 15m grid
    patch_stride: int = 4     # non-overlapping patches (n_patches = 32/4 = 8)
    # With T=32, patch_len=4, stride=4 -> 8 patches per sequence, each a
    # (patch_len * n_features) = 404-dim raw vector that's projected to d_model.

    # Time2Vec (reuse TransVol's module)
    time2vec_dim: int = 16

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 100     # usually converges faster than TransVol
    batch_size: int = 512
    grad_accum_steps: int = 2
    patience: int = 20
    grad_clip: float = 1.0

    # MTL (same as TransVol -- lets us re-use losses)
    loss_weight_qlike: float = 0.8
    loss_weight_jump: float = 0.2
    huber_delta: float = 5.0

    # Temporal splits (walkforward driver overrides these per-month)
    data_start: str = ""
    train_end: str = "2025-01-01"
    val_end: str = "2025-10-01"

    # Paths
    checkpoint_dir: str = "vol_model/output/patchtst"

    seed: int = 42

    @property
    def n_patches(self) -> int:
        """Derived: number of time patches per sequence."""
        return (self.seq_len - self.patch_len) // self.patch_stride + 1
