"""TransVol configuration."""

from dataclasses import dataclass


@dataclass
class TransVolConfig:
    # Architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.15

    # Input
    seq_len: int = 32       # T: consecutive 15m windows
    n_features: int = 101   # F: features per window (126 - 25 dropped OFI/depth/OI)

    # Time2Vec
    time2vec_dim: int = 16  # periodic components per time input
    n_time_inputs: int = 2  # duration_s, n_bars_in_window

    # Training
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 150
    batch_size: int = 512
    grad_accum_steps: int = 2    # effective batch = 1024
    patience: int = 20
    grad_clip: float = 1.0

    # MTL (fixed weights — dynamic weighting failed in V2)
    loss_weight_qlike: float = 0.8
    loss_weight_jump: float = 0.2
    huber_delta: float = 5.0    # Huber delta on normalized jump_var (mean=1.0, delta=5x mean)

    # Temporal splits (ISO dates, UTC)
    data_start: str = ""           # empty = use all data (no L2 filter)
    train_end: str = "2025-01-01"
    val_end: str = "2025-10-01"

    # Paths
    checkpoint_dir: str = "vol_model/output/transvol"

    seed: int = 42
