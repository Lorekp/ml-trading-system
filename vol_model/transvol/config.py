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

    # ── R2 (TransVol rehab, 2026-06-22): log-target for continuous_var ──
    # When True the continuous-variance head predicts log(continuous_var)
    # directly (raw skew 309 -> log skew -0.39, near-Gaussian) with a plain
    # MSE loss in log space — removing the softplus(bias=-11) + raw-1e-5
    # gradient pathology the raw-variance target required. The head bias is
    # re-initialised to ``cont_log_bias`` (train-mean of the log-target) so
    # the model starts calibrated. When False (PRODUCTION DEFAULT) the model
    # is byte-identical to the deployed ratio-form QLIKE + softplus head.
    target_log: bool = False
    cont_log_bias: float = 0.0   # train-mean of log(continuous_var); set per fold

    # Temporal splits (ISO dates, UTC)
    data_start: str = ""           # empty = use all data (no L2 filter)
    train_end: str = "2025-01-01"
    val_end: str = "2025-10-01"

    # Paths
    checkpoint_dir: str = "vol_model/output/transvol"

    seed: int = 42
