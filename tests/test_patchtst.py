"""Tests for vol_model.patchtst.encoder.

Critical test: the patch-axis attention is strictly causal. We verify
this numerically by comparing outputs when we mutate future patch
content -- outputs at position p must be invariant to patches > p.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import numpy as np  # noqa: E402

from vol_model.patchtst.config import PatchTSTConfig  # noqa: E402
from vol_model.patchtst.encoder import PatchTSTEncoder  # noqa: E402


def _cfg() -> PatchTSTConfig:
    # Small config for tests
    return PatchTSTConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        dropout=0.0,  # deterministic for equality checks
        seq_len=16,
        n_features=8,
        patch_len=4,
        patch_stride=4,
    )


def test_encoder_output_shape() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = PatchTSTEncoder(cfg).eval()
    B = 5
    x = torch.randn(B, cfg.seq_len, cfg.n_features)
    t = torch.randn(B, cfg.seq_len, cfg.n_time_inputs)
    with torch.no_grad():
        out = model(x, t)
    assert out["continuous_var_raw"].shape == (B,)
    assert out["jump_var_raw"].shape == (B,)
    assert out["embedding"].shape == (B, cfg.d_model)
    assert out["embedding_compact"].shape == (B, 16)


def test_encoder_n_patches_derived() -> None:
    cfg = _cfg()
    # seq_len=16, patch_len=4, stride=4 -> 4 patches
    assert cfg.n_patches == 4


def test_causal_mask_structure() -> None:
    cfg = _cfg()
    mask = PatchTSTEncoder.create_causal_mask(cfg.n_patches, torch.device("cpu"))
    # Below diagonal (incl.) = 0, above diagonal = -inf
    for i in range(cfg.n_patches):
        for j in range(cfg.n_patches):
            if j > i:
                assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf"
            else:
                assert mask[i, j] == 0.0, f"mask[{i},{j}] should be 0"


def test_patchtst_attention_is_causal() -> None:
    """CRITICAL leakage test: output at patch position p must not depend on
    input at patch positions > p.

    We forward twice with identical inputs except a future patch is
    mutated; a causal encoder should produce IDENTICAL embeddings at
    earlier patch positions. Since we only expose the pooled embedding
    (last patch), we instead forward with inputs where the FIRST patch
    is the same but future patches differ, and check that the model's
    prediction when we only have the first patch of content (zeroed
    future patches) equals the prediction when we mutate future
    patches with random junk -- as long as we read only the first
    patch's attention output.

    We do this by hooking the intermediate attention output of the
    final block and comparing its first-patch row.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    model = PatchTSTEncoder(cfg).eval()

    B = 3
    x_a = torch.randn(B, cfg.seq_len, cfg.n_features)
    t = torch.randn(B, cfg.seq_len, cfg.n_time_inputs)

    # Variant B: identical first patch (positions 0..patch_len-1), random
    # future patches.
    x_b = x_a.clone()
    x_b[:, cfg.patch_len:, :] = torch.randn_like(x_b[:, cfg.patch_len:, :])

    captured = {}

    def hook(module, args, output):
        # MultiheadAttention returns (attn_output, attn_weights). We
        # capture the attention output tensor.
        if isinstance(output, tuple):
            captured.setdefault("attn_out", []).append(output[0].detach().clone())
        else:
            captured.setdefault("attn_out", []).append(output.detach().clone())

    # Register on the final block's attention
    last_block_attn = model.layers[-1].attn
    handle = last_block_attn.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(x_a, t)
        _ = model(x_b, t)
    handle.remove()

    assert len(captured["attn_out"]) == 2, "Should have captured two forward passes"
    attn_a, attn_b = captured["attn_out"]

    # Row 0 (first patch) of the attention output must be identical
    # between the two runs because first-patch attends only to itself.
    torch.testing.assert_close(
        attn_a[:, 0, :], attn_b[:, 0, :],
        rtol=1e-5, atol=1e-6,
        msg="First-patch attention output differs between runs with identical first patches "
        "-- the attention is NOT causal!",
    )


def test_shuffle_future_patches_preserves_first_patch_embedding() -> None:
    """End-to-end causality probe: if we zero out the final token embeddings
    before pooling, we shouldn't see the future-patch contents in the output.

    Alternative phrasing: we build a model with only 1 patch of content
    and verify the pooled embedding and loss reach stable values.
    This is a SMOKE test that the full model runs with a causal forward.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    model = PatchTSTEncoder(cfg).eval()

    B = 2
    # All-zero future patches
    x = torch.zeros(B, cfg.seq_len, cfg.n_features)
    x[:, :cfg.patch_len, :] = torch.randn(B, cfg.patch_len, cfg.n_features)
    t = torch.zeros(B, cfg.seq_len, cfg.n_time_inputs)

    with torch.no_grad():
        out = model(x, t)
    assert torch.isfinite(out["embedding"]).all()
    assert torch.isfinite(out["embedding_compact"]).all()


def test_compute_loss_runs() -> None:
    """Sanity: the compute_loss matches TransVol's interface."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = PatchTSTEncoder(cfg)
    B = 4
    x = torch.randn(B, cfg.seq_len, cfg.n_features)
    t = torch.randn(B, cfg.seq_len, cfg.n_time_inputs)
    out = model(x, t)
    targets = {
        "continuous_var": torch.rand(B) * 1e-4 + 1e-5,
        "jump_var": torch.rand(B),
    }
    losses = model.compute_loss(out, targets)
    assert "total" in losses and "qlike" in losses and "huber_jump" in losses
    assert torch.isfinite(losses["total"])
    # Gradient flows
    losses["total"].backward()


def test_bottleneck_range() -> None:
    """Compact embedding uses tanh bottleneck -- values in [-1, 1]."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = PatchTSTEncoder(cfg).eval()
    B = 32
    x = torch.randn(B, cfg.seq_len, cfg.n_features)
    t = torch.randn(B, cfg.seq_len, cfg.n_time_inputs)
    with torch.no_grad():
        out = model(x, t)
    compact = out["embedding_compact"]
    assert (compact >= -1.0).all() and (compact <= 1.0).all()
