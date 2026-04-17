"""PatchTST encoder -- drop-in replacement candidate for TransVol.

See ``vol_model/patchtst/encoder.py`` for the model.
See ``vol_model/patchtst/train.py`` for the training driver (training
itself is deferred to an explicit user-approved run; see SIGNAL_RESEARCH_NOTES.md).

Reference:
    Nie, Nguyen, Sinthong, Kalagnanam. "A Time Series is Worth 64 Words:
    Long-term Forecasting with Transformers." ICLR 2023.

Design deviations from the reference paper (because we're an encoder for
a downstream LGBM, not a direct forecaster):
  1. **Cross-channel attention retained.** Reference PatchTST is
     channel-independent (each of F=101 features processed separately).
     We keep cross-feature attention because the LGBM downstream benefits
     from embeddings that encode *feature interactions*. We use patching
     in the TIME dimension only.
  2. **Last-patch pooling + persistence skip.** Matches TransVol's
     contract: bottleneck a (B, 16) compact embedding that LightGBM
     consumes.
  3. **Strict causality.** Attention mask blocks any patch from attending
     to strictly-later patches. Unit-tested.
"""
