"""Cycle-33 (2026-05-13): diagnostic instrumentation for live↔offline parity tracing.

The :mod:`bar_trace` submodule emits a per-bar JSONL trace capturing
every intermediate value in the signal pipeline so the offline
backtester's parallel trace can be diff'd field-by-field.

Default-OFF: nothing in this package is exercised unless
``PaperConfig.feature_trace_path`` is set (or the env var
``DERIBIT_PAPER_FEATURE_TRACE_PATH`` is non-empty at engine construction).
"""
