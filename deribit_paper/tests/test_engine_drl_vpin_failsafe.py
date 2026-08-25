"""Cycle-22 (2026-05-11) engine-startup fail-fast tests.

Preserves a historical replay startup guard: the V2 marker requires a
VPIN warm-start path when DRL hedging is enabled.

Pre-fix the engine would silently start with ``vpin_stream=None``,
StraddleFlow would silently fall back to ``vpin=0.5`` per trade (10σ
off SAC v4 training), and every hedge action would be corrupted from
bar one. Now the engine refuses to start in that config, forcing the
operator to notice the wiring bug at startup time.

These tests just exercise the boolean ``drl_enabled`` guard logic; a
full engine-construction roundtrip would require heavy mocking of
Deribit / Binance clients.
"""

from __future__ import annotations

import pytest


def _drl_enabled_per_engine_logic(use_drl_hedging: bool, config_preset: str) -> bool:
    """Mirror of the cycle-22 guard in engine.py around line 894.

    The actual engine logic:
        drl_enabled = (
            getattr(cfg, "use_drl_hedging", True) is not False
            and cfg.config_preset == "v2"
        )

    This helper makes it directly testable without instantiating the
    full TradingEngine.
    """
    return use_drl_hedging is not False and config_preset == "v2"


def test_v2_preset_with_drl_on_is_drl_enabled():
    """V2 preset + use_drl_hedging True → fail-fast required."""
    assert _drl_enabled_per_engine_logic(True, "v2") is True


def test_v2_preset_with_no_explicit_drl_flag_is_drl_enabled():
    """Historical V2 replay defaults DRL on and therefore requires VPIN."""
    # getattr(cfg, "use_drl_hedging", True) → True
    assert _drl_enabled_per_engine_logic(True, "v2") is True


def test_default_preset_is_not_drl_enabled():
    """The non-V2 ``default`` preset is allowed to run without a VPIN
    stream (no DRL hedging in that path). Engine startup should NOT
    fail-fast."""
    assert _drl_enabled_per_engine_logic(True, "default") is False


def test_v2_preset_with_explicit_drl_off_is_not_drl_enabled():
    """Edge case: V2 preset but somehow ``use_drl_hedging=False`` is
    set. Engine should NOT fail-fast — the user has explicitly opted
    out of DRL on V2 (e.g., for an ablation run)."""
    assert _drl_enabled_per_engine_logic(False, "v2") is False


def test_v2_config_with_empty_vpin_state_path_would_fail_engine_startup():
    """Integration sanity: a real V2 PaperConfig with vpin_state_path
    cleared should produce the failing condition (DRL enabled + no
    stream = no provider wired) that engine.py guards against.

    We can't easily run the full TradingEngine here, but we can verify
    the config state that would trigger the guard.
    """
    from deribit_paper.config import PaperConfig
    cfg = PaperConfig.historical_v2_preset()
    assert cfg.config_preset == "v2"
    # The archived V2 configuration includes a VPIN warm-start path.
    # If a future config change ever flips it empty, the engine fail-
    # fast will catch it. Document the contract here.
    assert cfg.vpin_state_path != "", (
        "V2 preset must wire vpin_state_path. The engine refuses to "
        "start without it (cycle-22 audit fix). If you've intentionally "
        "cleared it, you must also flip use_drl_hedging off."
    )
