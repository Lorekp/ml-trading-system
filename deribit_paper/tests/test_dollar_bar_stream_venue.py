"""Cycle-31 (2026-05-13) regression tests for the Binance dollar-bar
stream's venue.

Pre-cycle-31 the engine constructed
``BinanceRESTClient.for_spot()`` for the dollar-bar stream's REST
client — pointing at ``https://api.binance.com/api/v3/aggTrades``
(Binance SPOT). The F3/F6/F8 docstring claimed the offline pipeline
trained on Binance Vision SPOT aggTrades, but the canonical offline
downloader (``scripts/download_binance_aggtrades.sh:10``) actually
pulls from ``data.binance.vision/data/futures/um/monthly/aggTrades``
i.e. Binance USD-M FUTURES.

Live consuming spot vs offline trained on futures was a venue
mismatch the models were never trained for. Cycle-31 flips the live
REST client back to the default (futures).

These tests pin:
1. ``engine.py``'s dollar-bar REST client uses the futures endpoint,
   NOT ``BinanceRESTClient.for_spot()``.
2. The futures URL ``fapi.binance.com/fapi/v1/aggTrades`` is the
   canonical default of ``BinanceRESTClient``.
3. ``scripts/download_binance_aggtrades.sh`` references the futures
   Vision URL — sentinel against a future operator switching to spot.
"""

from __future__ import annotations

import re
from pathlib import Path


def test_engine_dollar_bar_uses_futures_rest_client():
    """Cycle-31: engine.py must construct BinanceRESTClient() (default
    futures) for the dollar-bar stream, NOT BinanceRESTClient.for_spot()."""
    src = Path(__file__).resolve().parents[1].joinpath("engine.py").read_text()

    # Locate the dollar-bar REST construction block. The line below
    # is the cycle-31 corrected form; the pre-cycle-31 form was
    # ``BinanceRESTClient.for_spot()``.
    assert re.search(
        r"self\._dollar_bar_rest\s*=\s*BinanceRESTClient\(\)",
        src,
    ), (
        "cycle-31: engine.py must construct BinanceRESTClient() "
        "(default futures) for the dollar-bar stream; "
        "BinanceRESTClient.for_spot() is the pre-cycle-31 venue-mismatch form"
    )
    # Must NOT use the spot factory for the dollar-bar stream.
    # (LiveVPINStream uses BinanceRESTClient() but the literal
    # ``_dollar_bar_rest = BinanceRESTClient.for_spot()`` is a
    # cycle-31 regression sentinel — it must not reappear.)
    assert not re.search(
        r"_dollar_bar_rest\s*=\s*BinanceRESTClient\.for_spot\(\)",
        src,
    ), (
        "cycle-31 regression: engine.py must not call "
        "BinanceRESTClient.for_spot() for the dollar-bar stream — "
        "use BinanceRESTClient() (default futures) to match the "
        "offline training venue"
    )


def test_binance_rest_client_default_is_futures():
    """Sanity: BinanceRESTClient's default base_url is the futures
    endpoint (``fapi.binance.com``)."""
    from deribit_paper.client.binance_rest import BinanceRESTClient

    client = BinanceRESTClient()
    # The base URL is captured at __init__ time; expose via the
    # internal field (no public getter — this is a test).
    assert "fapi.binance.com" in client._base_url, (
        f"expected futures default base_url, got {client._base_url!r}"
    )
    assert "/fapi/v1/aggTrades" in client._agg_trades_path, (
        f"expected futures aggTrades path, got {client._agg_trades_path!r}"
    )


def test_offline_downloader_targets_futures_vision():
    """Cross-reference: the canonical offline aggTrades downloader
    must target Binance Vision FUTURES URL (USD-M perp). A future
    refactor switching to the spot URL would reintroduce the cycle-31
    venue mismatch from the OPPOSITE side (live now correct, offline
    drifted to spot) — pin both sides."""
    src = Path(__file__).resolve().parents[2].joinpath(
        "scripts", "download_binance_aggtrades.sh",
    ).read_text()
    assert re.search(
        r"data\.binance\.vision/data/futures/um/[\w/]+aggTrades",
        src,
    ), (
        "cycle-31: scripts/download_binance_aggtrades.sh must reference "
        "the Binance Vision FUTURES (USD-M) URL — anchor for live-canon parity"
    )
