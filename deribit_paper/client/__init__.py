"""Deribit async client — REST (httpx) + WebSocket (websockets) + auth."""

from deribit_paper.client.auth import DeribitAuth
from deribit_paper.client.errors import (
    DeribitAuthError,
    DeribitError,
    DeribitOrderRejected,
    DeribitRateLimitError,
    DeribitRequestError,
    DeribitWebSocketError,
)
from deribit_paper.client.rest import DeribitRESTClient
from deribit_paper.client.ws import DeribitWebSocketClient

__all__ = [
    "DeribitAuth",
    "DeribitRESTClient",
    "DeribitWebSocketClient",
    "DeribitError",
    "DeribitAuthError",
    "DeribitRequestError",
    "DeribitRateLimitError",
    "DeribitOrderRejected",
    "DeribitWebSocketError",
]
