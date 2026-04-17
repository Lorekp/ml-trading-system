"""Deribit exception hierarchy.

Every error from the Deribit API surfaces as one of these subclasses.
The client module catches upstream ``httpx``/``websockets`` errors and
re-raises them as the appropriate ``DeribitError`` so the rest of the
application has a uniform exception API to match on.
"""

from __future__ import annotations


class DeribitError(Exception):
    """Base class for all Deribit client errors."""

    def __init__(
        self,
        message: str,
        code: int | None = None,
        data: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.data = data or {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={self.args[0]!r}, code={self.code})"


class DeribitAuthError(DeribitError):
    """OAuth2 client_credentials flow failed, or token expired + refresh failed."""


class DeribitRequestError(DeribitError):
    """A REST request failed with a non-auth, non-rate-limit error.

    Covers HTTP 4xx/5xx, network timeouts, and Deribit application errors
    returned as JSON-RPC error objects.
    """


class DeribitRateLimitError(DeribitError):
    """Deribit's rate limiter rejected the request (code 10028)."""


class DeribitOrderRejected(DeribitError):
    """An order was rejected by Deribit.

    Distinct from ``DeribitRequestError`` because the rejection reason
    determines whether the caller should retry (e.g. ``too_many_requests``
    → retry with backoff) or surface the error to the user
    (``insufficient_funds`` → halt).
    """

    def __init__(
        self,
        message: str,
        code: int | None = None,
        data: dict | None = None,
        retryable: bool = False,
    ):
        super().__init__(message, code=code, data=data)
        self.retryable = retryable


class DeribitWebSocketError(DeribitError):
    """WebSocket connection or subscription failure.

    Raised for reconnection failures, subscription rejections, and
    protocol violations.  Normal reconnect attempts are handled
    internally by ``DeribitWebSocketClient`` and do not raise; this
    exception only surfaces when reconnect attempts have been exhausted
    or the server has returned an unrecoverable error.
    """
