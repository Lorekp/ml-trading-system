"""Deribit API client with auth, rate limiting, retries, and backoff."""

from __future__ import annotations

import logging
import time
from json import JSONDecodeError

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from downloader.config import DownloadConfig
from downloader.rate_limiter import TokenBucketLimiter

logger = logging.getLogger(__name__)


class DeribitAPIError(Exception):
    def __init__(self, error: dict):
        self.code = error.get("code", 0)
        self.message = error.get("message", str(error))
        super().__init__(f"Deribit API error {self.code}: {self.message}")


class MaxRetriesExceeded(Exception):
    pass


class DeribitClient:
    """Thread-safe Deribit API client with credit-based rate limiting."""

    BASE_URL = "https://www.deribit.com/api/v2/public"
    HISTORY_URL = "https://history.deribit.com/api/v2/public"
    AUTH_URL = "https://www.deribit.com/api/v2/public/auth"

    # Endpoints with higher credit cost
    EXPENSIVE_ENDPOINTS = {"get_instruments": 10_000}

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.limiter = TokenBucketLimiter(
            pool=config.credit_pool,
            refill_per_sec=config.credit_refill_per_sec,
            cost=config.credits_per_request,
        )
        self._access_token: str | None = None
        self._token_expiry: float = 0

        # Session with connection pooling and transport-level retries
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10, pool_maxsize=10,
            max_retries=Retry(total=2, backoff_factor=0.3,
                              status_forcelist=[500, 502, 503]),
        )
        self.session.mount("https://", adapter)

        if config.authenticated:
            self._authenticate()

    def _authenticate(self) -> None:
        """Get access token via client_credentials grant."""
        resp = self.session.get(self.AUTH_URL, params={
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "grant_type": "client_credentials",
        }, timeout=15)
        data = resp.json()
        if "error" in data:
            logger.error("Auth failed: %s", data["error"])
            raise DeribitAPIError(data["error"])
        result = data["result"]
        self._access_token = result["access_token"]
        self._token_expiry = time.time() + result.get("expires_in", 900) - 60
        logger.info("Authenticated (token expires in %ds)", result.get("expires_in", 0))

    def _ensure_auth(self) -> dict:
        """Return auth headers, refreshing token if needed."""
        if not self.config.authenticated:
            return {}
        if time.time() > self._token_expiry:
            self._authenticate()
        return {"Authorization": f"Bearer {self._access_token}"}

    def request(self, endpoint: str, params: dict,
                use_history: bool = False) -> dict:
        """Single request with rate limiting, retries, and backoff.

        Args:
            endpoint: API method name (e.g. "get_volatility_index_data")
            params: Query parameters
            use_history: Use history.deribit.com instead of www.deribit.com

        Returns:
            The "result" field from the JSON response.
        """
        base = self.HISTORY_URL if use_history else self.BASE_URL
        url = f"{base}/{endpoint}"
        cost = self.EXPENSIVE_ENDPOINTS.get(endpoint, self.config.credits_per_request)

        for attempt in range(self.config.max_retries):
            self.limiter.acquire(cost)
            try:
                resp = self.session.get(
                    url, params=params, headers=self._ensure_auth(), timeout=30)
                data = resp.json()
            except requests.exceptions.Timeout:
                wait = min(self.config.backoff_base ** (attempt + 1), self.config.max_backoff)
                logger.debug("Timeout on %s, retry in %.1fs", endpoint, wait)
                time.sleep(wait)
                continue
            except requests.exceptions.ConnectionError:
                wait = min(self.config.backoff_base ** (attempt + 2), self.config.max_backoff)
                logger.debug("Connection error on %s, retry in %.1fs", endpoint, wait)
                time.sleep(wait)
                continue
            except JSONDecodeError:
                wait = min(self.config.backoff_base ** (attempt + 1), self.config.max_backoff)
                logger.debug("JSON decode error on %s, retry in %.1fs", endpoint, wait)
                time.sleep(wait)
                continue

            # Check for API-level errors
            if "error" in data:
                code = data["error"].get("code", 0)
                msg = data["error"].get("message", "")

                if code == 10028 or "too_many_requests" in msg.lower():
                    self.limiter.drain()
                    wait = min(self.config.backoff_base ** (attempt + 2), self.config.max_backoff)
                    logger.warning("Rate limited (10028), backing off %.1fs", wait)
                    time.sleep(wait)
                    continue

                # Non-retryable API error
                raise DeribitAPIError(data["error"])

            logger.debug("OK %s (%d credits left)", endpoint, self.limiter.stats["credits_available"])
            return data.get("result", data)

        raise MaxRetriesExceeded(f"{endpoint} after {self.config.max_retries} attempts")
