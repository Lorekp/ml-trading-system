"""Exchange adapter base — re-exports the protocol from core.interfaces."""

from live.core.interfaces import ExchangeAdapter, DataProvider

__all__ = ["ExchangeAdapter", "DataProvider"]
