"""Re-export notifier from parent package — already strategy-agnostic."""
from live.notifier import TelegramNotifier

__all__ = ["TelegramNotifier"]
