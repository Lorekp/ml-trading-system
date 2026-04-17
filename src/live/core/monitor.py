"""Execution quality monitor — rolling comparison of actual vs expected.

Checks fill rate, slippage, and latency against calibration baselines.
Alerts via Telegram when thresholds are exceeded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from live.core.state import StateDB
from live.notifier import TelegramNotifier

logger = logging.getLogger(__name__)

# Calibration baselines — kept as documentation defaults; runtime uses MonitorConfig
EXPECTED_MAKER_FILL_RATE = 0.77
EXPECTED_SPREAD_BPS = 1.22
EXPECTED_AVG_COST_BPS = 3.9
FILL_RATE_DRIFT_THRESHOLD = 0.10
SLIPPAGE_ALERT_BPS = 5.0
LATENCY_P95_ALERT_MS = 65_000


@dataclass
class TCAReport:
    """Summary of execution quality over a window."""
    window_hours: int
    n_trades: int
    maker_fill_rate: float
    maker_count: int
    taker_count: int
    blended_count: int
    avg_shortfall_bps: float
    avg_arrival_cost_bps: float
    avg_spread_bps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    fill_rate_alert: bool
    slippage_alert: bool
    latency_alert: bool
    expected_fill_rate: float = EXPECTED_MAKER_FILL_RATE

    def format_telegram(self) -> str:
        alerts = []
        if self.fill_rate_alert:
            alerts.append("FILL RATE")
        if self.slippage_alert:
            alerts.append("SLIPPAGE")
        if self.latency_alert:
            alerts.append("LATENCY")

        header = "<b>TCA SUMMARY</b>"
        if alerts:
            header += f" [ALERT: {', '.join(alerts)}]"

        return (
            f"{header}\n"
            f"Trades: {self.n_trades} ({self.maker_count}M/{self.taker_count}T/{self.blended_count}B)\n"
            f"Fill rate: {self.maker_fill_rate:.0%} "
            f"(expected {self.expected_fill_rate:.0%}, "
            f"drift {self.maker_fill_rate - self.expected_fill_rate:+.0%})\n"
            f"Shortfall: {self.avg_shortfall_bps:.1f} bps\n"
            f"Latency: p50={self.latency_p50_ms/1000:.1f}s "
            f"p95={self.latency_p95_ms/1000:.1f}s "
            f"p99={self.latency_p99_ms/1000:.1f}s"
        )


class ExecutionMonitor:
    """Monitor execution quality and alert on degradation."""

    def __init__(self, state: StateDB, notifier: Optional[TelegramNotifier] = None,
                 config=None):
        self._state = state
        self._notifier = notifier
        # Use config if provided, else module-level defaults
        if config is not None:
            self._expected_fill_rate = config.expected_maker_fill_rate
            self._fill_rate_drift = config.fill_rate_drift_threshold
            self._slippage_alert_bps = config.slippage_alert_bps
            self._latency_p95_alert_ms = config.latency_p95_alert_ms
        else:
            self._expected_fill_rate = EXPECTED_MAKER_FILL_RATE
            self._fill_rate_drift = FILL_RATE_DRIFT_THRESHOLD
            self._slippage_alert_bps = SLIPPAGE_ALERT_BPS
            self._latency_p95_alert_ms = LATENCY_P95_ALERT_MS

    async def generate_report(self, window_hours: int = 24) -> Optional[TCAReport]:
        """Generate TCA report for the last N hours."""
        records = await self._state.get_recent_tca(hours=window_hours)

        if not records:
            return None

        n = len(records)

        # Fill rate
        methods = [r["method"] for r in records]
        maker_count = sum(1 for m in methods if m == "maker")
        taker_count = sum(1 for m in methods if m == "taker")
        blended_count = sum(1 for m in methods if m == "blended")

        fill_rates = [r["maker_fill_rate"] for r in records if r["maker_fill_rate"] is not None]
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 0.0

        # Costs
        shortfalls = [r["impl_shortfall_bps"] for r in records
                      if r["impl_shortfall_bps"] is not None]
        arrival_costs = [r["arrival_cost_bps"] for r in records
                         if r["arrival_cost_bps"] is not None]
        spreads = [r["spread_cost_bps"] for r in records
                   if r["spread_cost_bps"] is not None]

        # Latencies
        latencies = [r["submit_to_complete_ms"] for r in records
                     if r["submit_to_complete_ms"] is not None and r["submit_to_complete_ms"] > 0]

        report = TCAReport(
            window_hours=window_hours,
            n_trades=n,
            maker_fill_rate=float(avg_fill_rate),
            maker_count=maker_count,
            taker_count=taker_count,
            blended_count=blended_count,
            avg_shortfall_bps=float(np.mean(shortfalls)) if shortfalls else 0.0,
            avg_arrival_cost_bps=float(np.mean(arrival_costs)) if arrival_costs else 0.0,
            avg_spread_bps=float(np.mean(spreads)) if spreads else 0.0,
            latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
            latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            latency_p99_ms=float(np.percentile(latencies, 99)) if latencies else 0.0,
            fill_rate_alert=abs(float(avg_fill_rate) - self._expected_fill_rate) > self._fill_rate_drift,
            slippage_alert=(float(np.mean(shortfalls)) if shortfalls else 0.0) > self._slippage_alert_bps,
            latency_alert=(float(np.percentile(latencies, 95)) if latencies else 0.0) > self._latency_p95_alert_ms,
            expected_fill_rate=self._expected_fill_rate,
        )

        # Alert if any threshold breached
        if report.fill_rate_alert or report.slippage_alert or report.latency_alert:
            logger.warning("TCA alert: fill_rate=%s slippage=%s latency=%s",
                           report.fill_rate_alert, report.slippage_alert, report.latency_alert)
            if self._notifier:
                await self._notifier.send(report.format_telegram())

        return report
