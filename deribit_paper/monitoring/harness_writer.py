"""Per-trade row writer for the V2 reproduction harness.

C6 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Each completed (or aborted) :class:`TradeOutcome` is consolidated into a
single row matching the V2 backtest trade-log schema, then appended to
``vol_arb/output/live_trades_harness.parquet``. The reproduction harness
(I4) replays the V2 backtester on the same bar window using these rows
to attribute residual PnL gap to {config drift / feature drift / strike
rounding budget / residual}.

Schema design notes:

* Columns mirror the V2 backtest trade log on every field that lives in
  both worlds. Fields the live engine doesn't yet capture (e.g.
  ``y_true``, the full slippage decomposition) are emitted as NaN; the
  harness's Stage A diff treats those as bucketed-as-residual.
* Append-only by design — restarting the engine doesn't truncate past
  rows. The parquet round-trip uses pyarrow under pandas to keep
  schema stable across Python versions.
* Failure to write does NOT raise — harness logging is observability,
  not a critical path. We log an error and continue.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Column order is the harness contract — DO NOT shuffle without
# updating scripts/h15_alignment_harness.py.
_HARNESS_COLUMNS: list[str] = [
    "trade_id",
    "started_at",
    "ended_at",
    "signal",                 # -1/0/+1
    "n_contracts_btc",
    "strike",
    "expiry_dt",
    "holding_minutes",
    "status",
    "y_pred_pct",             # rv_pred_pct from the predictor (post β*)
    "iv_used_pct",
    "edge_vol_pts",
    "term_slope_1w_2w",       # NaN until SVI calibrator exposes it
    "profit_scale",           # E.1.1: V2 empirical-Bayes shrinkage scalar
    "kelly_mult",              # 1.0 default
    "taker_slippage_pct",     # max_slippage_pct used at IOC
    "entry_call_price",
    "entry_put_price",
    "exit_call_price",
    "exit_put_price",
    "realized_pnl_usd",
    "hedge_ticks_executed",
    "index_price_at_entry",
    "config_preset",          # "default" or "v2"
]


def _row_from_outcome(
    outcome: Any,
    *,
    realized_pnl_usd: float | None,
    kelly_mult: float = 1.0,
    taker_slippage_pct: float = 0.5,
    term_slope_1w_2w: float | None = None,
    config_preset: str = "default",
    profit_scale: float = 1.0,
) -> dict[str, Any]:
    """Flatten a :class:`TradeOutcome` plus engine-side metadata into one row.

    ``outcome`` is a ``deribit_paper.execution.straddle_flow.TradeOutcome``
    — duck-typed here so this module doesn't need to import the dataclass.
    """
    signal = outcome.signal
    straddle = signal.straddle
    started = outcome.started_at
    ended = outcome.ended_at
    holding = (
        (ended - started).total_seconds() / 60.0
        if (started is not None and ended is not None)
        else float("nan")
    )

    def _avg_price(o: Any) -> float:
        return float(o.average_price) if o is not None else float("nan")

    return {
        "trade_id": outcome.trade_id or "",
        "started_at": started.isoformat() if started is not None else "",
        "ended_at": ended.isoformat() if ended is not None else "",
        "signal": int(signal.direction),
        "n_contracts_btc": float(outcome.n_contracts_btc),
        "strike": float(straddle.strike) if straddle is not None else float("nan"),
        "expiry_dt": (
            pd.Timestamp(straddle.expiry_dt).isoformat()
            if straddle is not None else ""
        ),
        "holding_minutes": holding,
        "status": outcome.status,
        "y_pred_pct": float(signal.rv_pred_pct),
        "iv_used_pct": float(signal.iv_used_pct),
        "edge_vol_pts": float(signal.edge_vol_pts),
        "term_slope_1w_2w": (
            float(term_slope_1w_2w)
            if term_slope_1w_2w is not None else float("nan")
        ),
        "profit_scale": float(profit_scale),
        "kelly_mult": float(kelly_mult),
        "taker_slippage_pct": float(taker_slippage_pct),
        "entry_call_price": _avg_price(outcome.entry_call),
        "entry_put_price": _avg_price(outcome.entry_put),
        "exit_call_price": _avg_price(outcome.exit_call),
        "exit_put_price": _avg_price(outcome.exit_put),
        "realized_pnl_usd": (
            float(realized_pnl_usd)
            if realized_pnl_usd is not None else float("nan")
        ),
        "hedge_ticks_executed": int(
            outcome.hedge_summary.n_ticks_executed
            if outcome.hedge_summary is not None else 0
        ),
        "index_price_at_entry": (
            float(straddle.index_price)
            if straddle is not None else float("nan")
        ),
        "config_preset": str(config_preset),
    }


class HarnessWriter:
    """Append-only writer for the harness parquet.

    Concurrency contract (E.1.3 2026-05-01):

    * Reads + writes are gated behind an :class:`asyncio.Lock` so two
      concurrent ``_execute_and_persist`` tasks never race each other.
      Without the lock the read-modify-write pattern below silently
      drops rows when both tasks read the same parquet then both write.
    * Writes go to a sibling ``.tmp.<uuid>`` file then ``os.replace`` to
      the final path. This is **atomic on POSIX and on Windows** for
      same-volume renames — a crash mid-write leaves the prior parquet
      intact rather than zeroing it.
    * The lock is per-instance; tests pass a fresh instance per case so
      no test sees state from another.

    Designed for low-frequency calls (one per closed trade) so the
    full read+rewrite cost is negligible at V2's ~150-300 trades/week.
    """

    def __init__(self, output_path: str | Path):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _atomic_write(self, df: pd.DataFrame) -> None:
        """Write ``df`` to ``self.path`` atomically via tmp+rename."""
        tmp = self.path.with_suffix(
            f".tmp.{uuid.uuid4().hex}{self.path.suffix}",
        )
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, self.path)
        except Exception:
            # Best-effort cleanup of orphan tmp on failure.
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise

    def _append_sync(self, row: dict[str, Any]) -> None:
        """Read-modify-write under no lock (called by both sync and async paths).
        The lock-acquisition is the caller's responsibility."""
        try:
            new_row = pd.DataFrame([row], columns=_HARNESS_COLUMNS)
            if self.path.exists():
                existing = pd.read_parquet(self.path)
                # Defensive: drop columns not in our contract; add missing
                # as NaN so old parquets can be migrated without breaking.
                for col in _HARNESS_COLUMNS:
                    if col not in existing.columns:
                        existing[col] = np.nan
                existing = existing[_HARNESS_COLUMNS]
                merged = pd.concat([existing, new_row], ignore_index=True)
            else:
                merged = new_row
            self._atomic_write(merged)
        except Exception as exc:  # noqa: BLE001 — observability, not critical path
            logger.error(
                "HarnessWriter append failed (%s): %s", self.path.name, exc,
            )

    def append(self, row: dict[str, Any]) -> None:
        """Synchronous append.

        Kept for back-compat with tests + non-async callers. The async
        engine path goes through :meth:`consolidate_and_write` which
        gates on :attr:`_lock`.
        """
        self._append_sync(row)

    async def consolidate_and_write(
        self,
        outcome: Any,
        *,
        realized_pnl_usd: float | None,
        kelly_mult: float = 1.0,
        taker_slippage_pct: float = 0.5,
        term_slope_1w_2w: float | None = None,
        config_preset: str = "default",
        profit_scale: float = 1.0,
    ) -> None:
        """High-level entry point: consolidate a TradeOutcome into a row
        and append. Public method used by the engine's _execute_and_persist
        hook.

        Held under the per-instance ``asyncio.Lock`` so two concurrent
        trades don't lose a row to a read-modify-write race.
        """
        if outcome is None or outcome.signal is None:
            return
        row = _row_from_outcome(
            outcome,
            realized_pnl_usd=realized_pnl_usd,
            kelly_mult=kelly_mult,
            taker_slippage_pct=taker_slippage_pct,
            term_slope_1w_2w=term_slope_1w_2w,
            config_preset=config_preset,
            profit_scale=profit_scale,
        )
        async with self._lock:
            self._append_sync(row)


__all__ = ["HarnessWriter", "_HARNESS_COLUMNS"]
