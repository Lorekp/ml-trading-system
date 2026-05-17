#!/usr/bin/env python
"""Cycle-38 (2026-05-17): replay-parity harness for the BTC vol-arb engine.

The parity campaign's open question: when the LIVE engine and the
OFFLINE backtester produce a different per-bar ``edge`` / ``signal``,
is it a CODE bug (the two pipelines compute differently) or a DATA
bug (the live WS capture diverged from the canonical offline parquet)?

This harness isolates the first from the second. It drives the LIVE
engine's ``process_bar`` feature/signal pipeline with the SAME
canonical offline parquet data the backtester consumes. Because the
inputs are now byte-identical to offline by construction, any residual
divergence in the resulting per-bar trace is a CODE divergence — the
live-WS data-capture noise has been removed from the equation.

It does NOT add new trace plumbing. The engine's ``process_bar``
already emits a :class:`~deribit_paper.diagnostics.bar_trace.BarTrace`
per bar via :class:`BarTraceWriter` when ``config.feature_trace_path``
is set (cycles 33-35). This harness simply:

  1. Builds a real :class:`TradingEngine` in paper + dry-run mode and
     ``await initialize()``-s it (paper mode connects unauthenticated;
     opening the REST session is fine; ``run()`` is NEVER called so no
     WS streaming starts).
  2. For each 15-min bar T in the CLI window, prewarms every rolling
     buffer to its bar-close-T state from canonical parquets capped
     ``as_of=T``, reconstructs the canonical :class:`Bar` for T, and
     ``await engine.process_bar(bar)`` — the existing ``BarTraceWriter``
     captures the C0-C8 "replay" trace.
  3. Emits a replay-trace JSONL identical in schema to the production
     ``bar_trace_live.jsonl``, ready for ``scripts/feature_trace_diff.py``
     to diff against the offline trace built by
     ``scripts/build_offline_bar_trace.py``.

Canonical data sources (all capped ``<= T``):

  * ``vol_arb/output/BTC_vol_arb_svi.parquet`` — per-15min-bar canon
    (index_price, vol_ann, jump_fraction, tsrv, svi_iv_matched, ...).
  * ``data/deribit/trades/BTC_option/<date>.parquet`` — Deribit option
    trades (SVI calibration input + ATM straddle selection).
  * ``data/deribit/trades/BTC_future/<date>.parquet`` — Deribit perp /
    future trades (``Bar.perp_trades``).
  * ``data/binance-aggTrades-parquet/BTCUSDT/<month>.parquet`` —
    Binance USD-M futures aggTrades (the TSRV / dollar-bar input,
    ``Bar.tsrv_trades``).
  * ``TSRV/output/BTCUSDT_tsrv_15m.parquet`` — TSRV history prewarm.
  * ``vol_model/output/BTCUSDT_features_15m.parquet`` — TransVol /
    HAR-PD feature prewarm (used internally by the assembler).

Usage::

    python scripts/replay_parity_harness.py \
        --start "2026-05-14 00:15:00+00:00" \
        --end   "2026-05-14 01:00:00+00:00" \
        --output deribit_paper/logs/bar_trace_replay.jsonl

Then diff against the offline trace::

    python scripts/feature_trace_diff.py \
        --live deribit_paper/logs/bar_trace_replay.jsonl \
        --offline vol_arb/output/bar_trace_offline.jsonl

The harness is PURELY ADDITIVE: it imports the production engine and
buffers but mutates nothing in them. The only engine-state override is
harness-local (a pre-populated :class:`StrikePicker` instrument cache
so ATM straddle selection is deterministic and network-free) — see
:func:`_install_offline_strike_picker`.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Make the repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deribit_paper.config import PaperConfig  # noqa: E402
from deribit_paper.market_data.aggregator import Bar  # noqa: E402
from deribit_paper.market_data.buffer import Trade  # noqa: E402
from deribit_paper.market_data.subscriber import (  # noqa: E402
    parse_option_instrument,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("replay_parity_harness")


# ── Canonical data-source paths (relative to repo root) ─────────────

CANON_SVI_PARQUET = "vol_arb/output/BTC_vol_arb_svi.parquet"
DERIBIT_OPTION_DIR = "data/deribit/trades/BTC_option"
DERIBIT_FUTURE_DIR = "data/deribit/trades/BTC_future"
BINANCE_AGGTRADES_DIR = "data/binance-aggTrades-parquet/BTCUSDT"

BAR_MINUTES = 15

# ``BTC-PERPETUAL`` is the Deribit instrument name for the perp.
_DERIBIT_FUTURE_RE = re.compile(r"^BTC-(PERPETUAL|\d{1,2}[A-Z]{3}\d{2})$")


# ── Reconstruction helpers ──────────────────────────────────────────


def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    """Coerce a string/Timestamp to a UTC-aware :class:`pd.Timestamp`."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _bar_grid(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return the list of 15-min bar-close timestamps in ``[start, end]``.

    Both ends are floored to the 15-min grid so an off-grid CLI value
    still produces canonical bar boundaries.
    """
    freq = f"{BAR_MINUTES}min"
    s = start.ceil(freq)
    e = end.floor(freq)
    if e < s:
        return []
    return list(pd.date_range(s, e, freq=freq))


def _direction_from_str(value: object) -> int:
    """Map a Deribit ``direction`` field to the engine's +1/-1 convention."""
    if isinstance(value, str):
        return 1 if value.lower() == "buy" else -1
    try:
        return 1 if int(value) >= 0 else -1
    except (TypeError, ValueError):
        return 1


def _load_deribit_day(directory: str, day: pd.Timestamp) -> pd.DataFrame:
    """Load one day's Deribit trade parquet, or an empty frame if absent."""
    path = _REPO_ROOT / directory / f"{day.strftime('%Y-%m-%d')}.parquet"
    if not path.exists():
        logger.warning("Deribit parquet missing: %s", path)
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows with ``start <= timestamp < end`` (half-open bar window)."""
    if df.empty or "timestamp" not in df.columns:
        return df.iloc[0:0]
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df.loc[mask]


def _perp_trades_for_window(
    future_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Trade]:
    """Reconstruct ``Bar.perp_trades`` from canonical Deribit future trades.

    Only ``BTC-PERPETUAL`` rows are kept — the engine's perp buffer is
    fed by ``trades.BTC-PERPETUAL`` in production. Future-dated
    instruments (``BTC-26JUN26`` etc.) are excluded so the slice
    matches the live subscriber's channel scope.
    """
    win = _slice_window(future_df, start, end)
    trades: list[Trade] = []
    for row in win.itertuples(index=False):
        name = getattr(row, "instrument_name", "BTC-PERPETUAL")
        if name != "BTC-PERPETUAL":
            continue
        price = float(getattr(row, "price", 0.0) or 0.0)
        trades.append(
            Trade(
                trade_id=str(getattr(row, "trade_id", "")),
                timestamp=getattr(row, "timestamp"),
                instrument_name=name,
                price=price,
                amount=float(getattr(row, "amount", 0.0) or 0.0),
                direction=_direction_from_str(getattr(row, "direction", "buy")),
                index_price=float(getattr(row, "index_price", price) or price),
                mark_price=float(getattr(row, "mark_price", price) or price),
            )
        )
    return trades


def _option_trades_for_window(
    option_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Trade]:
    """Reconstruct option :class:`Trade` rows for ``[start, end)``.

    Parses each ``instrument_name`` (``BTC-15MAY26-80000-C``) into
    ``(expiry_dt, strike, option_type)`` via the production
    :func:`parse_option_instrument` so the live SVI calibrator sees
    byte-identical instrument metadata.
    """
    win = _slice_window(option_df, start, end)
    trades: list[Trade] = []
    for row in win.itertuples(index=False):
        name = str(getattr(row, "instrument_name", ""))
        parsed = parse_option_instrument(name)
        if parsed is None:
            continue
        expiry_dt, strike, option_type = parsed
        iv = getattr(row, "iv", None)
        trades.append(
            Trade(
                trade_id=str(getattr(row, "trade_id", "")),
                timestamp=getattr(row, "timestamp"),
                instrument_name=name,
                price=float(getattr(row, "price", 0.0) or 0.0),
                amount=float(getattr(row, "amount", 0.0) or 0.0),
                direction=_direction_from_str(getattr(row, "direction", "buy")),
                strike=strike,
                option_type=option_type,
                expiry_dt=expiry_dt,
                iv=float(iv) if iv is not None and pd.notna(iv) else None,
                index_price=float(getattr(row, "index_price", 0.0) or 0.0),
                mark_price=float(getattr(row, "mark_price", 0.0) or 0.0),
            )
        )
    return trades


def _tsrv_trades_for_window(
    binance_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Trade]:
    """Reconstruct ``Bar.tsrv_trades`` from Binance USD-M aggTrades.

    The Binance Vision schema uses ``transact_time`` (ms int), ``price``,
    ``quantity`` and ``is_buyer_maker``. ``direction`` is ``+1`` when
    ``is_buyer_maker`` is False (the aggressor bought), ``-1`` otherwise
    — matching :meth:`TradeBuffer.prewarm_from_parquet`.
    """
    if binance_df.empty:
        return []
    ts = binance_df["binance_ts"]
    mask = (ts >= start) & (ts < end)
    win = binance_df.loc[mask]
    trades: list[Trade] = []
    for row in win.itertuples(index=False):
        is_maker = bool(getattr(row, "is_buyer_maker", False))
        trades.append(
            Trade(
                trade_id=f"binance:{int(getattr(row, 'agg_trade_id', 0))}",
                timestamp=getattr(row, "binance_ts"),
                instrument_name="BTCUSDT",
                price=float(getattr(row, "price", 0.0) or 0.0),
                amount=float(getattr(row, "quantity", 0.0) or 0.0),
                direction=-1 if is_maker else 1,
            )
        )
    return trades


def _canonical_index_at(svi_df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """Return the canonical index price at bar-close ``ts``.

    ``BTC_vol_arb_svi.parquet`` is indexed by 15-min bar-close
    timestamp; the engine's ``Bar.index_price`` is the close-of-bar
    index reference. An exact lookup matches the offline backtester's
    per-bar index; a missing row falls back to the latest at-or-before.
    """
    if svi_df.empty or "index_price" not in svi_df.columns:
        return None
    if ts in svi_df.index:
        return float(svi_df.at[ts, "index_price"])
    prior = svi_df.loc[svi_df.index <= ts]
    if prior.empty:
        return None
    return float(prior["index_price"].iloc[-1])


# ── Offline strike-picker (deterministic, network-free) ─────────────


def _install_offline_strike_picker(engine, option_df: pd.DataFrame) -> int:
    """Pre-populate the engine's :class:`StrikePicker` instrument cache.

    The production ``StrikePicker.pick_atm_straddle`` queries Deribit
    REST for the live active-instrument list. In a historical replay
    the live REST list does not reflect the expiries that were active
    at bar T, and a network call makes the harness non-deterministic.

    This harness-local override builds the instrument list from the
    DISTINCT option instruments present in the canonical option-trade
    parquet for the whole window and stuffs it into
    ``StrikePicker._instruments_cache`` with a far-future
    ``_instruments_cached_at`` so the picker never refreshes from REST.

    This mutates only the harness's own engine instance — it is NOT a
    production code change. Returns the instrument count installed.
    """
    picker = engine.strike_picker
    if picker is None:
        logger.warning("strike_picker is None — cannot install offline cache")
        return 0
    if option_df.empty or "instrument_name" not in option_df.columns:
        logger.warning("no option instruments to install in strike picker")
        return 0
    instruments: list[dict] = []
    for name in sorted(option_df["instrument_name"].astype(str).unique()):
        if parse_option_instrument(name) is None:
            continue
        instruments.append({"instrument_name": name})
    picker._instruments_cache = instruments
    # Far-future cache stamp so ``_get_instruments`` never deems it stale.
    picker._instruments_cached_at = pd.Timestamp("2099-01-01", tz="UTC")
    logger.info(
        "Installed offline strike-picker cache: %d option instruments",
        len(instruments),
    )
    return len(instruments)


# ── Per-bar replay driver ───────────────────────────────────────────


@dataclass
class ReplaySources:
    """Canonical data loaded once and reused across every bar in the window."""
    svi_df: pd.DataFrame                       # 15-min bar-close indexed canon
    option_by_day: dict[str, pd.DataFrame]     # date-str -> Deribit option trades
    future_by_day: dict[str, pd.DataFrame]     # date-str -> Deribit future trades
    binance_df: pd.DataFrame                   # Binance aggTrades for the window's months
    all_option_df: pd.DataFrame                # concat of every option day (strike picker)


def _load_sources(bars: list[pd.Timestamp]) -> ReplaySources:
    """Load every canonical parquet the replay window touches, once."""
    svi_df = pd.read_parquet(_REPO_ROOT / CANON_SVI_PARQUET)
    if svi_df.index.tz is None:
        svi_df.index = svi_df.index.tz_localize("UTC")
    svi_df = svi_df.sort_index()

    # Deribit trades: one parquet per UTC day. A bar at T covers
    # [T-15min, T) so the prior day's tail can matter for the 00:15 bar.
    days: set[pd.Timestamp] = set()
    for t in bars:
        days.add(t.normalize())
        days.add((t - pd.Timedelta(minutes=BAR_MINUTES)).normalize())
    option_by_day: dict[str, pd.DataFrame] = {}
    future_by_day: dict[str, pd.DataFrame] = {}
    for day in sorted(days):
        key = day.strftime("%Y-%m-%d")
        option_by_day[key] = _load_deribit_day(DERIBIT_OPTION_DIR, day)
        future_by_day[key] = _load_deribit_day(DERIBIT_FUTURE_DIR, day)

    # Binance aggTrades: one parquet per UTC month.
    months: set[str] = set()
    for t in bars:
        months.add(t.strftime("%Y-%m"))
        months.add((t - pd.Timedelta(minutes=BAR_MINUTES)).strftime("%Y-%m"))
    binance_frames: list[pd.DataFrame] = []
    for month in sorted(months):
        path = _REPO_ROOT / BINANCE_AGGTRADES_DIR / f"{month}.parquet"
        if not path.exists():
            logger.warning("Binance aggTrades parquet missing: %s", path)
            continue
        # Only the columns the harness needs — the monthly file is huge.
        cols = ["agg_trade_id", "price", "quantity", "transact_time",
                "is_buyer_maker"]
        bdf = pd.read_parquet(path, columns=cols)
        binance_frames.append(bdf)
    if binance_frames:
        binance_df = pd.concat(binance_frames, ignore_index=True)
        binance_df["binance_ts"] = pd.to_datetime(
            binance_df["transact_time"], unit="ms", utc=True,
        )
        # Keep only the rows the replay window can touch — bounds the
        # downstream per-bar slicing cost.
        lo = (bars[0] - pd.Timedelta(minutes=BAR_MINUTES))
        hi = bars[-1]
        binance_df = binance_df.loc[
            (binance_df["binance_ts"] >= lo) & (binance_df["binance_ts"] < hi)
        ].sort_values("binance_ts").reset_index(drop=True)
    else:
        binance_df = pd.DataFrame(columns=["agg_trade_id", "price", "quantity",
                                           "transact_time", "is_buyer_maker", "binance_ts"])

    option_nonempty = [d for d in option_by_day.values() if not d.empty]
    all_option_df = (
        pd.concat(option_nonempty, ignore_index=True)
        if option_nonempty else pd.DataFrame()
    )
    return ReplaySources(
        svi_df=svi_df,
        option_by_day=option_by_day,
        future_by_day=future_by_day,
        binance_df=binance_df,
        all_option_df=all_option_df,
    )


def reconstruct_bar(sources: ReplaySources, bar_close: pd.Timestamp) -> Bar:
    """Reconstruct the canonical :class:`Bar` for bar-close ``bar_close``.

    The bar covers ``[bar_close - 15min, bar_close)``:

      * ``perp_trades``  — Deribit ``BTC-PERPETUAL`` trades in the window.
      * ``option_trades``— Deribit option trades in the window.
      * ``tsrv_trades``  — Binance USD-M aggTrades in the window.
      * ``index_price``  — the canonical close-of-bar index from
                           ``BTC_vol_arb_svi.parquet``.

    Trades may straddle a UTC-day boundary (the 00:15 bar reads the
    prior day's 00:00-00:15 tail), so both day parquets are consulted.
    """
    window_start = bar_close - pd.Timedelta(minutes=BAR_MINUTES)
    day_keys = {window_start.strftime("%Y-%m-%d"),
                bar_close.strftime("%Y-%m-%d")}

    perp_trades: list[Trade] = []
    option_trades: list[Trade] = []
    for key in sorted(day_keys):
        fut = sources.future_by_day.get(key, pd.DataFrame())
        opt = sources.option_by_day.get(key, pd.DataFrame())
        perp_trades.extend(
            _perp_trades_for_window(fut, window_start, bar_close)
        )
        option_trades.extend(
            _option_trades_for_window(opt, window_start, bar_close)
        )
    perp_trades.sort(key=lambda t: t.timestamp)
    option_trades.sort(key=lambda t: t.timestamp)

    tsrv_trades = _tsrv_trades_for_window(
        sources.binance_df, window_start, bar_close,
    )

    index_price = _canonical_index_at(sources.svi_df, bar_close)

    return Bar(
        timestamp=bar_close,
        bar_minutes=BAR_MINUTES,
        perp_trades=perp_trades,
        option_trades=option_trades,
        index_price=index_price,
        tsrv_trades=tsrv_trades,
    )


def _prewarm_buffers_as_of(engine, sources: ReplaySources, as_of: pd.Timestamp) -> None:
    """Populate the engine's option buffer with all option trades ``< as_of``.

    The SVI calibrator recalibrates from ``option_buffer.to_dataframe()``
    inside ``process_bar`` (step 3). For that recalibration to fit the
    surface ``as_of`` T, the option buffer must hold the trailing
    window of option trades before T. The TSRV-history, dollar-bar and
    TransVol buffers are prewarmed ONCE at ``initialize()`` from their
    canonical parquets — re-prewarming them here would double-append.
    Instead the per-bar ``process_bar`` call advances them naturally
    (``tsrv_history_buffer.append_bar``, TransVol ``append``) so they
    track bar-close T exactly.

    This is a no-op when the option buffer is None (defensive).
    """
    if engine.option_buffer is None:
        return
    # The option buffer's own ``window_seconds`` eviction keeps it
    # bounded; we feed the full trailing window and let the buffer trim.
    horizon = pd.Timedelta(seconds=engine.option_buffer.window_seconds)
    lo = as_of - horizon
    if sources.all_option_df.empty:
        return
    df = sources.all_option_df
    mask = (df["timestamp"] >= lo) & (df["timestamp"] < as_of)
    win = df.loc[mask]
    added = 0
    for row in win.itertuples(index=False):
        name = str(getattr(row, "instrument_name", ""))
        parsed = parse_option_instrument(name)
        if parsed is None:
            continue
        expiry_dt, strike, option_type = parsed
        iv = getattr(row, "iv", None)
        trade = Trade(
            trade_id=str(getattr(row, "trade_id", "")),
            timestamp=getattr(row, "timestamp"),
            instrument_name=name,
            price=float(getattr(row, "price", 0.0) or 0.0),
            amount=float(getattr(row, "amount", 0.0) or 0.0),
            direction=_direction_from_str(getattr(row, "direction", "buy")),
            strike=strike,
            option_type=option_type,
            expiry_dt=expiry_dt,
            iv=float(iv) if iv is not None and pd.notna(iv) else None,
            index_price=float(getattr(row, "index_price", 0.0) or 0.0),
            mark_price=float(getattr(row, "mark_price", 0.0) or 0.0),
        )
        if engine.option_buffer.append(trade):
            added += 1
    logger.debug("Option buffer prewarmed +%d trades as_of %s", added, as_of)


async def run_replay(
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_path: Path,
    *,
    initial_equity_usd: float = 100_000.0,
) -> int:
    """Drive the live engine's ``process_bar`` over ``[start, end]``.

    Returns the number of bars successfully driven through the engine.
    Raises on a fatal setup error (missing canonical data, engine
    ``initialize`` failure).
    """
    bars = _bar_grid(start, end)
    if not bars:
        logger.error("empty bar grid for window [%s, %s]", start, end)
        return 0
    logger.info(
        "Replay window: %d bars from %s to %s",
        len(bars), bars[0], bars[-1],
    )

    logger.info("Loading canonical data sources ...")
    sources = _load_sources(bars)

    # ── Build the engine in paper + dry-run mode ──
    cfg = PaperConfig()
    cfg.switch_to_prod(paper_mode=True)   # mainnet URLs, paper (unauth) mode
    cfg.dry_run = True
    cfg.paper_initial_equity_usd = initial_equity_usd
    cfg.bar_minutes = BAR_MINUTES
    # Point the engine's BarTraceWriter at the harness output JSONL so
    # the existing per-bar trace instrumentation captures the replay.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.feature_trace_path = str(output_path)
    # A fresh run should not append onto a stale trace.
    if output_path.exists():
        output_path.unlink()

    from deribit_paper.engine import TradingEngine

    engine = TradingEngine(cfg)
    logger.info("Initialising engine (paper mode, unauthenticated) ...")
    await engine.initialize()
    if engine.bar_trace_writer is None or not engine.bar_trace_writer.enabled:
        raise RuntimeError(
            "BarTraceWriter did not enable — check cfg.feature_trace_path"
        )

    # Deterministic, network-free ATM straddle selection.
    _install_offline_strike_picker(engine, sources.all_option_df)

    driven = 0
    for bar_close in bars:
        # 1. Recalibrate SVI as_of T from canonical option trades so the
        #    IV chain (C3 ``iv_used``) reflects T's surface. We feed the
        #    option buffer here; ``process_bar`` step 3's bucket-boundary
        #    ``needs_refresh`` then fits the just-closed 15-min bucket.
        _prewarm_buffers_as_of(engine, sources, bar_close)

        # 2. Reconstruct the canonical Bar for T.
        bar = reconstruct_bar(sources, bar_close)
        if bar.index_price is None:
            logger.warning(
                "bar %s has no canonical index price — skipping", bar_close,
            )
            continue

        # 3. Drive the engine. ``process_bar`` runs TSRV -> SVI recal ->
        #    TransVol -> predictor -> signal and emits the BarTrace via
        #    the engine's own BarTraceWriter.
        try:
            await engine.process_bar(bar)
            driven += 1
        except Exception:  # noqa: BLE001
            logger.exception("process_bar failed at %s — continuing", bar_close)

    logger.info(
        "Replay complete: %d/%d bars driven; trace written to %s",
        driven, len(bars), output_path,
    )
    return driven


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Replay-parity harness — drive the live engine's "
                    "process_bar with canonical offline data.",
    )
    p.add_argument(
        "--start", required=True,
        help="Replay window start (ISO 8601, e.g. '2026-05-14 00:15:00+00:00')",
    )
    p.add_argument(
        "--end", required=True,
        help="Replay window end (inclusive, ISO 8601)",
    )
    p.add_argument(
        "--output", default="deribit_paper/logs/bar_trace_replay.jsonl",
        help="Path for the replay-trace JSONL (default: "
             "deribit_paper/logs/bar_trace_replay.jsonl)",
    )
    p.add_argument(
        "--offline-trace", default="vol_arb/output/bar_trace_offline.jsonl",
        help="Offline trace to suggest for the feature_trace_diff step",
    )
    p.add_argument(
        "--initial-equity", type=float, default=100_000.0,
        help="Synthetic paper-mode starting equity in USD",
    )
    args = p.parse_args()

    start = _to_utc(args.start)
    end = _to_utc(args.end)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    driven = asyncio.run(
        run_replay(
            start, end, output_path,
            initial_equity_usd=args.initial_equity,
        )
    )
    if driven == 0:
        logger.error("no bars driven — replay produced an empty trace")
        return 1

    # Print the diff command for the operator (do not auto-run: the
    # offline trace must be built first by build_offline_bar_trace.py).
    print("")
    print("=" * 70)
    print("  REPLAY TRACE READY")
    print("=" * 70)
    print(f"  replay trace : {output_path}")
    print(f"  bars driven  : {driven}")
    print("")
    print("  Diff against the offline trace with:")
    print("")
    print(f"    python scripts/feature_trace_diff.py \\")
    print(f"        --live {output_path} \\")
    print(f"        --offline {args.offline_trace}")
    print("")
    print("  (build the offline trace first via "
          "scripts/build_offline_bar_trace.py if it does not exist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
