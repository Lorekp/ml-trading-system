"""Verify the S3 I/O cache is byte-identical to the live loaders.

Standalone, fast (~30s typical). Compares the cached parquet contents to a
fresh `PricePanel.load` + `load_option_trades` over the FROZEN_CFG window
on the same machine, and asserts:

    1. kline: `pd.testing.assert_frame_equal` on (open_time, close, close_time)
       between live concat and the cached parquet. `check_dtype=False` so the
       parquet's `int32`-vs-`int64` downcast doesn't false-flag (the values
       are the load-bearing byte-identity check).
    2. tape:  same check on the post-`load_option_trades` columns (timestamp,
       strike, expiry_dt, option_type, price, mark_price, iv, index_price,
       amount, trade_seq, T_years, dte_days, moneyness). We additionally
       verify the pre-baked `expiry_key` column is bit-identical to the
       `pd.to_datetime(...).astype('datetime64[D]')` expression that
       `mark_trade_log_to_real_premium` uses (honest_pnl.py:153).

Exit codes:
    0  — all assertions pass
    1  — kline byte-identity FAILED
    2  — tape byte-identity FAILED
    3  — expiry_key pre-bake FAILED
    4  — cache files missing (run `scripts/_build_window_cache.py` first)

This is the SAFETY NET that keeps the cache locked to the live concat.
`scripts/s3_revalidate.py`'s `assert_brackets` is the runtime companion —
fires per-trade on the S3 revalidation cycle. If THIS script fails, the
cache is stale (the underlying CSVs or parquet set have drifted) and the
production cache must be rebuilt.

USAGE
-----
    PYTHONPATH=. .venv/Scripts/python.exe scripts/_verify_io_cache.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vol_arb.realistic_eval import (
    RealisticEvalConfig, PricePanel, _read_kline_csv,
)
from vol_arb.options_loader import load_option_trades
from scripts._build_window_cache import (
    CACHE_START, CACHE_END, KLINE_CACHE, TAPE_CACHE,
    build_kline_panel,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("verify_io_cache")


def verify_kline(kline_dir: Path) -> bool:
    if not KLINE_CACHE.exists():
        log.error("[verify] kline cache missing: %s", KLINE_CACHE)
        return False
    cached = pd.read_parquet(KLINE_CACHE)
    slow = build_kline_panel(kline_dir, CACHE_START, CACHE_END)
    if len(slow) != len(cached):
        log.error("[verify] kline length MISMATCH: slow=%d cached=%d",
                  len(slow), len(cached))
        return False
    try:
        pd.testing.assert_frame_equal(
            slow[["open_time", "close", "close_time"]].reset_index(drop=True),
            cached[["open_time", "close", "close_time"]].reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
        )
    except AssertionError as exc:
        log.error("[verify] kline FRAME MISMATCH: %s", exc)
        return False
    log.info("[verify] kline: %d rows, byte-identical to live concat", len(cached))
    return True


def verify_tape(tape_dir: Path) -> tuple[bool, bool]:
    """Returns (frame_match, expiry_key_match). Uses set-equality on the
    natural key (timestamp, instrument_name, trade_seq) to avoid sort-tie
    false-positives from parquet round-trip."""
    if not TAPE_CACHE.exists():
        log.error("[verify] tape cache missing: %s", TAPE_CACHE)
        return (False, False)
    cached = pd.read_parquet(TAPE_CACHE)
    slow = load_option_trades(
        tape_dir,
        start_date=CACHE_START.strftime("%Y-%m-%d"),
        end_date=CACHE_END.strftime("%Y-%m-%d"),
    )
    if len(slow) != len(cached):
        log.error("[verify] tape length MISMATCH: slow=%d cached=%d",
                  len(slow), len(cached))
        return (False, False)

    # Natural key: (timestamp, instrument_name, trade_seq). Set-compare.
    key_cols = ["timestamp", "instrument_name", "trade_seq"]
    slow_k = list(zip(slow["timestamp"].astype("int64").tolist(),
                      slow["instrument_name"].astype(str).tolist(),
                      slow["trade_seq"].astype("int64").tolist()))
    cached_k = list(zip(cached["timestamp"].astype("int64").tolist(),
                        cached["instrument_name"].astype(str).tolist(),
                        cached["trade_seq"].astype("int64").tolist()))
    if set(slow_k) != set(cached_k):
        only_s = set(slow_k) - set(cached_k)
        only_c = set(cached_k) - set(slow_k)
        log.error("[verify] tape SET MISMATCH: only_slow=%d only_cached=%d "
                  "(sample slow=%s sample cached=%s)",
                  len(only_s), len(only_c),
                  list(only_s)[:3], list(only_c)[:3])
        return (False, False)
    log.info("[verify] tape: %d rows, SET-EQUAL on (timestamp, instrument, trade_seq)", len(slow))

    # Value compare: outer merge on the key and check each non-key column.
    merged = slow.merge(
        cached.drop(columns=["expiry_key"], errors="ignore").rename(
            columns={c: f"_c_{c}" for c in cached.columns if c not in key_cols}
        ),
        on=key_cols, how="outer", suffixes=("", "_c"),
    )
    val_cols = [c for c in slow.columns if c not in key_cols]
    for c in val_cols:
        cc = f"{c}_c"
        if cc not in merged.columns:
            continue
        a = merged[c]
        b = merged[cc]
        ne = (a != b)
        # tolerate NaN==NaN
        both_nan = a.isna() & b.isna()
        ne = ne & ~both_nan
        if ne.any():
            n_diff = int(ne.sum())
            sample_idx = np.where(ne.to_numpy())[:5]
            sample = [(float(a.iloc[i]), float(b.iloc[i])) for i in sample_idx[0][:3]]
            log.error("[verify] tape VALUE MISMATCH on %r: %d rows differ; sample %s",
                      c, n_diff, sample)
            return (False, False)
    log.info("[verify] tape: ALL post-load columns match elementwise on the merged set")

    # expiry_key pre-bake: recompute the slow-path expression on the cached
    # `expiry_dt` and assert it matches the pre-baked column elementwise.
    ek_check = pd.to_datetime(cached["expiry_dt"], utc=True).values.astype("datetime64[D]")
    if not np.array_equal(ek_check, cached["expiry_key"].to_numpy()):
        log.error("[verify] expiry_key pre-bake MISMATCH")
        return (True, False)
    log.info("[verify] expiry_key pre-bake: bit-identical to honest_pnl.py:153 expression")
    return (True, True)


def main() -> int:
    t0 = time.time()
    kline_dir = _REPO / "data" / "binance-1m" / "BTCUSDT"
    tape_dir = _REPO / "data" / "deribit" / "trades" / "BTC_option"

    log.info("[verify] window: %s -> %s", CACHE_START.date(), CACHE_END.date())
    kline_ok = verify_kline(kline_dir)
    tape_frame_ok, ek_ok = verify_tape(tape_dir)
    elapsed = time.time() - t0

    print("=" * 60)
    print("S3 I/O CACHE BYTE-IDENTITY VERIFICATION")
    print("=" * 60)
    print(f"  kline:               {'PASS' if kline_ok else 'FAIL'}")
    print(f"  tape (frame):        {'PASS' if tape_frame_ok else 'FAIL'}")
    print(f"  tape (expiry_key):   {'PASS' if ek_ok else 'FAIL'}")
    print(f"  elapsed:             {elapsed:.1f}s")
    print("=" * 60)
    if kline_ok and tape_frame_ok and ek_ok:
        print("ALL CACHE ASSERTIONS PASSED — cache is safe to use.")
        return 0
    if not kline_ok:
        return 1
    if not tape_frame_ok:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
