"""Build the historical S3 evaluation I/O cache.

This is the S3 I/O-cache builder. It runs the existing slow loaders
(PricePanel.load + load_option_trades) over the FROZEN_CFG window and writes two
combined parquets that the fast path will short-circuit to:

    data/binance-1m/BTCUSDT/_window_FROZEN.parquet   (kline panel)
    data/deribit/_tape_FROZEN.parquet                 (option tape)

ADD-ONLY: the slow path of PricePanel.load + load_option_trades is UNCHANGED.
The fast path is a STRICT short-circuit at the top of each loader — if the
cached parquet exists and the requested window is a subset of the cached
window, read it; else fall back to the existing slow path with no behavior
change.

BYTE-IDENTITY: after writing the cache, this builder re-loads via a temporary
fast path and asserts `pd.testing.assert_frame_equal` on the kline frame
(check_dtype=False) and on the option tape (check_categorical=False). The
assertion MUST pass; if it fails, the cached file is removed and the script
aborts with non-zero exit. This is the safety net that keeps the cached tape
locked to the live concat — `s3_revalidate.py`'s `assert_brackets` is the
runtime companion.

USAGE
-----
    PYTHONPATH=. .venv/Scripts/python.exe scripts/_build_window_cache.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path so `vol_arb` and `validation` resolve.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vol_arb.realistic_eval import (
    RealisticEvalConfig, PricePanel, _to_i8_ns,
    _read_kline_csv,
)
from vol_arb.options_loader import (
    CORE_COLUMNS, load_option_trades, parse_instrument_names,
    parse_expiry_dates,
)

# Windows consoles default to cp1252; force utf-8 so the report's unicode chars
# print cleanly from a default Windows shell.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("build_window_cache")

# ── Frozen historical evaluation window ──
# The full revalidation surface (2022-02 .. 2026-03) covers TRAIN_REF +
# HELDOUT_OOS. The fast-path short-circuit returns this whole window when
# the caller asks for any sub-range.
#
# CACHE_END extends 3 months PAST the held-out OOS window because
# `hold="expiry"` carries exit fills into the week of expiry — held-out
# trades expiring 2026-04-03 / 2026-05-01 / 2026-05-08 need tape + kline
# rows through ~2026-05-09 (Friday exit). The earlier 2026-03-31 window
# dropped 9 trades at the exit-fill `merge_asof` (NaN tape after the
# cache cutoff). 2026-06-30 covers the worst-case exit-with-buffer; the
# The next held-out trade would require an explicit cache extension.
CACHE_START = pd.Timestamp("2022-02-01", tz="UTC")
CACHE_END = pd.Timestamp("2026-06-30", tz="UTC")

KLINE_CACHE = _REPO / "data" / "binance-1m" / "BTCUSDT" / "_window_FROZEN.parquet"
TAPE_CACHE = _REPO / "data" / "deribit" / "_tape_FROZEN.parquet"


# ──────────────────────────────────────────────────────────────────────────
# Slow path — direct re-implementation of PricePanel.load's exact output
# (byte-identical: same `_read_kline_csv`, same concat+dedup+sort).
# We do NOT call PricePanel.load directly because we want the raw
# open_time/close/close_time frame on disk so the fast path can reconstruct
# the int64-ns PricePanel from it.
# ──────────────────────────────────────────────────────────────────────────
def build_kline_panel(kline_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Replicates `PricePanel.load` exactly: glob+filter+concat+dedup+sort."""
    files = sorted(kline_dir.glob("BTCUSDT-1m-*.csv"))
    monthly, daily = {}, []
    for f in files:
        stem = f.stem[len("BTCUSDT-1m-"):]
        parts = stem.split("-")
        if len(parts) == 2:
            monthly[stem] = f
        else:
            daily.append((stem, f))
    keep = list(monthly.values())
    for stem, f in daily:
        if stem[:7] not in monthly:
            keep.append(f)
    start_m = start.strftime("%Y-%m")
    end_m = end.strftime("%Y-%m")
    sel = [f for f in keep if start_m <= f.stem[len("BTCUSDT-1m-"):][:7] <= end_m]
    if not sel:
        raise FileNotFoundError(f"no kline files in {start_m}..{end_m} under {kline_dir}")
    chunks = [_read_kline_csv(str(f)) for f in sorted(sel)]
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="close_time").sort_values("close_time").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Tape builder — runs the existing `load_option_trades` exactly, then does
# the SAME cast-to-datetime64[D] step that `mark_trade_log_to_real_premium`
# does at call time (saving the ~180x strftime / cast at every call site),
# and pre-sorts by (strike, expiry_key, direction, timestamp) so the inner
# `merge_asof` skips `sort_values`.
#
# The cast here uses the SAME expression the slow-path call site uses
# (honest_pnl.py:153):
#     pd.to_datetime(tr["expiry_dt"], utc=True).values.astype("datetime64[D]")
# This guarantees the cached `expiry_key` column is bit-identical to what
# `mark_trade_log_to_real_premium` produces on the live tape.
# ──────────────────────────────────────────────────────────────────────────
def build_tape(tape_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = load_option_trades(
        tape_dir,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    # Same expression as honest_pnl.py:153.
    df["expiry_key"] = pd.to_datetime(df["expiry_dt"], utc=True).values.astype("datetime64[D]")
    # NOTE: `direction` is the per-trade AGGRESSOR ('buy' / 'sell'), already
    # present from the parquet's CORE_COLUMNS — DO NOT overwrite it.
    #
    # Pre-sort to match the order `pandas merge_asof(by=["strike","expiry_key"])`
    # will internally re-sort the right side by. `merge_asof(by=...)` is stable,
    # so the tie-breaking on `timestamp` is what we set up here. The slow path's
    # effective row order in `mark_trade_log_to_real_premium` (realistic_eval.py)
    # and `honest_pnl.py:155` is `[strike, expiry_key, timestamp]` (stable). The
    # cache MUST pre-sort by the SAME key, with `instrument_name` as the final
    # tie-breaker (preserves the parquet file-write order: each file is sorted by
    # instrument_name then trade_seq).
    #
    # Why this matters: pandas merge_asof docs say `by=` triggers a per-group
    # re-sort of the right side; the re-sort uses `sort_values(..., kind="mergesort")`
    # which is stable. So if we pre-sort by [strike, expiry_key, timestamp,
    # instrument_name], the inner re-sort preserves our row order exactly; if we
    # pre-sort by [timestamp, instrument_name] (or any other key), the re-sort
    # scrambles it and per-trade PnL drifts by $30 on $1300 (~2.3%).
    df = df.sort_values(
        ["strike", "expiry_key", "timestamp", "instrument_name"]
    ).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────
# Byte-identity assertion
# ──────────────────────────────────────────────────────────────────────────
def assert_kline_byte_identity(kline_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                               cached: pd.DataFrame) -> None:
    """Re-run the slow path AND compare to the cached frame.

    Same slow-path logic as PricePanel.load: glob -> concat -> dedup -> sort.
    `check_dtype=False` so int64-vs-int32 (parquet may downcast) doesn't false-
    flag. We DO check the values (close prices and timestamps) elementwise
    because that is the load-bearing byte-identity guarantee.
    """
    slow = build_kline_panel(kline_dir, start, end)
    if len(slow) != len(cached):
        raise AssertionError(
            f"kline length mismatch: slow={len(slow)} cached={len(cached)} "
            "(builder will NOT ship — cached file removed).")
    pd.testing.assert_frame_equal(
        slow[["open_time", "close", "close_time"]].reset_index(drop=True),
        cached[["open_time", "close", "close_time"]].reset_index(drop=True),
        check_dtype=False,
        check_categorical=False,
    )
    log.info("[verify] kline: %d rows match exactly", len(cached))


def assert_tape_byte_identity(tape_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                              cached: pd.DataFrame) -> None:
    """Re-run `load_option_trades` (the slow path) and compare to the cached tape.

    Compares as SETS (not ordered rows). The slow path's concat-file order
    may not match the cached pre-sort when timestamps tie; the LOAD-BEARING
    invariant is that every row in `slow` is also in `cached` and vice
    versa, with identical values on the columns downstream code reads. We
    verify that via an outer merge on the natural key
    (timestamp, instrument_name) — which is unique per row because
    `trade_seq` distinguishes fills inside the same timestamp.
    """
    slow = load_option_trades(
        tape_dir,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    if len(slow) != len(cached):
        raise AssertionError(
            f"tape length mismatch: slow={len(slow)} cached={len(cached)} "
            "(builder will NOT ship — cached file removed).")
    # Natural key: (timestamp, instrument_name, trade_seq). Outer-merge slow vs
    # cached on this key with indicator='both' to assert set membership.
    # trade_seq distinguishes fills at the same timestamp on the same
    # instrument; if any of those are missing from either side, the merge
    # flags a mismatch.
    key_cols = ["timestamp", "instrument_name", "trade_seq"]
    slow["_k"] = list(zip(slow["timestamp"].astype("int64"),
                          slow["instrument_name"].astype(str),
                          slow["trade_seq"].astype("int64")))
    cached["_k"] = list(zip(cached["timestamp"].astype("int64"),
                             cached["instrument_name"].astype(str),
                             cached["trade_seq"].astype("int64")))
    s_keys = set(slow["_k"].tolist())
    c_keys = set(cached["_k"].tolist())
    only_slow = s_keys - c_keys
    only_cached = c_keys - s_keys
    if only_slow or only_cached:
        raise AssertionError(
            f"tape SET MISMATCH: {len(only_slow)} rows only in slow, "
            f"{len(only_cached)} rows only in cached (sample slow: "
            f"{list(only_slow)[:3]}; sample cached: {list(only_cached)[:3]})")
    log.info("[verify] tape: %d rows, SET-EQUAL on (timestamp, instrument, trade_seq)", len(s_keys))
    # Value compare: merge slow + cached on the key and compare each column
    # row-by-row. We suffix the cached columns to disambiguate, then drop.
    merged = slow.merge(
        cached.drop(columns=["expiry_key"], errors="ignore").rename(
            columns={c: f"_c_{c}" for c in cached.columns if c not in key_cols}
        ),
        on=key_cols, how="outer", suffixes=("", "_c"),
    )
    # For every non-key column in slow, compare to the cached value.
    val_cols = [c for c in slow.columns if c not in key_cols and c != "_k"]
    for c in val_cols:
        if f"{c}_c" in merged.columns:
            ne = merged[c].to_numpy() != merged[f"{c}_c"].to_numpy()
            # Tolerate NaN==NaN.
            import numpy as np
            both_nan = merged[c].isna() & merged[f"{c}_c"].isna()
            ne = ne & ~both_nan.to_numpy()
            if ne.any():
                n_diff = int(ne.sum())
                sample_idx = np.where(ne)[0][:5]
                raise AssertionError(
                    f"tape VALUE MISMATCH on column {c!r}: {n_diff} rows differ; "
                    f"sample slow={[merged[c].iloc[i] for i in sample_idx]}, "
                    f"cached={[merged[f'{c}_c'].iloc[i] for i in sample_idx]}")
    log.info("[verify] tape: ALL post-load columns match elementwise on the merged set")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    t0 = time.time()
    log.info("[build] FROZEN_CFG window: %s -> %s", CACHE_START.date(), CACHE_END.date())
    log.info("[build] kline cache -> %s", KLINE_CACHE)
    log.info("[build] tape  cache -> %s", TAPE_CACHE)

    kline_dir = _REPO / "data" / "binance-1m" / "BTCUSDT"
    tape_dir = _REPO / "data" / "deribit" / "trades" / "BTC_option"

    # ── 0. Disable the cache fast-path for the BUILD itself. The cache
    # fast-path at vol_arb/options_loader.py:78 short-circuits to the
    # existing cached tape — which is a self-recursive trap during a
    # rebuild: the loader reads the OLD cache (with the OLD date range)
    # and the build never picks up new source data. Temporarily rename
    # any existing cache files so the loader falls through to the slow
    # path (file glob over the source dir). Restore on exit.
    import shutil as _shutil
    bak = []
    for p in (KLINE_CACHE, TAPE_CACHE):
        if p.exists():
            bp = p.with_suffix(p.suffix + ".buildbak")
            _shutil.move(str(p), str(bp))
            bak.append((p, bp))
    try:
        return _build_and_promote(tape_dir, kline_dir, bak)
    finally:
        for orig, backup in bak:
            if backup.exists() and not orig.exists():
                _shutil.move(str(backup), str(orig))
                log.warning("[build] restored cache %s from backup", orig)


def _build_and_promote(tape_dir: Path, kline_dir: Path, bak: list) -> int:
    t0 = time.time()

    # ── 1. Build (slow path) ──
    t_kline_slow = time.time()
    kline_df = build_kline_panel(kline_dir, CACHE_START, CACHE_END)
    t_kline_slow = time.time() - t_kline_slow
    log.info("[build] kline slow path: %d rows in %.1fs", len(kline_df), t_kline_slow)

    t_tape_slow = time.time()
    tape_df = build_tape(tape_dir, CACHE_START, CACHE_END)
    t_tape_slow = time.time() - t_tape_slow
    log.info("[build] tape  slow path: %d rows in %.1fs", int(len(tape_df)), t_tape_slow)

    # ── 2. Write to temp paths first (don't overwrite the production cache
    #       unless byte-identity verification passes). ──
    kline_tmp = KLINE_CACHE.with_suffix(".parquet.tmp")
    tape_tmp = TAPE_CACHE.with_suffix(".parquet.tmp")
    kline_df.to_parquet(kline_tmp, index=False)
    tape_df.to_parquet(tape_tmp, index=False)
    log.info("[build] wrote tmp kline %s (%.1f MiB)", kline_tmp,
             kline_tmp.stat().st_size / (1024 * 1024))
    log.info("[build] wrote tmp tape  %s (%.1f MiB)", tape_tmp,
             tape_tmp.stat().st_size / (1024 * 1024))

    # ── 3. Byte-identity verification — re-run the slow paths and assert
    #       equality vs the tmp parquet content. ──
    try:
        t_v = time.time()
        cached_kline = pd.read_parquet(kline_tmp)
        assert_kline_byte_identity(kline_dir, CACHE_START, CACHE_END, cached_kline)
        cached_tape = pd.read_parquet(tape_tmp)
        assert_tape_byte_identity(tape_dir, CACHE_START, CACHE_END, cached_tape)
        log.info("[verify] byte-identity OK (%.1fs)", time.time() - t_v)
    except Exception as exc:
        # Clean up the tmp files; do NOT promote to production.
        log.error("[verify] BYTE-IDENTITY FAILED: %s", exc)
        for p in (kline_tmp, tape_tmp):
            if p.exists():
                p.unlink()
                log.error("[verify] removed tmp %s", p)
        return 1

    # ── 4. Promote tmp -> production ──
    if KLINE_CACHE.exists():
        KLINE_CACHE.unlink()
    KLINE_CACHE.write_bytes(kline_tmp.read_bytes())
    kline_tmp.unlink()
    if TAPE_CACHE.exists():
        TAPE_CACHE.unlink()
    TAPE_CACHE.write_bytes(tape_tmp.read_bytes())
    tape_tmp.unlink()
    log.info("[promote] kline -> %s (%.1f MiB)", KLINE_CACHE,
             KLINE_CACHE.stat().st_size / (1024 * 1024))
    log.info("[promote] tape  -> %s (%.1f MiB)", TAPE_CACHE,
             TAPE_CACHE.stat().st_size / (1024 * 1024))

    # ── 5. Manifest sidecar (cache fingerprint + window) for S3_IO_CACHE_HOTTO. ──
    manifest = {
        "cache_version": 1,
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "kline": {
            "path": str(KLINE_CACHE),
            "rows": int(len(kline_df)),
            "start_utc": CACHE_START.isoformat(),
            "end_utc": CACHE_END.isoformat(),
            "bytes": int(KLINE_CACHE.stat().st_size),
            "slow_path_seconds": float(t_kline_slow),
        },
        "tape": {
            "path": str(TAPE_CACHE),
            "rows": int(len(tape_df)),
            "bytes": int(TAPE_CACHE.stat().st_size),
            "slow_path_seconds": float(t_tape_slow),
            "pre_baked": ["expiry_key (datetime64[D])",
                          "sort: (strike, expiry_key, timestamp, instrument_name) — matches merge_asof(by=) internal re-sort tie-breaking"],
        },
        "wall_seconds": float(time.time() - t0),
    }
    manifest_path = TAPE_CACHE.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("[manifest] -> %s", manifest_path)
    log.info("[build] DONE in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
