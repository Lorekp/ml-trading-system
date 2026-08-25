#!/usr/bin/env python
"""Audit: why does current `build_dataset(VolModelConfig()).y` no longer byte-match
the saved `y_true` in `vol_model/output/BTCUSDT_pruned_svi_predictions.parquet`?

Writes per-row deltas to ``vol_model/output/label_shift_per_row.csv`` and prints
all diagnostic stats to stdout.

Usage:
    cd /path/to/ml-trading-system
    python -m vol_model.run_label_shift_audit
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("label_shift_audit")
logger.setLevel(logging.INFO)

SAVED_PATH = Path("vol_model/output/BTCUSDT_pruned_svi_predictions.parquet")
CSV_OUT = Path("vol_model/output/label_shift_per_row.csv")


def _print_block(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def diagnose(saved: pd.Series, current: pd.Series, label: str) -> pd.DataFrame:
    """Compute and print full diagnostics; return per-row delta DataFrame."""
    _print_block(label)
    n_saved = len(saved)
    n_current = len(current)
    common = saved.index.intersection(current.index)
    n_common = len(common)
    print(f"Row counts: saved={n_saved:,} current={n_current:,} common={n_common:,}")

    if n_common == 0:
        print("No common index — cannot diagnose shift.")
        return pd.DataFrame()

    s = saved.loc[common]
    c = current.loc[common]
    d = c - s  # current - saved

    print(f"\nsaved y_true  : mean={s.mean():+.4f}  median={s.median():+.4f}  std={s.std():.4f}  min={s.min():+.4f}  max={s.max():+.4f}")
    print(f"current y_full: mean={c.mean():+.4f}  median={c.median():+.4f}  std={c.std():.4f}  min={c.min():+.4f}  max={c.max():+.4f}")
    print(f"delta (c - s) : mean={d.mean():+.4f}  median={d.median():+.4f}  std={d.std():.4f}  min={d.min():+.4f}  max={d.max():+.4f}")

    exact_eq = (d.abs() < 1e-9).sum()
    print(f"\nExact matches (|delta| < 1e-9): {exact_eq:,}/{n_common:,} ({exact_eq/n_common:.4%})")

    # Rows where current is outside saved's min/max envelope
    s_min, s_max = s.min(), s.max()
    oob = (c < s_min) | (c > s_max)
    n_oob = int(oob.sum())
    print(f"Rows where current outside [saved_min, saved_max]: {n_oob:,} ({n_oob/n_common:.4%})")

    # Per-decile analysis (buckets on saved y_true)
    print("\nPer-decile shift (bucketed on saved y_true):")
    try:
        bucket = pd.qcut(s, q=10, labels=False, duplicates="drop")
        tbl = (
            pd.DataFrame({"saved": s, "current": c, "bucket": bucket})
            .groupby("bucket", observed=True)
            .agg(
                saved_mean=("saved", "mean"),
                current_mean=("current", "mean"),
                n=("saved", "size"),
            )
        )
        tbl["delta_mean"] = tbl["current_mean"] - tbl["saved_mean"]
        print(tbl.to_string(float_format=lambda x: f"{x:+.4f}"))
    except Exception as e:  # noqa: BLE001
        print(f"  decile analysis skipped: {e}")

    # Top 10 absolute deltas
    print("\nTop 10 largest |delta| deltas (index, saved, current, delta):")
    top = d.abs().nlargest(10)
    for ts in top.index:
        print(f"  {ts}  saved={s.loc[ts]:+.4f}  current={c.loc[ts]:+.4f}  delta={d.loc[ts]:+.4f}")

    # Lag hypothesis: does c[T] match s[T+15min] or s[T-15min]?
    # (current y shifted by one bar vs saved y)
    print("\nLag hypothesis — test if current[T] equals saved at a nearby timestamp:")
    for shift_min in (-15, +15, -30, +30):
        c_shifted_idx = c.index + pd.Timedelta(minutes=shift_min)
        s_on_shift = saved.reindex(c_shifted_idx)
        s_on_shift.index = c.index
        mask = s_on_shift.notna()
        if mask.sum() == 0:
            continue
        matches = (np.abs(c[mask] - s_on_shift[mask]) < 1e-9).sum()
        close = (np.abs(c[mask] - s_on_shift[mask]) < 1e-4).sum()
        total = int(mask.sum())
        print(f"  current[T] vs saved[T{shift_min:+d}min]: exact={matches:,}/{total:,}  close(<1e-4)={close:,}/{total:,}")

    return pd.DataFrame({"saved": s, "current": c, "delta": d})


def main() -> None:
    if not SAVED_PATH.exists():
        raise FileNotFoundError(f"Saved predictions not found at {SAVED_PATH}")

    print("=" * 70)
    print("  LABEL-SHIFT AUDIT — BTCUSDT pruned_svi predictions")
    print("=" * 70)
    saved = pd.read_parquet(SAVED_PATH)
    saved_y = saved["y_true"]
    print(f"Saved y_true: {len(saved_y):,} rows, "
          f"index [{saved_y.index.min()} .. {saved_y.index.max()}], "
          f"mean={saved_y.mean():+.4f}")

    # --- Pass 1: default config (whatever it is NOW) ---
    cfg_default = VolModelConfig()
    print(f"\nPass 1 config: asset={cfg_default.asset}  freq_s={cfg_default.freq_s}  "
          f"label_clip_q={cfg_default.label_clip_quantile}  "
          f"predict_excess_logvol={cfg_default.predict_excess_logvol}")
    _, y_default, _ = build_dataset(cfg_default)
    print(f"Pass 1 y_full: {len(y_default):,} rows, "
          f"index [{y_default.index.min()} .. {y_default.index.max()}], "
          f"mean={y_default.mean():+.4f}")
    df_default = diagnose(saved_y, y_default, "PASS 1 — default VolModelConfig() (today's defaults)")

    # --- Pass 2: explicit predict_excess_logvol=False ---
    cfg_explicit = VolModelConfig(predict_excess_logvol=False)
    print(f"\nPass 2 config: asset={cfg_explicit.asset}  predict_excess_logvol={cfg_explicit.predict_excess_logvol}")
    _, y_explicit, _ = build_dataset(cfg_explicit)
    df_explicit = diagnose(saved_y, y_explicit, "PASS 2 — predict_excess_logvol=False explicit")

    # Compare the two passes to each other
    _print_block("PASS 1 vs PASS 2 — are default and explicit identical?")
    common12 = y_default.index.intersection(y_explicit.index)
    d12 = (y_default.loc[common12] - y_explicit.loc[common12]).abs()
    print(f"Common rows: {len(common12):,}")
    print(f"max |delta|: {d12.max():.3e}   mean |delta|: {d12.mean():.3e}   exact-match: {(d12 < 1e-9).sum():,}/{len(common12):,}")
    if (d12 < 1e-9).all():
        print("=> default config and predict_excess_logvol=False produce IDENTICAL y_full.")
        print("   => predict_excess_logvol is NOT the cause of the shift.")
    else:
        print("=> default config and explicit-False DIFFER — predict_excess_logvol may be a factor.")

    # --- TSRV-backup cross-check: does saved_y match the OLD (start-stamped) TSRV file? ---
    _print_block("TSRV BACKUP CROSS-CHECK")
    bak_path = Path("TSRV/output/BTCUSDT_tsrv_15m.parquet.pre_end_stamp_bak")
    cur_path = Path("TSRV/output/BTCUSDT_tsrv_15m.parquet")
    if bak_path.exists():
        bak = pd.read_parquet(bak_path)
        cur = pd.read_parquet(cur_path)
        print(f"Backup TSRV ('pre_end_stamp_bak') exists. Shape={bak.shape}")
        print(f"Backup index range: {bak.index.min()} .. {bak.index.max()}")
        print(f"Current index range: {cur.index.min()} .. {cur.index.max()}")
        # Do they differ only by index?
        bak_shifted_index = bak.index + pd.Timedelta(minutes=15)
        if bak_shifted_index.equals(cur.index):
            print("=> Backup index is exactly current index - 15min. TSRV was re-stamped (shifted +15min).")
        else:
            overlap = bak_shifted_index.intersection(cur.index)
            print(f"=> Backup index+15min overlaps current index in {len(overlap):,} rows.")

        # Does saved_y on its own index match log(bak.vol_ann)?
        bak_ln = np.log(bak[bak["vol_ann"] > 0]["vol_ann"])
        common_bak = saved_y.index.intersection(bak_ln.index)
        if len(common_bak):
            delta_bak = (bak_ln.loc[common_bak] - saved_y.loc[common_bak]).abs()
            print(f"saved_y vs backup log(vol_ann) on shared index ({len(common_bak):,} rows): "
                  f"exact={(delta_bak < 1e-9).sum():,}  close(<1e-4)={(delta_bak < 1e-4).sum():,}")

        # Does current y at T match log(current TSRV at T) (end-stamped)?
        cur_ln = np.log(cur[cur["vol_ann"] > 0]["vol_ann"])
        common_cur = y_default.index.intersection(cur_ln.index)
        if len(common_cur):
            delta_cur = (cur_ln.loc[common_cur] - y_default.loc[common_cur]).abs()
            print(f"current_y vs current log(vol_ann) on shared index ({len(common_cur):,} rows): "
                  f"exact={(delta_cur < 1e-9).sum():,}  close(<1e-4)={(delta_cur < 1e-4).sum():,}")
    else:
        print(f"No backup TSRV file at {bak_path}.")

    # --- Write per-row deltas ---
    if not df_default.empty:
        CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        df_default.to_csv(CSV_OUT)
        print(f"\nPer-row deltas (pass 1) written to {CSV_OUT}")


if __name__ == "__main__":
    main()
