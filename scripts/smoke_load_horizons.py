#!/usr/bin/env python
"""Per-horizon LOAD smoke test for the 7-horizon vol-forecast fleet.

For each horizon H in {5,15,30,60,240,480,1440}, confirm that
``run_walkforward_pruned_svi_har_pd.py`` can LOAD its sidecars with the
MF-2 stamping-aware code — WITHOUT running a full (or even a 1-month)
train. We invoke the SAME loaders the walk-forward main() calls:

  - vol_model.labels.load_labels(cfg)                  (stamping-aware label shift)
  - run_walkforward._load_svi_atm_1d(ccy, H)           (SVI +H alignment shift)
  - run_walkforward._load_har_pd(asset, H)             (stamping-aware HAR-PD shift)

Pass criteria per horizon:
  - no FileNotFoundError / KeyError from any loader,
  - load_labels reports the correct stamping branch (h=15 END-stamped -> -freq
    shift; all other horizons START-stamped -> no shift, MF-2 path),
  - svi_atm_1d and HAR-PD non-null coverage printed.

Exit 0 iff every horizon loads clean.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.labels import detect_stamping, load_labels
from vol_model.run_walkforward_pruned_svi_har_pd import (
    _load_har_pd, _load_svi_atm_1d, ADD_HAR_PD_FEATURES, ADD_SVI_FEATURES,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

HORIZONS = [5, 15, 30, 60, 240, 480, 1440]
ASSET = "BTCUSDT"
CCY = "BTC"


def smoke_one(h: int) -> dict:
    cfg = VolModelConfig(asset=ASSET)
    cfg.freq_s = h * 60
    row: dict = {"h": h, "ok": False, "err": ""}

    # 1. labels — exercises the MF-2 stamping-aware shift inside load_labels.
    tsrv = pd.read_parquet(cfg.get_tsrv_path())
    stamping = detect_stamping(tsrv.index, cfg.freq_s)
    row["tsrv_stamping"] = stamping
    labels = load_labels(cfg)  # clip=True default fine for a load smoke
    row["n_labels"] = len(labels)
    # Expected MF-2 branch: shift applied ONLY when END-stamped (h=15 only).
    row["label_shift_branch"] = (
        "-freq (END)" if (cfg.tsrv_end_stamped and stamping == "END")
        else "0 (START/MF-2)"
    )

    # 2. SVI sidecar load (+H alignment shift).
    svi_add = _load_svi_atm_1d(CCY, horizon_minutes=cfg.horizon_minutes)
    row["n_svi"] = len(svi_add)
    row["svi_cov"] = float(svi_add[ADD_SVI_FEATURES[0]].notna().mean())

    # 3. HAR-PD sidecar load (stamping-aware shift).
    har = _load_har_pd(ASSET, horizon_minutes=cfg.horizon_minutes)
    row["n_har"] = len(har)
    row["har_cov"] = float(har[ADD_HAR_PD_FEATURES].notna().all(axis=1).mean())
    row["har_stamping"] = detect_stamping(
        pd.read_parquet(
            f"vol_model/output/{ASSET}_features_har_pd_{h}m.parquet"
        ).pipe(lambda d: d.set_index("timestamp") if "timestamp" in d.columns else d).index,
        cfg.freq_s,
    )

    row["ok"] = True
    return row


def main() -> int:
    rows = []
    all_ok = True
    for h in HORIZONS:
        try:
            r = smoke_one(h)
        except Exception as e:  # noqa: BLE001 — surface the precise loader failure
            r = {"h": h, "ok": False, "err": f"{type(e).__name__}: {e}"}
            all_ok = False
        rows.append(r)

    print("\n" + "=" * 100)
    print("PER-HORIZON LOAD SMOKE (loaders only, no train)")
    print("=" * 100)
    hdr = (f"{'H(min)':>7} | {'OK':>3} | {'tsrv_stamp':>10} | {'label_branch':>15} | "
           f"{'svi_rows':>9} {'svi_cov':>8} | {'har_rows':>9} {'har_cov':>8} {'har_stamp':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["ok"]:
            print(f"{r['h']:>7} | {'YES':>3} | {r['tsrv_stamping']:>10} | "
                  f"{r['label_shift_branch']:>15} | {r['n_svi']:>9} {100*r['svi_cov']:>7.1f}% | "
                  f"{r['n_har']:>9} {100*r['har_cov']:>7.1f}% {r['har_stamping']:>9}")
        else:
            print(f"{r['h']:>7} | {'NO':>3} | ERROR: {r['err']}")
    print("=" * 100)
    print("RESULT:", "ALL HORIZONS LOAD CLEAN" if all_ok else "ONE OR MORE HORIZONS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
