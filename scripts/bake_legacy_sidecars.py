#!/usr/bin/env python
"""Bake faithful ``{asset}_meta.json`` sidecars for the EXISTING deployed S5
Binance directional models — WITHOUT replacing them (Path A2: preserve + verify).

Background
----------
S5 is blocked: ``configs/paper_binance.yaml`` sets ``require_metadata: true`` but
ZERO ``{asset}_meta.json`` sidecars exist for the deployed models, so the live
loader (``src/live/signal_generator.py:91-109``) skips every model. The loader
reads exactly two fields per sidecar: ``vol_threshold`` and ``conf_threshold``.

Both are FIT-TIME statistics computed inside the walk-forward (``model.py``):
  * ``vol_threshold``  = train-window EWM-std(returns) 50th-pct  (model.py:242-243)
  * ``conf_threshold`` = 70th-pct of |train predict_proba - 0.5| (model.py:305-308)

They are not stored in the joblib, so we must reproduce the fit to recover them.

This script reproduces the deployed thresholds by replaying ``download_and_train.py``
``train_one`` on the SAME data **sliced to the original 2026-04-03 train cutoff**,
then VERIFIES the reproduction by comparing the freshly-fit ``coef_`` / ``intercept_``
(and scaler ``mean_`` / ``scale_``) against the deployed joblib. If the coef_ match
is within tolerance, the recovered thresholds ARE the exact deployed thresholds and
the sidecar can be written with confidence.

The deployed pipeline is deterministic (fixed seed=42, saga solver) so re-fitting
on identical data reproduces the same model bit-for-bit. The key subtlety: the cached
1m data now extends PAST the original April train; slicing to the cutoff makes the
walk-forward land on the same final month as the deployed model.

Modes
-----
* ``--smoke`` (default behaviour here): REPORT only. No writes to the live model dir.
  Optionally dumps recovered sidecars to a scratch dir ``scripts/_bake_smoke_out/``.
* ``--full``: write real ``{asset}_meta.json`` sidecars next to the deployed joblibs
  for every asset in ``data/live/models/metadata.json``. ONE flag flip from smoke.

Usage
-----
    PYTHONPATH=src .venv/Scripts/python.exe scripts/bake_legacy_sidecars.py \
        --assets BTC,ETH,SOL --tfs 15m,1h,3h,4h,24h \
        --cutoff "2026-04-03 23:54:12" --smoke --workers 2

    # full bake (DO NOT run during smoke validation):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/bake_legacy_sidecars.py \
        --full --workers 4
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make the repo root the cwd and put src on the path (mirror download_and_train.py).
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "scripts"))
warnings.filterwarnings("ignore")

# Windows consoles default to cp1252; make stdout UTF-8 so any stray non-ASCII
# in progress output never crashes the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# Reuse the EXACT production constants from the trainer so we cannot drift.
from download_and_train import OHLCV_FEATURES, DATA_DIR, MODEL_DIR, TFS  # noqa: E402

# The deployed models were trained on 2026-04-03 (metadata.json) but their LAST
# walk-forward retrain window is MARCH 2026 — the April monthly data was cached
# AFTER that train. Empirically verified: slicing the cached 1m data at end-of-March
# reproduces every deployed (asset x tf) coef_ BIT-FOR-BIT (max|dcoef| = 0.0,
# scaler mean_/scale_ exact) across BTC/ETH/SOL x all 5 TFs. The April-3 cutoff
# only reproduces the higher TFs (3h/4h/24h) because their tiny April test window
# is dropped; 15m/1h wrongly form an April final window under that cutoff.
DEFAULT_CUTOFF = "2026-03-31 23:59:59"
TRAINED_AT_ISO = "2026-04-03T23:54:12Z"  # canonical sidecar 'trained_at' for legacy models

# coef_ reproduction tolerance — PASS means we faithfully reproduced the deployed fit.
COEF_TOL = 1e-6

SCRATCH_OUT = Path("scripts/_bake_smoke_out")

# TF label -> pandas resample alias (subset of TFS keyed by label).
TF_RESAMPLE = {label: resample for (label, resample) in TFS}


# ════════════════════════════════════════════════════════════════════════════
#  Per-(asset, tf) worker — reproduce + verify + (optionally) write sidecar
# ════════════════════════════════════════════════════════════════════════════

def bake_one(args: tuple) -> dict:
    """Reproduce the deployed model for one (asset, tf) and recover its thresholds.

    Returns a result dict (JSON-serialisable) with verification + threshold info.
    """
    import numpy as np
    import pandas as pd
    import joblib

    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.config import (
        SystemConfig, ModelConfig, FeatureConfig, RegimeFilterConfig,
        ConfidenceFilterConfig, SignalConfig, DataConfig,
    )

    asset, sym, tf_label, tf_resample, cutoff_str, write_mode, out_root = args
    warnings.filterwarnings("ignore")
    t0 = time.time()

    res = {
        "asset": asset, "tf": tf_label,
        "max_coef_delta": None, "max_intercept_delta": None,
        "scaler_mean_match": None, "scaler_scale_match": None,
        "conf_threshold": None, "vol_threshold": None,
        "n_signals": 0, "wall_s": 0.0, "verify_pass": False,
        "wrote": None, "error": None,
    }

    model_path = MODEL_DIR / tf_label / f"{asset}_model.joblib"
    scaler_path = MODEL_DIR / tf_label / f"{asset}_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        res["error"] = "deployed joblib missing"
        res["wall_s"] = round(time.time() - t0, 2)
        return res

    try:
        # ── EXACT config from download_and_train.train_one (copied verbatim) ──
        config = SystemConfig(
            model=ModelConfig(
                C=0.10, penalty="elasticnet", solver="saga", max_iter=1000,
                seed=42, fwd_bars=4, signal_model_type="asset_logistic",
            ),
            features=FeatureConfig(
                feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
            ),
            regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
            confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
            signal=SignalConfig(step_size=0.1),
            data=DataConfig(
                data_dirs=[str(DATA_DIR)],
                timeframe=tf_resample, regime_timeframe="4h",
            ),
        )

        # ── Load 1m, then SLICE to the original train cutoff (the key step) ──
        ohlcv_1m = load_binance_1m(sym, [DATA_DIR])
        cutoff = pd.Timestamp(cutoff_str)
        if ohlcv_1m.index.tz is not None:
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            else:
                cutoff = cutoff.tz_convert("UTC")
        else:
            if cutoff.tzinfo is not None:
                cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        ohlcv_1m = ohlcv_1m[ohlcv_1m.index <= cutoff]

        if len(ohlcv_1m) == 0:
            res["error"] = "no data <= cutoff"
            res["wall_s"] = round(time.time() - t0, 2)
            return res

        # ── Reproduce the walk-forward fit (deterministic; seed=42) ──
        ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
        feat = compute_features(ohlcv_tf, config.features)
        try:
            wf = WalkForwardModel(config, force_monthly_retrain=True)
        except TypeError:
            wf = WalkForwardModel(config)
        sigs = wf.generate_signals_batch(ohlcv_tf, feat, asset)
        res["n_signals"] = int(len(sigs))

        if wf._cached_model is None or wf._cached_scaler is None:
            res["error"] = "no model after reproduction (no signals)"
            res["wall_s"] = round(time.time() - t0, 2)
            return res

        # ── VERIFY against the deployed artifacts ──
        dep_model = joblib.load(model_path)
        dep_scaler = joblib.load(scaler_path)
        repro_model = wf._cached_model
        repro_scaler = wf._cached_scaler

        max_coef_delta = float(np.max(np.abs(dep_model.coef_ - repro_model.coef_)))
        max_int_delta = float(
            np.max(np.abs(np.atleast_1d(dep_model.intercept_) - np.atleast_1d(repro_model.intercept_)))
        )
        mean_match = bool(np.allclose(dep_scaler.mean_, repro_scaler.mean_, atol=1e-9, rtol=0))
        scale_match = bool(np.allclose(dep_scaler.scale_, repro_scaler.scale_, atol=1e-9, rtol=0))

        res["max_coef_delta"] = max_coef_delta
        res["max_intercept_delta"] = max_int_delta
        res["scaler_mean_match"] = mean_match
        res["scaler_scale_match"] = scale_match
        res["conf_threshold"] = (
            float(wf.last_conf_threshold) if wf.last_conf_threshold is not None else None
        )
        res["vol_threshold"] = (
            float(wf.last_vol_threshold) if wf.last_vol_threshold is not None else None
        )
        res["verify_pass"] = bool(max_coef_delta < COEF_TOL)

        # ── Build the sidecar JSON (schema = download_and_train.py:225-235) ──
        meta = {
            "asset": asset,
            "tf": tf_label,
            "trained_at": TRAINED_AT_ISO,  # preserve the legacy train timestamp
            "features": list(OHLCV_FEATURES),
            "vol_filter_quantile": 0.50,
            "conf_threshold_pct": 70.0,
            "conf_threshold": res["conf_threshold"],
            "vol_threshold": res["vol_threshold"],
            "n_signals": int(len(sigs)),
        }

        # ── Write — gated on verification and mode ──
        if res["verify_pass"]:
            if write_mode == "full":
                # Real write next to the deployed joblib.
                meta_path = MODEL_DIR / tf_label / f"{asset}_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                res["wrote"] = str(meta_path)
            elif write_mode == "scratch":
                # Optional scratch write for inspection — never touches live dir.
                d = Path(out_root) / tf_label
                d.mkdir(parents=True, exist_ok=True)
                meta_path = d / f"{asset}_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                res["wrote"] = str(meta_path)
            # write_mode == "report": no write
        # If verify FAILED we never write a live sidecar (would be unfaithful).

        res["wall_s"] = round(time.time() - t0, 2)
        return res

    except FileNotFoundError:
        res["error"] = "no data"
        res["wall_s"] = round(time.time() - t0, 2)
        return res
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        res["wall_s"] = round(time.time() - t0, 2)
        return res


# ════════════════════════════════════════════════════════════════════════════
#  CLI / orchestration
# ════════════════════════════════════════════════════════════════════════════

def load_full_asset_list() -> list[str]:
    """All ~190 assets from the deployed metadata.json (for --full)."""
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return list(meta["assets"])


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bake legacy S5 sidecars (Path A2).")
    p.add_argument("--assets", type=str, default="BTC,ETH,SOL",
                   help="Comma list of assets (ignored when --full).")
    p.add_argument("--tfs", type=str, default="15m,1h,3h,4h,24h",
                   help="Comma list of TF labels.")
    p.add_argument("--cutoff", type=str, default=DEFAULT_CUTOFF,
                   help="Original train cutoff; data is sliced to <= this.")
    p.add_argument("--workers", type=int, default=2,
                   help="ProcessPool workers (keep <=3 while box is busy).")
    p.add_argument("--smoke", action="store_true",
                   help="Report only (no live writes). Default if neither flag set.")
    p.add_argument("--scratch", action="store_true",
                   help="In smoke mode, also dump sidecars to scripts/_bake_smoke_out/.")
    p.add_argument("--full", action="store_true",
                   help="Bake ALL ~190 assets and WRITE real sidecars. (heavy)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    T0 = time.time()

    if args.full:
        assets = load_full_asset_list()
        write_mode = "full"
        out_root = ""
    else:
        assets = [a.strip() for a in args.assets.split(",") if a.strip()]
        if args.scratch:
            write_mode = "scratch"
            out_root = str(SCRATCH_OUT)
        else:
            write_mode = "report"
            out_root = ""

    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    for tf in tfs:
        if tf not in TF_RESAMPLE:
            print(f"  ERROR: unknown TF '{tf}' (known: {list(TF_RESAMPLE)})")
            return 1

    print("=" * 78)
    print(f"  BAKE LEGACY SIDECARS -- Path A2 (preserve + verify)")
    print("=" * 78)
    print(f"  Mode       : {'FULL (writes live sidecars)' if args.full else 'SMOKE'}"
          f"  write_mode={write_mode}")
    print(f"  Assets     : {len(assets)} ({assets[:6]}{'...' if len(assets) > 6 else ''})")
    print(f"  TFs        : {tfs}")
    print(f"  Cutoff     : {args.cutoff}")
    print(f"  Workers    : {args.workers}")
    print(f"  coef_ tol  : {COEF_TOL}")
    print()

    work = []
    for asset in assets:
        sym = f"{asset}USDT"
        for tf in tfs:
            work.append((asset, sym, tf, TF_RESAMPLE[tf], args.cutoff, write_mode, out_root))

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(bake_one, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            results.append(r)
            status = "PASS" if r["verify_pass"] else ("ERR " if r["error"] else "FAIL")
            cd = r["max_coef_delta"]
            cd_s = f"{cd:.2e}" if cd is not None else "  n/a  "
            print(f"  [{done}/{len(work)}] {r['asset']:<10} {r['tf']:<4} "
                  f"coef_d={cd_s} conf={r['conf_threshold']} vol={r['vol_threshold']} "
                  f"n={r['n_signals']:>5} {r['wall_s']:>6.1f}s  {status}"
                  + (f"  ({r['error']})" if r["error"] else ""))

    # ── Summary table ──
    print()
    print("  " + "-" * 100)
    print(f"  {'asset':<10}{'tf':<5}{'max|coef_d|':<13}{'mean_ok':<8}{'scale_ok':<9}"
          f"{'conf_thr':<12}{'vol_thr':<12}{'n_sig':<8}{'sec':<7}{'VERIFY'}")
    print("  " + "-" * 100)
    n_pass = 0
    for r in sorted(results, key=lambda x: (x["asset"], x["tf"])):
        cd = r["max_coef_delta"]
        cd_s = f"{cd:.2e}" if cd is not None else "n/a"
        conf_s = f"{r['conf_threshold']:.6f}" if r["conf_threshold"] is not None else "None"
        vol_s = f"{r['vol_threshold']:.6f}" if r["vol_threshold"] is not None else "None"
        verdict = "PASS" if r["verify_pass"] else ("ERR" if r["error"] else "FAIL")
        if r["verify_pass"]:
            n_pass += 1
        print(f"  {r['asset']:<10}{r['tf']:<5}{cd_s:<13}"
              f"{str(r['scaler_mean_match']):<8}{str(r['scaler_scale_match']):<9}"
              f"{conf_s:<12}{vol_s:<12}{r['n_signals']:<8}{r['wall_s']:<7.1f}{verdict}")
    print("  " + "-" * 100)

    elapsed = time.time() - T0
    print(f"\n  {n_pass}/{len(results)} verified PASS  |  wall {elapsed:.0f}s  "
          f"|  workers={args.workers}")

    # Per-task timing aggregate (useful for extrapolation).
    timed = [r["wall_s"] for r in results if r["wall_s"] > 0]
    if timed:
        import statistics
        print(f"  per-task wall: mean {statistics.mean(timed):.1f}s  "
              f"median {statistics.median(timed):.1f}s  "
              f"min {min(timed):.1f}s  max {max(timed):.1f}s")

    # Machine-readable dump for the report.
    if not args.full:
        SCRATCH_OUT.mkdir(parents=True, exist_ok=True)
        (SCRATCH_OUT / "smoke_results.json").write_text(json.dumps(results, indent=2))
        print(f"  results JSON -> {SCRATCH_OUT / 'smoke_results.json'}")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
