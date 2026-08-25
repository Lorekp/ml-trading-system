#!/usr/bin/env python
"""Per-asset cutoff search for the 23-model sidecar gap.

Background
----------
``scripts/bake_legacy_sidecars.py`` Path A2 reproduces deployed S5 models
on the SAME data sliced to the 2026-03-31 cutoff. It works for ~920 of
the 943 deployed models (BTC, ETH, SOL × all 5 TFs, plus most of the
32-asset universe). 23 models do NOT reproduce under any single cutoff —
they are believed to be the 1-2 month-drift stragglers (models whose
walk-forward retrain landed on a different final month than the rest).

The live engine skips them under ``require_metadata: true``. The spec
needs to know: (a) which assets are blocked, (b) what live-coverage
hit we take, (c) whether per-asset cutoffs recover the missing 23 (cheap
fix), or whether a full monthly retrain is required (proper fix).

This script
-----------
For each (asset, tf) in the 23-set, sweeps a small grid of cutoffs and
records the smallest |coef_delta| it finds. The Path A2 logic is
identical to bake_legacy_sidecars.py:compute_one — we just loop cutoffs
per asset instead of using one global default.

Usage
-----
::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/testnet_per_asset_cutoff_search.py \\
        --assets-file data/live/models/sidecar_gap_23.txt \\
        --out vol_arb/output/S5_SIDECAR_GAP_FIX.md

Where ``sidecar_gap_23.txt`` is a one-asset-per-line list. If absent, the
script discovers the gap from ``data/live/models/metadata.json``
(assets in metadata) minus the 920 sidecars that already exist on disk.

The script is read-only by default; pass ``--write`` to emit
``{asset}_meta.json`` next to the deployed joblibs for any (asset, tf)
where the per-asset search finds a coef_delta < 1e-6.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

# Reuse Path A2 from bake_legacy_sidecars; the per-asset search is a
# thin wrapper that loops cutoffs.
from bake_legacy_sidecars import (
    bake_one, COEF_TOL, TRAINED_AT_ISO, OHLCV_FEATURES, SCRATCH_OUT, MODEL_DIR,
    load_full_asset_list, DEFAULT_CUTOFF,
)

logger = logging.getLogger("sidecar_gap")

# Cutoff grid — covers the 6-week window the bake_legacy_sidecars script
# explores.  We deliberately span the canonical 03-31 boundary AND the
# legacy 04-03 cutoff so both code paths are exercised.  The grid is
# small (6 cutoffs); the per-(asset,tf) cost is dominated by data load
# + walk-forward (~10-30s on the homeserver).
CUTOFF_GRID = [
    "2026-03-15 23:59:59",
    "2026-03-22 23:59:59",
    "2026-03-29 23:59:59",
    "2026-03-31 23:59:59",  # canonical default
    "2026-04-02 23:59:59",
    "2026-04-03 23:59:59",  # legacy metadata.json
]


def discover_gap(model_dir: Path = MODEL_DIR) -> list[str]:
    """Return assets that are in metadata.json but missing at least one TF sidecar."""
    meta = json.loads((model_dir / "metadata.json").read_text())
    assets = list(meta.get("assets", []))
    tfs = ["15m", "1h", "3h", "4h", "24h"]
    missing_assets: set[str] = set()
    for asset in assets:
        for tf in tfs:
            sidecar = model_dir / tf / f"{asset}_meta.json"
            if not sidecar.exists():
                missing_assets.add(asset)
                break
    return sorted(missing_assets)


def load_assets_file(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def per_asset_search(
    asset: str, sym: str | None,
    tfs: list[str], cutoffs: list[str],
    write: bool, workers: int,
) -> list[dict]:
    """For one asset × tf × cutoff, run bake_one; return best result per (tf)."""
    if sym is None:
        sym = f"{asset}USDT"

    # Build all (asset, tf, cutoff) jobs.
    tf_resample = {
        "15m": "15min", "1h": "1h", "3h": "3h", "4h": "4h", "24h": "1D",
    }
    write_mode = "full" if write else "scratch"
    jobs = []
    for tf in tfs:
        for cutoff in cutoffs:
            jobs.append((asset, sym, tf, tf_resample[tf], cutoff, write_mode,
                         str(SCRATCH_OUT)))

    results: list[dict] = []
    if workers <= 1:
        for j in jobs:
            results.append(bake_one(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(bake_one, j) for j in jobs]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    logger.warning("worker failed: %s", e)

    # For each (tf), keep the smallest |coef_delta| result.
    best_per_tf: dict[str, dict] = {}
    for r in results:
        tf = r["tf"]
        if tf not in best_per_tf:
            best_per_tf[tf] = r
            continue
        prev = best_per_tf[tf]
        if (r.get("max_coef_delta") is not None and
            (prev.get("max_coef_delta") is None or
             r["max_coef_delta"] < prev["max_coef_delta"])):
            best_per_tf[tf] = r
    return list(best_per_tf.values())


def render_report(rows: list[dict], out_path: Path, gap_assets: list[str]) -> None:
    """Write the S5_SIDECAR_GAP_FIX.md summary."""
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r.get("verify_pass"))
    n_fail = n_total - n_pass
    by_asset: dict[str, list[dict]] = {}
    for r in rows:
        by_asset.setdefault(r["asset"], []).append(r)

    lines = [
        "# S5 Sidecar Gap — Per-Asset Cutoff Search",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Gap assets searched: **{len(gap_assets)}**",
        f"(asset × tf) pairs: **{n_total}**, "
        f"recovered (coef_delta < {COEF_TOL}): **{n_pass}**, "
        f"unrecovered: **{n_fail}**",
        "",
        "Method: for each missing asset × each of the 5 TFs, sweep 6 cutoffs",
        "(see `CUTOFF_GRID` in `scripts/testnet_per_asset_cutoff_search.py`) and",
        "keep the result with smallest |coef_delta| against the deployed joblib.",
        "This is the cheap fix; if not enough models reproduce, the proper",
        "fix is the next monthly retrain queue pass (Path B).",
        "",
        "## Recovery summary",
        "",
        "| Asset | TFs recovered | TFs failed | Best cutoffs | Notes |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for asset, recs in by_asset.items():
        passed = [r for r in recs if r.get("verify_pass")]
        failed = [r for r in recs if not r.get("verify_pass")]
        cutoffs = sorted({r.get("cutoff", "") for r in recs if r.get("verify_pass")})
        notes = []
        for r in failed:
            err = r.get("error") or "coef_delta>tol"
            notes.append(f"{r['tf']}: {err}")
        lines.append(
            f"| {asset} | {len(passed)} | {len(failed)} | "
            f"{', '.join(cutoffs) if cutoffs else '-'} | "
            f"{'; '.join(notes) if notes else 'all tfs recover'} |"
        )

    lines += [
        "",
        "## Live-coverage impact (current sizing)",
        "",
        "The 23 missing assets are excluded by `require_metadata=true`. The",
        "asset selector (s8_sym15) trades 30 assets in the S8 core + selector;",
        "of these, the 23 gap assets miss every TF. Net effect:",
        "",
        "- Signal universe shrinks from 30 to ~7 (BTC, ETH, SOL, and the 4",
        "  other assets the bake recovered previously).",
        "- Per-asset sizing is unchanged (per-asset inverse-vol), but the",
        "  reduced universe concentrates capital in fewer assets. MaxDD in",
        "  drawdown regimes rises (a tail event on any of the 7 surviving",
        "  assets is unhedged).",
        "- Forecast-as-overlay: independent of sidecars (uses live retrain",
        "  pipeline). No impact.",
        "",
        "## Decision",
        "",
    ]
    if n_pass >= n_total * 0.8:
        lines += [
            f"**PASS — {n_pass}/{n_total} models recover under per-asset cutoffs.**",
            "Run the script with `--write` to emit the sidecars and re-validate",
            "the live paper engine. This is the cheap fix (no retrain needed).",
        ]
    elif n_pass >= n_total * 0.3:
        lines += [
            f"**PARTIAL — {n_pass}/{n_total} recover.**",
            "The cheap fix recovers enough for limited testnet research, but the full 30-asset",
            "universe requires the proper fix: queue a monthly retrain pass",
            "for the remaining assets (Path B).",
        ]
    else:
        lines += [
            f"**FAIL — only {n_pass}/{n_total} recover.**",
            "Cheap fix insufficient. The proper fix is the next monthly",
            "retrain pass (Path B). Record the coverage limitation; retraining",
            "alone does not validate S5 or authorize capital.",
        ]
    lines.append("")
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Per-asset sidecar cutoff search")
    p.add_argument("--assets", default="",
                   help="Comma list of assets to search. Default: discover from "
                        "metadata.json - existing sidecars.")
    p.add_argument("--assets-file", default="",
                   help="Optional path to a file with one asset per line.")
    p.add_argument("--tfs", default="15m,1h,3h,4h,24h",
                   help="Comma list of TF labels.")
    p.add_argument("--cutoffs", default="",
                   help="Override CUTOFF_GRID; comma list of ISO timestamps.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--write", action="store_true",
                   help="Actually write sidecars for recovered models "
                        "(gated on verify_pass).")
    p.add_argument("--out", default="vol_arb/output/S5_SIDECAR_GAP_FIX.md")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.assets_file:
        gap_assets = load_assets_file(Path(args.assets_file))
    elif args.assets:
        gap_assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    else:
        gap_assets = discover_gap()
    logger.info("Searching %d assets", len(gap_assets))

    cutoffs = ([c.strip() for c in args.cutoffs.split(",") if c.strip()]
               if args.cutoffs else CUTOFF_GRID)
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    rows: list[dict] = []
    for asset in gap_assets:
        recs = per_asset_search(
            asset, None, tfs, cutoffs, args.write, args.workers)
        rows.extend(recs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_report(rows, out_path, gap_assets)
    n_pass = sum(1 for r in rows if r.get("verify_pass"))
    print(f"recovered {n_pass}/{len(rows)} (asset × tf) pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
