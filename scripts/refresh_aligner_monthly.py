"""Monthly aligner β* refresh orchestrator.

Flagged-issue follow-up to E.2.3 (2026-05-01). The aligner sidecar
(``vol_model/output/aligner_betas_by_month.json``) needs to be
regenerated every month after the prior month's data is fully
available. Pre-this-script, the procedure was a manual, multi-step
sequence prone to "forgot a step" errors.

This script orchestrates the full pipeline for ONE target month:

  1. **Data availability check**: confirms the required raw inputs
     exist for the target month. Refuses to proceed when:
     - Binance Vision monthly aggTrades isn't published yet (via 404
       check); or
     - Deribit option trades have <80% calendar-day coverage; or
     - DVOL parquet doesn't extend into the target month.

  2. **TSRV + features rebuild** (delegates to vol_arb/run_pipeline.py
     and vol_model.features.har_pd) for the target month only.

  3. **LightGBM single-month walk-forward** (delegates to
     vol_model/run_walkforward_pruned_svi_har_pd.py with --start/--end
     bracketing the target month). Trains one new monthly model on
     trailing data and extracts that month's predictions.

  4. **Aligner run** (delegates to vol_model/run_walkforward_aligner.py)
     to re-fit per-month calibration on the freshly-extended
     predictions parquet.

  5. **β* JSON re-extraction** via scripts/extract_aligner_betas.py.

  6. **Sanity gate**: refuses to swap in the new JSON when the new
     month's β* falls outside the engine's safety band [-0.5, +0.5]
     (matches ``vol_arb.sidecars.ALIGNER_BETA_MIN/MAX``).

  7. **Atomic swap** of the JSON via tmp+rename (matches the
     HarnessWriter pattern in E.1.3).

Designed to be cron-runnable on the 5th of each month (Binance Vision
typically publishes the prior month's aggTrades by day 3-5).

Usage::

    cd C:/Claude/ml-trading-system
    PYTHONPATH=. python scripts/refresh_aligner_monthly.py --month 2026-04

    # Dry-run mode: print plan, run feasibility checks, no write.
    PYTHONPATH=. python scripts/refresh_aligner_monthly.py --month 2026-04 --dry-run

    # Auto-pick the most recent complete calendar month before today.
    PYTHONPATH=. python scripts/refresh_aligner_monthly.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS_RAW = REPO_ROOT / "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"
PREDS_ALIGNED = REPO_ROOT / "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
BETAS_JSON = REPO_ROOT / "vol_model/output/aligner_betas_by_month.json"

# Raw data inputs.
BTC_AGGTRADES_DIR = REPO_ROOT / "data/binance-aggTrades-parquet/BTCUSDT"
BTC_OPTIONS_DIR = REPO_ROOT / "data/deribit/trades/BTC_option"
DVOL_PATH = REPO_ROOT / "data/deribit/dvol/BTC_dvol_60m.parquet"

# Sanity bands (match vol_arb/sidecars.py).
BETA_MIN, BETA_MAX = -0.5, 0.5

# Binance Vision publish heuristic — daily files exist throughout the
# month; monthly files publish 3-5 days into the next month.
BINANCE_MONTHLY_URL = (
    "https://data.binance.vision/data/futures/um/monthly/aggTrades"
    "/BTCUSDT/BTCUSDT-aggTrades-{ym}.zip"
)


def _autopick_month() -> str:
    """Most recent complete calendar month strictly before today."""
    today = pd.Timestamp.now(tz="UTC")
    return str(pd.Period(today, freq="M") - 1)


# ── Phase 1: data availability ────────────────────────────────────────────


def _check_aggtrades(target_ym: str) -> tuple[bool, str]:
    """Either local monthly parquet exists, or Binance Vision monthly is
    published. Returns (ok, reason_string)."""
    local = BTC_AGGTRADES_DIR / f"{target_ym}.parquet"
    if local.exists():
        return True, f"local parquet present ({local.stat().st_size:,} bytes)"
    # Probe Binance Vision availability.
    url = BINANCE_MONTHLY_URL.format(ym=target_ym)
    try:
        r = requests.head(url, timeout=10)
    except requests.RequestException as e:
        return False, f"network probe failed: {e}"
    if r.status_code == 200:
        return True, f"Binance Vision monthly archive available ({url})"
    return False, (
        f"Binance Vision returned HTTP {r.status_code} for {url} — "
        f"monthly archive not yet published. Try again on day 3-5 of "
        f"the following month, OR manually download daily aggTrades "
        f"and concatenate."
    )


def _check_option_trades(target_ym: str) -> tuple[bool, str]:
    """≥80% calendar-day coverage of the target month."""
    if not BTC_OPTIONS_DIR.is_dir():
        return False, f"options dir missing ({BTC_OPTIONS_DIR})"
    period = pd.Period(target_ym, freq="M")
    days_in_month = period.days_in_month
    found = sorted(BTC_OPTIONS_DIR.glob(f"{target_ym}-*.parquet"))
    coverage_pct = 100.0 * len(found) / days_in_month
    if coverage_pct < 80.0:
        return False, (
            f"option trades coverage {coverage_pct:.0f}% "
            f"({len(found)}/{days_in_month} days). Need ≥80%; "
            f"download daily files via scripts/download_deribit_robust.py."
        )
    return True, f"options coverage {coverage_pct:.0f}% ({len(found)} day files)"


def _check_dvol(target_ym: str) -> tuple[bool, str]:
    """DVOL parquet covers through (or past) the end of the target month.

    The DVOL file stores ``timestamp`` as a regular column, not the
    index — read the column rather than checking ``df.index``.
    """
    if not DVOL_PATH.exists():
        return False, f"DVOL parquet missing ({DVOL_PATH})"
    try:
        df = pd.read_parquet(DVOL_PATH, columns=["timestamp"])
    except Exception as e:
        return False, f"DVOL parquet unreadable: {e}"
    if "timestamp" not in df.columns or len(df) == 0:
        return False, "DVOL has no timestamp column"
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return False, "DVOL timestamp column has no valid dates"
    last = ts.max()
    target_end = pd.Timestamp(target_ym + "-01", tz="UTC") + pd.offsets.MonthEnd(1)
    if last < target_end:
        return False, (
            f"DVOL last bar {last} < target month end {target_end} "
            f"— refresh DVOL via downloader/tasks/dvol.py"
        )
    return True, f"DVOL covers through {last}"


def phase1_data_availability(target_ym: str) -> bool:
    """All three inputs available? Logs each gate's result."""
    logger.info("Phase 1: data availability for %s", target_ym)
    aggok, aggmsg = _check_aggtrades(target_ym)
    optok, optmsg = _check_option_trades(target_ym)
    dvolok, dvolmsg = _check_dvol(target_ym)
    for ok, name, msg in [
        (aggok, "BTCUSDT aggTrades", aggmsg),
        (optok, "BTC option trades", optmsg),
        (dvolok, "DVOL", dvolmsg),
    ]:
        marker = "OK" if ok else "FAIL"
        logger.info("  [%s] %s: %s", marker, name, msg)
    return aggok and optok and dvolok


# ── Phases 2-5: delegate to existing tooling ──────────────────────────────


def phase2_features(target_ym: str, *, dry_run: bool) -> bool:
    """Run vol_arb pipeline for the target month."""
    logger.info("Phase 2: features rebuild for %s", target_ym)
    if dry_run:
        logger.info("  [DRY-RUN] would invoke vol_arb/run_pipeline.py")
        return True
    # vol_arb/run_pipeline.py rebuilds the merged dataset + SVI features
    # for the target currency. It's idempotent at the month boundary
    # (existing months are not recomputed unless inputs change).
    cmd = [sys.executable, "-m", "vol_arb.run_pipeline", "--currency", "BTC"]
    logger.info("  $ %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": "."})
    if res.returncode != 0:
        logger.error("vol_arb pipeline failed (rc=%d)", res.returncode)
        return False
    return True


def phase3_lgbm_predict(target_ym: str, *, dry_run: bool) -> bool:
    """Run a single-month LightGBM walk-forward for the target month.

    The runner's ``--start`` / ``--end`` flags bracket the months
    processed; setting both to the target month produces ONE new
    monthly model + that month's prediction set, appended to the
    existing artifacts in
    ``vol_model/output/models/expanding_pruned_svi_har_pd_v1/``.
    """
    logger.info("Phase 3: LightGBM walk-forward for %s", target_ym)
    if dry_run:
        logger.info("  [DRY-RUN] would invoke run_walkforward_pruned_svi_har_pd")
        return True
    cmd = [
        sys.executable, "-m", "vol_model.run_walkforward_pruned_svi_har_pd",
        "--start", target_ym, "--end", target_ym,
        "--max-trials", "50", "--patience", "15",   # faster preset for monthly increment
    ]
    logger.info("  $ %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": "."})
    if res.returncode != 0:
        logger.error("LightGBM walk-forward failed (rc=%d)", res.returncode)
        return False
    return True


def phase4_aligner(*, dry_run: bool) -> bool:
    """Run the prediction aligner on the (now-extended) predictions parquet."""
    logger.info("Phase 4: prediction aligner walk-forward")
    if dry_run:
        logger.info("  [DRY-RUN] would invoke run_walkforward_aligner")
        return True
    cmd = [sys.executable, "-m", "vol_model.run_walkforward_aligner",
           "--method", "bias"]
    logger.info("  $ %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": "."})
    if res.returncode != 0:
        logger.error("Aligner failed (rc=%d)", res.returncode)
        return False
    return True


def phase5_extract(*, dry_run: bool, output_path: Path) -> bool:
    """Re-run the β* extractor; write to a TEMP path so phase 6 can
    sanity-check before atomic-swapping."""
    logger.info("Phase 5: β* JSON extraction → %s", output_path)
    if dry_run:
        logger.info("  [DRY-RUN] would invoke extract_aligner_betas.py")
        return True
    cmd = [sys.executable, "scripts/extract_aligner_betas.py",
           "--output", str(output_path)]
    logger.info("  $ %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": "."})
    if res.returncode != 0:
        logger.error("Extract aligner betas failed (rc=%d)", res.returncode)
        return False
    return True


# ── Phase 6: sanity gate ──────────────────────────────────────────────────


def phase6_sanity_gate(new_json: Path, target_ym: str) -> bool:
    """Refuse to ship when the new month's β* is out of band, OR when
    the new JSON drops months that the old one carried."""
    if not new_json.exists():
        logger.error("Sanity gate: new JSON %s missing", new_json)
        return False
    with new_json.open("r", encoding="utf-8") as fh:
        new = {str(k): float(v) for k, v in json.load(fh).items()}
    if target_ym not in new:
        logger.error(
            "Sanity gate: target month %s not in new JSON keys (have %s)",
            target_ym, sorted(new.keys())[-3:],
        )
        return False
    new_beta = new[target_ym]
    if not (BETA_MIN <= new_beta <= BETA_MAX):
        logger.error(
            "Sanity gate: target month β* %+.4f outside [%+.2f, %+.2f]",
            new_beta, BETA_MIN, BETA_MAX,
        )
        return False
    # Coverage check vs current sidecar.
    if BETAS_JSON.exists():
        with BETAS_JSON.open("r", encoding="utf-8") as fh:
            old = {str(k): float(v) for k, v in json.load(fh).items()}
        dropped = set(old) - set(new)
        if dropped:
            logger.error(
                "Sanity gate: new JSON drops months %s — refusing to ship",
                sorted(dropped),
            )
            return False
    logger.info(
        "Sanity gate: target %s β=%+.4f in band; coverage preserved",
        target_ym, new_beta,
    )
    return True


# ── Phase 7: atomic swap ─────────────────────────────────────────────────


def phase7_atomic_swap(new_json: Path) -> bool:
    """Replace ``BETAS_JSON`` with ``new_json`` atomically. Mirrors the
    tmp+rename pattern in HarnessWriter._atomic_write (E.1.3)."""
    backup = BETAS_JSON.with_suffix(f".bak.{uuid.uuid4().hex[:8]}.json")
    if BETAS_JSON.exists():
        shutil.copy2(BETAS_JSON, backup)
        logger.info("Backed up old JSON to %s", backup.name)
    try:
        os.replace(new_json, BETAS_JSON)
    except Exception as e:
        logger.error("Atomic swap failed: %s", e)
        if backup.exists():
            shutil.copy2(backup, BETAS_JSON)
            logger.info("Rolled back from %s", backup)
        return False
    logger.info("Swapped: %s is now the live β* sidecar", BETAS_JSON)
    return True


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--month", default="",
                        help="Target month YYYY-MM (default: most recent complete month)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run availability + plan steps; skip pipeline + writes")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Bypass phase 1 (use only when you know data is in)")
    args = parser.parse_args()

    target_ym = args.month or _autopick_month()
    if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", target_ym):
        logger.error("Invalid --month: %s (expected YYYY-MM)", target_ym)
        return 2
    logger.info("=== Aligner monthly refresh — target %s ===", target_ym)

    if not args.skip_data_check:
        if not phase1_data_availability(target_ym):
            logger.error("Data not available for %s — aborting", target_ym)
            return 3

    if not phase2_features(target_ym, dry_run=args.dry_run):
        return 4
    if not phase3_lgbm_predict(target_ym, dry_run=args.dry_run):
        return 5
    if not phase4_aligner(dry_run=args.dry_run):
        return 6

    if args.dry_run:
        logger.info("=== Dry-run complete (no JSON written) ===")
        return 0

    new_json = BETAS_JSON.with_suffix(f".new.{uuid.uuid4().hex[:8]}.json")
    if not phase5_extract(dry_run=False, output_path=new_json):
        return 7
    if not phase6_sanity_gate(new_json, target_ym):
        if new_json.exists():
            new_json.unlink()
        return 8
    if not phase7_atomic_swap(new_json):
        return 9

    logger.info("=== Refresh complete for %s ===", target_ym)
    logger.info("Next step: redeploy to homeserver and restart engine")
    logger.info("    bash scripts/deploy_to_homeserver.sh --skip-tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
