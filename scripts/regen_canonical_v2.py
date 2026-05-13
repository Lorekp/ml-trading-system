"""Regenerate the canonical V2 trade log + Kelly sidecars.

Two-pass bootstrap (matches the standing recipe):

  Pass 1: V2 backtester with kelly_mult_path=None → draft trade log.
  Step 2: scripts/generate_kelly_sidecar.py reads the draft trade log
          + aligned predictions → BTC_v2_kelly_mult_{K1,K2,K3}.parquet.
  Step 3: scripts/generate_kelly_slope_amped_sidecar.py composes K1
          with the term-slope amplifier → BTC_v2_kelly_mult_K1_slope.parquet.
  Pass 2: V2 backtester with the new K1_slope sidecar → final trade log
          → vol_arb/output/BTC_v2_trade_log.parquet.

Used after any V2-preset config change (cycle-12: jf=1.0).
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regen_v2")

CANON_TL = REPO_ROOT / "vol_arb/output/BTC_v2_trade_log.parquet"
DRAFT_TL = REPO_ROOT / "vol_arb/output/BTC_v2_trade_log_draft.parquet"


def _detect_parquet_cadence(parquet_path: Path) -> str:
    """Cycle-32 (2026-05-13): infer the actual bucket cadence of an
    SVI params parquet from its timestamp row spacing.

    Returns ``"15min"``, ``"1h"``, or ``"{N}s"`` for unrecognized
    cadences. Cycle-28b only compared CONFIG defaults; this lets the
    guard also detect a drift between what the config CLAIMS and what
    the data ACTUALLY contains.
    """
    import pandas as pd

    params = pd.read_parquet(parquet_path)
    if "timestamp" in params.columns:
        ts = pd.to_datetime(params["timestamp"], utc=True)
    else:
        ts = pd.to_datetime(params.index, utc=True)
    unique_ts = pd.Series(ts).drop_duplicates().sort_values()
    if len(unique_ts) < 2:
        return "unknown"
    median_spacing_s = float(unique_ts.diff().dropna().dt.total_seconds().median())
    if abs(median_spacing_s - 900.0) < 60.0:
        return "15min"
    if abs(median_spacing_s - 3600.0) < 60.0:
        return "1h"
    return f"{median_spacing_s:.0f}s"


def _assert_svi_bucket_freq_parity() -> None:
    """Cycle-28b (2026-05-13, hardened cycle-32 2026-05-13): refuse to
    regen canonical V2 when live and offline have diverged on
    ``svi_bucket_freq``.

    The vol_arb pipeline (``vol_arb/pipeline.py``) reads
    ``VolArbConfig().svi_bucket_freq``; the live engine reads
    ``PaperConfig.with_v2_preset().svi_bucket_freq``. Both default
    align with the canonical offline pipeline cadence (cycle-30:
    ``"15min"``). A mismatch — typically a manual override like
    ``--svi-bucket-freq 1h`` passed to an upstream regen — means
    offline calibrated buckets at one cadence while live applies the
    F-4 contract (``bucket_ts + bucket_freq <= now``) at another. The
    consequence is silent IV drift at every hourly bar.

    Cycle-28b checked config-default strings only. Cycle-32 also
    inspects the ACTUAL svi_surface_params parquet's row spacing —
    catches drift where the config says one thing but the parquet was
    built at another cadence (the cycle-30 root-cause class).

    This guard catches the mismatch BEFORE the canonical trade log is
    overwritten. If it fires, fix the upstream pipeline's bucket
    cadence and re-run.
    """
    from deribit_paper.config import PaperConfig
    from vol_arb.config import VolArbConfig

    live_freq = PaperConfig.with_v2_preset().svi_bucket_freq
    offline_freq = VolArbConfig().svi_bucket_freq

    # Layer 1 (cycle-28b): config-default parity.
    if live_freq != offline_freq:
        raise SystemExit(
            f"Cycle-28b parity guard FAILED: "
            f"live svi_bucket_freq={live_freq!r}, "
            f"offline svi_bucket_freq={offline_freq!r}. "
            "Both must match for live-parity regen. "
            "Aborting to protect canonical trade log."
        )

    # Layer 2 (cycle-32): parquet-cadence parity. Inspect the actual
    # offline params parquet whose cadence the live F-4 contract MUST
    # match for observable-timestamp alignment. We check the matched-
    # cadence file the backtester reads (svi_surface_params_15min for
    # 15min cadence, svi_surface_params_1h for 1h, etc.).
    candidates = [
        REPO_ROOT / "vol_arb/output/svi_surface_params_15min.parquet",
        REPO_ROOT / "vol_arb/output/svi_surface_params_1h.parquet",
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        # Detect cadence of each existing file; whichever matches the
        # config gets greenlight, mismatch raises.
        cadences = {p.name: _detect_parquet_cadence(p) for p in existing}
        # Pick the parquet whose cadence matches live_freq if present;
        # otherwise flag every existing file with its detected cadence.
        matching = [
            name for name, cad in cadences.items() if cad == live_freq
        ]
        if not matching:
            raise SystemExit(
                f"Cycle-32 parity guard FAILED: live svi_bucket_freq="
                f"{live_freq!r} but no offline params parquet matches "
                f"that cadence. Detected: {cadences}. "
                "Either rebuild the offline parquet at the live cadence "
                "(via vol_arb/run_pipeline.py --svi-bucket-freq "
                f"{live_freq}) OR change live's "
                "LiveSVICalibrator.svi_config.bucket_freq to match the "
                "existing parquet."
            )
        logger.info(
            "Cycle-32 parquet-cadence guard: live=%r matches %s (OK)",
            live_freq, matching[0],
        )

    logger.info(
        "Cycle-28b config-parity guard: svi_bucket_freq=%r live==offline (OK)",
        live_freq,
    )


def _run_one_v2(kelly_mult_path: str | None) -> pd.DataFrame:
    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    cfg.kelly_mult_path = kelly_mult_path
    _, tl = VolArbBacktester(cfg).run_backtest()
    return tl


def main() -> int:
    _assert_svi_bucket_freq_parity()
    t0 = time.time()

    # ── Pass 1: draft (kelly off) ──
    logger.info("Pass 1/2: draft V2 backtest with kelly_mult_path=None ...")
    tl_draft = _run_one_v2(None)
    DRAFT_TL.parent.mkdir(parents=True, exist_ok=True)
    tl_draft.to_parquet(DRAFT_TL)
    exe = tl_draft[tl_draft["N_contracts_nc"] > 0]
    logger.info("  draft: NC PnL $%s, trades %d, shorts %d, longs %d",
                f"{float(exe['net_pnl_nc'].sum()):,.0f}",
                int(len(exe)),
                int((exe["signal"] == -1).sum()),
                int((exe["signal"] == 1).sum()))

    # ── Step 2: Kelly K1/K2/K3 from draft ──
    logger.info("Step 2/4: generating Kelly K1/K2/K3 sidecars from draft ...")
    res = subprocess.run(
        [sys.executable, "scripts/generate_kelly_sidecar.py",
         "--trade-log", str(DRAFT_TL)],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": ".", **__import__("os").environ},
    )
    if res.returncode != 0:
        logger.error("Kelly sidecar generation failed (rc=%d)", res.returncode)
        return 2

    # ── Step 3: Kelly K1+slope amplifier ──
    logger.info("Step 3/4: generating Kelly K1+slope sidecar ...")
    res = subprocess.run(
        [sys.executable, "scripts/generate_kelly_slope_amped_sidecar.py",
         "--trade-log", str(DRAFT_TL)],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": ".", **__import__("os").environ},
    )
    if res.returncode != 0:
        logger.error("Kelly K1+slope generation failed (rc=%d)", res.returncode)
        return 3

    # ── Pass 4: final V2 backtest with the new K1_slope sidecar ──
    kelly_slope = "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
    logger.info("Pass 4/4: final V2 backtest with kelly_mult_path=%s ...",
                kelly_slope)
    tl_final = _run_one_v2(kelly_slope)
    tl_final.to_parquet(CANON_TL)
    exe = tl_final[tl_final["N_contracts_nc"] > 0]

    logger.info("=" * 80)
    logger.info("Cycle-12 regen DONE in %.1fs", time.time() - t0)
    logger.info("Canonical trade log -> %s", CANON_TL)
    logger.info("  NC PnL:       $%s", f"{float(exe['net_pnl_nc'].sum()):,.0f}")
    logger.info("  Trades:       %d (%d short, %d long)",
                int(len(exe)),
                int((exe["signal"] == -1).sum()),
                int((exe["signal"] == 1).sum()))
    # Daily Sharpe NC
    import numpy as np
    daily = exe["net_pnl_nc"].resample("1D").sum()
    if daily.std(ddof=1) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))
        logger.info("  Daily Sharpe: %.2f", sharpe)
    # MDD
    eq = 100_000 + exe["net_pnl_nc"].cumsum()
    mdd = float((eq - eq.cummax()).min())
    logger.info("  MDD:          $%s", f"{mdd:,.0f}")
    # Per-year
    per_year = exe.groupby(exe.index.year)["net_pnl_nc"].sum().to_dict()
    for yr, v in sorted(per_year.items()):
        logger.info("  %d:           $%s", yr, f"{v:,.0f}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
