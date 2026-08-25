#!/usr/bin/env python
"""Testnet smoke-run driver.

Drives ``src/live/testnet/cli.py`` for a fixed wall-clock window and
saves the per-fill JSONL + reconciliation report.  The smoke run is
intentionally short (1-4 weeks) so it can be inspected quickly; the
ramp plan in the spec covers the full regime coverage.

Usage
-----
::

    # paper fallback (no testnet creds) — exercises the harness plumbing
    PYTHONPATH=src .venv/Scripts/python.exe scripts/testnet_smoke_run.py \\
        --mode paper --duration 7d --label s5_smoke_paper

    # real testnet (requires BINANCE_TESTNET_API_KEY/SECRET in env)
    PYTHONPATH=src .venv/Scripts/python.exe scripts/testnet_smoke_run.py \\
        --mode live-testnet --duration 14d --label s5_smoke_testnet

The script does NOT itself reconcile — that is the reconciler module's
job.  We just orchestrate the run + ensure the JSONL is in the right
place.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = ROOT / ".venv" / "Scripts" / "python.exe"
OUT_DIR = ROOT / "vol_arb" / "output"


def main() -> int:
    p = argparse.ArgumentParser(description="Testnet smoke run orchestrator")
    p.add_argument("--mode", default="paper",
                   choices=["paper", "live-testnet"])
    p.add_argument("--duration", default="7d",
                   help="Wall-clock duration the engine should run "
                        "(e.g. 7d, 12h, 30m).")
    p.add_argument("--label", default="s5_smoke",
                   help="Output filename prefix; final files become "
                        "<label>_fills.jsonl and <label>_recon.md.")
    p.add_argument("--config", default="configs/paper_binance.yaml")
    p.add_argument("--capital", type=float, default=250_000.0)
    p.add_argument("--reconcile", action="store_true",
                   help="Run the reconciler on the JSONL after the run.")
    args = p.parse_args()

    fills_path = OUT_DIR / f"{args.label}_fills.jsonl"
    recon_path = OUT_DIR / f"{args.label}_recon.md"

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    cmd = [
        str(PY), "-m", "live.testnet.cli",
        "--config", args.config,
        "--mode", args.mode,
        "--duration", args.duration,
        "--capital", str(args.capital),
        "--fills-out", str(fills_path),
    ]
    print("$", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
    print(f"exit={proc.returncode} wall_s={time.time()-t0:.1f}")

    if args.reconcile:
        recon_cmd = [
            str(PY), "-m", "live.testnet.reconciler",
            "--fills", str(fills_path),
            "--out", str(recon_path),
        ]
        print("$", " ".join(recon_cmd))
        subprocess.run(recon_cmd, env=env, cwd=str(ROOT))

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
