#!/usr/bin/env python
"""Overnight pipeline: validate data → parquetize → BERT + OFI test.

Runs unattended. Logs everything to experiments/overnight.log.

Steps:
1. Wait for re-download to finish, then validate all zips
2. Re-download any remaining corrupted files (loop until clean)
3. Start parquetization on homeserver (pyarrow, parallel)
4. Start BERT pre-training on local GPU
5. After parquets ready: compute OFI features on homeserver
6. Pull OFI features locally
7. Run LogReg A/B test (OHLCV vs OHLCV+OFI, 2-month lookback, production config)
8. Log all results
"""

import subprocess
import sys
import time
import logging
import os
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("experiments/overnight.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def ssh(cmd, timeout=600):
    """Run command on homeserver, return stdout."""
    r = subprocess.run(["ssh", "monitor", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.returncode


def ssh_python(code, timeout=3600):
    """Run python code on homeserver."""
    cmd = f'source ~/ml-trading-system/.venv/bin/activate && python3 -u -c "{code}"'
    r = subprocess.run(["ssh", "monitor", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


# ================================================================
# STEP 1: Wait for re-download to finish
# ================================================================
def wait_for_redownload():
    log.info("=== STEP 1: Waiting for re-download to finish ===")
    while True:
        out, _ = ssh("ps aux | grep 're-download\\|corrupted' | grep python | grep -v grep | wc -l")
        n = int(out.strip()) if out.strip().isdigit() else 0
        if n == 0:
            log.info("Re-download process finished")
            break
        log.info("Re-download still running (%s processes), waiting 60s...", n)
        time.sleep(60)


# ================================================================
# STEP 2: Validate + re-download loop
# ================================================================
def validate_and_fix():
    log.info("=== STEP 2: Validate all bookTicker zips ===")

    validate_code = """
import zipfile
from pathlib import Path
bad = []
total = 0
for sym_dir in sorted(Path('/hdd/binance-l2/bookTicker').iterdir()):
    if not sym_dir.is_dir(): continue
    for zp in sorted(sym_dir.glob('*.zip')):
        total += 1
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.namelist()
        except Exception:
            bad.append(str(zp))
print(f'{len(bad)}/{total}')
for b in bad:
    print(b)
"""

    fix_code = """
import zipfile, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
BASE = 'https://data.binance.vision'
bad = []
for sym_dir in sorted(Path('/hdd/binance-l2/bookTicker').iterdir()):
    if not sym_dir.is_dir(): continue
    for zp in sorted(sym_dir.glob('*.zip')):
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.namelist()
        except Exception:
            bad.append(zp)
print(f'Fixing {len(bad)} files (parallel)')
def fix_one(zp):
    sym = zp.parent.name
    url = f'{BASE}/data/futures/um/daily/bookTicker/{sym}/{zp.name}'
    zp.unlink()
    r = subprocess.run(['curl', '-s', '-o', str(zp), url], timeout=120)
    try:
        with zipfile.ZipFile(zp) as zf:
            zf.namelist()
        return True
    except Exception:
        if zp.exists():
            zp.unlink()
        return False
fixed = 0
with ThreadPoolExecutor(max_workers=10) as pool:
    futs = {pool.submit(fix_one, zp): zp for zp in bad}
    for f in as_completed(futs):
        if f.result():
            fixed += 1
print(f'Fixed {fixed}/{len(bad)}')
"""

    for attempt in range(5):
        out, _, rc = ssh_python(validate_code, timeout=600)
        lines = out.strip().split("\n")
        summary = lines[0] if lines else "0/0"
        n_bad = int(summary.split("/")[0])
        n_total = int(summary.split("/")[1]) if "/" in summary else 0
        log.info("Validation attempt %d: %s bad / %s total", attempt + 1, n_bad, n_total)

        if n_bad == 0:
            log.info("All files valid!")
            return True

        log.info("Fixing %d corrupted files...", n_bad)
        out2, err2, rc2 = ssh_python(fix_code, timeout=3600)
        log.info("Fix result: %s", out2)

    log.warning("Still have corrupted files after 5 attempts")
    return False


# ================================================================
# STEP 3: Parquetization on homeserver
# ================================================================
def run_parquetization():
    log.info("=== STEP 3: Converting bookTicker CSV -> Parquet (pyarrow) ===")
    out, _ = ssh(
        "source ~/ml-trading-system/.venv/bin/activate && "
        "cd /hdd && python -u convert_arrow.py 2>&1",
        timeout=7200,
    )
    log.info("Parquetization output:\n%s", out[-2000:] if len(out) > 2000 else out)
    return True


# ================================================================
# STEP 4: BERT pre-training (runs on local GPU)
# ================================================================
def start_bert():
    log.info("=== STEP 4: Starting BERT pre-training on GPU ===")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "models.bbo_bert.pretrain"],
        env={**os.environ, "PYTHONPATH": "src"},
        stdout=open("data/bbo_bert/pretrain_stdout.log", "a"),
        stderr=subprocess.STDOUT,
    )
    log.info("BERT launched as PID %d", proc.pid)
    return proc


# ================================================================
# STEP 5: Compute OFI features from parquets
# ================================================================
def compute_ofi_features():
    log.info("=== STEP 5: Computing OFI features from parquets ===")

    ofi_code = """
import time, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

L2_DIR = Path('/hdd/binance-l2')
PQ_DIR = L2_DIR / 'bookTicker_parquet'
CACHE_DIR = L2_DIR / 'features_15m'
CACHE_DIR.mkdir(exist_ok=True)
SYMBOLS = [
    'BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','ADAUSDT',
    'AVAXUSDT','DOTUSDT','LINKUSDT','NEARUSDT','UNIUSDT','AAVEUSDT',
    'LTCUSDT','BCHUSDT','HBARUSDT','TRXUSDT','SUIUSDT','APTUSDT',
    'ENJUSDT','SXPUSDT','ZECUSDT','XLMUSDT','FETUSDT','WLDUSDT',
    'ANKRUSDT','TAOUSDT','ENAUSDT',
]

def process_day(pq_path):
    df = pq.read_table(str(pq_path)).to_pandas()
    n = len(df)
    if n < 100:
        return pd.DataFrame()
    bbp = df['best_bid_price'].values.astype(np.float64)
    bbq = df['best_bid_qty'].values.astype(np.float64)
    bap = df['best_ask_price'].values.astype(np.float64)
    baq = df['best_ask_qty'].values.astype(np.float64)
    ts_ms = df['transaction_time'].values
    bp_d = np.empty(n); bp_d[0]=0; bp_d[1:]=bbp[1:]-bbp[:-1]
    bq_d = np.empty(n); bq_d[0]=0; bq_d[1:]=bbq[1:]-bbq[:-1]
    ap_d = np.empty(n); ap_d[0]=0; ap_d[1:]=bap[1:]-bap[:-1]
    aq_d = np.empty(n); aq_d[0]=0; aq_d[1:]=baq[1:]-baq[:-1]
    prev_bbq = np.empty(n); prev_bbq[0]=bbq[0]; prev_bbq[1:]=bbq[:-1]
    prev_baq = np.empty(n); prev_baq[0]=baq[0]; prev_baq[1:]=baq[:-1]
    bid_of = np.where(bp_d>0, bbq, np.where(bp_d==0, bq_d, -prev_bbq))
    ask_of = np.where(ap_d<0, -baq, np.where(ap_d==0, -aq_d, prev_baq))
    ofi = bid_of - ask_of
    mid = (bbp+bap)*0.5
    spread_pct = np.where(mid>0, (bap-bbp)/mid, 0.0).astype(np.float32)
    imb = ((bbq-baq)/(bbq+baq+1e-10)).astype(np.float32)
    bin_idx = (ts_ms // 900000).astype(np.int64)
    unique_bins = np.unique(bin_idx)
    rows = []
    for b in unique_bins:
        m = bin_idx == b
        if m.sum() < 100:
            continue
        o=ofi[m]; s=spread_pct[m]; im=imb[m]
        rows.append({'ts_bin':b, 'l2_ofi_sum':o.sum(), 'l2_ofi_mean':o.mean(),
            'l2_ofi_std':o.std(), 'l2_ofi_skew':float(pd.Series(o).skew()) if len(o)>2 else 0.0,
            'l2_spread_mean':s.mean(), 'l2_spread_std':s.std(), 'l2_spread_max':s.max(),
            'l2_imbalance_mean':im.mean(), 'l2_imbalance_end':im[-1],
            'l2_tick_count':float(len(o)), 'l2_mid_return':0.0})
    if not rows:
        return pd.DataFrame()
    bars = pd.DataFrame(rows)
    bars.index = pd.to_datetime(bars['ts_bin']*900000, unit='ms', utc=True)
    return bars.drop(columns=['ts_bin']).astype(np.float32)

def process_symbol(symbol):
    cache_path = CACHE_DIR / f'{symbol}_bookTicker_15m.parquet'
    if cache_path.exists():
        return f'{symbol}: cached'
    pq_dir = PQ_DIR / symbol
    if not pq_dir.exists():
        return f'{symbol}: no parquet dir'
    pqs = sorted(pq_dir.glob('*.parquet'))
    if not pqs:
        return f'{symbol}: no files'
    all_bars = []
    for p in pqs:
        try:
            bars = process_day(p)
            if len(bars) > 0:
                all_bars.append(bars)
        except Exception:
            pass
    if not all_bars:
        return f'{symbol}: all failed'
    combined = pd.concat(all_bars).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    combined.to_parquet(str(cache_path))
    return f'{symbol}: {len(combined)} bars'

t0 = time.time()
print(f'Computing OFI from parquets for {len(SYMBOLS)} symbols...')
with ProcessPoolExecutor(max_workers=4) as pool:
    futs = {pool.submit(process_symbol, s): s for s in SYMBOLS}
    for f in as_completed(futs):
        print(f'  {f.result()}')
print(f'Done in {time.time()-t0:.0f}s')
"""

    out, err, rc = ssh_python(ofi_code, timeout=7200)
    log.info("OFI computation output:\n%s", out[-2000:] if len(out) > 2000 else out)
    if err:
        log.warning("OFI stderr:\n%s", err[-1000:])
    return rc == 0


# ================================================================
# STEP 6: Pull OFI features locally
# ================================================================
def pull_ofi_features():
    log.info("=== STEP 6: Pulling OFI features locally ===")
    local_dir = Path("data/l2-features")
    local_dir.mkdir(exist_ok=True)
    r = subprocess.run(
        ["scp", "monitor:/hdd/binance-l2/features_15m/*_bookTicker_15m.parquet",
         str(local_dir) + "/"],
        capture_output=True, text=True, timeout=120,
    )
    n = len(list(local_dir.glob("*_bookTicker_15m.parquet")))
    log.info("Pulled %d bookTicker feature files", n)
    return n > 0


# ================================================================
# STEP 7: Run OFI A/B test
# ================================================================
def run_ofi_ab_test():
    log.info("=== STEP 7: Running OFI A/B test ===")
    sys.path.insert(0, "src")
    r = subprocess.run(
        [sys.executable, "-u", "experiments/l2_ab_test.py"],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True, text=True, timeout=3600,
    )
    log.info("A/B test output:\n%s", r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
    if r.stderr:
        log.warning("A/B stderr:\n%s", r.stderr[-1000:])
    return r.returncode == 0


# ================================================================
# MAIN
# ================================================================
def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("  OVERNIGHT PIPELINE STARTED")
    log.info("=" * 60)

    # Step 1: Wait for re-download
    wait_for_redownload()

    # Step 2: Validate + fix loop
    if not validate_and_fix():
        log.error("Could not fix all corrupted files. Continuing anyway.")

    # Step 3+4: Parquetization + BERT in parallel
    bert_proc = start_bert()

    parquet_ok = run_parquetization()
    if not parquet_ok:
        log.error("Parquetization failed")

    # Step 5: OFI features (needs parquets)
    if parquet_ok:
        ofi_ok = compute_ofi_features()
    else:
        ofi_ok = False
        log.error("Skipping OFI — parquets not ready")

    # Step 6: Pull features
    if ofi_ok:
        pull_ofi_features()

    # Step 7: A/B test
    if ofi_ok:
        run_ofi_ab_test()

    # Check BERT status
    if bert_proc.poll() is None:
        log.info("BERT still training (PID %d)", bert_proc.pid)
    else:
        log.info("BERT finished (exit code %d)", bert_proc.returncode)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("  OVERNIGHT PIPELINE COMPLETE (%.0f min)", elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
