#!/usr/bin/env python
"""Quick post-fix validation."""
import sys, os, warnings, sqlite3
import numpy as np, pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig
from live.signal_generator import SklearnBackend

try:
    from afml.bet_sizing import get_signal, discretize_signal
except ImportError:
    def get_signal(prob, num_classes=2): return 2.0 * prob - 1.0
    def discretize_signal(sig, step_size=0.1): return round(sig / step_size) * step_size

DB = sys.argv[1] if len(sys.argv) > 1 else "data/live/paper_binance_postfix3.db"
CUTOFF = sys.argv[2] if len(sys.argv) > 2 else "2026-04-08T18:15"

backend = SklearnBackend()
backend.load_models(Path("data/live/models"))
fc = FeatureConfig(
    feature_list=["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore",
                  "rsi_14","stoch_rsi","macd_hist","bb_pct_b","atr_14",
                  "dist_swing_high","dist_swing_low"],
    winsor_lo=0.01, winsor_hi=0.99, vol_filter_quantile=0.50,
    vol_zscore_window=30, swing_lookback=20,
)
TFS = [("15m","15min",15,2),("1h","1h",60,4),("3h","3h",180,16),
       ("4h","4h",240,6),("24h","1D",1440,12)]

conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute(f"SELECT timestamp, asset, side, timeframe FROM trades "
          f"WHERE action='open' AND timestamp >= '{CUTOFF}' ORDER BY timestamp")
paper = [(r[0][:16].replace("T"," "), r[1], r[2], r[3]) for r in c.fetchall()]
conn.close()

ASSETS = sorted(set(t[1] for t in paper) |
    {"DOGE","ETC","HYPE","JOE","PAXG","RED","XRP","LTC","ADA","BNB","SIREN","TAO","BULLA"})

print(f"Post-fix trades: {len(paper)}")
ohlcv = {}; ohlcv_4h = {}
for a in ASSETS:
    try:
        raw = load_binance_1m(f"{a}USDT", ["data/binance-1m"])
        if raw.index.tz is not None: raw.index = raw.index.tz_localize(None)
    except: continue
    for tl, tr, _, _ in TFS:
        try: ohlcv[(a, tl)] = resample_ohlcv(raw, tr)
        except: pass
    try: ohlcv_4h[a] = resample_ohlcv(raw, "4h")
    except: pass

def chk_regime(d, df):
    if len(df) < 25: return True
    e = df["close"].ewm(span=20, adjust=False).mean()
    r = np.where(df["close"] > e, 1.0, -1.0)
    v = pd.Series(r, index=df.index).shift(1).iloc[-1]
    if pd.isna(v): return True
    return not (d > 0 and v < 0) and not (d < 0 and v > 0)

lk = {}
for ts, a, s, tf in paper: lk[(ts, a, tf)] = s
bar_times = sorted(set(t[0] for t in paper))
m = mi = po = bo = 0

print(f"\n{'Time':>16s} {'Asset':>8s} {'TF':>4s} {'Paper':>6s} {'BT':>6s} "
      f"{'Prob':>8s} {'Conf':>8s} {'Status':>12s}")
print("-" * 78)

for ts in bar_times:
    bt = pd.Timestamp(ts); tm = bt.hour * 60 + bt.minute
    atfs = [(tl, h) for tl, _, mn, h in TFS if tm % mn == 0]
    pab = {(a, tf): s for (t, a, tf), s in lk.items() if t == ts}
    bab = {}
    for tl, h in atfs:
        for a in ASSETS:
            df = ohlcv.get((a, tl))
            if df is None or len(df) < 30: continue
            if not backend.has_model(a, tl): continue
            ds = df[df.index < bt]
            if len(ds) < 30: continue
            try: ft = compute_features(ds, fc)
            except: continue
            if len(ft) == 0: continue
            la = ft.iloc[[-1]]
            co = [c for c in fc.feature_list if c in la.columns]
            if len(co) < len(fc.feature_list) * 0.8: continue
            va = la[co]
            if va.isna().any().any(): continue
            pr = backend.predict(a, tl, va.values)
            if pr is None: continue
            d = 1.0 if pr > 0.5 else -1.0; cf = abs(pr - 0.5)
            if cf < 0.08: continue
            d4 = ohlcv_4h.get(a)
            if d4 is not None:
                d4s = d4[d4.index < bt]
                if not chk_regime(d, d4s): continue
            bab[(a, tl)] = ("Buy" if d > 0 else "Sell", pr, cf)

    for (a, tf) in sorted(set(pab) | set(bab)):
        ps = pab.get((a, tf)); bi = bab.get((a, tf))
        if ps and bi:
            bs, p, c = bi
            st = "MATCH" if ps == bs else "MISMATCH"
            if st == "MATCH": m += 1
            else: mi += 1
            print(f"{ts:>16s} {a:>8s} {tf:>4s} {ps:>6s} {bs:>6s} "
                  f"{p:>8.4f} {c:>8.4f} {st:>12s}")
        elif ps:
            po += 1
            print(f"{ts:>16s} {a:>8s} {tf:>4s} {ps:>6s} {'---':>6s} "
                  f"{'---':>8s} {'---':>8s} {'PAPER_ONLY':>12s}")
        elif bi:
            bs, p, c = bi; bo += 1
            print(f"{ts:>16s} {a:>8s} {tf:>4s} {'---':>6s} {bs:>6s} "
                  f"{p:>8.4f} {c:>8.4f} {'BT_ONLY':>12s}")

t = len(paper)
print(f"\n{'='*78}")
print(f"  RESULT: {m}/{t} MATCH ({m/max(t,1)*100:.0f}%) | "
      f"{mi} MISMATCH | {po} PAPER_ONLY | {bo} BT_ONLY")
print(f"{'='*78}")
