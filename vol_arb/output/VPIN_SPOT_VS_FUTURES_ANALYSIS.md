# VPIN feed analysis — Spot vs Futures (2026-05-07)

## Context

The live engine reads VPIN from `wss://fstream.binance.com/ws/btcusdt@aggTrade`
(Binance USD-M Futures). This endpoint is geo-blocked from the homeserver
(and from local Germany IPs) — the WebSocket handshake succeeds but **zero
aggTrade frames flow**, leaving the live engine on the warm-start VPIN value
(0.1520) for hours. Two bugs were fixed already:

1. **VPIN is_stale fix** (commit `f6a548d`): the live stream's staleness
   check returned False forever when `_last_update_ts is None`, masking
   the silent-feed failure indefinitely. Now alerts after the grace
   period elapses with no message.
2. **4 sister silent-failure bugs** (commit `987a57f`): same pattern in
   SVI calibrator, KillSwitch network-outage, RNDMonitor size_scale,
   and IndexPriceBuffer.latest().

The remaining question: **can we switch the VPIN feed from Binance Futures
to Binance Spot** (which is reachable) without retraining the DRL hedger?

## Method

- Window: April 2026 (1 month)
- Futures aggTrades: `data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet`
  (41.5M trades, 4.4M BTC volume, 101 BTC/min)
- Spot aggTrades: `data/binance-spot-aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-04.zip`
  (26.1M trades, ~330k BTC volume, 11 BTC/min)
- Computed VPIN with the same Easley-Prado-O'Hara streaming algorithm the
  live engine uses (`_LiveBucketAccumulator` equivalent). Rolling window 50
  imbalances on both sides.
- Tested two bucket sizings:
  - **Same bucket (1000 BTC)** — preserves the V2 calibration, but Spot
    fills buckets ~9× slower (91 min vs 10 min on Futures)
  - **Cadence-matched (109 BTC Spot, 1000 BTC Futures)** — same wall-clock
    update cadence, but per-bucket imbalance distributions differ
- Aligned to hourly grid (last-VPIN-in-hour) for cross-venue correlation

## Results

### Distribution shape

| Metric | Futures (1000 BTC) | Spot (1000 BTC) | Spot (109 BTC, cadence-matched) |
|---|---:|---:|---:|
| count (bucket closes / month) | 4329 | 424 (-90%) | 4292 (-1%) |
| mean | 0.1921 | 0.1187 (**-38%**) | 0.2643 (**+38%**) |
| std | 0.0323 | 0.0144 (-56%) | 0.0388 (+20%) |
| p50 | 0.1890 | 0.1179 | 0.2579 |
| p99 | 0.2781 | 0.1499 | 0.3622 |
| autocorr (lag=1h) | 0.77 | 0.97 | 0.88 |

**Both Spot configurations are decisively off-distribution from Futures.** With
the same bucket size, Spot's p99 is below Futures' p25 — these are practically
disjoint. With the cadence-matched bucket, Spot's mean is 38% above Futures'
mean (in the other direction).

### Cross-venue correlation

| Bucket sizing | Pearson | Spearman |
|---|---:|---:|
| Same (1000 BTC) | **0.05** | 0.02 |
| Cadence-matched (109 BTC Spot) | **0.07** | 0.05 |

Lagged correlation analysis (cadence-matched):

| Lag (hours, +ve = spot lags fut) | Pearson |
|---:|---:|
| -24 | -0.14 |
| -12 | -0.06 |
| -6 | -0.03 |
| -3 | -0.03 |
| -1 | +0.03 |
| 0 | +0.07 |
| +1 | +0.06 |
| +3 | +0.06 |
| +6 | +0.02 |
| +12 | -0.13 |
| +24 | -0.19 |

**No lag captures meaningful correlation.** Peak at lag=0 is 0.07; flips negative
beyond ±12h. There is no temporal structure that would let one feed proxy the
other.

### Regime-conditional correlation

| Regime (binned by Futures VPIN) | Pearson | n |
|---|---:|---:|
| Low (q<0.33) | +0.005 | 235 |
| Mid (0.33-0.67) | -0.046 | 242 |
| High (q>0.67) | **+0.224** | 235 |

The only modest signal is in **stress regimes** (~0.22 correlation when Futures
VPIN is in the top tercile). This is consistent with the intuition that during
genuine market stress, Spot and Futures both reflect aggressor-side toxicity
proportionally. Outside stress periods, the two feeds carry **independent**
information.

## Interpretation

The DRL hedger (production SAC v4) was trained with **Binance Futures VPIN** as
one of 8 observation slots, normalized through VecNormalize fit on the training
distribution (mean ~0.19, std ~0.03). Switching to Spot VPIN at any bucket
sizing means:

- **Distribution shift**: 38% mean offset in either direction (1.0σ–2.5σ in
  normalized units) — the policy's input is centered in a region it never
  trained on.
- **Variance change**: 50%–20% smaller/larger std means the normalized
  observation has a non-unit scale.
- **Information shift**: Pearson correlation 0.05–0.07 means Spot VPIN is
  **not a noisy version of Futures VPIN; it's a different feature**. The
  policy's learned VPIN-conditional behaviour does not transfer.

In W5 v4 backtest the DRL contributed +12.4% Sortino over no-DRL. We don't
know the elasticity of that improvement to VPIN distribution shift, but
empirically the policy is operating on a feature that is essentially
independent of the one it was trained on.

## Recommendation

**Do NOT switch the VPIN feed to Binance Spot.** The data is unambiguous:
Spot and Futures VPIN are statistically independent signals despite being
"the same metric on different venues". Off-distribution input to a trained
DRL policy is a known mode of degradation.

### Pragmatic path forward

The current state (warm-start VPIN frozen at 0.1520) is now alert-surfaced
by the bug fixes. So the engine is no longer pretending to be healthy. The
options, in increasing order of effort:

1. **Disable VPIN feed entirely; feed the training-distribution mean (`vpin_fallback=0.189`)
   to the DRL.** The current code already does this when `is_stale` is True
   AND the stream's `latest_vpin` falls back. Operationally, the engine
   runs with VPIN as a constant — the DRL's VPIN-conditional behaviour
   collapses to the unconditional mean, which is the safest off-distribution
   degradation. **Recommended for now.** Zero new code.

2. **Find a non-geo-blocked Futures source.** Candidates:
   - **Bybit BTCUSDT perp** (`wss://stream.bybit.com/v5/public/linear`,
     subscribe `publicTrade.BTCUSDT`). Reachable from EU. Closer participant
     mix to Binance Futures (leveraged perp flow, similar maker/taker
     dynamics). Requires verifying message format + downloading historical
     data for a calibration check.
   - **OKX BTC-USDT-SWAP**. Similar profile, similar verification cost.
   - Either swap is **a new venue dependency** with its own staleness/auth
     concerns. Worth a focused 1-day cycle.

3. **Tunnel via a non-geo-blocked IP.** Operationally cleanest, no code change.
   Cost: VPN/proxy infrastructure on the homeserver, ongoing cost, depends
   on TOS. Not scalable.

4. **Retrain the DRL without VPIN** (W5 v5 with 7 features). Major effort —
   re-validation of all gates, new VecNormalize, new manifest. Justified if
   we're going to permanently move off VPIN.

### Recommendation: Option 1 (disable / fallback) NOW; queue Option 2 (Bybit)

The sister-bug fixes already make the disabled-VPIN case visible (heartbeat
will show `VPIN_STALE`). The fallback value (0.189) is the training-mean — the
DRL operates as if VPIN is "neutral", which is the safest degradation under
distribution shift.

Bybit perp is the most plausible drop-in replacement and worth the 1-day
analysis cycle: characterize Bybit-VPIN distribution vs Binance-Futures-VPIN
on overlapping historical windows; if Pearson correlation > 0.7 and mean/std
are within 1σ, swap with confidence.

## Files

- Analysis script: `scripts/analyze_vpin_spot_vs_futures.py`
- Live diagnostic script: `scripts/diagnose_vpin_spot_vs_futures.py`
- Joined hourly time-series: `vol_arb/output/vpin_spot_vs_futures.parquet`
- Spot aggTrades (one month): `data/binance-spot-aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-04.zip`

## Reproducibility

```bash
PYTHONPATH=. python scripts/analyze_vpin_spot_vs_futures.py \
  --month 2026-04 --bucket-btc 1000 --rolling 50

PYTHONPATH=. python scripts/analyze_vpin_spot_vs_futures.py \
  --month 2026-04 --bucket-btc 1000 --bucket-btc-spot 109 --rolling 50
```
