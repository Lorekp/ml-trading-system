# Vol-Arb Execution Simulator: Detailed Backtest Report

## 1. Objective

Simulate a delta-neutral volatility arbitrage strategy on BTC options. The edge is the spread between our LightGBM-predicted 15-minute Realized Volatility (RV) and the Deribit market's Implied Volatility (IV). The instrument is a synthetic ATM straddle (long call + long put at the same strike), held for exactly 15 chronological minutes, with P&L attributed via the Black-Scholes Greeks.

---

## 2. Data Pipeline

### 2.1 Input Datasets

| Dataset | Source | Path | Rows | Date Range |
|---------|--------|------|------|------------|
| TSRV 15m | Binance aggTrades -> TSRV estimator | `TSRV/output/BTCUSDT_tsrv_15m.parquet` | 32,160 | 2023-05-01 to 2024-03-31 |
| OOF Predictions | LightGBM (purged K-fold) | `vol_model/output/BTCUSDT_oof_predictions.parquet` | 30,286 | 2023-05-21 to 2024-03-31 |
| Deribit Option Trades | Deribit REST API (history endpoint) | `data/deribit/trades/BTC_option/*.parquet` | 3,915,365 (overlap period) | 336 daily files |
| Deribit DVOL | Deribit REST API | `data/deribit/dvol/BTC_dvol_60m.parquet` | 43,945 | 2021-04-01 to 2026-04-06 |

### 2.2 TSRV (Two-Scale Realized Variance)

The label/target. Computed from Binance BTC perpetual tick data using the Zhang, Mykland & Ait-Sahalia (2005) two-scale estimator on $500K dollar bars, aggregated to 15-minute windows.

Key columns used in the backtest:
- `vol_ann`: annualized realized volatility = `sqrt(tsrv * 365.25)`, decimal (e.g., 0.3013 = 30.13%)
- `jump_fraction`: fraction of variance from jumps = `max(rv_slow - bpv, 0) / rv_slow`

### 2.3 OOF Predictions

The LightGBM model predicts `ln(vol_ann)` one step ahead. Out-of-fold predictions are used (each fold's prediction was made by a model that never saw that fold's data). This is the standard approach for backtesting ML models without leakage.

- `y_pred`: log-space prediction of next window's annualized RV
- Coverage: 30,286 of 32,160 bars (94.2%). First ~3 weeks are warmup/fold cutoff.
- Model performance: QLIKE 0.5281, vs-HAR -31.9%, RankIC 0.784

### 2.4 Deribit Option Trades

Downloaded via `get_last_trades_by_currency_and_time` from `history.deribit.com`. Paginated by timestamp advancement (`start_timestamp = last_trade_timestamp + 1ms`). Each trade includes:

- `timestamp`: trade execution time (datetime64[ns, UTC])
- `instrument_name`: e.g., `BTC-2JUN23-27000-C`
- `iv`: implied volatility (float64, percentage, e.g., 83.37 = 83.37%)
- `index_price`: BTC spot price at trade time (float64, USD)
- `price`: option price (float64, BTC fraction)
- `mark_price`: theoretical mark price (float64, BTC fraction)

### 2.5 Instrument Name Parsing

Fully vectorized via `pd.Series.str.split("-", expand=True)`:

```
"BTC-2JUN23-27000-C" -> ["BTC", "2JUN23", "27000", "C"]
  parts[0] = currency    ("BTC")
  parts[1] = expiry_str  ("2JUN23")
  parts[2] = strike      (27000, cast to float64)
  parts[3] = option_type ("C" or "P")
```

### 2.6 Expiry Date Parsing

```python
expiry_dt = pd.to_datetime(expiry_str, format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
```

Deribit options settle at 08:00:00 UTC. The `%d%b%y` format handles both `"2JUN23"` (1-digit day) and `"30JUN23"` (2-digit day) natively.

### 2.7 Time to Expiry

```python
T_years = (expiry_dt - trade_timestamp).total_seconds() / (365.25 * 86400)
dte_days = (expiry_dt - trade_timestamp).total_seconds() / 86400
```

### 2.8 Moneyness

```python
moneyness = strike / index_price    # K / S
```

### 2.9 ATM & DTE Filter

Keep only near-ATM, short-dated options:

```python
mask = (moneyness >= 0.95) & (moneyness <= 1.05) & (dte_days >= 1) & (dte_days <= 14)
```

Result: 1,059,617 ATM trades from 3,908,144 valid trades (27.1% pass rate).

### 2.10 DVOL

Deribit Volatility Index: composite ATM 30-day implied volatility, hourly resolution. Used as fallback when trade-level IV is NaN (only 1 bar out of 32,160).

---

## 3. Black-Scholes Greeks Engine

### 3.1 Assumptions

- Risk-free rate: `r = 0` (crypto, no risk-free rate)
- Model: standard Black-Scholes (1973), USD-denominated
- Deribit options are BTC-settled (inverse), but we use USD Greeks for the vol-arb signal

### 3.2 Core Formulas

Given spot `S`, strike `K`, time to expiry `T` (years), implied vol `sigma` (decimal), and `r = 0`:

```
d1 = [ln(S/K) + (r + 0.5 * sigma^2) * T] / [sigma * sqrt(T)]
d2 = d1 - sigma * sqrt(T)
```

With `r = 0`, this simplifies to:

```
d1 = [ln(S/K) + 0.5 * sigma^2 * T] / [sigma * sqrt(T)]
d2 = d1 - sigma * sqrt(T)
```

### 3.3 Greeks

**Delta:**
```
Call:  delta = N(d1)
Put:   delta = N(d1) - 1
```
Where `N(x) = scipy.stats.norm.cdf(x)`.

**Gamma** (same for calls and puts):
```
gamma = n(d1) / [S * sigma * sqrt(T)]
```
Where `n(x) = scipy.stats.norm.pdf(x)`.

**Vega** (per 1 percentage point IV move):
```
vega = S * n(d1) * sqrt(T) * 0.01
```
The `* 0.01` converts from per-unit-sigma to per-1%-IV convention. Units: USD per 1 BTC notional per 1 vol point.

**Theta** (per calendar day, with r=0):
```
theta = -[S * n(d1) * sigma] / [2 * sqrt(T)] / 365.25
```
Full formula with r: `theta_call = -[S * n(d1) * sigma] / [2 * sqrt(T)] - r * K * exp(-rT) * N(d2)`. The second term vanishes when `r = 0`. Units: USD per 1 BTC notional per calendar day.

### 3.4 Validity Guard

Rows where `T <= 0`, `sigma <= 0`, `S <= 0`, or `K <= 0` receive `NaN` for all Greeks. Implemented via a boolean mask applied after computation — no branching, fully vectorized.

### 3.5 Input Conversion

```python
sigma = iv / 100.0    # 83.37 -> 0.8337
is_call = (option_type == "C")
```

---

## 4. Point-in-Time Merge

### 4.1 Options -> TSRV Merge

Each 15-minute TSRV bar receives the most recent ATM option trade that occurred **before** the bar's timestamp. Implemented via `pd.merge_asof`:

```python
merged = pd.merge_asof(
    tsrv_features.sort_values("timestamp"),        # left: 32,160 15-min bars
    options_atm.sort_values("timestamp")            # right: 962,839 unique ATM trade timestamps
        .drop_duplicates("timestamp", keep="last"),
    on="timestamp",
    direction="backward"                            # no lookahead
)
```

When multiple ATM trades share the same millisecond timestamp, only the last (by `trade_seq` sort order) is kept.

**Coverage:** 32,159 of 32,160 bars (100.0%) have trade-level IV after the merge.

### 4.2 DVOL -> Merged

A second `merge_asof` (backward) attaches the most recent hourly DVOL reading:

```python
merged = pd.merge_asof(merged, dvol[["timestamp", "dvol_close"]],
                        on="timestamp", direction="backward")
```

**Coverage:** 32,160 of 32,160 bars (100.0%).

### 4.3 Derived Columns

```python
rv_pct         = vol_ann * 100                  # realized vol in %, e.g., 30.13
rv_iv_spread   = rv_pct - iv                    # realized vs trade IV
rv_dvol_spread = rv_pct - dvol_close            # realized vs DVOL index
iv_premium     = iv - dvol_close                # trade IV vs DVOL index
```

### 4.4 Final Merged Dataset

`vol_arb/output/BTC_vol_arb.parquet`: 32,160 rows x 33 columns.

---

## 5. Backtest Configuration (Default Parameters)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `initial_equity` | $100,000 | Starting capital |
| `risk_per_trade` | 0.01 (1%) | Fraction of initial equity risked per trade |
| `assumed_iv_shock` | 5.0 vol pts | Denominator for vega-based position sizing |
| `base_spread_iv` | 1.5 vol pts | Base IV bid-ask spread |
| `spread_jump_k` | 3.0 | Jump sensitivity for spread widening |
| `entry_threshold` | 0.5 vol pts | Additional edge required beyond half-spread |
| `holding_minutes` | 15 | Holding period (chronological) |
| `option_taker_fee` | 0.0003 | 0.03% of underlying per option leg |
| `perp_taker_fee` | 0.0004 | 0.04% of underlying per perp trade |
| `hedge_rebalances` | 1 | Assumed delta rebalances per holding period |

---

## 6. Backtest Pipeline (Step by Step)

### 6.1 Step 1: Load & Prepare Data

1. Load `BTC_vol_arb.parquet` (32,160 rows x 33 cols).
2. Join OOF predictions on timestamp index: `df.join(oof[["y_pred"]], how="left")`.
3. Convert log prediction to annualized RV percentage:

```python
rv_pred_pct = exp(y_pred) * 100
```

The model predicts `ln(vol_ann)`. Exponentiating recovers `vol_ann` (decimal), then `* 100` converts to percentage for comparison with `iv` (which is in %).

4. IV fallback: `iv_used = iv.fillna(dvol_close)`. Use trade-level IV where available, DVOL otherwise.
5. Drop rows where `rv_pred_pct`, `iv_used`, or `vega` is NaN.

**Result:** 30,286 tradeable bars.

### 6.2 Step 2: Dynamic Liquidity Spread

Deribit IV spreads widen during market stress. We model this using the jump fraction from the TSRV decomposition:

```python
simulated_spread_iv = base_spread_iv * exp(spread_jump_k * clip(jump_fraction, 0, 1))
```

With defaults: `spread = 1.5 * exp(3.0 * jump_fraction)`.

| jump_fraction | spread (vol pts) |
|---------------|-----------------|
| 0.00 | 1.50 |
| 0.05 | 1.74 |
| 0.10 | 2.02 |
| 0.20 | 2.73 |
| 0.50 | 6.72 |
| 1.00 | 30.13 |

**Observed distribution:**
- Mean spread: 2.120 vol pts
- Median spread: 1.744 vol pts
- Min: 1.500, Max: 27.581

### 6.3 Step 3: Signal Generation

The theoretical edge is the difference between model-predicted RV and market IV:

```python
edge = rv_pred_pct - iv_used
```

Signal rules:

```python
signal = +1  (long straddle)   if edge > (spread / 2) + entry_threshold
signal = -1  (short straddle)  if edge < -(spread / 2) - entry_threshold
signal =  0  (flat)            otherwise
```

With defaults (`entry_threshold = 0.5`): enter long if edge > ~1.25 vol pts, short if edge < ~-1.25 vol pts.

**Observed signals:**
- Long straddle (+1): 7,137 bars (23.6%)
- Short straddle (-1): 21,561 bars (71.2%)
- Flat (0): 1,588 bars (5.2%)
- Active rate: 94.8%

The heavy short bias (3:1 short vs long) reflects the persistent **variance risk premium** in crypto: the model's RV prediction is systematically below market IV, consistent with options being systematically overpriced.

**Edge distribution (active trades):**
- Mean: -7.42 vol pts (short bias)
- Median: -10.60 vol pts
- Std: 17.93 vol pts

### 6.4 Step 4: Chronological Exit Alignment

Our index is 15-minute volume-time (dollar bar windows). The exit must be exactly 15 chronological minutes forward. We use `pd.merge_asof` to find the market state at the exit time:

```python
# For each entry bar, compute target exit time
exit_target = entry_timestamp + Timedelta(minutes=15)

# Build exit lookup from the same dataset
exit_lookup = DataFrame({
    "ts":        df.index,           # all bar timestamps
    "iv_exit":   df["iv_used"],      # IV at each bar
    "price_exit": df["index_price"]  # spot price at each bar
})

# Match: find the most recent bar at or before exit_target
aligned = merge_asof(
    entry.sort_values("exit_target"),
    exit_lookup.sort_values("ts"),
    left_on="exit_target",
    right_on="ts",
    direction="backward"
)
```

Since the TSRV grid is already 15-minute regular, this almost always matches the next bar exactly. The `merge_asof` handles any gaps gracefully.

**Observed 15-min changes:**
- Mean dIV: -0.0850 vol pts, Std: 6.4830 vol pts
- Mean dS: +$1.46, Std: $111.39
- Non-zero dIV: 99.1% (28,428 of 28,698 trades)

### 6.5 Step 5: Transaction Costs

All fees computed per 1 BTC notional, then scaled by position size `N`.

**Options entry + exit fee:**
```python
options_fee = 4 * option_taker_fee * S
```
Explanation: 2 legs (call + put) x 2 (open + close) = 4 option transactions. Each costs 0.03% of the underlying spot price.

Example: S = $37,785 -> options_fee = 4 * 0.0003 * 37,785 = **$45.34**

**Initial delta hedge fee:**
```python
init_hedge = perp_taker_fee * 0.02 * S
```
ATM straddle has near-zero net delta (~0.02 residual). Hedge this with a perpetual futures trade.

Example: 0.0004 * 0.02 * 37,785 = **$0.30**

**Rebalance hedge fee:**
```python
rebalance_fee = hedge_rebalances * perp_taker_fee * |gamma_straddle * dS| * S
```
During the 15-min holding period, spot moves by `dS`. The delta adjustment required is `gamma_straddle * dS`, traded on perpetuals at `perp_taker_fee`.

Example: 1 * 0.0004 * |0.000567 * 111| * 37,785 = **$0.95**

**Total fee per 1 BTC notional:**
```python
fee_total = options_fee + init_hedge + rebalance_fee
```
Mean observed: **$46.59** per 1 BTC notional round-trip.

### 6.6 Step 6: P&L Attribution (Greeks Method)

The position is a synthetic straddle: long 1 ATM call + long 1 ATM put (for `signal = +1`) or short both (for `signal = -1`).

**Straddle Greeks** (ATM call and put have nearly identical gamma, vega, theta):
```python
vega_straddle  = 2 * vega_single     # mean observed: 25.92 USD/vol-pt/BTC
gamma_straddle = 2 * gamma_single    # mean observed: 0.000567 /USD/BTC
theta_straddle = 2 * theta_single    # mean observed: -191.52 USD/day/BTC
```

**P&L components** (all per 1 BTC notional, all using ENTRY-row Greeks):

**Vega P&L** — profit/loss from IV change:
```python
vega_pnl = signal * vega_straddle * (iv_exit - iv_entry)
```
Long straddle profits when IV rises. Short straddle profits when IV falls.

**Theta P&L** — time decay cost/benefit:
```python
dt_days = 15 / 1440    # = 0.010417 days
theta_pnl = signal * theta_straddle * dt_days
```
Long straddle pays theta (negative P&L). Short straddle collects theta (positive P&L). Note: `theta_straddle` is negative (e.g., -191.52), so `signal * theta * dt` is negative for longs, positive for shorts.

**Gamma P&L** — convexity gain/loss from spot movement:
```python
gamma_pnl = signal * 0.5 * gamma_straddle * dS^2
```
Long straddle benefits from spot movement (positive gamma). Short straddle is hurt by spot movement (negative gamma).

**IMPORTANT CAVEAT:** This is the raw second-order Taylor expansion of the option position value. For a truly delta-hedged position, the correct P&L would be `0.5 * gamma * S^2 * (sigma_realized^2 - sigma_implied^2) * dt`. The current implementation uses the unhedged gamma convexity term while also charging hedge fees. This is a known limitation flagged for correction.

**Net P&L per 1 BTC notional:**
```python
gross_pnl = vega_pnl + theta_pnl + gamma_pnl
net_pnl   = gross_pnl - fee_total
```

### 6.7 Step 7: Position Sizing (Constant Capital)

Every trade is sized identically from `initial_equity` — no compounding, no equity dependence. This enables fully vectorized computation (the entire P&L array is computed in one pass).

The position size `N` (in BTC notional) is chosen so that a `assumed_iv_shock`-point IV move risks `risk_per_trade` fraction of initial equity:

```python
N = (risk_per_trade * initial_equity) / (assumed_iv_shock * vega_straddle * S)
```

Substituting defaults:
```python
N = (0.01 * 100,000) / (5.0 * 2 * vega * S)
  = 1,000 / (5.0 * vega_straddle * S)
```

Cap: `N <= initial_equity / S` (cannot exceed full equity as notional).

Zero `N` where `vega_straddle` is non-finite or non-positive.

**Observed sizing:**
- Mean N: 0.000380 BTC
- Mean notional: $12.07

**Final trade P&L:**
```python
trade_pnl = N * net_pnl    (only where signal != 0; 0 otherwise)
```

### 6.8 Step 8: Performance Metrics

Computed from the `net_pnl` array (30,286 entries, 28,698 non-zero).

**Annualization constant:**
```python
bars_per_year = 365.25 * 24 * 4 = 35,064
```

**Sharpe Ratio:**
```python
sharpe = (mean(trade_pnl) / std(trade_pnl, ddof=1)) * sqrt(bars_per_year)
```

**Sortino Ratio:**
```python
downside_pnl = trade_pnl[trade_pnl < 0]
sortino = (mean(trade_pnl) / std(downside_pnl, ddof=1)) * sqrt(bars_per_year)
```

**Max Drawdown:**
```python
equity_curve = initial_equity + cumsum(pnl_all_bars)   # includes 0s for flat bars
running_max  = maximum.accumulate(equity_curve)
drawdowns    = (equity_curve - running_max) / running_max
max_dd       = min(drawdowns)
```

**Win Rate:**
```python
win_rate = count(trade_pnl > 0) / count(trade_pnl != 0)
```

**Profit Factor:**
```python
profit_factor = sum(trade_pnl[trade_pnl > 0]) / abs(sum(trade_pnl[trade_pnl < 0]))
```

---

## 7. Results (Default Parameters)

### 7.1 Tearsheet

| Metric | Value |
|--------|-------|
| Total Trades | 28,698 |
| Total PnL | -$298.72 |
| Total Return | -0.30% |
| Sharpe Ratio | -51.17 |
| Sortino Ratio | -68.50 |
| Max Drawdown | -0.30% |
| Win Rate | 34.1% |
| Profit Factor | 0.46 |
| Avg Trade PnL | -$0.0104 |
| Avg Win | +$0.0260 |
| Avg Loss | -$0.0292 |

### 7.2 P&L Decomposition

| Component | Total | Mean/Trade | Std/Trade |
|-----------|-------|-----------|-----------|
| Vega P&L | +$112.88 | +$0.00393 | $0.03678 |
| Theta P&L | +$7.89 | +$0.00027 | $0.00058 |
| Gamma P&L | +$1.91 | +$0.00007 | $0.00284 |
| **Gross P&L** | **+$122.68** | **+$0.00428** | |
| Fee Total | -$421.39 | -$0.01468 | $0.01385 |
| **Net P&L** | **-$298.72** | **-$0.01041** | $0.03809 |

**Interpretation:** The strategy has a small positive gross edge (+$123), but transaction costs ($421) are 3.4x the gross profit. At 15-minute frequency with taker fees on 4 option legs, the per-trade cost ($0.015) exceeds the per-trade edge ($0.004).

### 7.3 Long vs Short Breakdown

| Direction | Trades | Net PnL |
|-----------|--------|---------|
| Long straddles | 7,137 | -$38.06 |
| Short straddles | 21,561 | -$260.66 |

### 7.4 Threshold Sensitivity

| Threshold | Trades | PnL | Sharpe | Win Rate | Profit Factor |
|-----------|--------|-----|--------|----------|---------------|
| 0.5 (default) | 28,698 | -$299 | -51.2 | 34.1% | 0.46 |
| 2.0 | 27,106 | -$274 | -49.6 | 34.3% | 0.47 |
| 5.0 | 23,748 | -$217 | -45.2 | 35.4% | 0.50 |
| 10.0 | 17,641 | -$116 | -32.6 | 37.9% | 0.61 |
| 15.0 | 11,665 | -$36 | -15.4 | 41.4% | 0.79 |
| **20.0** | **6,841** | **+$4** | **+2.69** | **44.7%** | **1.04** |

The strategy turns profitable at threshold ~20 vol pts, where only high-conviction signals are traded. Higher selectivity reduces trade count (fewer fee events) and improves per-trade edge.

---

## 8. Known Limitations

### 8.1 Gamma P&L Model

The current gamma P&L formula `0.5 * gamma * dS^2` represents the raw second-order option value change, not the P&L of a delta-hedged position. For a properly delta-hedged straddle, the correct formula is:

```
hedge_pnl = 0.5 * gamma * S^2 * (sigma_realized^2 - sigma_implied^2) * dt
```

The current implementation charges delta-hedging fees but credits raw gamma convexity, creating an inconsistency. This should be corrected in a future iteration.

### 8.2 RV vs IV Horizon Mismatch

`rv_pred_pct` is the annualized RV measured over 15-minute windows (high-frequency, noisy). `iv` is the market's expectation of volatility over 1-14 days (smoothed, lower). These are fundamentally different horizons, leading to a persistent edge distribution with mean -7.4 vol pts and most bars triggering signals.

### 8.3 Single-Option Greeks as Straddle Proxy

The merged dataset attaches Greeks from a single ATM option trade (either a call or put, whichever traded most recently). For the straddle, we multiply by 2. This is exact at-the-money but introduces small errors for slightly OTM options (moneyness 0.95-1.05).

### 8.4 No Slippage Model Beyond Spread

The dynamic spread model accounts for market impact on IV, but does not model spot slippage for the perpetual hedge legs or queue position effects.

### 8.5 Constant Capital Sizing

All trades use `initial_equity` for sizing, ignoring accumulated P&L. This simplifies vectorization but does not reflect real compounding. Given the small total return (-0.30%), the impact is negligible.

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `vol_arb/config.py` | `VolArbConfig`: ATM filter params, paths |
| `vol_arb/greeks.py` | Vectorized BS Greeks (Delta, Gamma, Vega, Theta), r=0 |
| `vol_arb/options_loader.py` | Load Deribit parquets, parse instrument names, compute T/moneyness, ATM filter |
| `vol_arb/pipeline.py` | Two merge_asof joins (options + DVOL -> TSRV), derived columns |
| `vol_arb/backtester.py` | `VolArbBacktester` class: spread model, signals, exit alignment, fees, Greeks P&L, metrics |
| `vol_arb/run_pipeline.py` | CLI: build merged dataset |
| `vol_arb/run_backtest.py` | CLI: run backtest, print tearsheet + decomposition |

---

## 10. Reproduction

```bash
cd ml-trading-system

# Step 1: Build merged dataset (requires Deribit data + TSRV output)
PYTHONPATH="." python vol_arb/run_pipeline.py

# Step 2: Run backtest with default params
PYTHONPATH="." python vol_arb/run_backtest.py

# Step 3: Run with optimized threshold
PYTHONPATH="." python vol_arb/run_backtest.py --threshold 20.0

# Step 4: Custom parameters
PYTHONPATH="." python vol_arb/run_backtest.py \
  --threshold 15.0 \
  --base-spread 2.0 \
  --spread-k 4.0 \
  --risk 0.02 \
  --iv-shock 3.0 \
  --equity 50000

# Step 5: Save trade log for analysis
PYTHONPATH="." python vol_arb/run_backtest.py --threshold 20.0 \
  --output vol_arb/output/trade_log.parquet
```
